"""Score reconstructions against their ground truth: IoU, GMS and Chamfer distance.

    # the tree agent_run.py writes; GT defaults to each part's own input mesh
    python evaluate.py --predictions runs/ --out results/

    # flat predictions (<id>.stl) of any method against a GT directory (<id>.stl)
    python evaluate.py --predictions preds/ --gt data/meshes/ --out results/

    # predictions written in the CADENA convention (pred / 200 + 0.5 = unit cube)
    python evaluate.py --predictions preds/ --gt data/meshes/ --pred-frame unit200

For every part the prediction is looked up as `<predictions>/<id>/<id>.stl`
(agent_run.py's layout) or `<predictions>/<id>.stl`. The GT is
`<gt>/<id>.<ext>` when --gt is given, otherwise `<predictions>/<id>/part.stl`,
the copy of the input mesh agent_run.py reconstructed from. With --gt, the set
of parts is the GT directory, so a part with no prediction counts as invalid.

Both meshes are put in the GT's frame (GT box centred at 0.5, longest side 1);
see metrics/pair_metrics.py for the exact protocol. Reported, as in the paper:

    IoU   volumetric IoU (boolean on closed bodies, voxel fallback otherwise),
          mean over parts that produced geometry
    GMS   Generalized Match Score (CADENA), normalised, mean over the same parts
    CD    symmetric Chamfer distance x1000 on 8192 points, MEDIAN over the same parts
    IR    invalid rate: share of parts with no usable prediction

Writes <out>/per_part.csv (one row per part, every metric) and
<out>/results.json (the aggregates, overall and per group with --groups).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "metrics"))

from pair_metrics import PRED_FRAMES, PairMetrics, score_pair  # noqa: E402

MESH_SUFFIXES = (".stl", ".obj", ".ply", ".off")
COLUMNS = ["id", "group"] + [c for c in PairMetrics.columns() if c != "figure_id"]


def _mesh_in(directory: Path, stem: str) -> Path | None:
    for suffix in MESH_SUFFIXES:
        p = directory / f"{stem}{suffix}"
        if p.is_file():
            return p
    return None


def collect(predictions: Path, gt: Path | None) -> list[tuple[str, Path, Path]]:
    """(id, gt path, prediction path) for every part to score."""
    def pred_of(pid: str) -> Path:
        nested = predictions / pid / f"{pid}.stl"
        if nested.is_file():
            return nested
        return _mesh_in(predictions, pid) or nested     # a missing file scores as no_prediction

    if gt is not None:
        ids = sorted(p.stem for p in gt.iterdir()
                     if p.is_file() and p.suffix.lower() in MESH_SUFFIXES)
        return [(pid, _mesh_in(gt, pid), pred_of(pid)) for pid in ids]
    parts = sorted(d for d in predictions.iterdir() if (d / "part.stl").is_file())
    if not parts:
        raise SystemExit(f"evaluate: no <id>/part.stl under {predictions}; pass --gt "
                         f"for flat predictions")
    return [(d.name, d / "part.stl", pred_of(d.name)) for d in parts]


def read_groups(path: Path | None) -> dict[str, str]:
    """id -> group from a CSV with `id` and `group` columns."""
    if path is None:
        return {}
    with open(path, newline="") as f:
        return {row["id"]: row["group"] for row in csv.DictReader(f)}


def _finite(v):
    return None if isinstance(v, float) and not math.isfinite(v) else v


def _score(job, repair: bool, pred_frame: str, repaired_dir: Path | None) -> dict:
    pid, gt, pred = job
    save = repaired_dir / f"{pid}.stl" if repaired_dir else None
    r = score_pair(gt, pred, figure_id=pid, repair=repair, save_repaired=save,
                   pred_frame=pred_frame)
    d = {k: _finite(v) for k, v in r.to_dict().items()}
    d.pop("figure_id", None)
    return {"id": pid, **d}


def _mean(xs):
    return statistics.fmean(xs) if xs else None


def summarize(rows: list[dict]) -> dict:
    """The paper's aggregates over one set of rows."""
    valid = [r for r in rows if not r.get("failure")]
    iou = [r["iou"] for r in valid if r.get("iou") is not None]
    gms = [r["gms_norm"] for r in valid if r.get("gms_norm") is not None]
    cd = [r["cd_8k"] for r in valid if r.get("cd_8k") is not None]
    return {
        "n_parts": len(rows),
        "n_valid": len(valid),
        "invalid_rate": (len(rows) - len(valid)) / len(rows) if rows else None,
        "iou_mean": _mean(iou),
        "gms_mean": _mean(gms),
        "cd_median_x1000": statistics.median(cd) if cd else None,
        "cd_mean_x1000": _mean(cd),
        "score_mean": _mean([r["score"] for r in rows if r.get("score") is not None]),
        "watertight_rate": (sum(1 for r in valid if r.get("pred_watertight")) / len(valid)
                            if valid else None),
    }


def _fmt(v, pct=False):
    if v is None:
        return "-"
    return f"{100 * v:.2f}" if pct else f"{v:.3f}"


def print_table(summary: dict) -> None:
    head = f"{'group':24s} {'n':>5s} {'IoU':>7s} {'GMS':>7s} {'CD med':>8s} {'IR %':>6s}"
    print(head)
    print("-" * len(head))
    for name, s in summary.items():
        print(f"{name:24s} {s['n_parts']:5d} {_fmt(s['iou_mean'], True):>7s} "
              f"{_fmt(s['gms_mean'], True):>7s} {_fmt(s['cd_median_x1000']):>8s} "
              f"{_fmt(s['invalid_rate'], True):>6s}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--predictions", type=Path, required=True,
                    help="agent_run.py output root, or a directory of <id>.stl")
    ap.add_argument("--gt", type=Path, default=None,
                    help="directory of GT meshes <id>.stl (default: <predictions>/<id>/part.stl)")
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--pred-frame", choices=PRED_FRAMES, default="gt",
                    help="gt: predictions are in the GT mesh's own frame (agent_run.py "
                         "exports there); unit200: pred / 200 + 0.5 is the GT in the unit cube")
    ap.add_argument("--groups", type=Path, default=None,
                    help="optional CSV with columns id,group for a per-group table")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-gt-repair", action="store_true",
                    help="score the GT exactly as loaded (default: repair known defects "
                         "of GT that is not a valid solid, e.g. double-sided exports)")
    args = ap.parse_args(argv)

    jobs = collect(args.predictions, args.gt)
    groups = read_groups(args.groups)
    args.out.mkdir(parents=True, exist_ok=True)
    repaired_dir = None if args.no_gt_repair else args.out / "gt_repaired"
    print(f"scoring {len(jobs)} part(s) from {args.predictions} with {args.workers} worker(s)",
          flush=True)

    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_score, j, not args.no_gt_repair, args.pred_frame,
                               repaired_dir): j for j in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            pid, gt, pred = futures[fut]
            try:
                rows.append(fut.result())
            except Exception as exc:  # a crashed worker must not lose the run
                rows.append({"id": pid, "gt_path": str(gt), "pred_path": str(pred),
                             "failure": "scoring_error", "error": repr(exc),
                             "score": 0.0, "score_strict": 0.0})
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} scored, {time.time() - t0:.0f}s", flush=True)
    for r in rows:
        r["group"] = groups.get(r["id"], "")
    rows.sort(key=lambda r: r["id"])

    with open(args.out / "per_part.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    summary = {"all": summarize(rows)}
    for g in sorted({r["group"] for r in rows if r["group"]}):
        summary[g] = summarize([r for r in rows if r["group"] == g])
    (args.out / "results.json").write_text(json.dumps(
        {"predictions": str(args.predictions), "gt": str(args.gt) if args.gt else None,
         "pred_frame": args.pred_frame, "summary": summary}, indent=2) + "\n")
    print()
    print_table(summary)
    print(f"\nIoU and GMS in %, CD x1000 (median over valid parts), IR = invalid rate.\n"
          f"wrote {args.out / 'per_part.csv'} and {args.out / 'results.json'}")


if __name__ == "__main__":
    main()
