"""Score one reconstruction against its ground truth: CD, IoU and GMS.

The formulas, constants and operation order are those of the CADENA reference
metrics (`utils/metrics.py` in https://github.com/kulibinai/cadena), so the
numbers are comparable with it:
CD x1000 on 8192 seeded samples (also reported at 30k), boolean IoU on closed bodies with the
1.05 / 1e-3 plausibility guards, GMS from the vendored CADENA
`utils/gms.py` (`gms.py` next to this file) with ``gms_norm = 1 - aoc/25``.

Six deliberate differences:

* Frame. CADENA normalises GT by its own extents and the prediction by
  ``/200`` (the scale its model writes in). Our predictions are exported in the
  target's own frame (mm, or whatever the dataset STL uses), so BOTH meshes get
  the GT's transform: centre of the GT box to (0.5, 0.5, 0.5), longest GT side
  to 1. A translated or scaled prediction stays translated or scaled.
* Voxel IoU fallback. Boolean IoU needs closed bodies on both sides. When the GT
  or the prediction is not watertight, or the boolean result is rejected by the
  guards, IoU comes from occupancy on one shared lattice instead
  (`scripts/compare.py`: one parity fill per axis, majority of the three, so an
  open or self-intersecting mesh still gets a sensible interior). `iou_method`
  says which one produced the number.
* Open predictions are measured, not failed. CADENA floors a non-watertight
  prediction to score 0 with no metrics; here it gets CD, voxel IoU and GMS.
  `score_strict` reproduces the CADENA contract (0 for an open prediction),
  `score` is the plain mean of IoU and normalised GMS.
* Cavities. Boolean IoU subtracts closed inward-facing shells (cavity walls)
  instead of dropping them, see `solid_bodies`.
* GT repair. A GT that is not a valid solid is repaired when a known defect
  explains it and the result is checkably closed (`repair_gt`; today: the
  duplicated triangles of double-sided exports). Valid GT is never touched;
  `gt_repair` records what happened.
* GMS seeding actually reaches trimesh (`_seeded_sampling`), so GMS is
  deterministic; through the snapshot alone it is not on trimesh >= 4.

Usable as a module (`score_pair`) or from the shell:

    python metrics/pair_metrics.py GT.stl PRED.stl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                       # vendored CADENA gms.py
sys.path.insert(0, str(HERE.parent / "scripts"))     # scripts/compare.py (voxel occupancy)

import compare  # noqa: E402
from gms import A_in_B_ball_matching_multiangle_v3, TrimeshHandler  # noqa: E402

logger = logging.getLogger(__name__)

# Reference constants (CADENA utils/metrics.py).
CD_N_POINTS = 8192
CD_SCALE = 1000.0
# PairMetrics field prefix -> samples per mesh. 8k is the reference; 30k is
# reported alongside because 8k CD carries visible sampling noise.
CD_VERSIONS = {"cd_8k": CD_N_POINTS, "cd_30k": 30000}
IOU_REJECT_ABOVE = 1.05
INTERSECTION_EXCESS_TOL = 1e-3
GMS_N_POINTS = 8192
GMS_N_ANGLES = 125
GMS_REL_DIST_TOL = 0.05
GMS_UPPER_BOUND_TOL_RT = 25
GMS_HOW_MANY_MEDIANS = 5.0
GMS_MAX_POSSIBLE_N = 100000
METRICS_SAMPLE_SEED = 42

# Where a prediction mesh lives. `gt`: the target's own frame (our harness
# exports there). `unit200`: the CADENA model convention, where
# pred / 200 + 0.5 lands on the GT normalised into the unit cube.
PRED_FRAME_GT = "gt"
PRED_FRAME_UNIT200 = "unit200"
PRED_FRAMES = (PRED_FRAME_GT, PRED_FRAME_UNIT200)
UNIT200_TO_CUBE = (trimesh.transformations.translation_matrix([0.5, 0.5, 0.5])
                   @ trimesh.transformations.scale_matrix(1.0 / 200))

FAILURE_NO_PREDICTION = "no_prediction"
FAILURE_LOAD_ERROR = "load_error"
FAILURE_EMPTY = "empty_mesh"

IOU_BOOLEAN = "boolean"
IOU_VOXEL = "voxel"


@dataclass
class PairMetrics:
    """Metrics of one GT/prediction pair, plus where the meshes came from."""

    gt_path: str
    pred_path: str
    figure_id: str | None = None

    gt_watertight: bool | None = None
    # what repair_gt did to the GT (None: it was a valid solid as loaded)
    gt_repair: str | None = None
    pred_watertight: bool | None = None
    failure: str | None = None

    iou: float | None = None
    iog: float | None = None
    iop: float | None = None
    iou_method: str | None = None
    # why the voxel path was taken: gt_not_watertight, pred_not_watertight,
    # boolean_rejected (both closed, the boolean result failed a guard)
    iou_fallback_reason: str | None = None
    voxel_cells: int | None = None

    gms_raw: float | None = None
    gms_norm: float | None = None
    # CD at the reference 8192 samples and at 30k (less sampling noise), see CD_VERSIONS
    cd_8k: float | None = None
    cd_8k_precision: float | None = None
    cd_8k_recall: float | None = None
    cd_30k: float | None = None
    cd_30k_precision: float | None = None
    cd_30k_recall: float | None = None

    gt_faces: int | None = None
    pred_faces: int | None = None
    elapsed_s: float | None = None
    error: str | None = None

    @property
    def score(self) -> float:
        """Mean of IoU and normalised GMS (whichever exist); 0 on failure."""
        if self.failure is not None:
            return 0.0
        available = [v for v in (self.iou, self.gms_norm) if v is not None]
        if not available:
            return 0.0
        return float(np.clip(np.mean(available), 0.0, 1.0))

    @property
    def score_strict(self) -> float:
        """The CADENA contract: an open prediction is a failure (0)."""
        return self.score if self.pred_watertight else 0.0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["score"] = self.score
        data["score_strict"] = self.score_strict
        return data

    @classmethod
    def columns(cls) -> list[str]:
        return [f.name for f in fields(cls)] + ["score", "score_strict"]


# --- loading and frame --------------------------------------------------------


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """One Trimesh from an STL/OBJ/PLY/...; a multi-geometry scene is concatenated."""
    mesh = trimesh.load(str(path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    return mesh


def gt_frame(gt_mesh: trimesh.Trimesh) -> np.ndarray:
    """4x4 transform taking the GT into the unit cube centred at (0.5, 0.5, 0.5)."""
    center = (gt_mesh.bounds[0] + gt_mesh.bounds[1]) / 2.0
    extent = float(np.max(gt_mesh.extents))
    scale = 1.0 / extent if extent > 1e-7 else 1.0
    m = trimesh.transformations.translation_matrix([0.5, 0.5, 0.5])
    m = m @ trimesh.transformations.scale_matrix(scale)
    return m @ trimesh.transformations.translation_matrix(-center)


def normalize_pair(gt_mesh: trimesh.Trimesh, pred_mesh: trimesh.Trimesh):
    """Copies of both meshes in the GT's unit-cube frame."""
    m = gt_frame(gt_mesh)
    gt_n, pred_n = gt_mesh.copy(), pred_mesh.copy()
    gt_n.apply_transform(m)
    pred_n.apply_transform(m)
    return gt_n, pred_n


# --- watertight status --------------------------------------------------------


def is_watertight(mesh: trimesh.Trimesh) -> bool:
    """Prediction validity (CADENA): the whole mesh is closed and a volume."""
    try:
        return bool(mesh.is_watertight and mesh.is_volume)
    except Exception:
        return False


def is_valid_gt(mesh: trimesh.Trimesh) -> bool:
    """GT validity for boolean IoU (CADENA): closed and |volume| > 0."""
    try:
        return bool(mesh.is_watertight and abs(float(mesh.volume)) > 0)
    except Exception:
        return False


def repair_gt(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, str | None]:
    """Repair a GT that is not a valid solid, if a known, checkable defect explains it.

    Only GT that fails `is_valid_gt` is touched, and a repair is kept only when
    its result passes `is_valid_gt`; otherwise the mesh is returned unchanged
    with the reason. Predictions are never repaired: their defects are the
    model's to answer for.

    Known defect: duplicated triangles. A double-sided export stores every
    triangle twice, once per orientation, so every edge has four faces: nothing
    is closed, a parity fill sees two crossings per wall (voxel IoU 0) and GMS
    sees both normals. One copy of each vertex triple is kept, then winding and
    outward normals are fixed.

    Returns ``(mesh, note)``: note is None when nothing was needed, otherwise
    what was done (``dedup_faces:<n removed>``) or why not (``repair_failed:...``).
    """
    if is_valid_gt(mesh):
        return mesh, None
    key = np.sort(mesh.faces, axis=1)
    _, first = np.unique(key, axis=0, return_index=True)
    n_dup = len(mesh.faces) - len(first)
    if n_dup == 0:
        return mesh, "repair_failed:no_known_defect"
    fixed = mesh.copy()
    fixed.update_faces(np.isin(np.arange(len(mesh.faces)), first))
    fixed.remove_unreferenced_vertices()
    trimesh.repair.fix_winding(fixed)
    trimesh.repair.fix_normals(fixed)
    if not is_valid_gt(fixed):
        return mesh, f"repair_failed:dedup_faces:{n_dup}_still_open"
    return fixed, f"dedup_faces:{n_dup}"


# --- CD -----------------------------------------------------------------------


def compute_cd(gt_mesh, pred_mesh, n_points: int = CD_N_POINTS):
    """Symmetric Chamfer distance x1000. ``(cd, precision, recall)``."""
    gt_points, _ = trimesh.sample.sample_surface(gt_mesh, n_points, seed=METRICS_SAMPLE_SEED)
    pred_points, _ = trimesh.sample.sample_surface(pred_mesh, n_points, seed=METRICS_SAMPLE_SEED)
    gt_distance, _ = cKDTree(gt_points).query(pred_points, k=1)
    pred_distance, _ = cKDTree(pred_points).query(gt_points, k=1)
    precision = float(np.mean(np.square(gt_distance)))
    recall = float(np.mean(np.square(pred_distance)))
    return (precision + recall) * CD_SCALE, precision * CD_SCALE, recall * CD_SCALE


# --- boolean IoU --------------------------------------------------------------


def merge_bodies(parts):
    """Union of overlapping bodies, so shared volume is not counted twice."""
    if len(parts) < 2:
        return parts
    try:
        merged = trimesh.boolean.union(parts)
    except Exception:
        return parts
    if merged is None or not merged.is_volume:
        return parts
    return [merged]


def solid_bodies(mesh):
    """Closed bodies with volume, merged, with their cavities cut out.

    Deviation from the reference, which keeps only `is_volume` parts: a closed
    inward-facing shell (negative volume) is the wall of a cavity, and dropping
    it fills the cavity in, so a hollowed prediction reads up to several percent
    too much volume. The shells are inverted and subtracted from the merged
    body; if that fails, the reference result stands.
    """
    parts = [m for m in mesh.split() if m.is_watertight]
    bodies = merge_bodies([m for m in parts if m.is_volume])
    cavities = []
    for m in parts:
        if not m.is_volume and m.is_winding_consistent and m.volume < 0:
            c = m.copy()
            c.invert()
            cavities.append(c)
    if len(bodies) != 1 or not cavities:
        return bodies
    try:
        cut = trimesh.boolean.difference([bodies[0], *cavities])
    except Exception:
        return bodies
    if cut is None or cut.is_empty or not cut.is_volume:
        return bodies
    return [cut]


def iou_components(mesh):
    """GT bodies: winding fixed, an inside-out mesh inverted, then `solid_bodies`."""
    if mesh.is_watertight and not mesh.is_winding_consistent:
        mesh = mesh.copy()
        trimesh.repair.fix_winding(mesh)
    if mesh.is_watertight and mesh.volume < 0:
        mesh = mesh.copy()
        mesh.invert()
    return solid_bodies(mesh)


def compute_iou(gt_mesh, pred_mesh):
    """Boolean IoU (manifold3d). ``(iou, iog, iop)`` or three ``None``."""
    try:
        gt_meshes = iou_components(gt_mesh)
        pred_meshes = solid_bodies(pred_mesh)
        if not gt_meshes or not pred_meshes:
            return None, None, None

        intersection_volume = 0.0
        for gt_i in gt_meshes:
            for pred_i in pred_meshes:
                inter = gt_i.intersection(pred_i)
                if inter is None or inter.is_empty or len(inter.faces) == 0:
                    continue
                vol = inter.volume
                intersection_volume += vol if np.isfinite(vol) else 0.0

        gt_volume = sum(m.volume for m in gt_meshes)
        pred_volume = sum(m.volume for m in pred_meshes)
        union_volume = gt_volume + pred_volume - intersection_volume
        if union_volume <= 0 or gt_volume <= 0 or pred_volume <= 0:
            return None, None, None
        if intersection_volume < 0:
            return None, None, None
        if intersection_volume > min(gt_volume, pred_volume) * (1.0 + INTERSECTION_EXCESS_TOL):
            return None, None, None

        iou = intersection_volume / union_volume
        iog = intersection_volume / gt_volume
        iop = intersection_volume / pred_volume
        if iou > IOU_REJECT_ABOVE:
            return None, None, None
        return (float(np.clip(iou, 0.0, 1.0)), float(np.clip(iog, 0.0, 1.0)),
                float(np.clip(iop, 0.0, 1.0)))
    except Exception:
        logger.exception("boolean IoU failed")
        return None, None, None


# --- voxel IoU (fallback) -----------------------------------------------------


def _or_block(target: np.ndarray, mesh, axes) -> None:
    """OR the occupancy of `mesh` into `target`, filling only its own bounding box."""
    sl = []
    for k in range(3):
        a = axes[k]
        lo = max(int(np.searchsorted(a, mesh.bounds[0][k])) - 1, 0)
        hi = min(int(np.searchsorted(a, mesh.bounds[1][k])) + 1, len(a))
        if hi - lo < 2:
            return
        sl.append(slice(lo, hi))
    sub = [axes[k][sl[k]] for k in range(3)]
    target[tuple(sl)] |= compare.occupancy(mesh, sub)


def body_occupancy(mesh, axes) -> np.ndarray:
    """Occupancy that respects bodies: closed bodies OR-ed, cavities cut, open rest voted.

    A parity fill of the whole mesh (compare.occupancy) counts the overlap of two
    overlapping bodies as outside: it is crossed twice. Here each closed
    component is filled on its own and OR-ed in, a closed inward-facing one (a
    cavity wall) is subtracted, and the open components, for which no
    orientation is trustworthy, are filled together by the three-axis majority.
    """
    if mesh.is_watertight and mesh.is_winding_consistent and mesh.volume < 0:
        mesh = mesh.copy()
        mesh.invert()
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return compare.occupancy(mesh, axes)
    occ = np.zeros([len(a) for a in axes], dtype=bool)
    cavity = np.zeros_like(occ)
    loose = []
    for p in parts:
        if p.is_watertight and p.is_winding_consistent:
            _or_block(cavity if p.volume < 0 else occ, p, axes)
        else:
            loose.append(p)
    if loose:
        _or_block(occ, trimesh.util.concatenate(loose), axes)
    return occ & ~cavity


def voxel_iou(gt_mesh, pred_mesh, cells: int = compare.VOXEL_CELLS):
    """Occupancy IoU on one lattice spanning both meshes. ``(iou, iog, iop, cells)``.

    The lattice is sized by the GT and refined (x2, up to compare.VOXEL_CELLS_MAX)
    while it misjudges a closed GT's solid volume (merged bodies, cavities cut)
    by more than compare.GRID_ERR_PCT; an open GT has no exact volume to check
    against and stays at `cells`.
    """
    ref_volume = None
    if is_valid_gt(gt_mesh):
        bodies = iou_components(gt_mesh)
        ref_volume = float(sum(b.volume for b in bodies)) if bodies else None
    while True:
        pitch = float(max(gt_mesh.extents) / cells)
        lo = np.minimum(gt_mesh.bounds[0], pred_mesh.bounds[0]) - pitch
        hi = np.maximum(gt_mesh.bounds[1], pred_mesh.bounds[1]) + pitch
        axes = compare._lattice(lo, hi, pitch, compare.JITTER)
        og = body_occupancy(gt_mesh, axes)
        if not ref_volume or cells >= compare.VOXEL_CELLS_MAX:
            break
        grid_err = abs(float(og.sum()) * pitch ** 3 - ref_volume) / ref_volume * 100.0
        if not grid_err > compare.GRID_ERR_PCT:
            break
        cells *= 2
    op = body_occupancy(pred_mesh, axes)
    inter = int(np.count_nonzero(og & op))
    union = int(np.count_nonzero(og | op))
    n_gt, n_pred = int(og.sum()), int(op.sum())
    iou = inter / union if union else 0.0
    iog = inter / n_gt if n_gt else None
    iop = inter / n_pred if n_pred else None
    return float(iou), iog, iop, int(cells)


# --- GMS ----------------------------------------------------------------------


@contextmanager
def _seeded_sampling(base: int):
    """Make `Trimesh.sample` deterministic while a GMS handler is built.

    The snapshot seeds the global numpy RNG (`np.random.seed(seed + N)`), but
    trimesh >= 4 samples through `np.random.default_rng(seed)` and ignores it,
    so GMS varied by about 0.001 between identical calls (CADENA's included,
    on this environment). The same seed the snapshot intends is passed to
    trimesh directly; `gms.py` itself stays unmodified.
    """
    original = trimesh.Trimesh.sample

    def sample(self, count, return_index=False, face_weight=None, seed=None):
        if seed is None:
            seed = (int(base) + int(count)) % (2 ** 32)
        return original(self, count, return_index=return_index, face_weight=face_weight, seed=seed)

    trimesh.Trimesh.sample = sample
    try:
        yield
    finally:
        trimesh.Trimesh.sample = original


def gms_handler(mesh, n_points: int = GMS_N_POINTS, cube_trick: bool = True,
                autofix_sampling: bool = False):
    with _seeded_sampling(METRICS_SAMPLE_SEED):
        return TrimeshHandler(
            mesh,
            Np=n_points,
            use_cube_trick=cube_trick,
            how_many_medians=GMS_HOW_MANY_MEDIANS,
            max_possible_N=GMS_MAX_POSSIBLE_N,
            compute_median=autofix_sampling,
            sample_seed=METRICS_SAMPLE_SEED,
        )


def compute_gms(gt_mesh, pred_mesh, n_points: int = GMS_N_POINTS, n_angles: int = GMS_N_ANGLES,
                rel_dist_tol: float = GMS_REL_DIST_TOL, cube_trick: bool = True,
                upper_bound_tol_rt: int = GMS_UPPER_BOUND_TOL_RT,
                autofix_sampling: bool = False):
    """GMS. ``(gms_raw, gms_norm)``, raw is an AOC (lower is better)."""
    tol_angles = np.linspace(0, upper_bound_tol_rt, n_angles)
    gt_handler = gms_handler(gt_mesh, n_points, cube_trick, autofix_sampling)
    pred_handler = gms_handler(pred_mesh, n_points, cube_trick, autofix_sampling)

    absolute_distance_threshold = rel_dist_tol * gt_handler.relative_unit_length("Lmax")
    if autofix_sampling:
        gt_handler.autofix_sampling(absolute_distance_threshold)
        pred_handler.autofix_sampling(absolute_distance_threshold)

    def _match_scores(mh_a, mh_b):
        query_k = max(1, len(mh_b.pc) // 100)
        return A_in_B_ball_matching_multiangle_v3(
            mh_A=mh_a,
            mh_B=mh_b,
            distance_threshold=absolute_distance_threshold,
            angle_tolerances=tol_angles,
            use_abs_in_cos=False,
            query_k=query_k,
            max_query_k=query_k,
        )

    gt_in_pred_scores = _match_scores(gt_handler, pred_handler)
    pred_in_gt_scores = _match_scores(pred_handler, gt_handler)

    recall = gt_in_pred_scores.sum(axis=1) / gt_in_pred_scores.shape[1]
    precision = pred_in_gt_scores.sum(axis=1) / pred_in_gt_scores.shape[1]

    inv_recall = np.divide(1.0, recall, out=np.zeros_like(recall), where=recall > 0)
    inv_precision = np.divide(1.0, precision, out=np.zeros_like(precision), where=precision > 0)
    denom = inv_recall + inv_precision - 1.0
    valid = (recall > 0) & (precision > 0) & np.isfinite(denom) & (denom != 0)
    gms_scores_all = np.divide(1.0, denom, out=np.zeros_like(denom), where=valid)

    stop_index = next((i for i, s in enumerate(gms_scores_all) if i > 0 and s >= 0.99), None)
    gms_scores = gms_scores_all if stop_index is None else gms_scores_all[: stop_index + 1]

    number_of_angles = len(gms_scores)
    tol_angles = tol_angles[:number_of_angles]
    max_possible_area = 0.0 if number_of_angles < 2 else tol_angles[-1] - tol_angles[0]
    aoc = float(max_possible_area - np.trapezoid(gms_scores, tol_angles))
    return aoc, float(1 - aoc / upper_bound_tol_rt)


# --- the pair -----------------------------------------------------------------


def _note(result: PairMetrics, what: str, exc: Exception) -> None:
    logger.exception("%s failed for %s", what, result.figure_id or result.pred_path)
    result.error = f"{result.error or ''} {what}: {type(exc).__name__}: {exc}".strip()


def score_pair(gt_path: str | Path, pred_path: str | Path, figure_id: str | None = None,
               gt_mesh: trimesh.Trimesh | None = None, repair: bool = True,
               save_repaired: str | Path | None = None,
               pred_frame: str = PRED_FRAME_GT) -> PairMetrics:
    """All metrics of one pair. ``gt_mesh`` may be passed already loaded (raw frame).

    ``repair`` runs `repair_gt` on the GT; ``save_repaired`` is a path the
    repaired GT is exported to when a repair was applied. ``pred_frame`` says
    where the prediction lives (see PRED_FRAMES); it is mapped into the GT's
    frame before scoring.
    """
    t0 = time.time()
    result = PairMetrics(gt_path=str(gt_path), pred_path=str(pred_path), figure_id=figure_id)
    try:
        gt = gt_mesh if gt_mesh is not None else load_mesh(gt_path)
    except Exception as exc:
        result.failure = FAILURE_LOAD_ERROR
        _note(result, "load gt", exc)
        return result
    if repair:
        gt, result.gt_repair = repair_gt(gt)
        if save_repaired and result.gt_repair and not result.gt_repair.startswith("repair_failed"):
            Path(save_repaired).parent.mkdir(parents=True, exist_ok=True)
            gt.export(str(save_repaired))
    result.gt_faces = int(len(gt.faces))
    result.gt_watertight = is_valid_gt(gt)

    if not Path(pred_path).is_file():
        result.failure = FAILURE_NO_PREDICTION
        return result
    try:
        pred = load_mesh(pred_path)
    except Exception as exc:
        result.failure = FAILURE_LOAD_ERROR
        _note(result, "load pred", exc)
        return result
    result.pred_faces = int(len(pred.faces))
    if len(gt.faces) == 0 or len(pred.faces) == 0:
        result.failure = FAILURE_EMPTY
        return result
    result.pred_watertight = is_watertight(pred)
    if pred_frame == PRED_FRAME_UNIT200:
        pred = pred.copy()
        pred.apply_transform(np.linalg.inv(gt_frame(gt)) @ UNIT200_TO_CUBE)

    gt, pred = normalize_pair(gt, pred)

    try:
        result.gms_raw, result.gms_norm = compute_gms(gt, pred)
    except Exception as exc:
        _note(result, "gms", exc)

    for name, n_points in CD_VERSIONS.items():
        try:
            cd, precision, recall = compute_cd(gt, pred, n_points)
        except Exception as exc:
            _note(result, name, exc)
            continue
        setattr(result, name, cd)
        setattr(result, f"{name}_precision", precision)
        setattr(result, f"{name}_recall", recall)

    if result.gt_watertight and result.pred_watertight:
        iou, iog, iop = compute_iou(gt, pred)
        if iou is not None:
            result.iou, result.iog, result.iop = iou, iog, iop
            result.iou_method = IOU_BOOLEAN
        else:
            result.iou_fallback_reason = "boolean_rejected"
    else:
        result.iou_fallback_reason = ("gt_not_watertight" if not result.gt_watertight
                                      else "pred_not_watertight")
    if result.iou is None:
        try:
            result.iou, result.iog, result.iop, result.voxel_cells = voxel_iou(gt, pred)
            result.iou_method = IOU_VOXEL
        except Exception as exc:
            _note(result, "voxel iou", exc)

    result.elapsed_s = round(time.time() - t0, 2)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("gt")
    ap.add_argument("pred")
    ap.add_argument("--id", default=None)
    ap.add_argument("--no-gt-repair", action="store_true", help="score the GT exactly as loaded")
    ap.add_argument("--pred-frame", choices=PRED_FRAMES, default=PRED_FRAME_GT)
    args = ap.parse_args()
    r = score_pair(args.gt, args.pred, figure_id=args.id, repair=not args.no_gt_repair,
                   pred_frame=args.pred_frame)
    out = {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
           for k, v in r.to_dict().items()}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
