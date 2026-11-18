"""Choose which attempt a retried part should keep, and which version of its
program a run delivers.

    python scripts/pick_best.py runs/<part>
    python scripts/pick_best.py --versions runs/<part>/run runs/<part>/part.stl

The runner retries a part until one attempt finishes -- judged by an empty
residual, never by a score. When one does, that attempt is the answer and this
changes nothing.

When NONE of them finishes, keeping whichever ran last is an accident rather
than a choice: attempts on the same part can differ wildly. This picks by the
smallest residual instead -- the same evidence the model itself is given,
available without any ground truth, and the only measure of "closer" the
harness is allowed to use.

Prints what it chose and why. Leaves the attempts in place; it only decides which
one `run/` should be.
"""

import functools
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mesh2cad.frame import declared_frame, describe_frame  # noqa: E402

# Two instruments print residuals and they print them differently.
#
# `sketch --check` prints one line:
#   "MISSING 4109.0 mm^3 in 2 bodies, EXTRA 31.2 mm^3 in 1 body"
# The standalone `residual` instrument prints a body table instead:
#   "MISSING (in the target, not yet built): 2 bodies"
#   "  D0      58082.568  ( -38.95, ...)"
# so the summary volume never appears on a line at all -- it has to be added up
# from the rows. Both forms are parsed.
VOLUME = re.compile(r"(MISSING|EXTRA)\s+(?:about\s+)?([0-9.]+)\s*mm\^3")
BODY_ROW = re.compile(r"^\s{2}[DE]\d+\s+([0-9.]+)\s")
# The table lists ten bodies a side and sums the rest on one line; leaving that
# line out under-counted every program with more than ten leftovers.
REST_ROW = re.compile(r"^\s{2}\.\.\. and \d+ smaller, ([0-9.]+) mm\^3 in all")
SECTION = re.compile(r"^(MISSING|EXTRA)\s+\(")


def parse(line):
    """Total unexplained volume on a residual line, or None.

    Missing and extra are added rather than netted: a feature in the wrong place
    contributes to both, and cancelling them would score it as perfect.
    """
    if not line:
        return None
    # A preview describes a program that was never written, so it is not
    # evidence about this attempt at all -- and it is the flattering kind of
    # non-evidence, since a preview of the finished part reads as zero.
    if "[PREVIEW:" in line:
        return None
    # Both say no body is above the floor; slivers are left out of every other
    # line too, so counting the fragments here would compare unlike with unlike.
    if ("nothing above the sliver floor" in line
            or "nothing SINGLE body is above the floor" in line):
        return 0.0
    found = VOLUME.findall(line)
    return sum(float(v) for _, v in found) if found else None


def parse_table(text):
    """Total unexplained volume from the `residual` instrument's body tables.

    Summed from the D0/E0 rows, because the instrument reports per body and
    never prints a total. "none" under both headings means nothing is left.
    Returns None when the output has no residual section at all -- a crash, or a
    program that would not build.
    """
    seen, total = False, 0.0
    for ln in text.splitlines():
        if SECTION.match(ln):
            seen = True
            continue
        row = BODY_ROW.match(ln) or REST_ROW.match(ln)
        if row:
            total += float(row.group(1))
    return total if seen else None


def badness(attempt, mesh, python, m2c):
    """What this attempt's FINAL program actually leaves unexplained.

    Measured here rather than read from the trace. `last_residual` is whichever
    check the model happened to run last, and a model that checks, then builds
    two more features, then stops, leaves a number describing a program that no
    longer exists.

    Falls back to the recorded line only when the program will not build, since
    a program that cannot run has no residual of its own to measure.
    """
    prog = attempt / "reconstruction.py"
    if prog.exists() and mesh.exists():
        try:
            proc = subprocess.run(
                [python, str(m2c), "residual", str(Path(mesh).resolve()), "--built",
                 str(prog.resolve())],
                capture_output=True, text=True, timeout=900, cwd=attempt)
            got = parse_table(proc.stdout)
            if got is not None:
                return got
        except Exception:
            pass
    try:
        trace = json.loads((attempt / "trace.json").read_text())
    except Exception:
        return None
    return parse(trace.get("last_residual"))


# --- versions inside one run ------------------------------------------------
#
# A run delivers run/reconstruction.py, which is whatever was written last --
# possibly a broken write after a valid version. Every version is already kept,
# with the residual its --check printed, in .m2c_history/reconstruction/ -- so
# choose there.

TARGET = re.compile(r"\(target volume (about )?([0-9.eE+-]+)\)")
PITCH = re.compile(r"measured on a ([0-9.eE+-]+) mm voxel grid")
OCCUPANCY = re.compile(r"missing about ([0-9.]+).*?extra about ([0-9.]+)")
SAME_TARGET = 0.01      # a check of another mesh than the part is not evidence


def _ingested(mesh):
    """The part as every check sees it -- `ingest`ed, so scaled and rotated
    the way the checks measured it -- or None when it cannot be read."""
    try:
        return _ingest_once(str(Path(mesh).resolve()))
    except Exception:                                              # noqa: BLE001
        return None


@functools.lru_cache(maxsize=4)
def _ingest_once(path):
    from mesh2cad.normalize import ingest
    return ingest(path)


def target_volume(mesh):
    """The part's volume as a check line prints it ("target volume ..."): in
    the frame ingest put it in, or None when the mesh cannot be read.

    The raw .stl's volume is not that number on a part ingest scaled: a 2 mm
    part is checked at x10, 1000 times the volume, and every check of it would
    read as a check of another mesh."""
    got = _ingested(mesh)
    v = float(got[0].volume) if got else 0.0
    return v if v > 0 else None


def target_area(mesh):
    """The part's surface area in the same frame, or None."""
    got = _ingested(mesh)
    return float(got[0].area) if got else None


def target_frame(mesh):
    """The frame ingest put the part in (4x4), or None when it neither scaled
    nor rotated it, or the mesh cannot be read."""
    got = _ingested(mesh)
    if not got:
        return None
    from mesh2cad.frame import ingest_frame
    return ingest_frame(got[1])


def same_target(line_volume, volume, about=False, pitch=None, area=None):
    """Whether a check line's target volume is this part's.

    An occupancy check prints the volume of the cells the target fills, which
    differs from the mesh's by up to a cell layer over its surface -- area x
    pitch -- and on an open target the mesh volume means little at all, so a
    fixed percentage is the wrong test. Derived, not picked."""
    tol = SAME_TARGET * volume
    if about and pitch and area:
        tol = max(tol, area * pitch)
    return abs(line_volume - volume) <= tol


def version_rows(run):
    """[(version, source, index row)] for the saved versions still on disk, in
    order. An undone version is renamed .undone and so is not offered; its
    number is reused, and the last index row for a number is the one that
    describes the file."""
    hist = Path(run) / ".m2c_history" / "reconstruction"
    idx = hist / "index.jsonl"
    if not idx.exists():
        return []
    rows = {}
    for ln in idx.read_text().splitlines():
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if isinstance(r.get("version"), int):
            rows[r["version"]] = r
    out = []
    for f in sorted(hist.glob("[0-9][0-9][0-9].py")):
        out.append((int(f.stem), f.read_text(), rows.get(int(f.stem), {})))
    return out


def version_badness(row, volume=None, area=None, frame=None, src=None):
    """(missing + extra, None) for a version its check measured as a valid
    solid against this part, else (None, why not).

    With `frame` (the part was scaled or rotated at ingest), a version whose
    source declares no `_FRAME` is not a candidate: export and scoring would
    take it as built, 10x or 100x the part."""
    if frame is not None and src is not None and declared_frame(src) is None:
        return None, (f"written without _FRAME, though the part was "
                      f"{describe_frame(frame)} at ingest: delivered, it would "
                      f"be scored in the wrong frame")
    if row.get("check_failed"):
        return None, f"its check failed ({row['check_failed']})"
    health = row.get("health") or ""
    if health.startswith("NOT A VALID SOLID"):
        return None, "not a valid solid"
    line = row.get("residual")
    if not line:
        return None, "never checked"
    if "NOT MEASURED" in line:
        return None, "not measured by its check"
    got = TARGET.search(line)
    pitch = PITCH.search(line)
    if volume and got and not same_target(float(got.group(2)), volume, bool(got.group(1)),
                                          pitch and float(pitch.group(1)), area):
        return None, (f"checked against another mesh (target volume "
                      f"{float(got.group(2)):g}, the part's is {volume:g})")
    b = parse(line)
    return (b, None) if b is not None else (None, "no residual on its line")


def measure(prog, mesh, python, m2c):
    """(missing + extra, None) for a program by the `residual` instrument, as
    `badness` measures an attempt, else (None, why not)."""
    # Absolute, because it runs in the program's folder: agent_run passes the
    # mesh as `part.stl`, relative to the run.
    prog, mesh = Path(prog).resolve(), Path(mesh).resolve()
    try:
        proc = subprocess.run(
            [python, str(m2c), "residual", str(mesh), "--built", str(prog)],
            capture_output=True, text=True, timeout=900, cwd=prog.parent)
    except Exception as exc:                                       # noqa: BLE001
        return None, f"residual could not run ({type(exc).__name__})"
    # A build that does not close is measured on a lattice; it is a measure
    # all the same, and often a better version than the last valid one.
    occ = OCCUPANCY.search(proc.stdout)
    if "OCCUPANCY" in proc.stdout and occ:
        return float(occ.group(1)) + float(occ.group(2)), None
    got = parse_table(proc.stdout)
    return (got, None) if got is not None else (None, "residual could not measure it")


# Versions with no measure against the part (never checked, or checked against
# another mesh) are measured here, newest first, at most this many: each is a
# `residual` run, up to minutes on a dense mesh, at the end of the run.
MEASURE_AT_MOST = 4


def _unmeasured(why):
    """Versions this part has no usable measure of. NOT MEASURED is among them:
    it said the check could not measure, not that the program is broken."""
    return (why in ("never checked", "not measured by its check")
            or why.startswith("checked against another mesh"))


def deliver_best(run, mesh, python=sys.executable, m2c=None):
    """Make run/reconstruction.py the best valid version the run saved.

    Best = the smallest missing + extra among versions measured as a valid
    solid against this part -- by their own --check, or here, by `residual`,
    for a version that was never checked, could not be measured by its check
    (it is measured by occupancy if it does not close), or was checked against
    another mesh. The later one wins a tie, so a finished run is never touched.
    A version whose own check said NOT A VALID SOLID, or whose check failed, is
    never chosen. The program on disk is kept unless a version beats it; the
    replaced one is kept as reconstruction.last.py. Returns a record of what was
    decided and why, and prints one line saying it.
    """
    run = Path(run)
    m2c = m2c or Path(__file__).resolve().parent / "m2c.py"
    prog = run / "reconstruction.py"
    hist = run / ".m2c_history" / "reconstruction"
    rec = {"delivered": "last", "why": ""}

    def say(why, **kw):
        rec.update(why=why, **kw)
        print(f"deliver: {why}")
        return rec

    if not prog.exists():
        return say("no program written")
    versions = version_rows(run)
    if not versions:
        return say("no saved versions; the program on disk is delivered")
    volume, area, frame = target_volume(mesh), target_area(mesh), target_frame(mesh)
    scored = [[v, src, *version_badness(row, volume, area, frame, src)]
              for v, src, row in versions]
    pending = [s for s in scored if s[2] is None and _unmeasured(s[3])][::-1]
    for s in pending[:MEASURE_AT_MOST]:
        got, why = measure(hist / f"{s[0]:03d}.py", mesh, python, m2c)
        s[2], s[3] = got, (f"{s[3]}; measured here" if got is not None
                           else f"{s[3]}; {why}")
    notes = []
    if pending:
        notes.append(f"{min(len(pending), MEASURE_AT_MOST)} version(s) with no check "
                     f"of this part measured here")
    if len(pending) > MEASURE_AT_MOST:
        notes.append(f"{len(pending) - MEASURE_AT_MOST} older one(s) not measured and "
                     f"not considered")
    if not volume:
        notes.append("part volume unreadable: checks of another mesh not caught")
    note = f" ({'; '.join(notes)})" if notes else ""

    valid = [s for s in scored if s[2] is not None]
    final = prog.read_text()
    mine = [s for s in scored if s[1] == final]
    if mine:
        cur_v, cur, cur_why = mine[-1][0], mine[-1][2], mine[-1][3]
    else:
        cur_v = None
        cur, why_m = measure(prog, mesh, python, m2c)
        cur_why = "not a saved version" + ("" if cur is not None else f"; {why_m}")
    if cur is not None and frame is not None and declared_frame(final) is None:
        cur, cur_why = None, (f"written without _FRAME, though the part was "
                              f"{describe_frame(frame)} at ingest")
    label = f"version {cur_v}" if cur_v else "the program on disk"
    if not valid and cur is None:
        return say(f"no saved version was measured as a valid solid; the last "
                   f"write is delivered{note}")
    best = min(reversed(valid), key=lambda s: s[2]) if valid else None
    if cur is not None and (best is None or cur <= best[2]):
        return say(f"{label} delivered: {cur:.1f} mm^3 left, no saved valid version "
                   f"leaves less{note}", version=cur_v, residual=cur)
    (run / "reconstruction.last.py").write_text(final)
    prog.write_text(best[1])
    was = f"{cur:.1f} mm^3 left" if cur is not None else cur_why
    return say(f"delivered version {best[0]} ({best[2]:.1f} mm^3 left) instead of "
               f"{label} ({was}); that one is kept as reconstruction.last.py{note}",
               delivered=best[0], version=best[0], residual=best[2],
               replaced=cur_v, replaced_why=was)


def candidates(work):
    """Every attempt this part produced, as (label, directory)."""
    out = []
    for d in sorted(work.iterdir()):
        if not d.is_dir():
            continue
        if d.name == "run" or d.name.startswith("attempt_"):
            # The NESTED copy wins when both exist. `cp -r run attempt_N` nests
            # when attempt_N already exists, leaving a stale program at the top
            # and the real one in attempt_N/run; reading the top one would pair
            # one program with another program's residual.
            nested = d / "run" / "trace.json"
            trace = nested if nested.exists() else d / "trace.json"
            if trace.exists():
                out.append((d.name, trace.parent))
    return out


def pick_attempt(work, python=sys.executable, m2c=None):
    """Make work/run the attempt to keep: the first that finished, else the one
    with the smallest residual. Attempts are work/run and work/attempt_<k>."""
    work = Path(work)
    m2c = Path(m2c) if m2c else Path(__file__).resolve().parent / "m2c.py"
    mesh = work / "part.stl"

    found = candidates(work)
    if len(found) < 2:
        print(f"pick_best: {len(found)} attempt(s), nothing to choose")
        return

    scored = []
    for label, d in found:
        try:
            trace = json.loads((d / "trace.json").read_text())
        except Exception as exc:
            print(f"pick_best: {label} unreadable ({exc}); skipped")
            continue
        b = badness(d, mesh, python, m2c)
        scored.append((label, d, trace.get("complete", False), b))
        shown = "complete" if trace.get("complete") else (
            f"{b:.1f} mm^3 unexplained" if b is not None else "never checked")
        print(f"pick_best: {label:12s} {shown}")

    done = [s for s in scored if s[2]]
    if done:
        print(f"pick_best: keeping {done[0][0]} -- it finished")
        best = done[0]
    else:
        # An attempt that never ran a check is not evidence of anything, so it
        # only wins if nothing else is left.
        measured = [s for s in scored if s[3] is not None]
        if not measured:
            print("pick_best: no attempt checked its work; keeping run/ as-is")
            return
        best = min(measured, key=lambda s: s[3])
        print(f"pick_best: no attempt finished; keeping {best[0]} "
              f"({best[3]:.1f} mm^3 unexplained, the smallest)")

    run = work / "run"
    if best[1] == run:
        return
    # Replace run/ with the winner so everything downstream -- scoring, videos,
    # the report -- reads one place and reads the same thing.
    if run.exists():
        shutil.rmtree(run, ignore_errors=True)
    shutil.copytree(best[1], run)
    print(f"pick_best: run/ is now {best[0]}")


def main():
    if sys.argv[1:2] == ["--versions"]:
        run, mesh = Path(sys.argv[2]), Path(sys.argv[3])
        python = sys.argv[4] if len(sys.argv) > 4 else sys.executable
        deliver_best(run, mesh, python)
        return
    pick_attempt(Path(sys.argv[1]),
                 sys.argv[2] if len(sys.argv) > 2 else sys.executable,
                 sys.argv[3] if len(sys.argv) > 3 else None)


if __name__ == "__main__":
    main()
