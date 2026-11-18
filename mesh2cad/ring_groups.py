"""The repeated primitives of a long ring, as groups.

An 84-tooth gear ring is 336 primitives, and `section` listed each one: 55-78k
characters a call, up to 712k, and the harness then cut the middle at 50k with
no idea what it was cutting. A tooth is one fact, said 84 times. So a long ring
is read as runs of congruent primitives, and a run prints as one line.

Fitting does not always cooperate: on a real gear the fit gives neighbouring
teeth different primitive sequences, and nothing repeats exactly. What is left
after grouping is therefore capped too, at the longest primitives, because those
are the ones a model names. Nothing is lost: `section --detail R#` lists the ring.
"""

import numpy as np

# Under this many primitives a ring is listed whole, as it always was.
GROUP_MIN_PRIMITIVES = 24
MAX_PERIOD = 64
MIN_REPEATS = 3
# One line has to replace at least this many.
MIN_RUN = 8
# Congruent within this fraction of the size, or SIZE_FLOOR mm for tiny ones.
SIZE_FRAC = 0.03
SIZE_FLOOR = 2e-3
SWEEP_FLOOR_DEG = 1.0
# How many primitives outside any group are listed one by one.
LIST_MAX = 40
# A run of starts lying this close (as a fraction of the step or radius) to a
# line or a circle has that pitch; anything looser gets no pitch at all.
PITCH_FRAC = 0.02


def _close(a, b, floor=SIZE_FLOOR):
    return abs(a - b) <= max(floor, SIZE_FRAC * max(abs(a), abs(b)))


def congruent(a, b):
    if a["kind"] != b["kind"]:
        return False
    if a["kind"] == "line":
        return _close(a["length"], b["length"])
    return (_close(a["radius"], b["radius"])
            and _close(a["sweep_deg"], b["sweep_deg"], SWEEP_FLOOR_DEG)
            and bool(a.get("ccw")) == bool(b.get("ccw")))


def _repeats(segs, i, p):
    """How many times the `p` primitives from `i` repeat, the first included.

    Always against the first period, so a slow drift cannot pass as a pattern.
    """
    k = 1
    while i + (k + 1) * p <= len(segs) and all(
            congruent(segs[i + j], segs[i + k * p + j]) for j in range(p)):
        k += 1
    return k


def find_runs(segs):
    """[(start, period, repeats), ...] in ring order, greedily longest first."""
    runs, i, n = [], 0, len(segs)
    while i < n:
        best = None
        for p in range(1, min(MAX_PERIOD, (n - i) // MIN_REPEATS) + 1):
            k = _repeats(segs, i, p)
            if (k >= MIN_REPEATS and p * k >= MIN_RUN
                    and (best is None or p * k > best[1] * best[2])):
                best = (i, p, k)
        if best:
            runs.append(best)
            i += best[1] * best[2]
        else:
            i += 1
    return runs


def pitch_of(points):
    """How the starts of a run step: a translation, a rotation, or "" if neither."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3 or not np.isfinite(pts).all():
        return ""
    steps = np.diff(pts, axis=0)
    mean = steps.mean(axis=0)
    size = float(np.hypot(*mean))
    if size > 0 and np.hypot(*(steps - mean).T).max() <= PITCH_FRAC * size:
        return f"step ({mean[0]:.4f}, {mean[1]:.4f})"
    a = np.column_stack([2 * pts, np.ones(len(pts))])
    b = (pts ** 2).sum(axis=1)
    sol = np.linalg.lstsq(a, b, rcond=None)[0]
    centre = sol[:2]
    rel = pts - centre
    radii = np.hypot(*rel.T)
    if radii.mean() <= 0 or np.abs(radii - radii.mean()).max() > PITCH_FRAC * radii.mean():
        return ""
    turns = np.diff(np.unwrap(np.arctan2(rel[:, 1], rel[:, 0])))
    deg = float(np.degrees(np.median(turns)))
    cx, cy = (round(float(c), 4) + 0.0 for c in centre)
    return f"pitch {abs(deg):.3f}deg about ({cx:.4f}, {cy:.4f})"


def _shape(sg):
    if sg["kind"] == "line":
        return f"L {sg['length']:.4f}"
    return f"A R{sg['radius']:.4f}"


def _span(segs, start, count):
    return f"{segs[start]['name']}..{segs[start + count - 1]['name']}"


def run_line(segs, run, at):
    """One run as one line: its shape, its pitch, and how its names go."""
    i, p, k = run
    pattern = ", ".join(_shape(sg) for sg in segs[i:i + p])
    starts = [at.get(segs[i + r * p]["from"]) for r in range(k)]
    pitch = pitch_of(starts) if all(starts) else ""
    last = i + (k - 1) * p
    names = (f"first {_span(segs, i, p)}, next {_span(segs, i + p, p)}, "
             f"last {_span(segs, last, p)}" if p > 1
             else f"{segs[i]['name']} to {segs[last]['name']}")
    return (f"    {k}x [{pattern}]" + (f" {pitch}," if pitch else ",") + f" {names}")


def _length(sg):
    if sg["kind"] == "line":
        return sg["length"]
    return sg["radius"] * np.radians(abs(sg["sweep_deg"]))


def group(segs):
    """(runs, listed, unlisted): what to print for one long ring.

    `listed` are the indices printed one by one, in ring order; `unlisted` is how
    many primitives neither a run nor that list covers.
    """
    runs = find_runs(segs)
    grouped = set()
    for i, p, k in runs:
        grouped.update(range(i, i + p * k))
    rest = [i for i in range(len(segs)) if i not in grouped]
    keep = sorted(sorted(rest, key=lambda i: -_length(segs[i]))[:LIST_MAX])
    return runs, keep, len(rest) - len(keep)
