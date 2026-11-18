"""Stage 4: turn a section polyline into named lines and arcs.

A mesh section is a dense polyline -- the intersection of a plane with triangles
-- so a true cylinder arrives as a fan of short chords and a true plane as a run
of nearly-collinear points. The job here is to invert the tessellation: recover
the primitives the CAD model was built from, name them, and hand the model a
vocabulary it can sketch with ("through N0, N1, then arc A2").

The approach is recursive split-and-fit rather than dynamic programming. DP over
a closed loop with a few thousand points is expensive, and the structure we are
recovering is strongly hierarchical: sharp corners are unambiguous breakpoints,
and between two corners the piece is nearly always a single line or a single arc.
So: break at corners, try to explain each piece with one primitive, and split at
the worst-fitting point only when that fails.

Tolerances are absolute millimetres and should be set from the tessellation
quality, not guessed -- a chord error of 0.01mm means anything below ~0.03mm is
noise and fitting it is overfitting.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Primitive:
    kind: str                 # "line" | "arc"
    p0: np.ndarray            # start point (u, v)
    p1: np.ndarray            # end point
    residual: float           # max deviation of the source points, mm
    n_points: int
    centre: np.ndarray = None  # arc only
    radius: float = None       # arc only
    ccw: bool = None           # arc only, sweep direction from p0 to p1
    sweep: float = None        # arc only, radians
    snapped: list = field(default_factory=list)   # human-readable snap notes

    def length(self):
        if self.kind == "line":
            return float(np.linalg.norm(self.p1 - self.p0))
        return float(abs(self.sweep) * self.radius)


# ---------------------------------------------------------------- fitting ----

def fit_line(pts):
    """Total-least-squares line through pts. Returns (point, unit direction)."""
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    return c, vt[0] / np.linalg.norm(vt[0])


def line_residual(pts, c, d):
    n = np.array([-d[1], d[0]])
    return float(np.abs((pts - c) @ n).max())


def fit_circle(pts, iters=64):
    """Kasa algebraic fit refined by the Landau geometric iteration.

    The algebraic fit alone is biased on short arcs -- exactly the case we hit
    on a fillet -- so the refinement is not optional.
    """
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([x, y, np.ones(len(pts))])
    b = x ** 2 + y ** 2
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None, None
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    c = np.array([cx, cy])
    disc = sol[2] + cx ** 2 + cy ** 2
    if disc <= 0:
        return None, None
    r = float(np.sqrt(disc))

    for _ in range(iters):
        d = pts - c
        dist = np.linalg.norm(d, axis=1)
        if (dist < 1e-12).any():
            break
        r_new = float(dist.mean())
        u = d / dist[:, None]
        # Landau step: c <- mean(p) + r * mean((c - p)/|c - p|)
        c_new = pts.mean(axis=0) + r_new * (-u).mean(axis=0)
        if np.linalg.norm(c_new - c) < 1e-12 and abs(r_new - r) < 1e-12:
            c, r = c_new, r_new
            break
        c, r = c_new, r_new
    return c, r


def circle_residual(pts, c, r):
    return float(np.abs(np.linalg.norm(pts - c, axis=1) - r).max())


# Enough samples that a metre-long edge is measured every millimetre, and few
# enough that the recursion stays cheap. Only long chords gain points.
MAX_FIT_SAMPLES = 2000


def densify(pts, step):
    """Insert points along any chord longer than `step`.

    A section outline is a POLYLINE, and its straight stretches carry no
    interior vertices at all -- a flat wall needs two triangles, so it arrives
    as a single edge however long it is. Measuring a fit only at the vertices
    therefore leaves those stretches completely unconstrained, and the residual
    that comes back is not a statement about the edge, it is a statement about
    two points.

    Vertices crowded into a corner fillet, then one far away across a flat
    face with nothing in between, fit a large circle with a tiny residual that
    bulges outside the part's own bounding box, midway along the gap where
    nothing was ever measured.

    A point on a chord IS a point on the boundary, so this adds no fiction. On
    a tessellated curve the chord midpoint sits inside the true arc by the
    tessellation sagitta -- far under the derived tolerance -- so real arcs are
    unaffected.

    It also fixes the weighting: least squares counts vertices, and a few of
    them inside a corner would outvote a long flat wall.
    """
    pts = np.asarray(pts, dtype=float)
    if step <= 0 or len(pts) < 2:
        return pts
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = int(np.linalg.norm(b - a) / step)
        if n > 1:
            out.extend(a + t * (b - a) for t in np.linspace(0.0, 1.0, n + 1)[1:-1])
        out.append(b)
    return np.asarray(out)


def align_rings(ring_a, ring_b):
    """Pair ring_b's primitives with ring_a's, or None if they do not match.

    Two sections of the same tapered prism decompose the same way, but the
    slicer is free to start the polygon anywhere, so index 0 of one need not be
    index 0 of the other. Every cyclic rotation is tried and the one with the
    least total displacement wins; if even the best leaves primitives a quarter
    of the ring apart, or the kinds do not line up, this refuses rather than
    pairing a line with an arc and calling it a taper.
    """
    sa, sb = ring_a["segments"], ring_b["segments"]
    if len(sa) != len(sb) or not sa:
        return None
    pa = np.array([s["from_xy"] for s in sa], dtype=float)
    diag = float(np.linalg.norm(pa.max(axis=0) - pa.min(axis=0))) or 1.0

    best, best_cost, best_spread = None, np.inf, np.inf
    for k in range(len(sb)):
        rot = sb[k:] + sb[:k]
        if any(x["kind"] != y["kind"] for x, y in zip(sa, rot, strict=True)):
            continue
        pb = np.array([s["from_xy"] for s in rot], dtype=float)
        d = np.linalg.norm(pa - pb, axis=1)
        cost = float(d.max())
        if cost < best_cost:
            # How UNEVEN the pairing is, which is the tell that matters. A real
            # taper moves every vertex by about the same amount, so the spread
            # is near 1. A pairing that lines up two rings whose fitter split an
            # edge in different places moves one vertex far more than the rest,
            # and a plain max-displacement test waves it through: the ring
            # emitted can have a last corner that misses the first, so `.close()`
            # adds a sliver edge and the loft has broken faces and no closed body.
            # The floor matters as much as the ratio. A bore that does not
            # taper moves no vertex at all, so the median displacement is zero
            # and any float noise in the last digit makes the ratio infinite --
            # which would refuse a pairing of two IDENTICAL rings. Anything under
            # a thousandth of the ring is not evidence of anything.
            med = max(float(np.median(d)), 1e-3 * diag)
            best_spread = cost / med
            best, best_cost = rot, cost
    if best is None or best_cost > 0.25 * diag or best_spread > 3.0:
        return None
    return best


def blend_ring(ring_a, ring_b, t):
    """One ring linearly blended between two measured rings.

    `t` is in units of the a->b interval: 0 is ring_a, 1 is ring_b, and outside
    [0, 1] it extrapolates. Returns None when the two rings cannot be paired.

    This exists because a loft cannot measure its own end profiles. A section
    taken exactly on a face is degenerate, so both profiles are measured a
    little way inside -- which costs a prism nothing, and costs a TAPER
    everything, because the section a fraction of the way along is not the
    section at the end: the taper comes out too shallow, proud at one end and
    short at the other.

    A ruled loft is linear by construction, so two measurements anywhere along
    it determine both ends exactly. Nothing is guessed: the extrapolation uses
    only what was measured, and it is exact for the draft and chamfer cases
    lofting is for. It is NOT exact for a curved taper -- and a curved taper is
    not a ruled loft either, so it was already the wrong operation.
    """
    if (ring_a.get("conformed") is not None
            and ring_a.get("conformed") == ring_b.get("conformed")
            and len(ring_a["segments"]) == len(ring_b["segments"])):
        # Rebuilt together, vertex for vertex: searching the pairing again
        # can find a shorter one across turned blades, or none.
        aligned = ring_b["segments"]
    else:
        aligned = align_rings(ring_a, ring_b)
    if aligned is None:
        return None
    sa = ring_a["segments"]
    n = len(sa)

    # Blend the ring's VERTICES, then rebuild each segment from them. Blending
    # every segment's own endpoints instead lets a segment's end and the next
    # one's start drift apart, and the ring emitted from that does not close --
    # closing the wire then draws a hundredth-of-a-millimetre edge across the
    # gap, and the sliver costs the whole solid. Rebuilding from a single vertex
    # list makes that impossible rather than unlikely.
    verts = []
    for x, y in zip(sa, aligned, strict=True):
        u = np.asarray(x["from_xy"], float)
        v = np.asarray(y["from_xy"], float)
        verts.append(u + t * (v - u))

    out = dict(ring_a)
    segs = []
    for i, (x, y) in enumerate(zip(sa, aligned, strict=True)):
        s = dict(x)
        s["from_xy"] = tuple(verts[i])
        s["to_xy"] = tuple(verts[(i + 1) % n])
        for key in ("centre",):
            u, v = x.get(key), y.get(key)
            if u is not None and v is not None:
                u, v = np.asarray(u, float), np.asarray(v, float)
                s[key] = tuple(u + t * (v - u))
        if x.get("radius") is not None and y.get("radius") is not None:
            s["radius"] = float(x["radius"] + t * (y["radius"] - x["radius"]))
        segs.append(s)
    out["segments"] = segs
    return out


# The smallest spacing, in millimetres, between two vertices a conformed ring
# may have. Emitted numbers carry four decimals, so anything much closer than
# this rounds two vertices onto one point and `.close()` adds a sliver edge.
_MIN_VERTEX_GAP = 5e-4


def ring_polyline(ring, rel_tol=5e-4):
    """The ring as a closed counter-clockwise polyline, without the repeat.

    Lines contribute their start vertex; arcs are sampled so that no chord
    strays more than `rel_tol` of the ring's size from the arc (never less than
    2e-4 mm, which the four-decimal grid could not show anyway).
    """
    segs = ring["segments"]
    ends = np.array([s["from_xy"] for s in segs], float)
    size = max(float(np.linalg.norm(ends.max(axis=0) - ends.min(axis=0))),
               max((2.0 * float(s.get("radius") or 0.0) for s in segs), default=0.0),
               1e-9)
    tol = max(2e-4, rel_tol * size)
    pts = []
    for s in segs:
        p0 = np.asarray(s["from_xy"], float)
        pts.append(p0)
        if s["kind"] != "arc":
            continue
        c = np.asarray(s["centre"], float)
        r = float(s["radius"])
        sweep = np.radians(abs(float(s["sweep_deg"])))
        # Same rule as emit.ring_code: a lone arc that ends where it started is
        # a circle, whatever sweep was recorded.
        span = float(np.linalg.norm(p0 - np.asarray(s["to_xy"], float)))
        if len(segs) == 1 and (abs(np.degrees(sweep) - 360) < 1
                               or span <= 1e-6 * max(1.0, r)):
            sweep = 2 * np.pi
        step = 2 * np.arccos(max(-1.0, 1.0 - tol / r)) if r > tol else np.pi / 2
        k = int(np.clip(np.ceil(sweep / step), 2, 90))
        a0 = np.arctan2(p0[1] - c[1], p0[0] - c[0])
        sign = 1.0 if s.get("ccw", True) else -1.0
        for j in range(1, k):
            a = a0 + sign * sweep * j / k
            pts.append(c + r * np.array([np.cos(a), np.sin(a)]))
    pts = np.array(pts)
    pts = pts[np.linalg.norm(pts - np.roll(pts, 1, axis=0), axis=1) > 1e-9]
    if _ring_area(pts) < 0:
        pts = pts[::-1]
    return pts


def _arclength(pts):
    closed = np.vstack([pts, pts[:1]])
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    return cum / cum[-1], float(cum[-1])


def _point_at(pts, u_vertices, u):
    closed = np.vstack([pts, pts[:1]])
    return np.column_stack([np.interp(u, u_vertices, closed[:, 0]),
                            np.interp(u, u_vertices, closed[:, 1])])


def ring_thinness(ring):
    """Mean width of the ring's region divided by its extent.

    2*area/perimeter is the mean width of a strip, so a sliver -- a crescent
    between two nearly equal arcs -- scores near zero while a thin-walled but
    real section keeps a few percent.
    """
    pts = ring_polyline(ring)
    _, perimeter = _arclength(pts)
    extent = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) or 1.0
    return 2.0 * abs(_ring_area(pts)) / max(perimeter, 1e-12) / extent


# A turn of the far ring this small is no turn: the overlap a rotation buys
# below it is the fit's own noise.
TWIST_MIN_DEG = 0.5


def twist_between(ring_a, ring_b, most_deg=45.0):
    """(degrees, overlap at 0, overlap turned): how far ring_b is turned about
    its own centre from ring_a, both centres put together -- the rotation that
    overlays b on a best, the least one of any that do as well (a 5-blade
    impeller overlays itself every 72 deg). 0 when turning buys nothing.
    Paired by arc length from where each ring happened to start, the starts
    of a twisted section can be far apart and the loft shears it into a star."""
    from shapely import affinity
    from shapely.geometry import Polygon

    pa = Polygon(ring_polyline(ring_a)).buffer(0)
    pb = Polygon(ring_polyline(ring_b)).buffer(0)
    if pa.is_empty or pb.is_empty:
        return 0.0, 0.0, 0.0
    ca, cb = pa.centroid, pb.centroid
    pb = affinity.translate(pb, ca.x - cb.x, ca.y - cb.y)

    def iou(t):
        q = affinity.rotate(pb, t, origin=ca)
        inter = pa.intersection(q).area
        return inter / (pa.union(q).area or 1.0)
    base = iou(0.0)
    coarse = np.arange(-most_deg, most_deg + 1e-9, 1.0)
    vals = np.array([iou(t) for t in coarse])
    top = float(vals.max())
    t0 = float(min(coarse[vals >= top - 0.005], key=abs))
    fine = np.arange(t0 - 1.0, t0 + 1.0 + 1e-9, 0.05)
    fv = np.array([iou(t) for t in fine])
    t1 = float(fine[int(np.argmax(fv))])
    best = float(fv.max())
    if abs(t1) < TWIST_MIN_DEG or best - base < 0.02:
        return 0.0, base, base
    # rotate() turns b by t onto a, so b is turned -t from a
    return -t1, base, best


def start_rotated(ring_a, ring_b, twist_deg=0.0):
    """(ring_b with the start vertex facing ring_a's, degrees it moved): the
    vertex of b nearest in angle, about b's centre, to a's start turned by
    the twist -- so a loft that pairs the rings from their starts pairs like
    with like."""
    segs = ring_b["segments"]
    if len(segs) < 2:
        return ring_b, 0.0
    pa = ring_polyline(ring_a)
    ca = pa.mean(axis=0)
    a0 = np.asarray(ring_a["segments"][0]["from_xy"], float) - ca
    want = np.arctan2(a0[1], a0[0]) + np.radians(twist_deg)
    cb = ring_polyline(ring_b).mean(axis=0)
    v = np.array([s["from_xy"] for s in segs], float) - cb
    ang = np.arctan2(v[:, 1], v[:, 0])
    off = np.abs((ang - want + np.pi) % (2 * np.pi) - np.pi)
    k = int(np.argmin(off))
    moved = float(np.degrees(abs((ang[k] - ang[0] + np.pi) % (2 * np.pi) - np.pi)))
    out = dict(ring_b)
    out["segments"] = segs[k:] + segs[:k]
    return out, moved


def conform_rings(rings, samples=128, twists=None):
    """Every ring rebuilt from straight lines, all with the SAME vertex count
    and with vertex i of each ring corresponding to vertex i of the next.

    This exists because OpenCascade's loft cannot be trusted to do it. Given
    sections with different edge counts, ThruSections splits edges itself by
    casting rays from each section's centre, and that only works for a section
    every ray crosses once. A U-shaped section is not one: the same U, resampled
    to two vertex counts and lofted onto itself, fails with `BRep_API:
    command not done`, and so does a loft between two measured sections
    whose fitter split them into different numbers of primitives.

    Correspondence is by normalised arc length, both rings turned
    counter-clockwise, with the start of each chosen by the cyclic shift that
    best overlays it on the ring before it once their centres and sizes are
    removed -- so a taper that also moves sideways is not read as a twist. The
    vertices of EVERY ring are kept, as parameters on every other ring, so no
    corner of any measured section is cut off; arcs become chords within
    `ring_polyline`'s tolerance.

    `twists` (degrees, one per ring, from `twist_between` against the first)
    turns each ring back by its twist before the overlay, so a section that
    turns as it goes is paired turned with it, not sheared.
    """
    polys = [ring_polyline(r) for r in rings]
    params = [_arclength(p) for p in polys]
    grid = np.arange(samples) / samples
    turns = [np.radians(float(t)) for t in (twists or [0.0] * len(rings))]

    def _normalised(p, u, shift, turn=0.0):
        q = _point_at(p, u, (grid + shift) % 1.0)
        q = q - q.mean(axis=0)
        if turn:
            c, s = np.cos(-turn), np.sin(-turn)
            q = q @ np.array([[c, s], [-s, c]])
        return q / (np.sqrt((q ** 2).sum(axis=1).mean()) or 1.0)

    starts, prev = [], None
    for p, (u, _), turn in zip(polys, params, turns, strict=True):
        if prev is None:
            starts.append(0.0)
            prev = _normalised(p, u, 0.0, turn)
            continue
        q = _normalised(p, u, 0.0, turn)
        cost = [float(((prev - np.roll(q, -k, axis=0)) ** 2).sum()) for k in range(samples)]
        coarse = int(np.argmin(cost)) / samples
        # The grid step is a whole 1/128 of the perimeter, which leaves matched
        # vertices visibly apart on anything but a coarse part. Refine within it.
        fine = coarse + np.linspace(-1.0, 1.0, 33) / samples
        cost = [float(((prev - _normalised(p, u, f, turn)) ** 2).sum()) for f in fine]
        start = float(fine[int(np.argmin(cost))]) % 1.0
        starts.append(start)
        prev = _normalised(p, u, start, turn)

    ts = np.sort(np.concatenate(
        [[0.0]] + [(u[:-1] - s) % 1.0 for (u, _), s in zip(params, starts, strict=True)]))
    gap = _MIN_VERTEX_GAP / min(length for _, length in params)
    keep = [0.0]
    for t in ts[1:]:
        if t - keep[-1] >= gap and 1.0 - t >= gap:
            keep.append(float(t))
    t = np.array(keep)

    out = []
    key = id(polys)
    for ring, p, (u, _), s in zip(rings, polys, params, starts, strict=True):
        pts = _point_at(p, u, (t + s) % 1.0)
        segs = [{"kind": "line",
                 "from_xy": tuple(float(v) for v in pts[i]),
                 "to_xy": tuple(float(v) for v in pts[(i + 1) % len(pts)])}
                for i in range(len(pts))]
        conformed = dict(ring)
        conformed["segments"] = segs
        conformed["n_primitives"] = len(segs)
        conformed["conformed"] = key          # vertex i pairs with vertex i
        out.append(conformed)
    return out


def fit_samples(pts):
    """The points a fit should actually be measured on: `pts`, gaps filled.

    The step is the run's own median chord, so a uniformly sampled run is
    returned untouched and only genuinely unsampled stretches gain points.
    """
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 3:
        return pts
    chords = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    step = float(np.median(chords))
    total = float(chords.sum())
    if step <= 0 or total <= 0:
        return pts
    return densify(pts, max(step, total / MAX_FIT_SAMPLES))


# ------------------------------------------------------------ segmentation ---

def turning_angles(pts, window=2):
    """Signed turn at each vertex, measured over a small window.

    A window wider than one edge is essential: on a tessellated arc each
    individual edge turns by a tiny amount and single-edge turning is dominated
    by quantisation noise, so corners and arcs become indistinguishable.
    """
    n = len(pts)
    ang = np.zeros(n)
    for i in range(n):
        a = pts[(i - window) % n]
        b = pts[i]
        c = pts[(i + window) % n]
        v1, v2 = b - a, c - b
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-12 or n2 < 1e-12:
            continue
        v1, v2 = v1 / n1, v2 / n2
        ang[i] = np.arctan2(v1[0] * v2[1] - v1[1] * v2[0], v1 @ v2)
    return ang


def estimate_tolerance(pts, k=3.0, floor=2e-4, ceiling=0.5, corner_deg=20.0):
    """Fit tolerance implied by how finely this polyline was sampled.

    A hardcoded tolerance is a guess about someone else's mesh. On a finely
    tessellated part, a fixed one can both hide a real small fillet and invent
    large arcs in its place -- confidently, with no signal that anything is
    wrong. The sampling density is right there in the data, so
    measure it instead.

    For a smooth run sampled from a circle of radius R at angular step theta,
    the chord is L = 2R*sin(theta/2) and the sagitta -- the most a chord can
    deviate from the true curve -- is approximately L*theta/8. Corners are
    excluded, because a real corner's turn says nothing about sampling density.
    The tolerance is a few sagittas: tight enough to keep real features, loose
    enough not to fit the tessellation itself.
    """
    pts = np.asarray(pts, float)
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    # Duplicate and near-duplicate vertices must go first. A zero-length edge
    # makes turning_angles skip the vertex, leaving its turn recorded as zero --
    # so a genuine 90 degree corner reads as smooth, a window straddling it is
    # asked to fit one circle through a right angle, and the residual it returns
    # drives the whole estimate to its ceiling.
    if len(pts) > 2:
        step = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)
        pts = pts[np.concatenate([[True], step[:-1] > 1e-9])]
    n = len(pts)
    if n < 8:
        return 0.05

    ang = np.abs(turning_angles(pts, window=1))
    keep = ang < np.radians(corner_deg)          # corners say nothing about sampling

    # How well is the polyline locally explained by the primitives we fit?
    #
    # Not deviation from a chord: on a curved run that is real curvature, which
    # an arc reproduces exactly, and counting it makes the tolerance so loose
    # that small fillets disappear. Not the sampling sagitta either: that
    # sits below the vertex noise of a flat face, and the fitter then explains
    # a straight cylinder wall with a huge arc.
    #
    # Fit a small circle to each local window and take its residual. On a
    # cleanly sampled arc or line that residual is ~0 regardless of how curved
    # or how coarse the run is; what remains is exactly the noise the fitter has
    # to tolerate.
    edge = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)
    w = 3
    stride = max(1, n // 400)
    res = []
    for i in range(0, n, stride):
        idx = [(i + j) % n for j in range(-w, w + 1)]
        # Skip any window CONTAINING a corner, not merely centred on one: one
        # circle cannot pass through a right angle, and on a profile with few
        # vertices most windows straddle one.
        if not keep[idx].all():
            continue
        # Skip windows that straddle a change of sampling density. A meridian
        # can carry a 0.03 mm edge on a fillet and a 44 mm edge on a flat wall;
        # a window spanning both is not describing noise, it is describing the
        # transition, and its residual is meaningless.
        e = edge[[(i + j) % n for j in range(-w, w)]]
        if e.min() <= 0 or e.max() / e.min() > 5.0:
            continue
        win = pts[idx]
        c, r = fit_circle(win)
        if c is not None and r is not None and np.isfinite(r) and r < 1e6:
            res.append(circle_residual(win, c, r))
        else:
            cc, dd = fit_line(win)
            res.append(line_residual(win, cc, dd))
    if len(res) < 4:
        return float(min(ceiling, max(floor, 0.01)))
    # A high percentile, not the median: the tolerance has to clear the
    # worst-sampled stretch, since that is the one that gets overfitted.
    return float(min(ceiling, max(floor, k * float(np.percentile(res, 90)))))


def find_corners(pts, corner_deg=18.0, window=2):
    """Indices of sharp vertices: local maxima of |turn| above the threshold."""
    ang = turning_angles(pts, window=window)
    mag = np.abs(ang)
    thresh = np.radians(corner_deg)
    n = len(pts)
    corners = []
    for i in range(n):
        if mag[i] < thresh:
            continue
        # keep only the local peak so one physical corner yields one index
        if mag[i] >= mag[(i - 1) % n] and mag[i] >= mag[(i + 1) % n]:
            corners.append(i)
    return corners


def _fit_piece(pts, tol, min_pts, arc_min_sweep_deg):
    """Explain one open piece with as few primitives as possible."""
    if len(pts) < 2:
        return []
    if len(pts) == 2:
        return [Primitive("line", pts[0], pts[-1], 0.0, 2)]

    # Fit and measure on the run's GEOMETRY, not on its vertex list. See
    # densify(): an unsampled straight stretch otherwise constrains nothing,
    # and a residual computed there is a fact about two points.
    q = fit_samples(pts)

    c, d = fit_line(q)
    res_line = line_residual(q, c, d)
    if res_line <= tol:
        return [Primitive("line", pts[0], pts[-1], res_line, len(pts))]

    best_arc, res_arc = None, np.inf
    if len(pts) >= max(min_pts, 4):
        cc, r = fit_circle(q)
        if cc is not None and r is not None and np.isfinite(r) and r > 1e-9:
            res_arc = circle_residual(q, cc, r)
            a0 = np.arctan2(pts[0][1] - cc[1], pts[0][0] - cc[0])
            a1 = np.arctan2(pts[-1][1] - cc[1], pts[-1][0] - cc[0])
            # Which of the two arcs between the endpoints is the one the points
            # actually lie on. Taking it from the accumulated turning angle
            # inverts on a nearly straight run: the turn is tiny, its sign is
            # noise, and the emitted arc then bulges on the WRONG side, out
            # past the part's own bounding box.
            #
            # Ask instead whether the mid-point of the run sits inside the
            # counter-clockwise sweep, which is exact however small the turn.
            amid = np.arctan2(q[len(q) // 2][1] - cc[1],
                              q[len(q) // 2][0] - cc[0])
            ccw = ((amid - a0) % (2 * np.pi)) <= ((a1 - a0) % (2 * np.pi))
            sweep = (a1 - a0) % (2 * np.pi) if ccw else (a0 - a1) % (2 * np.pi)
            if np.degrees(sweep) >= arc_min_sweep_deg:
                best_arc = Primitive("arc", pts[0], pts[-1], res_arc, len(pts),
                                     centre=cc, radius=float(r), ccw=ccw,
                                     sweep=float(sweep))

    if best_arc is not None and res_arc <= tol and res_arc < res_line:
        return [best_arc]

    # Neither primitive explains it: split at the worst point of the better fit
    # and recurse. Splitting on the better fit's residual puts the break where
    # the single-primitive assumption actually fails.
    if res_arc < res_line and best_arc is not None:
        dev = np.abs(np.linalg.norm(pts - best_arc.centre, axis=1) - best_arc.radius)
    else:
        nvec = np.array([-d[1], d[0]])
        dev = np.abs((pts - c) @ nvec)
    k = int(np.argmax(dev[1:-1])) + 1
    if k < min_pts or len(pts) - k < min_pts:
        k = len(pts) // 2
    return (_fit_piece(pts[:k + 1], tol, min_pts, arc_min_sweep_deg)
            + _fit_piece(pts[k:], tol, min_pts, arc_min_sweep_deg))


def decompose_ring(coords, tol=None, corner_deg=18.0, min_pts=4,
                   arc_min_sweep_deg=4.0):
    """Decompose one closed ring into lines and arcs.

    `coords` is an (N, 2) array of the ring's vertices, not repeating the first
    point at the end.
    """
    pts = np.asarray(coords, dtype=float)
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    if tol is None:
        tol = estimate_tolerance(pts)
    # Drop duplicate consecutive points; they break every angle computation.
    keep = np.ones(len(pts), bool)
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[i - 1]) < 1e-9:
            keep[i] = False
    pts = pts[keep]
    if len(pts) < 3:
        return []

    corners = find_corners(pts, corner_deg=corner_deg)
    n = len(pts)
    if not corners:
        # A fully smooth ring (a circle, say) has no corners: cut it in two so
        # each piece is an open arc that the fitter can handle.
        corners = [0, n // 2]

    prims = []
    for a, b in zip(corners, corners[1:] + [corners[0] + n]):
        idx = np.arange(a, b + 1) % n
        prims.extend(_fit_piece(pts[idx], tol, min_pts, arc_min_sweep_deg))
    return merge(prims, tol)


def _arc_points(a, n=16):
    """n points along arc a, from p0 to p1 in its own direction."""
    t0 = np.arctan2(*(a.p0 - a.centre)[::-1])
    t = t0 + (1 if a.ccw else -1) * abs(a.sweep) * np.linspace(0.0, 1.0, n)
    return a.centre + a.radius * np.column_stack([np.cos(t), np.sin(t)])


def _arc_union(q, p, tol):
    """One arc for two neighbours that are pieces of one circle, or None.

    The 1% rule in `_merged` joins arcs that came out the same circle. On a
    coarse mesh the pieces of one fillet each fit their own few points, and
    come back as several arcs of different radii side by side. Two
    neighbours are one arc when a single circle runs within the fit tolerance
    of both -- counting what each already missed the section by.
    """
    pts = np.vstack([_arc_points(q), _arc_points(p)])
    c, r = fit_circle(pts)
    if c is None:
        return None
    dev = circle_residual(pts, c, r) + max(q.residual, p.residual)
    if dev > tol:
        return None
    return Primitive("arc", q.p0, p.p1, dev, q.n_points + p.n_points,
                     centre=c, radius=r, ccw=q.ccw, sweep=q.sweep + p.sweep)


def _merged(q, p, tol):
    """Primitive p absorbed into its predecessor q, or None if they are two."""
    if q.kind == "line" and p.kind == "line":
        d1, d2 = q.p1 - q.p0, p.p1 - p.p0
        n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
        if (n1 > 1e-12 and n2 > 1e-12
                and abs(float(d1 @ d2) / (n1 * n2)) > np.cos(np.radians(1.0))):
            return Primitive("line", q.p0, p.p1, max(q.residual, p.residual),
                             q.n_points + p.n_points)
        return None
    if q.kind == "arc" and p.kind == "arc" and q.ccw == p.ccw:
        if (abs(q.radius - p.radius) < max(tol, 0.01 * q.radius)
                and np.linalg.norm(q.centre - p.centre) < max(tol, 0.01 * q.radius)):
            return Primitive("arc", q.p0, p.p1, max(q.residual, p.residual),
                             q.n_points + p.n_points,
                             centre=(q.centre + p.centre) / 2,
                             radius=(q.radius + p.radius) / 2, ccw=q.ccw,
                             sweep=q.sweep + p.sweep)
        return _arc_union(q, p, tol)
    return None


def merge(prims, tol, closed=True):
    """Join adjacent primitives that were split unnecessarily.

    The wrap-around case matters more than it looks: the ring's first point is
    wherever the mesh section happened to start, which is usually the middle of
    an edge rather than a corner. Without closing the loop, a plain rectangle
    comes back as five lines with one edge arbitrarily cut in two.
    """
    if len(prims) < 2:
        return prims
    out = [prims[0]]
    for p in prims[1:]:
        m = _merged(out[-1], p, tol)
        if m is not None:
            out[-1] = m
        else:
            out.append(p)

    # Close the ring: the split between the last and first primitive is an
    # artefact of where the polyline happened to start, not a real vertex.
    while closed and len(out) > 2:
        m = _merged(out[-1], out[0], tol)
        if m is None:
            break
        out[0] = m
        out.pop()
    return out


# ---------------------------------------------------------------- snapping ---

def snap(prims, angle_tol_deg=1.5, radius_tol=0.05, grid=None):
    """Nudge near-regular geometry onto exact regularity.

    This is what separates a CAD reconstruction from a mesh trace: a face that
    is 0.4 degrees off axis was axis-aligned in the original and is off only
    because of tessellation. Every adjustment is recorded on the primitive so
    nothing is changed silently.
    """
    axes = {0: np.array([1.0, 0.0]), 1: np.array([0.0, 1.0])}
    for p in prims:
        if p.kind != "line":
            continue
        d = p.p1 - p.p0
        L = np.linalg.norm(d)
        if L < 1e-12:
            continue
        d = d / L
        for ax, v in axes.items():
            ang = np.degrees(np.arccos(min(1.0, abs(float(d @ v)))))
            if 0 < ang <= angle_tol_deg:
                mid = (p.p0 + p.p1) / 2
                sign = np.sign(float(d @ v)) or 1.0
                p.p0 = mid - v * sign * L / 2
                p.p1 = mid + v * sign * L / 2
                p.snapped.append(f"axis-aligned to {'UV'[ax]} ({ang:.2f}deg)")
                break

    radii = [p.radius for p in prims if p.kind == "arc"]
    for i, p in enumerate(prims):
        if p.kind != "arc":
            continue
        near = [r for r in radii if abs(r - p.radius) <= radius_tol]
        if len(near) > 1:
            r_new = float(np.mean(near))
            if abs(r_new - p.radius) > 1e-9:
                p.snapped.append(f"radius {p.radius:.4f} -> {r_new:.4f} (equalised)")
                p.radius = r_new

    if grid:
        for p in prims:
            for attr in ("p0", "p1", "centre"):
                v = getattr(p, attr, None)
                if v is None:
                    continue
                snapped_v = np.round(np.asarray(v) / grid) * grid
                if np.linalg.norm(snapped_v - v) > 1e-12 and np.linalg.norm(snapped_v - v) <= grid / 2:
                    setattr(p, attr, snapped_v)
                    p.snapped.append(f"{attr} snapped to {grid}mm grid")
    return prims


# ------------------------------------------------------------------ naming ---

def name_ring(prims, ring_id=0, node_start=0):
    """Assign N-names to junctions, L/A-names to primitives, R to the ring.

    The ring is the thing `--use` actually selects -- naming a primitive only
    ever meant "the ring this belongs to" -- and without a name of its own the
    caller has to guess which index happens to be a line: `--use "L0_0"` fails
    on a ring whose segment 0 is an arc. R0 says what was meant.
    """
    nodes, segs = [], []
    for i, p in enumerate(prims):
        nodes.append({"name": f"N{node_start + i}",
                      "uv": [round(float(p.p0[0]), 4), round(float(p.p0[1]), 4)]})
    for i, p in enumerate(prims):
        a = f"N{node_start + i}"
        b = f"N{node_start + (i + 1) % len(prims)}"
        # Endpoint coordinates travel with the segment, not just the node names:
        # code generation needs the numbers, and resolving them back through the
        # node table is an easy place to introduce an off-by-one.
        xy = {"from_xy": [round(float(p.p0[0]), 4), round(float(p.p0[1]), 4)],
              "to_xy": [round(float(p.p1[0]), 4), round(float(p.p1[1]), 4)]}
        if p.kind == "line":
            segs.append({"name": f"L{ring_id}_{i}", "kind": "line", "from": a, "to": b,
                         **xy,
                         "length": round(p.length(), 4),
                         "residual": round(p.residual, 5),
                         "snapped": p.snapped})
        else:
            segs.append({"name": f"A{ring_id}_{i}", "kind": "arc", "from": a, "to": b,
                         **xy,
                         "radius": round(float(p.radius), 4),
                         "centre": [round(float(p.centre[0]), 4), round(float(p.centre[1]), 4)],
                         "sweep_deg": round(float(np.degrees(p.sweep)), 3),
                         "ccw": bool(p.ccw),
                         "residual": round(p.residual, 5),
                         "snapped": p.snapped})
    return nodes, segs


def decompose_polygon(poly, tol=None, grid=None, node_start=0, ring_start=0,
                      cut=None, **kw):
    """Decompose a shapely Polygon (exterior + holes) into named primitives.

    node_start/ring_start let a caller number several disjoint loops of one
    section continuously, so N-names are unique across the whole sketch rather
    than restarting per loop.
    """
    rings = []
    node_i, ring_i = node_start, ring_start
    for ring, role in [(poly.exterior, "outer")] + [(r, "hole") for r in poly.interiors]:
        coords = np.asarray(ring.coords)
        ring_tol = estimate_tolerance(coords) if tol is None else tol
        prims = decompose_ring(coords, tol=ring_tol, **kw)
        if cut is not None:
            prims = repair_after_cut(prims, ring_tol, cut[0], cut[1])
        prims = snap(prims, grid=grid)
        nodes, segs = name_ring(prims, ring_id=ring_i, node_start=node_i)
        rings.append({"role": role, "name": f"R{ring_i}",
                      "nodes": nodes, "segments": segs,
                      "tol": round(ring_tol, 5),
                      "n_primitives": len(prims),
                      "n_lines": sum(1 for p in prims if p.kind == "line"),
                      "n_arcs": sum(1 for p in prims if p.kind == "arc"),
                      "max_residual": round(max([p.residual for p in prims], default=0.0), 5),
                      "_prims": prims})
        node_i += len(nodes)
        ring_i += 1
    return rings


def detect_patterns(rings, radius_tol=0.05, angle_tol_deg=0.5):
    """Group equal-radius circular holes and recognise bolt-circle patterns.

    Given a correct list of hole centres, a model computing the pitch-circle
    start angle by hand can get it wrong and rotate every hole in the
    reconstruction. Reporting the pattern directly removes the step where that
    can happen -- and the volumetric score barely moves, so nothing downstream
    would catch it either.
    """
    circles = []
    for r in rings:
        if r["role"] != "hole" or r["n_primitives"] != 1:
            continue
        seg = r["segments"][0]
        if seg["kind"] != "arc" or abs(abs(seg["sweep_deg"]) - 360.0) > 1.0:
            continue
        circles.append({"radius": seg["radius"], "centre": seg["centre"],
                        "name": seg["name"]})

    groups, used = [], set()
    for i, c in enumerate(circles):
        if i in used:
            continue
        members = [c]
        used.add(i)
        for j, d in enumerate(circles):
            if j in used or abs(d["radius"] - c["radius"]) > radius_tol:
                continue
            members.append(d)
            used.add(j)
        groups.append(members)

    out = []
    for members in groups:
        radius = float(np.mean([m["radius"] for m in members]))
        entry = {"count": len(members), "hole_radius": round(radius, 4),
                 "hole_diameter": round(2 * radius, 4),
                 "names": [m["name"] for m in members]}
        if len(members) >= 3:
            cen = np.array([m["centre"] for m in members], dtype=float)
            # Pitch circle centre: for a regular pattern the centroid is the
            # centre, and a constant radius to it confirms the pattern.
            pc = cen.mean(axis=0)
            rad = np.linalg.norm(cen - pc, axis=1)
            if rad.std() <= radius_tol and rad.mean() > radius_tol:
                ang = np.degrees(np.arctan2(cen[:, 1] - pc[1], cen[:, 0] - pc[0])) % 360.0
                ang.sort()
                diffs = np.diff(np.concatenate([ang, [ang[0] + 360.0]]))
                even = float(diffs.std()) <= angle_tol_deg
                entry.update({
                    "pattern": "circular" if even else "circular-irregular",
                    "pitch_circle_centre": [round(float(pc[0]), 4), round(float(pc[1]), 4)],
                    "pitch_circle_radius": round(float(rad.mean()), 4),
                    "start_angle_deg": round(float(ang[0]), 4),
                    "spacing_deg": round(float(diffs.mean()), 4) if even else None,
                    "angles_deg": [round(float(a), 3) for a in ang],
                })
        out.append(entry)
    return sorted(out, key=lambda g: -g["count"])


def repair_after_cut(prims, tol, axis_idx, axis_val):
    """Clean up the artefacts a straight cut leaves in a fitted profile.

    A section polygon is a chord approximation, so a cut rarely lands on one of
    its vertices. It lands mid-chord, at a point that sits inside the true
    circle by the sagitta -- and the decomposition then sees a kink there and
    splits off a short stub that is really part of the arc: many times the fit
    tolerance in length, but well inside it in shape.

    So the test is shape, not length: a primitive lying within `tol` of an
    adjacent arc's circle belongs to that arc. Absorb it, then put the arc's
    endpoint exactly on the cut line, where the geometry says it should be.
    """
    if not prims:
        return prims

    def on_circle(p, centre, radius):
        for q in (p.p0, p.p1):
            if abs(float(np.linalg.norm(np.asarray(q, float) - centre)) - radius) > tol:
                return False
        return True

    out, absorbed = list(prims), True
    while absorbed and len(out) > 1:
        absorbed = False
        for i, p in enumerate(out):
            if p.kind != "line":
                continue
            for j in ((i - 1) % len(out), (i + 1) % len(out)):
                nb = out[j]
                if nb.kind != "arc" or not on_circle(p, nb.centre, nb.radius):
                    continue
                # Hand the stub's far end to the arc and drop the stub.
                if j == (i + 1) % len(out):
                    nb.p0 = p.p0
                else:
                    nb.p1 = p.p1
                a0 = np.arctan2(nb.p0[1] - nb.centre[1], nb.p0[0] - nb.centre[0])
                a1 = np.arctan2(nb.p1[1] - nb.centre[1], nb.p1[0] - nb.centre[0])
                nb.sweep = float((a1 - a0) % (2 * np.pi) if nb.ccw
                                 else (a0 - a1) % (2 * np.pi))
                out.pop(i)
                absorbed = True
                break
            if absorbed:
                break

    # Put every endpoint that should be on the cut line exactly on it. For an
    # arc, that means the true intersection of its circle with the line, not the
    # chord point the polygon happened to give us.
    for p in out:
        for attr in ("p0", "p1"):
            q = np.asarray(getattr(p, attr), float).copy()
            if abs(q[axis_idx] - axis_val) > tol:
                continue
            q[axis_idx] = axis_val
            if p.kind == "arc":
                other = 1 - axis_idx
                d2 = p.radius ** 2 - (axis_val - p.centre[axis_idx]) ** 2
                if d2 > 0:
                    off = float(np.sqrt(d2))
                    cand = [p.centre[other] - off, p.centre[other] + off]
                    q[other] = min(cand, key=lambda v: abs(v - q[other]))
            setattr(p, attr, q)
        if p.kind == "arc":
            a0 = np.arctan2(p.p0[1] - p.centre[1], p.p0[0] - p.centre[0])
            a1 = np.arctan2(p.p1[1] - p.centre[1], p.p1[0] - p.centre[0])
            p.sweep = float((a1 - a0) % (2 * np.pi) if p.ccw
                            else (a0 - a1) % (2 * np.pi))
    return out


def half_arc_options(seg, u_name="U", v_name="V"):
    """Split a full circle into halves, each usable as a revolve profile.

    A sphere is a revolve of a half circle about its own diameter, so a sketch
    vocabulary that only offers whole circles cannot express one. Each option
    carries the diameter it was cut on, because that line is exactly the axis
    the half has to be revolved about.
    """
    if seg["kind"] != "arc" or abs(abs(seg["sweep_deg"]) - 360.0) > 1.0:
        return []
    cu, cv = seg["centre"]
    r = seg["radius"]
    out = []
    # `alias` is what a person actually says -- "the upper half of A0_0" -- so
    # both spellings are accepted wherever a primitive can be named.
    for axis_name, keep, alias, p0, p1, axis in (
        (v_name, "+", "upper", (cu - r, cv), (cu + r, cv), ((cu - r, cv), (cu + r, cv))),
        (v_name, "-", "lower", (cu + r, cv), (cu - r, cv), ((cu - r, cv), (cu + r, cv))),
        (u_name, "+", "right", (cu, cv - r), (cu, cv + r), ((cu, cv - r), (cu, cv + r))),
        (u_name, "-", "left", (cu, cv + r), (cu, cv - r), ((cu, cv - r), (cu, cv + r))),
    ):
        out.append({
            "name": f"{seg['name']}@{axis_name}{keep}",
            "alias": f"{seg['name']}@{alias}",
            "keeps": f"{axis_name} {'>=' if keep == '+' else '<='} {cv if axis_name == v_name else cu:.4f}",
            "from": [round(p0[0], 4), round(p0[1], 4)],
            "to": [round(p1[0], 4), round(p1[1], 4)],
            "radius": seg["radius"],
            "centre": seg["centre"],
            "sweep_deg": 180.0,
            "revolve_axis": [[round(axis[0][0], 4), round(axis[0][1], 4)],
                             [round(axis[1][0], 4), round(axis[1][1], 4)]],
        })
    return out


def arc_points(prim, n=64):
    """Sample a fitted arc for drawing."""
    a0 = np.arctan2(prim.p0[1] - prim.centre[1], prim.p0[0] - prim.centre[0])
    sweep = prim.sweep if prim.ccw else -prim.sweep
    t = a0 + np.linspace(0.0, sweep, n)
    return np.column_stack([prim.centre[0] + prim.radius * np.cos(t),
                            prim.centre[1] + prim.radius * np.sin(t)])


# The most primitives a section picture labels one by one (see plot_series).
PNG_LABEL_LIMIT = 30


def plot_series(series, u_name, v_name, title, out_path, show_raw=True):
    """Draw several sections in one frame, each in its own colour.

    At the boundary between two prisms the cross-section is discontinuous: one
    profile just above it, a different one just below. Drawing them separately
    hides the only thing that matters for placing the next feature -- how the
    smaller sits inside the larger. Concentric? Offset? Tangent? One frame
    answers it; two frames do not.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 9.5), dpi=110)

    # Above PNG_LABEL_LIMIT primitives a label per primitive and per node is a
    # solid block of text, unreadable. Then each ring gets one label, and the
    # title says where the primitive names went.
    n_prims = sum(len(r["segments"]) for s in series for r in s["rings"])
    per_ring = n_prims > PNG_LABEL_LIMIT
    # Arc centres off the part only widen the axes and shrink the drawing; with
    # per-ring labels no centre can be tied to its arc, so none are drawn.
    bounds = [p.bounds for s in series for p in s["polys"]]
    if bounds:
        b = np.array(bounds, float)
        pad = 0.05 * max(b[:, 2].max() - b[:, 0].min(), b[:, 3].max() - b[:, 1].min())
        box = (b[:, 0].min() - pad, b[:, 1].min() - pad,
               b[:, 2].max() + pad, b[:, 3].max() + pad)
    else:
        box = None

    def on_part(c):
        return box is None or (box[0] <= c[0] <= box[2] and box[1] <= c[1] <= box[3])

    for s in series:
        line_c, arc_c, node_c = s["colours"]
        # Distinct style as well as colour. Where two sections share a feature
        # -- a hole running through both prisms, which is precisely the thing
        # worth seeing -- identical solid strokes overplot and the one drawn
        # first disappears entirely.
        ls = s.get("style", "-")
        lw = s.get("lw", 2.2)
        dx, dy = s.get("label_offset", (5, 5))
        if show_raw:
            for poly in s["polys"]:
                for ring in [poly.exterior] + list(poly.interiors):
                    xy = np.asarray(ring.coords)
                    ax.plot(xy[:, 0], xy[:, 1], "-", color=line_c, lw=3.4,
                            alpha=0.20, zorder=1, solid_capstyle="round")

        for r in s["rings"]:
            for prim, seg in zip(r["_prims"], r["segments"]):
                if prim.kind == "line":
                    xy = np.array([prim.p0, prim.p1])
                    ax.plot(xy[:, 0], xy[:, 1], ls, color=line_c, lw=lw, zorder=3)
                    mid = (prim.p0 + prim.p1) / 2
                else:
                    pts = arc_points(prim)
                    ax.plot(pts[:, 0], pts[:, 1], ls, color=arc_c, lw=lw, zorder=3)
                    mid = pts[len(pts) // 2]
                    if not per_ring and on_part(prim.centre):
                        ax.plot([prim.centre[0]], [prim.centre[1]], "+", color=arc_c,
                                ms=7, mew=1.3, zorder=4)
                if per_ring:
                    continue
                label = seg["name"] + (f" R{prim.radius:.3f}" if prim.kind == "arc" else "")
                ax.annotate(label, mid, fontsize=7.5,
                            color=arc_c if prim.kind == "arc" else line_c,
                            ha="center", va="center", zorder=5,
                            bbox=dict(boxstyle="round,pad=0.16", fc="white",
                                      ec=arc_c if prim.kind == "arc" else line_c,
                                      lw=0.5, alpha=0.92))
            if per_ring:
                if r["_prims"]:
                    at = r["_prims"][0].p0
                    ax.annotate(f"{r['name']} ({len(r['segments'])})", (at[0], at[1]),
                                fontsize=8.5, color=node_c, weight="bold", zorder=7,
                                xytext=(dx, dy), textcoords="offset points",
                                bbox=dict(boxstyle="round,pad=0.16", fc="white",
                                          ec=node_c, lw=0.5, alpha=0.9))
                continue
            for prim, node in zip(r["_prims"], r["nodes"]):
                ax.plot([prim.p0[0]], [prim.p0[1]], "o", color=node_c, ms=4.5, zorder=6)
                ax.annotate(node["name"], (prim.p0[0], prim.p0[1]), fontsize=8,
                            color=node_c, weight="bold", zorder=7,
                            xytext=(dx, dy), textcoords="offset points")
        ax.plot([], [], ls, color=arc_c, lw=lw, label=s["label"])
    if per_ring:
        title += (f"\n{n_prims} primitives: one label per ring (name, primitive "
                  f"count); the primitive names are in the text output")

    ax.set_aspect("equal")
    ax.grid(True, which="major", ls="-", lw=0.6, alpha=0.45)
    ax.grid(True, which="minor", ls=":", lw=0.4, alpha=0.30)
    ax.minorticks_on()
    ax.set_xlabel(f"{u_name} (mm)")
    ax.set_ylabel(f"{v_name} (mm)")
    ax.set_title(title, fontsize=10)
    if len(series) > 1:
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_rings(rings, polys, u_name, v_name, title, out_path, show_raw=True):
    """Draw the decomposed section with every node and segment named.

    The raw polyline is kept underneath in grey so the fit is auditable: if a
    primitive has wandered off the mesh, it is visible rather than buried in a
    residual number. Naming is the actual product here -- the model's next
    instruction is phrased in these labels.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 9), dpi=110)

    if show_raw:
        for poly in polys:
            for ring in [poly.exterior] + list(poly.interiors):
                xy = np.asarray(ring.coords)
                ax.plot(xy[:, 0], xy[:, 1], "-", color="#b9bec6", lw=3.2,
                        alpha=0.55, zorder=1, solid_capstyle="round")

    for r in rings:
        for prim, seg in zip(r["_prims"], r["segments"]):
            if prim.kind == "line":
                xy = np.array([prim.p0, prim.p1])
                ax.plot(xy[:, 0], xy[:, 1], "-", color="#1f5fa8", lw=2.0, zorder=3)
                mid = (prim.p0 + prim.p1) / 2
                ax.annotate(seg["name"], mid, fontsize=7.5, color="#1f5fa8",
                            ha="center", va="center", zorder=5,
                            bbox=dict(boxstyle="round,pad=0.16", fc="white",
                                      ec="#1f5fa8", lw=0.5, alpha=0.9))
            else:
                pts = arc_points(prim)
                ax.plot(pts[:, 0], pts[:, 1], "-", color="#b3341f", lw=2.2, zorder=3)
                mid = pts[len(pts) // 2]
                ax.annotate(f"{seg['name']} R{prim.radius:.3f}", mid, fontsize=7.5,
                            color="#b3341f", ha="center", va="center", zorder=5,
                            bbox=dict(boxstyle="round,pad=0.16", fc="white",
                                      ec="#b3341f", lw=0.5, alpha=0.9))
                ax.plot([prim.centre[0]], [prim.centre[1]], "+", color="#b3341f",
                        ms=6, mew=1.2, zorder=4)

        for prim, node in zip(r["_prims"], r["nodes"]):
            ax.plot([prim.p0[0]], [prim.p0[1]], "o", color="#12304f", ms=4.5, zorder=6)
            ax.annotate(node["name"], (prim.p0[0], prim.p0[1]), fontsize=8,
                        color="#12304f", weight="bold", zorder=7,
                        xytext=(5, 5), textcoords="offset points")

    ax.set_aspect("equal")
    ax.grid(True, which="major", ls="-", lw=0.6, alpha=0.45)
    ax.grid(True, which="minor", ls=":", lw=0.4, alpha=0.30)
    ax.minorticks_on()
    ax.set_xlabel(f"{u_name} (mm)")
    ax.set_ylabel(f"{v_name} (mm)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _ring_extent(ring):
    """Bounding box and centroid of a ring, from its segment endpoints."""
    pts = np.array([s["from_xy"] for s in ring["segments"]], dtype=float)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return (hi - lo), (lo + hi) / 2.0


def _ring_signature(ring, places=3):
    """What the ring is made of, independent of where it sits.

    Two fins of the same size are the same shape; two fins where one carries a
    relief notch are not. Comparing sizes alone would call them identical and
    comparing coordinates would call every one of them different, so compare the
    multiset of primitive kinds and their lengths or radii.
    """
    bits = []
    for s in ring["segments"]:
        if s["kind"] == "line":
            bits.append(("line", round(float(s["length"]), places)))
        else:
            bits.append(("arc", round(float(s["radius"]), places)))
    return tuple(sorted(bits))


def detect_arrays(rings, size_tol=0.05, pitch_tol=0.02, min_count=3):
    """Group congruent rings that sit on a line at a constant pitch.

    The linear counterpart of `detect_patterns`, and it exists for the same
    reason. A finned block of seven fins costs seven names --
    `--use "L0_0,L1_0,L2_0,L3_0,L4_0,L5_0,L6_0"` -- none of which says that the
    part is one fin arrayed seven times. That is the design intent, it is what
    the original program almost certainly expressed with a loop or an rarray,
    and it is thrown away by the time the model sees the section.

    Grouping is by bounding-box size rather than by exact shape, then outliers
    are reported. If the two outermost fins carry a relief notch the middle five
    do not, strict congruence would split that into a group of five and two
    singletons and lose the array, while size grouping finds all seven and says
    which two differ. The difference is the useful part -- an array whose end
    members are not like the rest is exactly the thing a person scanning the
    plot misses.
    """
    if len(rings) < min_count:
        return []

    entries = []
    for r in rings:
        if not r.get("segments"):
            continue          # a ring fitted away to nothing has no extent
        span, centre = _ring_extent(r)
        entries.append({"ring": r, "span": span, "centre": centre,
                        "sig": _ring_signature(r),
                        "name": r["segments"][0]["name"]})

    groups, used = [], set()
    for i, a in enumerate(entries):
        if i in used:
            continue
        members = [a]
        used.add(i)
        for j, b in enumerate(entries):
            if j in used:
                continue
            if np.all(np.abs(b["span"] - a["span"]) <= size_tol):
                members.append(b)
                used.add(j)
        if len(members) >= min_count:
            groups.append(members)

    out = []
    for members in groups:
        cen = np.array([m["centre"] for m in members], dtype=float)
        # Direction of the run: the axis the centres actually spread along.
        spread = cen.max(axis=0) - cen.min(axis=0)
        axis = int(np.argmax(spread))
        other = 1 - axis
        if spread[other] > size_tol:
            continue                       # scattered, not a line
        order = np.argsort(cen[:, axis])
        members = [members[k] for k in order]
        coords = cen[order][:, axis]
        steps = np.diff(coords)
        if len(steps) == 0 or steps.mean() <= 0:
            continue
        even = float(steps.std()) <= pitch_tol

        sigs = {}
        for m in members:
            sigs.setdefault(m["sig"], []).append(m["name"])
        odd = []
        if len(sigs) > 1:
            majority = max(sigs.values(), key=len)
            odd = [n for names in sigs.values() if names is not majority
                   for n in names]

        out.append({
            "count": len(members),
            "size": [round(float(members[0]["span"][0]), 4),
                     round(float(members[0]["span"][1]), 4)],
            "axis": axis,
            "pitch": round(float(steps.mean()), 4) if even else None,
            "even": even,
            "from": round(float(coords[0]), 4),
            "to": round(float(coords[-1]), 4),
            "names": [m["name"] for m in members],
            "odd_members": sorted(odd),
        })
    return sorted(out, key=lambda g: -g["count"])


def grow_edges(rings, spec):
    """Push named edges outward along their own normals, keeping rings closed.

    For joining a feature to something already built. Two independent fits of
    one physical surface agree to about four decimals and disagree below that,
    and OCC cannot sew two faces that nearly-but-not-quite coincide -- which is
    what leaves broken faces at every tessellation tolerance. Overlapping the
    join removes the shared surface, and with it the disagreement.

    Deliberately per-edge and by name. Buffering the whole profile would move
    every edge, so a spoke grown 1 mm to reach its rim would also come out 2 mm
    wider -- trading a meshing failure for a silent geometry error in a
    dimension that was measured exactly.

    `spec` is "NAME=DISTANCE[,NAME=DISTANCE...]", e.g. "L0_3=1.0".
    """
    wanted = {}
    for part in spec.split(","):
        name, _, dist = part.partition("=")
        if not dist:
            raise SystemExit(f'--grow wants "NAME=DISTANCE", got {part!r}')
        wanted[name.strip()] = float(dist)

    seen = set()
    for ring in rings:
        segs = ring["segments"]
        pts = [np.asarray(s["from_xy"], float) for s in segs]
        n = len(pts)
        for i, s in enumerate(segs):
            if s["name"] not in wanted:
                continue
            if s["kind"] != "line":
                raise SystemExit(f"--grow: {s['name']} is an arc; only straight "
                                 f"edges can be pushed out this way")
            seen.add(s["name"])
            d = wanted[s["name"]]
            a, b = pts[i], pts[(i + 1) % n]
            edge = b - a
            L = float(np.linalg.norm(edge))
            if L < 1e-9:
                continue
            normal = np.array([edge[1], -edge[0]]) / L
            # Outward is whichever sign increases the enclosed area; "grow"
            # should never mean "shrink", whatever the ring's winding.
            if _ring_area(pts) * _cross2(edge, normal) > 0:
                normal = -normal
            off = normal * d
            prev_p, next_p = pts[(i - 1) % n], pts[(i + 2) % n]
            pts[i] = _line_meet(prev_p, a, a + off, b + off, a + off)
            pts[(i + 1) % n] = _line_meet(b, next_p, a + off, b + off, b + off)

        for i, s in enumerate(segs):
            s["from_xy"] = [round(float(v), 4) for v in pts[i]]
            s["to_xy"] = [round(float(v), 4) for v in pts[(i + 1) % n]]
            if s["kind"] == "line":
                s["length"] = round(float(np.linalg.norm(pts[(i + 1) % n] - pts[i])), 4)

    missing = set(wanted) - seen
    if missing:
        raise SystemExit(f"--grow named {', '.join(sorted(missing))}, which is "
                         f"not in this profile")
    return rings


def _cross2(a, b):
    """The z of the cross product of two 2-D vectors. np.cross refuses 2-D
    input in NumPy 2.x."""
    return float(a[0] * b[1] - a[1] * b[0])


def _ring_area(pts):
    p = np.asarray(pts, float)
    return 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1)
                              - np.roll(p[:, 0], -1) * p[:, 1]))


def _line_meet(a1, a2, b1, b2, fallback):
    """Intersection of line a1a2 with line b1b2, or `fallback` if parallel."""
    d1, d2 = np.asarray(a2) - a1, np.asarray(b2) - b1
    den = _cross2(d1, d2)
    if abs(den) < 1e-12:
        return np.asarray(fallback, float)
    t = _cross2(np.asarray(b1) - a1, d2) / den
    return np.asarray(a1, float) + t * d1
