"""Pipe paths: a tube's centreline, chained from its round faces.

A tube bent in 3-D comes back from `faces` as straight runs (cylinders) and
bends (tori), each often in several pieces, wall and bore alike, with freeform
patches between -- and nothing joins them: the centreline is not planar, so no
section names a strip path, and the model would revolve the bends one at a time.
Every one of those faces is on the same centreline. A cylinder's axis is a
straight run of it, a torus's centre circle a bend; the faces of one run or one
bend are merged, wall and bore alike. A run meets a bend where its axis touches
the bend's circle: at the foot of the bend's centre on the axis, the axis lying
in the bend's plane at the bend radius. The joint is that tangent point, from
the axes and radii, not from where the segmentation happened to stop a face.

Paths are lists of pieces in world coordinates, ("line", p0, p1),
("arc", p0, mid, p1) and ("spline", p0, ..., pn) -- a smooth curve through
its points -- each starting where the one before ends.
"""

import numpy as np

MERGE_DEG = 1.0        # two axes this close in direction are one axis
TANGENT_DEG = 5.0      # a run leaves a bend along its tangent within this
END_DEG = 20.0         # a joint this close to a bend's measured end is at that end
TUBE_DEG = 180.0       # a run or bend whose faces go this far round is a tube
KINK_DEG = 2.0         # pieces meeting at more than this are not one smooth path
TUBE_SAMPLES = 200     # points taken along each wall of a tube
TUBE_SPREAD = 3.0      # a tube's widest place over its narrowest, at most
TUBE_TURN_DEG = 30.0   # two walls of a tube run this parallel, over 90% of them


class PipeError(ValueError):
    """No such path, or a profile that does not sit on it; says why."""


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _basis(n):
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = _unit(np.cross(n, a))
    return e1, np.cross(n, e1)


def _covered(th):
    """(start, sweep) of the arc that angles `th` cover: the circle less its
    widest gap between them."""
    t = np.sort(np.mod(np.asarray(th, float), 2 * np.pi))
    if len(t) < 2:
        return float(t[0]) if len(t) else 0.0, 0.0
    gaps = np.diff(np.concatenate([t, [t[0] + 2 * np.pi]]))
    k = int(np.argmax(gaps))
    return float(t[(k + 1) % len(t)]), float(2 * np.pi - gaps[k])


def _ang_dist(a, b):
    d = abs((a - b + np.pi) % (2 * np.pi) - np.pi)
    return float(d)


# --- pieces of centreline --------------------------------------------------------

def _round_pieces(mesh, seg):
    """One piece per straight run (coaxial cylinders merged) and per bend
    (tori on one centre circle merged): their centrelines, the radii of the
    faces on them and how far round the faces go."""
    V = np.asarray(mesh.vertices, float)
    tol0 = 10.0 * float(getattr(seg, "tol", 0.0) or 0.0)
    pieces = []
    for s in seg.surfaces:
        if s.kind not in ("cylinder", "torus"):
            continue
        pts = V[np.unique(mesh.faces[s.faces])]
        if s.kind == "cylinder":
            a, p, r = _unit(s.params["axis"]), np.asarray(s.params["point"], float), \
                float(s.params["radius"])
            if r <= 0:
                continue
            tol = max(0.02 * r, tol0)
            t = (pts - p) @ a
            q = next((q for q in pieces if q["kind"] == "line"
                      and abs(float(q["a"] @ a)) > np.cos(np.radians(MERGE_DEG))
                      and np.linalg.norm(np.cross(p - q["p"], q["a"])) < tol), None)
            if q is not None:
                t = (pts - q["p"]) @ q["a"]
                if t.min() > q["t1"] + 2 * r or t.max() < q["t0"] - 2 * r:
                    q = None                    # the same axis further on: another run
            if q is None:
                q = {"kind": "line", "p": p, "a": a, "t0": float(t.min()),
                     "t1": float(t.max()), "faces": [], "walls": []}
                pieces.append(q)
            q["t0"], q["t1"] = min(q["t0"], float(t.min())), max(q["t1"], float(t.max()))
            rel = pts - q["p"]
            w = rel - np.outer(rel @ q["a"], q["a"])
            e1, e2 = _basis(q["a"])
            phi = np.arctan2(w @ e2, w @ e1)
        else:
            n, c = _unit(s.params["axis"]), np.asarray(s.params["point"], float)
            R, r = float(s.params["major"]), float(s.params["minor"])
            if r <= 0 or R <= r:
                continue                       # a spindle torus is no bend
            tol = max(0.02 * r, tol0)
            q = next((q for q in pieces if q["kind"] == "arc"
                      and abs(float(q["n"] @ n)) > np.cos(np.radians(MERGE_DEG))
                      and np.linalg.norm(c - q["c"]) < tol
                      and abs(R - q["R"]) < max(tol, 0.005 * R)), None)
            if q is None:
                e1, e2 = _basis(n)
                q = {"kind": "arc", "c": c, "n": n, "R": R, "e1": e1, "e2": e2,
                     "th": [], "faces": [], "walls": []}
                pieces.append(q)
            rel = pts - q["c"]
            along = rel @ q["n"]
            inplane = rel - np.outer(along, q["n"])
            q["th"].extend(np.arctan2(inplane @ q["e2"], inplane @ q["e1"]).tolist())
            rho = np.linalg.norm(inplane, axis=1)
            phi = np.arctan2(along, rho - q["R"])
        q["faces"].append((s.name, float(s.area)))
        q["walls"].append((float(s.params.get("radius", s.params.get("minor", 0.0))),
                           s.sense, np.degrees(_covered(phi)[1])))
    for q in pieces:
        if q["kind"] == "arc":
            q["start"], q["sweep"] = _covered(q["th"])
    return [q for q in pieces if q["kind"] == "line" or q["sweep"] < 2 * np.pi - 1e-3]


def _radii(q):
    return [w[0] for w in q["walls"]]


def _share_radius(a, b, tol):
    return any(abs(x - y) <= tol for x in _radii(a) for y in _radii(b))


def _line_arc_joint(L, A, tol0):
    """(residual, J, line end, arc end) where run L leaves bend A along its
    tangent, or None."""
    r = max(_radii(L) + _radii(A))
    tol = max(0.02 * r, tol0)
    if abs(float(A["n"] @ L["a"])) > np.sin(np.radians(TANGENT_DEG)):
        return None
    if abs(float(A["n"] @ (L["p"] - A["c"]))) > max(tol, 0.01 * A["R"]):
        return None
    tJ = float((A["c"] - L["p"]) @ L["a"])
    J = L["p"] + L["a"] * tJ
    res = abs(float(np.linalg.norm(J - A["c"])) - A["R"])
    if res > max(tol, 0.01 * A["R"]):
        return None
    if not _share_radius(L, A, max(0.03 * r, tol0)):
        return None
    if not (L["t0"] - 2 * r <= tJ <= L["t1"] + 2 * r):
        return None
    v = J - A["c"]
    thJ = float(np.arctan2(v @ A["e2"], v @ A["e1"]))
    da = _ang_dist(thJ, A["start"])
    db = _ang_dist(thJ, A["start"] + A["sweep"])
    lim = np.radians(END_DEG)
    if min(da, db) > lim or min(da, db) > 0.5 * A["sweep"]:
        return None
    end = "a" if da <= db else "b"
    out = _unit(np.cross(A["n"], v)) * (1.0 if end == "b" else -1.0)
    if abs(float(out @ L["a"])) < np.cos(np.radians(TANGENT_DEG)):
        return None
    up = float(out @ L["a"]) > 0
    if (L["t1"] - tJ if up else tJ - L["t0"]) < 0.25 * r:
        return None                           # the run is on the other side
    return res / max(tol, 1e-12), J, ("lo" if up else "hi"), end, tJ, thJ


def pipe_paths(mesh, seg):
    """[{"name", "pieces", "faces", "tube", "bore", "length", "planar",
    "ends": [(point, face or None), (point, face or None)]}] -- every pipe
    path of the part, longest first, named W0, W1, ..."""
    tol0 = 10.0 * float(getattr(seg, "tol", 0.0) or 0.0)
    P = _round_pieces(mesh, seg)
    cands = []
    for i, L in enumerate(P):
        if L["kind"] != "line":
            continue
        for j, A in enumerate(P):
            if A["kind"] != "arc":
                continue
            got = _line_arc_joint(L, A, tol0)
            if got is not None:
                cands.append((got[0], i, j, got))
    cands.sort(key=lambda c: c[0])
    joint = {}                                 # (piece, end) -> (other, end, J, param)
    for _, i, j, (_, J, lend, aend, tJ, thJ) in cands:
        if (i, lend) in joint or (j, aend) in joint:
            continue
        joint[(i, lend)] = (j, aend, J, tJ)
        joint[(j, aend)] = (i, lend, J, thJ)

    def ends_of(k):
        return ("lo", "hi") if P[k]["kind"] == "line" else ("a", "b")

    seen, chains = set(), []
    for k in range(len(P)):
        if k in seen or not any((k, e) in joint for e in ends_of(k)):
            continue
        # walk back to a free end, then forward
        cur, came, guard = k, None, 0
        while guard <= len(P):
            nxt = [e for e in ends_of(cur) if (cur, e) in joint and e != came]
            if not nxt:
                break
            o, oe = joint[(cur, nxt[0])][:2]
            if o == k:
                break                          # a closed loop of pipe: start anywhere
            cur, came, guard = o, oe, guard + 1
        start = cur
        free = [e for e in ends_of(start) if (start, e) not in joint]
        enter = free[0] if free else ends_of(start)[0]
        chain, cur, e_in = [], start, enter
        while cur is not None and cur not in seen:
            seen.add(cur)
            e_out = [e for e in ends_of(cur) if e != e_in][0]
            chain.append((cur, e_in, e_out))
            if (cur, e_out) in joint:
                cur, e_in = joint[(cur, e_out)][:2]
            else:
                cur = None
        chains.append(chain)

    out = []
    for chain in chains:
        if len(chain) < 2 or not any(P[k]["kind"] == "arc" for k, _, _ in chain):
            continue
        walls = [w for k, _, _ in chain for w in P[k]["walls"]]
        senses = {w[1] for w in walls}
        if not (any(w[2] >= TUBE_DEG for w in walls) or
                ("convex" in senses and "concave" in senses)):
            continue                           # a fillet running round an edge
        pieces, faces = [], []
        for k, e_in, e_out in chain:
            q = P[k]
            if q["kind"] == "line":
                t_lo = joint[(k, "lo")][3] if (k, "lo") in joint else q["t0"]
                t_hi = joint[(k, "hi")][3] if (k, "hi") in joint else q["t1"]
                a = joint[(k, e_in)][2] if (k, e_in) in joint else \
                    q["p"] + q["a"] * (t_lo if e_in == "lo" else t_hi)
                b = joint[(k, e_out)][2] if (k, e_out) in joint else \
                    q["p"] + q["a"] * (t_hi if e_in == "lo" else t_lo)
                if np.linalg.norm(b - a) < 1e-9:
                    continue
                pieces.append(("line", tuple(a), tuple(b)))
            else:
                th_a = joint[(k, "a")][3] if (k, "a") in joint else q["start"]
                th_b = joint[(k, "b")][3] if (k, "b") in joint else q["start"] + q["sweep"]
                sw = (th_b - th_a) % (2 * np.pi)
                ths = (th_a, th_a + sw / 2, th_a + sw) if e_in == "a" else \
                    (th_a + sw, th_a + sw / 2, th_a)
                pts = [q["c"] + q["R"] * (np.cos(t) * q["e1"] + np.sin(t) * q["e2"])
                       for t in ths]
                for end, idx in ((e_in, 0), (e_out, 2)):
                    if (k, end) in joint:
                        pts[idx] = joint[(k, end)][2]   # the very point the run shares
                # the middle at the ends' radius: three points off one circle by
                # a hundredth would visibly change the bend's radius
                rm = 0.5 * (np.linalg.norm(pts[0] - q["c"]) + np.linalg.norm(pts[2] - q["c"]))
                pts[1] = q["c"] + rm * _unit(pts[1] - q["c"])
                pieces.append(("arc", tuple(pts[0]), tuple(pts[1]), tuple(pts[2])))
            faces.append([n for n, _ in sorted(q["faces"], key=lambda f: -f[1])])
        tube = _main_radius(chain, P, "convex") or max(w[0] for w in walls)
        bore = _main_radius(chain, P, "concave", below=tube)
        pieces, ends = _end_faces(pieces, seg, tube, tol0)
        out.append({"pieces": pieces, "faces": faces, "tube": tube, "bore": bore,
                    "length": length(pieces), "planar": _planar(pieces, tube),
                    "ends": ends})
    out.sort(key=lambda w: -w["length"])
    for i, w in enumerate(out):
        w["name"] = f"W{i}"
    return out


def _main_radius(chain, P, sense, below=None):
    """The radius of `sense` with the most face area on the chain: the wall,
    not a flange's rim on the same axis."""
    area = {}
    for k, _, _ in chain:
        for (_, a), (r, sn, _) in zip(P[k]["faces"], P[k]["walls"], strict=True):
            if sn == sense and (below is None or r < below):
                key = next((x for x in area if abs(x - r) <= 0.02 * r), r)
                area[key] = area.get(key, 0.0) + a
    return max(area, key=area.get) if area else None


def _end_faces(pieces, seg, tube, tol0):
    """The flat face closing each end of a path, if one does: a plane square
    to the path at that end. A straight end is carried onto it."""
    pieces = list(pieces)
    ends = []
    for at in (0, 1):
        q = pieces[0] if at == 0 else pieces[-1]
        E = np.asarray(q[1] if at == 0 else q[-1], float)
        T = tangent_at(q, 0.0 if at == 0 else 1.0) * (-1.0 if at == 0 else 1.0)
        best = None
        for s in seg.surfaces:
            if s.kind != "plane":
                continue
            n = _unit(s.params["normal"])
            if abs(float(n @ T)) < np.cos(np.radians(3.0)):
                continue
            d = float(s.params["offset"]) - float(n @ E)
            step = d / float(n @ T)            # along the outward tangent
            if abs(step) > 0.25 * tube + tol0:
                continue
            lo, hi = (np.asarray(b, float) for b in s.extent["bbox"])
            P = E + T * step
            if np.any(P < lo - 0.5 * tube) or np.any(P > hi + 0.5 * tube):
                continue
            if best is None or abs(step) < abs(best[1]):
                best = (s.name, step)
        if best is not None and q[0] == "line":
            P = tuple(E + T * best[1])
            if at == 0:
                pieces[0] = ("line", P, q[2])
            else:
                pieces[-1] = ("line", q[1], P)
            E = np.asarray(P)
        ends.append((tuple(E), best[0] if best else None))
    return pieces, ends


def _planar(pieces, tube):
    pts = np.array([p for q in pieces for p in q[1:]], float)
    c = pts.mean(axis=0)
    _, sv, _ = np.linalg.svd(pts - c)
    return bool(sv[-1] / np.sqrt(len(pts)) < 0.01 * tube)


# --- path geometry ---------------------------------------------------------------

def _arc(q):
    """(centre, unit normal, sweep in radians) of ("arc", p0, mid, p1)."""
    p0, m, p1 = (np.asarray(x, float) for x in q[1:])
    u, v = m - p0, p1 - p0
    w = np.cross(u, v)
    ww = float(w @ w)
    if ww < 1e-18:
        return None
    c = p0 + (float(u @ u) * np.cross(v, w) + float(v @ v) * np.cross(w, u)) / (2 * ww)
    a, b = p0 - c, m - c
    n = _unit(np.cross(a, b))
    half = np.arctan2(float(np.linalg.norm(np.cross(a, b))), float(a @ b))
    return c, n, 2.0 * half


def _rot(v, n, t):
    return v * np.cos(t) + np.cross(n, v) * np.sin(t) + n * float(n @ v) * (1 - np.cos(t))


def _poly(q):
    """(points, cumulative lengths) of a spline's points."""
    pts = np.asarray(q[1:], float)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    return pts, s


def _poly_at(q, f):
    pts, s = _poly(q)
    t = min(max(f, 0.0), 1.0) * s[-1]
    i = int(np.clip(np.searchsorted(s, t, side="right") - 1, 0, len(pts) - 2))
    g = (t - s[i]) / ((s[i + 1] - s[i]) or 1.0)
    return pts[i] + (pts[i + 1] - pts[i]) * g, i


def point_at(q, f):
    if q[0] == "spline":
        return _poly_at(q, f)[0]
    if q[0] == "line":
        a, b = np.asarray(q[1], float), np.asarray(q[2], float)
        return a + (b - a) * f
    g = _arc(q)
    if g is None:
        return np.asarray(q[1], float) + (np.asarray(q[3], float) - np.asarray(q[1], float)) * f
    c, n, sw = g
    return c + _rot(np.asarray(q[1], float) - c, n, sw * f)


def tangent_at(q, f):
    if q[0] == "spline":
        pts = np.asarray(q[1:], float)
        i = _poly_at(q, f)[1]
        return _unit(pts[i + 1] - pts[i])
    if q[0] == "line" or _arc(q) is None:
        return _unit(np.asarray(q[-1], float) - np.asarray(q[1], float))
    c, n, _ = _arc(q)
    return _unit(np.cross(n, point_at(q, f) - c))


def piece_length(q):
    if q[0] == "spline":
        return float(_poly(q)[1][-1])
    if q[0] == "line" or _arc(q) is None:
        return float(np.linalg.norm(np.subtract(q[-1], q[1])))
    c, _, sw = _arc(q)
    return float(np.linalg.norm(np.subtract(q[1], c)) * sw)


def length(pieces):
    return float(sum(piece_length(q) for q in pieces))


def _part(q, f0, f1):
    a, b = point_at(q, f0), point_at(q, f1)
    if q[0] == "spline":
        pts, s = _poly(q)
        lo, hi = sorted((f0, f1))
        # a point within a third of a step of the new end would put a kink in
        # the curve through them
        gap = s[-1] / max(len(pts) - 1, 1) / 3.0 / (s[-1] or 1.0)
        mid = [tuple(p) for p, t in zip(pts, s / (s[-1] or 1.0), strict=True)
               if lo + gap < t < hi - gap]
        got = [tuple(point_at(q, lo))] + mid + [tuple(point_at(q, hi))]
        return _straight(("spline",) + tuple(got if f0 <= f1 else got[::-1]))
    if q[0] == "line":
        return ("line", tuple(a), tuple(b))
    return ("arc", tuple(a), tuple(point_at(q, 0.5 * (f0 + f1))), tuple(b))


def _straight(q):
    """A spline whose points lie on one line is that line: swept, a curve
    through collinear points has no frame, and the kernel gives up on it."""
    pts = np.asarray(q[1:], float)
    a, b = pts[0], pts[-1]
    L = float(np.linalg.norm(b - a))
    if len(pts) == 2 or L == 0.0:
        return ("line", tuple(a), tuple(b))
    u = (b - a) / L
    rel = pts - a
    off = np.linalg.norm(rel - np.outer(rel @ u, u), axis=1)
    return ("line", tuple(a), tuple(b)) if off.max() <= 1e-4 * L else q


def max_kink(pieces):
    """The largest turn, in degrees, where one piece meets the next."""
    k = 0.0
    for a, b in zip(pieces, pieces[1:], strict=False):
        c = float(np.clip(tangent_at(a, 1.0) @ tangent_at(b, 0.0), -1.0, 1.0))
        k = max(k, float(np.degrees(np.arccos(c))))
    return k


def smooth_spline(points, width):
    """("spline", ...) through a strip's middle points: resampled once a
    width and averaged over three, ends kept, so the curve follows the path
    and not the trace noise. Resampled finer, a spline can bend tighter than
    its own half width in a bend, and the kernel cannot close the sweep."""
    pts = np.asarray(points, float)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    n = max(4, int(np.ceil(s[-1] / width)) + 1)
    t = np.linspace(0.0, s[-1], n)
    res = np.column_stack([np.interp(t, s, pts[:, j]) for j in range(pts.shape[1])])
    out = res.copy()
    for i in range(1, n - 1):
        lo, hi = i - 1, i + 2
        out[i] = res[lo:hi].mean(axis=0)
    return _straight(("spline",) + tuple(tuple(p) for p in out))


def tube_midline(a, b, wall):
    """{"points", "width": (narrowest, widest), "length"}: the middle of a tube
    cut along its axis, from the middle lines `a` and `b` of its two walls, or
    None if they aren't two walls of one tube. Each is taken at even fractions
    of its own length, since a bend's outer wall is longer than its inner one.
    Two walls run side by side, a bore's width apart, and much wider than
    `wall`, their thickness."""
    def even(pts):
        pts = np.asarray(pts, float)
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        t = np.linspace(0.0, s[-1], TUBE_SAMPLES)
        return np.column_stack([np.interp(t, s, pts[:, j]) for j in range(pts.shape[1])])

    p, q = even(a), even(b)
    if np.linalg.norm(p - q[::-1], axis=1).mean() < np.linalg.norm(p - q, axis=1).mean():
        q = q[::-1]
    w = np.linalg.norm(p - q, axis=1)
    if w.min() < 2.0 * wall or w.max() > TUBE_SPREAD * w.min():
        return None
    tp, tq = np.gradient(p, axis=0), np.gradient(q, axis=0)
    tp /= np.linalg.norm(tp, axis=1, keepdims=True) + 1e-12
    tq /= np.linalg.norm(tq, axis=1, keepdims=True) + 1e-12
    if np.mean(np.sum(tp * tq, axis=1) >= np.cos(np.radians(TUBE_TURN_DEG))) < 0.9:
        return None
    mid = 0.5 * (p + q)
    length = float(np.linalg.norm(np.diff(mid, axis=0), axis=1).sum())
    if length < 0.5 * float(w.mean()):
        return None
    return {"points": mid, "width": (float(w.min()), float(w.max())), "length": length}


def reverse(pieces):
    return [(q[0],) + tuple(q[1:][::-1]) for q in pieces[::-1]]


def to_world(pieces2d, spec, level):
    """A section's path pieces in (u, v) at `level` as world pieces."""
    def w(p):
        v = [0.0, 0.0, 0.0]
        v[spec["u"]], v[spec["v"]], v[spec["axis"]] = float(p[0]), float(p[1]), float(level)
        return tuple(v)
    return [(q[0],) + tuple(w(p) for p in q[1:]) for q in pieces2d]


def crossings(pieces, k, level):
    """[(piece index, fraction)] where the path crosses coordinate k = level."""
    out = []
    for i, q in enumerate(pieces):
        fs = (np.linspace(0.0, 1.0, 2) if q[0] == "line" else
              _poly(q)[1] / (_poly(q)[1][-1] or 1.0) if q[0] == "spline" else
              np.linspace(0.0, 1.0, 65))
        g = np.array([point_at(q, f)[k] - level for f in fs])
        for j in range(len(fs) - 1):
            if g[j] == 0.0 and (i, float(fs[j])) not in out:
                out.append((i, float(fs[j])))
            elif g[j] * g[j + 1] < 0:
                lo, hi = fs[j], fs[j + 1]
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    if (point_at(q, mid)[k] - level) * g[j] > 0:
                        lo = mid
                    else:
                        hi = mid
                out.append((i, float(0.5 * (lo + hi))))
    if pieces and abs(point_at(pieces[-1], 1.0)[k] - level) == 0.0:
        out.append((len(pieces) - 1, 1.0))
    return out


def from_plane(pieces, k, level, inside, reach):
    """The path as swept from the profile plane coordinate k = level: a list
    of one path (the profile at an end) or two (a section across the middle,
    swept both ways), each starting on the plane, and a note on what was done.

    An end within `reach` of the plane is trimmed or carried straight onto it;
    otherwise the crossing whose point `inside(point)` holds is taken. A path
    the plane does not meet near the profile is refused: PipeError."""
    best = None
    for at in (0, 1):
        q = pieces[0] if at == 0 else pieces[-1]
        E = point_at(q, float(at))
        T = tangent_at(q, float(at)) * (1.0 if at else -1.0)   # outward
        if abs(T[k]) < 0.1:
            continue
        step = (level - E[k]) / T[k]
        if abs(step) <= reach and inside(E + T * step) and \
                (best is None or abs(step) < abs(best[1])):
            best = (at, step, E, T)
    if best is not None:
        at, step, E, T = best
        path = list(pieces) if at == 0 else reverse(pieces)
        if step > 1e-4:
            path.insert(0, ("line", tuple(E + T * step), tuple(E)))
            return [path], f"carried {step:.4f} straight on to the profile's plane"
        if step < -1e-4:
            cut = [(i, f) for i, f in crossings(path, k, level) if i <= 1]
            if cut:
                i, f = cut[0]
                return ([[_part(path[i], f, 1.0)] + path[i + 1:]],
                        f"trimmed {-step:.4f} back to the profile's plane")
        return [path], "the profile is at its end"
    xs = [(i, f) for i, f in crossings(pieces, k, level)
          if abs(tangent_at(pieces[i], f)[k]) >= 0.1 and inside(point_at(pieces[i], f))]
    if not xs:
        raise PipeError("the profile's plane does not meet the path inside the profile")
    i, f = xs[0]
    fwd = [_part(pieces[i], f, 1.0)] + list(pieces[i + 1:])
    back = reverse(list(pieces[:i]) + [_part(pieces[i], 0.0, f)])
    paths = [p for p in (fwd, back) if length(p) > 1e-6]
    return paths, "the profile is a section across it: swept both ways from there"


def tightest_bend(paths):
    """(radius, centre) of the tightest arc on the paths, or (None, None)."""
    best = (None, None)
    for path in paths:
        for q in path:
            g = _arc(q) if q[0] == "arc" else None
            if g is not None:
                r = float(np.linalg.norm(np.subtract(q[1], g[0])))
                if best[0] is None or r < best[0]:
                    best = (r, tuple(g[0]))
    return best


def reach_of_face(mesh, seg, name, point):
    """How far face `name` reaches from `point`: its farthest vertex."""
    s = seg.by_name(name)
    pts = np.asarray(mesh.vertices, float)[np.unique(mesh.faces[s.faces])]
    return float(np.linalg.norm(pts - np.asarray(point, float), axis=1).max())


def across_run(w):
    """(axis index, level) of the middle of the path's longest straight run
    when it lies along X, Y or Z -- where a cut across the tube is its own
    section -- or None."""
    runs = [q for q in w["pieces"] if q[0] == "line"]
    if not runs:
        return None
    q = max(runs, key=piece_length)
    d = np.abs(_unit(np.subtract(q[2], q[1])))
    k = int(np.argmax(d))
    if d[k] < np.cos(np.radians(1.0)):
        return None
    return k, float(0.5 * (q[1][k] + q[2][k]))


def describe(w):
    """`W0 pipe: F2 -> F7+F8+F10 -> F5 ...` and its measurements."""
    kinds = [q[0] for q in w["pieces"]]
    bends = []
    for q in w["pieces"]:
        if q[0] == "arc" and _arc(q) is not None:
            c, _, sw = _arc(q)
            bends.append(f"{np.degrees(sw):.1f} deg R{np.linalg.norm(np.subtract(q[1], c)):.4f}")
    chain = " -> ".join("+".join(f) for f in w["faces"])
    head = (f"{w['name']} pipe: {chain}   tube R{w['tube']:.4f}"
            + (f", bore R{w['bore']:.4f}" if w["bore"] else "")
            + f"; {len(kinds)} pieces ({kinds.count('line')} straight, "
            f"{kinds.count('arc')} bends: {', '.join(bends)}), "
            f"{w['length']:.4f} long, {'planar' if w['planar'] else 'not planar'}")
    ends = []
    for (p, face), which in zip(w["ends"], ("from", "to"), strict=True):
        ends.append(f"{which} ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})"
                    + (f" [end face {face}]" if face else " [open]"))
    return head, "   " + "  ".join(ends)
