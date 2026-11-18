"""The centreline of a strip ring: the path a swept wire follows.

A wire or pipe swept along a planar path, cut in its own plane, comes back as
a strip -- a ring made of the path offset half the wire's width to each side,
closed by the wire's ends -- and nothing names the path inside it, so the
model would stack bands into a square wire where one sweep along the path is
right. Pairing the section's concentric arcs (R6.5 / R5.5 -> a path of R6)
does not work on a traced ring -- each designed arc comes back as three to
five sub-arcs with centres a few hundredths apart -- so the path is taken
from the strip's shape instead: its medial axis (the points half the width
from both sides, from a Voronoi diagram of its boundary), fitted as lines
and arcs like any other trace.
"""

import numpy as np

CAP_TOL = 0.15          # x width: an end arc this close to half the width is a rounded end


class StripError(ValueError):
    """A ring that is no strip, or a centreline that is not one path; says why."""


def _ring_polygon(ring):
    from shapely.geometry import Polygon
    pts = []
    for s in ring["segments"]:
        a = np.asarray(s["from_xy"], float)
        if s["kind"] == "arc":
            c = np.asarray(s["centre"], float)
            t0 = np.arctan2(*(a - c)[::-1])
            sw = np.radians(s["sweep_deg"]) * (1.0 if s["ccw"] else -1.0)
            for t in t0 + sw * np.linspace(0.0, 1.0, 16, endpoint=False):
                pts.append(c + s["radius"] * np.array([np.cos(t), np.sin(t)]))
        else:
            pts.append(a)
    return Polygon(pts).buffer(0)


def _middle(poly, r_in, width):
    """The strip's middle line, ordered from one end to the other: every
    boundary point moved half the width inwards (a side lands on the middle, a
    rounded end on its centre), the points that do sit half the width from the
    boundary kept, and the longest of the shortest paths through them taken."""
    import shapely
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    from scipy.spatial import cKDTree
    from shapely.geometry.polygon import orient

    ring = np.asarray(shapely.segmentize(orient(poly, 1.0).exterior, width / 10.0).coords)[:-1]
    t = np.roll(ring, -1, axis=0) - np.roll(ring, 1, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    q = ring + np.column_stack([-t[:, 1], t[:, 0]]) * r_in      # interior on the left
    bnd = poly.boundary
    d = np.asarray([bnd.distance(shapely.Point(x)) for x in q])
    q = q[(np.abs(d - r_in) <= 0.15 * r_in) & shapely.contains_xy(poly, q[:, 0], q[:, 1])]
    if len(q) < 4:
        raise StripError("no middle line found")
    tree = cKDTree(q)
    pairs = np.asarray(list(tree.query_pairs(width / 3.0)))
    if not len(pairs):
        raise StripError("no middle line found")
    dist = np.linalg.norm(q[pairs[:, 0]] - q[pairs[:, 1]], axis=1)
    g = coo_matrix((dist, (pairs[:, 0], pairs[:, 1])), shape=(len(q), len(q)))
    d0 = dijkstra(g, directed=False, indices=0)
    a = int(np.nanargmax(np.where(np.isfinite(d0), d0, -1)))
    da, pred = dijkstra(g, directed=False, indices=a, return_predecessors=True)
    b = int(np.nanargmax(np.where(np.isfinite(da), da, -1)))
    path, k = [b], b
    while k != a and pred[k] >= 0:
        k = int(pred[k])
        path.append(k)
    if da[b] < 0.5 * max(poly.length / 2.0 - 2 * width, width):
        raise StripError("its middle is not one path (it does not run the strip's length)")
    line = shapely.LineString(q[path[::-1]]).simplify(0.02 * width)
    return np.asarray(line.coords)


def _circle(a, m, b):
    a, m, b = (np.asarray(x, float) for x in (a, m, b))
    d = 2 * (a[0] * (m[1] - b[1]) + m[0] * (b[1] - a[1]) + b[0] * (a[1] - m[1]))
    c = np.array([((a @ a) * (m[1] - b[1]) + (m @ m) * (b[1] - a[1]) + (b @ b) * (a[1] - m[1])) / d,
                  ((a @ a) * (b[0] - m[0]) + (m @ m) * (a[0] - b[0]) + (b @ b) * (m[0] - a[0])) / d])
    return c, float(np.linalg.norm(a - c))


def _snap(pieces, ring, width):
    """The fitted pieces snapped to the strip's own sides: an arc to the mean of
    the concentric arcs either side of it (R6.5 / R5.5 -> R6), a line to the
    midline of the lines facing each other across it; joints rebuilt where the
    snapped pieces meet (a tangent point, an intersection). A piece with nothing
    either side to snap to is a kink of the fit at a joint and is dropped."""
    segs = ring["segments"]
    out = []
    for p in pieces:
        if p[0] == "arc":
            c, r = _circle(*p[1:])
            near = [s for s in segs if s["kind"] == "arc"
                    and np.linalg.norm(np.asarray(s["centre"]) - c) <= 0.5 * width]
            ins = [s for s in near if s["radius"] < r and r - s["radius"] < width]
            outs = [s for s in near if s["radius"] > r and s["radius"] - r < width]
            if not ins or not outs:
                continue
            wts = [abs(s["sweep_deg"]) * s["radius"] for s in ins + outs]
            cc = np.average([s["centre"] for s in ins + outs], axis=0, weights=wts)
            ri = np.average([s["radius"] for s in ins], weights=[abs(s["sweep_deg"]) * s["radius"] for s in ins])
            ro = np.average([s["radius"] for s in outs], weights=[abs(s["sweep_deg"]) * s["radius"] for s in outs])
            v0, v1 = np.asarray(p[1]) - c, np.asarray(p[2]) - c
            ccw = float(v0[0] * v1[1] - v0[1] * v1[0]) > 0
            out.append(("arc", cc, 0.5 * (ri + ro), ccw))
        else:
            a, b = np.asarray(p[1], float), np.asarray(p[2], float)
            d = (b - a) / (np.linalg.norm(b - a) or 1.0)
            mid = 0.5 * (a + b)
            nrm = np.array([-d[1], d[0]])
            found = []
            for s in segs:
                if s["kind"] != "line":
                    continue
                qa, qb = np.asarray(s["from_xy"], float), np.asarray(s["to_xy"], float)
                L = np.linalg.norm(qb - qa)
                if L < 1e-9 or abs(np.dot((qb - qa) / L, nrm)) > np.sin(np.radians(2.0)):
                    continue
                if abs(abs(float(np.dot(0.5 * (qa + qb) - mid, nrm))) - width / 2) < 0.35 * width:
                    u = (qb - qa) / L
                    found.append((0.5 * (qa + qb), u if np.dot(u, d) > 0 else -u, L))
            if not found:
                continue
            # Direction and offsets from the sides themselves, measured from the
            # piece's middle: the fitted line leans with the trace noise, and a
            # degree of lean on a short piece puts the path a noticeable fraction
            # of the width off where its sides were measured across it.
            d2 = np.average([f[1] for f in found], axis=0, weights=[f[2] for f in found])
            d2 /= np.linalg.norm(d2)
            n2 = np.array([-d2[1], d2[0]])
            offs = [(float(np.dot(f[0] - mid, n2)), f[2]) for f in found]
            pos = [o for o in offs if o[0] > 0]
            neg = [o for o in offs if o[0] < 0]
            if not pos or not neg:
                continue
            po = np.average([o[0] for o in pos], weights=[o[1] for o in pos])
            ne = np.average([o[0] for o in neg], weights=[o[1] for o in neg])
            out.append(("line", mid + n2 * 0.5 * (po + ne), d2))
    if not out:
        return pieces
    # Two fitted pieces that snapped to one circle or one line are one piece:
    # a bend can come out of the fit in two, with their "joint" on the centre.
    merged = [out[0]]
    for q in out[1:]:
        m = merged[-1]
        if q[0] == m[0] == "arc" and np.linalg.norm(q[1] - m[1]) <= 0.25 * width \
                and abs(q[2] - m[2]) <= 0.25 * width:
            continue
        if q[0] == m[0] == "line" and abs(abs(float(np.dot(q[2], m[2]))) - 1.0) < 1e-3 \
                and abs(float(np.dot(q[1] - m[1], np.array([-m[2][1], m[2][0]])))) <= 0.25 * width:
            continue
        merged.append(q)
    out = merged
    joints = [_free_end(out[0], pieces[0][1])]
    for q, r in zip(out, out[1:], strict=False):
        joints.append(_joint(q, r))
    joints.append(_free_end(out[-1], pieces[-1][-1]))
    rebuilt = []
    for k, q in enumerate(out):
        a, b = joints[k], joints[k + 1]
        if q[0] == "line":
            rebuilt.append(("line", tuple(a), tuple(b)))
        else:
            c, r, ccw = q[1], q[2], q[3]
            t0 = np.arctan2(*(a - c)[::-1])
            t1 = np.arctan2(*(b - c)[::-1])
            sweep = (t1 - t0) % (2 * np.pi) if ccw else -((t0 - t1) % (2 * np.pi))
            tm = t0 + 0.5 * sweep
            rebuilt.append(("arc", tuple(a), tuple(c + r * np.array([np.cos(tm), np.sin(tm)])), tuple(b)))
    return rebuilt


def _free_end(q, near):
    near = np.asarray(near, float)
    if q[0] == "line":
        p0, d = q[1], q[2]
        return p0 + d * float(np.dot(near - p0, d))
    c, r = q[1], q[2]
    v = near - c
    return c + r * v / (np.linalg.norm(v) or 1.0)


def _joint(q, r):
    """Where two snapped pieces meet: tangent point, or intersection."""
    if q[0] == "line" and r[0] == "line":
        p, d, p2, d2 = q[1], q[2], r[1], r[2]
        m = np.array([d, -d2]).T
        if abs(np.linalg.det(m)) < 1e-9:
            return 0.5 * (p + p2)
        t = np.linalg.solve(m, p2 - p)[0]
        return p + d * t
    if q[0] == "arc" and r[0] == "arc":
        c1, r1, c2, r2 = q[1], q[2], r[1], r[2]
        v = c2 - c1
        L = np.linalg.norm(v) or 1.0
        # tangent circles touch on the line of centres
        return c1 + v / L * (r1 if abs(L - (r1 + r2)) < abs(L - abs(r1 - r2)) else np.sign(r1 - r2 or 1) * r1)
    ln, ac = (q, r) if q[0] == "line" else (r, q)
    p, d, c = ln[1], ln[2], ac[1]
    return p + d * float(np.dot(c - p, d))          # the foot of the centre on the line


def fit_pieces(points, tol):
    """Lines and arcs through a middle line's points, as path pieces."""
    from mesh2cad.fit import _fit_piece, merge
    pieces = []
    for p in merge(_fit_piece(np.asarray(points, float), tol, 4, 4.0), tol, closed=False):
        if p.kind == "line":
            pieces.append(("line", tuple(p.p0), tuple(p.p1)))
        else:
            a0 = np.arctan2(*(p.p0 - p.centre)[::-1])
            mid = a0 + 0.5 * p.sweep * (1.0 if p.ccw else -1.0)
            pieces.append(("arc", tuple(p.p0),
                           tuple(p.centre + p.radius * np.array([np.cos(mid), np.sin(mid)])),
                           tuple(p.p1)))
    return pieces


def centreline(ring, poly=None):
    """{"width", "pieces": [("line", p0, p1) | ("arc", p0, mid, p1)], "caps",
    "length"} for a strip ring, in its section's (u, v); StripError if it is
    no strip or its middle is not one path. `poly` is the section's own
    polygon the ring was fitted to: rebuilt from the fitted arcs instead, a
    1 mm strip came out with a 1.07 mm bulge and its middle broke apart."""
    import shapely

    poly = _ring_polygon(ring) if poly is None else poly.buffer(0)
    if poly.is_empty or poly.geom_type != "Polygon" or poly.interiors:
        raise StripError("not a strip: not one solid ring")
    minx, miny, maxx, maxy = poly.bounds
    size = max(maxx - minx, maxy - miny)
    r_in = float(shapely.maximum_inscribed_circle(poly, 1e-4 * size).length)
    width = 2.0 * r_in
    if width <= 0 or width > 0.25 * size or poly.area < 2.0 * width * width:
        raise StripError(f"not a strip: it is {width:.4f} wide at most, not a thin band "
                         f"along a path (its size is {size:.4f})")
    pts = _middle(poly, r_in, width)
   # Ends: a strip closed by a half-circle of half its width is a wire with a
    # rounded end, and the medial axis already runs to the circle's centre; a
    # flat end stops half a width short, so it is carried out by that much.
    caps = []
    for end, nxt in ((0, 1), (-1, -2)):
        p = pts[end]
        rounded = any(s["kind"] == "arc" and abs(s["radius"] - r_in) <= CAP_TOL * width
                      and np.linalg.norm(np.asarray(s["centre"]) - p) <= 0.5 * width
                      for s in ring["segments"])
        if rounded:
            caps.append(tuple(p))
        else:
            d = p - pts[nxt]
            pts[end] = p + d / (np.linalg.norm(d) or 1.0) * r_in
    dense = np.asarray(shapely.segmentize(shapely.LineString(pts), width / 6.0).coords)
    # The width from the area the path sweeps, not from the largest circle
    # inside: that one sits in a bend and overestimates the wire; area /
    # length is closer. A rounded end adds a half disc, pi w^2 / 8.
    length = float(shapely.LineString(dense).length)
    for _ in range(3):
        width = poly.area / (length + len(caps) * np.pi * width / 8.0)
    # The middle points carry the boundary's trace noise; at 1% of the width
    # a path can come out in dozens of pieces, at 5% in a handful.
    pieces = fit_pieces(dense, max(0.05 * width, float(ring.get("tol") or 0.0)))
    pieces = _snap(pieces, ring, width)
    for a, b in zip(pieces, pieces[1:], strict=False):
        if np.linalg.norm(np.subtract(a[-1], b[1])) > 0.25 * width:
            raise StripError("its middle line breaks between two pieces")
    return {"width": width, "pieces": pieces, "caps": caps, "length": length,
            "points": [tuple(p) for p in dense]}
