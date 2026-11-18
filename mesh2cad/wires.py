"""The centrelines of a body's thin round wires, read off its mesh.

Pipe paths chain cylinder and torus faces. A wire cage has none to chain: on a
dense export `faces` finds thousands of cylinder fragments a few tenths of a mm
long, and no section shows a wire along its length.

A wire's surface says where its centreline is without any fitting. The surface
of a wire of radius r curves by 1/r across it, so that radius stands out of the
histogram of the mesh's curvature, and each surface point moved r against its
normal lands on the centreline. Those points are a thin cloud along the wires;
where wires cross it stops being a line, so it is cut there, and the pieces
that carry on through a crossing are joined again. This costs seconds on a
mesh that `faces` can't finish.

Only what goes most of the way round is a wire: a fillet is curved as tightly,
but its normals turn through a quarter (`ROUND_DEG`).
"""

import numpy as np

CURVE_BINS = 80        # of the histogram of log radius
RADIUS_SPREAD = 0.3    # x r: a vertex this close to the peak radius is on a wire
MIN_SHARE = 0.1        # of the vertices at the peak radius, or there are no wires
MAX_RADIUS = 0.02      # x the mesh's diagonal: thicker is a body, not a wire
CELL = 0.5             # x r: the cloud is thinned to one point in a cell this big
STRAIGHT = 0.75        # a point this straight or more isn't at a crossing
TURN_COS = 0.7         # neighbours in one piece turn less than this
JOIN_GAP = 10.0        # x r: pieces this far apart can be one wire across a crossing
JOIN_COS = 0.95        # ... when both run on along the gap this well
MIN_LENGTH = 8.0       # x r: a shorter piece is the remains of a crossing
ROUND_DEG = 240.0      # the surface's normals go at least this far round a wire
SAMPLE = 0.5           # x r: the spacing of the points sampled on the surface
MAX_SAMPLES = 1_000_000
STEP = 3.0             # x r: spacing of the points kept along a wire
COPY_LENGTH = 0.02     # lengths this close, in share, can be one wire's copies


def _radius_of_vertices(mesh):
    """Each vertex's sharpest curvature, as a radius."""
    v, n, e = mesh.vertices, mesh.vertex_normals, mesh.edges_unique
    dp = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)
    dn = np.linalg.norm(n[e[:, 0]] - n[e[:, 1]], axis=1)
    k = np.zeros(len(e))
    ok = dp > 1e-9
    k[ok] = dn[ok] / dp[ok]
    kv = np.zeros(len(v))
    np.maximum.at(kv, e[:, 0], k)
    np.maximum.at(kv, e[:, 1], k)
    return 1.0 / np.maximum(kv, 1e-9)


def _peak_radius(r, diag):
    """The wire radius, or None: the top of the histogram of `r`, if enough of
    the mesh is there and it is thin for the part."""
    good = (r > 1e-3) & (r < MAX_RADIUS * diag)
    if not good.any():
        return None
    h, edges = np.histogram(np.log10(r[good]), bins=CURVE_BINS)
    i = int(np.argmax(h))
    r0 = float(10 ** (0.5 * (edges[i] + edges[i + 1])))
    return r0 if np.mean(np.abs(r - r0) <= RADIUS_SPREAD * r0) >= MIN_SHARE else None


def _samples(mesh, near, r):
    """(points, unit normals) spread over the triangles whose corners are all
    `near`. Not the vertices: a straight wire's mesh has its vertices at its
    ends, and long triangles between."""
    faces = mesh.faces[near[mesh.faces].all(axis=1)]
    if not len(faces):
        return np.zeros((0, 3)), np.zeros((0, 3))
    tri = mesh.vertices[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    n = int(min(MAX_SAMPLES, max(area.sum() / (SAMPLE * r) ** 2, len(faces))))
    rng = np.random.default_rng(0)
    pick = rng.choice(len(faces), n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    w = np.column_stack([1.0 - u - v, u, v])[:, :, None]
    points = (mesh.vertices[faces[pick]] * w).sum(axis=1)
    normals = (mesh.vertex_normals[faces[pick]] * w).sum(axis=1)
    return points, normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)


def _cloud(points, r):
    """`points` thinned to the mean of each cell, leaving out lone stray ones."""
    cell = np.floor(points / (CELL * r)).astype(np.int64)
    _, inv, cnt = np.unique(cell, axis=0, return_inverse=True, return_counts=True)
    p = np.zeros((len(cnt), 3))
    np.add.at(p, inv.ravel(), points)
    return (p / cnt[:, None])[cnt >= 2]


def _directions(p, r):
    """(each point's direction along the wire, how straight it is there)."""
    from scipy.spatial import cKDTree
    d, straight = np.zeros((len(p), 3)), np.zeros(len(p))
    for i, near in enumerate(cKDTree(p).query_ball_point(p, 2.0 * r)):
        if len(near) < 3:
            continue
        q = p[near] - p[near].mean(axis=0)
        w, vec = np.linalg.eigh(q.T @ q)
        d[i], straight[i] = vec[:, 2], 1.0 - np.sqrt(max(w[1], 0.0) / (w[2] + 1e-12))
    return d, straight


def _ordered(q, r):
    """The points of one piece, end to end: the longest of the shortest paths."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    from scipy.spatial import cKDTree
    pairs = cKDTree(q).query_pairs(1.6 * r, output_type="ndarray")
    if not len(pairs):
        return q
    w = np.linalg.norm(q[pairs[:, 0]] - q[pairs[:, 1]], axis=1)
    g = coo_matrix((w, (pairs[:, 0], pairs[:, 1])), shape=(len(q), len(q)))
    d0 = dijkstra(g, directed=False, indices=0)
    a = int(np.argmax(np.where(np.isfinite(d0), d0, -1.0)))
    d1, pred = dijkstra(g, directed=False, indices=a, return_predecessors=True)
    b = int(np.argmax(np.where(np.isfinite(d1), d1, -1.0)))
    path = [b]
    while path[-1] != a and pred[path[-1]] >= 0:
        path.append(int(pred[path[-1]]))
    return q[path[::-1]]


def _length(p):
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def _pieces(p, r):
    """The cloud cut at its crossings: ordered pieces of wire."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    d, straight = _directions(p, r)
    keep = straight >= STRAIGHT
    q, dq = p[keep], d[keep]
    if len(q) < 3:
        return []
    pairs = cKDTree(q).query_pairs(1.6 * r, output_type="ndarray")
    if not len(pairs):
        return []
    pairs = pairs[np.abs(np.sum(dq[pairs[:, 0]] * dq[pairs[:, 1]], axis=1)) > TURN_COS]
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(q), len(q)))
    n, lab = connected_components(g, directed=False)
    out = []
    for k in range(n):
        s = q[lab == k]
        if len(s) >= 6:
            s = _ordered(s, r)
            if _length(s) >= 4.0 * r:
                out.append(s)
    return out


def _end(s, e, span=6):
    """(the point, the outward direction) of end `e` (0 or 1) of a piece."""
    a = s[:span] if e == 0 else s[-span:]
    u = a[0] - a[-1] if e == 0 else a[-1] - a[0]
    return (s[0] if e == 0 else s[-1]), u / (np.linalg.norm(u) or 1.0)


def _joined(pieces, r):
    """Pieces that carry on through a crossing, joined, nearest ends first."""
    wires = [np.asarray(s) for s in pieces]
    while True:
        best = None
        for i, a in enumerate(wires):
            for ei in (0, 1):
                pa, ta = _end(a, ei)
                for j in range(i + 1, len(wires)):
                    for ej in (0, 1):
                        pb, tb = _end(wires[j], ej)
                        gap = pb - pa
                        dist = float(np.linalg.norm(gap))
                        if dist > JOIN_GAP * r or dist < 1e-9:
                            continue
                        u = gap / dist
                        if ta @ u >= JOIN_COS and tb @ (-u) >= JOIN_COS and (
                                best is None or dist < best[0]):
                            best = (dist, i, ei, j, ej)
        if best is None:
            return wires
        _, i, ei, j, ej = best
        a = wires[i][::-1] if ei == 0 else wires[i]
        b = wires[j] if ej == 0 else wires[j][::-1]
        wires[i] = np.vstack([a, b])
        del wires[j]


def _smooth(w, r):
    """The wire's points, averaged along it and kept every STEP radii."""
    n = max(int(round(1.5 * r / max(_length(w) / max(len(w) - 1, 1), 1e-9))), 1)
    pad = np.vstack([np.repeat(w[:1], n, axis=0), w, np.repeat(w[-1:], n, axis=0)])
    kernel = np.ones(2 * n + 1) / (2 * n + 1)
    s = np.column_stack([np.convolve(pad[:, i], kernel, "valid") for i in range(3)])
    s[0], s[-1] = w[0], w[-1]
    at = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(s, axis=0), axis=1))]
    want = np.linspace(0.0, at[-1], max(int(at[-1] / (STEP * r)) + 1, 4))
    return np.column_stack([np.interp(want, at, s[:, i]) for i in range(3)])


def _round_all_the_way(w, surface, normals, r):
    """(whether the surface near `w` goes ROUND_DEG round it, its measured
    radius). Each surface point's normal, across the wire, sets its bearing."""
    from scipy.spatial import cKDTree
    dist, idx = cKDTree(w).query(surface, distance_upper_bound=2.0 * r)
    near = np.isfinite(dist)
    if near.sum() < 12:
        return False, r
    t = np.gradient(w, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    a, n = t[idx[near]], normals[near]
    across = n - a * np.sum(n * a, axis=1, keepdims=True)
    off = surface[near] - w[idx[near]]
    off = off - a * np.sum(off * a, axis=1, keepdims=True)
    # square to the wire: whichever world axis it runs least along, so never zero
    world = np.eye(3)[np.argmin(np.abs(a), axis=1)]
    ref = np.cross(a, world)
    ref /= np.linalg.norm(ref, axis=1, keepdims=True) + 1e-12
    other = np.cross(a, ref)
    bearing = np.degrees(np.arctan2(np.sum(across * other, axis=1), np.sum(across * ref, axis=1)))
    bins = np.unique(np.floor((bearing + 180.0) / 15.0).astype(int))
    return len(bins) * 15.0 >= ROUND_DEG, float(np.median(np.linalg.norm(off, axis=1)))


def wire_paths(mesh, start=0):
    """[{"name": W#, "radius", "points": [(x, y, z)...], "length", "copies":
    [names of wires of its size]}], longest first, numbered from `start`; []
    when the mesh has no thin round wires. Wires of one radius are found: the
    tallest peak of its curvature."""
    r = _radius_of_vertices(mesh)
    diag = float(np.linalg.norm(mesh.extents))
    r0 = _peak_radius(r, diag)
    if r0 is None:
        return []
    surface, normals = _samples(mesh, np.abs(r - r0) <= RADIUS_SPREAD * r0, r0)
    cloud = _cloud(surface - normals * r0, r0)
    if len(cloud) < 6:
        return []
    found = []
    for w in _joined(_pieces(cloud, r0), r0):
        if _length(w) < MIN_LENGTH * r0:
            continue
        w = _smooth(w, r0)
        round_it, rad = _round_all_the_way(w, surface, normals, r0)
        if round_it:
            found.append((_length(w), rad, w))
    found.sort(key=lambda x: -x[0])
    out = [{"name": f"W{start + i}", "radius": rad, "length": length,
            "points": [tuple(float(x) for x in p) for p in w], "copies": []}
           for i, (length, rad, w) in enumerate(found)]
    for a in out:
        a["copies"] = [b["name"] for b in out if b is not a
                       and abs(b["length"] - a["length"]) <= COPY_LENGTH * a["length"]]
    return out
