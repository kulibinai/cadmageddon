"""What is left over between a target and a build: features, not tessellation.

A boolean difference between two tessellations is mostly film: a fitted true
circle sits just outside the target's polygon everywhere, and the difference is
a shell a few hundredths of a millimetre thick over every curved wall. A floor
on each body's VOLUME cannot tell that film from a feature, and it fails both
ways: it shows a large film a few hundredths thick as the body left, and hides
a small real feature (a plug in a side hole, hole details) under the floor --
or fused to the film, so it is not even a body of its own.

So each body is judged by its THICKNESS (mean 2V/A, against the part's
diagonal) as well as its volume, and a thin body big enough to hide something
is probed inside for thick material -- a lump fused to a film:

- block   a body at least COMPACT x diagonal thick on average: a feature;
- lump    thick material inside a thin body, at least the floor in volume;
- offset  a thin body thicker than the film limit, min(FILM x diagonal,
          FILM_MAX): a face built a fraction of a millimetre off, over that area;
- film    thinner than that on average: tessellation, not a feature;
- sliver  below the volume floor, whatever its shape.

Blocks, lumps and offsets are reported; films and slivers are counted.

The thicknesses never go below the meshing tolerance: on a part one unit across
0.2% of the diagonal is thinner than the chord error of the build's own
tessellation, and a lens between a polygon and its circle is not a feature.
"""
from dataclasses import dataclass, field

import numpy as np

FLOOR = 1e-4      # x target volume: smaller pieces are slivers
COMPACT = 0.002   # x diagonal: thicker on average is a block (0.25 mm at 125 mm)
FILM = 0.001      # x diagonal: thinner on average is film. At 0.05% the base
                  # `--op hole` cone run-on (0.038 mm on a 51 mm plate) read as
                  # a face off, a difference no command can measure finer
FILM_MAX = 0.05   # mm: and never thicker than this. A film is the gap between two
                  # tessellations of one surface: the build's is within 0.01 mm
                  # (STL_TOL), and on finished runs films stay under ~0.04 mm.
                  # 0.1% of a large part's diagonal is several tenths of a mm,
                  # enough to call a face built visibly off "film -- tessellation"
LINK = 0.015      # x diagonal: thick samples nearer than this form one lump
PROBED = 4        # thin bodies probed for lumps, largest first
RAYS_MIN = 2000   # samples per probed body, at least ...
RAYS_MAX = 20000  # ... and at most (about one per (LINK/2)^2 of its area)
MIN_SAMPLES = 5   # a lump has at least this many thick samples; fewer is a ray
                  # that slipped between two triangles of a film
SPIKE = 0.005     # x diagonal: a body reaching this far off the target is a fin, whatever its volume
SPIKE_SAMPLES = 200000  # points on the build, and on the target to screen them against
SPIKE_POINTS = 5        # points of those on one patch make it a spike; fewer is a needle of no area
ON_SAMPLES = 1500 # points per leftover body read off the target's faces
ON_SHARE = 0.05   # a face under less of a body than this is not named
ON_LISTED = 3     # faces named per body


@dataclass
class Item:
    kind: str                     # "block", "lump", "offset"
    volume: float
    thickness: float              # mm: mean 2V/A (block, offset) or median depth (lump)
    centre: np.ndarray
    extents: np.ndarray
    mesh: object = None
    on: list = field(default_factory=list)   # [(surface, area)]: the target faces it lies on


@dataclass
class Leftover:
    items: list = field(default_factory=list)       # reported, largest first
    films: list = field(default_factory=list)       # (volume, thickness)
    slivers: list = field(default_factory=list)     # volumes below the floor
    floor: float = 0.0

    @property
    def film_volume(self):
        return float(sum(v for v, _t in self.films))

    @property
    def sliver_volume(self):
        return float(sum(self.slivers))


def film_limit(diag, tol=0.01):
    """Mean thickness under which a thin body is film, in mm: 0.1% of the
    diagonal, at most FILM_MAX, at least the meshing tolerance."""
    return max(min(FILM * diag, FILM_MAX), tol)


def classify(diff, target_volume, diag, floor=None, tol=0.01, rng=None):
    """Split a boolean difference into blocks, lumps, offsets, films, slivers.
    `tol` is the meshing tolerance (mm) the two meshes were made at."""
    out = Leftover(floor=FLOOR * target_volume if floor is None else floor)
    compact, film = max(COMPACT * diag, 2 * tol), film_limit(diag, tol)
    if diff is None or not len(getattr(diff, "faces", [])):
        return out
    rng = np.random.default_rng(0) if rng is None else rng
    sized = sorted(_bodies(diff), key=lambda x: -x[0])
    probed = 0
    for v, b in sized:
        if v < out.floor:
            out.slivers.append(v)
            continue
        a = float(b.area)
        t = 2.0 * v / a if a > 0 else 0.0
        if t >= compact:
            out.items.append(_item("block", v, t, b))
            continue
        lumps = []
        if probed < PROBED:
            probed += 1
            lumps = [q for q in _lumps(b, rng, compact, LINK * diag) if q.volume >= out.floor]
        out.items.extend(lumps)
        # What is left once the lumps are out is judged on its own: a plug
        # inside a 0.02 mm film lifts the body's mean thickness, not the film's.
        rest = max(v - sum(q.volume for q in lumps), 0.0)
        t_rest = 2.0 * rest / a if a > 0 else 0.0
        if t_rest >= film and rest >= out.floor:
            out.items.append(_item("offset", rest, t_rest, b))
        elif rest > 0:
            out.films.append((rest, t_rest))
    out.items.sort(key=lambda q: -q.volume)
    return out


def _bodies(diff):
    """[(volume, mesh)] per body of a difference.

    Connected faces are not bodies. A build that is proud of the target all
    round leaves a hollow film: an outer shell and an inverted inner one that
    share no edge, so `split` returns two closed shells whose volumes are the
    whole box and minus the whole target. A closed shell with a negative
    volume is a cavity; it belongs to the smallest closed shell around it."""
    import trimesh

    with np.errstate(divide="ignore", invalid="ignore"):
        comps = diff.split(only_watertight=False)
    if not len(comps):
        comps = [diff]
    shells = []
    for c in comps:
        with np.errstate(divide="ignore", invalid="ignore"):
            shells.append((float(c.volume), c, bool(c.is_watertight)))
    outer = [s for s in shells if s[0] >= 0 or not s[2]]
    inner = [s for s in shells if s[0] < 0 and s[2]]
    groups = {id(s[1]): [s] for s in outer}
    for s in inner:
        lo, hi = s[1].bounds
        around = [o for o in outer if o[2] and o[0] > 0
                  and np.all(o[1].bounds[0] <= lo + 1e-9) and np.all(o[1].bounds[1] >= hi - 1e-9)]
        if around:
            host = min(around, key=lambda o: float(np.prod(o[1].extents)))
            groups[id(host[1])].append(s)
        else:
            groups[id(s[1])] = [s]
    out = []
    for g in groups.values():
        if len(g) == 1:
            out.append((abs(g[0][0]), g[0][1]))
        else:
            out.append((abs(sum(s[0] for s in g)), trimesh.util.concatenate([s[1] for s in g])))
    return out


def spikes(built, target, diag, tol=0.01):
    """[(height, tip, volume)], tallest first: patches of the build's surface
    that stand further off the target than SPIKE x diagonal, however little
    volume they hold. The volume floor is blind to them: p197's loft left 3.4 mm
    fins under a 30 mm^3 floor and the run read complete, fins and all.

    Measured on points spread over the build by area, not on the vertices of a
    boolean difference: those have needles of no area in them (a tilted plate,
    0.94 mm tall, no volume). A patch needs SPIKE_POINTS points with material
    under them: a cut as deep as the plate left a sheet of no thickness across
    a tilted plate's slot, 3 mm off the target and nothing there. Its volume is half its
    area times its thickness, a fin having two sides."""
    import trimesh
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    limit = max(SPIKE * diag, 10.0 * tol)
    pts, tri = trimesh.sample.sample_surface(built, SPIKE_SAMPLES, seed=0)
    # Never nearer than the true distance, so nothing tall is dropped here.
    screen = cKDTree(trimesh.sample.sample_surface(target, SPIKE_SAMPLES, seed=0)[0]).query(pts)[0]
    far, inward = pts[screen >= limit], -np.asarray(built.face_normals)[tri[screen >= limit]]
    if len(far) < SPIKE_POINTS:
        return []
    h = trimesh.proximity.closest_point(target, far)[1]
    far, inward, h = far[h >= limit], inward[h >= limit], h[h >= limit]
    start = far + 2.0 * tol * inward
    solid = built.ray.contains_points(start) if len(far) else np.zeros(0, bool)
    far, inward, h, start = far[solid], inward[solid], h[solid], start[solid]
    if len(far) < SPIKE_POINTS:
        return []
    thick = h.copy()
    at, ray, _t = built.ray.intersects_location(start, inward, multiple_hits=False)
    thick[ray] = np.minimum(h[ray], np.linalg.norm(at - start[ray], axis=1))
    pairs = cKDTree(far).query_pairs(LINK * diag, output_type="ndarray")
    n = len(far)
    _k, label = connected_components(coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)),
                                     directed=False)
    each = float(built.area) / SPIKE_SAMPLES
    out = []
    for k in np.unique(label):
        on = label == k
        if on.sum() >= SPIKE_POINTS:
            top = int(np.argmax(h[on]))
            out.append((float(h[on][top]), far[on][top], 0.5 * float(thick[on].sum()) * each))
    return sorted(out, key=lambda s: -s[0])


def spike_text(found, vol):
    """The residual line for `spikes`. It says EXTRA with a volume on purpose:
    the stopping rule reads this line, and a part with fins on is not finished."""
    h, tip, _v = found[0]
    n = len(found)
    return (f"EXTRA about {vol(sum(s[2] for s in found))} mm^3 in {n} thin spike{'s' if n != 1 else ''} under the "
            f"per-body floor, standing up to {h:.2f} mm off the target at ({tip[0]:.2f}, {tip[1]:.2f}, {tip[2]:.2f}) "
            f"-- too little volume to count, but not part of the target. Look at what ends there: a loft or "
            f"a blend running past the part")


def _item(kind, volume, thickness, mesh):
    lo, hi = np.asarray(mesh.bounds)
    return Item(kind, volume, thickness, (lo + hi) / 2, hi - lo, mesh)


def _lumps(body, rng, probe, link):
    """Thick material inside a thin body: rays inward from points sampled on
    its surface by area; points whose ray runs at least `probe` before the next wall are
    joined into lumps (within `link`), and each lump is the body inside the
    box around its samples."""
    import trimesh
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    pts, t = _depths(body, rng, link)
    thick = np.nan_to_num(t, nan=0.0) >= probe
    if thick.sum() < MIN_SAMPLES:
        return []
    p, tt = pts[thick], t[thick]
    # Joined within `link`, or within 2.5 times the mean spacing of the samples
    # when they are sparser than that -- random points leave gaps, and a 100 mm
    # plug sampled every 2.5 mm came out as twelve lumps at a 2.6 mm link.
    link = max(link, 2.5 * float(np.sqrt(body.area / len(pts))))
    pairs = cKDTree(p).query_pairs(link, output_type="ndarray")
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(p), len(p)))
    k, lab = connected_components(g, directed=False)
    count = np.bincount(lab, minlength=k)
    out = []
    for q in np.argsort(-count):
        if count[q] < MIN_SAMPLES:
            break
        pq = p[lab == q]
        lo, hi = pq.min(axis=0) - probe, pq.max(axis=0) + probe
        box = trimesh.creation.box(extents=np.maximum(hi - lo, 1e-9))
        box.apply_translation((lo + hi) / 2)
        try:
            m = trimesh.boolean.intersection([body, box])
        except Exception:                                          # noqa: BLE001
            continue
        if m is None or not len(m.faces):
            continue
        out.append(Item("lump", float(abs(m.volume)), float(np.median(tt[lab == q])),
                        (lo + hi) / 2, hi - lo, m))
    return out


def _depths(body, rng, link, n=None):
    """(points, depth) for points sampled on the surface by area: how far a
    ray from the point runs inward before the next wall. A back face lying on
    the point (a sheet of no thickness) gives 0; no wall at all (an open sheet)
    gives NaN.

    Points, not face centres: a CAD export makes a cylinder of strips as long
    as the cylinder, and their centres put a 100 mm plug's samples on two rings
    33 mm apart -- two thin "lumps" instead of one."""
    import pyvista as pv
    import trimesh
    import vtk

    nf = len(body.faces)
    if n is None:
        n = int(np.clip(float(body.area) / (0.5 * link) ** 2, RAYS_MIN, RAYS_MAX))
    c, ids = trimesh.sample.sample_surface(body, n, seed=int(rng.integers(1 << 31)))
    faces = np.hstack([np.full((nf, 1), 3), body.faces]).ravel()
    # vtkCellLocator, not vtkOBBTree: on a large film it is an order of magnitude
    # faster and misses only a handful of hits -- a missed back face reads as one
    # deep sample, which MIN_SAMPLES absorbs.
    tree = vtk.vtkCellLocator()
    tree.SetDataSet(pv.PolyData(np.asarray(body.vertices, float), faces))
    tree.BuildLocator()
    pts, hit = vtk.vtkPoints(), vtk.vtkIdList()
    nrm = body.face_normals[ids]
    reach = float(np.linalg.norm(body.extents)) + 1.0
    t = np.full(len(ids), np.nan)
    for j, i in enumerate(ids):
        tree.IntersectWithLine(c[j] + nrm[j] * 1e-9 * reach, c[j] - nrm[j] * reach, 0.0, pts, hit)
        for k in range(hit.GetNumberOfIds()):
            if hit.GetId(k) == i:
                continue
            d = max(float(np.dot(np.array(pts.GetPoint(k)) - c[j], -nrm[j])), 0.0)
            if not d >= t[j]:
                t[j] = d
    return c, t


def faces_on(bodies, target, seg, tol=0.01, rng=None):
    """For each leftover body, the target's faces it lies on: [(surface,
    volume)], largest first, the faces under less than ON_SHARE of it left out.

    A leftover is bounded partly by the target's own surface -- where the
    material should be (missing), or where extra material meets it -- and
    partly by the build's. Points sampled on it by area that lie on the target,
    within a quarter of the meshing tolerance, are read off the target's
    segmentation, so a body is named by the faces `faces` lists: the top 1 mm
    of a boss is "cone F12 + cone F23", not a centre and a box, which alone can
    send the model to the wrong end of the part.

    Each point weighs its area times the depth of the body behind it, so a face
    is ranked by the leftover VOLUME against it: a body that is an uncut groove,
    an unbuilt chamfer and a thin skin along a bore would otherwise be named by
    the skin's faces, by area alone.

    Points on no named surface (a facet the segmentation left out) are dropped;
    a body all of whose points miss the target gets an empty list."""
    rng = np.random.default_rng(0) if rng is None else rng
    near = _near(target, seg)
    eps = max(0.25 * tol, 1e-6 * float(np.linalg.norm(target.extents)))
    out = []
    for body in bodies:
        if body is None or not len(getattr(body, "faces", [])) or body.area <= 0:
            out.append([])
            continue
        pts, depth = _depths(body, rng, 0.0, n=ON_SAMPLES)
        w = float(body.area) / len(pts) * np.nan_to_num(depth, nan=0.0)
        out.append(near(pts, w, eps))
    return out


def _near(target, seg):
    """near(points, weight, reach) -> [(surface, weight)]: the target's named
    faces within `reach` of the points, each weighted by how many fall on it."""
    import pyvista as pv
    import vtk

    nf = len(target.faces)
    faces = np.hstack([np.full((nf, 1), 3), np.asarray(target.faces)]).ravel()
    tree = vtk.vtkCellLocator()
    tree.SetDataSet(pv.PolyData(np.asarray(target.vertices, float), faces))
    tree.BuildLocator()
    labels = np.asarray(seg.labels)
    closest, cell, sub, d2 = [0.0, 0.0, 0.0], vtk.reference(0), vtk.reference(0), vtk.reference(0.0)

    def near(points, weight, reach):
        weight = np.broadcast_to(np.asarray(weight, float), (len(points),))
        got = {}
        for p, w in zip(points, weight, strict=True):
            tree.FindClosestPoint(p, closest, cell, sub, d2)
            if float(d2) > reach * reach:
                continue
            k = int(labels[int(cell)]) if 0 <= int(cell) < len(labels) else -1
            if k >= 0:
                got[k] = got.get(k, 0.0) + float(w)
        # One surface cut into several faces -- a groove split by a bore, a
        # cylinder in four quarters -- is one feature, and named once.
        fam = {}
        for k, a in got.items():
            sk = seg.surfaces[k]
            f = fam.setdefault(_family(sk), [[], 0.0])
            f[0].append(sk)
            f[1] += a
        on = sum(a for _ss, a in fam.values())
        named = sorted(((Named.of(ss), a) for ss, a in fam.values()
                        if a >= ON_SHARE * on), key=lambda sa: -sa[1])
        return named[:ON_LISTED]
    return near


@dataclass(frozen=True)
class Named:
    """The faces of one target surface a leftover lies on."""
    kind: str
    names: tuple
    size: str
    surfaces: tuple = field(compare=False, hash=False)

    @classmethod
    def of(cls, surfaces):
        ss = sorted(surfaces, key=lambda x: int(x.name[1:]) if x.name[1:].isdigit() else 0)
        p = ss[0].params or {}
        size = ""
        if ss[0].kind in ("cylinder", "sphere") and "radius" in p:
            size = f" R{float(p['radius']):.4g}"
        elif ss[0].kind == "torus" and "major" in p:
            size = f" R{float(p['major']):.4g}/r{float(p['minor']):.4g}"
        return cls(ss[0].kind, tuple(x.name for x in ss), size, tuple(ss))


def _family(s):
    """A key shared by the faces of one surface: the same plane, or the same
    axis line with the same radius (cylinder, torus) or the same generator
    (cone: slope and radius where the axis passes the origin's foot)."""
    p = s.params or {}
    if s.kind == "plane" and "normal" in p:
        n, off = np.asarray(p["normal"], float), float(p["offset"])
        if n[int(np.argmax(np.abs(n)))] < 0:
            n, off = -n, -off
        return ("plane", tuple(np.round(n, 3)), round(off, 3))
    if "axis" in p and "point" in p:
        d = np.asarray(p["axis"], float)
        d = d / np.linalg.norm(d)
        if d[int(np.argmax(np.abs(d)))] < 0:
            d = -d
        c = np.asarray(p["point"], float)
        t = float(np.dot(c, d))
        foot = tuple(np.round(c - t * d, 2))
        if s.kind == "cone" and "slope" in p:
            sl = float(p["slope"]) * float(np.sign(np.dot(np.asarray(p["axis"], float), d)))
            return ("cone", tuple(np.round(d, 3)), foot, round(sl, 3),
                    round(float(p["offset"]) - sl * t, 2))
        size = tuple(round(float(p[k]), 3) for k in ("radius", "major", "minor") if k in p)
        if size:
            return (s.kind, tuple(np.round(d, 3)), foot, size)
    return (s.kind, s.name)


# Three lattices, each off the face planes by a different fraction of a cell:
# an axis-aligned lattice over an axis-aligned part puts sample points exactly
# coplanar with faces, where ray casting is ambiguous. Fixed, so the answer is
# reproducible; three, so it says how far a cell's worth of shift moves it.
LATTICES = ((0.137, 0.219, 0.311), (0.583, 0.671, 0.457), (0.829, 0.373, 0.761))


@dataclass
class Voxels:
    missing: float               # mm^3, mean over the lattices
    extra: float
    target: float
    pitch: float
    missing_range: tuple         # (min, max) over the lattices
    extra_range: tuple
    bodies: dict = field(default_factory=dict)   # "missing"/"extra" -> [Item]


def voxel_residual(target, built, max_points=100_000, floor=0.0, seg=None):
    """Missing and extra volume by occupancy, for meshes that will not close.

    A boolean needs both solids watertight; occupancy does not, at the price of
    being quantised to the cell. The lattice is fixed to the TARGET -- its
    pitch is 1/64 of the target's longest side and its nodes sit at fixed
    fractions of a cell from the target's corner -- and only extended by whole
    cells to cover what was built, so two builds of one target are measured on
    the same cells (a lattice spanning target and build together would give one
    target different volumes). The spread over three lattices is returned with
    the mean.

    The first lattice's leftover cells are also split into bodies (face-
    connected), each at least `floor` in volume, named by the target faces
    within a cell of them when `seg` is given."""
    from scipy import ndimage

    t_lo, t_hi = np.asarray(target.bounds[0], float), np.asarray(target.bounds[1], float)
    lo = np.minimum(t_lo, built.bounds[0])
    hi = np.maximum(t_hi, built.bounds[1])
    pitch = float((t_hi - t_lo).max()) / 64.0
    if pitch <= 0:
        return None
    while np.prod(np.maximum(np.ceil((hi - lo) / pitch) + 1, 1)) > max_points:
        pitch *= 1.15
    cell = pitch ** 3
    miss, over, tgt, first = [], [], [], None
    for jit in LATTICES:
        start = t_lo + pitch * np.asarray(jit) - pitch * np.ceil((t_lo - lo) / pitch)
        axes = [np.arange(start[i], hi[i], pitch) for i in range(3)]
        if not all(len(a) for a in axes):
            return None
        shape = tuple(len(a) for a in axes)
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        a, b = _inside(target, grid), _inside(built, grid)
        miss.append(np.count_nonzero(a & ~b) * cell)
        over.append(np.count_nonzero(b & ~a) * cell)
        tgt.append(np.count_nonzero(a) * cell)
        if first is None:
            first = (grid, shape, a, b)
    out = Voxels(float(np.mean(miss)), float(np.mean(over)), float(np.mean(tgt)), pitch,
                 (float(min(miss)), float(max(miss))), (float(min(over)), float(max(over))))
    grid, shape, a, b = first
    near = _near(target, seg) if seg is not None else None
    for side, mask in (("missing", a & ~b), ("extra", b & ~a)):
        lab, n = ndimage.label(mask.reshape(shape))
        items = []
        for k in range(1, n + 1):
            idx = np.flatnonzero(lab.ravel() == k)
            v = len(idx) * cell
            if v < floor or not len(idx):
                continue
            p = grid[idx]
            items.append(Item("block", v, 0.0, (p.min(axis=0) + p.max(axis=0)) / 2,
                              p.max(axis=0) - p.min(axis=0) + pitch,
                              on=near(p, cell, 1.5 * pitch) if near else []))
        items.sort(key=lambda q: -q.volume)
        out.bodies[side] = items
    return out


def _inside(mesh, grid):
    """Which lattice points lie inside: the sign of the distance to the nearest
    facet (vtkImplicitPolyDataDistance), which needs no closed surface.
    `trimesh.contains` casts rays and sizes its intermediates by points times
    facets, which on a dense target runs out of memory; this takes seconds and
    lands within a small fraction of a percent of the true volume."""
    import pyvista as pv
    faces = np.hstack([np.full((len(mesh.faces), 1), 3), np.asarray(mesh.faces)]).ravel()
    surf = pv.PolyData(np.asarray(mesh.vertices, float), faces)
    d = pv.PolyData(np.asarray(grid, float)).compute_implicit_distance(surf)["implicit_distance"]
    return np.asarray(d) < 0


def face_word(n):
    """`cone F12`, `cylinder F31+F32 R3`: a surface as the model will look it up."""
    return f"{n.kind} {'+'.join(n.names)}{n.size}"


def on_text(named):
    """`cone F12 + plane F3`, or "" when nothing was named."""
    return " + ".join(face_word(s) for s, _a in named)


def film_note(left, fmt=lambda x: f"{x:.4g}"):
    """`2 film bodies, 6699 mm^3, 0.035 mm thick on average -- tessellation, not features`."""
    if not left.films:
        return ""
    v = left.film_volume
    t = sum(vv * tt for vv, tt in left.films) / max(v, 1e-30)
    n = len(left.films)
    return (f"{n} film bod{'y' if n == 1 else 'ies'}, {fmt(v)} mm^3, {t:.3g} mm thick on "
            f"average -- tessellation, not features")
