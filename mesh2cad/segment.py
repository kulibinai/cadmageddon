"""The mesh as the analytic faces it was tessellated from.

Every instrument before this one reads the part through cuts: a sweep of slices,
and where the slice changes, a level. That works, but it measures a surface
indirectly -- a cylinder lying across the sweep axis makes the area ramp, and a
ramp is read as a run of "prism boundaries" that no operation ever made. On one
bracket that was most of the eighteen levels along Z.

The meshes this harness is given are not scans. They are tessellations of a
B-rep, and a tessellator puts every VERTEX exactly on the face it is meshing and
only the chords between vertices off it. So the faces are recoverable exactly:
fit each candidate surface to vertices, not to facets, and a cylinder comes back
with its true radius and axis rather than a polygon's.

    planar patches   adjacent triangles that are exactly coplanar. The
                     tolerance is read off the mesh: coplanar neighbours agree
                     to float precision, and the nearest pair that is not
                     coplanar differs by orders of magnitude more (R5).
    round surfaces   grown from patches: a cylinder, cone, sphere or torus is
                     accepted only while EVERY vertex of every patch it takes
                     lies on it. A fillet stops where the plane it is tangent to
                     begins, crease or no crease, because the plane's far
                     vertices are not on the fillet.
    what is left     each remaining patch is a plane; a smooth run of small
                     patches that no analytic surface fits is a freeform face,
                     and is called that rather than forced into a primitive.
                     An exact flat patch that dwarfs every leftover patch it
                     meets smoothly is a plane of its own even so: a pad's wall
                     tangent to a fillet cut into two facets was glued to it,
                     and on one pump housing 28 planes -- 27% of the area --
                     went freeform that way.
    revolves         a freeform piece whose vertices all sit on three or more
                     coaxial circles is a surface of revolution with a profile
                     no analytic kind has -- a spline revolved. It is kept as
                     kind "revolve": its axis, and the (t, R) of each ring.

When the mesh shows no gap between coplanar and not, the tolerance falls back
to the float32 floor, which can be tighter than the tessellator was: vertices
on a revolve's seam sat one to two floors off their torus. Then a surface that
has been seeded within the floor may GROW onto facets within FLOOR_MARGIN
floors of it. Seeding itself keeps the floor -- with the margin, every few
facets of a spline fitted some cone or torus -- and planes keep it too.

This module decides nothing about operations. It says which surfaces the part
is made of and how they meet, and everything is evidence (A2): the levels, the
directions and the feature hypotheses of later stages are read off it.
"""

import hashlib
import heapq
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Bump when anything below changes what a cached result would contain.
ALGO_VERSION = 5
EPS32 = float(np.finfo(np.float32).eps)
DOF = {"plane": 3, "sphere": 4, "cylinder": 5, "cone": 6, "torus": 7}
ROUND_KINDS = ("sphere", "cylinder", "cone", "torus")
MERIDIAN_ORDER = ("cylinder", "cone", "sphere", "torus")

# Facets that meet at 90 degrees or more are never offered to a round surface as
# a smooth continuation: a circle cut into four or fewer facets is a square, and
# no measurement can say otherwise. Not a tolerance -- a bound on what a
# tessellation of a round face can look like, used only to prune the search.
SMOOTH_BOUND_DEG = 90.0
# A round surface cut into fewer than twelve facets per turn is COARSE: at that
# density a cylinder and a prism with that many sides tessellate identically,
# so a coarse surface is only believed when the mesh's own chord tolerance --
# measured on its finely cut surfaces -- says it would have been cut that coarse.
COARSE_FACET_DEG = 30.0
# Least-squares refinement works on at most this many vertices; acceptance is
# always checked on every vertex, so this bounds time, not correctness.
FIT_SAMPLE = 400
# A tessellator cuts one round face into facets of about ONE angular step, and
# a chord's bow grows with the square of its width. A patch WIDER than three
# steps and DEEPER than nine bows is not a facet but a flat that happens to have
# its corners on the face: a D-shaft's flat spans ten of the shaft's strips and
# sits six hundred bows in, a disc capping a fillet covers the whole turn and
# sits a radius away. Three, because tessellators do lay the odd facet two
# steps wide -- a cone here had five, each bowing four times its neighbours.
# Width alone is not enough either: a tessellator lays a flat ring of triangles
# round the crown of a torus, all the way round and hardly any depth, and that
# ring is a facet.
SPREAD = 3.0
# The step is read off the facets a surface takes, up to this many; after that
# it is settled, and re-reading it on every facet would cost more than it tells.
SPAN_SAMPLE = 200
# ...and it is the coarse end of what they show, not the middle. A tessellator
# cuts finer where something crosses the face -- round a hole through a wall
# the strips were a quarter of their width elsewhere -- and a surface that
# happened to start growing there took its fine facets for its step and turned
# away its own full-width strips as flats.
STEP_QUANTILE = 90
# A seed's first neighbourhood takes only neighbours within this factor of its
# own area. A tessellator cuts one face into facets of one size, so the big
# neighbour at a small angle is the cap or the plane the face runs into -- and
# a fit that has to go through it as well bends to suit it. Big neighbours are
# still offered to the surface once it exists, and judged then.
HOOD_AREA_RATIO = 4.0
# Changing a surface's kind mid-growth (a band of a sphere is exactly a cone
# until the next band arrives) is only tried while the surface is young; a
# large one blocked by a neighbour has met its real boundary.
UPGRADE_PATCHES = 60
# Iteration cap for the trial fits a seed's neighbourhood gets (see _first_fit).
QUICK_NFEV = 40

# Switches for the three behaviours the module docstring describes, so a test
# or an A/B run can turn one off; all are on in the harness.
KEEP_DOMINANT_PLANES = True
FLOOR_GROWTH_MARGIN = True
# How far a growing round surface may sit from a facet, in floor tolerances,
# when the tolerance is the float32 floor.
FLOOR_MARGIN = 3.0
# A leftover flat patch of two or more triangles is a face when it is this many
# times larger than the largest leftover patch it meets smoothly.
PLANE_DOMINANCE = HOOD_AREA_RATIO
# A freeform piece whose vertices all lie on coaxial circles is a revolve of a
# profile no analytic surface has: kind "revolve", with the axis and the
# (t, rho) of each ring as its profile.
FIND_REVOLVES = True
REVOLVE_MIN_RINGS = 3         # two rings are a cone's worth of evidence, no more
REVOLVE_RING_EDGE = 0.1       # an edge runs round the axis when its step along the profile is under this share of it
REVOLVE_K = 4.0               # ring vertices within this many round tolerances of the circle,
REVOLVE_REL = 2e-5            # or within this share of the part's diagonal: a spline revolve
                              # is often stored as a B-spline that is round only to a micron
REVOLVE_MIN_TURN = 60.0       # degrees a ring must go round to place the axis
REVOLVE_STRAY = 0.25          # share of vertices allowed off the rings (seam vertices)
REVOLVE_STRAY_K = 20.0        # how far those may sit from a spline through the rings, in limits


# ------------------------------------------------------------------ results

@dataclass(eq=False)
class Surface:
    kind: str                 # plane | cylinder | cone | sphere | torus | freeform
    faces: np.ndarray         # triangle indices
    params: dict              # world-frame parameters of the fitted surface
    area: float
    residual: float = 0.0     # max vertex distance to the fitted surface
    sagitta: float = 0.0      # max facet-centroid distance: the chord error
    facet_angle: float = 0.0  # tessellation step inside the surface, degrees
    sense: str = "flat"       # convex | concave | flat | mixed
    extent: dict = field(default_factory=dict)
    loops: int = 0            # boundary loops of the face
    notes: list = field(default_factory=list)
    name: str = ""


@dataclass
class Joint:
    a: int
    b: int
    length: float
    angle: float              # length-weighted mean dihedral, degrees
    kind: str                 # convex | concave | tangent | mixed


@dataclass
class Segmentation:
    surfaces: list
    labels: np.ndarray        # surface index per triangle
    joints: list
    tol: float
    tol_how: str
    diag: float
    area: float
    n_faces: int
    watertight: bool
    notes: list = field(default_factory=list)
    demoted: list = field(default_factory=list)
    # True when a time budget stopped the search for round surfaces early; what
    # was not examined is reported as planes or freeform, and a note says so.
    partial: bool = False
    # The budget a partial result was cut at: it serves only a caller allowed
    # as little time or less, so `faces` never inherits the level table's 90 s.
    budget: float = None
    # What a partial search left untried: candidate facets, and their share of
    # the surface, so "cut short" says how much and where.
    untried: int = 0
    untried_area: float = 0.0

    def by_name(self, name):
        return next((s for s in self.surfaces if s.name == name), None)

    def to_dict(self):
        def clean(v):
            if isinstance(v, np.ndarray):
                return [clean(x) for x in v.tolist()]
            if isinstance(v, (list, tuple)):
                return [clean(x) for x in v]
            if isinstance(v, dict):
                return {k: clean(x) for k, x in v.items()}
            if isinstance(v, (np.floating, float)):
                return round(float(v), 9)
            if isinstance(v, np.integer):
                return int(v)
            return v
        return {
            "tolerance": self.tol, "tolerance_how": self.tol_how,
            "diag": self.diag, "area": self.area, "n_faces": self.n_faces,
            "watertight": self.watertight, "notes": self.notes,
            "surfaces": [{
                "name": s.name, "kind": s.kind, "params": clean(s.params),
                "area": round(s.area, 9), "area_frac": round(s.area / self.area, 6),
                "n_faces": int(len(s.faces)), "residual": s.residual,
                "sagitta": s.sagitta, "facet_angle": s.facet_angle,
                "sense": s.sense, "extent": clean(s.extent), "loops": s.loops,
                "notes": s.notes} for s in self.surfaces],
            "joints": [{"a": self.surfaces[j.a].name, "b": self.surfaces[j.b].name,
                        "length": round(j.length, 9), "angle": round(j.angle, 4),
                        "kind": j.kind} for j in self.joints],
        }


# ---------------------------------------------------------------- tolerance

def derive_tolerance(mesh):
    """How far a vertex may sit off a surface and still be ON it, from the mesh.

    Coplanar neighbours in a B-rep tessellation agree to float precision, and the
    nearest pair that is not coplanar differs by orders of magnitude more: on
    the meshes in this repo the two populations are separated by five to eight
    decades. The tolerance goes just above the first population, and never below
    what float32 storage of the coordinates can promise, because STL is float32.

    Returns (tol, sentence saying where it came from).
    """
    diag = float(np.linalg.norm(mesh.extents)) or 1.0
    quantum = EPS32 * float(np.abs(mesh.vertices).max() or 1.0)
    floor = 4.0 * quantum
    d = _pair_offsets(mesh)
    # A target ingest zipped (normalize.zip_cracks): the triangles it made or
    # bent sit a crack's width off their neighbours. Read among them, they would
    # drive the tolerance down to the float32 floor and leave real bores
    # freeform. They are left out here, and the tolerance never goes below
    # how far the repair moved a vertex.
    zipped = mesh.metadata.get("zipped_faces") if hasattr(mesh, "metadata") else None
    moved = float(mesh.metadata.get("zip_move") or 0.0) if zipped is not None else 0.0
    if zipped is not None and len(zipped) and len(d) == len(mesh.face_adjacency):
        bad = np.zeros(len(mesh.faces), bool)
        bad[np.asarray(zipped, int)] = True
        d = d[~bad[mesh.face_adjacency].any(axis=1)]
    tol, how = _tolerance_from(d, diag, quantum, floor)
    if moved > tol:
        return moved, (f"{moved:.2e} mm: the {tol:.2e} the facets give ({how}), "
                       f"raised to how far ingest's repair moved a vertex")
    return tol, how


def _tolerance_from(d, diag, quantum, floor):
    """`derive_tolerance` on the pair offsets `d`."""
    rel = d[d > 0] / diag
    if not len(rel):
        return floor, (f"{floor:.2e} mm: four float32 steps at this part's "
                       f"coordinates (every neighbouring pair is exactly coplanar)")
    lo_floor = np.log10(floor / diag)
    lg = np.sort(np.log10(rel))
    # Anything below float32 precision is rounding by definition, however it is
    # distributed -- exact zeros and 1e-12 of double-precision noise are one
    # population, not two, and the gap between them is not the one sought.
    lg = lg[(lg <= -2.0) & (lg > lo_floor - 1.0)]
    # The widest empty stretch of the log-offset distribution below 1% of the
    # part. The leading stretch -- from the float floor up to the first offset --
    # counts too: a mesh whose coplanar pairs all come out exactly zero has no
    # noise population at all.
    edges = np.concatenate([[lo_floor - 1.0], lg])
    if len(edges) < 2:
        return floor, f"{floor:.2e} mm: four float32 steps (no offsets to read)"
    gaps = np.diff(edges)
    k = int(np.argmax(gaps))
    if gaps[k] < 1.0:
        return floor, (f"{floor:.2e} mm: four float32 steps -- the offsets between "
                       f"neighbouring facets show no clean gap between coplanar "
                       f"and not, so nothing looser can be justified")
    noise = 10 ** edges[k] * diag if k > 0 else 0.0
    geom = 10 ** edges[k + 1] * diag
    tol = max(floor, 4.0 * noise)
    if tol > geom / 4.0:
        tol = max(floor, geom / 4.0)
    how = (f"{tol:.2e} mm: coplanar facets agree to "
           + (f"{noise:.1e}" if noise > 0 else "exactly")
           + f", the closest pair that is not coplanar differs by {geom:.1e}"
           + f" (float32 step here {quantum:.1e})")
    return tol, how


def _pair_offsets(mesh):
    """Per adjacent pair: the distance of the smaller face's free vertex from the
    larger face's plane. The larger face, because a sliver's own normal is noise."""
    adj = mesh.face_adjacency
    if not len(adj):
        return np.zeros(0)
    fa = mesh.area_faces
    first_big = fa[adj[:, 0]] >= fa[adj[:, 1]]
    big = np.where(first_big, adj[:, 0], adj[:, 1])
    other = np.where(first_big, mesh.face_adjacency_unshared[:, 1],
                     mesh.face_adjacency_unshared[:, 0])
    v0 = mesh.vertices[mesh.faces[big, 0]]
    return np.abs(np.einsum("ij,ij->i", mesh.vertices[other] - v0,
                            mesh.face_normals[big]))


# ---------------------------------------------------------- planar patches

def planar_patches(mesh, tol):
    """Label every triangle with the planar patch it belongs to.

    Two neighbours join when the smaller one's free vertex lies on the larger
    one's plane and they face the same way. Components are then checked as a
    whole, because a chain of pairwise-coplanar slivers could in principle drift;
    a component that does not fit one plane is regrown from its largest face.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(mesh.faces)
    adj = np.asarray(mesh.face_adjacency).view(np.ndarray)
    d = _pair_offsets(mesh)
    fn = np.asarray(mesh.face_normals).view(np.ndarray)
    # Plain arrays, read once. Every `mesh.<attribute>` read checks trimesh's
    # cache by hashing the whole mesh, and reading them per patch and per facet
    # below was 658 of the 1859 s a `residual --plane` spent on a 109k-face
    # impeller (and, with the regrow scanning every adjacency pair of the whole
    # mesh once per patch, 1184 s more). Same arithmetic, same order.
    F = np.asarray(mesh.faces).view(np.ndarray)
    V = np.asarray(mesh.vertices).view(np.ndarray)
    area = np.asarray(mesh.area_faces).view(np.ndarray)
    same_side = np.einsum("ij,ij->i", fn[adj[:, 0]], fn[adj[:, 1]]) > 0
    # A sliver's normal is noise, so which way it faces says nothing; its
    # vertices lying on the neighbour's plane is the whole test for it.
    sliver = _sliver(mesh)
    ok = (d <= tol) & (same_side | sliver[adj[:, 0]] | sliver[adj[:, 1]])
    e = adj[ok]
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    _, labels = connected_components(g, directed=False)

    # Verify: every vertex of a patch on one plane.
    labels = labels.astype(np.int64)
    order = np.argsort(labels, kind="stable")
    bounds = np.flatnonzero(np.diff(labels[order])) + 1
    groups = np.split(order, bounds)
    next_id = labels.max() + 1
    arrays = (V, F, area, fn, adj)
    for grp in groups:
        if len(grp) < 3:
            continue
        vid = np.unique(F[grp])
        pts = V[vid]
        normal, off = _plane_of(pts, fn[grp], area[grp])
        if np.abs(pts @ normal - off).max() <= 2.0 * tol:
            continue
        for sub in _regrow_planes(mesh, grp, tol, arrays):
            labels[sub] = next_id
            next_id += 1
    # Dense, deterministic ids: by first face.
    _, dense = np.unique(labels, return_inverse=True)
    return dense.astype(np.int64)


def _sliver(mesh):
    """Triangles too thin for their normal to mean anything: area under a
    ten-thousandth of the square of the longest edge."""
    tri = mesh.triangles
    e = np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2).max(axis=1)
    return mesh.area_faces < 1e-4 * np.maximum(e * e, 1e-300)


def _plane_of(pts, normals, areas):
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[-1]
    if (normals * areas[:, None]).sum(axis=0) @ n < 0:
        n = -n
    return n, float(c @ n)


def _regrow_planes(mesh, faces, tol, arrays=None):
    """Split a component that does not fit one plane, seeding from big faces.

    `arrays` is (vertices, faces, areas, normals, adjacency) as plain arrays;
    the neighbour pairs are the adjacency rows with both faces in this
    component, in adjacency order, exactly as a scan of every pair found them.
    """
    if arrays is None:
        arrays = tuple(np.asarray(a).view(np.ndarray) for a in (
            mesh.vertices, mesh.faces, mesh.area_faces, mesh.face_normals,
            mesh.face_adjacency))
    V, F, area, fn, adj = arrays
    faces = set(int(f) for f in faces)
    member = np.zeros(len(F), dtype=bool)
    member[list(faces)] = True
    nbr = {}
    for a, b in adj[member[adj[:, 0]] & member[adj[:, 1]]].tolist():
        nbr.setdefault(a, []).append(b)
        nbr.setdefault(b, []).append(a)
    left = sorted(faces, key=lambda f: -area[f])
    done, out = set(), []
    for seed in left:
        if seed in done:
            continue
        n0 = fn[seed]
        o0 = float(V[F[seed, 0]] @ n0)
        grp, stack = [seed], [seed]
        done.add(seed)
        while stack:
            f = stack.pop()
            for g in nbr.get(f, ()):
                if g in done:
                    continue
                if np.abs(V[F[g]] @ n0 - o0).max() <= tol:
                    done.add(g)
                    grp.append(g)
                    stack.append(g)
        out.append(np.array(grp))
    return out


# ------------------------------------------------------------------ models

def _basis(a):
    a = a / np.linalg.norm(a)
    h = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(a, h)
    u /= np.linalg.norm(u)
    return u, np.cross(a, u)


def canonical_dir(a):
    """An axis and its opposite are one axis: point the largest component up."""
    a = np.asarray(a, float)
    a = a / np.linalg.norm(a)
    k = int(np.argmax(np.abs(a)))
    return -a if a[k] < 0 else a


class Model:
    """A fitted analytic surface. `signed` is positive on the outer side:
    away from an axis, a centre, or a torus's spine circle."""

    def __init__(self, kind, **p):
        self.kind = kind
        self.p = p
        self.dof = DOF[kind]
        self.max_res = np.inf
        self.sag = np.inf
        self._uv = None

    def basis(self):
        """Two directions across the axis, computed once per surface."""
        if self._uv is None:
            self._uv = _basis(self.p["axis"])
        return self._uv

    def _meridian(self, P):
        a, c = self.p["axis"], self.p["point"]
        q = P - c
        t = q @ a
        radial = q - np.outer(t, a)
        rho = np.linalg.norm(radial, axis=1)
        return t, rho, radial

    def signed(self, P):
        P = np.atleast_2d(P)
        k = self.kind
        if k == "sphere":
            return np.linalg.norm(P - self.p["centre"], axis=1) - self.p["radius"]
        t, rho, _ = self._meridian(P)
        if k == "cylinder":
            return rho - self.p["radius"]
        if k == "cone":
            s, b = self.p["slope"], self.p["offset"]
            return (rho - s * t - b) / np.sqrt(1.0 + s * s)
        return np.hypot(rho - self.p["major"], t) - self.p["minor"]

    def outward(self, P):
        P = np.atleast_2d(P)
        if self.kind == "sphere":
            v = P - self.p["centre"]
        else:
            a = self.p["axis"]
            t, rho, radial = self._meridian(P)
            r_hat = radial / np.maximum(rho, 1e-300)[:, None]
            if self.kind == "cylinder":
                v = r_hat
            elif self.kind == "cone":
                v = r_hat - self.p["slope"] * a
            else:
                v = (rho - self.p["major"])[:, None] * r_hat + t[:, None] * a
        return v / np.maximum(np.linalg.norm(v, axis=1), 1e-300)[:, None]

    def local_radius(self, P):
        """The tighter principal radius near P: what a facet's chord bows by."""
        k = self.kind
        if k in ("sphere", "cylinder"):
            return float(self.p["radius"])
        if k == "torus":
            return float(self.p["minor"])
        t, _, _ = self._meridian(np.atleast_2d(P))
        rho = np.abs(self.p["slope"] * t + self.p["offset"])
        return float(np.mean(rho) * np.sqrt(1.0 + self.p["slope"] ** 2))


def _circle2d(x, y):
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    cx, cy, c = sol
    r2 = c + cx * cx + cy * cy
    return cx, cy, np.sqrt(max(r2, 0.0))


def _axis_candidates(N, W, Q, kind=None):
    """Axis directions worth trying for a surface of revolution.

    Three, because each is exact for a different surface and useless for
    another: a cylinder's normals are all perpendicular to its axis, a cone's all
    make one angle with it, and every normal of any surface of revolution meets
    the axis -- which is the only one that finds a torus, and the one that is
    degenerate on a cylinder cut into full-height strips, whose normal lines all
    lie in one plane.
    """
    out = []
    S = (N * W[:, None]).T @ N
    out.append(np.linalg.eigh(S)[1][:, 0])
    mu = (N * W[:, None]).sum(axis=0) / W.sum()
    C = ((N - mu) * W[:, None]).T @ (N - mu)
    out.append(np.linalg.eigh(C)[1][:, 0])
    M = np.hstack([np.cross(Q, N), N]) * np.sqrt(W)[:, None]
    try:
        x = np.linalg.svd(M, full_matrices=False)[2][-1]
        if np.linalg.norm(x[:3]) > 1e-9:
            out.append(x[:3])
    except np.linalg.LinAlgError:
        pass
    # The one exact for this kind goes first; see the docstring for which.
    first = {"cylinder": 0, "cone": 1, "torus": 2}.get(kind, 0)
    if first < len(out):
        out = [out[first]] + out[:first] + out[first + 1:]
    uniq = []
    for a in out:
        a = canonical_dir(a)
        if all(abs(a @ b) < 1 - 1e-8 for b in uniq):
            uniq.append(a)
    return uniq


def _axis_point(a, Q, N, W):
    """Where the facet normals, seen along `a`, cross: a point on the axis."""
    u, v = _basis(a)
    q = np.column_stack([Q @ u, Q @ v])
    d = np.column_stack([N @ u, N @ v])
    s = np.linalg.norm(d, axis=1)
    keep = s > 1e-6
    if keep.sum() < 2:
        return None
    d = d[keep] / s[keep, None]
    n2 = np.column_stack([-d[:, 1], d[:, 0]])
    w = W[keep] * s[keep] ** 2
    A = (n2 * w[:, None]).T @ n2
    if np.linalg.cond(A) > 1e10:
        return None
    b = (n2 * w[:, None]).T @ np.einsum("ij,ij->i", n2, q[keep])
    c2 = np.linalg.solve(A, b)
    return c2[0] * u + c2[1] * v


def _lm(fj, x0, iters, give_up=None):
    """Levenberg-Marquardt for the small problems here.

    `fj(x)` returns the residuals and their Jacobian together, because both are
    read off one computation of the geometry. scipy's solver is the same
    method, but on twelve points its bookkeeping -- copies, comparisons,
    wrappers -- cost ten times the arithmetic, and a spline face asks for
    thousands of these fits, nearly all of them to fail.

    `give_up`, an RMS residual: a trial still that far off after eight steps is
    abandoned. On exact data a right surface is within a hair of it by then.
    """
    x = np.asarray(x0, float)
    r, J = fj(x)
    cost = float(r @ r)
    lam = None
    n = max(len(r), 1)
    for it in range(iters):
        if give_up is not None and it >= 8 and cost > give_up * give_up * n:
            break
        A = J.T @ J
        g = J.T @ r
        d = np.diag(A).copy()
        if lam is None:
            lam = 1e-3 * (float(d.max()) or 1.0)
        try:
            dx = np.linalg.solve(A + lam * np.diag(d + 1e-12 * (d.max() or 1.0)), -g)
        except np.linalg.LinAlgError:
            lam *= 10.0
            continue
        xn = x + dx
        rn, Jn = fj(xn)
        cn = float(rn @ rn)
        if np.isfinite(cn) and cn < cost:
            done = (cost - cn) <= 1e-12 * cost or float(np.abs(dx).max()) <= 1e-13
            x, r, J, cost = xn, rn, Jn, cn
            lam = max(lam / 3.0, 1e-15)
            if done:
                break
        else:
            lam *= 4.0
            if lam > 1e16:
                break
    return x


def _refine(kind, P, a0, c0, init, nfev=100, give_up=None):
    """Least squares on vertex distance, from an initial guess.

    The axis is perturbed in the plane perpendicular to its first guess and its
    point along two directions in that plane, so no parameter is free to slide
    along the axis and make the problem singular (a torus's centre alone gets
    the third direction, because it has a place along its axis).

    The Jacobians are analytic. A finite-difference one costs a residual
    evaluation per parameter per step, and on a part with a spline face the
    fits run by the thousand: numerical derivatives were four fifths of a
    486-second segmentation.
    """
    u, v = _basis(a0)
    E = (u, v, a0)
    ones = np.ones(len(P))

    def geom(x):
        w = a0 + x[0] * u + x[1] * v
        nw = np.linalg.norm(w)
        a = w / nw
        # d(a)/d(x0), d(a)/d(x1): the unit vector's derivative.
        da = [(e - a * (a @ e)) / nw for e in (u, v)]
        c = c0 + x[2] * u + x[3] * v
        if kind == "torus":
            c = c + x[4] * a0
        q = P - c
        t = q @ a
        rv = q - np.outer(t, a)
        rho = np.sqrt(np.einsum("ij,ij->i", rv, rv))
        rhat = rv / np.maximum(rho, 1e-300)[:, None]
        return a, da, q, t, rho, rhat

    # Per parameter: how rho and t move. Axis tilt: rho by -t (rhat . da), t by
    # q . da. Axis shift by e: rho by -(rhat . e), t by -(a . e).
    def moves(a, da, q, t, rhat):
        d_rho = [-t * (rhat @ g) for g in da]
        d_t = [q @ g for g in da]
        shifts = E if kind == "torus" else E[:2]
        d_rho += [-(rhat @ e) for e in shifts]
        d_t += [-(a @ e) * ones for e in shifts]
        return d_rho, d_t

    if kind == "cylinder":
        def fj(x):
            a, da, q, t, rho, rhat = geom(x)
            d_rho, _ = moves(a, da, q, t, rhat)
            return rho - x[4], np.column_stack(d_rho + [-ones])
        x0 = [0, 0, 0, 0, init["radius"]]
    elif kind == "cone":
        def fj(x):
            a, da, q, t, rho, rhat = geom(x)
            k, sk = x[4], np.sqrt(1 + x[4] ** 2)
            d_rho, d_t = moves(a, da, q, t, rhat)
            res = (rho - k * t - x[5]) / sk
            cols = [(dr - k * dt) / sk for dr, dt in zip(d_rho, d_t, strict=True)]
            cols += [-t / sk - res * k / sk ** 2, -ones / sk]
            return res, np.column_stack(cols)
        x0 = [0, 0, 0, 0, init["slope"], init["offset"]]
    else:
        def fj(x):
            a, da, q, t, rho, rhat = geom(x)
            d = np.maximum(np.hypot(rho - x[5], t), 1e-300)
            d_rho, d_t = moves(a, da, q, t, rhat)
            cols = [((rho - x[5]) * dr + t * dt) / d
                    for dr, dt in zip(d_rho, d_t, strict=True)]
            cols += [-(rho - x[5]) / d, -ones]
            return d - x[6], np.column_stack(cols)
        x0 = [0, 0, 0, 0, init["shift"], init["major"], init["minor"]]
    if len(P) < len(x0):
        return None
    try:
        x = _lm(fj, x0, nfev, give_up)
    except Exception:                                              # noqa: BLE001
        return None
    if not np.all(np.isfinite(x)):
        return None
    a = a0 + x[0] * u + x[1] * v
    a = a / np.linalg.norm(a)
    c = c0 + x[2] * u + x[3] * v
    if kind == "cylinder":
        return Model("cylinder", axis=a, point=c, radius=abs(x[4]))
    if kind == "cone":
        return Model("cone", axis=a, point=c, slope=x[4], offset=x[5])
    return Model("torus", axis=a, point=c + x[4] * a0, major=x[5], minor=abs(x[6]))


def _fit_sphere(P, nfev=100, give_up=None):
    A = np.column_stack([2 * P, np.ones(len(P))])
    sol, *_ = np.linalg.lstsq(A, (P * P).sum(axis=1), rcond=None)
    c0 = sol[:3]
    r0 = np.sqrt(max(sol[3] + c0 @ c0, 0.0))
    ones = np.ones(len(P))

    def fj(x):
        d = P - x[:3]
        n = np.maximum(np.sqrt(np.einsum("ij,ij->i", d, d)), 1e-300)
        return n - x[3], np.column_stack([-d / n[:, None], -ones])
    try:
        x = _lm(fj, np.r_[c0, r0], nfev, give_up)
    except Exception:                                              # noqa: BLE001
        return None
    if not np.all(np.isfinite(x)):
        return None
    return Model("sphere", centre=x[:3], radius=abs(x[3]))


def fit_round(kind, P, Q, N, W, start=None, tol=None, nfev=100):
    """Fit one kind of round surface to vertices P (facet centroids Q, normals N,
    areas W steer the start). Coordinates are normalised first: the parts here
    run from a 0.007 mm spring to a 670 mm frame, and a least-squares problem
    posed in raw millimetres at either end is badly scaled.

    `start`, a Model of the same kind, replaces the search over starting axes
    with that one surface: what a growing region needs when it refits.
    """
    mu = P.mean(axis=0)
    L = float(np.abs(P - mu).max()) or 1.0
    Pn, Qn = (P - mu) / L, (Q - mu) / L
    Ps = Pn[:: max(1, len(Pn) // FIT_SAMPLE)]
    tol_n = None if tol is None else tol / L
    give_up = 1e3 * tol_n if (tol_n is not None and nfev < 100) else None
    best = None
    if kind == "sphere":
        best = _fit_sphere(Ps, nfev, give_up)
    elif start is not None and start.kind == kind:
        a = start.p["axis"]
        c = (start.p["point"] - mu) / L
        c = c - (c @ a) * a
        t0 = ((start.p["point"] - mu) / L) @ a
        if kind == "cylinder":
            init = {"radius": start.p["radius"] / L}
        elif kind == "cone":
            init = {"slope": start.p["slope"],
                    "offset": (start.p["offset"] - start.p["slope"] * t0 * L) / L}
        else:
            init = {"shift": t0, "major": start.p["major"] / L,
                    "minor": start.p["minor"] / L}
        best = _refine(kind, Ps, a, c, init, nfev, give_up)
        if best is not None:
            best.max_res = float(np.abs(best.signed(Pn)).max())
    else:
        for k, a in enumerate(_axis_candidates(N, W, Qn, kind)):
            if (k and nfev < 100 and tol_n is not None and best is not None
                    and best.max_res > 1e3 * tol_n):
                # A trial fit from the axis that is exact for this kind and
                # still a thousand tolerances off: the other starts will not
                # rescue it, and on a spline face every trial is this one.
                break
            c = _axis_point(a, Qn, N, W)
            if c is None:
                c = Pn.mean(axis=0) - (Pn.mean(axis=0) @ a) * a
            q = Ps - c
            t = q @ a
            rho = np.linalg.norm(q - np.outer(t, a), axis=1)
            if kind == "cylinder":
                init = {"radius": float(np.median(rho))}
            elif kind == "cone":
                A = np.column_stack([t, np.ones_like(t)])
                (s, b), *_ = np.linalg.lstsq(A, rho, rcond=None)
                init = {"slope": s, "offset": b}
            else:
                t0, R, r = _circle2d(t, rho)
                init = {"shift": t0, "major": R, "minor": r}
            m = _refine(kind, Ps, a, c, init, nfev, give_up)
            if m is None:
                continue
            m.max_res = float(np.abs(m.signed(Pn)).max())
            if best is None or m.max_res < best.max_res:
                best = m
            if tol_n is not None and best.max_res <= tol_n:
                break
    if best is None:
        return None
    # Back to world units.
    p = dict(best.p)
    if kind == "sphere":
        p["centre"] = mu + L * p["centre"]
        p["radius"] *= L
    else:
        p["point"] = mu + L * p["point"]
        p["axis"] = canonical_dir(p["axis"]) if kind != "cone" else p["axis"]
        for key in ("radius", "offset", "major", "minor"):
            if key in p:
                p[key] = p[key] * L
    m = Model(kind, **p)
    if kind == "cone":
        # A flipped axis flips the sign of the slope; keep the axis canonical.
        ca = canonical_dir(m.p["axis"])
        if ca @ m.p["axis"] < 0:
            m.p["axis"] = ca
            m.p["slope"] = -m.p["slope"]
    m.max_res = float(np.abs(m.signed(P)).max())
    return m


def best_round(P, Q, N, W, mids, tol, nfev=100):
    """The simplest round surface that every vertex lies on, or None.

    Vertices alone cannot choose. A cylinder cut into full-height strips has all
    its vertices on two circles, and two equal circles lie on a sphere, a cone
    and a torus too -- so a sphere fits every vertex of such a cylinder exactly.
    What the sphere cannot do is contain the strip's long edges: those are
    straight rulings lying ON the cylinder, while on the sphere they are chords
    that sag. So among the surfaces that fit the vertices, the one the mesh's
    edges sag from least is the one it was cut from.

    Ties go to the simpler MERIDIAN, not to fewer parameters: a straight line
    parallel to the axis, then any straight line, then a circle centred on the
    axis, then any circle. A band one facet tall -- a chamfer round the top of
    a boss -- has its vertices on two circles and nothing else, and a sphere
    through them has fewer parameters than the cone; but the straight meridian
    is the plainer account of two points, and it is the chamfer.
    """
    fits = []
    quick = nfev < 100
    for kind in (MERIDIAN_ORDER if quick else ROUND_KINDS):
        if len(P) < DOF[kind] + 2:
            continue
        m = fit_round(kind, P, Q, N, W, tol=tol, nfev=nfev)
        if m is None or not np.isfinite(m.max_res) or m.max_res > tol:
            continue
        if not _plausible(m, P):
            continue
        m.sag = float(np.abs(m.signed(mids)).mean()) if len(mids) else 0.0
        fits.append(m)
        if quick:
            # A trial takes the plainest meridian that fits and stops: the
            # comparison of every kind is made once, on the finished surface.
            break
    if not fits:
        return None
    lo = min(m.sag for m in fits)
    near = [m for m in fits if m.sag <= lo * 1.1 + tol]
    return min(near, key=lambda m: MERIDIAN_ORDER.index(m.kind))


def _plausible(m, P):
    """Reject fits that are really a plane or a line in disguise."""
    ext = float(np.linalg.norm(P.max(axis=0) - P.min(axis=0))) or 1.0
    if m.kind in ("sphere", "cylinder"):
        return m.p["radius"] < 1e3 * ext
    if m.kind == "cone":
        return abs(m.p["slope"]) < 1e3 and abs(m.p["offset"]) < 1e3 * ext
    return m.p["minor"] < 1e3 * ext and abs(m.p["major"]) < 1e3 * ext


# ---------------------------------------------------------- the segmenter

class _Patches:
    """Per-patch geometry, precomputed once."""

    def __init__(self, mesh, plabels):
        self.mesh = mesh
        self.labels = plabels
        # Plain copies. trimesh's arrays check a hash of themselves on every
        # read, and the growth loop reads them once per candidate facet: on a
        # 31k-triangle torus that check alone was a quarter of the run.
        self.V = np.array(mesh.vertices, dtype=np.float64)
        self.tc = np.array(mesh.triangles_center, dtype=np.float64)
        n = int(plabels.max()) + 1
        order = np.argsort(plabels, kind="stable")
        cuts = np.flatnonzero(np.diff(plabels[order])) + 1
        self.faces = np.split(order, cuts)
        fa, fn = mesh.area_faces, mesh.face_normals
        tc = mesh.triangles_center
        self.area = np.array([fa[f].sum() for f in self.faces])
        self.normal = np.zeros((n, 3))
        self.centroid = np.zeros((n, 3))
        self.verts = []
        self.mids = []
        edges = mesh.edges_unique
        emid = mesh.vertices[edges].mean(axis=1)
        f2e = mesh.faces_unique_edges
        for i, f in enumerate(self.faces):
            w = fa[f]
            s = w.sum() or 1.0
            nv = (fn[f] * w[:, None]).sum(axis=0)
            self.normal[i] = nv / (np.linalg.norm(nv) or 1.0)
            self.centroid[i] = (tc[f] * w[:, None]).sum(axis=0) / s
            self.verts.append(np.unique(mesh.faces[f]))
            self.mids.append(emid[np.unique(f2e[f])])
        # Patch adjacency with the dihedral between patch normals.
        adj = mesh.face_adjacency
        pa, pb = plabels[adj[:, 0]], plabels[adj[:, 1]]
        cross = pa != pb
        pairs = np.unique(np.sort(np.column_stack([pa[cross], pb[cross]]), axis=1),
                          axis=0)
        self.nbrs = [[] for _ in range(n)]
        for a, b in pairs:
            ang = float(np.degrees(np.arccos(np.clip(
                self.normal[a] @ self.normal[b], -1.0, 1.0))))
            self.nbrs[a].append((ang, int(b)))
            self.nbrs[b].append((ang, int(a)))
        for lst in self.nbrs:
            lst.sort()

    def gather(self, ids):
        ids = list(ids)
        V = np.unique(np.concatenate([self.verts[i] for i in ids]))
        return (self.V[V], self.centroid[ids], self.normal[ids],
                self.area[ids], np.concatenate([self.mids[i] for i in ids]))


def _revolve_evidence(P, E, Q, N, W, rtol):
    """Axis and rings of a piece that might be a surface of revolution.

    P are its vertices, E its edges (pairs of indices into P). First axis: the
    line every facet normal line meets (Pluecker). A ring is then read off the
    mesh itself: vertices joined by edges that run round the axis -- an edge
    whose step along the profile, in (t, rho), is a small part of its length.
    That holds however finely the rings are spaced and however far the first
    axis is off, where splitting heights at a threshold did not. Vertices on no
    ring are seam vertices: the tessellator cuts the seam edge finer than the
    face and leaves one between rings, on the profile itself. The axis is
    refitted through the rings that go a good way round (a few vertices on a
    short arc give no usable centre): their planes' normal is its direction,
    their centres are on it. Returns a dict, or None without rings.
    """
    from scipy.interpolate import CubicSpline
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    mu = Q.mean(axis=0)
    L = float(np.abs(Q - mu).max()) or 1.0
    M = np.hstack([np.cross((Q - mu) / L, N), N]) * np.sqrt(W)[:, None]
    try:
        x = np.linalg.svd(M, full_matrices=False)[2][-1]
    except np.linalg.LinAlgError:
        return None
    if np.linalg.norm(x[:3]) < 1e-9:
        return None
    a = x[:3] / np.linalg.norm(x[:3])
    c = mu + L * np.cross(a, x[3:] / np.linalg.norm(x[:3]))
    extent = float(np.linalg.norm(P.max(axis=0) - P.min(axis=0))) or 1.0
    n = len(P)
    rings, stray = None, None
    for _ in range(3):
        t = (P - c) @ a
        rho = np.linalg.norm((P - c) - np.outer(t, a), axis=1)
        dt = t[E[:, 1]] - t[E[:, 0]]
        dr = rho[E[:, 1]] - rho[E[:, 0]]
        length = np.linalg.norm(P[E[:, 1]] - P[E[:, 0]], axis=1)
        round_edge = np.hypot(dt, dr) <= REVOLVE_RING_EDGE * np.maximum(length, 1e-300)
        e = E[round_edge]
        g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
        _, lab = connected_components(g, directed=False)
        comps = [np.flatnonzero(lab == k) for k in np.unique(lab)]
        arcs = [cmp for cmp in comps if len(cmp) >= 3]
        stray = np.concatenate([cmp for cmp in comps if len(cmp) < 3] or [np.zeros(0, int)])
        if len(arcs) < 2:
            return None
        # Arcs of one ring broken by the seam are one ring: same height.
        arcs.sort(key=lambda r: float(t[r].mean()))
        rings = [arcs[0]]
        for r in arcs[1:]:
            if t[r].mean() - t[rings[-1]].mean() <= max(50.0 * rtol, 1e-6 * extent):
                rings[-1] = np.concatenate([rings[-1], r])
            else:
                rings.append(r)
        u, v = _basis(a)
        good = []
        for r in rings:
            q = P[r] - c
            if _turn(np.degrees(np.arctan2(q @ v, q @ u))) >= REVOLVE_MIN_TURN:
                good.append(r)
        if len(good) < 2:
            return None
        nsum, centres, wts = np.zeros(3), [], []
        for r in good:
            R = P[r]
            cr = R.mean(axis=0)
            nrm = np.linalg.svd(R - cr, full_matrices=False)[2][-1]
            if nrm @ a < 0:
                nrm = -nrm
            u2, v2 = _basis(nrm)
            cx, cy, _ = _circle2d((R - cr) @ u2, (R - cr) @ v2)
            nsum += len(r) * nrm
            centres.append(cr + cx * u2 + cy * v2)
            wts.append(len(r))
        a = nsum / np.linalg.norm(nsum)
        c = np.average(np.array(centres), axis=0, weights=wts)
    t = (P - c) @ a
    rho = np.linalg.norm((P - c) - np.outer(t, a), axis=1)
    ring_dev = max(max(np.ptp(rho[r]), np.ptp(t[r])) / 2.0 for r in rings)
    prof = sorted((float(t[r].mean()), float(rho[r].mean()), len(r)) for r in rings)
    stray_dev = 0.0
    ts = [q_[0] for q_ in prof]
    if len(stray) and len(prof) >= 3 and np.all(np.diff(ts) > 0):
        cs = CubicSpline(ts, [q_[1] for q_ in prof])
        inside = (t[stray] >= prof[0][0]) & (t[stray] <= prof[-1][0])
        if inside.any():
            stray_dev = float(np.abs(rho[stray][inside] - cs(t[stray][inside])).max())
    return {"axis": a, "point": c, "rings": rings, "profile": prof,
            "dev": float(ring_dev), "stray": int(len(stray)), "stray_dev": stray_dev,
            "t": t, "rho": rho}


def _as_revolve(mesh, pt, s, rtol, diag):
    """Make a freeform piece a revolve if all its vertices sit on circles round
    one axis. Every other kind here is a revolve too, of a line or a circle;
    this one's profile is only known at its rings, and is given as those."""
    ids = [int(i) for i in s._patches]
    if len(ids) < 3:
        return False
    _, Q, N, W, _ = pt.gather(ids)
    vid, inv = np.unique(mesh.faces[s.faces], return_inverse=True)
    if len(vid) < 3 * REVOLVE_MIN_RINGS:
        return False
    P = np.asarray(mesh.vertices, float)[vid]
    tri = inv.reshape(-1, 3)
    E = np.unique(np.sort(tri[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1), axis=0)
    ev = _revolve_evidence(P, E, Q, N, W, rtol)
    if ev is None or len(ev["rings"]) < REVOLVE_MIN_RINGS:
        return False
    lim = max(REVOLVE_K * rtol, REVOLVE_REL * diag)
    if ev["dev"] > lim:
        return False
    # Seam vertices are few and lie on the profile; a spline through the rings
    # only approximates it between them, so they get more room.
    if ev["stray"] > REVOLVE_STRAY * len(P) or ev["stray_dev"] > REVOLVE_STRAY_K * lim:
        return False
    a, c = canonical_dir(ev["axis"]), ev["point"]
    t = (P - c) @ a
    rho = np.linalg.norm((P - c) - np.outer(t, a), axis=1)
    prof = sorted((float(t[r].mean()), float(rho[r].mean())) for r in ev["rings"])
    s.kind = "revolve"
    s.params = {"axis": a, "point": c, "profile": [list(p) for p in prof]}
    s.residual = float(ev["dev"])
    # which way it faces: normals along the radial direction or against it
    radial = (Q - c) - np.outer((Q - c) @ a, a)
    radial /= np.maximum(np.linalg.norm(radial, axis=1), 1e-300)[:, None]
    agree = float((np.einsum("ij,ij->i", N, radial) * W).sum() / (W.sum() or 1.0))
    s.sense = "convex" if agree > 0.5 else "concave" if agree < -0.5 else "mixed"
    u, v = _basis(a)
    q = P - c
    ang = np.sort(np.degrees(np.arctan2(q @ v, q @ u)) % 360.0)
    gaps = np.diff(np.r_[ang, ang[0] + 360.0])
    s.extent["along"] = [float(t.min()), float(t.max())]
    s.extent["radius"] = [float(rho.min()), float(rho.max())]
    s.extent["sweep"] = 360.0 if gaps.max() <= 1.5 * max(s.facet_angle, 10.0) else float(360.0 - gaps.max())
    s.notes.append(f"a revolve of a profile through {len(prof)} rings of vertices "
                   f"(every vertex within {s.residual:.2g} mm of its ring's circle); "
                   f"no plane, cylinder, cone, sphere or torus fits it")
    return True


def segment(mesh, tol=None, budget=None):
    """Segment a mesh into named analytic surfaces. See the module docstring.

    `budget`, in seconds, bounds the search for round surfaces. `faces` has
    none; the level table, which every `section` and `sketch` builds, has one,
    because on a part that is mostly spline a complete search takes minutes and
    the levels degrade gracefully without it (the sweep's own levels fill in).
    Stopping early is recorded in `partial` and said in a note.
    """
    import time
    t0 = time.time()
    diag = float(np.linalg.norm(mesh.extents)) or 1.0
    if tol is None:
        tol, how = derive_tolerance(mesh)
    else:
        how = f"{tol:.2e} mm: given with --tol"
    plabels = planar_patches(mesh, tol)
    pt = _Patches(mesh, plabels)
    P = len(pt.faces)
    owner = np.full(P, -1)
    regions = []
    # A round surface is seeded within `tol` and grows within `rtol`; planar
    # patches use `tol` throughout.
    rtol = tol
    # Only when the floor was a fallback -- no gap between coplanar and not --
    # not when a measured gap happened to put the tolerance on the floor.
    at_floor = "no clean gap" in how or "no offsets to read" in how
    if FLOOR_GROWTH_MARGIN and at_floor:
        rtol = tol * FLOOR_MARGIN

    seeds = [i for i in range(P)
             if sum(1 for a, _ in pt.nbrs[i] if a < SMOOTH_BOUND_DEG) >= 2]
    seeds.sort(key=lambda i: (pt.area[i], i))
    # A seed whose neighbourhood fits nothing retires its same-sized neighbours
    # as seeds too: their neighbourhoods overlap its own, and on a spline face
    # -- where nothing fits anywhere -- trying every facet was 1300 failed
    # starts and four fifths of a five-minute run. They stay free to be TAKEN
    # by a surface grown from somewhere else; only starting from them is skipped.
    retired = np.zeros(len(pt.faces), bool)
    stopped_at = None
    for k_seed, seed in enumerate(seeds):
        if budget is not None and time.time() - t0 > budget:
            stopped_at = k_seed
            break
        if owner[seed] >= 0 or retired[seed]:
            continue
        got = _grow(pt, seed, owner, rtol, seed_tol=tol)
        if got is None:
            a0 = max(pt.area[seed], 1e-300)
            for hood in _neighbourhoods(pt, seed, owner):
                for q in hood:
                    r = max(pt.area[q], 1e-300) / a0
                    if 1 / HOOD_AREA_RATIO <= r <= HOOD_AREA_RATIO:
                        retired[q] = True
            continue
        ids, model = got
        rid = len(regions)
        owner[ids] = rid
        regions.append((ids, model))
    regions = _absorb(pt, owner, regions, rtol)
    owner[:] = -1
    for rid, (ids, _) in enumerate(regions):
        owner[ids] = rid
    regions = _sweep_leftovers(pt, owner, regions, rtol)

    surfaces, notes, demoted = _assemble(mesh, pt, owner, regions, tol)
    if FIND_REVOLVES:
        for s in surfaces:
            if s.kind == "freeform":
                _as_revolve(mesh, pt, s, rtol, diag)
    if rtol != tol:
        notes.append(f"the tolerance is the float32 floor; round surfaces were "
                     f"allowed {FLOOR_MARGIN:g} of it ({rtol:.2e} mm), planes "
                     f"the floor itself")
    labels = np.full(len(mesh.faces), -1, dtype=np.int64)
    for k, s in enumerate(surfaces):
        labels[s.faces] = k
    joints = _joints(mesh, labels, surfaces)
    for s in surfaces:
        s.loops = _loops(mesh, s.faces)
    watertight = bool(mesh.is_watertight)
    if not watertight:
        notes.append("the mesh is OPEN: surfaces that run into a hole in it are "
                     "measured only as far as the hole")
    seg = Segmentation(surfaces=surfaces, labels=labels, joints=joints, tol=tol,
                       tol_how=how, diag=diag, area=float(mesh.area),
                       n_faces=int(len(mesh.faces)), watertight=watertight,
                       notes=notes, demoted=demoted)
    _exactness_note(seg, pt)
    if stopped_at is not None:
        left = [q for q in seeds[stopped_at:] if owner[q] < 0 and not retired[q]]
        seg.partial = True
        seg.budget = float(budget)
        seg.untried = len(left)
        seg.untried_area = float(sum(pt.area[q] for q in left)) / max(float(mesh.area), 1e-300)
        seg.notes.append(
            f"stopped looking for round surfaces after {budget:.0f} s with "
            f"{len(left)} candidate facets untried; what they belong to is "
            f"reported as planes or freeform")
    return seg


def _grow(pt, seed, owner, tol, seed_tol=None):
    """Grow one round surface from a seed patch, or None if none fits."""
    start = None
    for hood in _neighbourhoods(pt, seed, owner):
        got = _first_fit(pt, hood, tol if seed_tol is None else seed_tol)
        if got is not None:
            start = got
            break
    if start is None:
        return None
    region, model = set(start[0]), start[1]
    heap = []
    for i in region:
        for ang, q in pt.nbrs[i]:
            heapq.heappush(heap, (ang, q))
    blocked = set()
    upgrades = 8
    fitted_at = len(region)
    step = _Step(model, pt, region)
    while heap:
        ang, q = heapq.heappop(heap)
        if q in region or owner[q] >= 0 or q in blocked:
            continue
        if ang >= SMOOTH_BOUND_DEG:
            continue
        ok = _takes(model, pt, q, tol, step)
        # Only a facet the size of the surface's own can be the surface's next
        # facet; the big neighbour it is blocked by is where the face ends.
        if (not ok and upgrades > 0 and len(region) <= UPGRADE_PATCHES
                and _similar(pt, q, region)):
            # The surface may be something the first few facets could not tell
            # apart from what was fitted: a band of a sphere is exactly a cone.
            upgrades -= 1
            m2 = best_round(*pt.gather(region | {q}), tol, nfev=QUICK_NFEV)
            if m2 is not None:
                s2 = _Step(m2, pt, region)
                if _takes(m2, pt, q, tol, s2):
                    model, ok, step = m2, True, s2
        if not ok:
            blocked.add(q)
            continue
        region.add(q)
        step.add(model, pt, q)
        for a2, q2 in pt.nbrs[q]:
            if q2 not in region and owner[q2] < 0:
                heapq.heappush(heap, (a2, q2))
        if len(region) >= 1.5 * fitted_at:
            fitted_at = len(region)
            m2 = fit_round(model.kind, *pt.gather(region)[:4], start=model)
            if m2 is not None and m2.max_res <= tol:
                model = m2
    if len(region) < 3:
        return None
    ids = np.array(sorted(region))
    final = best_round(*pt.gather(ids), tol)
    return ids, (final or model)


def _similar(pt, q, ids):
    ref = float(np.median(pt.area[sorted(ids)[:SPAN_SAMPLE]])) or 1e-300
    r = max(pt.area[q], 1e-300) / ref
    return 1.0 / HOOD_AREA_RATIO <= r <= HOOD_AREA_RATIO


def _sag(model, pt, i):
    """How far a patch bows from the surface: the furthest of its triangles'
    centres. Per triangle, not per patch: a ring of facets round the crown of a
    torus has its centroid on the axis, a whole radius away, while every one of
    its triangles lies on the torus."""
    return float(np.abs(model.signed(pt.tc[pt.faces[i]])).max())


def _span(model, pt, i):
    """How much of the surface one patch covers, in degrees: its turn about the
    axis, or for a sphere the angle it subtends at the centre."""
    V = pt.V[pt.verts[i]]
    if model.kind == "sphere":
        d = V - model.p["centre"]
        d = d / np.maximum(np.linalg.norm(d, axis=1), 1e-300)[:, None]
        m = d.mean(axis=0)
        m = m / (np.linalg.norm(m) or 1.0)
        return float(2.0 * np.degrees(np.arccos(np.clip(d @ m, -1.0, 1.0)).max()))
    u, v = model.basis()
    q = V - model.p["point"]
    return _turn(np.degrees(np.arctan2(q @ v, q @ u)))


def _turn(ang):
    """The smallest arc containing every angle, in degrees."""
    ang = np.sort(np.asarray(ang) % 360.0)
    if len(ang) < 2:
        return 0.0
    gaps = np.diff(np.r_[ang, ang[0] + 360.0])
    return float(360.0 - gaps.max())


def _first_fit(pt, hood, tol):
    """A first surface for a seed's neighbourhood, with the outliers dropped.

    Extra parameters can absorb a patch that does not belong: a disc capping a
    fillet has all its vertices on the fillet's edge circle, and a sphere of
    radius 328 goes through that circle and the fillet's first ring of facets
    alike. The disc gives itself away by how much of the surface it claims to
    cover -- dozens of times the step the fillet was cut at -- so a fit whose
    patches disagree that much is refitted without the ones that stand out.
    """
    # A trial: on exact data a round surface converges in a handful of
    # steps, and one that is still wandering at forty is not going to fit.
    m = best_round(*pt.gather(hood), tol, nfev=QUICK_NFEV)
    if m is None:
        return None
    ref = sorted(hood)[:SPAN_SAMPLE]
    step = _Step(m, pt, ref)
    keep = [i for i in hood
            if _sag(m, pt, i) <= _chord_bound(m, pt, i, tol)
            and not step.stands_out(_span(m, pt, i), _sag(m, pt, i), tol,
                                    skip=ref.index(i) if i in ref else None)]
    if len(keep) == len(hood):
        return hood, m
    if len(keep) < 3:
        return None
    m = best_round(*pt.gather(keep), tol)
    return (keep, m) if m is not None else None


def _chord_bound(model, pt, i, tol):
    """How far a facet of this surface can be from it: a chord across up to
    90 degrees of arc bows by (1 - cos 45) of the local radius."""
    return ((1.0 - np.cos(np.radians(SMOOTH_BOUND_DEG / 2)))
            * model.local_radius(pt.centroid[i]) + tol)


def _neighbourhoods(pt, seed, owner):
    """Small sets of patches around a seed to fit a first surface to.

    The whole smooth one-ring first, grown until it holds enough vertices to
    over-determine a torus; then each one-sided chain, for a seed that sits on
    the boundary between two surfaces and would poison a fit taken both ways.
    Neighbours of the seed's own size come first: a tessellator cuts one face
    into facets of one size, and the big neighbour at a small angle is usually
    the plane a fillet runs into.
    """
    a0 = max(pt.area[seed], 1e-300)

    def order(items):
        return sorted(items, key=lambda aq: (round(abs(np.log(
            max(pt.area[aq[1]], 1e-300) / a0)) * 2) / 2, aq[0]))

    def chain(first):
        hood, frontier = [seed] + first, list(first)
        while frontier and _n_verts(pt, hood) < 12 and len(hood) < 10:
            nxt = []
            for f in frontier:
                for ang, q in order(pt.nbrs[f]):
                    if (ang < SMOOTH_BOUND_DEG and q not in hood and owner[q] < 0
                            and similar(q)):
                        hood.append(q)
                        nxt.append(q)
                        break
            frontier = nxt
        return hood

    def similar(q):
        r = max(pt.area[q], 1e-300) / a0
        return 1.0 / HOOD_AREA_RATIO <= r <= HOOD_AREA_RATIO

    ring = [q for a, q in order(pt.nbrs[seed])
            if a < SMOOTH_BOUND_DEG and owner[q] < 0 and similar(q)]
    out = []
    if len(ring) >= 2:
        out.append(chain(ring[:4]))
    for q in ring[:2]:
        out.append(chain([q]))
    seen, uniq = set(), []
    for h in out:
        key = tuple(sorted(h))
        if len(h) >= 3 and key not in seen:
            seen.add(key)
            uniq.append(h)
    return uniq


def _n_verts(pt, ids):
    return len(np.unique(np.concatenate([pt.verts[i] for i in ids])))


class _Step:
    """The tessellation step of a growing surface: how much of it one facet
    covers and how far one facet bows from it, read off the first facets."""

    def __init__(self, model, pt, ids):
        ids = sorted(ids)[:SPAN_SAMPLE]
        self.spans = [_span(model, pt, i) for i in ids]
        self.sags = [_sag(model, pt, i) for i in ids]
        self._med = None

    def add(self, model, pt, q):
        if len(self.spans) < SPAN_SAMPLE:
            self.spans.append(_span(model, pt, q))
            self.sags.append(_sag(model, pt, q))
            self._med = None

    def stands_out(self, span, sag, tol, skip=None):
        """`skip` leaves one facet out of the reference, so a patch is never
        measured against a median it helped set."""
        if skip is None:
            if self._med is None:
                self._med = (_coarse(self.spans), _coarse(self.sags),
                             len(self.spans))
            med_span, med_sag, n = self._med
        else:
            spans = [x for k, x in enumerate(self.spans) if k != skip]
            sags = [x for k, x in enumerate(self.sags) if k != skip]
            n = len(spans)
            med_span, med_sag = _coarse(spans), _coarse(sags)
        if n < 2:
            return False
        wide = span > SPREAD * med_span + 1e-9
        deep = sag > SPREAD ** 2 * max(med_sag, tol)
        return wide and deep


def _coarse(values):
    return float(np.percentile(values, STEP_QUANTILE)) if len(values) else 0.0


def _takes(model, pt, q, tol, step):
    """Does patch q belong to this round surface?

    Every vertex on it; the patch no further from it than a chord of it can
    bow; and the patch not standing out from the facets already taken, both
    wider and deeper (see SPREAD). The last is what keeps a D-shaft's flat out
    of the shaft: all four of the flat's corners lie on the cylinder, but it
    spans ten strips' worth of turn and its middle is the flat's depth away.
    """
    V = pt.V[pt.verts[q]]
    if np.abs(model.signed(V)).max() > tol:
        return False
    sag = _sag(model, pt, q)
    if sag > _chord_bound(model, pt, q, tol):
        return False
    return not step.stands_out(_span(model, pt, q), sag, tol)


def _sweep_leftovers(pt, owner, regions, tol):
    """Offer every unclaimed facet to the finished surfaces beside it, again.

    While a surface grows it is judged by the model it had at the time, fitted
    to its first few facets and good near them only; a facet it met early and
    turned away is never asked again. With every surface settled, the same
    test is run once more with the final model -- a facet that was a bad fit
    for a guess and a good fit for the answer is part of the answer.
    """
    changed = True
    while changed:
        changed = False
        for r, (ids, model) in enumerate(regions):
            step = _Step(model, pt, ids)
            took = []
            members = set(ids.tolist())
            frontier = {q for i in ids for ang, q in pt.nbrs[i]
                        if ang < SMOOTH_BOUND_DEG and owner[q] < 0}
            while frontier:
                q = frontier.pop()
                if owner[q] >= 0 or not _takes(model, pt, q, tol, step):
                    continue
                owner[q] = r
                took.append(q)
                members.add(q)
                frontier |= {q2 for ang, q2 in pt.nbrs[q]
                             if ang < SMOOTH_BOUND_DEG and owner[q2] < 0}
            if took:
                regions[r] = (np.array(sorted(members)), model)
                changed = True
    return regions


def _absorb(pt, owner, regions, tol):
    """Fold small round regions into a neighbour whose surface they lie on.

    A handful of facets can seed a surface of their own before the big one
    they belong to reaches them -- a few strips at the small end of a cone, fit
    exactly by a slightly tilted cylinder because five parameters will go
    through eight points. Once the big surface exists, it says whether they are
    on it; if every vertex is, they are one face.
    """
    changed = True
    while changed:
        changed = False
        order = sorted(range(len(regions)), key=lambda r: -len(regions[r][0]))
        for r in order:
            ids, model = regions[r]
            if not len(ids):
                continue
            near = set()
            for i in ids:
                for _, q in pt.nbrs[i]:
                    o = owner[q]
                    if o >= 0 and o != r and 0 < len(regions[o][0]) <= len(ids):
                        near.add(int(o))
            for o in sorted(near):
                other = regions[o][0]
                V = pt.V[np.unique(np.concatenate(
                    [pt.verts[i] for i in other]))]
                if np.abs(model.signed(V)).max() > tol:
                    continue
                merged = np.array(sorted(set(ids.tolist()) | set(other.tolist())))
                m2 = fit_round(model.kind, *pt.gather(merged)[:4], start=model)
                if m2 is None or m2.max_res > tol:
                    continue
                owner[other] = r
                regions[o] = (np.array([], dtype=np.int64), None)
                ids, model = merged, m2
                regions[r] = (ids, model)
                changed = True
    return [(ids, m) for ids, m in regions if len(ids)]


# ----------------------------------------------------------- assembling

def _assemble(mesh, pt, owner, regions, tol):
    """Round surfaces from the regions, planes and freeform from the rest."""
    fa = mesh.area_faces
    notes, demoted = [], []

    round_s = []
    tmp = np.full(len(mesh.faces), -1, dtype=np.int64)
    for r, (ids, model) in enumerate(regions):
        faces = np.concatenate([pt.faces[i] for i in ids])
        tmp[faces] = r
        round_s.append((ids, faces, model))
    steps = _facet_angles(mesh, tmp)

    built = []
    for r, (ids, faces, model) in enumerate(round_s):
        V = mesh.vertices[np.unique(mesh.faces[faces])]
        s = Surface(kind=model.kind, faces=faces, params=dict(model.p),
                    area=float(fa[faces].sum()))
        s.residual = float(np.abs(model.signed(V)).max())
        s.sagitta = float(np.abs(model.signed(mesh.triangles_center[faces])).max())
        s.facet_angle = steps.get(r, 0.0)
        s._model = model
        s._patches = ids
        built.append(s)

    # The coarse ones: fewer than twelve facets a turn. A hexagonal prism and a
    # cylinder cut into six strips are the same triangles, so the only honest
    # arbiter is the mesh's own tessellation rule, read off the surfaces it cut
    # finely: a real round face at this radius would bow no more than those do.
    fine = [s for s in built if s.facet_angle <= COARSE_FACET_DEG]
    chord = max((s.sagitta for s in fine), default=None)
    keep = []
    for s in built:
        if s.facet_angle <= COARSE_FACET_DEG:
            keep.append(s)
            continue
        spans = [_span(s._model, pt, i) for i in s._patches]
        if max(spans) > SPREAD * max(min(spans), 1e-9):
            # A coarse surface is believed only if it looks like a
            # tessellation: every facet the same step. Faces of unequal width
            # whose corners happen to share a circle -- a box's top and its
            # chamfers -- are flat faces, and are simply left as planes.
            owner[s._patches] = -1
        elif chord is not None and s.sagitta > 2.0 * max(chord, tol):
            demoted.append((s.kind, len(s._patches), s.facet_angle, s.sagitta, chord))
            owner[s._patches] = -1
        else:
            if chord is None:
                s.notes.append(f"coarse: {360.0 / max(s.facet_angle, 1e-9):.0f} "
                               f"facets a turn -- could equally be a prism with "
                               f"that many flat sides")
            keep.append(s)
    for kind, n, ang, sag, ch in demoted:
        notes.append(f"{n} flat faces meeting at {ang:.0f} deg fit a {kind}, but "
                     f"bow {sag:.3g} mm from it where this mesh's finely cut round "
                     f"faces bow at most {ch:.3g} mm -- kept as flat faces "
                     f"(a polygonal prism, not a round one)")

    # Whatever no round surface took: planes, or freeform where a smooth run of
    # patches fits nothing analytic.
    rest = [i for i in range(len(pt.faces)) if owner[i] < 0]
    # Facets meeting at no more than a tessellation step are one smooth face:
    # the coarsest step this mesh cut its round faces at, and never less than
    # the coarse bound -- a loft has almost no round faces to read a step off,
    # and with none it came apart into five hundred single-facet "planes".
    smooth = max(max((s.facet_angle for s in keep), default=0.0) * 1.25,
                 COARSE_FACET_DEG)
    rest_set = set(rest)
    # An exact flat patch that dwarfs every leftover patch it meets smoothly is
    # a face that happens to be tangent to something unrecognised -- a pad's
    # wall running into a fillet cut into two facets -- not one link of a
    # spline's chain of similar facets. It stays a plane.
    solo = set()
    if KEEP_DOMINANT_PLANES:
        for i in rest:
            nb = [pt.area[q] for ang, q in pt.nbrs[i] if ang <= smooth and q in rest_set]
            if (len(pt.faces[i]) >= 2 and nb
                    and pt.area[i] >= PLANE_DOMINANCE * max(nb)):
                solo.add(i)
    rest_set -= solo
    comp = {}
    for i in rest:
        if i in comp or i in solo:
            continue
        cid = i
        stack = [i]
        comp[i] = cid
        while stack:
            f = stack.pop()
            for ang, q in pt.nbrs[f]:
                if ang <= smooth and q in rest_set and q not in comp:
                    comp[q] = cid
                    stack.append(q)
    groups = {}
    for i in rest:
        if i in solo:
            groups[("solo", i)] = [i]
        else:
            groups.setdefault(comp[i], []).append(i)
    out = list(keep)
    tmp[:] = -1
    free = []
    for members in groups.values():
        if len(members) >= 3:
            faces = np.concatenate([pt.faces[i] for i in members])
            tmp[faces] = len(free)
            free.append((members, faces))
    steps = _facet_angles(mesh, tmp)
    for g, (members, faces) in enumerate(free):
        s = Surface(kind="freeform", faces=faces, params={},
                    area=float(fa[faces].sum()), sense="mixed")
        s.facet_angle = steps.get(g, 0.0)
        s._patches = np.array(members)
        out.append(s)
    for members in groups.values():
        if len(members) >= 3:
            continue
        for i in members:
            faces = pt.faces[i]
            V = mesh.vertices[pt.verts[i]]
            n = pt.normal[i]
            off = float(pt.centroid[i] @ n)
            s = Surface(kind="plane", faces=faces,
                        params={"normal": n, "offset": off},
                        area=float(pt.area[i]))
            s.residual = float(np.abs(V @ n - off).max())
            s._patches = np.array([i])
            out.append(s)

    for s in out:
        _describe(mesh, s)
    out.sort(key=lambda s: (-s.area, s.kind, tuple(np.round(
        mesh.triangles_center[s.faces].mean(axis=0), 6))))
    for k, s in enumerate(out):
        s.name = f"F{k}"
    return out, notes, demoted


def _facet_angles(mesh, face_label):
    """The tessellation step inside each labelled set: the 95th percentile of
    the dihedral angle between neighbouring facets that both belong to it."""
    adj = mesh.face_adjacency
    la, lb = face_label[adj[:, 0]], face_label[adj[:, 1]]
    inside = (la == lb) & (la >= 0)
    if not inside.any():
        return {}
    lab = la[inside]
    ang = np.degrees(mesh.face_adjacency_angles[inside])
    order = np.argsort(lab, kind="stable")
    lab, ang = lab[order], ang[order]
    cuts = np.flatnonzero(np.diff(lab)) + 1
    return {int(g[0]): float(np.percentile(a, 95))
            for g, a in zip(np.split(lab, cuts), np.split(ang, cuts), strict=True)}


def _describe(mesh, s):
    """Sense and extent, in terms a reader can check against the renders."""
    faces = s.faces
    V = np.asarray(mesh.vertices)[np.unique(mesh.faces[faces])]
    s.extent["bbox"] = [np.array(V.min(axis=0)), np.array(V.max(axis=0))]
    if s.kind in ("plane", "freeform"):
        return
    m = s._model
    fn = mesh.face_normals[faces]
    w = mesh.area_faces[faces]
    out = m.outward(mesh.triangles_center[faces])
    agree = float((np.einsum("ij,ij->i", fn, out) * w).sum() / (w.sum() or 1.0))
    s.sense = "convex" if agree > 0.5 else "concave" if agree < -0.5 else "mixed"
    if s.kind == "sphere":
        return
    a = s.params["axis"]
    t = (V - s.params["point"]) @ a
    s.extent["along"] = [float(t.min()), float(t.max())]
    # Where along the axis in world terms: the axis coordinate itself when the
    # axis is a principal one, which is how every other instrument names places.
    u, v = _basis(a)
    q = V - s.params["point"]
    ang = np.sort(np.degrees(np.arctan2(q @ v, q @ u)) % 360.0)
    gaps = np.diff(np.r_[ang, ang[0] + 360.0])
    k = int(np.argmax(gaps))
    gap = float(gaps[k])
    full = gap <= 1.5 * max(s.facet_angle, 1e-6) + 1e-6
    s.extent["sweep"] = 360.0 if full else 360.0 - gap
    if s.kind == "cone":
        rho = np.abs(m.p["slope"] * t + m.p["offset"])
        s.extent["radius"] = [float(rho.min()), float(rho.max())]
        s.params["half_angle"] = float(np.degrees(np.arctan(abs(m.p["slope"]))))


def _joints(mesh, labels, surfaces):
    """How each pair of surfaces meets, along their shared mesh edges."""
    adj = mesh.face_adjacency
    la, lb = labels[adj[:, 0]], labels[adj[:, 1]]
    x = (la != lb) & (la >= 0) & (lb >= 0)
    if not x.any():
        return []
    ev = mesh.vertices[mesh.face_adjacency_edges[x]]
    length = np.linalg.norm(ev[:, 0] - ev[:, 1], axis=1)
    ang = np.degrees(mesh.face_adjacency_angles[x])
    convex = mesh.face_adjacency_convex[x]
    a, b = np.minimum(la[x], lb[x]), np.maximum(la[x], lb[x])
    key = a * (len(surfaces) + 1) + b
    out = []
    for k in np.unique(key):
        sel = key == k
        i, j = int(a[sel][0]), int(b[sel][0])
        L = float(length[sel].sum()) or 1e-300
        mean = float((ang[sel] * length[sel]).sum() / L)
        cvx = float((length[sel] * convex[sel]).sum() / L)
        si, sj = surfaces[i], surfaces[j]
        step = max(si.facet_angle, sj.facet_angle)
        # At a tangent seam each curved side's first facet leans off the shared
        # tangent plane by half its step. A plane does not lean, so a fillet
        # meets its plane at half a step; two curved faces lean opposite ways
        # and meet at up to a whole one -- which is how a chain of arcs on a
        # spline profile read as a run of convex creases at 7 to 10 degrees.
        both_curved = si.kind != "plane" and sj.kind != "plane"
        limit = (1.25 if both_curved else 0.75) * step
        if (si.kind != "plane" or sj.kind != "plane") and step > 0 \
                and mean <= limit + 1e-6:
            kind = "tangent"
        elif cvx >= 0.9:
            kind = "convex"
        elif cvx <= 0.1:
            kind = "concave"
        else:
            kind = "mixed"
        out.append(Joint(i, j, L, mean, kind))
    return out


def _loops(mesh, faces):
    """Boundary loops of a face set: one for a disc, one more per hole in it."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    e = np.sort(mesh.faces[faces][:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    uniq, cnt = np.unique(e, axis=0, return_counts=True)
    border = uniq[cnt == 1]
    if not len(border):
        return 0
    nodes, inv = np.unique(border, return_inverse=True)
    inv = inv.reshape(-1, 2)
    g = coo_matrix((np.ones(len(inv)), (inv[:, 0], inv[:, 1])),
                   shape=(len(nodes), len(nodes)))
    return int(connected_components(g, directed=False)[0])


def _exactness_note(seg, pt):
    """Say so when the mesh does not behave like a B-rep tessellation.

    On a scan or a remeshed part no two facets are coplanar and no vertex lies
    on anything analytic, so every patch is one triangle and every surface is a
    guess. The segmentation still runs; this is what stops it being read as a
    measurement.

    The test is how much of the part is EXACTLY flat (two or more triangles on one
    plane) or exactly round -- not how much is freeform. A loft is an exact
    tessellation of a face that happens not to be analytic: its caps are flat
    to float precision, and calling the whole mesh a scan for its sides would
    be wrong about everything else on it.
    """
    exact = sum(s.area for s in seg.surfaces
                if s.kind in ROUND_KINDS or (s.kind == "plane" and len(s.faces) >= 2))
    frac = exact / (seg.area or 1.0)
    if frac < 0.05:
        seg.notes.append(
            f"only {frac:.1%} of the area is exactly flat or exactly round: this "
            f"does not look like an exact CAD tessellation (a scan, or a remeshed "
            f"part), so the surfaces below are approximate -- prefer `section` "
            f"and `spectrum` here")


# ------------------------------------------------------------------ cache

_MEMO = {}
# id(turned mesh) -> (turned mesh, target mesh, R): targets turned into a frame
# by mesh2cad.frames, whose segmentation is the target's, turned.
_FRAMED = {}


def frame_source(turned, world, R):
    """Say that `turned` is `world` rotated by R, so segment_cached turns
    world's segmentation instead of searching `turned` again."""
    _FRAMED[id(turned)] = (turned, world, np.asarray(R, float), {})


def segment_cached(mesh, cache_dir=None, tol=None, budget=None):
    """`segment`, memoised on the mesh bytes, the tolerance and ALGO_VERSION.

    Every m2c call re-ingests its mesh; segmenting it again each time would make
    the cheapest question the slowest one. A cache that cannot be read or
    written is skipped, never trusted and never fatal.

    Memoised in-process too: one `sketch` asks for the level table more than
    once. A result cut short by a budget serves any caller that also has one,
    and is recomputed for a caller that has none.

    A target turned into a frame (`frame_source`) is never segmented itself:
    its target's segmentation, asked for with the same tolerance and budget,
    is turned with it, so every frame names the same F# and C#.
    """
    src = _FRAMED.get(id(mesh))
    if src is not None and src[0] is mesh:
        from mesh2cad.frames import turn_segmentation
        _turned, world, R, done = src
        base = segment_cached(world, cache_dir=cache_dir, tol=tol, budget=budget)
        if id(base) not in done:
            done[id(base)] = (base, turn_segmentation(base, R, mesh))
        return done[id(base)][1]
    h = hashlib.sha1()
    h.update(f"v{ALGO_VERSION};tol={tol!r};{KEEP_DOMINANT_PLANES},{FLOOR_GROWTH_MARGIN},"
             f"{FLOOR_MARGIN},{FIND_REVOLVES},{REVOLVE_MIN_TURN},{REVOLVE_STRAY},"
             f"{REVOLVE_RING_EDGE};".encode())
    h.update(np.ascontiguousarray(mesh.vertices, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(mesh.faces, dtype=np.int64).tobytes())
    key = h.hexdigest()[:20]

    def usable(seg):
        if seg is None:
            return False
        if not seg.partial:
            return True
        # A cut-short result serves only a caller allowed no more time than it
        # had; older cached results without a recorded budget count as 0.
        return budget is not None and (getattr(seg, "budget", None) or 0.0) >= budget

    hit = _MEMO.get(key)
    if usable(hit):
        return hit
    from mesh2cad.budget import stage
    stage(f"finding the part's faces (segmenting {len(mesh.faces)} triangles)")
    if cache_dir is None:
        seg = segment(mesh, tol=tol, budget=budget)
        _MEMO[key] = seg
        return seg
    path = Path(cache_dir) / f"segment_{key}.pkl"
    try:
        with open(path, "rb") as fh:
            seg = pickle.load(fh)
        if usable(seg):
            _MEMO[key] = seg
            return seg
    except Exception:                                              # noqa: BLE001
        pass
    seg = segment(mesh, tol=tol, budget=budget)
    _MEMO[key] = seg
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(seg, fh)
        tmp.replace(path)
    except Exception:                                              # noqa: BLE001
        pass
    return seg


# -------------------------------------------------------------- directions

def directions(seg, mesh):
    """Directions the surfaces carry, by area: plane normals and round axes.

    An extrusion leaves caps whose normal is its direction and round walls whose
    axis is; a revolve leaves axes. So this is the census of directions the part
    was built along, and one that is not X, Y or Z is an operation made at an
    angle. Clustered within what the fits can resolve: a direction measured on a
    face of extent L to a tolerance tol is known to about tol/L radians.
    """
    items = []
    for k, s in enumerate(seg.surfaces):
        if s.kind == "plane":
            d = s.params["normal"]
        elif s.kind in ("cylinder", "cone", "torus", "revolve"):
            d = s.params["axis"]
        else:
            continue
        lo, hi = s.extent["bbox"]
        L = float(np.linalg.norm(np.asarray(hi) - np.asarray(lo))) or seg.diag
        # Known to about (how far its vertices sit off it) / (its extent): a
        # plane to float precision, a torus fitted to a spline face far worse.
        items.append((canonical_dir(d), s.area, k,
                      max(max(s.residual, seg.tol) / L, 1e-9)))
    items.sort(key=lambda x: -x[1])
    groups = []
    for d, area, k, sig in items:
        for g in groups:
            ang = np.arccos(min(1.0, abs(float(d @ g["dir"]))))
            if ang <= 3.0 * max(sig, g["sigma"]) + 1e-9:
                g["area"] += area
                g["members"].append(k)
                break
        else:
            groups.append({"dir": d, "area": area, "members": [k], "sigma": sig})
    for g in groups:
        g["frac"] = g["area"] / (seg.area or 1.0)
        k = int(np.argmax(np.abs(g["dir"])))
        off = float(np.arccos(min(1.0, abs(g["dir"][k]))))
        g["principal"] = "XYZ"[k] if off <= 3.0 * g["sigma"] + 1e-9 else None
        g["off_axis_deg"] = float(np.degrees(off))
    groups.sort(key=lambda g: -g["area"])
    return groups


def principal_of(d, tol_rad=1e-6):
    """'X', 'Y' or 'Z' when a direction is one of them to within tol_rad."""
    d = np.asarray(d, float)
    k = int(np.argmax(np.abs(d)))
    ang = np.arccos(min(1.0, abs(d[k]) / np.linalg.norm(d)))
    return "XYZ"[k] if ang <= tol_rad else None


def equal_groups(seg):
    """Round surfaces repeated at the same size along parallel axes."""
    def key(s):
        # The turn a face covers belongs in the key: a full bore and a quarter
        # fillet of the same radius are not one feature repeated.
        p, sweep = s.params, s.extent.get("sweep", 360.0)
        if s.kind == "cylinder":
            return ("cylinder", p["radius"], sweep)
        if s.kind == "sphere":
            return ("sphere", p["radius"])
        if s.kind == "torus":
            return ("torus", p["major"], p["minor"], sweep)
        if s.kind == "cone":
            return ("cone", p.get("half_angle", 0.0),
                    s.extent.get("radius", [0, 0])[0],
                    s.extent.get("radius", [0, 0])[1], sweep)
        return None

    groups = []
    for k, s in enumerate(seg.surfaces):
        kk = key(s)
        if kk is None:
            continue
        for g in groups:
            ref = seg.surfaces[g[0]]
            rk = key(ref)
            if rk[0] != kk[0] or s.sense != ref.sense:
                continue
            if len(kk) != len(rk) or any(
                    abs(x - y) > 4 * seg.tol + 1e-6 * abs(y)
                    + (1e-3 if i == len(kk) - 2 else 0)
                    for i, (x, y) in enumerate(zip(kk[1:], rk[1:], strict=True))):
                continue
            if "axis" in s.params and abs(s.params["axis"] @ ref.params["axis"]) < 1 - 1e-6:
                continue
            g.append(k)
            break
        else:
            groups.append([k])
    return [g for g in groups if len(g) > 1]


def coaxial_groups(seg, min_size=2):
    """Round surfaces sharing one axis line: a revolve profile's worth of faces.

    Groups of fewer than `min_size` surfaces are dropped; mesh2cad.holes asks
    for 1, since a plain drilled hole is one cylinder."""
    axial = [k for k, s in enumerate(seg.surfaces)
             if s.kind in ("cylinder", "cone", "torus", "revolve")]
    groups = []
    for k in axial:
        s = seg.surfaces[k]
        for g in groups:
            r = seg.surfaces[g[0]]
            a, b = s.params["axis"], r.params["axis"]
            if abs(a @ b) < 1 - 1e-6:
                continue
            w = s.params["point"] - r.params["point"]
            off = np.linalg.norm(w - (w @ b) * b)
            if off <= 8 * seg.tol + 1e-7 * seg.diag:
                g.append(k)
                break
        else:
            groups.append([k])
    # Spheres sit on any axis through their centre; they join a group whose axis
    # passes through it.
    for k, s in enumerate(seg.surfaces):
        if s.kind != "sphere":
            continue
        for g in groups:
            r = seg.surfaces[g[0]]
            w = s.params["centre"] - r.params["point"]
            b = r.params["axis"]
            if np.linalg.norm(w - (w @ b) * b) <= 8 * seg.tol + 1e-7 * seg.diag:
                g.append(k)
                break
    return [g for g in groups if len(g) >= min_size]
