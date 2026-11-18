"""Pieces, through-holes and sealed cavities of a build, read off its B-rep.

The benchmark scores topology on these three counts of the delivered STEP, and
per-feature meshing cannot see them: extra solids, extra through-holes,
cross-holes skinned into cavities and open shells all pass as "solid OK".

The rule:
- a shell's genus is (2 - chi) / 2, chi = V - E + F - R + meetings, over the
  edges and vertices that bound its faces: a seam counts once, a degenerate edge
  (a pole) is a point, an internal edge or vertex is no boundary. R counts a
  face's wires beyond the first; a meeting is a point where a face's boundary
  passes through itself;
- solids that touch or overlap are one piece;
- through-holes are the genus of each solid's outer shell and of its cavities,
  solid by solid;
- inner shells that touch are one cavity, and one that touches the outer shell
  is open to it, so no cavity;
- a shell with open edges, or a face whose wires don't run end to start, has no
  genus to read; its counts are kept, with `unreadable` saying why.

The target has no B-rep here, so `mesh_counts` reads it off its mesh.

Read through FreeCAD's Part API (the B-rep is OCC's either way): a face's
wires are walked with `OrderedEdges`, which, unlike `Edges`, keeps both uses
of a seam with their orientation, and each use's (u, v) ends come from the
face's own p-curve of it (`Face.curveOnSurface`).
"""

from dataclasses import dataclass, field

import numpy as np

MEET_SPAN = 0.25      # of a face's (u, v) extent: a vertex's edge ends closer are one point
TOUCH_MAX = 1e-4      # mm: the most two shapes lie apart and still touch
RIM_EDGES = 3000      # edges in a shell: past it, `rims` doesn't look for its holes


@dataclass
class Cavity:
    volume: float
    centre: tuple
    low: tuple
    high: tuple
    holes: int = 0


@dataclass
class Solid:
    volume: float
    low: tuple
    high: tuple
    holes: int = 0


@dataclass
class Topology:
    pieces: int
    holes: int
    cavities: int
    solids: list = field(default_factory=list)
    voids: list = field(default_factory=list)
    inner: list = field(default_factory=list)        # inner shells touching the outer one
    open_edges: list = field(default_factory=list)   # (x, y, z) mid-points
    malformed: int = 0

    @property
    def unreadable(self):
        """Why a topology scorer would read the mesh instead, or None."""
        if self.open_edges:
            return f"a shell with {len(self.open_edges)} open edge(s)"
        if self.malformed:
            return f"{self.malformed} face(s) whose wires do not run end to start"
        return None


def _shape(shape):
    """The Part.Shape of a shape or of a Build (mesh2cad.fc.runtime)."""
    return getattr(shape, "shape", shape)


class _Index:
    """Shapes numbered by identity (OCC's IsSame: the same TShape and location,
    whatever the orientation), 1 upwards, as TopTools_IndexedMapOfShape does."""

    def __init__(self):
        self._by_hash, self.keys = {}, []

    def find(self, s):
        for t, i in self._by_hash.get(s.hashCode(), ()):
            if t.isSame(s):
                return i
        return 0

    def add(self, s):
        i = self.find(s)
        if i:
            return i
        self.keys.append(s)
        i = len(self.keys)
        self._by_hash.setdefault(s.hashCode(), []).append((s, i))
        return i

    def __len__(self):
        return len(self.keys)


def _uses(wire):
    """Every use of an edge in a wire, with its orientation: a seam twice."""
    return list(wire.OrderedEdges)


def _boundary(wire):
    """The edges of `wire`, as it holds them, that bound a face."""
    return [e for e in _uses(wire) if e.Orientation in ("Forward", "Reversed")]


def _meetings(face, index):
    """How many more passes a face's boundary makes through its own points than
    it has points, or None when its wires don't run end to start. Each boundary
    edge leaves one point of the face's (u, v) domain and arrives at another;
    one vertex's ends are one point unless a seam's period apart, and the two
    ends of a degenerate edge are one point."""
    u0, u1, v0, v1 = face.ParameterRange
    du, dv = MEET_SPAN * max(abs(u1 - u0), 1e-12), MEET_SPAN * max(abs(v1 - v0), 1e-12)
    ends, poles = [], []                 # (vertex, u, v, +1 leaves / -1 arrives / 0 pole)
    for wire in face.Wires:
        for e in _boundary(wire):
            got = face.curveOnSurface(e)
            if got is None:
                return None
            c, t0, t1 = got
            if e.Orientation == "Reversed":
                t0, t1 = t1, t0
            pole = bool(e.Degenerated)
            first = len(ends)
            for vx, t, way in ((e.firstVertex(True), t0, 1), (e.lastVertex(True), t1, -1)):
                p = c.value(t)
                ends.append((index.find(vx), p.x, p.y, 0 if pole else way))
            if pole:
                poles.append((first, first + 1))
    root = list(range(len(ends)))

    def find(i):
        while root[i] != i:
            root[i] = root[root[i]]
            i = root[i]
        return i
    by_vertex = {}
    for i, end in enumerate(ends):
        by_vertex.setdefault(end[0], []).append(i)
    for same in by_vertex.values():
        for n, i in enumerate(same):
            for j in same[:n]:
                if abs(ends[i][1] - ends[j][1]) <= du and abs(ends[i][2] - ends[j][2]) <= dv:
                    root[find(i)] = find(j)
    for a, b in poles:
        root[find(a)] = find(b)
    passes = {}
    for i, end in enumerate(ends):
        if end[3]:
            ways = passes.setdefault(find(i), [0, 0])
            ways[end[3] < 0] += 1
    if any(a != b for a, b in passes.values()):
        return None
    return sum(max(0, a - 1) for a, _ in passes.values())


def _sides(shell):
    """(the shell's edges numbered, {edge number: how many face uses it has}):
    TopExp::MapShapesAndAncestors counts a face once per use, a seam's face
    twice."""
    edges, uses = _Index(), {}
    for f in shell.Faces:
        for w in f.Wires:
            for e in _uses(w):
                i = edges.add(e)
                uses[i] = uses.get(i, 0) + 1
    return edges, uses


def shell_euler(shell):
    """(chi, the mid-points of its open edges, faces whose wires don't close)."""
    vertices, edges = _Index(), _Index()
    faces = list(shell.Faces)
    for f in faces:
        for w in f.Wires:
            for e in _boundary(w):
                vertices.add(e.firstVertex())
                vertices.add(e.lastVertex())
    extra_wires = meets = malformed = 0
    for f in faces:
        wires = 0
        for w in f.Wires:
            bounds = False
            for e in _boundary(w):
                bounds = True
                if not e.Degenerated:
                    edges.add(e)
            wires += bounds
        extra_wires += max(wires - 1, 0)
        m = _meetings(f, vertices)
        if m is None:
            malformed += 1
        else:
            meets += m
    every, uses = _sides(shell)
    opened = []
    for e in edges.keys:
        i = every.find(e)
        if not i or uses.get(i, 0) != 2:
            p = e.valueAt(0.5 * (e.FirstParameter + e.LastParameter))
            opened.append((p.x, p.y, p.z))
    chi = len(vertices) - len(edges) + len(faces) - extra_wires + meets
    return chi, opened, malformed


def _box(shape, tight=False):
    """The shape's bounding box: OCC's quick one, or (tight) the optimal one."""
    return shape.optimalBoundingBox(False, False) if tight else shape.BoundBox


def _volume(shape):
    return float(shape.Volume)


def _touch(a, b, tol):
    try:
        return a.distToShape(b)[0] <= tol
    except Exception:                                              # noqa: BLE001
        return False


def _classes(items, same):
    """`items` grouped by the transitive closure of `same(a, b)`."""
    root = list(range(len(items)))

    def find(i):
        while root[i] != i:
            root[i] = root[root[i]]
            i = root[i]
        return i
    for i in range(len(items)):
        for j in range(i):
            if find(i) != find(j) and same(items[i], items[j]):
                root[find(i)] = find(j)
    groups = {}
    for i, it in enumerate(items):
        groups.setdefault(find(i), []).append(it)
    return list(groups.values())


def _ends(box):
    return (box.XMin, box.YMin, box.ZMin), (box.XMax, box.YMax, box.ZMax)


def read(shape, quick=False):
    """The `Topology` of a Part shape (or a Build): its solids, each shell's
    genus and the sealed cavities, by the benchmark's rule. `quick` skips the
    distances, the slow part (a minute or more on threads): solids are pieces and
    inner shells cavities, enough to say which feature changed a count."""
    shape = _shape(shape)
    solids = [s for s in shape.Solids if _volume(s) > 0]
    if not solids:
        raise ValueError("no solid to read")
    lo, hi = _ends(_box(shape))
    span = sum((b - a) ** 2 for a, b in zip(lo, hi, strict=True)) ** 0.5
    tol = min(TOUCH_MAX, max(1e-6, 1e-7 * span))

    out = Topology(0, 0, 0)
    for sol in solids:
        outer = sol.OuterShell
        inner, holes = [], 0
        for sh in sol.Shells:
            chi, opened, malformed = shell_euler(sh)
            genus = (2 - chi) // 2
            out.open_edges.extend(opened)
            out.malformed += malformed
            if sh.isSame(outer):
                holes += genus
            else:
                inner.append((sh, genus))
        groups = ([[x] for x in inner] if quick else
                  _classes(inner, lambda a, b: _touch(a[0], b[0], tol)))
        for group in groups:
            ends = [_ends(_box(sh, tight=True)) for sh, _ in group]
            low = tuple(min(e[0][i] for e in ends) for i in range(3))
            high = tuple(max(e[1][i] for e in ends) for i in range(3))
            cav = Cavity(sum(abs(_volume(sh)) for sh, _ in group),
                         tuple((a + b) / 2 for a, b in zip(low, high, strict=True)),
                         low, high, sum(g for _, g in group))
            if not quick and any(_touch(sh, outer, tol) for sh, _ in group):
                out.inner.append(cav)
                continue
            out.voids.append(cav)
            holes += cav.holes
        low, high = _ends(_box(sol, tight=True))
        out.solids.append(Solid(_volume(sol), low, high, holes))
    boxes = []
    for s in solids:
        b = _box(s, tight=True)
        b.enlarge(tol)
        boxes.append(b)
    index = list(range(len(solids)))
    out.pieces = len(solids) if quick else len(_classes(
        index, lambda i, j: boxes[i].intersect(boxes[j]) and _touch(solids[i], solids[j], tol)))
    out.holes = sum(s.holes for s in out.solids)
    out.cavities = len(out.voids)
    return out


def rims(shell):
    """Where a shell's through-holes are: up to its genus of the shortest closed
    runs of edges that don't separate it and share no vertex. Each runs around
    one hole or handle, so its middle says where that hole is. [(length, middle,
    half its span, points along it)], shortest first; fewer than the genus when
    the shell has more than RIM_EDGES edges or the runs don't reach them all."""
    import heapq

    genus = (2 - shell_euler(shell)[0]) // 2
    faces, vertices = list(shell.Faces), _Index()
    for v in shell.Vertexes:
        vertices.add(v)
    edges, face_of = _Index(), {}
    for fi, f in enumerate(faces):
        for w in f.Wires:
            for e in _uses(w):
                face_of.setdefault(edges.add(e), set()).add(fi)
    links, ends, lengths, shapes, near = {}, {}, {}, {}, {}
    for i, e in enumerate(edges.keys, start=1):
        if e.Degenerated:
            continue
        links[i] = sorted(face_of[i])
        a = vertices.find(e.firstVertex())
        b = vertices.find(e.lastVertex())
        ends[i], lengths[i], shapes[i] = (a, b), float(e.Length), e
        near.setdefault(a, []).append(i)
        near.setdefault(b, []).append(i)
    if not genus or len(links) > RIM_EDGES:
        return []

    def run_through(start):
        """The shortest closed run of edges through edge `start`."""
        a, b = ends[start]
        if a == b:
            return [start]
        best, back, todo = {a: 0.0}, {}, [(0.0, a)]
        while todo:
            d, v = heapq.heappop(todo)
            if v == b:
                break
            if d > best.get(v, np.inf):
                continue
            for e in near[v]:
                if e == start:
                    continue
                u = ends[e][1] if ends[e][0] == v else ends[e][0]
                if d + lengths[e] < best.get(u, np.inf):
                    best[u], back[u] = d + lengths[e], (v, e)
                    heapq.heappush(todo, (best[u], u))
        if b not in back:
            return None
        run, v = [start], b
        while v != a:
            v, e = back[v]
            run.append(e)
        return run

    def whole(cut):
        root = list(range(len(faces)))

        def find(i):
            while root[i] != i:
                root[i] = root[root[i]]
                i = root[i]
            return i
        for k, ids in links.items():
            if k not in cut:
                for f in ids[1:]:
                    root[find(ids[0])] = find(f)
        return len({find(i) for i in range(len(root))}) == 1

    runs = {}
    for e in links:
        run = run_through(e)
        if run:
            runs.setdefault(frozenset(run), sum(lengths[k] for k in run))
    taken, cut, used = [], set(), set()
    for run, length in sorted(runs.items(), key=lambda kv: kv[1]):
        if len(taken) >= genus:
            break
        here = {v for k in run for v in ends[k]}
        if here & used or not whole(cut | run):
            continue
        cut |= run
        used |= here
        pts = []
        for k in run:
            c = shapes[k]
            for t in (0.0, 0.25, 0.5, 0.75):
                p = c.valueAt(c.FirstParameter + t * (c.LastParameter - c.FirstParameter))
                pts.append((p.x, p.y, p.z))
        low, high = np.min(pts, axis=0), np.max(pts, axis=0)
        taken.append((length, tuple((low + high) / 2), tuple((high - low) / 2), pts))
    return taken


def outer_rims(shape):
    """`rims` of the outer shell of each solid of `shape`."""
    out = []
    for s in _shape(shape).Solids:
        out.extend(rims(s.OuterShell))
    return out


def hits(shape, origin, along):
    """Where the ray from `origin` along the unit vector `along` crosses the
    faces of `shape`: its distances, in order. The ray is a segment reaching
    past the shape's far side, sectioned with it."""
    from mesh2cad.fc import freecad
    FreeCAD = freecad()
    import Part
    shape = _shape(shape)
    o, d = np.asarray(origin, float), np.asarray(along, float)
    bb = shape.BoundBox
    corners = np.array([[x, y, z] for x in (bb.XMin, bb.XMax) for y in (bb.YMin, bb.YMax)
                        for z in (bb.ZMin, bb.ZMax)])
    reach = float(np.max(np.linalg.norm(corners - o, axis=1))) * 2.0 + 1.0
    ray = Part.LineSegment(FreeCAD.Vector(*o), FreeCAD.Vector(*(o + reach * d))).toShape()
    got = sorted(float((np.array([v.Point.x, v.Point.y, v.Point.z]) - o) @ d)
                 for v in shape.section(ray).Vertexes)
    got = [t for t in got if t > 1e-9]
    return [t for i, t in enumerate(got) if i == 0 or t - got[i - 1] > 1e-6]


def mesh_counts(mesh):
    """(pieces, through-holes, cavities) of a closed mesh, or None for an open
    one: a closed shell of negative volume is a cavity, and a shell's genus
    comes from its Euler number."""
    if not mesh.is_watertight:
        return None
    shells = mesh.split()
    if not len(shells):
        return None
    pieces = sum(1 for s in shells if s.volume > 0)
    cavities = len(shells) - pieces
    holes = sum(max(0, (2 - int(s.euler_number)) // 2) for s in shells)
    return pieces, holes, cavities
