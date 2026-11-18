"""Pieces, through-holes and sealed cavities of a build, read off its B-rep.

Topology is scored on these three counts of the delivered solid, and
per-feature meshing alone would say "1 body" whatever the B-rep held: several
solids, extra through-holes, cross-holes skinned into cavities, or an open
shell would all pass as "solid OK".

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


def _wrapped(shape):
    """The TopoDS_Shape of a Workplane, a cadquery shape or itself."""
    if hasattr(shape, "vals"):
        import cadquery as cq
        got = shape.vals()
        shape = got[0] if len(got) == 1 else cq.Compound.makeCompound(got)
    return getattr(shape, "wrapped", shape)


def _kids(shape, kind):
    from OCP.TopExp import TopExp_Explorer
    ex = TopExp_Explorer(shape, kind)
    while ex.More():
        yield ex.Current()
        ex.Next()


def _boundary(shape):
    """The edges of `shape`, as its wires hold them, that bound a face."""
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FORWARD, TopAbs_REVERSED
    from OCP.TopoDS import TopoDS
    for e in _kids(shape, TopAbs_EDGE):
        if e.Orientation() in (TopAbs_FORWARD, TopAbs_REVERSED):
            yield TopoDS.Edge(e)


def _meetings(face, index):
    """How many more passes a face's boundary makes through its own points than
    it has points, or None when its wires don't run end to start. Each boundary
    edge leaves one point of the face's (u, v) domain and arrives at another;
    one vertex's ends are one point unless a seam's period apart, and the two
    ends of a degenerate edge are one point."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve2d
    from OCP.BRepTools import BRepTools
    from OCP.TopAbs import TopAbs_REVERSED, TopAbs_WIRE
    from OCP.TopExp import TopExp

    u0, u1, v0, v1 = BRepTools.UVBounds_s(face)
    du, dv = MEET_SPAN * max(abs(u1 - u0), 1e-12), MEET_SPAN * max(abs(v1 - v0), 1e-12)
    ends, poles = [], []                 # (vertex, u, v, +1 leaves / -1 arrives / 0 pole)
    for wire in _kids(face, TopAbs_WIRE):
        for e in _boundary(wire):
            c = BRepAdaptor_Curve2d(e, face)
            t0, t1 = c.FirstParameter(), c.LastParameter()
            if e.Orientation() == TopAbs_REVERSED:
                t0, t1 = t1, t0
            pole = BRep_Tool.Degenerated_s(e)
            first = len(ends)
            for vx, t, way in ((TopExp.FirstVertex_s(e, True), t0, 1),
                               (TopExp.LastVertex_s(e, True), t1, -1)):
                p = c.Value(t)
                ends.append((index.FindIndex(vx), p.X(), p.Y(), 0 if pole else way))
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


def shell_euler(shell):
    """(chi, the mid-points of its open edges, faces whose wires don't close)."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_WIRE
    from OCP.TopExp import TopExp
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import (
        TopTools_IndexedDataMapOfShapeListOfShape,
        TopTools_IndexedMapOfShape,
    )

    vertices, edges = TopTools_IndexedMapOfShape(), TopTools_IndexedMapOfShape()
    faces = [TopoDS.Face(f) for f in _kids(shell, TopAbs_FACE)]
    seen = TopTools_IndexedMapOfShape()
    faces = [f for f in faces if seen.Add(f) and True]
    for f in faces:
        for e in _boundary(f):
            vertices.Add(TopExp.FirstVertex_s(e))
            vertices.Add(TopExp.LastVertex_s(e))
    extra_wires = meets = malformed = 0
    for f in faces:
        wires = 0
        for w in _kids(f, TopAbs_WIRE):
            bounds = False
            for e in _boundary(w):
                bounds = True
                if not BRep_Tool.Degenerated_s(e):
                    edges.Add(e)
            wires += bounds
        extra_wires += max(wires - 1, 0)
        m = _meetings(f, vertices)
        if m is None:
            malformed += 1
        else:
            meets += m
    sides = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shell, TopAbs_EDGE, TopAbs_FACE, sides)
    opened = []
    for i in range(1, edges.Extent() + 1):
        e = edges.FindKey(i)
        if not sides.Contains(e) or sides.FindFromKey(e).Size() != 2:
            c = BRepAdaptor_Curve(TopoDS.Edge(e))
            p = c.Value(0.5 * (c.FirstParameter() + c.LastParameter()))
            opened.append((p.X(), p.Y(), p.Z()))
    chi = vertices.Extent() - edges.Extent() + len(faces) - extra_wires + meets
    return chi, opened, malformed


def _box(shape, tight=False):
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    b = Bnd_Box()
    if tight:
        BRepBndLib.AddOptimal_s(shape, b, False, False)
    else:
        BRepBndLib.Add_s(shape, b, False)
    return b


def _volume(shape):
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, g)
    return g.Mass()


def _touch(a, b, tol):
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    d = BRepExtrema_DistShapeShape(a, b)
    return d.IsDone() and d.Value() <= tol


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
    x0, y0, z0, x1, y1, z1 = box.Get()
    return (x0, y0, z0), (x1, y1, z1)


def read(shape, quick=False):
    """The `Topology` of a cadquery shape or a TopoDS_Shape: its solids, each
    shell's genus and the sealed cavities. `quick` skips the distances, the slow
    part on threaded parts: solids are pieces and inner shells cavities, enough
    to say which feature changed a count."""
    from OCP.BRepClass3d import BRepClass3d
    from OCP.TopAbs import TopAbs_SHELL, TopAbs_SOLID
    from OCP.TopoDS import TopoDS

    wrapped = _wrapped(shape)
    solids = [TopoDS.Solid(s) for s in _kids(wrapped, TopAbs_SOLID)]
    solids = [s for s in solids if _volume(s) > 0]
    if not solids:
        raise ValueError("no solid to read")
    lo, hi = _ends(_box(wrapped))
    span = sum((b - a) ** 2 for a, b in zip(lo, hi, strict=True)) ** 0.5
    tol = min(TOUCH_MAX, max(1e-6, 1e-7 * span))

    out = Topology(0, 0, 0)
    for sol in solids:
        outer = BRepClass3d.OuterShell_s(sol)
        inner, holes = [], 0
        for sh in _kids(sol, TopAbs_SHELL):
            sh = TopoDS.Shell(sh)
            chi, opened, malformed = shell_euler(sh)
            genus = (2 - chi) // 2
            out.open_edges.extend(opened)
            out.malformed += malformed
            if sh.IsSame(outer):
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
        b.Enlarge(tol)
        boxes.append(b)
    index = list(range(len(solids)))
    out.pieces = len(solids) if quick else len(_classes(
        index, lambda i, j: not boxes[i].IsOut(boxes[j]) and _touch(solids[i], solids[j], tol)))
    out.holes = sum(s.holes for s in out.solids)
    out.cavities = len(out.voids)
    return out


def _length(edge):
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    g = GProp_GProps()
    BRepGProp.LinearProperties_s(edge, g)
    return g.Mass()


def rims(shell):
    """Where a shell's through-holes are: up to its genus of the shortest closed
    runs of edges that don't separate it and share no vertex. Each runs around
    one hole or handle, so its middle says where that hole is. [(length, middle,
    half its span, points along it)], shortest first; fewer than the genus when
    the shell has more than RIM_EDGES edges or the runs don't reach them all."""
    import heapq

    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX
    from OCP.TopExp import TopExp
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import (
        TopTools_IndexedDataMapOfShapeListOfShape,
        TopTools_IndexedMapOfShape,
    )

    genus = (2 - shell_euler(shell)[0]) // 2
    faces, vertices = TopTools_IndexedMapOfShape(), TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shell, TopAbs_FACE, faces)
    TopExp.MapShapes_s(shell, TopAbs_VERTEX, vertices)
    sides = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shell, TopAbs_EDGE, TopAbs_FACE, sides)
    links, ends, lengths, shapes, near = {}, {}, {}, {}, {}
    for i in range(1, sides.Extent() + 1):
        e = TopoDS.Edge(sides.FindKey(i))
        if BRep_Tool.Degenerated_s(e):
            continue
        links[i] = sorted({faces.FindIndex(f) - 1 for f in sides.FindFromIndex(i)})
        a = vertices.FindIndex(TopExp.FirstVertex_s(e))
        b = vertices.FindIndex(TopExp.LastVertex_s(e))
        ends[i], lengths[i], shapes[i] = (a, b), _length(e), e
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
        root = list(range(faces.Extent()))

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
            c = BRepAdaptor_Curve(shapes[k])
            for t in (0.0, 0.25, 0.5, 0.75):
                p = c.Value(c.FirstParameter() + t * (c.LastParameter() - c.FirstParameter()))
                pts.append((p.X(), p.Y(), p.Z()))
        low, high = np.min(pts, axis=0), np.max(pts, axis=0)
        taken.append((length, tuple((low + high) / 2), tuple((high - low) / 2), pts))
    return taken


def outer_rims(shape):
    """`rims` of the outer shell of each solid of `shape`."""
    from OCP.BRepClass3d import BRepClass3d
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopoDS import TopoDS
    wrapped = _wrapped(shape)
    out = []
    for s in _kids(wrapped, TopAbs_SOLID):
        out.extend(rims(BRepClass3d.OuterShell_s(TopoDS.Solid(s))))
    return out


def hits(shape, origin, along):
    """Where the ray from `origin` along the unit vector `along` crosses the
    faces of `shape`: its distances, in order."""
    from OCP.BRepIntCurveSurface import BRepIntCurveSurface_Inter
    from OCP.gp import gp_Dir, gp_Lin, gp_Pnt
    ray = BRepIntCurveSurface_Inter()
    ray.Init(_wrapped(shape), gp_Lin(gp_Pnt(*map(float, origin)), gp_Dir(*map(float, along))), 1e-7)
    got = []
    while ray.More():
        if ray.W() > 1e-9:
            got.append(float(ray.W()))
        ray.Next()
    got.sort()
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
