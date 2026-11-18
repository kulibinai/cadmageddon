"""Execute a CadQuery program and tessellate it.

Shared because three callers need the same thing -- the CLI's `--built`, the
scorer, and now the residual instrument -- and each had grown its own copy with
slightly different tolerances. A tolerance that differs between the tool that
checks a reconstruction and the tool that scores it is a way to disagree with
yourself about whether a part is finished.
"""

import os
import sys
import tempfile
from pathlib import Path

# The STL tolerance everything in the harness meshes at, so "looks right here"
# and "scores here" cannot diverge for a reason nobody can see.
STL_TOL = 0.01
ANGULAR_TOL = 0.1


def load_program(path, tol=STL_TOL):
    """Run a CadQuery program and return its tessellation as a trimesh.

    Prefers combining the FEATURES as meshes over meshing the fused solid.
    OCC will happily union a spoke into a cylindrical hub and then fail to mesh
    the result -- unstitched faces at every tolerance, through both
    tessellators, because it cannot make a planar face's triangulation agree
    with a cylindrical one along their shared trim curve. Each feature meshes
    watertight on its own, so mesh them separately and let manifold3d do the
    boolean.

    Falls back to the fused result when a program predates `_STEPS`.
    """
    import cadquery as cq
    import trimesh

    ns = {"show_object": lambda o, *a, **k: None,
          "debug": lambda *a, **k: None}
    exec(compile(Path(path).read_text(), str(path), "exec"), ns)

    steps = ns.get("_STEPS")
    if steps:
        try:
            return _mesh_steps(steps, tol)
        except Exception as exc:                                   # noqa: BLE001
            # Say it. A quiet fall-back to the fused solid would put the broken
            # mesh back without anything on screen changing, and the next thing
            # to look wrong would be somewhere else entirely.
            print(f"note: per-feature meshing failed ({type(exc).__name__}: "
                  f"{exc}); falling back to meshing the fused solid, which may "
                  f"not be watertight", file=sys.stderr)
    obj = next((ns[n] for n in ("result", "r", "part", "shape")
                if hasattr(ns.get(n), "val") or hasattr(ns.get(n), "wrapped")), None)
    if obj is None:
        raise RuntimeError(f"no result in {path}")
    shape = obj.toCompound() if hasattr(obj, "toCompound") else (
        obj.val() if hasattr(obj, "val") else obj)

    f = tempfile.mktemp(suffix=".stl")
    try:
        cq.exporters.export(shape, f, tolerance=tol, angularTolerance=ANGULAR_TOL)
        m = trimesh.load(f, force="mesh")
    finally:
        if os.path.exists(f):
            os.unlink(f)
    m.merge_vertices()
    return m


def _mesh_steps(steps, tol):
    """Apply the recorded add/cut operands as meshes, grouped by operation.

    Consecutive operations go into ONE boolean call. Folding them in pairwise
    instead -- union(a,b), then union(that,c) -- reintroduces exactly the defect
    this route exists to avoid: seven features that each mesh watertight came
    back open after six sequential unions, and closed when handed to manifold3d
    together. Every intermediate result is another chance to make a sliver.
    """
    import trimesh

    if any(op == "body" for op, _ in steps):
        return _mesh_bodies(steps, tol)
    out, i = None, 0
    while i < len(steps):
        op = steps[i][0]
        group = []
        while i < len(steps) and steps[i][0] == op:
            m = _tessellate(steps[i][1], tol)
            if not m.is_watertight:
                raise RuntimeError(
                    f"feature {i} does not mesh watertight on its own, so "
                    f"combining meshes cannot help here")
            group.append(m)
            i += 1
        if op == "add":
            out = trimesh.boolean.union(group if out is None else [out] + group)
        elif out is None:
            raise RuntimeError(f"a {op} with nothing before it")
        elif op == "intersect":
            out = trimesh.boolean.intersection([out] + group)
        else:
            out = trimesh.boolean.difference([out] + group)
    if out is None:
        raise RuntimeError("no additive feature in _STEPS")
    # Deliberately NOT merge_vertices(). manifold3d returns a closed mesh;
    # merging its vertices afterwards fuses ones that only happen to coincide
    # and turns four edges non-manifold. The same seven features gave 0 broken
    # faces without this call and 4 with it.
    return _drop_empty_shells(out)


def _touch(a, b):
    """Whether two closed meshes share material or a face: their union is one
    piece. What OCC's fuse does with them, read off the meshes."""
    import trimesh
    if (a.bounds[0] > b.bounds[1] + 1e-6).any() or (b.bounds[0] > a.bounds[1] + 1e-6).any():
        return False
    return len(trimesh.boolean.union([a, b]).split()) == 1


def _mesh_bodies(steps, tol):
    """`_mesh_steps` for a program with bodies of its own (`--combine
    separate`): every body meshed apart and put side by side, as the
    program's compound holds them. An added feature fuses with the bodies it
    touches (or stands alone if it touches none), a cut or an intersection
    applies to each body, and a body step starts a new one. Folding the bodies
    into one union, as the single-body route does, would fuse separate parts
    that merely touch."""
    import trimesh

    bodies = []
    for i, (op, wp) in enumerate(steps):
        m = _tessellate(wp, tol)
        if not m.is_watertight:
            raise RuntimeError(f"feature {i} does not mesh watertight on its own, so "
                               f"combining meshes cannot help here")
        if op == "body" or not bodies:
            if op not in ("add", "body"):
                raise RuntimeError(f"a {op} with nothing before it")
            bodies.append(m)
        elif op == "add":
            hit = [b for b in bodies if _touch(b, m)]
            rest = [b for b in bodies if not any(b is h for h in hit)]
            bodies = rest + [trimesh.boolean.union(hit + [m]) if hit else m]
        else:
            fn = (trimesh.boolean.intersection if op == "intersect"
                  else trimesh.boolean.difference)
            kept = []
            for b in bodies:
                r = fn([b, m])
                if r is not None and len(r.faces):
                    kept.extend(p for p in r.split() if p.volume > 0)
            bodies = kept
    if not bodies:
        raise RuntimeError("nothing is left of the program's bodies")
    return _drop_empty_shells(trimesh.util.concatenate(bodies))


def _drop_empty_shells(m):
    """Leave out closed shells that enclose (next to) no volume.

    They are where two operands' faces nearly coincide, not material: the
    B-spline pieces of a hole cutter (frames.bspline_code) end on each other's
    caps, and cutting them out of the body leaves dozens of near-empty shells
    beside the part, which `--check` would report as a part in that many pieces.
    Below 1e-9 of the part's volume nothing is real.
    """
    parts = m.split(only_watertight=False)
    if len(parts) < 2:
        return m
    floor = 1e-9 * max(abs(float(p.volume)) for p in parts)
    keep = [p for p in parts if abs(float(p.volume)) > floor]
    if len(keep) == len(parts) or not keep:
        return m
    import trimesh
    return trimesh.util.concatenate(keep)


def _tessellate(wp, tol):
    import cadquery as cq
    import trimesh

    # A workplane holding several solids that overlap -- a hole cutter's
    # B-spline pieces, gathered and never fused (frames.bspline_code) --
    # meshes as their union. `.val()` alone took the first piece and dropped
    # the rest without a word.
    if hasattr(wp, "vals") and len(wp.vals()) > 1:
        return trimesh.boolean.union([_tessellate(v, tol) for v in wp.vals()])
    shape = wp.toCompound() if hasattr(wp, "toCompound") else (
        wp.val() if hasattr(wp, "val") else wp)
    f = tempfile.mktemp(suffix=".stl")
    try:
        cq.exporters.export(shape, f, tolerance=tol, angularTolerance=ANGULAR_TOL)
        m = trimesh.load(f, force="mesh")
    finally:
        if os.path.exists(f):
            os.unlink(f)
    m.merge_vertices()
    _drop_degenerate(m)
    return m


def _drop_degenerate(m):
    """Remove zero-area triangles left at a revolve pole.

    Every solid with a revolve pole came out of the exporter with exactly four
    unmatched edges, two per pole, and no tolerance fixed it: 0.01 down to
    0.0005 all gave four. The cause is one degenerate triangle at each apex --
    OCC emits the pole once per surface patch, the copies differ only in the
    sign of a zero, and the triangle spanning them survives merge_vertices with
    two identical corners. Four unmatched edges is enough for the scorer to
    call the part INVALID.

    It is not specific to anything this harness emits. `cq.Workplane.sphere`
    unioned into a cylinder does it too, which is why a capsule, a dome, a
    bullet nose and a ball could not be built as a single feature at all.

    Dropping those triangles is a topology repair, not a geometric one: the
    volume is unchanged to every digit it prints. If that ever stops being true
    this raises, because a "repair" that moves the shape is a wrong answer
    wearing a fix's clothes.
    """
    keep = m.nondegenerate_faces()
    if keep.all():
        return 0
    before = m.volume
    m.update_faces(keep)
    m.remove_unreferenced_vertices()
    moved = abs(m.volume - before)
    if moved > max(abs(before), 1.0) * 1e-9:
        raise RuntimeError(
            f"dropping {int((~keep).sum())} degenerate triangles changed the "
            f"volume by {moved:.6g} mm^3; that is not a tessellation artefact "
            f"and must not be repaired away")
    return int((~keep).sum())
