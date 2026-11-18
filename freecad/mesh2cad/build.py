"""Execute a FreeCAD program and tessellate it.

Shared because every caller needs the same thing -- the CLI's `--built`, the
scorer, the residual instrument, export -- with the same tolerances. A tolerance that differs between the tool
that checks a reconstruction and the tool that scores it is a way to disagree
with yourself about whether a part is finished.

The program runs in FreeCAD (mesh2cad.fc.runtime). The features' own tool solids,
read back from the feature tree, are meshed one by one and combined by
manifold3d, because that closes parts the kernel's mesher will not.
"""

import sys

from mesh2cad.fc.mesh import ANGULAR_TOL, STL_TOL, drop_degenerate  # noqa: F401
from mesh2cad.fc.mesh import tessellate as _mesh_shape

_drop_degenerate = drop_degenerate      # the name tests and older callers use


def load_program(path, tol=STL_TOL):
    """Run a FreeCAD program and return its tessellation as a trimesh.

    Prefers combining the FEATURES as meshes over meshing the fused solid.
    OCC will happily union a spoke into a cylindrical hub and then fail to mesh
    the result -- four unstitched faces per spoke, at every tolerance, because
    it cannot make a planar face's triangulation agree with a cylindrical one
    along their shared trim curve. Each feature meshes watertight on its own,
    so mesh them separately and let manifold3d do the boolean.

    Falls back to the fused part when per-feature meshing fails, and says so.
    """
    from mesh2cad.fc.runtime import ProgramError, run_program

    b = run_program(path)
    if b.failed:
        raise ProgramError("FreeCAD refused " + "; ".join(f.line() for f in b.failed))
    if b.steps:
        try:
            return _mesh_steps(b.steps, tol)
        except Exception as exc:                                   # noqa: BLE001
            # Say it. A quiet fall-back to the fused solid would put the broken
            # mesh back without anything on screen changing, and the next thing
            # to look wrong would be somewhere else entirely.
            print(f"note: per-feature meshing failed ({type(exc).__name__}: "
                  f"{exc}); falling back to meshing the fused solid, which may "
                  f"not be watertight", file=sys.stderr)
    return _tessellate(b.shape, tol)


def _mesh_steps(steps, tol):
    """Apply the recorded add/cut operands as meshes, grouped by operation.

    Consecutive operations go into ONE boolean call. Folding them in pairwise
    instead -- union(a,b), then union(that,c) -- reintroduces exactly the defect
    this route exists to avoid: features that each mesh watertight can come
    back open after sequential unions, and closed when handed to manifold3d
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
    # and turns edges non-manifold.
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
    into one union, as the single-body route does, would fuse bodies the
    program keeps apart."""
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

    They are where two operands' faces nearly coincide, not material: B-spline
    pieces of a hole cutter (frames.bspline_code) that end on each other's caps
    can leave tiny shells beside the part when cut out of the body, which
    `--check` would report as extra pieces. Below 1e-9 of the part's volume nothing is real.
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


def _tessellate(shape, tol=STL_TOL):
    """One FreeCAD shape as a trimesh.

    A shape holding several solids that overlap -- a hole cutter's B-spline
    pieces, gathered and never fused -- meshes as their union: meshing the
    compound whole would keep the overlaps twice."""
    import trimesh

    from mesh2cad.fc.mesh import real_solids
    solids = real_solids(shape)
    if len(solids) > 1:
        return trimesh.boolean.union([_mesh_shape(s, tol) for s in solids])
    return _mesh_shape(solids[0] if solids else shape, tol)

