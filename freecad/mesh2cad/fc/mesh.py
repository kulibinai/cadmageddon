"""Tessellate a FreeCAD shape the way the scorer does.

`MeshPart.meshFromShape` is OCC's BRepMesh_IncrementalMesh, called with two
numbers (0.01, 0.1 rad) and RELATIVE linear deflection: each edge's deflection
is 0.01 of that edge's size, not 0.01 mm. The meshing tolerance decides what
"watertight" and "scores" mean, so it must stay fixed.
"""

import numpy as np

# The STL tolerance everything in the harness meshes at, so "looks right here"
# and "scores here" cannot diverge for a reason nobody can see.
STL_TOL = 0.01
ANGULAR_TOL = 0.1
RELATIVE = True         # deflection relative to edge size; see the module docstring


def real_solids(shape):
    """The shape's solids without the degenerate ones: a solid enclosing no
    volume (below 1e-9 of the largest) is two coincident faces, not material.

    A profile whose outline runs back along itself -- a bore slot cut to the
    full depth of the wall, a zero-width bridge between two regions --
    revolves into its real rings and a third "solid" of ~1e-12 mm^3. Meshed
    apart (as a compound of several solids is, see build._tessellate) it is
    not a volume, the union refuses it, and `--check` would say BAD
    TESSELLATION for a sound part."""
    solids = list(getattr(shape, "Solids", []) or [])
    if not solids:
        return []
    top = max(abs(float(x.Volume)) for x in solids)
    return [x for x in solids if abs(float(x.Volume)) > 1e-9 * top]


def without_degenerate(shape):
    """`shape`, or a compound of its real solids when it holds degenerate ones."""
    solids = list(getattr(shape, "Solids", []) or [])
    keep = real_solids(shape)
    if len(keep) == len(solids) or not keep:
        return shape
    import Part
    return keep[0] if len(keep) == 1 else Part.makeCompound(keep)


def tessellate(shape, tol=STL_TOL, angular=ANGULAR_TOL):
    """A Part.Shape as a trimesh, vertices merged, degenerate triangles at
    revolve poles dropped (see drop_degenerate)."""
    import MeshPart
    import trimesh

    from mesh2cad.fc import freecad
    freecad()
    if shape is None or shape.isNull():
        raise ValueError("nothing to mesh: the shape is empty")
    shape = without_degenerate(shape)
    m = MeshPart.meshFromShape(Shape=shape, LinearDeflection=float(tol),
                               AngularDeflection=float(angular), Relative=RELATIVE)
    pts, tri = m.Topology
    if not tri:
        raise ValueError("the shape meshed to no triangles")
    out = trimesh.Trimesh(np.array([(p.x, p.y, p.z) for p in pts], dtype=float),
                          np.array(tri, dtype=np.int64), process=False)
    out.merge_vertices()
    drop_degenerate(out)
    return out


def drop_degenerate(m):
    """Remove zero-area triangles left at a revolve pole.

    The mesher emits a pole once per surface patch and the copies differ only
    in the sign of a zero, so a triangle spanning them survives merging with
    two identical corners -- four unmatched edges per solid with a pole, which
    the scorer calls INVALID. Dropping them is a topology repair, not a
    geometric one, and it raises if the volume moves: a "repair" that moves the
    shape is a wrong answer wearing a fix's clothes.
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
