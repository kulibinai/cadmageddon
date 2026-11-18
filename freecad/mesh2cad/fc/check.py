"""Is what was built a solid? The kernel's half of `--check`, in the harness's
own wording.

The body-count half (bodies vs the target's, hairline gaps, fused parts) works
on meshes and stays in scripts/m2c.py; this is what needs the kernel itself:
the BRep's validity, the two failures that look identical from a mesh -- a
shape the kernel rejects, and a valid shape whose tessellation is open -- and
whether the part FreeCAD holds is the part the features' meshes make.
"""

from mesh2cad.fc.mesh import STL_TOL, tessellate


def validity(shape):
    """(valid, why): `why` is FreeCAD's own complaint when the BRep is not valid."""
    try:
        if shape is None or shape.isNull():
            return False, "the shape is empty"
        if shape.isValid():
            return True, ""
        try:
            shape.check(True)
        except Exception as exc:                                   # noqa: BLE001
            return False, str(exc).strip().splitlines()[0][:200]
        return False, "BRepCheck rejects it"
    except Exception as exc:                                       # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def bbox(shape):
    """The shape's TIGHT bounding box.

    `Shape.BoundBox` is OCC's quick box: it encloses the control net of every
    B-spline, torus and revolved face, so it is loose, which `--check` would
    report as material OUTSIDE THE TARGET. `optimalBoundingBox` is the tight
    one, computed on the exact geometry."""
    try:
        return shape.optimalBoundingBox(False, False)
    except Exception:                                              # noqa: BLE001
        return shape.BoundBox


def summary(shape):
    """The `check: builds OK` numbers: volume and bounding box."""
    bb = bbox(shape)
    return {"volume": float(shape.Volume), "solids": len(shape.Solids),
            "bbox": (bb.XMin, bb.YMin, bb.ZMin, bb.XMax, bb.YMax, bb.ZMax)}


def fused_verdict(shape, tol=STL_TOL):
    """Mesh the fused part and say whether it is a solid -- the fallback when
    per-feature meshing could not run."""
    try:
        m = tessellate(shape, tol)
    except Exception as exc:                                       # noqa: BLE001
        return f"tessellation FAILED  {type(exc).__name__}: {exc}"
    bodies = len(m.split())
    if m.is_watertight:
        return f"solid OK   watertight, {bodies} {'body' if bodies == 1 else 'bodies'}"
    import trimesh
    broken = len(trimesh.repair.broken_faces(m))
    valid, why = validity(shape)
    if not valid:
        return (f"NOT A VALID SOLID   the kernel rejects the shape itself"
                f"{' (' + why + ')' if why else ''}: {broken} broken faces, "
                f"{bodies} closed bodies. Usual cause: two features share a face "
                f"fitted separately at two different sections, so their union "
                f"leaves a sliver -- measure both from the same section, or build "
                f"the shared face once.")
    return (f"VALID SOLID, BAD TESSELLATION   the BRep is valid ({bodies} closed "
            f"bodies) but meshing at tol={tol} leaves {broken} broken faces, and "
            f"that is what scoring meshes. IoU and GMS will come back undefined. "
            f"Pole degeneracies are already repaired, so this is a real sliver: "
            f"look for two faces that nearly coincide -- usually a feature joined "
            f"to another exactly on a face rather than overlapping it. Meshing "
            f"finer will NOT fix it; a broken-face count that does not move with "
            f"tolerance never does.")


# How far the part FreeCAD holds may be from the part its features' meshes make,
# as a fraction of the meshes' volume. Meshing at STL_TOL usually puts the two
# well under 1% apart; a broken part is far more off. But the exact volume of a
# sound part with large curved faces can sit nearly 1% off its meshes by the
# meshing alone. So a gap past it is measured again like with like before it
# counts (brep_mismatch).
BREP_VS_MESH = 0.01


def _feature_ref(f):
    """`F01_Pad (feature #1, Pad)`: the tree name, and the number `--replace` takes."""
    what = f.type.split("::")[-1]
    num = f.name[1:].split("_", 1)[0] if f.name[:1] == "F" else ""
    return f"{f.name} (feature #{int(num)}, {what})" if num.isdigit() else f"{f.name} ({what})"


def _close(a, b, rel):
    return abs(a - b) <= rel * max(abs(b), 1e-9)


def _meshed_whole(shape, tol):
    """The part's volume meshed as one shape, the way the score meshes, or
    None when it does not mesh."""
    try:
        return float(tessellate(shape, tol).volume)
    except Exception:                                              # noqa: BLE001
        return None


def brep_mismatch(build, mesh_volume, rel=BREP_VS_MESH, tol=STL_TOL):
    """The verdict when the part FreeCAD holds is not the part its features
    make, or None.

    Everything the agent is shown -- `--check`'s body count, `residual`, the
    STL -- combines the features' own solids as meshes; the .FCStd and the
    scorer are FreeCAD's part. When FreeCAD builds a feature as an invalid
    solid and builds on it, the two part ways and nothing on screen says so.
    Worded to start like
    the kernel's other rejection, so a version it fires on is never the one
    delivered (scripts/pick_best.py). `build` must come from
    `run_program(..., validate=True)` for the cause to be named.

    The exact volume of a sound part is off its meshes by the meshing alone,
    and with large curved faces that is most of `rel`. So a gap past `rel`
    is measured again like with like before it counts: the part meshed whole
    at `tol`, as the score meshes it, against the features meshed apart. A
    broken feature stays as far off meshed. A part
    with a feature FreeCAD built invalid keeps the exact comparison: its
    volume is the thing in doubt, and its mesh may be open."""
    vb, vm = float(build.shape.Volume), float(mesh_volume)
    if _close(vb, vm, rel):
        return None
    whole = None
    if not build.invalid:
        whole = _meshed_whole(build.shape, tol)
        if whole is not None and _close(whole, vm, rel):
            return None
    ref = whole if whole is not None else vb
    meshed = f" ({whole:.1f} meshed whole)" if whole is not None else ""
    bad = build.invalid
    if bad:
        more = f" ({len(bad) - 1} more after it)" if len(bad) > 1 else ""
        cause = (f"FreeCAD built {_feature_ref(bad[0])} as an invalid solid without "
                 f"marking it failed{more}, and every later feature in its Body is "
                 f"built on it. Rebuild that feature (`--replace N`): usually its "
                 f"profile crosses or touches itself, or its faces lie exactly on "
                 f"faces of the solid it joins.")
    else:
        cause = ("No feature is an invalid solid on its own, so FreeCAD combines "
                 "them differently than the meshes do: look at the feature that "
                 "changed the volume last.")
    return (f"NOT A VALID SOLID   the part FreeCAD holds is not the part its "
            f"features make: {vb:.1f} mm^3{meshed} against {vm:.1f} mm^3 from the "
            f"features' own meshes ({100 * abs(ref - vm) / max(abs(vm), 1e-9):.0f}% off). The "
            f".FCStd and the score are FreeCAD's part; the body count here, "
            f"`residual` and the STL are the features', which is why they look "
            f"right. {cause}")


def invalid_note(build):
    """A warning when a feature, or the part itself, is not a valid solid but
    the part still has its features' volume (else `brep_mismatch` says it), or
    None. Not a rejection: the score is not affected yet."""
    bad = build.invalid
    valid, why = validity(build.shape)
    if valid and not bad:
        return None
    said = []
    if bad:
        more = f" and {len(bad) - 1} more" if len(bad) > 1 else ""
        said.append(f"FreeCAD built {_feature_ref(bad[0])}{more} as an invalid "
                    f"solid without marking it failed")
    if not valid:
        said.append(f"the kernel rejects the part itself{' (' + why + ')' if why else ''}")
    return ("BREP WARNING   " + "; ".join(said) + ". The part still has the volume "
            "its features make, so the score is not affected yet, but every later "
            "feature in that Body is built on it: if the next change comes out "
            "wrong, rebuild this one first.")


def feature_failures(build):
    """One line per feature FreeCAD refused, in tree order -- what to say
    first, because every later feature in that Body is built on it."""
    return [f.line() for f in build.failed]


def block_program(code, mode="add", name="F00"):
    """A whole program around one feature block's body: the prelude, a fresh
    document, the block, `_finish`. What a block is checked and tested in."""
    from mesh2cad.fc import prelude_source
    body = "\n".join(code.strip().splitlines())
    return ("# -*- coding: utf-8 -*-\n" + prelude_source()
            + '\ndoc, body = _start("check")\n'
            + f'_b = _block(doc, "{name}", "{mode}")\n{body}\n_end(_b)\n'
            + "\nresult, _STEPS = _finish(doc)\n")


def block_problem(code, kinds=("Loft",), tol=STL_TOL):
    """Why a feature block would not build as a scoreable solid, or None.

    The block is built on its own, and every feature whose type contains one
    of `kinds` must be accepted by FreeCAD, valid, of positive volume, and
    mesh closed on its own -- the loft precheck: a loft the kernel refuses
    says only `BRep_API: command not done`, and a smooth loft through many
    short edges can come back a valid BRep whose mesh is open."""
    from mesh2cad.fc.runtime import ProgramError, run_program
    try:
        b = run_program("<block>", src=block_program(code), keep=True)
    except ProgramError as exc:
        return str(exc)
    try:
        for f in b.features:
            if not any(k in f.type for k in kinds):
                continue
            if f.failed:
                return f.line()
            s = b.doc.getObject(f.name).AddSubShape
            if s.isNull() or not s.isValid():
                return f"{f.name} builds but is not a valid solid"
            if s.Volume <= 0:
                return f"{f.name} builds inside out (volume {s.Volume:.4g})"
            if not tessellate(s, tol).is_watertight:
                return f"{f.name} builds but does not mesh closed at tol={tol}"
        return None
    finally:
        import FreeCAD
        FreeCAD.closeDocument(b.doc.Name)
