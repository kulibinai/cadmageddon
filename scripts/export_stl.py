#!/usr/bin/env python
"""Export an emitted program as the deliverable STL.

    python scripts/export_stl.py run/reconstruction.py part_name.stl

The STL is meshed the way `sketch --check` meshes it (per feature, then one
boolean -- mesh2cad.build.load_program), at the scorer's tolerance, and put
back in the frame of the target .stl: a program that declares `_FRAME` was
written in the rotated frame `ingest` aligned the target to, and an export left
in that frame sits rotated against the mesh it reproduces.

Not an `m2c.py` command on purpose: every m2c command is listed in the agent's
prompt, and exporting is the harness's job after the run, not the model's.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mesh2cad.build import STL_TOL, load_program  # noqa: E402
from mesh2cad.frame import describe_frame, ingest_frame, program_frame  # noqa: E402


class FrameMissing(RuntimeError):
    """The program declares no `_FRAME`, and the target needs one undone."""


def missing_frame(program, target):
    """Why `program` cannot be put back in `target`'s frame, or None.

    A program written against a target `ingest` scaled or rotated is in that
    frame; without its `_FRAME` the export is written as built -- for a tiny
    part ingest scaled up, 10x or 100x the target's size.
    """
    T, _ = program_frame(program)
    if T is not None:
        return None
    from mesh2cad.normalize import ingest
    need = ingest_frame(ingest(str(target))[1])
    if need is None:
        return None
    return (f"refused: {program} declares no _FRAME, but ingest "
            f"{describe_frame(need)} {target}, so the program's coordinates are "
            f"in that frame and an STL written as built would not be the "
            f"target's size or place. Re-emit it with `sketch ... --base` or "
            f"`edit` (both write the target's _FRAME now), or pass --frame "
            f"program to write it as built anyway")


def export(program, out, frame="target", tol=STL_TOL, target=None):
    """Write `program` to `out` and return the lines that say what was written.

    With `target`, a program that should declare a frame and does not is
    refused (FrameMissing) rather than written in the wrong one."""
    if target is not None and frame == "target":
        why = missing_frame(program, target)
        if why:
            raise FrameMissing(why)
    mesh = load_program(program, tol=tol)
    T, how = program_frame(program)
    notes = []
    if frame == "target" and T is not None:
        mesh.apply_transform(np.linalg.inv(T))
        notes.append(f"frame: undid the program's _FRAME ({how}, "
                     f"{describe_frame(T)}) so the STL sits where the target does")
    elif T is not None:
        notes.append(f"frame: left in the program's frame "
                     f"({describe_frame(T)}), as asked")
    else:
        notes.append("frame: the program declares no _FRAME; written as built")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out)
    notes.append(f"wrote {out}: {len(mesh.faces)} faces, volume {mesh.volume:.3f}, "
                 f"watertight {mesh.is_watertight}, "
                 f"{mesh.body_count} bodies, tolerance {tol}")
    return notes


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("program", help="the emitted program (sketch --emit)")
    ap.add_argument("out", help="the .stl to write")
    ap.add_argument("--frame", choices=("target", "program"), default="target",
                    help="target (default): the frame of the input .stl; "
                         "program: the rotated frame the program was written in")
    ap.add_argument("--tol", type=float, default=STL_TOL,
                    help=f"linear meshing tolerance (default {STL_TOL}, the scorer's)")
    ap.add_argument("--target", default=None,
                    help="the .stl the program reconstructs: a program without "
                         "_FRAME is refused when ingest scaled or rotated it")
    a = ap.parse_args()
    try:
        lines = export(a.program, a.out, a.frame, a.tol, target=a.target)
    except FrameMissing as exc:
        print(exc)
        sys.exit(2)
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
