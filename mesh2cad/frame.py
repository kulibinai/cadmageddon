"""The frame a reconstruction was written in, and how to compare against it.

`ingest` rotates a target whose faces do not lie on the axes, because an
axis-aligned sketch is the only kind this harness can write. Every level, every
ring name, and therefore every emitted program is in that rotated frame -- but
the .stl on disk is not. Scoring one against the other would charge the model
for a rotation the harness performed.

Kept out of the scoring scripts so it can be imported -- and tested --
on its own.
"""

import ast
import json
import os
import re
from pathlib import Path

import numpy as np

# The whole `_FRAME = [[...]]` statement together with its comment block, so a
# base program can be inlined without dragging a stale copy along.
#
# Trailing `\s*` rather than `\n`: `--base` rstrips the source before it strips
# the block, so on the last program in a chain there is no newline left to
# match, and the pattern quietly failed to fire -- leaving two `_FRAME` lines
# with the second shadowing the first.
FRAME_BLOCK = re.compile(r"\n*# --- frame ---.*?^_FRAME = \[\[.*?\]\]\s*",
                         re.S | re.M)
_FRAME_VALUE = re.compile(r"^_FRAME = (\[\[.*?\]\])\s*$", re.S | re.M)


def frame_block(T):
    """The rotation and scale `ingest` applied, as source to append to a program.

    Empty when the target was already axis-aligned and of millimetre size,
    which is most parts -- so the common case costs nothing and reads unchanged.
    """
    if T is None:
        return ""
    rows = ",\n          ".join(
        "[" + ", ".join(f"{v:.12g}" for v in row) + "]" for row in T)
    s = scale_of(T)
    what = ("rotated" if abs(s - 1.0) < 1e-9 else
            f"scaled by {s:g}" if angle_of(T) < 1e-6 else f"rotated and scaled by {s:g}")
    return ("\n\n# --- frame ---\n"
            f"# ingest {what} the target with this 4x4 (to put its faces on the\n"
            "# axes, and a part far from millimetre size at millimetre size), and\n"
            "# `result` above is built in that frame. Apply the inverse to put the\n"
            "# reconstruction back in the frame of the input mesh.\n"
            f"_FRAME = [{rows}]\n")


def scale_of(T):
    """The uniform scale a frame carries (1 for a pure rotation)."""
    return float(abs(np.linalg.det(np.asarray(T, float)[:3, :3])) ** (1.0 / 3.0))


def angle_of(T):
    """How far a frame rotates, in degrees -- recorded so it is never silent.
    Its scale is divided out first."""
    M = np.asarray(T, float)[:3, :3] / scale_of(T)
    # The angle from its sine and cosine together, not arccos of the cosine
    # alone: a pure x10 scale divides by a cube root a few ulps off 10, and the
    # arccos of a cosine that far below 1 is 1.4e-6 deg -- printed as "rotated
    # 0.00 deg and scaled by 10" for a part that was never turned.
    s = np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])
    return float(np.degrees(np.arctan2(0.5 * np.linalg.norm(s), 0.5 * (np.trace(M) - 1.0))))


def describe_frame(T):
    """'rotated 10.16 deg', 'scaled by 1000', or both."""
    parts = []
    if angle_of(T) >= 1e-6:
        parts.append(f"rotated {angle_of(T):.2f} deg")
    if abs(scale_of(T) - 1.0) > 1e-9:
        parts.append(f"scaled by {scale_of(T):g}")
    return " and ".join(parts) or "identity"


def program_frame(program_path, gt_path=None):
    """The frame a program was written in, as (4x4 or None, how).

    Preferred source is the program's own `_FRAME`, so the number scored is the
    one the program declares. Programs emitted before `_FRAME` existed carry no
    such line; for those the transform is recomputed from the target, which is
    sound because `align_to_axes` is deterministic and every program here came
    out of `sketch --emit` against its output. Which of the two was used is
    returned rather than assumed: the declared one is a fact and the recomputed
    one is an inference, and a caller that prints them the same way is lying by
    a little.
    """
    try:
        T = declared_frame(Path(program_path).read_text())
    except ValueError as exc:
        raise ValueError(f"{program_path}: {exc}") from None
    if T is not None:
        return T, "declared"
    if gt_path is None:
        return None, "none"
    from mesh2cad.normalize import align_to_axes, load
    _, T, rep = align_to_axes(load(str(gt_path)))
    if not rep.get("applied"):
        return None, "none"
    return np.asarray(T, dtype=float), "recomputed"


def declared_frame(src):
    """The 4x4 `_FRAME` a program's source declares, or None."""
    m = _FRAME_VALUE.search(src)
    if not m:
        return None
    T = np.asarray(ast.literal_eval(m.group(1)), dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"_FRAME is not 4x4, it is {T.shape}")
    return T


def same_frame(a, b):
    """Whether two frames (4x4, or None for none) are the same move."""
    A = np.eye(4) if a is None else np.asarray(a, float)
    B = np.eye(4) if b is None else np.asarray(b, float)
    return bool(np.allclose(A, B, rtol=1e-9, atol=1e-9 * max(scale_of(A), 1.0)))


def ingest_frame(info):
    """The frame `ingest` put a mesh in, from its info dict, or None when it
    neither rotated nor scaled it."""
    if info["alignment"].get("applied") or info.get("scale", 1.0) != 1.0:
        return np.asarray(info["transform"], float)
    return None


# `residual --export` writes the leftover in the frame `ingest` put the target
# in, not in the frame of the .stl on disk -- and a leftover is millimetre-sized
# however small the part, so its own ingest finds nothing to undo. A sketch on
# it would write a program with no `_FRAME` at all, delivered at 10x or 100x
# its size. So the export says, beside itself, which target it came from and
# in what frame.
SOURCE_SUFFIX = ".source.json"


def source_record(mesh_path):
    """Where a mesh's provenance is kept: `left.stl` -> `left.stl.source.json`."""
    p = Path(mesh_path)
    return p.with_name(p.name + SOURCE_SUFFIX)


def write_export_source(export_path, target_path, T):
    """Record that `export_path` is a residual export of `target_path`, written
    in frame `T` (the target's ingest frame, or None). An export of an export
    names the first target, in the frame the chain is in."""
    up = export_source(target_path)
    root, named = Path(target_path).resolve(), str(target_path)
    if up is not None:
        root, named = up["source"] or Path(up["named"]), up["named"]
        T = (up["frame"] if T is None else
             np.asarray(T, float) @ (np.eye(4) if up["frame"] is None else up["frame"]))
    here = Path(export_path).resolve().parent
    rec = {"kind": "residual export",
           "target": named,
           "source": os.path.relpath(root, here),
           "source_abs": str(root),
           "frame": None if T is None else np.asarray(T, float).tolist()}
    source_record(export_path).write_text(json.dumps(rec, indent=1) + "\n")


def export_source(mesh_path):
    """{'named', 'source', 'frame'} when `mesh_path` is a residual export, else
    None. `named` is the target as `residual` was given it; `source` its path,
    or None when it is no longer there; `frame` the 4x4 the export's
    coordinates are in, or None."""
    try:
        rec = json.loads(source_record(mesh_path).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(rec, dict) or not rec.get("source"):
        return None
    here = Path(mesh_path).resolve().parent
    found = next((c.resolve() for c in (here / rec["source"], Path(rec.get("source_abs") or ""))
                  if str(c) not in ("", ".") and c.is_file()), None)
    T = rec.get("frame")
    return {"named": rec.get("target") or rec["source"], "source": found,
            "frame": None if T is None else np.asarray(T, float)}


def to_program_frame(gt_raw, T):
    """Put the ground truth in the frame the program was written in.

    The same kind of move CADENA already makes with `transform_mesh_0_1`, which
    puts both meshes in a common box before comparing. Here the common frame is
    the one the program was written in.
    """
    if T is None:
        return gt_raw
    out = gt_raw.copy()
    out.apply_transform(np.asarray(T, float))
    return out
