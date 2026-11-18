"""The emitted program as a list of features, so a feature can be replaced,
dropped or undone instead of rebuilding the whole part.

`sketch --base` writes a program that is a chain of blocks. Each block starts
with the `# sketch on ...` header `_finish_sketch` writes, holds the CadQuery
for ONE feature, and is joined to the block before it by two fixed glue lines:

    <block 0>
    _prev = result
    _PREV_STEPS = list(_STEPS)

    <block 1 header + code>
    result = _prev.union(result)
    _STEPS = _PREV_STEPS + _STEPS
    ...
    <optional _FRAME block>
    show_object(result)

A block whose header carries `# combine: cut` (or `intersect`) is joined by
`_prev.cut(...)` / `_prev.intersect(...)` instead, with the feature's finished
solid as the one tool -- see GLUES.

A block without its glue is a complete program for that feature alone, which
is what makes editing safe: removing or replacing one block and re-joining the
rest with the same glue gives exactly the program the harness would have
written had that feature been built differently in the first place. Nothing is
re-measured and no geometry is re-derived -- the edit is a list operation.

Every program written with `--emit` is also kept as a numbered snapshot next
to it, so the last write can be undone. The snapshot is what the reference
harness calls an attempt: a broken emit (an invalid union, a feature on the
wrong level) used to overwrite the only copy of the good program.
"""

import json
import re
from pathlib import Path

from mesh2cad.frame import FRAME_BLOCK

HEADER = re.compile(r"(?m)^(?=# sketch on )")
# The solids `--emit` leaves out of a program. Not a feature: it is written
# last, and worked out again at every write from what the program then builds.
SLIVER_BLOCK = re.compile(r"\n*# --- slivers ---.*?^# --- end slivers ---[ \t]*\n?",
                          re.S | re.M)
GLUE = "_prev = result\n_PREV_STEPS = list(_STEPS)"
UNION = "result = _prev.union(result)\n_STEPS = _PREV_STEPS + _STEPS"
# How a block joins the ones before it. A cutting feature subtracts its
# FINISHED solid -- outline minus its own inner rings -- as one tool: turning
# each of its steps into a cut would subtract the outline and then the holes
# too, which is not the same solid.
GLUES = {
    "add": UNION,
    "cut": ("_tool = result\nresult = _prev.cut(_tool)\n"
            "_STEPS = _PREV_STEPS + [(\"cut\", _tool)]"),
    # Fuzzy, at the meshing tolerance: two faces of one surface measured by
    # different instruments -- a hole stack's bore and the meridian's, a few
    # thousandths apart -- make OCC's exact common return nothing at all.
    "intersect": ("_tool = result\nresult = _prev.intersect(_tool, tol=0.01)\n"
                  "_STEPS = _PREV_STEPS + [(\"intersect\", _tool)]"),
    # A body of its own: put beside what is there, never fused into it.
    # A later feature fuses only with the bodies it touches -- OCC does not
    # intersect the pieces of one compound with each other -- so a screw stays
    # a screw beside the plate it sits in.
    "separate": ("_tool = result\nresult = cq.Workplane(obj=cq.Compound.makeCompound("
                 "_prev.solids().vals() + _tool.solids().vals()))\n"
                 "_STEPS = _PREV_STEPS + [(\"body\", _tool)]"),
}
COMBINE = re.compile(r"(?m)^# combine: (cut|intersect|separate)\b")


# The header lines of a feature folded into another by `--combine intersect
# --with N` start with this, so they neither split the program (HEADER) nor set
# the composite's mode (COMBINE).
CLIPPED = "#   clipped by: "


def intersect_into(feature, tool):
    """`feature` clipped by `tool` alone, as ONE block that takes its place.

    `--combine intersect` clips everything built so far; this clips one feature
    and leaves the rest alone -- a joggled plate is its outline extrusion
    intersected with the strip a cut across the bend gives, whatever else the
    part already has. Both blocks are standalone programs, so running one
    after the other and intersecting their results is exact; the composite keeps
    `feature`'s header and mode, and records the tool's header beneath it.
    """
    lines, first = [], True
    for ln in tool.strip().splitlines():
        if ln.startswith("# "):
            ln = (CLIPPED if first else "#   ") + ln[2:]
            first = False
        lines.append(ln)
    return (feature.strip() + "\n_feat = result\n\n" + "\n".join(lines)
            + "\n_tool = result\nresult = _feat.intersect(_tool, tol=0.01)\n"
            + "_STEPS = [(\"add\", result)]")


def mode_of(feature):
    """"add", "cut", "intersect" or "separate", from the block's header."""
    m = COMBINE.search(feature)
    return m.group(1) if m else "add"
IMPORT = "import cadquery as cq"
SHOW = "show_object(result)"


class ProgramError(ValueError):
    """A program this module cannot take apart; the message says why."""


def without_slivers(src):
    """`src` without its sliver block."""
    return SLIVER_BLOCK.sub("\n", src)


def sliver_block(n, floor):
    """Source that leaves the `n` solids under `floor` mm^3 out of `result`."""
    return (f"# --- slivers ---\n"
            f"# {n} solid(s) under the sliver floor, left out of the result\n"
            f"_kept = [_s for _s in result.solids().vals() if _s.Volume() >= {floor:.6g}]\n"
            f"result = cq.Workplane(\"XY\").newObject(\n"
            f"    _kept if len(_kept) == 1 else [cq.Compound.makeCompound(_kept)])\n"
            f"# --- end slivers ---\n")


def split_program(src):
    """(features, frame) of an emitted program.

    `features` is a list of standalone blocks, header included. `frame` is the
    `_FRAME` block text ("" when the target was not rotated), carried through
    untouched so an edit never changes the frame the program was measured in.
    """
    m = FRAME_BLOCK.search(src)
    frame = m.group(0).strip() if m else ""
    body = without_slivers(FRAME_BLOCK.sub("\n", src))
    body = body.replace(IMPORT + "\n", "").replace(SHOW, "")
    chunks = HEADER.split(body)
    lead, chunks = chunks[0].strip(), chunks[1:]
    if not chunks:
        raise ProgramError("no `# sketch on` header found: this program was "
                           "not written by `sketch`, so it has no features to "
                           "edit")
    if lead:
        raise ProgramError("the program has code before its first feature "
                           "header; it was edited by hand and cannot be split "
                           "safely")
    feats = []
    for i, c in enumerate(chunks):
        c = c.strip()
        if c.endswith(GLUE):
            c = c[: -len(GLUE)].rstrip()
        if i > 0:
            glue = GLUES[mode_of(c)]
            if not c.endswith(glue):
                raise ProgramError(f"feature #{i} does not end with the "
                                   f"{mode_of(c)} glue `sketch --base` writes; "
                                   "the program was edited by hand")
            c = c[: -len(glue)].rstrip()
        feats.append(c)
    return feats, frame


def join_program(features, frame=""):
    """The inverse of split_program: the program `sketch` would have written."""
    if not features:
        raise ProgramError("a program needs at least one feature")
    if mode_of(features[0]) != "add":
        raise ProgramError(f"feature #0 would {mode_of(features[0])} from "
                           "nothing: the first feature must add material")
    parts = [features[0].strip()]
    for f in features[1:]:
        parts.append(GLUE + "\n\n" + f.strip() + "\n" + GLUES[mode_of(f)])
    out = IMPORT + "\n\n" + "\n".join(parts)
    if frame:
        out += "\n\n" + frame.strip() + "\n"
    return out + "\n\n" + SHOW + "\n"


_RANGE = re.compile(r"([XYZ])=(-?[\d.]+(?:e-?\d+)?)\D+?\1=(-?[\d.]+(?:e-?\d+)?)")


def describe(feature):
    """One short line for a feature, read from the header `sketch` wrote.

    `extrude XY Z 0.00->1.20`, `revolve XZ (360 deg about X=0,Y=0 ...)`.
    """
    lines = [ln[2:] for ln in feature.splitlines() if ln.startswith("# ")]
    plane = "?"
    if lines and lines[0].startswith("sketch on "):
        plane = lines[0].split()[2]
    op, detail = "?", ""
    for ln in lines[1:]:
        if ln.startswith("profile measured"):
            continue
        if ":" in ln:
            op, detail = (s.strip() for s in ln.split(":", 1))
            break
    r = _RANGE.search(detail)
    where = (f"{r.group(1)} {float(r.group(2)):.2f}->{float(r.group(3)):.2f}"
             if r else detail[:48])
    mode = mode_of(feature)
    clipped = " clipped" if CLIPPED in feature else ""
    return (("" if mode == "add" else mode + " ") + f"{op} {plane} {where}{clipped}").strip()


def summary_line(src):
    """`program: 3 features -- #0 ... | #1 ... | #2 ...`, or "" if unreadable."""
    try:
        feats, _ = split_program(src)
    except ProgramError:
        return ""
    body = " | ".join(f"#{i} {describe(f)}" for i, f in enumerate(feats))
    return (f"program: {len(feats)} feature{'s' if len(feats) != 1 else ''} -- "
            f"{body}")


def replace_feature(src, index, new_feature):
    feats, frame = split_program(src)
    _check_index(index, feats)
    feats[index] = new_feature.strip()
    return join_program(feats, frame)


def drop_feature(src, index):
    feats, frame = split_program(src)
    _check_index(index, feats)
    if len(feats) == 1:
        raise ProgramError("#0 is the only feature; dropping it would leave "
                           "nothing. Build a new first feature without --base "
                           "instead")
    del feats[index]
    return join_program(feats, frame)


def _check_index(index, feats):
    if not 0 <= index < len(feats):
        raise ProgramError(f"there is no feature #{index}: the program has "
                           f"{len(feats)} (#0..#{len(feats) - 1})")


# --- history ----------------------------------------------------------------

def history_dir(program_path):
    p = Path(program_path)
    return p.parent / ".m2c_history" / p.stem


def _entries(program_path):
    d = history_dir(program_path)
    return sorted(d.glob("[0-9][0-9][0-9].py")) if d.is_dir() else []


def snapshot(program_path, src, note="", residual=None):
    """Keep `src` as the next numbered version of `program_path`."""
    d = history_dir(program_path)
    d.mkdir(parents=True, exist_ok=True)
    have = _entries(program_path)
    n = int(have[-1].stem) + 1 if have else 1
    (d / f"{n:03d}.py").write_text(src)
    with open(d / "index.jsonl", "a") as f:
        f.write(json.dumps({"version": n, "note": note,
                            "residual": residual}) + "\n")
    return n


def keep_current(program_path, note):
    """Snapshot the file on disk if the history doesn't already end with it.

    A program written before history existed, or by hand, has no saved
    version; overwriting it would leave `edit --undo` nothing to go back to.
    """
    p = Path(program_path)
    if not p.exists():
        return None
    have = _entries(p)
    src = p.read_text()
    if have and have[-1].read_text() == src:
        return None
    return snapshot(p, src, note)


def last_residual(program_path):
    """The residual line recorded with the newest saved version, or None."""
    idx = history_dir(program_path) / "index.jsonl"
    if not idx.exists():
        return None
    rows = [json.loads(ln) for ln in idx.read_text().splitlines() if ln.strip()]
    for r in reversed(rows):
        if r.get("residual"):
            return r["residual"]
    return None


def last_volume(program_path):
    """The volume recorded with the newest checked version, or None."""
    idx = history_dir(program_path) / "index.jsonl"
    if not idx.exists():
        return None
    rows = [json.loads(ln) for ln in idx.read_text().splitlines() if ln.strip()]
    for r in reversed(rows):
        if r.get("volume") is not None:
            return r["volume"]
    return None


def record_residual(program_path, residual, **check):
    """Attach a residual line, and what else the check said (`health`, `volume`,
    `check_failed`), to the newest snapshot's index row."""
    idx = history_dir(program_path) / "index.jsonl"
    if not idx.exists():
        return
    rows = [json.loads(ln) for ln in idx.read_text().splitlines() if ln.strip()]
    if rows:
        rows[-1]["residual"] = residual
        rows[-1].update(check)
        idx.write_text("".join(json.dumps(r) + "\n" for r in rows))


def undo(program_path, commit=True):
    """(source to restore, version it came from) -- the program as it was
    before the last write. With `commit`, the undone versions are renamed (not
    deleted) so a second undo walks one step further back; without it (a
    preview), the history is left exactly as it was."""
    p = Path(program_path)
    current = p.read_text() if p.exists() else None
    have = _entries(p)
    # Drop the versions that ARE the current file: the newest snapshot normally
    # equals it, and an undo should go one step back from what is on disk.
    while have and current is not None and have[-1].read_text() == current:
        top = have.pop()
        if commit:
            top.rename(top.with_suffix(".undone"))
    if not have:
        raise ProgramError(f"nothing to undo: {p} has no earlier saved version")
    return have[-1].read_text(), int(have[-1].stem)


_VOL = re.compile(r"(MISSING|EXTRA)(?: about)? ([0-9.eE+-]+)")


def residual_totals(line):
    """(missing, extra) mm^3 from a residual line; 0 when a side is absent."""
    if not line:
        return None
    if "nothing above the sliver floor" in line:
        return 0.0, 0.0
    got = dict((k, float(v)) for k, v in _VOL.findall(line))
    if not got:
        return None
    return got.get("MISSING", 0.0), got.get("EXTRA", 0.0)


def delta_line(before, after):
    """`vs last saved version: missing 871 -> 402, extra 594 -> 610 mm^3`."""
    a, b = residual_totals(before), residual_totals(after)
    if a is None or b is None:
        return ""
    fmt = lambda v: f"{v:.4g}"                                    # noqa: E731
    word = ("better" if sum(b) < sum(a) * 0.999 else
            "worse" if sum(b) > sum(a) * 1.001 else "unchanged")
    return (f"vs last saved version: missing {fmt(a[0])} -> {fmt(b[0])}, "
            f"extra {fmt(a[1])} -> {fmt(b[1])} mm^3 ({word})")
