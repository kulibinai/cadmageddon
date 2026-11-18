"""The emitted program as a list of features, so a feature can be replaced,
dropped or undone instead of rebuilding the whole part.

A program `sketch` writes is a FreeCAD script that builds a PartDesign feature
tree. It is the prelude (mesh2cad/fc/prelude.py, inlined), a document, one
BLOCK per feature, and the part read back from the tree:

    # -*- coding: utf-8 -*-
    <prelude>
    doc, body = _start("reconstruction")

    # sketch on XY at Z = 0.0000  (part face)       <- the feature's header
    # extrude: from Z=0.0000 to Z=5.0000
    _b = _block(doc, "F00", "add", "#00 extrude XY Z 0.00->5.00")
    sk = _sketch(_b, _plane_at("XY", 0))           <- the feature's body
    ...
    _end(_b)

    # sketch on ...
    # combine: cut (subtracted from everything before it)
    _b = _block(doc, "F01", "cut", "#01 cut extrude ...")
    ...
    _end(_b)

    <optional _FRAME block>
    result, _STEPS = _finish(doc)

A FEATURE, as this module hands it out, is its header and its body -- the
block's opening and closing lines are not part of it. They are written by
`join_program` from the feature's position and header: the block's name is its
index (F00, F01 ...), its mode the header's `# combine:` line, its label the
`describe` line. So removing or replacing one feature and joining the rest gives
exactly the program the harness would have written had that feature been built
differently in the first place, tree names included. Nothing is re-measured and
no geometry is re-derived -- the edit is a list operation.

Every program written with `--emit` is also kept as a numbered snapshot next
to it, so the last write can be undone. The snapshot is what the reference
harness calls an attempt: a broken emit (an invalid union, a feature on the
wrong level) would otherwise overwrite the only copy of the good program.
"""

import json
import re
from pathlib import Path

from mesh2cad.fc import BEGIN as PRELUDE_BEGIN
from mesh2cad.fc import END as PRELUDE_END
from mesh2cad.fc import prelude_source
from mesh2cad.frame import FRAME_BLOCK

HEADER = re.compile(r"(?m)^(?=# sketch on )")
# The solids `--emit` leaves out of a program. Not a feature: it is
# written last, after `_finish`, and worked out again at every write from what
# the program then builds.
SLIVER_BLOCK = re.compile(r"\n*# --- slivers ---.*?^# --- end slivers ---[ \t]*\n?",
                          re.S | re.M)
COMBINE = re.compile(r"(?m)^# combine: (cut|intersect|separate)\b")
MODES = ("add", "cut", "intersect", "separate")

# The opening and closing lines of a feature's block, written by join_program.
OPEN = re.compile(r'(?m)^_b = _block\(doc, "F\d+", "(add|cut|intersect|separate)"'
                  r'(?:, .*)?\)\n')
CLOSE = "_end(_b)"

HEAD = ("# -*- coding: utf-8 -*-\n"
        "# m2c reconstruction: a FreeCAD PartDesign feature tree, one block per\n"
        "# feature, written by `sketch`. Run it in FreeCAD (Macro > Execute macro)\n"
        "# or with `FreeCADCmd <this file>`; the harness reads `result` and `_STEPS`.\n")
START = 'doc, body = _start("reconstruction")'
FINISH = "result, _STEPS = _finish(doc)"

# The header lines of a feature folded into another by `--combine intersect
# --with N` start with this, so they neither split the program (HEADER) nor set
# the composite's mode (COMBINE).
CLIPPED = "#   clipped by: "


class ProgramError(ValueError):
    """A program this module cannot take apart; the message says why."""


def mode_of(feature):
    """"add", "cut", "intersect" or "separate", from the block's header."""
    m = COMBINE.search(feature)
    return m.group(1) if m else "add"


def _head_and_body(feature):
    """(header lines, body) of a feature: the comment lines it starts with."""
    lines = feature.strip().split("\n")
    k = 0
    while k < len(lines) and lines[k].startswith("#"):
        k += 1
    return "\n".join(lines[:k]), "\n".join(lines[k:])


def intersect_into(feature, tool):
    """`feature` clipped by `tool` alone, as ONE feature that takes its place.

    `--combine intersect` clips everything built so far; this clips one feature
    and leaves the rest alone -- a joggled plate is its outline extrusion
    intersected with the strip a cut across the bend gives, whatever else the
    part already has. The composite keeps `feature`'s header and mode,
    and records the tool's header beneath it. In the tree, the feature is built
    in a Body of its own, the tool after `_clip(_b)` in another, and the first
    is clipped by the second (a Common) before it joins the part.
    """
    head, body = _head_and_body(feature)
    t_lines, first = [], True
    for ln in tool.strip().splitlines():
        if ln.startswith("# "):
            ln = (CLIPPED if first else "#   ") + ln[2:]
            first = False
        t_lines.append(ln)
    t_head = [ln for ln in t_lines if ln.startswith("#")]
    t_body = [ln for ln in t_lines if not ln.startswith("#")]
    return "\n".join([head] + t_head + ["_b.clipped = True", body, "_clip(_b)"] + t_body)


def _strip(src):
    """The feature part of a program: no head, prelude, start, frame or finish."""
    if PRELUDE_BEGIN in src and PRELUDE_END in src:
        a, b = src.index(PRELUDE_BEGIN), src.index(PRELUDE_END) + len(PRELUDE_END)
        src = src[:a] + src[b:]
    m = FRAME_BLOCK.search(src)
    frame = m.group(0).strip() if m else ""
    body = without_slivers(FRAME_BLOCK.sub("\n", src))
    for line in HEAD.splitlines() + [START, FINISH]:
        body = body.replace(line + "\n", "").replace(line, "")
    return body, frame


def without_slivers(src):
    """`src` without its sliver block."""
    return SLIVER_BLOCK.sub("\n", src)


def sliver_block(n, floor):
    """Source that leaves the `n` solids under `floor` mm^3 out of `result`:
    written after `_finish`, so the tree (and its .FCStd) is the program's as
    built, and the part the harness reads and delivers is without them."""
    return (f"# --- slivers ---\n"
            f"# {n} solid(s) under the sliver floor, left out of the result\n"
            f"_kept = [_s for _s in result.Solids if _s.Volume >= {floor:.6g}]\n"
            f"result = _kept[0] if len(_kept) == 1 else Part.makeCompound(_kept)\n"
            f"# --- end slivers ---\n")


def split_program(src):
    """(features, frame) of an emitted program.

    `features` is a list of standalone features, header included, without the
    block lines. `frame` is the `_FRAME` block text ("" when the target was not
    rotated), carried through untouched so an edit never changes the frame the
    program was measured in.
    """
    if "import cadquery" in src:
        raise ProgramError("this is a CadQuery program. This harness builds with "
                           "FreeCAD only and cannot edit CadQuery code; rebuild "
                           "the part with `sketch --emit`")
    body, frame = _strip(src)
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
        m = OPEN.search(c)
        if m is None or not c.endswith(CLOSE):
            raise ProgramError(f"feature #{i} is not one block (`_b = _block(...)` "
                               f"... `_end(_b)`) as `sketch` writes it; the program "
                               "was edited by hand")
        if m.group(1) != mode_of(c):
            raise ProgramError(f"feature #{i}'s block says {m.group(1)!r} but its "
                               f"header says {mode_of(c)!r}; the program was edited "
                               "by hand")
        c = (c[:m.start()] + c[m.end():])[: -len(CLOSE)].rstrip()
        feats.append(c.strip())
    return feats, frame


def join_program(features, frame=""):
    """The inverse of split_program: the program `sketch` would have written."""
    if not features:
        raise ProgramError("a program needs at least one feature")
    if mode_of(features[0]) not in ("add", "separate"):
        raise ProgramError(f"feature #0 would {mode_of(features[0])} from "
                           "nothing: the first feature must add material")
    parts = []
    for i, f in enumerate(features):
        head, body = _head_and_body(f)
        label = f"#{i:02d} {describe(f)}"
        parts.append(f'{head}\n_b = _block(doc, "F{i:02d}", "{mode_of(f)}", {label!r})\n'
                     f"{body.strip()}\n{CLOSE}")
    out = HEAD + prelude_source() + "\n" + START + "\n\n" + "\n\n".join(parts)
    if frame:
        out += "\n\n" + frame.strip() + "\n"
    return out + "\n\n" + FINISH + "\n"


def shown(src):
    """The program as the model is shown it: the prelude -- the same few
    hundred lines in every program, and no measurement in them -- as one line.
    Printing it after every `sketch` would cost more than the feature."""
    if PRELUDE_BEGIN not in src or PRELUDE_END not in src:
        return src
    a, b = src.index(PRELUDE_BEGIN), src.index(PRELUDE_END) + len(PRELUDE_END)
    n = src[a:b].count("\n") + 1
    return (src[:a] + f"# [m2c prelude: {n} lines of FreeCAD helpers, the same in every "
            f"program -- mesh2cad/fc/prelude.py]" + src[b:])


def fcstd_path(program_path):
    """Where the .FCStd of a program is kept: beside it, same stem."""
    return Path(program_path).with_suffix(".FCStd")


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
