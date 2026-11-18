"""The SUMMARY: block every instrument ends with.

When an output gets old the harness keeps only this block (agent_run.py),
so it holds what a later turn needs: the names in play, the levels, the residual.
It is picked out of what the instrument just printed rather than computed again,
so it cannot say anything the output did not. Its lines are indented on purpose:
the harness reads residuals from lines that START with `check:`, and a copy of one
would count twice.
"""
import re

MAX_LINE = 170
MAX_LINES = 14
LEVEL = re.compile(r"\s+([A-Z]\d+)\s+(-?\d+\.\d+)\s")
RING = re.compile(r"--- ring (R\d+) \((.*?)\): (\d+ primitives[^,]*(?:,[^,]*arcs\))?)")


def _cut(line, width=MAX_LINE):
    line = line.strip()
    return line if len(line) <= width else line[:width - 3] + "..."


def _starting(lines, *prefixes):
    return [ln for ln in lines if ln.strip().startswith(prefixes)]


def _rows(lines, pattern, most, width=110, label=""):
    """The first `most` rows of a table, and how many more there were. `label`
    keeps a row from reading as a row of the table itself to whatever parses it."""
    rows = [ln for ln in lines if re.match(pattern, ln)]
    more = [f"... and {len(rows) - most} more"] if len(rows) > most else []
    return [label + _cut(r, width) for r in rows[:most]] + more


def _levels(lines):
    """A level table as one line: names and coordinates, not what they are."""
    found = {}
    for ln in lines:
        m = LEVEL.match(ln)
        if m:
            found.setdefault(m.group(1), m.group(2))
    if not found:
        return []
    return ["levels: " + ", ".join(f"{k} {v}" for k, v in found.items())]


def _section(lines):
    out = _starting(lines, "SKETCH PLANE:", "meridian")[:3]
    rings = []
    for i, ln in enumerate(lines):
        m = RING.match(ln)
        if m:
            shape = lines[i + 1].strip() if i + 1 < len(lines) else ""
            rings.append(f"{m.group(1)} {m.group(2)}: {m.group(3)}"
                         + (f"; {shape}" if shape.startswith("shape:") else ""))
    out += rings[:7]
    if len(rings) > 7:
        out.append(f"... and {len(rings) - 7} more rings")
    return out + _rows(lines, r"\s+[MX]\d+\s", 3) + _levels(lines)


def _sketch(lines):
    return (_starting(lines, "feature:", "wrote ", "undo:", "check:", "program:")
            + _rows(lines, r"\s+#\d+\s", 8))


def _overview(lines):
    return (_starting(lines, "extents", "volume", "watertight", "bodies", "regime",
                      "revolve", "helix")
            + _rows(lines, r"\s+H\d+\s", 4))


def _spectrum(lines):
    return lines[:1] + _rows(lines, r"\s+c\d+\s", 5)


def _ortho(lines):
    return _starting(lines, "section plane") + _levels(lines)


def _residual(lines):
    return (_starting(lines, "target volume")
            + _starting(lines, "MISSING (") + _rows(lines, r"\s+D\d+\s", 3)
            + _starting(lines, "EXTRA (") + _rows(lines, r"\s+E\d+\s", 3))


def _faces(lines):
    return (_starting(lines, "faces of", "area:") + _rows(lines, r"\s+F\d+\s", 6, 80)
            + _rows(lines, r"\s+C\d+\s", 4, 80, "hole "))


READERS = {
    "overview": _overview,
    "survey": lambda lines: _starting(lines, "recommended:"),
    "spectrum": _spectrum,
    "ortho": _ortho,
    "section": _section,
    "sketch": _sketch,
    "edit": _sketch,
    "progress": lambda lines: _starting(lines, "target ", "built ", "difference",
                                        "missing "),
    "residual": _residual,
    "faces": _faces,
    "symmetry": lambda lines: _rows(lines, r"\s*[A-Z]*S\d+\b", 8),
}


def block(command, text):
    """The block for one instrument's output, or "" for a command that has none."""
    read = READERS.get(command)
    if read is None:
        return ""
    picked = [_cut(ln) for ln in read(text.splitlines())][:MAX_LINES]
    if not picked:
        return ""
    return "\nSUMMARY:\n" + "\n".join(f"  {ln}" for ln in picked) + "\n"


def recall(text, grep=None, ring=None, span=None):
    """Part of a past output, or all of it. What matched nothing says so, and says
    what there is instead."""
    lines = text.splitlines()
    if span:
        first, _, last = span.partition(":")
        first, last = int(first or 1), int(last or len(lines))
        return "\n".join(lines[first - 1:last])
    if grep:
        hits = [f"{n}: {ln}" for n, ln in enumerate(lines, 1)
                if grep.lower() in ln.lower()]
        return "\n".join(hits) or f"nothing in this output matches {grep!r}"
    if ring:
        out, inside = [], False
        for ln in lines:
            if ln.startswith("--- ring "):
                inside = ln.startswith(f"--- ring {ring.upper()} ")
            elif not ln.strip():
                inside = False
            if inside:
                out.append(ln)
        have = re.findall(r"^--- ring (R\d+)", text, re.M)
        return "\n".join(out) or (f"no ring {ring} in this output; it has "
                                  f"{', '.join(have) or 'no rings'}")
    return text
