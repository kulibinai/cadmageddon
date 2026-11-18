"""What every contribution to this harness has to satisfy.

The project has one governing idea: **the model never touches geometry**. An
algorithm processes the geometry first, gives a name to everything the model
might want to refer to, and hands back a short block of text plus a picture. The
model's job is to choose names.

That idea is only worth anything if it survives contact with other people's
code, so the parts of it that can be checked mechanically live here and run in
CI over every registered contribution. The parts that cannot be checked
mechanically are enforced by review.

What is checked here:

  1. Every label matches the naming grammar, so the model can quote it back and
     an instrument can resolve it without guessing.
  2. Every label the skill declares actually appears in the text the model sees.
     A name the model cannot see is a name it cannot use.
  3. The text stays within a line budget. This is the token argument made
     concrete: a section is ten named primitives, not three thousand vertices.
  4. Numbers come attached to names. A handful of summary lines is fine; a wall
     of bare coordinates is the failure mode this whole design exists to avoid.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

# N0 · A0_0 · L0_3 · A0_0@X+ · A0_0@upper · c0 · S1
LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*(?:@[A-Za-z0-9+\-]+)?$")
MAX_LABEL_LEN = 24

# A float, possibly signed, possibly in exponent form.
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])[-+]?\d+\.\d+(?:[eE][-+]?\d+)?")

DEFAULT_LINE_BUDGET = 140
DEFAULT_UNLABELLED_NUMERIC_BUDGET = 40


@dataclass
class Result:
    """What one instrument call hands back to the model.

    `text` is what it reads, `images` is what it looks at, and `labels` is the
    contract between the two: every name in here is something the model may pass
    to a later command.
    """

    text: str = ""
    images: list = field(default_factory=list)
    labels: dict = field(default_factory=dict)

    def line(self, s=""):
        self.text += s + "\n"
        return self

    def label(self, name, meaning):
        self.labels[name] = meaning
        return self

    def image(self, path):
        self.images.append(Path(path))
        return self

    def render(self):
        """The exact bytes the CLI prints. IMAGE lines are how the agent is told
        a render exists; the harness's transcript reader keys off them."""
        head = "".join(f"IMAGE: {p}\n" for p in self.images)
        return head + self.text


def check(result, line_budget=DEFAULT_LINE_BUDGET,
          unlabelled_numeric_budget=DEFAULT_UNLABELLED_NUMERIC_BUDGET):
    """Return a list of contract violations. Empty list means it conforms."""
    problems = []
    text = result.text if isinstance(result, Result) else str(result)
    labels = result.labels if isinstance(result, Result) else {}

    lines = text.splitlines()
    if len(lines) > line_budget:
        problems.append(
            f"output is {len(lines)} lines, budget is {line_budget}. Summarise, "
            f"or name the things you are printing so the model can ask for one.")

    for name in labels:
        if len(name) > MAX_LABEL_LEN:
            problems.append(f"label {name!r} is longer than {MAX_LABEL_LEN} chars")
        elif not LABEL_RE.match(name):
            problems.append(
                f"label {name!r} does not match the naming grammar "
                f"{LABEL_RE.pattern}")
        elif name not in text:
            problems.append(
                f"label {name!r} is declared but never printed; the model cannot "
                f"refer to a name it was not shown")

    bare = _unlabelled_numeric_lines(lines, labels)
    if len(bare) > unlabelled_numeric_budget:
        problems.append(
            f"{len(bare)} lines carry numbers with no label attached (budget "
            f"{unlabelled_numeric_budget}). A number the model cannot name is a "
            f"number it has to retype. First offender: {bare[0]!r}")
    return problems


def _unlabelled_numeric_lines(lines, labels):
    """Lines that print a float without naming anything the model can refer to."""
    out = []
    for ln in lines:
        if not NUMBER_RE.search(ln):
            continue
        if any(lab in ln for lab in labels):
            continue
        # A line that carries its own name -- "bbox 95.40 x 123.48" -- is fine;
        # what we are hunting is the anonymous coordinate dump.
        if _has_leading_word(ln):
            continue
        out.append(ln.strip())
    return out


def _has_leading_word(line):
    stripped = line.strip()
    if not stripped:
        return False
    first = stripped.split()[0]
    return bool(re.match(r"^[A-Za-z][A-Za-z0-9_\-]*$", first))
