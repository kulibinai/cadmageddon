"""The registry every contribution lands in.

A contribution is one file under `mesh2cad/contrib/<kind>/`. Dropping it there
is the whole installation step: the CLI grows a subcommand, the agent-facing
reference grows an entry, and CI starts checking it against the contract. Nobody
has to edit a central list, which is the point -- a central list is where merge
conflicts and gatekeeping live.

Four kinds, because the project needs all four and they share the same
principles:

    instrument  an algorithm the agent calls; returns named primitives
    part        a reference part, so a change can be judged on more than four
    policy      a procedure document -- a different way to drive the same loop
    skill       a Claude Code skill package, for running the loop outside the CLI

`kind` decides which conformance tests apply and where the docs generator files
it. Everything else is common.
"""

import importlib
import pkgutil
from dataclasses import dataclass, field

KINDS = ("instrument", "part", "policy", "skill")

_ENTRIES = {}
_DISCOVERED = False


@dataclass
class Contribution:
    name: str
    kind: str
    summary: str
    author: str = ""
    module: str = ""
    # instrument
    configure: object = None      # (parser) -> None
    run: object = None            # (args) -> Result | None
    line_budget: int = None
    # part
    build: object = None          # () -> a CadQuery solid
    # policy / skill
    path: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(
                f"{self.name}: kind {self.kind!r} is not one of {KINDS}")
        if self.kind == "instrument" and self.run is None:
            raise ValueError(f"{self.name}: an instrument needs a run(args)")
        if self.kind == "part" and self.build is None:
            raise ValueError(f"{self.name}: a part needs a build()")
        if not self.summary:
            raise ValueError(
                f"{self.name}: needs a one-line summary -- it is what the agent "
                f"reads when deciding whether to call this at all")


def register(**kw):
    """Register a contribution. Call this at module scope in your file."""
    c = Contribution(**kw)
    if c.name in _ENTRIES:
        other = _ENTRIES[c.name]
        raise ValueError(
            f"{c.name!r} is already registered by {other.module or 'a built-in'}; "
            f"names are the agent's vocabulary, so they have to be unique")
    _ENTRIES[c.name] = c
    return c


def discover(force=False):
    """Import every module under mesh2cad.contrib so registrations happen."""
    global _DISCOVERED
    if _DISCOVERED and not force:
        return
    _DISCOVERED = True
    import mesh2cad.contrib as root
    for kind_pkg in pkgutil.iter_modules(root.__path__):
        if not kind_pkg.ispkg:
            continue
        pkg = importlib.import_module(f"mesh2cad.contrib.{kind_pkg.name}")
        for mod in pkgutil.iter_modules(pkg.__path__):
            if mod.name.startswith("_"):
                continue
            importlib.import_module(f"mesh2cad.contrib.{kind_pkg.name}.{mod.name}")


def entries(kind=None):
    discover()
    out = [c for c in _ENTRIES.values() if kind is None or c.kind == kind]
    return sorted(out, key=lambda c: c.name)


def get(name):
    discover()
    return _ENTRIES.get(name)


def reset():
    """Test helper: forget everything and re-discover on next use."""
    global _DISCOVERED
    _ENTRIES.clear()
    _DISCOVERED = False
