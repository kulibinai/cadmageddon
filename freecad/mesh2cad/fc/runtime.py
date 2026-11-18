"""Run a FreeCAD program and read back what it built.

One function, `run_program`, used by everything that needs a program's solid:
`--check`, `--built`, residual, progress, the scorers, export. It executes the
program in a fresh namespace (the program creates its own document through
the prelude's `_start`), recomputes, and returns a `Build`: the part, the
per-feature tool solids, and the state of every feature in the tree -- so a
feature FreeCAD refused is reported by name, not discovered later as a
missing volume. The document is closed afterwards unless asked to keep it.

In-process, not a FreeCADCmd subprocess: importing FreeCAD costs 0.14-0.64 s,
less than starting a process would.
"""

import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from mesh2cad.fc import freecad


class ProgramError(RuntimeError):
    """A program that did not run, or ran and left no part; the message says why."""


@dataclass
class Feature:
    name: str
    label: str
    type: str
    body: str
    state: list
    error: str = ""
    valid: bool = True            # False: FreeCAD built it, but not as a valid solid

    @property
    def failed(self):
        return any(s in ("Invalid", "Error") for s in self.state)

    def line(self):
        what = self.type.split("::")[-1]
        lab = f" ({self.label})" if self.label and self.label != self.name else ""
        why = self.error or ", ".join(self.state) or ("" if self.valid else "not a valid solid")
        return f"{self.name}{lab} [{what}]: {why}"


@dataclass
class Build:
    path: str
    shape: object                 # Part.Shape -- the part
    steps: list                   # [(op, Part.Shape)] from the tree, in build order
    features: list = field(default_factory=list)
    step_names: list = field(default_factory=list)   # the object each step came from
    volume_before_last: float | None = None          # the part before its last feature
    fcstd: str | None = None
    seconds: float = 0.0
    states: list | None = None    # the part after each feature block (run with `states`)

    @property
    def failed(self):
        return [f for f in self.features if f.failed]

    @property
    def invalid(self):
        """Features FreeCAD built without complaint as an invalid solid. Only
        known when the program was run with `validate`; empty otherwise."""
        return [f for f in self.features if not f.valid]

    def block_shape(self, i):
        """What feature #i contributed on its own: its steps' solids fused, or
        None when it contributed none."""
        prefix = f"F{i:02d}_"
        own = [s for (_, s), n in zip(self.steps, self.step_names, strict=False)
               if n.startswith(prefix)]
        if not own:
            return None
        return own[0] if len(own) == 1 else own[0].fuse(own[1:])

    def tree(self):
        """The feature tree, one line per object, as FreeCAD shows it."""
        return [f"{f.body}/{f.name}  {f.type.split('::')[-1]}"
                + (f"  \"{f.label}\"" if f.label != f.name else "")
                + ("  FAILED: " + (f.error or ",".join(f.state)) if f.failed else "")
                + ("  INVALID SOLID" if not f.valid else "")
                for f in self.features]


def _features(doc):
    out = []
    for o in doc.Objects:
        if not (o.TypeId.startswith("PartDesign::") or o.TypeId.startswith("Sketcher::")
                or o.TypeId == "Part::Feature"):
            continue
        if o.TypeId == "PartDesign::Body":
            continue
        body = getattr(o, "_Body", None)
        if body is None:
            parents = [p for p in o.InList if p.TypeId == "PartDesign::Body"]
            body = parents[0] if parents else None
        err = ""
        if any(s in ("Invalid", "Error") for s in o.State):
            try:
                err = o.getStatusString()
            except Exception:                                      # noqa: BLE001
                err = ""
        out.append(Feature(o.Name, o.Label, o.TypeId, body.Name if body else "-",
                           list(o.State), err))
    return out


def _mark_invalid(doc, features):
    """Mark the PartDesign features whose own result is not a valid solid.

    FreeCAD does not mark them itself: a Pad can come back an invalid BRep
    with its State clean, and a feature built on that can leave a fraction of
    the part -- while the per-feature meshes that `--check`, `residual` and the
    STL use make the whole part. A BRepCheck per feature takes up to seconds
    per program, which is why only `--check` and export ask for it."""
    for f in features:
        if not f.type.startswith("PartDesign::") or f.failed:
            continue
        s = getattr(doc.getObject(f.name), "Shape", None)
        if s is not None and not s.isNull() and not s.isValid():
            f.valid = False


_BLOCK = None


def _block_of(obj):
    """The feature block an object belongs to (F03_Pad -> 3), or None."""
    global _BLOCK
    if _BLOCK is None:
        import re
        _BLOCK = re.compile(r"F(\d+)_")
    m = _BLOCK.match(obj.Name)
    return int(m.group(1)) if m else None


def _block_states(doc, ns):
    """The part after each feature block, in block order: for each part Body,
    the last object of its chain built by that block or one before it -- what
    the Body held then -- and the Bodies made later left out."""
    import Part
    bodies, chain = ns["_part_bodies"](doc), ns["_chain"]
    blocks = sorted({k for o in doc.Objects for k in [_block_of(o)] if k is not None})
    chains = []
    for b in bodies:
        base = b.BaseFeature
        chains.append(([base] if base is not None else []) + chain(b))
    out = []
    for i in blocks:
        shapes = []
        for objs in chains:
            last = None
            for o in objs:
                k = _block_of(o)
                if k is not None and k > i:
                    break
                last = o
            if last is not None and _block_of(last) is not None and not last.Shape.isNull():
                shapes.append(last.Shape.copy())
        out.append(shapes[0] if len(shapes) == 1 else Part.makeCompound(shapes))
    return out


def _where(exc, path):
    """`line N: <source line>` of the program frame an exception came from."""
    for fr in reversed(traceback.extract_tb(exc.__traceback__)):
        if fr.filename == str(path):
            return f"line {fr.lineno}: {(fr.line or '').strip()[:120]}"
    return ""


def run_program(path, src=None, save_as=None, keep=False, validate=False, states=False):
    """Execute the FreeCAD program at `path` (or the source `src`, reported
    as `path`) and return its Build. With `save_as`, the document is written
    there as .FCStd. With `keep`, the document stays open (build.doc). With
    `validate`, every feature is checked for being a valid solid
    (build.invalid). With `states`, the part after each feature block is read
    off the tree (build.states)."""
    App = freecad()
    path = str(path)
    if src is None:
        src = Path(path).read_text(encoding="utf-8")
    if "import cadquery" in src:
        raise ProgramError(f"{path} is a CadQuery program. This harness builds "
                           "with FreeCAD only and does not run CadQuery code; "
                           "rebuild the part with `sketch --emit`")
    if "_finish(" not in src:
        raise ProgramError(f"{path} does not end with `result, _STEPS = "
                           "_finish(doc)`: it was not written by `sketch`")
    ns = {"__name__": "__m2c_program__", "__file__": path,
          "_M2C_SAVE_AS": str(save_as) if save_as else None,
          "show_object": lambda *a, **k: None}
    before = set(App.listDocuments())
    t = time.time()
    try:
        exec(compile(src, path, "exec"), ns)
    except Exception as exc:                                       # noqa: BLE001
        where = _where(exc, path)
        _close_new(App, before)
        raise ProgramError(f"the program stopped at {where or 'an unknown line'}: "
                           f"{type(exc).__name__}: {exc}") from exc
    new = [d for n, d in App.listDocuments().items() if n not in before]
    doc = ns.get("doc") or (new[0] if new else None)
    try:
        result, steps = ns.get("result"), ns.get("_STEPS")
        if result is None or doc is None:
            raise ProgramError(f"{path} ran but left no `result`")
        b = Build(path=path, shape=result, steps=list(steps or []),
                  features=_features(doc), step_names=list(ns.get("_STEP_NAMES") or []),
                  volume_before_last=ns.get("_VOLUME_BEFORE_LAST"),
                  fcstd=str(save_as) if save_as else None,
                  seconds=round(time.time() - t, 3))
        if validate:
            _mark_invalid(doc, b.features)
        if states:
            b.states = _block_states(doc, ns)
        if keep:
            b.doc = doc
        return b
    finally:
        if not keep:
            _close_new(App, before)


def _close_new(App, before):
    for name in list(App.listDocuments()):
        if name not in before:
            App.closeDocument(name)


def load_shape(path):
    """The part a program builds, as one Part.Shape."""
    return run_program(path).shape


if __name__ == "__main__":
    # python -m mesh2cad.fc.runtime program.py [out.FCStd]: print the tree.
    b = run_program(sys.argv[1], save_as=sys.argv[2] if len(sys.argv) > 2 else None)
    print("\n".join(b.tree()))
    print(f"volume {b.shape.Volume:.4f}  steps {len(b.steps)}  {b.seconds} s")
