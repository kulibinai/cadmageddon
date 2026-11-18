"""FreeCAD as the harness's CAD kernel.

    runtime   run a FreeCAD program, read its feature tree, close it
    mesh      tessellate a FreeCAD shape exactly the way the scorer does
    check     validity verdicts in the harness's wording
    prelude   the helpers every emitted program carries (inlined, not imported)

This package is the only place that imports FreeCAD. Nothing in the harness
imports cadquery or OCP.
"""

import os
import sys
from pathlib import Path

_PRELUDE = Path(__file__).with_name("prelude.py")
BEGIN, END = "# >>> m2c prelude >>>", "# <<< m2c prelude <<<"


def freecad():
    """`import FreeCAD`, making it importable first if this is a conda env whose
    FreeCAD sits beside its libraries rather than in site-packages (what
    scripts/setup_freecad_env.py fixes once and for all), or if M2C_FREECAD
    names the directory that holds it."""
    try:
        import FreeCAD
    except ImportError:
        dirs = [os.environ.get("M2C_FREECAD", ""), str(Path(sys.prefix) / "lib"),
                str(Path(sys.prefix) / "Library" / "bin")]
        for d in dirs:
            if d and any((Path(d) / f"FreeCAD{x}").exists() for x in (".so", ".pyd")):
                sys.path.append(d)
                break
        try:
            import FreeCAD
        except ImportError as exc:
            raise ImportError(
                "FreeCAD is not importable from this python. Activate the "
                "mesh2cad-freecad env and run `python scripts/setup_freecad_env.py` "
                "once (docs/RUN_freecad.md), or set M2C_FREECAD to the directory "
                "holding FreeCAD.so / FreeCAD.pyd") from exc
    # Overwriting an .FCStd would otherwise leave a timestamped .FCBak beside it.
    FreeCAD.ParamGet("User parameter:BaseApp/Preferences/Document").SetBool(
        "CreateBackupFiles", False)
    return FreeCAD


def prelude_source():
    """The prelude block, verbatim, as it is inlined at the top of a program.
    Read from the file, so an emitter needs no FreeCAD to write a program."""
    src = _PRELUDE.read_text(encoding="utf-8")
    a, b = src.index(BEGIN), src.index(END) + len(END)
    return src[a:b] + "\n"
