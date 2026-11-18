"""Finish a new cadmageddon-freecad env. Run once after creating it, on any platform:

    conda env create -f freecad/environment.yml
    conda activate cadmageddon-freecad
    python freecad/scripts/setup_freecad_env.py

1. conda-forge's freecad puts its Python modules (FreeCAD, Part, PartDesign, ...)
   beside its shared libraries -- `<env>/lib` on Linux/macOS, `<env>\\Library\\bin`
   on Windows -- not in site-packages, so a plain `import FreeCAD` fails. This
   writes a `freecad_lib.pth` into site-packages that adds that directory to
   sys.path, and proves it by importing FreeCAD and the workbenches the harness
   builds with.
2. On Linux (WSL included) it installs `mesalib<25.1` into the env when the env
   has no OSMesa yet. With no display, VTK renders through OSMesa, and Mesa 25.1
   and later no longer ship it. `mesh2cad` selects the OSMesa window by itself
   when there is no DISPLAY / WAYLAND_DISPLAY. Windows and macOS render through
   the system's OpenGL and get nothing: on Windows, conda-forge's mesalib would
   put its own opengl32.dll in front of the graphics driver's.
   It installs with the tool that made the env (MAMBA_EXE / CONDA_EXE, else
   micromamba, mamba or conda on PATH); `--no-osmesa` skips this step.

Idempotent.
"""

import importlib
import os
import shutil
import site
import subprocess
import sys
from pathlib import Path

MESALIB = "mesalib<25.1"


def find_freecad_dir(prefix):
    for d in (prefix / "lib", prefix / "Library" / "bin", prefix / "Library" / "lib"):
        if any((d / f"FreeCAD{ext}").exists() for ext in (".so", ".pyd", ".dylib")):
            return d
    return None


def link_freecad(prefix):
    d = find_freecad_dir(prefix)
    if d is None:
        raise SystemExit(f"no FreeCAD module under {prefix}: is this the "
                         "cadmageddon-freecad env, and is freecad installed in it?")
    sp = Path(site.getsitepackages()[0])
    pth = sp / "freecad_lib.pth"
    pth.write_text(str(d) + "\n", encoding="utf-8")
    if str(d) not in sys.path:
        sys.path.append(str(d))
    for mod in ("FreeCAD", "Part", "PartDesign", "Sketcher", "MeshPart"):
        importlib.import_module(mod)
    import FreeCAD
    print(f"ok: {pth} -> {d}")
    print(f"FreeCAD {'.'.join(FreeCAD.Version()[:3])}: FreeCAD, Part, PartDesign, "
          "Sketcher, MeshPart import")


def package_manager():
    """The conda-compatible installer to add a package to this env with, or None."""
    for var in ("MAMBA_EXE", "CONDA_EXE"):
        exe = os.environ.get(var)
        if exe and Path(exe).is_file():
            return exe
    for name in ("micromamba", "mamba", "conda"):
        exe = shutil.which(name)
        if exe:
            return exe
    return None


def ensure_osmesa(prefix):
    """On Linux, put mesalib's OSMesa into the env unless it is there already."""
    if not sys.platform.startswith("linux"):
        print(f"{sys.platform}: renders use the system's OpenGL; no OSMesa needed")
        return
    if any((prefix / "lib").glob("libOSMesa.so*")):
        print(f"ok: OSMesa already in {prefix / 'lib'}")
        return
    exe = package_manager()
    cmd = [exe or "conda", "install", "-y", "-p", str(prefix), "-c", "conda-forge", MESALIB]
    if exe is None:
        raise SystemExit("no micromamba / mamba / conda found to install OSMesa with; run:\n  "
                         + " ".join(f'"{c}"' if "<" in c else c for c in cmd))
    print("installing OSMesa (headless rendering): " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    if not any((prefix / "lib").glob("libOSMesa.so*")):
        raise SystemExit(f"{MESALIB} installed, but no libOSMesa.so in {prefix / 'lib'}")
    print(f"ok: OSMesa in {prefix / 'lib'}")


def main():
    prefix = Path(sys.prefix)
    link_freecad(prefix)
    if "--no-osmesa" not in sys.argv[1:]:
        ensure_osmesa(prefix)
    try:
        import cadquery  # noqa: F401
        print("WARNING: cadquery is importable in this env; the FreeCAD harness "
              "must not depend on it")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
