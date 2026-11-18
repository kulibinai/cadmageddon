"""mesh2cad: a visual-algorithmic environment for reconstructing CAD from meshes."""

import os as _os
import sys as _sys

__version__ = "0.1.0"

# The instruments render off screen with pyvista/VTK. On Linux with no display (a
# cluster node, WSL without WSLg) VTK's default X window cannot open, so pick the
# OSMesa one: the env's mesalib (scripts/setup_freecad_env.py installs it on Linux).
# Set before anything creates a render window; an explicit setting wins.
if (_sys.platform.startswith("linux") and not _os.environ.get("DISPLAY")
        and not _os.environ.get("WAYLAND_DISPLAY")):
    _os.environ.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkOSOpenGLRenderWindow")
