"""Four isometric views of a mesh: the corners of a tetrahedron.

`visualize_isometric(mesh, mosaic)` takes a trimesh mesh and returns either one
2x2 mosaic or a dict of the four panes keyed "1", "2", "3", "4".

`visualize_isometric_clusters(mesh, clusters, colors)` renders exactly the same
four views, but paints each cluster of triangles - a segmented CAD face, say -
in a color of its own instead of the one grey material.  It is a thin wrapper:
it turns the clusters into a per-triangle color array and hands that to
`visualize_isometric(face_colors=...)`, so the two share every camera, framing
and pixel decision below and cannot drift apart.  A triangle no cluster claims
keeps the same grey, drawn by the same material, so the unclaimed part of a
colored sheet is pixel for pixel the plain one.

Everything is VTK: the shading, the orientation triad, the pane titles, the
montage and the PNG on disk.  The renderer settings -- light kit, eye-dome
lighting, optional SSAO and feature edges, supersample-and-box-filter -- are the
ones orthographic_views.py uses, so the two scripts produce matching images.
Unlike that script there is no grid, no frame, no ticks and no axis names: an
isometric pane has no meaningful 2D axes to put numbers on.  Orientation is
carried instead by VTK's own axes actor, drawn at the left of the header strip
beside the title -- outside the part's image, so it never covers it -- on an
overlay layer that shares the pane's camera.

The four camera directions are the vertices of a regular tetrahedron inscribed
in the bounding cube: (+a,-b,+u), (-a,+b,+u), (+a,+b,-u), (-a,-b,-u), where u is
the up axis.  They sit 109.47 degrees apart -- the widest four directions can be
spread -- and every face normal (+/-X, +/-Y, +/-Z) has a positive dot product
with exactly two of them.  Four panes therefore see the whole part, with each
side shown twice from different angles.  All four keep the same view-up, so the
part does not tumble between panes.

Framing is as tight as the four views allow.  Each view measures its own
silhouette -- the vertices projected onto its image plane, not the bounding box,
which would overstate the part by its corners and could not tell the four views
apart anyway (see view_frame).  The pane takes the largest width and the largest
height found over the four, so every pane is one size and the mosaic is a plain
grid; inside that shared shape each view is then framed and centred on its own,
so it fills the pane on at least one axis.  Panes are therefore not square in
general -- they are as oblong as the part is.  `uniform=True` frames all four
alike instead, which keeps the part the same size in every pane at the cost of
white space in the narrower views.

The part is rendered edge to edge, and the header -- triad and title -- is a
separate strip stacked on top of the finished image rather than a margin carved
out of it.  So the header costs the part nothing, which is what leaves room for
a legible triad and title.

Every pane's width and height is a multiple of 32 px, and so is the mosaic's.
That is arranged before the camera is set, not by padding afterwards: the part's
image is rounded up to the alignment and the parallel scale is then solved for
that exact pixel aspect, so the rounding costs at most a few percent of slack
around the part rather than a strip of white.  The header strip is separately a
multiple of 32 tall, so stacking preserves the alignment.

Panes are rendered at `ss` x resolution and box-filtered down, so ss^2 samples
land in every output pixel: that anti-aliases the geometry, the triad and the
triad's letters together, without fighting the EDL render pass (which is why
MSAA is not used instead).  Pane titles are the exception -- they are rasterised
at 1:1 and multiplied over the finished pane, because FreeType has already
anti-aliased and hinted each glyph and box-filtering that down a second time is
what makes downscaled text look muddy.

usage:
  python isometric_views.py OUT in.stl [more.stl ...] [options]
    OUT is a .png path when there is exactly one input, otherwise a directory
    that receives <stem>_iso.png for each input.

    --separate      write the four panes as <stem>_iso_1.png ... not a mosaic
    --both          write the mosaic *and* the four separate panes
    --max-side N    longest side of the part's image, in px (default 512);
                    the header strip is stacked on top of that
    --ss N          supersampling factor (default 4; costs N^2 in pixels)
    --no-edl        disable eye-dome lighting (depth-gradient detail pass)
    --ssao          screen-space ambient occlusion (shallow engravings)
    --edges         overlay sharp feature edges (crisp engravings)
    --no-axes       hide the orientation triad
    --no-titles     hide the pane titles
    --flat          flat (per-facet) shading, to inspect the tessellation
    --uniform       frame all four views alike, so the part is the same size in
                    every pane.  Off by default: each view is framed on its own,
                    which is what keeps the white space around it minimal
    --shared        extend --uniform across every mesh in the run, so two parts
                    are comparable in size -- at the cost of white space around
                    the smaller ones
    --up X|Y|Z      which model axis is "up" (default Z)
"""
import os
import sys

import numpy as np
import pyvista as pv  # mesh IO / normals only; it is VTK
import vtk
from vtkmodules.util import numpy_support

AXIS = {0: "X", 1: "Y", 2: "Z"}
KEYS = ("1", "2", "3", "4")
REF = 512.0                              # max_side the metrics below are drawn for
FONT = dict(title=15, axis=14)           # rasterised at 1:1, so real px
SS = 4                                   # default supersampling factor
ALIGN = 32                               # every image dimension is a multiple
INK = (0.16, 0.16, 0.16)
FIT = 1.03                               # slack around the tightest camera fit
AXES_BOX = 88                            # the triad's square in the header, px
AXES_PAD = 4                             # clearance around that square, px
AXES_FILL = 0.80                         # of its box the triad spans; < 1 for letters
AXES_LABEL = 1.22                        # letters pushed this far past the tips
TITLE_GAP = 12                           # header height above the title, px

RGB_ARRAY = "face_rgb"                   # per-triangle colors, on the cell data
GREY_ARRAY = "face_grey"                 # which of them are unassigned

#: The default material: the grey `visualize_isometric` paints a whole part in,
#: and the grey `visualize_isometric_clusters` leaves an unclaimed triangle.
#: One definition, because two greys that are nearly the same read as a bug.
GREY = (0.725, 0.749, 0.776)                            # VTK wants 0-1
GREY_MATERIAL = dict(ambient=0.16, diffuse=0.72, specular=0.22, power=20)

#: The same color as `GREY`, in the 0-255 the cluster colors are given in.
UNASSIGNED = tuple(int(round(c * 255)) for c in GREY)

#: Material for the colored variant.  Mostly ambient, and far less specular
#: than the grey default: a specular highlight on a saturated face reads as a
#: different, paler color, and the point of coloring the clusters is that the
#: color can be named.  Shading scales the color it is given, so the ratio
#: between the channels - the hue - survives exactly whatever the lighting;
#: what the lighting decides is how dark the face lands, and a face at half
#: brightness is maroon rather than red to anything reading the picture.
#: Measured on 2.stl over four saturated colors, the most common facet renders
#: at this share of its nominal color: 0.56 at ambient 0.34, 0.70 at 0.55,
#: 0.73 at 0.70, while the spread between the darkest and brightest lit facet
#: falls 0.35 -> 0.30 -> 0.28.  0.55 is where the color stops getting brighter
#: and the shading has not yet flattened.
COLOR_MATERIAL = dict(ambient=0.55, diffuse=0.45, specular=0.05, power=12)

#: Enough plain color names to call `visualize_isometric_clusters` without
#: importing a palette from anywhere.  Anything with an `.rgb` attribute (a
#: `segment_mesh.Color`, say), a "#rrggbb" string, or an RGB triple works too.
NAMED_RGB = {
    "red": (255, 0, 0), "lime": (0, 255, 0), "blue": (0, 0, 255),
    "yellow": (255, 255, 0), "cyan": (0, 255, 255), "magenta": (255, 0, 255),
    "orange": (255, 165, 0), "dark green": (0, 128, 0), "brown": (101, 67, 33),
    "green": (0, 128, 0), "purple": (128, 0, 128), "olive": (128, 128, 0),
    "pink": (255, 105, 180), "teal": (0, 128, 128), "navy": (0, 0, 128),
    "maroon": (128, 0, 0), "gray": (128, 128, 128), "grey": (128, 128, 128),
    "white": (255, 255, 255), "black": (0, 0, 0),
}


def parse_args(argv):
    out = argv[1]
    src, opt = [], dict(edl=True, ssao=False, edges=False, axes=True, titles=True,
                        smooth=True, up="Z", max_side=512, ss=SS, uniform=False)
    mode, shared = dict(mosaic=True, separate=False), False
    i = 2
    while i < len(argv):
        a = argv[i]
        if a == "--no-edl":
            opt["edl"] = False
        elif a == "--ssao":
            opt["ssao"] = True
        elif a == "--edges":
            opt["edges"] = True
        elif a == "--no-axes":
            opt["axes"] = False
        elif a == "--no-titles":
            opt["titles"] = False
        elif a == "--flat":
            opt["smooth"] = False
        elif a == "--separate":
            mode = dict(mosaic=False, separate=True)
        elif a == "--both":
            mode = dict(mosaic=True, separate=True)
        elif a == "--uniform":
            opt["uniform"] = True
        elif a == "--shared":
            shared = opt["uniform"] = True
        elif a == "--max-side":
            i += 1
            opt["max_side"] = float(argv[i])
        elif a == "--ss":
            i += 1
            opt["ss"] = int(argv[i])
        elif a == "--up":
            i += 1
            opt["up"] = argv[i].upper()
        elif a.startswith("--"):
            raise SystemExit(f"unknown option {a}")
        else:
            src.append(a)
        i += 1
    if not src:
        raise SystemExit("usage: " + __doc__.strip().rsplit("usage:", 1)[-1])
    return out, src, opt, mode, shared


def dir_label(vec):
    """(0.577, -0.577, 0.577) -> '+X -Y +Z'"""
    return " ".join(("+" if vec[k] > 0 else "-") + AXIS[k] for k in range(3))


def iso_views(up_axis):
    """The four tetrahedral isometric views, in mosaic order.

    Yields (key, unit camera direction, view-up).  See the module docstring for
    why these four and not the eight cube corners.  Keys "1" and "2" look down
    on the part and fill the mosaic's first row, "3" and "4" look up at it.
    """
    k_up = {"X": 0, "Y": 1, "Z": 2}[up_axis]
    k_a, k_b = [k for k in range(3) if k != k_up]
    e = np.eye(3)

    # (side, depth, up) signs; -b is "front", so +a/-b/+u is the classic iso
    signs = [(1, -1, 1), (-1, 1, 1), (1, 1, -1), (-1, -1, -1)]

    views = []
    for key, (s_a, s_b, s_u) in zip(KEYS, signs):
        d = s_a * e[k_a] + s_b * e[k_b] + s_u * e[k_up]
        views.append((key, d / np.linalg.norm(d), e[k_up]))
    return views


def camera_frame(d, up):
    """The camera's right and true-up vectors, given a view direction and up.

    `d` points from the focal point towards the camera, so the camera looks
    along -d.  view-up is only a hint: the true up is re-derived so the frame is
    orthonormal even when `up` is not perpendicular to the view.
    """
    f = -np.asarray(d, float)
    r = np.cross(f, np.asarray(up, float))
    r /= np.linalg.norm(r)
    return r, np.cross(r, f)


def view_frame(points, center, d, up):
    """How much of the image plane this view's silhouette covers, and where.

    Returns the (half_width, half_height) of the projected point cloud and the
    focal point that centres it.  Measuring the *vertices* rather than the
    bounding box matters twice over.  The box wastes the margin between itself
    and the part -- its corners project further out than any real silhouette,
    which for a round flange is most of a pane.  And the box cannot tell the
    four views apart at all: they are sign flips of a pair of axes, which map a
    box centred on its own centre exactly onto itself, so every view would
    measure the same and "fit each view" would silently do nothing.

    The focal point keeps the mesh centre's depth and slides in-plane onto the
    silhouette's midpoint, so a part whose mass sits off to one side is centred
    in the pane instead of leaving white space opposite it.
    """
    right, vup = camera_frame(d, up)
    h, k = points @ right, points @ vup
    h0, h1, k0, k1 = h.min(), h.max(), k.min(), k.max()
    focus = (np.asarray(center, float)
             + right * ((h0 + h1) / 2 - center @ right)
             + vup * ((k0 + k1) / 2 - center @ vup))
    return np.array([(h1 - h0) / 2, (k1 - k0) / 2]), focus


def view_frames(mesh, views):
    """view_frame for each view: extents (n, 2) and focal points (n, 3)."""
    pts = np.asarray(mesh.points, float)
    c = np.asarray(mesh.center, float)
    out = [view_frame(pts, c, d, up) for _, d, up in views]
    return np.maximum([e for e, _ in out], 1e-9), np.array([f for _, f in out])


def fit_extent(meshes, views):
    """The tightest (half-width, half-height) that holds every view of them all.

    Used to size the pane -- every pane in a mosaic must be one size -- and, when
    `uniform` is set, as the framing of each view too, which keeps the part the
    same size in all four panes and (given more than one mesh) keeps two parts
    comparable in size.
    """
    e = np.vstack([view_frames(m, views)[0] for m in meshes])
    return tuple(e.max(axis=0))


def to_rgb(color):
    """Anything color-shaped -> a (3,) uint8 RGB triple.

    Accepts an object carrying its own `.rgb` (a `segment_mesh.Color`), a name
    from `NAMED_RGB`, a "#rrggbb" or "#rgb" string, or a triple.  A triple of
    Python ints is read as 0-255 and a triple of floats as 0-1, so (1, 0, 0)
    is very nearly black while (1.0, 0.0, 0.0) is red - the two conventions
    cannot be told apart by value, only by type.
    """
    if hasattr(color, "rgb"):                       # e.g. segment_mesh.Color
        color = color.rgb
    if isinstance(color, str):
        key = color.strip().lower()
        if key in NAMED_RGB:
            return np.array(NAMED_RGB[key], dtype=np.uint8)
        if key.startswith("#"):
            digits = key[1:]
            if len(digits) == 3:
                digits = "".join(c * 2 for c in digits)
            if len(digits) == 6:
                return np.array([int(digits[i:i + 2], 16)
                                 for i in (0, 2, 4)], dtype=np.uint8)
        raise ValueError(f"unknown color {color!r}; pass an RGB triple, a "
                         f"'#rrggbb' string, or one of {sorted(NAMED_RGB)}")
    arr = np.asarray(color)
    if arr.shape != (3,):
        raise ValueError(f"a color needs three components, got {color!r}")
    if arr.dtype.kind == "f":
        arr = np.round(np.clip(arr, 0.0, 1.0) * 255.0)
    return np.clip(arr, 0, 255).astype(np.uint8)


def cluster_face_colors(n_faces, clusters, colors=None, unassigned=UNASSIGNED):
    """Per-triangle RGB for a list of triangle clusters, one color each.

    `clusters` is either a sequence of triangle-index arrays with a parallel
    `colors` sequence, or a sequence of (triangles, color) pairs with `colors`
    left out.  Triangles claimed by no cluster keep `unassigned`; a triangle
    named by two clusters takes the color of the later one.
    """
    if colors is None:
        try:
            pairs = [(triangles, color) for triangles, color in clusters]
        except (TypeError, ValueError):
            raise ValueError("pass colors=..., or make clusters a sequence of "
                             "(triangles, color) pairs") from None
    else:
        colors = list(colors)
        clusters = list(clusters)
        if len(colors) != len(clusters):
            raise ValueError(f"{len(clusters)} clusters but {len(colors)} "
                             f"colors")
        pairs = list(zip(clusters, colors))

    rgb = np.tile(np.asarray(to_rgb(unassigned), dtype=np.uint8), (n_faces, 1))
    for triangles, color in pairs:
        index = np.asarray(triangles, dtype=np.int64).ravel()
        if index.size and (index.min() < 0 or index.max() >= n_faces):
            raise IndexError(f"triangle index out of range for a mesh of "
                             f"{n_faces} triangles")
        rgb[index] = to_rgb(color)
    return rgb


def to_polydata(mesh):
    """trimesh.Trimesh (or anything with .vertices/.faces) -> pv.PolyData."""
    if isinstance(mesh, pv.PolyData):
        pd = mesh
    else:
        v = np.asarray(mesh.vertices, dtype=float)
        f = np.asarray(mesh.faces, dtype=np.int64)
        cells = np.hstack([np.full((len(f), 1), f.shape[1], np.int64), f]).ravel()
        pd = pv.PolyData(v, cells)
    pd.point_data.clear()
    pd.cell_data.clear()                      # drop any imported normals
    return pd


def prepare(pd, smooth):
    """Below the feature angle normals are averaged, above it vertices split.

    That keeps curved surfaces smooth while genuine CAD edges stay perfectly
    sharp -- which is what the isometric views are mostly there to show.
    """
    if not smooth:
        return pd
    return pd.compute_normals(feature_angle=35.0, split_vertices=True,
                              consistent_normals=True, auto_orient_normals=True)


# ---------------------------------------------------------------- 2D overlay

def text_actor(text, x, y, size, color=INK, halign="center", valign="center"):
    a = vtk.vtkTextActor()
    a.SetInput(text)
    a.SetPosition(x, y)
    p = a.GetTextProperty()
    p.SetFontFamilyToCourier()
    p.SetFontSize(int(round(size)))
    p.SetColor(*color)
    p.SetJustificationToLeft() if halign == "left" else (
        p.SetJustificationToRight() if halign == "right" else p.SetJustificationToCentered())
    p.SetVerticalJustificationToTop() if valign == "top" else (
        p.SetVerticalJustificationToCentered() if valign == "center"
        else p.SetVerticalJustificationToBottom())
    return a


def label_renderer(title, x, y, u, valign="top"):
    """Dark text on white: the pane title, rasterised at the header's own size."""
    ren = vtk.vtkRenderer()
    ren.SetLayer(0)
    ren.SetBackground(1, 1, 1)
    ren.InteractiveOff()
    ren.AddViewProp(text_actor(title, x, y, FONT["title"] * u, valign=valign))
    return ren


# ------------------------------------------------------------------ 3D pane

def axes_renderer(d, up, rect, W, H, s):
    """VTK's own axes actor, in the header band at the pane's top-left.

    It sits beside the title, above the plot, so it never covers the part -- the
    band is sized to hold it (see pane_geometry).  Its own renderer on layer 1
    with the colour buffer preserved, the trick vtkOrientationMarkerWidget uses,
    done by hand because there is no interactor off-screen.  The camera matches
    the part's, so the triad reports this pane's true model orientation.
    """
    axes = vtk.vtkAxesActor()
    axes.SetShaftTypeToCylinder()
    axes.SetCylinderRadius(0.035)
    axes.SetConeRadius(0.38)
    axes.SetNormalizedShaftLength(0.85, 0.85, 0.85)
    axes.SetNormalizedTipLength(0.24, 0.24, 0.24)
    # the caption sits *on* the cone at the default 1.0, so push it past the tip
    axes.SetNormalizedLabelPosition(AXES_LABEL, AXES_LABEL, AXES_LABEL)
    for cap in (axes.GetXAxisCaptionActor2D(),
                axes.GetYAxisCaptionActor2D(),
                axes.GetZAxisCaptionActor2D()):
        # the caption otherwise scales itself to its box and turns to mush
        cap.GetTextActor().SetTextScaleModeToNone()
        p = cap.GetCaptionTextProperty()
        p.SetFontFamilyToCourier()
        p.SetFontSize(int(round(FONT["axis"] * s)))
        p.SetColor(*INK)
        p.SetBold(True)
        p.SetItalic(False)
        p.SetShadow(False)

    ren = vtk.vtkRenderer()
    ren.SetLayer(1)
    ren.SetPreserveColorBuffer(True)        # draw over the part, do not clear
    ren.SetPreserveDepthBuffer(False)
    ren.InteractiveOff()
    ren.SetViewport(rect[0] / W, rect[1] / H, rect[2] / W, rect[3] / H)
    ren.AddActor(axes)

    # Frame it by hand rather than with ResetCamera().  vtkAxesActor reports
    # its bounds as the whole [-1, 1]^3 cube even though the three arms only
    # occupy the positive octant, so ResetCamera() fits twice what is actually
    # drawn and centres on the cube's centre rather than the triad's -- the
    # triad lands at about a quarter size, off to one side, and the slack shows
    # up as white between the header and the part.  The arm tips project the
    # same way the part does, so view_frame answers this too.
    tips = np.array([[0, 0, 0], [AXES_LABEL, 0, 0],
                     [0, AXES_LABEL, 0], [0, 0, AXES_LABEL]])
    ext, focus = view_frame(tips, tips.mean(axis=0), d, up)
    cam = ren.GetActiveCamera()
    cam.ParallelProjectionOn()
    cam.SetFocalPoint(*focus)
    cam.SetPosition(*(focus + np.asarray(d, float) * 6.0))
    cam.SetViewUp(*np.asarray(up, float))
    cam.SetParallelScale(ext.max() / AXES_FILL)   # the box is square
    ren.ResetCameraClippingRange()
    return ren


def split_by_material(mesh):
    """The mesh as [(polydata, material)] - one entry per material needed.

    A colored mesh and a grey one cannot share an actor, because the material
    is a property of the actor and the two want different ones: the grey is
    the specular, low-ambient default the whole script is drawn in, while a
    saturated color needs most of its light ambient to stay the color it was
    named (see COLOR_MATERIAL).  Rendering the unassigned triangles with the
    color material instead was a visible bug - the same grey came out
    (134, 134, 137) here and (84, 87, 90) in `visualize_isometric`.

    So the unclaimed triangles are split off into their own actor with the
    grey material, and come out pixel for pixel what `visualize_isometric`
    would have drawn.  An all-grey or all-colored mesh stays a single actor.
    """
    if not (hasattr(mesh, "cell_data") and RGB_ARRAY in mesh.cell_data):
        return [(mesh, GREY_MATERIAL)]
    if GREY_ARRAY not in mesh.cell_data:
        return [(mesh, COLOR_MATERIAL)]

    grey = np.asarray(mesh.cell_data[GREY_ARRAY], dtype=bool)
    if grey.all():
        return [(mesh, GREY_MATERIAL)]
    if not grey.any():
        return [(mesh, COLOR_MATERIAL)]
    # extract_cells keeps the point normals `prepare` computed, so the two
    # halves shade exactly as the whole did
    surface = dict(algorithm="dataset_surface")   # pyvista's default, pinned
    return [(mesh.extract_cells(~grey).extract_surface(**surface),
             COLOR_MATERIAL),
            (mesh.extract_cells(grey).extract_surface(**surface),
             GREY_MATERIAL)]


def part_actor(mesh, material, smooth):
    """One actor for one material; colored when it carries the RGB array."""
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(mesh)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()

    if material is COLOR_MATERIAL:
        # Address the array by name rather than by making it the active
        # scalars: `prepare` runs the mesh through vtkPolyDataNormals, which
        # passes cell arrays through in order but has no reason to preserve
        # which one was "active".  DirectScalars means the uint8 triples are
        # the color, with no lookup table in between, so a face comes out the
        # RGB it was given (times the shading below).
        mapper.SetScalarModeToUseCellFieldData()
        mapper.SelectColorArray(RGB_ARRAY)
        mapper.SetColorModeToDirectScalars()
        mapper.ScalarVisibilityOn()
    else:
        mapper.ScalarVisibilityOff()
        prop.SetColor(*GREY)

    prop.SetAmbient(material["ambient"])
    prop.SetDiffuse(material["diffuse"])
    prop.SetSpecular(material["specular"])
    prop.SetSpecularPower(material["power"])
    prop.SetInterpolationToPhong() if smooth else prop.SetInterpolationToFlat()
    return actor


def part_renderer(mesh, edges, d, up, focus, scale, rect, W, H, s, opt, ssao_r):
    """Layer-0 renderer holding the part, centred and at the pane's scale."""
    ren = vtk.vtkRenderer()
    ren.SetLayer(0)
    ren.SetBackground(1, 1, 1)
    ren.SetViewport(rect[0] / W, rect[1] / H, rect[2] / W, rect[3] / H)

    for part, material in split_by_material(mesh):
        actor = part_actor(part, material, opt["smooth"])
        ren.AddActor(actor)
    vtk.vtkLightKit().AddLightsToRenderer(ren)

    if edges is not None:
        em = vtk.vtkPolyDataMapper()
        em.SetInputData(edges)
        em.ScalarVisibilityOff()
        ea = vtk.vtkActor()
        ea.SetMapper(em)
        ea.GetProperty().SetColor(0.2, 0.22, 0.24)
        ea.GetProperty().SetLineWidth(1.4 * s)
        ea.GetProperty().SetLighting(False)
        ren.AddActor(ea)

    # render passes: steps -> [ssao] -> [edl]
    chain = vtk.vtkRenderStepsPass()
    if opt["ssao"]:
        # screen-space ambient occlusion darkens crevices, which is what makes a
        # shallow engraving readable on a face pointing straight at the camera
        p = vtk.vtkSSAOPass()
        p.SetDelegatePass(chain)
        p.SetRadius(ssao_r)
        p.SetBias(ssao_r * 0.02)
        p.SetKernelSize(192)
        p.BlurOn()
        chain = p
    if opt["edl"]:
        p = vtk.vtkEDLShading()
        p.SetDelegatePass(chain)
        chain = p
    ren.SetPass(chain)

    d = np.asarray(d, float)
    radius = float(np.linalg.norm(np.ptp(np.array(mesh.bounds).reshape(3, 2), axis=1))) / 2
    focus = np.asarray(focus, float)
    cam = ren.GetActiveCamera()
    cam.ParallelProjectionOn()                              # true isometric
    cam.SetFocalPoint(*focus)
    cam.SetPosition(*(focus + d * max(radius, 1e-9) * 6))
    cam.SetViewUp(*np.asarray(up, float))
    cam.SetParallelScale(scale)                             # plot height / 2
    ren.ResetCameraClippingRange()
    return ren


def grab(renderers, W, H, layers=1):
    """Render an off-screen window of the given renderers, top-down RGB."""
    rw = vtk.vtkRenderWindow()
    rw.SetOffScreenRendering(1)
    rw.SetSize(W, H)
    rw.SetNumberOfLayers(layers)
    for ren in renderers:
        rw.AddRenderer(ren)
    rw.Render()

    grabber = vtk.vtkWindowToImageFilter()
    grabber.SetInput(rw)
    grabber.SetInputBufferTypeToRGB()
    grabber.ReadFrontBufferOff()
    grabber.Update()
    img = grabber.GetOutput()
    nx, ny, _ = img.GetDimensions()
    arr = numpy_support.vtk_to_numpy(img.GetPointData().GetScalars())
    arr = arr.reshape(ny, nx, 3)[::-1].copy()   # VTK images start bottom-left
    rw.Finalize()
    return arr


def header_image(d, up, title, geom, opt):
    """The strip that goes above the part: triad at the left, title beside it.

    Rendered on its own rather than carved out of the pane, so the part keeps
    the whole of its own image.  The triad goes through the supersampled pass
    with the geometry; the title is rasterised at 1:1 and multiplied over the
    result, for the reason given in the module docstring.
    """
    k, u, W, hh = geom["ss"], geom["u"], geom["W"] // geom["ss"], geom["header"]
    box, pad = geom["box"], geom["pad"]

    # a renderer with no actors still clears its viewport, which is how the
    # strip gets its white ground under the layer-1 triad
    ground = vtk.vtkRenderer()
    ground.SetLayer(0)
    ground.SetBackground(1, 1, 1)
    ground.InteractiveOff()
    rens = [ground]
    if opt["axes"]:
        rens.append(axes_renderer(d, up, (pad * k, pad * k, (pad + box) * k,
                                          (pad + box) * k), W * k, hh * k, geom["s"]))
    arr = grab(rens, W * k, hh * k, layers=2).astype(np.float32)
    arr = arr.reshape(hh, k, W, k, 3).mean((1, 3))

    if opt["titles"]:
        # centred in what the triad leaves, not in the pane: on a narrow pane
        # centring on the pane would put the title under the triad
        x0 = (pad + box + pad) if opt["axes"] else 0
        # top-aligned, a hair below the edge: centred in the strip it reads as
        # floating, with a band of white above it
        labels = grab([label_renderer(title, (x0 + W) / 2, hh - pad, u)],
                      W, hh).astype(np.float32)
        arr = arr * (labels / 255.0)
    return np.clip(arr, 0, 255).astype(np.uint8)


def render_pane(mesh, edges, d, up, opt, focus, scale, title, ssao_r, geom):
    """One pane: the part over its whole image, with the header stacked on top."""
    W, H, k = geom["W"], geom["H"], geom["ss"]
    ren = part_renderer(mesh, edges, d, up, focus, scale, (0, 0, W, H), W, H,
                        geom["s"], opt, ssao_r)
    arr = grab([ren], W, H).astype(np.float32)

    # box-filter the supersampled pane back down: an unweighted average over the
    # k x k block is the pixel's area integral, which is what supersampling is
    arr = arr.reshape(H // k, k, W // k, k, 3).mean((1, 3))
    part = np.clip(arr, 0, 255).astype(np.uint8)

    if not geom["header"]:
        return part
    return np.vstack([header_image(d, up, title, geom, opt), part])


# --------------------------------------------------------------- composition

def align_up(n, k=ALIGN):
    return int(np.ceil(n / k - 1e-9) * k)


def pane_geometry(half_w, half_h, max_side, ui, ss, titles, axes):
    """Pixel layout of one pane: the part's image, plus the header above it.

    The part is rendered edge to edge -- no margins are carved out of it -- and
    its width and height are each rounded up to a multiple of ALIGN before the
    camera is set, so the rounding buys a little slack around the part instead
    of a strip of white beside it.  The header is a separate strip, itself a
    multiple of ALIGN, stacked on afterwards.  Pane and mosaic are therefore
    both 32-aligned on both axes, and the triad and title cost the part nothing.
    """
    need = 0
    if axes:
        need = max(need, AXES_BOX + 2 * AXES_PAD)
    if titles:
        need = max(need, FONT["title"] + TITLE_GAP)
    header = align_up(round(need * ui)) if need else 0

    # the silhouette's aspect, fitted into max_side on whichever side binds
    pw, ph = (max_side, max_side * half_h / half_w) if half_w >= half_h else \
             (max_side * half_w / half_h, max_side)
    W, H = align_up(pw), align_up(ph)
    if W <= 0 or H <= 0:
        raise ValueError(f"max_side={max_side} is too small")
    return dict(ss=ss, s=ss * ui, u=ui, header=header,
                box=round(AXES_BOX * ui), pad=round(AXES_PAD * ui),
                W=W * ss, H=H * ss, pw=W * ss, ph=H * ss)


def camera_scale(half_w, half_h, geom):
    """The parallel scale that just fits (half_w, half_h) into geom's plot area.

    Parallel scale is half the plot's height in model units; the half-width that
    buys follows from the pixel aspect, so both constraints hold at once.
    """
    return max(half_h, half_w * geom["ph"] / geom["pw"]) * FIT


def visualize_isometric(mesh, mosaic=True, max_side=512, ss=SS, up="Z",
                        edl=True, ssao=False, edges=False, axes=True,
                        titles=True, smooth=True, uniform=False, extent=None,
                        face_colors=None):
    """Render the four isometric views of a trimesh mesh.

    mesh      trimesh.Trimesh (anything with .vertices/.faces), or a PolyData
    mosaic    True  -> one 2x2 montage image, panes "1" "2" on the first row
              False -> dict of the four panes, keyed "1", "2", "3", "4"
    max_side  longest side of one pane in px, margins and title included.
              Rounded up to a multiple of 32, as is the other side; the mosaic
              is two panes by two, so it is 32-aligned as well
    ss        supersampling factor: the pane is rendered `ss` times larger and
              box-filtered back down, so ss x ss samples land in each output
              pixel.  Costs ss^2 in pixels rendered
    uniform   False -> every view is framed on its own, so each pane is filled
              as far as the shared pane shape allows.  True -> all four share
              one framing, so the part is the same size in every pane and the
              four stay comparable, at the cost of white space in the views
              whose projection is smaller than the largest
    extent    the (half_width, half_height) that sets the pane's shape, from
              fit_extent().  None measures this mesh alone; passing one measured
              over several meshes (and uniform=True) keeps those meshes
              comparable in size
    face_colors  (n_triangles, 3) uint8, one RGB per triangle, in the mesh's
              own triangle order.  None renders the single grey material.
              `visualize_isometric_clusters` builds this from a list of
              clusters; `cluster_face_colors` is the same builder on its own.
              Triangles left at UNASSIGNED are drawn with the grey material,
              identically to a plain `visualize_isometric` of the same part

    Returns the mosaic array, or the dict of four panes.
    """
    opt = dict(edl=edl, ssao=ssao, edges=edges, axes=axes, titles=titles,
               smooth=smooth)
    raw = to_polydata(mesh)                   # this clears any cell data, so
    if face_colors is not None:               # the colors go on afterwards
        rgb = np.ascontiguousarray(face_colors, dtype=np.uint8)
        if rgb.shape != (raw.n_cells, 3):
            raise ValueError(f"face_colors is {rgb.shape}, expected "
                             f"({raw.n_cells}, 3) - one RGB per triangle")
        raw.cell_data[RGB_ARRAY] = rgb
        # Triangles left at the default grey are drawn by the default grey
        # material, so they come out exactly as `visualize_isometric` draws
        # them - see split_by_material.  Read off the color rather than from a
        # separate mask, which keeps "this triangle is UNASSIGNED grey" and
        # "this triangle looks like the rest of the script's grey" the same
        # statement.
        grey = (rgb == np.asarray(UNASSIGNED, dtype=np.uint8)).all(axis=1)
        if grey.any():
            raw.cell_data[GREY_ARRAY] = grey.astype(np.uint8)
    views = iso_views(up)
    # the pane is one shape for all four views (a mosaic is a grid), but each
    # view is framed on its own inside it unless `uniform` says otherwise
    per_view, focuses = view_frames(raw, views)
    pane_ext = np.array(extent if extent is not None else per_view.max(axis=0))
    if uniform:
        per_view = np.repeat(pane_ext[None, :], len(views), axis=0)

    bnds = np.array(raw.bounds).reshape(3, 2)
    ssao_r = float(np.linalg.norm(np.ptp(bnds, axis=1))) / 2 * 0.05

    body = prepare(raw, smooth)
    e = None
    if edges:
        e = body.extract_feature_edges(feature_angle=25.0, boundary_edges=True,
                                       feature_edges=True, non_manifold_edges=False,
                                       manifold_edges=False)

    geom = pane_geometry(*pane_ext, max_side, max_side / REF, ss, titles, axes)
    panes = {key: render_pane(body, e, d, upv, opt, focus,
                              camera_scale(*ext, geom), dir_label(d), ssao_r, geom)
             for (key, d, upv), ext, focus in zip(views, per_view, focuses)}

    if not mosaic:
        return {k: panes[k] for k in KEYS}
    return np.vstack([np.hstack([panes[KEYS[0]], panes[KEYS[1]]]),
                      np.hstack([panes[KEYS[2]], panes[KEYS[3]]])])


def visualize_isometric_clusters(mesh, clusters, colors=None,
                                 unassigned=UNASSIGNED, **kwargs):
    """The four isometric views, with each cluster of triangles in its color.

    Same views, framing, header and pixel layout as `visualize_isometric` -
    this only supplies the per-triangle colors and forwards everything else -
    so a colored sheet and a grey one of the same part line up exactly.

    mesh        trimesh.Trimesh (anything with .vertices/.faces), or a PolyData
    clusters    a sequence of triangle-index arrays, one per cluster, with a
                parallel `colors`; or a sequence of (triangles, color) pairs
                and no `colors`.  Indices are into the mesh's own triangles,
                which is what `segment_mesh.find_faces` returns
    colors      one color per cluster: an RGB triple, a "#rrggbb" string, a
                name from NAMED_RGB, or anything carrying its own `.rgb`
    unassigned  color for triangles no cluster claimed (default light grey)
    **kwargs    passed to `visualize_isometric`: mosaic, max_side, ss, up,
                edl, ssao, edges, axes, titles, smooth, uniform, extent

    Returns the mosaic array, or the dict of four panes.

    A whole `segment_mesh.FaceGraph` renders as

        visualize_isometric_clusters(
            graph.mesh, [(c.triangles, c.color) for c in graph.clusters])

    since a `Color` carries its own `.rgb`.
    """
    faces = np.asarray(mesh.faces)
    n_faces = len(faces) if faces.ndim == 2 else int(mesh.n_cells)
    rgb = cluster_face_colors(n_faces, clusters, colors, unassigned)
    return visualize_isometric(mesh, face_colors=rgb, **kwargs)


def write_png(path, arr):
    """Write an RGB numpy array with VTK's own PNG writer."""
    h, w, _ = arr.shape
    img = vtk.vtkImageData()
    img.SetDimensions(w, h, 1)
    flat = np.ascontiguousarray(arr[::-1].reshape(-1, 3))
    vtk_arr = numpy_support.numpy_to_vtk(flat, deep=True,
                                         array_type=vtk.VTK_UNSIGNED_CHAR)
    vtk_arr.SetNumberOfComponents(3)
    img.GetPointData().SetScalars(vtk_arr)
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(path)
    writer.SetInputData(img)
    writer.Write()


def main():
    import trimesh

    out, srcs, opt, mode, shared = parse_args(sys.argv)
    meshes = [trimesh.load(s, force="mesh") for s in srcs]

    extent = None
    if shared:
        extent = fit_extent([to_polydata(m) for m in meshes], iso_views(opt["up"]))

    single_png = len(srcs) == 1 and out.lower().endswith(".png")
    if not single_png:
        os.makedirs(out, exist_ok=True)

    for src, mesh in zip(srcs, meshes):
        size = mesh.extents
        print(f"{src}: {len(mesh.faces)} triangles, "
              f"bbox {size[0]:.4g} x {size[1]:.4g} x {size[2]:.4g}")
        stem = os.path.splitext(os.path.basename(src))[0]
        base = out[:-4] if single_png else os.path.join(out, f"{stem}_iso")

        if mode["mosaic"]:
            img = visualize_isometric(mesh, mosaic=True, extent=extent, **opt)
            write_png(base + ".png", img)
            print(f"  -> {base}.png ({img.shape[1]}x{img.shape[0]})")
        if mode["separate"]:
            for key, img in visualize_isometric(mesh, mosaic=False,
                                                extent=extent, **opt).items():
                path = f"{base}_{key}.png"
                write_png(path, img)
                print(f"  -> {path} ({img.shape[1]}x{img.shape[0]})")


if __name__ == "__main__":
    main()
