"""Six orthographic projections of a mesh: +X, -X, +Y, -Y, +Z, -Z.

`visualize_orthographic(mesh)` takes a trimesh object and returns either one
2x3 mosaic or a dict of six separate panes, plus the dict of pane titles.

`visualize_orthographic_clusters(mesh, clusters, colors)` renders the same six
panes with each cluster of triangles - a segmented CAD face, say - in a color
of its own.  It is a thin wrapper over `visualize_orthographic(face_colors=)`,
so grid, ticks, framing and scale are decided in one place for both.  The
color vocabulary and the split into a colored and a grey material are imported
from isometric_views rather than restated here: two copies of "the default
grey" is precisely the bug this file and that one would drift into.

A reflected pane (see below) keeps its colors: the reflection and the
re-winding preserve cell order, so triangle i is still cluster i's.

Everything is VTK: the shading, the blue grid, the tick numbers, the pane
titles, the montage and the PNG on disk.  matplotlib is not involved (it has no
depth buffer -- sorting triangles on their centroid draws large or slanted
facets in the wrong order, which is what produced the stray-triangle speckle
and the fake "engraved pattern" in the original version of this script).

Each pane is one off-screen render window with two layered renderers:
  layer 0  the part, parallel projection, light kit, eye-dome lighting,
           confined to a sub-viewport so the margins stay clear
  layer 1  a 2D overlay -- frame, ticks, labels, title -- addressed directly in
           pixels.  The projection is linear and parallel_scale is known, so
           model coordinates map onto pixels exactly and the tick numbers are
           true model units, printed as they are.  The grid step is measured off
           the mesh -- about GRID_LINES lines across its largest side, at a
           1/2/5 x 10^k spacing, ticks at half of it -- so a part 1 unit across
           and one 10000 units across are both gridded, and no label runs past
           MAX_DECIMALS decimal places.
The blue grid belongs behind the part, which no renderer ordering can do (2D
actors always draw last, and a grid on its own layer is overwritten by the EDL
pass).  So it is painted in numpy and the pane is composited over it, using a
second flat white-on-black pass of the same camera as the coverage mask.
Panes are rendered at `ss` x resolution and box-filtered down, so ss^2 samples
land in every output pixel: that anti-aliases the geometry, the grid and the
text together, without fighting the EDL render pass (which is why MSAA is not
used instead).

Each pane's window is [0, max] x [0, max] by default -- the origin in the
lower-left corner, the model's extent on that axis at the far edge.  Pass
window="bounds" and it becomes [min, max] instead, framing the part rather than
its position: a part that does not reach the origin otherwise spends the gap on
empty frame, and one that reaches a negative coordinate is cut off by the
default window altogether.  Tick numbers are true model coordinates in both.
One pixels-per-unit
is shared by all six panes, so they are directly comparable, and the pane with
the largest span is exactly `max_side` px on its longer side (margins, ticks
and title included).  Panes therefore differ in aspect ratio; the two panes of
a column share both axes, so a column has one width, and panes are centred
vertically in a common height.  Every returned image is padded (right and
bottom, white) to a multiple of 32 px.

Every pane reads like a chart: horizontal axis increasing right, vertical axis
increasing up.  A camera that does that is uniquely determined -- cross(h, v) --
and for three of the six faces it sits on the opposite side.  Those panes are
made by reflecting the mesh along the view's own depth axis and shooting from
there, which leaves the two in-plane axes (and so every tick number) untouched.
They are mirror images of the true view, and titled "(mirrored)".

usage:
  python orthographic_views.py in.stl out.png [options]
    --separate      write the six panes as out_+X.png ... instead of a mosaic
    --max-side N    longest side of the largest pane, in px (default 512)
    --ss N          supersampling factor (default 4; costs N^2 in pixels)
    --no-edl        disable eye-dome lighting (depth-gradient detail pass)
    --ssao          screen-space ambient occlusion (shallow engravings)
    --edges         overlay sharp feature edges (crisp engravings)
    --no-grid       hide the grid, ticks and axis names
    --flat          flat (per-facet) shading, to inspect the tessellation
    --up X|Y|Z      which model axis is "up" (default Y)
    --window MODE   origin (default): each axis runs [0, max], so the origin is
                    in the lower-left corner and the picture says where the part
                    sits.  bounds: each axis runs [min, max], framing the part
                    itself -- which is what you want when the detail matters
                    more than the position, or when the part reaches negative
                    coordinates and "origin" would leave it outside the frame
"""
import sys

import numpy as np
import pyvista as pv  # mesh IO / normals only; it is VTK
import vtk
from vtkmodules.util import numpy_support

# the per-triangle color vocabulary, shared so the two scripts cannot disagree
# about what "no color" looks like
from isometric_views import (
    GREY_ARRAY,
    RGB_ARRAY,
    UNASSIGNED,
    cluster_face_colors,
    part_actor,
    split_by_material,
)

AXIS = {0: "X", 1: "Y", 2: "Z"}
KEYS = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
REF = 512.0                              # max_side the metrics below are drawn for
MARGIN = dict(left=60, right=20, bottom=54, top=30)    # room for ticks + title
FONT = dict(tick=16, name=18, title=19)   # rasterised at 1:1, so real px
GRID_LINES = 10                          # grid lines across the part's largest side
TICK_RATIO = 0.5                         # tick marks at this fraction of the grid step
MAX_DECIMALS = 4                         # a tick label never states more than this
MIN_STEP = 10.0 ** -MAX_DECIMALS         # ...which is what floors the step
TICK_LEN = 4                             # px a tick sticks out of the frame
# What each axis is shown over.  "origin" keeps the origin in the lower-left
# corner, which says where the part sits in model space; "bounds" frames the
# part itself, which spends no frame on the gap between it and the origin.  A
# part that does not reach the origin makes these very different pictures: one
# 0.202 across sitting at 0.399 fills 34% of an "origin" axis and all of a
# "bounds" one.  Neither is right in general -- position and detail are both
# real -- so it is a parameter and "origin" stays the default.
WINDOW_MODES = ("origin", "bounds")
WINDOW_DEFAULT = "origin"
SS = 4                                   # default supersampling factor
ALIGN = 32                               # output dims are a multiple of this
BLUE = (0.12, 0.44, 0.88)
INK = (0.16, 0.16, 0.16)


def parse_args(argv):
    src, dst = argv[1], argv[2]
    opt = dict(edl=True, ssao=False, edges=False, grid=True, smooth=True,
               up="Y", mosaic=True, max_side=512, ss=SS, window=WINDOW_DEFAULT)
    i = 3
    while i < len(argv):
        a = argv[i]
        if a == "--no-edl":
            opt["edl"] = False
        elif a == "--ssao":
            opt["ssao"] = True
        elif a == "--edges":
            opt["edges"] = True
        elif a == "--no-grid":
            opt["grid"] = False
        elif a == "--flat":
            opt["smooth"] = False
        elif a == "--separate":
            opt["mosaic"] = False
        elif a == "--max-side":
            i += 1
            opt["max_side"] = float(argv[i])
        elif a == "--ss":
            i += 1
            opt["ss"] = int(argv[i])
        elif a == "--up":
            i += 1
            opt["up"] = argv[i].upper()
        elif a == "--window":
            i += 1
            opt["window"] = argv[i].lower()
        else:
            raise SystemExit(f"unknown option {a}")
        i += 1
    return src, dst, opt


def axis_label(vec):
    """(0, 0, -1) -> '-Z'"""
    k = int(np.argmax(np.abs(vec)))
    return ("+" if vec[k] > 0 else "-") + AXIS[k]


def six_views(up_axis):
    """The six axis-aligned views, laid out so every pane reads like a chart.

    Yields (name, camera direction, view-up, h axis, v axis, reflect axis, face).
    See the module docstring for why three of them reflect the mesh.  Ordered so
    the 2x3 mosaic pairs each view with its opposite in the same column.
    """
    k_up = {"X": 0, "Y": 1, "Z": 2}[up_axis]
    k_side, k_depth = [k for k in range(3) if k != k_up]
    e = np.eye(3)

    faces = [("FRONT",   e[k_depth], k_side, k_up),
             ("RIGHT",   e[k_side],  k_depth, k_up),
             ("TOP",     e[k_up],    k_side, k_depth),
             ("BACK",   -e[k_depth], k_side, k_up),
             ("LEFT",   -e[k_side],  k_depth, k_up),
             ("BOTTOM", -e[k_up],    k_side, k_depth)]

    views = []
    for name, face, kh, kv in faces:
        cam = np.cross(e[kh], e[kv])          # the only camera with h right, v up
        mirror = float(cam @ face) < 0        # ...but it looks at the far side
        views.append((name, cam, e[kv], kh, kv,
                      int(np.argmax(np.abs(face))) if mirror else None, face))
    return views


def to_polydata(mesh):
    """trimesh.Trimesh (or anything with .vertices/.faces) -> pv.PolyData."""
    if isinstance(mesh, pv.PolyData):
        return mesh
    v = np.asarray(mesh.vertices, dtype=float)
    f = np.asarray(mesh.faces, dtype=np.int64)
    cells = np.hstack([np.full((len(f), 1), f.shape[1], np.int64), f]).ravel()
    pd = pv.PolyData(v, cells)
    pd.point_data.clear()
    pd.cell_data.clear()                      # drop any imported normals
    return pd


def mesh_variants(raw, center, smooth):
    """Lazily build the mesh as seen by each pane, keyed by reflection axis.

    None is the model itself; 0/1/2 is the model reflected along X/Y/Z about
    its own centre.  A reflection has determinant -1, so it turns every
    triangle inside out: flip_faces() restores the winding, and the normals are
    recomputed afterwards.  Skip either step and the part renders inside-out
    and almost black.
    """
    cache = {}

    def prepare(m):
        if not smooth:
            return m
        # below the feature angle normals are averaged (smooth), above it the
        # vertices are split so genuine CAD edges stay perfectly sharp
        return m.compute_normals(feature_angle=35.0, split_vertices=True,
                                 consistent_normals=True, auto_orient_normals=True)

    def get(k):
        if k not in cache:
            m = raw if k is None else \
                (raw.flip_x, raw.flip_y, raw.flip_z)[k](point=center).flip_faces()
            cache[k] = prepare(m)
        return cache[k]

    return get


def nice_step(target):
    """The 1/2/5 x 10^k step nearest `target`, floored at MIN_STEP.

    1, 2 and 5 are the multipliers whose multiples a reader adds up without
    arithmetic, and each of them lands on a terminating decimal -- which is what
    keeps a label inside MAX_DECIMALS instead of trailing off into rounding.
    Nearest rather than the largest that fits, so the line count lands near
    GRID_LINES from either side rather than only ever above it.
    """
    if not np.isfinite(target) or target <= 0:
        return MIN_STEP
    k = np.floor(np.log10(target))
    best = min((m * 10.0 ** k for m in (1.0, 2.0, 5.0, 10.0)),
               key=lambda c: abs(np.log(c / target)))
    return max(float(best), MIN_STEP)


def grid_steps(span, grid_step=None, tick_step=None):
    """(grid, tick) in model units: about GRID_LINES lines across `span`.

    Derived from the part rather than fixed, because a fixed step is only ever
    right for one size of part: at 1000 model units a part 1 unit across drew a
    single grid line, at the origin, and the axes carried no scale at all --
    which is the one thing a measured view has to carry.  It is measured once
    over the mesh's largest side and shared by all six panes, so a square of
    grid means the same distance wherever the reader is looking.
    """
    g = float(grid_step) if grid_step else nice_step(span / GRID_LINES)
    return g, (float(tick_step) if tick_step else g * TICK_RATIO)


def step_decimals(step):
    """Decimals needed to state a multiple of `step` exactly, MAX_DECIMALS at most.

    `nice_step` only ever returns 1/2/5 x 10^k and floors at MIN_STEP, so this
    is exact for anything it produced; a caller-supplied step of 1/3 is the case
    that hits the cap and gets rounded labels rather than a runaway one.
    """
    for d in range(MAX_DECIMALS + 1):
        scaled = step * 10.0 ** d
        if abs(scaled - round(scaled)) <= 1e-6:
            return d
    return MAX_DECIMALS


def marks(lo, hi, step):
    """Every multiple of `step` in [lo, hi].

    Multiples of the step, not offsets from `lo`: a grid line has to fall on a
    round number whatever the window starts at, or the labels come out at
    0.3991, 0.4991 and the grid stops being something a reader can count on.
    """
    first = np.ceil(lo / step - 1e-6) * step
    return np.arange(first, hi + step * 1e-6, step)


def label_width(lo, hi, step, decimals):
    """Characters in the widest label an axis over [lo, hi] will print.

    fmt_tick(hi) is not it: `hi` is the model's extent, rarely a multiple of
    the step, so 8.639 would measure as "8.639" and reserve room for five
    characters where the axis only ever prints "8.5".
    """
    got = marks(lo, hi, step)
    return max((len(fmt_tick(t, decimals)) for t in got), default=1)


def label_step(step, ppu, need):
    """Widen `step` until consecutive labels are `need` pixels apart."""
    for m in (1, 2, 5, 10, 20, 50):
        if step * m * ppu >= need:
            return step * m
    return step * 100


# ---------------------------------------------------------------- 2D overlay

def text_actor(text, x, y, size, color=INK, halign="center", valign="bottom",
               angle=0.0, bold=False):
    a = vtk.vtkTextActor()
    a.SetInput(text)
    a.SetPosition(x, y)
    p = a.GetTextProperty()
    p.SetFontFamilyToCourier()
    p.SetFontSize(int(round(size)))
    p.SetColor(*color)
    p.SetBold(bold)
    p.SetJustificationToLeft() if halign == "left" else (
        p.SetJustificationToRight() if halign == "right" else p.SetJustificationToCentered())
    p.SetVerticalJustificationToTop() if valign == "top" else (
        p.SetVerticalJustificationToCentered() if valign == "center"
        else p.SetVerticalJustificationToBottom())
    p.SetOrientation(angle)
    return a


def fmt_tick(t, decimals):
    """A grid line's model coordinate, printed plainly in model units.

    Model units, not a scaled-down stand-in for them: the numbers on the axis
    are the numbers a `section` or a `sketch` is given, and a label that has to
    be multiplied by anything before it can be typed back is a step at which
    the reader can be wrong.  `decimals` comes from the step, so a whole-number
    grid prints whole numbers and nothing prints a digit the step cannot place.
    """
    if abs(t) < MIN_STEP / 10:
        t = 0.0                                   # -0 is tick arithmetic
    return f"{t:.{decimals}f}"


def cover(i, c, width):
    """Exact coverage of pixel bins i (bin i spans [i, i+1]) by a band at c."""
    i = np.asarray(i, np.float32)
    return np.clip(np.minimum(c + width / 2, i + 1) -
                   np.maximum(c - width / 2, i), 0, 1)


def paint_vlines(img, xs, y0, y1, color, width, alpha):
    """Blend vertical bands at `xs`, spanning y0..y1, into a float RGB image.

    Coordinates are bottom-up pixels, `img` rows are top-down.  Coverage is
    computed exactly, so the lines are anti-aliased at the supersampled
    resolution before the pane is filtered back down -- and, unlike a
    vtkActor2D, a width below or above one pixel is honoured (OpenGL core
    profile clamps glLineWidth to 1).
    """
    H, W, _ = img.shape
    r0, r1 = int(round(H - y1)), int(round(H - y0))
    col = np.array(color, np.float32) * 255.0
    for x in xs:
        c0 = max(int(np.floor(x - width / 2)), 0)
        c1 = min(int(np.ceil(x + width / 2)) + 1, W)
        if c1 <= c0:
            continue
        w = (cover(np.arange(c0, c1), x, width) * alpha)[None, :, None]
        img[r0:r1, c0:c1] = img[r0:r1, c0:c1] * (1 - w) + col * w
    return img


def paint_hlines(img, ys, x0, x1, color, width, alpha):
    """paint_vlines, turned ninety degrees."""
    H, W, _ = img.shape
    c0, c1 = int(round(x0)), int(round(x1))
    col = np.array(color, np.float32) * 255.0
    for y in ys:
        r0 = max(int(np.floor(H - y - width / 2)), 0)
        r1 = min(int(np.ceil(H - y + width / 2)) + 1, H)
        if r1 <= r0:
            continue
        w = (cover(np.arange(r0, r1), H - y, width) * alpha)[:, None, None]
        img[r0:r1, c0:c1] = img[r0:r1, c0:c1] * (1 - w) + col * w
    return img


def label_renderer(rect, labels_h, labels_v, to_px, names, title, grid, top, u,
                   decimals):
    """Dark text on white: tick numbers, axis names and the title.

    Rendered on its own, at the pane's *final* size rather than supersampled.
    FreeType already anti-aliases and hints each glyph; box-filtering that down
    would anti-alias it a second time, and the hinting would have snapped the
    stems to the supersampled grid instead of the one they end up on -- which is
    exactly what makes downscaled text look muddy.  All of this lives in the
    margins, so it is multiplied over the finished pane (see render_pane).
    """
    ren = vtk.vtkRenderer()
    ren.SetLayer(0)
    ren.SetBackground(1, 1, 1)
    ren.InteractiveOff()
    x0, y0, x1, y1 = rect
    gap = (TICK_LEN + 3) * u                    # clear of the tick marks

    if grid:
        for t in labels_h:
            ren.AddViewProp(text_actor(fmt_tick(t, decimals), to_px(t, 0)[0], y0 - gap,
                                       FONT["tick"] * u, halign="center", valign="top"))
        for t in labels_v:
            ren.AddViewProp(text_actor(fmt_tick(t, decimals), x0 - gap, to_px(0, t)[1],
                                       FONT["tick"] * u, halign="right", valign="center"))
        ren.AddViewProp(text_actor(names[0], (x0 + x1) / 2,
                                   y0 - 31 * u, FONT["name"] * u,
                                   halign="center", valign="top"))
        ren.AddViewProp(text_actor(names[1], x0 - 48 * u,
                                   (y0 + y1) / 2, FONT["name"] * u,
                                   halign="center", valign="center", angle=90))

    ren.AddViewProp(text_actor(title, (x0 + x1) / 2, y1 + top / 2,
                               FONT["title"] * u, halign="center", valign="center"))
    return ren


# ------------------------------------------------------------------ 3D pane

def part_renderer(mesh, edges, cam_dir, up, box, center, kh, kv, rect, W, H, s,
                  opt, ssao_r, silhouette=False):
    """Layer-0 renderer holding the part, framed on [lo_h, hi_h] x [lo_v, hi_v].

    silhouette=True swaps the shading for flat white on black: an exact
    coverage mask of the part, in the very same pixels as the shaded render.
    """
    lo_h, hi_h, lo_v, hi_v = box
    ren = vtk.vtkRenderer()
    ren.SetLayer(0)
    ren.SetBackground(0, 0, 0) if silhouette else ren.SetBackground(1, 1, 1)
    ren.SetViewport(rect[0] / W, rect[1] / H, rect[2] / W, rect[3] / H)

    if silhouette:
        # a flat white stencil: no scalars, no lighting.  A colored part has to
        # mask the grid exactly as a grey one does, so this pass must not know
        # how the part is painted
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(mesh)
        mapper.ScalarVisibilityOff()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 1, 1)
        actor.GetProperty().SetLighting(False)
        ren.AddActor(actor)
    else:
        # one actor per material: the triangles left at the default grey are
        # drawn by the default grey material, so an uncolored part of a colored
        # pane is pixel for pixel what visualize_orthographic draws
        for part, material in split_by_material(mesh):
            ren.AddActor(part_actor(part, material, opt["smooth"]))
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
            # screen-space ambient occlusion darkens crevices, which is what
            # makes a shallow engraving readable on a face pointing straight at
            # the camera
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

    d = np.asarray(cam_dir, float)
    d = d / np.linalg.norm(d)
    radius = float(np.linalg.norm(np.ptp(np.array(mesh.bounds).reshape(3, 2), axis=1))) / 2
    # aim at the centre of the window, not at the centre of the mesh: the two
    # in-plane axes are pinned to [lo, hi], the depth axis only has to contain
    # the part, so it keeps the mesh centre
    focus = np.array(center, float)
    focus[kh], focus[kv] = (lo_h + hi_h) / 2, (lo_v + hi_v) / 2
    cam = ren.GetActiveCamera()
    cam.ParallelProjectionOn()                              # true orthographic
    cam.SetFocalPoint(*focus)
    cam.SetPosition(*(focus + d * max(radius, 1e-9) * 6))
    cam.SetViewUp(*np.asarray(up, float))
    cam.SetParallelScale((hi_v - lo_v) / 2)                 # window height / 2
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


def render_pane(mesh, edges, cam_dir, up, opt, box, center, kh, kv, title,
                ssao_r, geom, steps):
    """One pane: grid, then the part over it, then the labels -> RGB uint8.

    `box` is (lo_h, hi_h, lo_v, hi_v), the model window shown on each axis;
    `geom` is the pane's pixel layout from pane_geometry().
    """
    lo_h, hi_h, lo_v, hi_v = box
    s = geom["s"]                                 # px per REF-unit, supersampled
    W, H, pw, ph = geom["W"], geom["H"], geom["pw"], geom["ph"]
    rect = (geom["left"], geom["bottom"], geom["left"] + pw, geom["bottom"] + ph)

    # model coordinates -> window pixels; exact, because the projection is
    # parallel and parallel_scale is half the viewport height in model units
    ppu = ph / (hi_v - lo_v)

    def to_px(h, v):
        return (rect[0] + pw / 2 + (h - (lo_h + hi_h) / 2) * ppu,
                rect[1] + ph / 2 + (v - (lo_v + hi_v) / 2) * ppu)

    # grid lines every grid_step model units, tick marks every tick_step, and a
    # number on every grid line that has room for one
    u, k = geom["u"], geom["ss"]
    # Measured once over the whole mesh, so every pane's grid is the same grid
    # -- see grid_steps.  Deriving it per pane from that pane's own two axes
    # would give the thin views a coarser grid than the wide ones, and a square
    # of grid would stop meaning one distance.
    g_step, t_step, decimals = steps
    wide = (label_width(lo_h, hi_h, g_step, decimals) * 0.62 * FONT["tick"] + 8) * u
    tall = (FONT["tick"] + 5) * u

    def part(silhouette):
        return part_renderer(mesh, edges, cam_dir, up, box, center, kh, kv,
                             rect, W, H, s, opt, ssao_r, silhouette)

    arr = grab([part(False)], W, H).astype(np.float32)

    if opt["grid"]:
        back = np.full((H, W, 3), 255.0, np.float32)
        paint_vlines(back, [to_px(t, 0)[0] for t in marks(lo_h, hi_h, g_step)],
                     rect[1], rect[3], BLUE, 0.8 * s, 0.45)
        paint_hlines(back, [to_px(0, t)[1] for t in marks(lo_v, hi_v, g_step)],
                     rect[0], rect[2], BLUE, 0.8 * s, 0.45)
        # the grid shows through wherever the pane is bare background; the part,
        # a white specular highlight included, comes from its own silhouette pass
        keep = (grab([part(True)], W, H)[..., 0] > 127) | (arr != 255).any(-1)
        arr = np.where(keep[..., None], arr, back)

        # frame, and tick marks that only ever stick *out* of it
        paint_vlines(arr, [rect[0], rect[2]], rect[1], rect[3], BLUE, 1.1 * s, 0.6)
        paint_hlines(arr, [rect[1], rect[3]], rect[0], rect[2], BLUE, 1.1 * s, 0.6)
        paint_vlines(arr, [to_px(t, 0)[0] for t in marks(lo_h, hi_h, t_step)],
                     rect[1] - TICK_LEN * s, rect[1], BLUE, 1.0 * s, 0.6)
        paint_hlines(arr, [to_px(0, t)[1] for t in marks(lo_v, hi_v, t_step)],
                     rect[0] - TICK_LEN * s, rect[0], BLUE, 1.0 * s, 0.6)

    # box-filter the supersampled pane back down: an unweighted average over the
    # k x k block is the pixel's area integral, which is what supersampling is
    ny, nx = (H // k) * k, (W // k) * k
    arr = arr[:ny, :nx].reshape(ny // k, k, nx // k, k, 3).mean((1, 3))

    # the text goes on last, rasterised at 1:1 and multiplied over the pane
    labels = grab([label_renderer(tuple(c / k for c in rect),
                                  marks(lo_h, hi_h, label_step(g_step, ppu / k, wide)),
                                  marks(lo_v, hi_v, label_step(g_step, ppu / k, tall)),
                                  lambda h, v: tuple(c / k for c in to_px(h, v)),
                                  (AXIS[kh], AXIS[kv]), title, opt["grid"],
                                  geom["top"], u, decimals)],
                  W // k, H // k).astype(np.float32)
    return np.clip(arr * (labels / 255.0), 0, 255).astype(np.uint8)


# --------------------------------------------------------------- composition

def align_up(n, k=ALIGN):
    return int(np.ceil(n / k - 1e-9) * k)


def margins(ui):
    """Left and top margins, in final pixels and a multiple of ALIGN."""
    return align_up(MARGIN["left"] * ui), align_up(MARGIN["top"] * ui)


def pane_geometry(span_h, span_v, ppu, ui, ss):
    """Pixel layout of one pane: final pixels, scaled by `ss` for rendering.

    The left and top margins are rounded up to a multiple of ALIGN and the
    right and bottom ones absorb whatever the padding to ALIGN adds, so the
    plot's top-left corner sits at a multiple of ALIGN in the finished image --
    and so do the image's own dimensions.
    """
    left, top = margins(ui)
    pw, ph = int(span_h * ppu), int(span_v * ppu)
    W = align_up(left + pw + MARGIN["right"] * ui)
    H = align_up(top + ph + MARGIN["bottom"] * ui)
    return dict(ss=ss, s=ss * ui, u=ui, top=top, W=W * ss, H=H * ss,
                pw=pw * ss, ph=ph * ss,
                left=left * ss, bottom=(H - top - ph) * ss)


def pad_to(img, width, height):
    """White canvas of width x height, image in its top-left corner."""
    out = np.full((height, width, 3), 255, np.uint8)
    h, w, _ = img.shape
    out[:h, :w] = img
    return out


def _widen(lo, hi):
    """A window with width on every axis, whatever came in.

    A perfectly flat axis -- a plate, a mesh of one triangle -- would otherwise
    divide by zero the moment ppu is computed.
    """
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return lo, np.where(hi - lo < 1e-9, lo + 1e-9, hi)


def window_bounds(mesh, window=WINDOW_DEFAULT):
    """The [lo, hi] each axis is shown over, as two arrays of three.

    "origin" pins lo to zero, so anything at a negative coordinate falls outside
    the frame -- the caller is warned, but the geometry is gone.  "bounds" takes
    the mesh's own extent on each axis and cannot lose anything.  An explicit
    (lo, hi) pair is passed straight through, which is how two meshes are drawn
    on one window: see fit_window.
    """
    if not isinstance(window, str):
        try:
            lo, hi = window
        except (TypeError, ValueError):
            raise ValueError(f"window must be one of {WINDOW_MODES} or an "
                             f"(lo, hi) pair, got {window!r}") from None
        return _widen(lo, hi)
    if window not in WINDOW_MODES:
        raise ValueError(f"window must be one of {WINDOW_MODES} or an (lo, hi) "
                         f"pair, got {window!r}")
    b = np.array(to_polydata(mesh).bounds).reshape(3, 2)
    hi = np.array(b[:, 1], float)
    if window == "origin":
        return np.zeros(3), np.maximum(hi, 1e-9)
    return _widen(b[:, 0], hi)


def fit_window(meshes, window=WINDOW_DEFAULT):
    """One (lo, hi) window holding every one of them, to pass back as `window`.

    The orthographic counterpart of isometric_views.fit_extent.  Two meshes
    windowed separately are two different rulers: the same feature lands at a
    different pixel on each sheet, and a reader flipping between them reads the
    difference in framing as a difference in the parts.  Measuring the window
    over both is what makes the two sheets comparable -- and what lets a
    prediction that is slightly smaller than its target render slightly smaller,
    instead of being blown up to fill the same page.
    """
    los, his = zip(*(window_bounds(m, window) for m in meshes), strict=True)
    return _widen(np.min(los, axis=0), np.max(his, axis=0))


def pane_frames(mesh, up="Y", max_side=512, ss=SS, window=WINDOW_DEFAULT):
    """Where each pane will put the model, in that pane's own finished pixels.

    Returns {key: frame}: the two world axes the pane shows, the [0, hi] window
    it shows them over, its finished size, the plot rectangle inside that, and
    pixels per model unit.  `visualize_orthographic` calls this to lay itself
    out, so a caller that asks the same question gets the answer the picture was
    actually drawn from.

    It is public because a caller may want to put something else on the pane --
    a section level, say -- and it has to land on the number the axis prints
    beside it.  Deriving that mapping independently is the version that goes
    quietly wrong the first time a margin changes: the line moves two pixels off
    the tick it is supposed to name, and nothing anywhere says so.
    """
    lo, hi = window_bounds(mesh, window)
    extent = hi - lo
    views = six_views(up)
    ui = max_side / REF
    left, top = margins(ui)
    span_h = max(extent[v[3]] for v in views)
    span_v = max(extent[v[4]] for v in views)
    # one pixels-per-unit for all six panes: the largest pane hits max_side on
    # whichever of its two sides is the binding one, and nothing exceeds it
    budget = (int(max_side) // ALIGN) * ALIGN    # so the padding cannot overrun
    ppu = min((budget - left - MARGIN["right"] * ui) / span_h,
              (budget - top - MARGIN["bottom"] * ui) / span_v)
    if ppu <= 0:
        raise ValueError(f"max_side={max_side} is too small for the margins")

    frames = {}
    for name, cam, upv, kh, kv, k_reflect, face in views:
        geom = pane_geometry(extent[kh], extent[kv], ppu, ui, ss)
        frames[axis_label(face)] = dict(
            name=name, kh=kh, kv=kv,
            lo_h=float(lo[kh]), hi_h=float(hi[kh]),
            lo_v=float(lo[kv]), hi_v=float(hi[kv]),
            width=geom["W"] // ss, height=geom["H"] // ss,
            # bottom-up, like everything else that addresses a pane in pixels
            rect=(geom["left"] / ss, geom["bottom"] / ss,
                  (geom["left"] + geom["pw"]) / ss,
                  (geom["bottom"] + geom["ph"]) / ss),
            ppu=(geom["ph"] / ss) / max(float(extent[kv]), 1e-12),
            mirrored=k_reflect is not None,
            _view=(name, cam, upv, kh, kv, k_reflect, face), _geom=geom)
    return frames


def frame_row(frame, v):
    """Top-down pixel row of `v` on the pane's VERTICAL axis, for PIL."""
    x0, y0, x1, y1 = frame["rect"]
    mid = (frame["lo_v"] + frame["hi_v"]) / 2
    return frame["height"] - (y0 + (y1 - y0) / 2 + (v - mid) * frame["ppu"])


def frame_col(frame, h):
    """Left-to-right pixel column of `h` on the pane's HORIZONTAL axis."""
    x0, y0, x1, y1 = frame["rect"]
    mid = (frame["lo_h"] + frame["hi_h"]) / 2
    return x0 + (x1 - x0) / 2 + (h - mid) * frame["ppu"]


def visualize_orthographic(mesh, mosaic=True, max_side=512, ss=SS, up="Y",
                           grid_step=None, tick_step=None, edl=True,
                           ssao=False, edges=False, grid=True, smooth=True,
                           face_colors=None, window=WINDOW_DEFAULT):
    """Render the six axis-aligned orthographic views of a trimesh mesh.

    mesh      trimesh.Trimesh (anything with .vertices/.faces), or a PolyData
    mosaic    True  -> one 2x3 montage image (rows +/-, columns X/Y/Z pairs)
              False -> dict of six panes, keyed "+X", "-X", "+Y", ... "-Z"
    max_side  the pane with the largest spans is this many pixels on its longer
              side, margins/ticks/title included; every other pane is smaller by
              the same pixels-per-model-unit, so all six share one scale.
              Rounded down to a multiple of 32 so the padding below cannot push
              a pane past it
    ss        supersampling factor: every pane is rendered `ss` times larger
              and box-filtered back down, so `ss` x `ss` samples land in each
              output pixel.  Costs ss^2 in pixels rendered
    grid_step model units between grid lines, tick_step between the tick marks
              outside the frame.  Both default to None, which measures them off
              the mesh: about GRID_LINES lines across its largest side, rounded
              to a 1/2/5 x 10^k step, with the ticks at half of that.  Pass a
              number to override either one
    face_colors  (n_triangles, 3) uint8, one RGB per triangle, in the mesh's own
              triangle order.  None renders the single grey material.
              `visualize_orthographic_clusters` builds this from a list of
              clusters; `cluster_face_colors` is the same builder on its own.
              Triangles left at UNASSIGNED are drawn with the grey material,
              identically to a plain `visualize_orthographic` of the same part
    window    "origin" (default): each axis runs [0, max], the origin in the
              lower-left corner, so the picture says where the part sits in
              model space -- at the cost of the frame between the origin and the
              part, which for a part that does not reach it is most of the pane.
              "bounds": each axis runs [min, max], framing the part itself.  Use
              it when detail matters more than absolute position, and whenever
              the part reaches a negative coordinate, which "origin" puts
              outside the frame entirely.  An explicit (lo, hi) pair of
              three-vectors draws that exact window -- `fit_window` builds one
              spanning several meshes, so two of them can be drawn to one ruler.
              Tick numbers are true model coordinates in every case

    Returns (images, titles): `images` is the montage or the dict of panes,
    `titles` is always a dict of the six one-line pane titles.  Image
    dimensions, and the margins left of and above every plot, are all multiples
    of 32 px; the white padding that buys that goes on the right and the bottom.
    """
    opt = dict(edl=edl, ssao=ssao, edges=edges, grid=grid, smooth=smooth)
    raw = to_polydata(mesh)
    if face_colors is not None:
        if raw is mesh:
            raw = raw.copy()          # do not stamp arrays onto the caller's
        rgb = np.ascontiguousarray(face_colors, dtype=np.uint8)
        if rgb.shape != (raw.n_cells, 3):
            raise ValueError(f"face_colors is {rgb.shape}, expected "
                             f"({raw.n_cells}, 3) - one RGB per triangle")
        raw.cell_data[RGB_ARRAY] = rgb
        # the arrays ride along through the reflections and the normals pass,
        # both of which keep cell order, so a mirrored pane colors the same
        grey = (rgb == np.asarray(UNASSIGNED, dtype=np.uint8)).all(axis=1)
        if grey.any():
            raw.cell_data[GREY_ARRAY] = grey.astype(np.uint8)
    bnds = np.array(raw.bounds).reshape(3, 2)
    center = np.array(raw.center)
    ssao_r = float(np.linalg.norm(np.ptp(bnds, axis=1))) / 2 * 0.05

    lo, hi = window_bounds(raw, window)
    if window == "origin" and (bnds[:, 0] < -1e-9 * hi.max()).any():
        print(f"warning: geometry reaches {bnds[:, 0].min():.4g} < 0 and is "
              f"clipped by the [0, max] window; window='bounds' frames it "
              f"instead", file=sys.stderr)

    variant = mesh_variants(raw, center, smooth)
    # The layout is decided in one place, and a caller drawing over a pane asks
    # that same place where the model landed -- see pane_frames.
    frames = pane_frames(raw, up, max_side, ss, window)

    # One grid for all six panes, off the mesh's largest side
    g_step, t_step = grid_steps(float(np.max(hi - lo)), grid_step, tick_step)
    steps = (g_step, t_step, step_decimals(g_step))

    panes, titles, order = {}, {k: "" for k in KEYS}, []
    for key in (axis_label(v[6]) for v in six_views(up)):
        frame = frames[key]
        name, cam, upv, kh, kv, k_reflect, face = frame["_view"]
        order.append(key)
        mirrored_info = ""  # (mirrored)
        titles[key] = (f"{name} {key}{f'  {mirrored_info}' if k_reflect is not None else ''}"
                       f"   ({AXIS[kh]} right, {AXIS[kv]} up)")
        m = variant(k_reflect)
        e = None
        if edges:
            e = m.extract_feature_edges(feature_angle=25.0,
                                        boundary_edges=True, feature_edges=True,
                                        non_manifold_edges=False, manifold_edges=False)
        panes[key] = render_pane(m, e, cam, upv, opt,
                                 (lo[kh], hi[kh], lo[kv], hi[kv]), center,
                                 kh, kv, titles[key], ssao_r, frame["_geom"],
                                 steps)

    if not mosaic:
        return {key: panes[key] for key in KEYS}, titles

    # panes differ in aspect ratio, so a row is as tall as its tallest pane and
    # every pane sits at its top: that keeps each plot's offset within the
    # mosaic a multiple of ALIGN, exactly as it is within its own pane
    row_h = max(p.shape[0] for p in panes.values())
    cells = [pad_to(panes[k], panes[k].shape[1], row_h) for k in order]
    return np.vstack([np.hstack(cells[0:3]), np.hstack(cells[3:6])]), titles


def visualize_orthographic_clusters(mesh, clusters, colors=None,
                                   unassigned=UNASSIGNED, **kwargs):
    """The six orthographic views, with each cluster of triangles in its color.

    Same panes, grid, ticks, scale and titles as `visualize_orthographic` -
    this only supplies the per-triangle colors and forwards everything else -
    so a colored sheet and a grey one of the same part line up exactly.

    mesh        trimesh.Trimesh (anything with .vertices/.faces), or a PolyData
    clusters    a sequence of triangle-index arrays, one per cluster, with a
                parallel `colors`; or a sequence of (triangles, color) pairs
                and no `colors`.  Indices are into the mesh's own triangles,
                which is what `segment_mesh.find_faces` returns
    colors      one color per cluster: an RGB triple, a "#rrggbb" string, a
                name from isometric_views.NAMED_RGB, or anything carrying its
                own `.rgb`
    unassigned  color for triangles no cluster claimed (default light grey)
    **kwargs    passed to `visualize_orthographic`: mosaic, max_side, ss, up,
                grid_step, tick_step, edl, ssao, edges, grid, smooth, window

    Returns (images, titles), exactly as `visualize_orthographic` does.

    A whole `segment_mesh.FaceGraph` renders as

        visualize_orthographic_clusters(
            graph.mesh, [(c.triangles, c.color) for c in graph.clusters])

    since a `Color` carries its own `.rgb`.
    """
    faces = np.asarray(mesh.faces)
    n_faces = len(faces) if faces.ndim == 2 else int(mesh.n_cells)
    rgb = cluster_face_colors(n_faces, clusters, colors, unassigned)
    return visualize_orthographic(mesh, face_colors=rgb, **kwargs)


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

    src, dst, opt = parse_args(sys.argv)
    mesh = trimesh.load(src, force="mesh")
    images, titles = visualize_orthographic(mesh, **opt)

    size = mesh.extents
    print(f"{src}: {len(mesh.faces)} triangles, "
          f"bbox {size[0]:.4g} x {size[1]:.4g} x {size[2]:.4g}")
    if opt["mosaic"]:
        write_png(dst, images)
        print(f"  -> {dst} ({images.shape[1]}x{images.shape[0]})")
    else:
        stem = dst[:-4] if dst.lower().endswith(".png") else dst
        for key, img in images.items():
            path = f"{stem}_{key}.png"
            write_png(path, img)
            print(f"  -> {path} ({img.shape[1]}x{img.shape[0]})  {titles[key]}")


if __name__ == "__main__":
    main()
