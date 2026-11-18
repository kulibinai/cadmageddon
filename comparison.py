"""Two meshes on one ruler: where a predicted surface leaves the target one.

`comparative_mesh_vis(target_mesh, predict_mesh)` returns four images -- the
four isometric views of the target, the same four of the prediction, the six
orthographic views of the target, the same six of the prediction -- with every
triangle that sits further than a threshold from the *other* mesh painted in
the color of the worst threshold it clears.

It is a thin wrapper.  The distances decide a per-triangle color array and that
array is handed to `visualize_isometric(face_colors=)` and
`visualize_orthographic(face_colors=)`, so camera, framing, header, grid, ticks
and every pixel decision are the ones those two scripts already make and cannot
drift from them.  A triangle inside the tightest threshold keeps the plain grey
those scripts draw a whole part in, drawn by the same material, so the matching
part of a comparison sheet is pixel for pixel the ordinary view.

Distance.  For each triangle of the mesh being drawn, the unsigned distance
from its centroid to the closest point of the other mesh's *surface* -- VTK's
vtkImplicitPolyDataDistance through a cell locator, so it is a true
point-to-triangle distance and not a distance to the nearest vertex.  Sign is
dropped: a surface pushed out and one pushed in are the same error here, and
the sign is meaningless anyway on a mesh that is not closed.

One centroid per triangle is the whole measurement, which is why both meshes
are drawn rather than just the prediction.  A triangle much longer than the
deviation can sit with its centroid inside a threshold and its ends outside it;
CAD tessellation makes plenty of those (this project's parts cut cylinders into
strips of about 200:1), so a long strip reports the error at its middle only.
And the measurement is one-sided per sheet: geometry the prediction *left out*
has no triangle of its own to paint, and shows up only on the target sheet,
while geometry it *invented* shows up only on the prediction sheet.  The two
sheets together are what makes a difference visible, whichever way it went.

Thresholds are fractions of the largest axis span of the two meshes' shared
bounding box -- by default 0.1% and 1%.  Measuring the span over both
meshes, rather than each mesh alone, is what lets the two sheets be compared:
the same physical error falls in the same band on both.  The bands are nested,
and the worst one wins, so the last band means "more than 1% out", not "between
1% and the next band".

The paint differs by mesh even though the bands do not: the target sheets cool
lime -> dark green, the prediction sheets warm orange -> red.  A pane therefore
says which mesh it is of on its own, which matters because the two sheets are
otherwise framed and scaled to be interchangeable -- they are meant to be
flipped between, and a screenshot of one out of context would otherwise be
unattributable.  `target_colors` and `predict_colors` override both.

Both sheets are also drawn at one scale: the isometric views share a framing
(`fit_extent`, `uniform=True`) and the orthographic views share a window
(`fit_window`), so a part that is slightly smaller than its counterpart renders
slightly smaller instead of being blown up to the same page.  Pass
`shared=False` to frame each mesh on its own, which is tighter and not
comparable.

usage:
  python comparison.py TARGET.stl PREDICT.stl OUTDIR [options]
    writes target_iso.png, predict_iso.png, target_ortho.png,
    predict_ortho.png, and prints how many triangles fall in each band and what
    share of the surface AREA that is: dense tessellation (hole walls, fillets)
    makes triangle counts overstate small features (a part with only its holes
    missing can read as mostly off by count)

    --max-side N    longest side of one isometric pane / the largest
                    orthographic pane, in px (default 512)
    --ss N          supersampling factor (default 4; costs N^2 in pixels)
    --fractions A,B  the thresholds, as fractions of the max axis span
                    (default 0.001,0.01).  The target sheets paint them lime and
                    dark green, the prediction sheets orange and red;
                    `comparative_mesh_vis` takes both palettes
    --mm A,B        the thresholds in model units (mm) instead.  On a large part
                    0.1% of the span is coarse and can make a wrong build look
                    clean; the run says so when the tight band is over COARSE_MM
    --grid-step N   model units between orthographic grid lines, and
    --tick-step N   between its tick marks.  Both default to measuring
                    themselves off the parts
    --window MODE   bounds (default here) or origin, as orthographic_views takes
                    it; measured over both meshes so the sheets stay comparable.
                    origin frames [0, max] and shows only the +X/+Y quadrant of a
                    part centred on the origin (hiding three quarters of it), so
                    the command line frames the parts' bounds
    --up X|Y|Z      which model axis is "up" in the isometric views (default Z)
    --ortho-up X|Y|Z   the same for the orthographic views (default Y)
    --separate      also write the panes one PNG each
    --no-shared     frame each mesh on its own instead of on one shared scale
"""
import os
import sys

import numpy as np
import pyvista as pv  # mesh IO / locators only; it is VTK

import isometric_views as iso
import orthographic_views as ortho
from isometric_views import UNASSIGNED, cluster_face_colors, write_png

#: The two deviation bands, as fractions of the largest axis span.  Ordered
#: loosest first: a triangle clears both at once and `cluster_face_colors`
#: gives it the color of the last cluster that names it, which is the worst
#: band it clears.
#:
#: The 0.01% band that used to sit below these is gone.  At that distance a
#: triangle is reporting the tessellation, not the reconstruction - a chord
#: through a curve leaves the chord's own sagitta between the two meshes even
#: where they are the same surface - so it painted most of every curved face in
#: a third colour that meant "these agree".  Two bands, and grey means agreement.
FRACTIONS = (0.001, 0.01)                               # 0.1%, 1%

#: The color for a triangle further out than each band, one palette per mesh.
#: The bands are the same distances on both sheets; only the paint differs, so
#: that a pane on its own says which mesh it came from - the target cools
#: through lime to dark green, the prediction warms through orange to red.
TARGET_COLORS = ("lime", "dark green")
PREDICT_COLORS = ("orange", "red")
BAND_COLORS = TARGET_COLORS              # the default of `deviation_colors`

KEYS = ("target_iso", "predict_iso", "target_ortho", "predict_ortho")

#: The command line's window. `comparative_mesh_vis` keeps orthographic_views'
#: own default for its library callers (mesh2cad.render passes its own).
CLI_WINDOW = "bounds"

#: A tight band wider than this (model units) gets a note suggesting --mm.
COARSE_MM = 0.5


def parse_args(argv):
    if len(argv) < 4:
        raise SystemExit("usage: " + __doc__.strip().rsplit("usage:", 1)[-1])
    target, predict, out = argv[1], argv[2], argv[3]
    opt = dict(max_side=512, ss=iso.SS, up="Z", ortho_up="Y", shared=True,
               fractions=FRACTIONS, grid_step=None, tick_step=None,
               window=CLI_WINDOW, bands_mm=None)
    separate = False
    i = 4
    while i < len(argv):
        a = argv[i]
        if a == "--separate":
            separate = True
        elif a == "--no-shared":
            opt["shared"] = False
        elif a in ("--max-side", "--ss", "--up", "--ortho-up", "--fractions",
                   "--grid-step", "--tick-step", "--window", "--mm"):
            i += 1
            v = argv[i]
            if a == "--max-side":
                opt["max_side"] = float(v)
            elif a == "--ss":
                opt["ss"] = int(v)
            elif a == "--up":
                opt["up"] = v.upper()
            elif a == "--ortho-up":
                opt["ortho_up"] = v.upper()
            elif a == "--grid-step":
                opt["grid_step"] = float(v)
            elif a == "--tick-step":
                opt["tick_step"] = float(v)
            elif a == "--window":
                opt["window"] = v.lower()
            elif a == "--mm":
                opt["bands_mm"] = tuple(float(x) for x in v.split(","))
            else:
                opt["fractions"] = tuple(float(x) for x in v.split(","))
        else:
            raise SystemExit(f"unknown option {a}")
        i += 1
    return target, predict, out, opt, separate


def as_polydata(mesh):
    """Anything mesh-shaped -> a pv.PolyData carrying no arrays of its own.

    A copy, always: `isometric_views.to_polydata` clears the point and cell
    data of a PolyData handed to it and then stamps the color array onto it,
    and neither belongs on the caller's mesh.
    """
    if isinstance(mesh, pv.PolyData):
        mesh = mesh.copy()
    return iso.to_polydata(mesh)


def max_axis_span(meshes):
    """The largest single-axis extent over all of them, in model units.

    Over *all* of them, so a pair of meshes gets one ruler even when one is
    slightly larger than the other - which is what makes the same physical
    deviation the same color on both sheets.
    """
    span = 0.0
    for m in meshes:
        b = np.array(m.bounds).reshape(3, 2)
        span = max(span, float(np.ptp(b, axis=1).max()))
    return span


def surface_distance(mesh, other):
    """Unsigned distance from each triangle's centroid to `other`'s surface.

    Returns one distance per cell of `mesh`, in its own cell order.  VTK's
    vtkImplicitPolyDataDistance walks a cell locator, so this is the distance
    to the closest point of the closest *triangle* of `other`, not to its
    closest vertex.  Its sign says which side of `other` the point is on, which
    needs `other` to be closed and oriented to mean anything; the magnitude
    does not, so the sign is dropped here.
    """
    centers = mesh.cell_centers()
    d = centers.compute_implicit_distance(other)["implicit_distance"]
    return np.abs(np.asarray(d, dtype=float))


def deviation_colors(distances, thresholds, colors=BAND_COLORS,
                     unassigned=UNASSIGNED):
    """Per-triangle RGB for one mesh's distances: worst band a triangle clears.

    `thresholds` are in model units and must be increasing; a triangle strictly
    further out than the k-th takes the k-th color, and one inside the first
    keeps `unassigned`, the grey the plain views draw the whole part in.
    """
    thresholds = [float(t) for t in thresholds]
    if len(thresholds) != len(colors):
        raise ValueError(f"{len(thresholds)} thresholds but {len(colors)} "
                         f"colors")
    if any(b <= a for a, b in zip(thresholds, thresholds[1:])):
        raise ValueError(f"thresholds must increase, got {thresholds}")
    distances = np.asarray(distances, dtype=float)
    bands = [np.flatnonzero(distances > t) for t in thresholds]
    return cluster_face_colors(len(distances), bands, colors, unassigned)


def band_counts(distances, thresholds):
    """How many triangles land in each band, loosest first, plus the matched.

    Counts are of the band a triangle is *painted* in - the worst one it clears
    - so they and the "within" count partition the mesh.
    """
    distances = np.asarray(distances, dtype=float)
    over = [int((distances > t).sum()) for t in thresholds]
    return [int(len(distances) - (over[0] if over else 0))] + \
           [a - b for a, b in zip(over, over[1:])] + over[-1:]


def band_areas(distances, areas, thresholds):
    """The same partition as `band_counts`, weighted by triangle area, as
    fractions of the mesh's surface."""
    distances = np.asarray(distances, dtype=float)
    areas = np.asarray(areas, dtype=float)
    total = float(areas.sum()) or 1.0
    over = [float(areas[distances > t].sum()) for t in thresholds]
    within = total - (over[0] if over else 0.0)
    return [x / total for x in
            [within] + [a - b for a, b in zip(over, over[1:])] + over[-1:]]


def cell_areas(mesh):
    return np.asarray(mesh.compute_cell_sizes(length=False, area=True,
                                              volume=False)["Area"], float)


def thresholds_for(span, fractions=FRACTIONS, bands_mm=None):
    """The band edges in model units: `bands_mm` as given, else fractions of
    the span."""
    t = ([float(x) for x in bands_mm] if bands_mm is not None
         else [float(f) * span for f in fractions])
    if any(b <= a for a, b in zip(t, t[1:])):
        raise ValueError(f"thresholds must increase, got {t}")
    return t


def comparative_mesh_vis(target_mesh, predict_mesh, fractions=FRACTIONS,
                         target_colors=TARGET_COLORS,
                         predict_colors=PREDICT_COLORS, shared=True,
                         mosaic=True, max_side=512, ss=iso.SS, up="Z",
                         ortho_up="Y", window=ortho.WINDOW_DEFAULT,
                         iso_kwargs=None, ortho_kwargs=None, bands_mm=None):
    """Four sheets comparing two meshes, with the deviations painted on.

    target_mesh, predict_mesh
              trimesh.Trimesh (anything with .vertices/.faces), or PolyData.
              They are compared both ways round: every triangle of each is
              measured against the other's surface, and each gets its own two
              sheets, because a difference only paints on the mesh that has
              triangles there (see the module docstring)
    fractions the thresholds, as fractions of the largest axis span of the two
              meshes together.  Increasing; one per color
    target_colors, predict_colors
              the color for a triangle past each threshold, one palette per
              sheet: an RGB triple, a "#rrggbb" string, a name from
              isometric_views.NAMED_RGB, or anything carrying its own `.rgb`.
              The worst band a triangle clears is the one it is painted in.
              The two palettes differ so a pane says which mesh it is of
              without its filename; the bands themselves are the same
              distances on both.  Pass the same tuple twice to paint them alike
    shared    True  -> both meshes are framed and scaled alike, so the two
              isometric sheets and the two orthographic sheets can be flipped
              between.  This costs the isometric sheets their per-view framing
              (`uniform=True`), so a view whose silhouette is smaller than the
              largest gets white space around it
              False -> each mesh is framed on its own, tighter but not
              comparable in size
    mosaic    True  -> each sheet is one montage image (2x2 isometric,
              2x3 orthographic).  False -> each is the dict of its panes,
              keyed as isometric_views.KEYS and orthographic_views.KEYS
    max_side, ss, up
              passed to both renderers; `ortho_up` is the up axis of the
              orthographic sheets, whose natural default differs from the
              isometric one
    window    how the orthographic sheets window each axis, exactly as
              `orthographic_views.visualize_orthographic` takes it: "origin"
              ([0, max], the default), "bounds" ([min, max], framing the parts
              rather than their distance from the origin), or an explicit
              (lo, hi) pair.  Under `shared` it is measured over BOTH meshes
              (`fit_window`) whichever mode is asked for, so the two sheets stay
              one ruler
    bands_mm  the thresholds in model units instead of `fractions`; one per
              color, increasing
    iso_kwargs, ortho_kwargs
              anything else for `visualize_isometric` / `visualize_orthographic`
              - edl, ssao, edges, smooth, and for the orthographic sheets grid,
              grid_step and tick_step.  Both step defaults are None, which
              measures them off the parts, so a mesh one unit across and one ten
              thousand across are each gridded

    Returns a dict of the four images, keyed "target_iso", "predict_iso",
    "target_ortho", "predict_ortho" - the montages, or the dicts of panes.
    """
    for name, colors in (("target_colors", target_colors),
                         ("predict_colors", predict_colors)):
        n = len(bands_mm) if bands_mm is not None else len(fractions)
        if n != len(colors):
            raise ValueError(f"{n} thresholds but {len(colors)} {name}")
    target, predict = as_polydata(target_mesh), as_polydata(predict_mesh)

    span = max_axis_span([target, predict])
    thresholds = thresholds_for(span, fractions, bands_mm)
    face_colors = {
        "target": deviation_colors(surface_distance(target, predict),
                                   thresholds, target_colors),
        "predict": deviation_colors(surface_distance(predict, target),
                                    thresholds, predict_colors),
    }

    iso_opt = dict(mosaic=mosaic, max_side=max_side, ss=ss, up=up)
    iso_opt.update(iso_kwargs or {})
    ortho_opt = dict(mosaic=mosaic, max_side=max_side, ss=ss, up=ortho_up)
    ortho_opt.update(ortho_kwargs or {})
    if shared:
        # one framing and one window for both meshes, so the same feature is
        # the same size and in the same place on either sheet
        iso_opt.setdefault("uniform", True)
        iso_opt.setdefault("extent", iso.fit_extent([target, predict],
                                                    iso.iso_views(iso_opt["up"])))
        ortho_opt.setdefault("window",
                             ortho.fit_window([target, predict], window))
    else:
        ortho_opt.setdefault("window", window)

    out = {}
    for name, mesh in (("target", target), ("predict", predict)):
        rgb = face_colors[name]
        out[f"{name}_iso"] = iso.visualize_isometric(
            mesh.copy(), face_colors=rgb, **iso_opt)
        out[f"{name}_ortho"] = ortho.visualize_orthographic(
            mesh.copy(), face_colors=rgb, **ortho_opt)[0]     # drop the titles
    return out


def main():
    target_path, predict_path, out, opt, separate = parse_args(sys.argv)
    target = pv.read(target_path)
    predict = pv.read(predict_path)

    span = max_axis_span([target, predict])
    thresholds = thresholds_for(span, opt["fractions"], opt["bands_mm"])
    if opt["bands_mm"] is not None:
        print(f"max axis span {span:.6g}; thresholds (given in mm) "
              + ", ".join(f"{t:.4g}" for t in thresholds))
    else:
        print(f"max axis span {span:.6g}; thresholds "
              + ", ".join(f"{f * 100:g}% = {t:.4g}"
                          for f, t in zip(opt["fractions"], thresholds)))
        if thresholds[0] > COARSE_MM:
            print(f"  note: the tight band is {thresholds[0]:.3g} mm on this part; "
                  f"errors below that show as grey. --mm sets the bands in mm "
                  f"(e.g. --mm 0.05,0.5)")
    print(f"  window {opt['window']}")
    for name, a, b, colors in (("target", target, predict, TARGET_COLORS),
                               ("predict", predict, target, PREDICT_COLORS)):
        d = surface_distance(a, b)
        counts = band_counts(d, thresholds)
        shares = band_areas(d, cell_areas(a), thresholds)
        print(f"  {name:<7} {a.n_cells} triangles: "
              + ", ".join(f"{n} {label}" for n, label in
                          zip(counts, ["within"] + list(colors))))
        print(f"  {'':<7} by area: "
              + ", ".join(f"{100 * x:.2f}% {label}" for x, label in
                          zip(shares, ["within"] + list(colors))))

    kwargs = {k: opt[k] for k in ("fractions", "shared", "max_side", "ss",
                                  "up", "ortho_up", "window", "bands_mm")}
    ortho_kwargs = dict(grid_step=opt["grid_step"], tick_step=opt["tick_step"])

    os.makedirs(out, exist_ok=True)
    images = comparative_mesh_vis(target, predict, ortho_kwargs=ortho_kwargs,
                                  **kwargs)
    for key in KEYS:
        path = os.path.join(out, f"{key}.png")
        write_png(path, images[key])
        print(f"  -> {path} ({images[key].shape[1]}x{images[key].shape[0]})")

    if separate:
        panes = comparative_mesh_vis(target, predict, mosaic=False,
                                     ortho_kwargs=ortho_kwargs, **kwargs)
        for key in KEYS:
            for name, img in panes[key].items():
                path = os.path.join(out, f"{key}_{name}.png")
                write_png(path, img)
                print(f"  -> {path} ({img.shape[1]}x{img.shape[0]})")


if __name__ == "__main__":
    main()
