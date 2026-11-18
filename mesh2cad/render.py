"""Stage 1: the overview render.

Design constraints, in priority order:
  1. Every face of the part must be visible in at least one view.
  2. Planar faces must read flat and cylindrical faces must read curved, so the
     model can tell a chamfer from a fillet.
  3. No triangle wireframe -- tessellation noise is not part information.
  4. Absolute scale must be readable without measuring pixels.

(1) is why the default view set is two views from above and two from below
rather than the conventional four upper isometrics: four upper views never see
a blind pocket in the underside, and the model has no way to know it is missing.
(2) is crease-angle shading -- flat shading everywhere facets cylinders, smooth
shading everywhere melts chamfers. (4) is the graduated bounds box plus caption.
"""

from dataclasses import dataclass, field

import numpy as np
import pyvista as pv

# Four isometric views at ONE elevation, differing only in azimuth: 90 degree
# steps of a rotation about Z.
#
# The earlier set covered all eight octants, two views from below, which sees
# every face but is unreadable. The camera is always re-uprighted to up=+Z
# rather than rolled rigidly, so a view from below is not the rigid rotation of
# a view from above: two of the three axes project identically and the third
# reverses, which is exactly what a mirror does. Readers -- correctly -- read
# those pairs as mirrored and then find the geometry contradicts it.
#
# With a single elevation every view IS a rigid rotation of its neighbour about
# the vertical, so up=+Z is the honest up in all four and nothing is ambiguous.
# The cost is that the underside is never shown; `underside=True` renders the
# mirrored set for parts that need it, as a separate, explicitly labelled pass.
_EL = 0.8165        # tan of 39.2 deg; near-isometric without hiding vertical faces
VIEWS = (
    ("from +X +Y", (1, 1, _EL)),
    ("from -X +Y", (-1, 1, _EL)),
    ("from -X -Y", (-1, -1, _EL)),
    ("from +X -Y", (1, -1, _EL)),
)
VIEWS_UNDER = tuple((lab.replace("from", "under"), (d[0], d[1], -d[2]))
                    for lab, d in VIEWS)

CLAY = dict(
    color="#c8ccd2",
    specular=0.12,
    specular_power=24,
    ambient=0.28,
    diffuse=0.78,
    smooth_shading=True,
    split_sharp_edges=True,
    feature_angle=30.0,
)


@dataclass
class RenderConfig:
    tile: int = 760           # px per view, before compositing
    background: str = "white"
    feature_angle: float = 30.0
    ssao: bool = True
    # Graduated axes box. Off by default: it is the only honest way to read
    # absolute position off the image, but in practice its rotated tick labels
    # collide with each other and its wireframe crosses the part, which costs
    # more legibility than the numbers buy. Dimensions live in the caption
    # instead, and orientation comes from the gnomon.
    bounds_box: bool = False
    gnomon: bool = True
    feature_edges: bool = True
    edge_color: str = "#22262c"
    edge_width: float = 2.4
    views: tuple = VIEWS
    zoom: float = 0.88        # <1 leaves margin for the bounds box labels
    lights: bool = True
    _unused: dict = field(default_factory=dict)


def _add_lighting(p):
    """Three-point rig: cool key from upper left, warm fill, dim rim.

    A single headlight flattens coplanar faces at different depths into the same
    grey, which is exactly the discrimination we need for reading steps.
    """
    p.remove_all_lights()
    # Camera-relative so all four tiles shade identically and a face's grey means
    # the same thing in every view -- with world-fixed lights the two lower views
    # come out markedly darker than the upper pair and read as a different part.
    for pos, intensity, color in (
        ((-0.7, -0.5, 1.0), 1.05, "#ffffff"),   # key, upper left
        ((0.9, 0.3, 0.45), 0.55, "#fff4e6"),    # warm fill, opposite side
        ((0.1, 0.9, -0.5), 0.35, "#e2eaff"),    # cool rim from behind
    ):
        light = pv.Light(position=pos, light_type="camera light", intensity=intensity)
        light.diffuse_color = color
        p.add_light(light)


def _feature_edges(poly, angle):
    """Crease + boundary edges only -- never the full triangle wireframe.

    The brief said "without edges", which is right about tessellation edges but
    wrong about feature edges: a thin crease overlay is the single biggest
    legibility win on a shaded grey part, because it pins down exactly where a
    face ends when two adjacent faces happen to shade to similar values.
    """
    return poly.extract_feature_edges(
        feature_angle=angle,
        boundary_edges=True,
        feature_edges=True,
        non_manifold_edges=False,
        manifold_edges=False,
    )


def render_view(poly, direction, cfg: RenderConfig, bounds=None,
                mesh=None, plane=None, offset=None):
    """Render one orthographic view, returned as an RGB array."""
    p = pv.Plotter(off_screen=True, window_size=(cfg.tile, cfg.tile))
    p.background_color = cfg.background

    style = dict(CLAY, feature_angle=cfg.feature_angle)
    p.add_mesh(poly, **style)

    if cfg.feature_edges:
        edges = _feature_edges(poly, cfg.feature_angle)
        if edges.n_cells:
            actor = p.add_mesh(edges, color=cfg.edge_color, line_width=cfg.edge_width,
                               render_lines_as_tubes=False, lighting=False)
            # The edges lie exactly on the surface they came from, so without a
            # depth bias they lose the depth test to their own faces and vanish
            # -- the lines are drawn, just never seen.
            mapper = actor.GetMapper()
            mapper.SetResolveCoincidentTopologyToPolygonOffset()
            mapper.SetRelativeCoincidentTopologyLineOffsetParameters(-1.0, -6.0)

    if plane is not None and offset is not None and mesh is not None:
        _add_section_plane(p, mesh, plane, offset)

    if cfg.lights:
        _add_lighting(p)
    if cfg.ssao:
        diag = float(np.linalg.norm(poly.bounds[1::2] - np.array(poly.bounds[0::2])))
        p.enable_ssao(radius=diag * 0.08, bias=0.005, blur=True)
    p.enable_anti_aliasing("fxaa")

    if cfg.bounds_box:
        p.show_bounds(
            grid="back", location="outer", ticks="both",
            n_xlabels=3, n_ylabels=3, n_zlabels=3,
            xtitle="X", ytitle="Y", ztitle="Z",
            font_size=13, color="#5a6068", use_3d_text=False,
            padding=0.0, bounds=bounds,
        )

    # The triad is drawn after the screenshot, in pixel space, rather than by
    # add_axes -- see draw_gnomon for why the widget is not sufficient.

    # Parallel projection: sizes stay comparable across the part, which is what
    # makes visual proportion estimates usable at all.
    p.camera.SetParallelProjection(True)
    p.camera_position = [tuple(direction), (0, 0, 0), (0, 0, 1)]
    p.reset_camera()
    p.camera.Zoom(cfg.zoom)

    img = p.screenshot(return_img=True)
    p.close()
    if cfg.gnomon:
        img = draw_gnomon(img, direction)
    return img


def draw_gnomon(img, direction, up=(0.0, 0.0, 1.0), size=None, margin=18,
                corner="bl"):
    """Draw the axis triad ourselves, in pixel space, with a depth cue.

    Two things the VTK widget cannot do that matter here: keep the labels from
    landing on top of each other when two axes project to the same side, and
    show which axis points toward the viewer. Without the second, two different
    cameras can produce an identical triad -- the ambiguity that made these
    images misread.
    """
    from PIL import Image, ImageDraw, ImageFont

    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil, "RGBA")
    W, H = pil.size
    size = size or int(min(W, H) * 0.115)

    view = np.asarray(direction, float)
    view = -view / np.linalg.norm(view)          # camera -> target
    up = np.asarray(up, float)
    right = np.cross(view, up)
    right /= np.linalg.norm(right)
    up_s = np.cross(right, view)

    # The triad needs a clear corner. In the orthographic views the bottom-left
    # is already taken by the level ruler and the horizontal tick labels, so the
    # caller picks a free one rather than everything piling up in one place.
    pad = margin + size + 16
    cx, cy = {
        "bl": (pad, H - pad),
        "br": (W - pad, H - pad),
        "tl": (pad, pad),
        "tr": (W - pad, pad),
    }[corner]

    def fnt(sz):
        for n in ("/System/Library/Fonts/SFNSMono.ttf",
                  "/System/Library/Fonts/Supplemental/Arial.ttf"):
            try:
                return ImageFont.truetype(n, sz)
            except OSError:
                continue
        return ImageFont.load_default()

    f = fnt(max(13, size // 3))
    axes = [("X", (1, 0, 0), (200, 58, 46)),
            ("Y", (0, 1, 0), (34, 140, 68)),
            ("Z", (0, 0, 1), (44, 92, 200))]

    # Draw the axes pointing away first so the near ones overlay them.
    order = sorted(axes, key=lambda a: -float(np.asarray(a[1], float) @ view))
    for name, vec, rgb in order:
        v = np.asarray(vec, float)
        hx, hy = float(v @ right), float(v @ up_s)
        depth = float(v @ view)                   # >0 means pointing away
        away = depth > 0.02
        ex, ey = cx + hx * size, cy - hy * size

        col = rgb + (110,) if away else rgb + (255,)
        d.line([(cx, cy), (ex, ey)], fill=col, width=2 if away else 4)

        # arrowhead
        ang = np.arctan2(ey - cy, ex - cx)
        hlen = size * 0.26
        for s in (+1, -1):
            a = ang + s * 0.42
            d.line([(ex, ey), (ex - hlen * np.cos(a), ey - hlen * np.sin(a))],
                   fill=col, width=2 if away else 4)

        # Label just past the tip, pushed clear of the arrow direction so two
        # axes on the same side cannot collide.
        lx = cx + hx * size * 1.42
        ly = cy - hy * size * 1.42
        txt = name if not away else f"({name})"
        tw = d.textlength(txt, font=f)
        d.text((lx - tw / 2, ly - f.size / 2), txt, font=f, fill=rgb + (255 if not away else 150,))

    d.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(40, 46, 56, 255))
    # Legend sits under the triad, not in a fixed screen corner, so it travels
    # with it instead of landing on whatever else occupies that corner.
    f_leg = fnt(11)
    legend = "( ) = away"
    lw = d.textlength(legend, font=f_leg)
    # Above the triad in the bottom corners, below it in the top ones -- placed
    # unconditionally under it, the bottom-corner variant falls off the frame.
    ly = cy - size * 1.9 if corner.startswith("b") else cy + size * 1.72
    d.text((cx - lw / 2, ly), legend, font=f_leg, fill=(110, 120, 132, 255))
    return np.asarray(pil.convert("RGB"))


def _wrap_caption(caption, fnt, width_px, draw):
    """Break a caption on spaces so no line is clipped by the canvas edge."""
    out = []
    for para in str(caption).split("\n"):
        line = ""
        for word in para.split(" "):
            trial = f"{line} {word}".strip()
            if line and draw.textlength(trial, font=fnt) > width_px:
                out.append(line)
                line = word
            else:
                line = trial
        out.append(line)
    return out


def composite(images, labels, caption, out_path, pad=12, header=64):
    """Tile 4 views into one image with per-view labels and a caption strip.

    One image rather than four: same pixel budget, a quarter of the per-image
    overhead, and the model can compare views without cross-referencing.
    """
    from PIL import Image, ImageDraw, ImageFont

    def font(size):
        for name in ("/System/Library/Fonts/SFNSMono.ttf",
                     "/System/Library/Fonts/Supplemental/Arial.ttf"):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        return ImageFont.load_default()

    f_label, f_caption = font(26), font(22)
    h, w = images[0].shape[:2]
    W = w * 2 + pad * 3

    # Wrap the caption and grow the header to fit it. It used to take the first
    # two lines and clip each to the canvas width, so anything past that was
    # simply gone -- the legend explaining what a level's uncertainty band means
    # never appeared at all, and the line before it was cut mid-word. A caption
    # the reader cannot finish is not a caption.
    lines = _wrap_caption(caption, f_caption, W - pad * 2 - 8, ImageDraw.Draw(
        Image.new("RGB", (8, 8))))
    header = max(header, 20 + 26 * len(lines))
    H = h * 2 + pad * 3 + header

    canvas = Image.new("RGB", (W, H), "#f4f5f7")
    draw = ImageDraw.Draw(canvas)

    for i, (img, label) in enumerate(zip(images, labels)):
        col, row = i % 2, i // 2
        x = pad + col * (w + pad)
        y = header + pad + row * (h + pad)
        canvas.paste(Image.fromarray(img), (x, y))
        draw.rectangle([x, y, x + w - 1, y + h - 1], outline="#c9ced6", width=1)
        # Label sits on a solid chip so it stays readable over any render.
        tw = draw.textlength(label, font=f_label)
        draw.rectangle([x + 8, y + 8, x + 20 + tw, y + 44], fill="#2b3038")
        draw.text((x + 14, y + 13), label, font=f_label, fill="#ffffff")

    for i, line in enumerate(lines):
        draw.text((pad + 4, 10 + i * 26), line, font=f_caption, fill="#2b3038")

    canvas.save(out_path)
    return out_path


def _add_section_plane(p, mesh, plane, offset, opacity=0.28):
    """Draw the active section plane into the scene.

    Adding it as real 3D geometry rather than a 2D overlay is what lets one
    mechanism serve both jobs: in an isometric view it reads as a translucent
    sheet through the part, and in a view along the plane it collapses to
    exactly the level line we want. The bold border is what survives the
    edge-on case, where the translucent quad itself is a sub-pixel sliver.
    """
    from .section import PLANES

    spec = PLANES[plane]
    lo, hi = mesh.bounds
    centre = (lo + hi) / 2
    centre[spec["axis"]] = offset

    # Only slightly larger than the part: a big overshoot both wastes frame and
    # drags the auto-fit camera outwards, shrinking the part we came to look at.
    su = (hi[spec["u"]] - lo[spec["u"]]) * 1.12 + 1e-6
    sv = (hi[spec["v"]] - lo[spec["v"]]) * 1.12 + 1e-6

    quad = pv.Plane(center=centre, direction=spec["normal"],
                    i_size=su * 0.86, j_size=sv * 0.86,
                    i_resolution=1, j_resolution=1)
    p.add_mesh(quad, color="#e8a33d", opacity=opacity, lighting=False,
               show_edges=False)
    border = quad.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                        manifold_edges=False, non_manifold_edges=False)
    actor = p.add_mesh(border, color="#c2761a", line_width=4.0, lighting=False)
    mp = actor.GetMapper()
    mp.SetResolveCoincidentTopologyToPolygonOffset()
    mp.SetRelativeCoincidentTopologyLineOffsetParameters(-1.0, -8.0)
    return quad


def _draw_level_ruler(img, axis_name, lo_a, hi_a, centre_a, scale, offset, n_ticks=7):
    """Draw the normal-axis ruler and level callout ourselves, in pixel space.

    VTK's cube axes size their text by which bounding-box corner they attach to,
    which on a wide elevation puts three labels on top of each other in some
    tiles and not others. Because the camera here is set analytically -- focal
    point at the part centre, parallel scale fixed -- the mapping from a world
    coordinate to a pixel row is exact and we can just draw it. Deterministic
    beats negotiating with the layout heuristic.
    """
    from PIL import Image, ImageDraw, ImageFont

    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    w, h = pil.size

    def font(size):
        for name in ("/System/Library/Fonts/SFNSMono.ttf",
                     "/System/Library/Fonts/Supplemental/Arial.ttf"):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        return ImageFont.load_default()

    f = font(max(16, h // 26))
    # Parallel projection: half the viewport height is `scale` world units.
    px_per_unit = h / (2.0 * scale)

    def row(value):
        return h / 2.0 - (value - centre_a) * px_per_unit

    x0 = int(draw.textlength("-000.00", font=f)) + 18
    draw.line([(x0, row(hi_a)), (x0, row(lo_a))], fill="#4a5058", width=2)
    for v in np.linspace(lo_a, hi_a, n_ticks):
        y = row(v)
        draw.line([(x0 - 8, y), (x0, y)], fill="#4a5058", width=2)
        label = f"{v:.2f}"
        draw.text((x0 - 12 - draw.textlength(label, font=f), y - f.size * 0.62),
                  label, font=f, fill="#4a5058")
    draw.text((x0 - 8, row(hi_a) - f.size * 2.1), axis_name, font=f, fill="#2b3038")

    # The level callout, on a solid chip so it never fights the render behind it.
    y = row(offset)
    label = f"{axis_name} = {offset:.3f}"
    tw = draw.textlength(label, font=f)
    draw.line([(x0, y), (w - 8, y)], fill="#e8a33d", width=2)
    draw.rectangle([w - tw - 22, y - f.size - 6, w - 8, y + f.size * 0.5 + 4],
                   fill="#c2761a")
    draw.text((w - tw - 15, y - f.size - 2), label, font=f, fill="#ffffff")
    return np.asarray(pil)


TARGET_COLOUR = "#4a6b8a"      # steel blue, the mesh we are trying to match
BUILT_COLOUR = "#e0a458"       # amber, what has been reconstructed so far


# The six straight-on directions. Distinct from the elevations used when
# choosing a section offset: those are picked so one particular plane is
# edge-on, these are fixed and always the same six, so a face seen here means
# the same thing on every part.
VIEWS_ORTHO6 = (
    ("from +X", (1, 0, 0), (0, 0, 1)),
    ("from -X", (-1, 0, 0), (0, 0, 1)),
    ("from +Y", (0, 1, 0), (0, 0, 1)),
    ("from -Y", (0, -1, 0), (0, 0, 1)),
    ("from +Z", (0, 0, 1), (0, 1, 0)),
    ("from -Z", (0, 0, -1), (0, 1, 0)),
)


def composite_groups(groups, caption, out_path, pad=12, header=64, group_gap=40):
    """Tile several groups of views, each group on its own grid.

    The isometric and orthographic sets answer different questions and are laid
    out separately rather than poured into one grid, so a reader is never left
    working out which convention a given tile follows.
    """
    from PIL import Image, ImageDraw, ImageFont

    def font(size):
        for name in ("/System/Library/Fonts/SFNSMono.ttf",
                     "/System/Library/Fonts/Supplemental/Arial.ttf"):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        return ImageFont.load_default()

    f_label, f_caption, f_group = font(24), font(22), font(19)

    h, w = groups[0][1][0][0].shape[:2]
    max_cols = max(g[2] for g in groups)
    W = w * max_cols + pad * (max_cols + 1)

    # Wrapped and given the room it needs, as in `composite`: keeping only the
    # first two lines would drop the residual's legend for what a dim body means.
    lines = _wrap_caption(caption, f_caption, W - pad * 2 - 8, ImageDraw.Draw(
        Image.new("RGB", (8, 8))))
    header = max(header, 20 + 26 * len(lines))
    H = header
    for _, items, cols in groups:
        rows = (len(items) + cols - 1) // cols
        H += group_gap + rows * (h + pad)
    H += pad

    canvas = Image.new("RGB", (W, H), "#f4f5f7")
    draw = ImageDraw.Draw(canvas)
    for i, line in enumerate(lines):
        draw.text((pad + 4, 10 + i * 26), line, font=f_caption, fill="#2b3038")

    y = header
    for title, items, cols in groups:
        draw.text((pad + 4, y + 10), title, font=f_group, fill="#5a6270")
        y += group_gap
        row_w = cols * w + (cols - 1) * pad
        x_off = (W - row_w) // 2
        for i, (img, label) in enumerate(items):
            cx, cy = i % cols, i // cols
            x = x_off + cx * (w + pad)
            yy = y + cy * (h + pad)
            canvas.paste(Image.fromarray(img), (x, yy))
            draw.rectangle([x, yy, x + w - 1, yy + h - 1], outline="#c9ced6", width=1)
            tw = draw.textlength(label, font=f_label)
            draw.rectangle([x + 8, yy + 8, x + 20 + tw, yy + 42], fill="#2b3038")
            draw.text((x + 14, yy + 12), label, font=f_label, fill="#ffffff")
        y += ((len(items) + cols - 1) // cols) * (h + pad)

    canvas.save(out_path)
    return out_path


def projections(target, built, out_path, cfg: RenderConfig = None, caption=None,
                views="all", residual=None):
    """Ten views -- four isometric plus the six straight-on -- in one image.

    Always composed by colour channel, target green and reconstruction red, even
    before anything has been reconstructed. Keeping one convention for the whole
    session means a green shape never has to be re-interpreted: it is the target,
    at every stage.

    One scale across all ten tiles, so a feature's size can be compared between
    views instead of only within one.

    With `residual` (a Residual) the tiles are its missing and extra bodies in
    front of the build (`residual_compose`); target and built then only frame
    the views.
    """
    cfg = cfg or RenderConfig()
    meshes = [m for m in (target, built) if m is not None and len(m.vertices)]
    lo = np.min([m.bounds[0] for m in meshes], axis=0)
    hi = np.max([m.bounds[1] for m in meshes], axis=0)
    centre = (lo + hi) / 2
    corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) - centre

    iso_views = [(lab, d, (0, 0, 1)) for lab, d in cfg.views]
    if views == "iso":
        all_views, n_iso = iso_views, len(iso_views)
    elif views == "ortho":
        all_views, n_iso = list(VIEWS_ORTHO6), 0
    else:
        all_views = iso_views + list(VIEWS_ORTHO6)
        n_iso = len(iso_views)

    def half_extent(direction, up):
        v = -np.asarray(direction, float)
        v /= np.linalg.norm(v)
        r = np.cross(v, np.asarray(up, float))
        r /= np.linalg.norm(r)
        u = np.cross(r, v)
        return max(float(np.abs(corners @ r).max()), float(np.abs(corners @ u).max()))

    scale = max(half_extent(d, u) for _, d, u in all_views) * 1.10

    tiles = []
    for label, direction, up in all_views:
        size = (cfg.tile, cfg.tile)
        dv = np.asarray(direction, float)
        dv = dv / np.linalg.norm(dv)
        proj = corners @ dv
        drange = (float(proj.min()) + float(centre @ dv),
                  float(proj.max()) + float(centre @ dv))
        if residual is not None:
            comp = _residual_tile(residual, direction, up, centre, scale, size, cfg,
                                  drange)
        else:
            it = _solo_shot(target, direction, up, centre, scale, size, cfg, drange)
            ib = (_solo_shot(built, direction, up, centre, scale, size, cfg, drange)
                  if built is not None and len(built.vertices)
                  else np.zeros_like(it))
            comp, _, _ = channel_compose(ib, it)
        if cfg.gnomon:
            comp = draw_gnomon(comp, direction, up=up)
        tiles.append((comp, label))

    if caption is None:
        state = ("green = target, red = reconstruction, yellow = both agree"
                 if built is not None else
                 "green = target (nothing reconstructed yet)")
        caption = (state + "\nsame scale in every tile; isometric views share one "
                   "elevation, orthographic views are the six straight-on directions")
    groups = []
    if n_iso:
        groups.append((f"{n_iso} isometric", tiles[:n_iso], 4))
    if len(tiles) > n_iso:
        groups.append((f"{len(tiles) - n_iso} orthographic", tiles[n_iso:], 3))
    return composite_groups(groups, caption, out_path)


def _solo_shot(mesh_i, direction, up, centre, scale, size, cfg, depth_range=None):
    """One mesh, shaded by DEPTH along the view axis, on black.

    Not lit shading. In a straight-on orthographic view two coaxial cylinders
    present two faces that are both perpendicular to the camera and therefore
    shade identically -- the step between them vanishes, which is exactly how a
    boss disappeared into the disc behind it in the -Y view. Colouring by
    distance towards the camera makes the nearer face brighter, so the step is
    visible whatever the lighting would have done. (CADENA's own visualiser
    does the same thing, per-axis coordinate as the scalar.)

    `depth_range` is shared between target and reconstruction so that the same
    physical depth maps to the same brightness in both, and agreement composites
    to clean yellow instead of a gradient.
    """
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)

    poly = pv.wrap(mesh_i)
    depth = poly.points @ d                     # larger = nearer the camera
    if depth_range is None:
        depth_range = (float(depth.min()), float(depth.max()))
    lo_d, hi_d = depth_range
    rng = max(hi_d - lo_d, 1e-9)
    # Floor at 0.22 so the far side stays clearly distinct from the background.
    poly["depth"] = (0.22 + 0.78 * np.clip((depth - lo_d) / rng, 0, 1)) * 255.0

    p = pv.Plotter(off_screen=True, window_size=size)
    p.background_color = "black"
    p.add_mesh(poly, scalars="depth", cmap="gray", clim=(0, 255),
               show_scalar_bar=False, lighting=False)
    p.enable_anti_aliasing("fxaa")
    p.camera.SetParallelProjection(True)
    # Place the camera at centre + direction, NOT at `direction` itself. Using
    # the direction vector as an absolute position puts the camera a millimetre
    # from the world origin, and the actual view direction becomes
    # (origin - part centre) -- nearly the same for every requested direction,
    # so all four tiles come out identical.
    p.camera_position = [tuple(centre + d * scale * 6.0), tuple(centre), tuple(up)]
    p.camera.SetParallelScale(scale)
    img = p.screenshot(return_img=True)
    p.close()
    return img


def channel_compose(img_red, img_green):
    """Two depth-shaded renders combined by colour channel.

    Red channel from one solid, green from the other, exactly as CADENA merges
    them. Both arrive black-on-black outside the part, so agreement is yellow,
    target-only is green, reconstruction-only is red, and empty space is black
    with no masking or thresholding required.
    """
    r = img_red.astype(np.float32).mean(axis=2)
    g = img_green.astype(np.float32).mean(axis=2)
    out = np.zeros((*r.shape, 3), np.uint8)
    out[..., 0] = np.clip(r, 0, 255).astype(np.uint8)
    out[..., 1] = np.clip(g, 0, 255).astype(np.uint8)
    return out, r > 8, g > 8


# How the build is drawn behind a residual: bright enough to say where a body
# sits, dim enough that none of it reads as a body.
CONTEXT_LEVEL = 0.30
# A skin -- a face built off the target's, `face off` in the residual table --
# at this share of a solid body's brightness: it covers a large area and holds
# little volume, and drawn as bright as a block it would read as a block.
SKIN_LEVEL = 0.45
RESIDUAL_LEGEND = ("green = MISSING (the D bodies), red = EXTRA (the E bodies), "
                   "grey = built so far, for where they sit; every body is drawn "
                   "in front of the build\n"
                   "dim = a face built off the target's (a skin: large area, "
                   "little volume); yellow/orange = a missing and an extra body "
                   "in one line of sight (a face built off on both sides)")
RESIDUAL_SOLO_LEGEND = ("the residual ALONE: green = MISSING (the D bodies), red = "
                        "EXTRA (the E bodies); neither target nor build is drawn\n"
                        "dim = a face built off the target's (a skin: large area, "
                        "little volume); yellow/orange = a missing and an extra "
                        "body in one line of sight")


@dataclass
class Residual:
    """What a residual render draws: [(mesh, brightness)] missing and extra,
    1.0 for a solid piece and SKIN_LEVEL for a skin, and the build behind
    them (None to draw the residual alone)."""
    missing: list
    extra: list
    context: object = None


def _layer_shot(parts, direction, up, centre, scale, size, cfg, drange):
    """Several meshes in one grey channel, each at its own brightness."""
    import trimesh
    img = None
    for level in sorted({lv for _, lv in parts}):
        ms = [m for m, lv in parts if lv == level and m is not None and len(m.vertices)]
        if not ms:
            continue
        shot = _solo_shot(trimesh.util.concatenate(ms), direction, up, centre,
                          scale, size, cfg, drange).astype(np.float32).mean(axis=2)
        img = shot * level if img is None else np.maximum(img, shot * level)
    return img


def residual_compose(missing, extra, context=None):
    """Missing green, extra red, the build dim grey behind both.

    The two-solid compose cannot draw a residual. Given the leftover in green
    against the build in red, every extra body lies inside the build and comes
    out yellow -- the colour of agreement -- and the build red wherever nothing
    was left in front of it, so a thin skin of leftover reads as all wrong while
    the extra bodies of the table read as right. Here each colour is one half of
    the table: green the D bodies, red the E bodies, and the build only grey.
    Arguments are grey images (H, W), or None for nothing drawn.
    """
    shape = next(a.shape for a in (missing, extra, context) if a is not None)
    zero = np.zeros(shape, np.float32)
    m = zero if missing is None else missing
    e = zero if extra is None else extra
    lump = (m > 8) | (e > 8)
    out = np.zeros((*shape, 3), np.uint8)
    if context is not None:
        grey = np.clip(context * CONTEXT_LEVEL, 0, 255).astype(np.uint8)
        for k in range(3):
            out[..., k] = np.where(lump, 0, grey)
    out[..., 0] = np.where(lump, np.clip(e, 0, 255), out[..., 0]).astype(np.uint8)
    out[..., 1] = np.where(lump, np.clip(m, 0, 255), out[..., 1]).astype(np.uint8)
    return out


def _residual_tile(res, direction, up, centre, scale, size, cfg, drange):
    shots = [_layer_shot(parts, direction, up, centre, scale, size, cfg, drange)
             for parts in (res.missing, res.extra,
                           [(res.context, 1.0)] if res.context is not None else [])]
    if all(s is None for s in shots):
        return np.zeros((size[1], size[0], 3), np.uint8)
    return residual_compose(*shots)


def channel_views(target, built, out_path, cfg: RenderConfig = None, caption=None):
    """Iso views with target and reconstruction composited by channel."""
    cfg = cfg or RenderConfig()
    meshes = [m for m in (target, built) if m is not None and len(m.vertices)]
    lo = np.min([m.bounds[0] for m in meshes], axis=0)
    hi = np.max([m.bounds[1] for m in meshes], axis=0)
    centre = (lo + hi) / 2
    corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) - centre

    images, labels = [], []
    for label, direction in cfg.views:
        view = -np.asarray(direction, float)
        view /= np.linalg.norm(view)
        right = np.cross(view, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        upv = np.cross(right, view)
        scale = max(float(np.abs(corners @ right).max()),
                    float(np.abs(corners @ upv).max())) * 1.12

        size = (cfg.tile, cfg.tile)
        dv = -view
        proj = corners @ dv
        drange = (float(proj.min()) + float(centre @ dv),
                  float(proj.max()) + float(centre @ dv))
        it = _solo_shot(target, direction, (0, 0, 1), centre, scale, size, cfg, drange)
        ib = (_solo_shot(built, direction, (0, 0, 1), centre, scale, size, cfg, drange)
              if built is not None and len(built.vertices) else np.zeros_like(it))
        comp, _, _ = channel_compose(ib, it)
        images.append(draw_gnomon(comp, direction) if cfg.gnomon else comp)
        labels.append(label)

    if caption is None:
        caption = ("green = target only (missing)   red = reconstruction only "
                   "(extra)   yellow = both agree\nblack = empty space; each solid "
                   "rendered opaque and combined by colour channel")
    return composite(images, labels, caption, out_path)


def overlay_views(layers, out_path, cfg: RenderConfig = None, caption=None):
    """The four isometric views with several meshes superimposed.

    `layers` is a list of (mesh, colour, opacity) drawn in order.

    Note what this deliberately does NOT do: draw the target and the
    reconstruction both translucent on top of each other. Over most of their
    area those two solids share the same surface, and two coincident
    semi-transparent faces fight over depth -- the result is a mess of radial
    streaks that reads as neither shape. Passing the boolean difference instead
    means no two layers share a surface, and each colour marks a region that
    genuinely belongs to only one of them.

    Depth peeling is still needed for the translucent layer to composite in the
    right order regardless of how the actors were added.
    """
    cfg = cfg or RenderConfig()
    meshes = [m for m, _, _ in layers if m is not None and len(m.vertices)]
    if not meshes:
        raise ValueError("nothing to draw")
    lo = np.min([m.bounds[0] for m in meshes], axis=0)
    hi = np.max([m.bounds[1] for m in meshes], axis=0)
    centre = (lo + hi) / 2

    corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) - centre

    images, labels = [], []
    for label, direction in cfg.views:
        p = pv.Plotter(off_screen=True, window_size=(cfg.tile, cfg.tile))
        p.background_color = cfg.background
        p.enable_depth_peeling(number_of_peels=12, occlusion_ratio=0.0)

        for mesh_i, colour, opacity in layers:
            if mesh_i is None or not len(mesh_i.vertices):
                continue
            p.add_mesh(pv.wrap(mesh_i), color=colour, opacity=opacity,
                       smooth_shading=True, split_sharp_edges=True,
                       feature_angle=30.0, specular=0.12, ambient=0.32,
                       diffuse=0.74)
        _add_lighting(p)
        p.enable_anti_aliasing("fxaa")

        p.camera.SetParallelProjection(True)
        _d = np.asarray(direction, float)
        _d = _d / np.linalg.norm(_d)
        view = -_d
        right = np.cross(view, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        upv = np.cross(right, view)
        half = max(float(np.abs(corners @ right).max()),
                   float(np.abs(corners @ upv).max()))
        p.camera_position = [tuple(centre + _d * half * 6.0), tuple(centre), (0, 0, 1)]
        p.camera.SetParallelScale(half * 1.12)

        img = p.screenshot(return_img=True)
        p.close()
        images.append(draw_gnomon(img, direction) if cfg.gnomon else img)
        labels.append(label)

    if caption is None:
        caption = ("amber = reconstruction so far (translucent)   "
                   "blue = target material still missing   red = built but not in "
                   "the target")
    return composite(images, labels, caption, out_path)


def _nice_ticks(lo, hi, target=6):
    """Round tick values covering [lo, hi] at a 1/2/5 x 10^k step."""
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / max(target, 1)
    mag = 10.0 ** np.floor(np.log10(raw))
    for mult in (1, 2, 2.5, 5, 10):
        step = mult * mag
        if span / step <= target:
            break
    first = np.ceil(lo / step) * step
    out, v = [], first
    while v <= hi + step * 1e-6:
        out.append(round(float(v), 6))
        v += step
    return out


def _decorate_ortho(img, axis_name, h_axis_name, v_to_row, h_to_col,
                    v_range, h_range, offset, marks, gutter_l=104, gutter_b=48,
                    gutter_t=34, gutter_r=18, dark=False):
    """Wrap a rendered elevation in its own coordinate frame.

    The numbers live in gutters outside the render, never on top of it: drawing
    a scale over the part is what made the earlier version unreadable. Grid
    lines run horizontally only, because the single quantity being chosen here
    is a level along the section normal -- vertical rules would just be noise.
    Feature levels from the sweep are drawn in their own colour so the choice
    becomes "pick a band", not "guess a number".
    """
    from PIL import Image, ImageDraw, ImageFont

    src = Image.fromarray(img)
    iw, ih = src.size
    W, H = gutter_l + iw + gutter_r, gutter_t + ih + gutter_b
    canvas = Image.new("RGB", (W, H), "#ffffff")
    canvas.paste(src, (gutter_l, gutter_t))
    d = ImageDraw.Draw(canvas, "RGBA")

    def fnt(sz):
        for n in ("/System/Library/Fonts/SFNSMono.ttf",
                  "/System/Library/Fonts/Supplemental/Arial.ttf"):
            try:
                return ImageFont.truetype(n, sz)
            except OSError:
                continue
        return ImageFont.load_default()

    f = fnt(max(12, ih // 34))
    f_ax = fnt(max(13, ih // 30))
    f_mk = fnt(max(11, ih // 38))

    # Over a depth-shaded render the background is black, so the grid has to
    # invert or it simply is not there.
    GRID = (210, 218, 228, 90) if dark else (120, 132, 148, 70)
    AXIS = (86, 96, 110, 255)
    TEXT = (70, 80, 94, 255)
    MARK = (80, 150, 235, 255) if dark else (43, 108, 176, 255)
    END = (60, 190, 130, 255) if dark else (22, 120, 78, 255)
    LEVEL = (240, 165, 60, 255) if dark else (194, 118, 26, 255)

    x0, x1 = gutter_l, gutter_l + iw
    y0, y1 = gutter_t, gutter_t + ih

    mark_rows = [(gutter_t + v_to_row(m["value"]), m) for m in (marks or [])]
    mark_rows = [(r, m) for r, m in mark_rows if y0 - 2 <= r <= y1 + 2]

    # Thin the LABELS where rules crowd together, keeping part ends over
    # boundaries. Every rule is still drawn -- only the text is suppressed, and
    # the count of suppressed ones is reported -- because five overlapping
    # labels within a millimetre render as an unreadable smear and hide the one
    # that matters.
    order = sorted(mark_rows, key=lambda rm: (rm[1].get("kind") != "end", rm[0]))
    kept, hidden = [], 0
    for row, m in order:
        if any(abs(row - r2) < f.size * 1.25 for r2, _ in kept):
            hidden += 1
            continue
        kept.append((row, m))
    labelled = {id(m) for _, m in kept}

    # Horizontal grid + vertical-axis numbers, skipping any tick that would
    # land on a feature level -- the feature label is the more informative one.
    for v in _nice_ticks(*v_range):
        row = gutter_t + v_to_row(v)
        if not (y0 - 1 <= row <= y1 + 1):
            continue
        if any(abs(row - r) < f.size * 0.95 for r, _ in mark_rows):
            continue
        d.line([(x0, row), (x1, row)], fill=GRID, width=1)
        d.line([(x0 - 6, row), (x0, row)], fill=AXIS, width=1)
        t = f"{v:g}"
        d.text((x0 - 10 - d.textlength(t, font=f), row - f.size * 0.62), t,
               font=f, fill=TEXT)

    # Feature levels, weighted by how much each one is actually worth.
    #
    # A part end is exact -- it comes from the bounding box, not from the sweep
    # -- so it is always drawn at full strength. A prism boundary is a
    # heuristic: it fires whenever the cross-section area moves more than a
    # threshold between adjacent samples, which on a curved part produces a
    # crowd of levels that mean nothing. So boundary weight FALLS as their
    # number rises: one or two boundaries are probably real and are drawn
    # boldly; nine are probably curvature and are drawn as hairlines. The
    # picture then carries its own confidence.
    n_bounds = sum(1 for _, m in mark_rows if m.get("kind") != "end")
    b_width = 3.0 if n_bounds <= 2 else max(1.0, 3.0 - 0.34 * (n_bounds - 2))
    b_alpha = 255 if n_bounds <= 2 else max(90, int(255 - 18 * (n_bounds - 2)))

    # Uncertainty, drawn at the same scale as everything else in the picture.
    #
    # A level known to +/-0.38 mm should be drawn as a
    # BAND, not a hairline, because a decision made by looking is only as good
    # as the picture and every level used to be one crisp pixel whether it was
    # a bounding-box face or a guess. Only the label distinguished them, and a
    # label is not what the eye reads first.
    #
    # The band is the level's own uncertainty converted through v_to_row, so it
    # is a real measurement rather than a decoration: a bisected boundary known
    # to 0.001 mm has a band far under a pixel and stays a hairline, while a
    # plateau edge's error bar is often several pixels tall. The distinction
    # draws itself, and the two now LOOK as different as they are.
    bands = 0
    for _row, m in mark_rows:
        unc = float(m.get("uncertainty", 0.0) or 0.0)
        if unc <= 0:
            continue
        top = gutter_t + v_to_row(m["value"] + unc)
        bot = gutter_t + v_to_row(m["value"] - unc)
        lo_r, hi_r = (min(top, bot), max(top, bot))
        if hi_r - lo_r < 1.6:
            continue          # sub-pixel: the hairline already says "exact"
        lo_r, hi_r = max(lo_r, y0), min(hi_r, y1)
        if hi_r <= lo_r:
            continue
        base = END if m.get("kind") == "end" else MARK
        d.rectangle([x0, lo_r, x1, hi_r], fill=base[:3] + (64,))
        for edge in (lo_r, hi_r):
            d.line([(x0, edge), (x1, edge)], fill=base[:3] + (105,), width=1)
        bands += 1

    for row, m in mark_rows:
        end = m.get("kind") == "end"
        show_text = id(m) in labelled
        t = f"{m['value']:g}"
        tag = m.get("label", "") if show_text else ""

        if end:
            # Always prominent: white halo so it survives crossing the part,
            # solid rule, arrowheads in both gutters, filled label chip.
            d.line([(x0, row), (x1, row)], fill=(255, 255, 255, 225), width=8)
            d.line([(x0, row), (x1, row)], fill=END, width=4)
            tri = f.size * 0.6
            d.polygon([(x0 - 3, row), (x0 - 3 - tri, row - tri * 0.62),
                       (x0 - 3 - tri, row + tri * 0.62)], fill=END)
            d.polygon([(x1 + 3, row), (x1 + 3 + tri, row - tri * 0.62),
                       (x1 + 3 + tri, row + tri * 0.62)], fill=END)
            if show_text:
                d.text((x0 - 9 - tri - d.textlength(t, font=f_ax),
                        row - f_ax.size * 0.62), t, font=f_ax, fill=END)
            if tag:
                tw = d.textlength(tag, font=f_mk)
                ty = row + 5 if row - y0 < f_mk.size * 3 else row - f_mk.size - 7
                d.rectangle([x1 - tw - 22, ty - 3, x1 - 8, ty + f_mk.size + 3], fill=END)
                d.text((x1 - tw - 15, ty), tag, font=f_mk, fill=(255, 255, 255, 255))
            continue

        col = MARK[:3] + (b_alpha,)
        d.line([(x0, row), (x1, row)], fill=col, width=int(round(b_width)))
        d.line([(x0 - 7, row), (x0, row)], fill=col, width=int(round(b_width)))
        if show_text:
            d.text((x0 - 11 - d.textlength(t, font=f_mk), row - f_mk.size * 0.62),
                   t, font=f_mk, fill=col)
        if tag:
            tw = d.textlength(tag, font=f_mk)
            ty = row + 4 if row - y0 < f_mk.size * 2.4 else row - f_mk.size - 4
            d.text((x1 - tw - 8, ty), tag, font=f_mk, fill=col)

    # Current level, last so it sits on top of everything. Absent while the
    # offset is still being chosen -- drawing a "current" level before one has
    # been picked invents a decision the reader has not made yet.
    if offset is not None:
        row = gutter_t + v_to_row(offset)
        if y0 - 2 <= row <= y1 + 2:
            d.line([(x0, row), (x1, row)], fill=LEVEL, width=3)
            chip = f"{axis_name} = {offset:.3f}"
            cw = d.textlength(chip, font=f)
            d.rectangle([x1 - cw - 16, row - f.size - 6, x1 - 2, row + f.size * 0.5 + 4],
                        fill=LEVEL)
            d.text((x1 - cw - 9, row - f.size - 2), chip, font=f,
                   fill=(255, 255, 255, 255))

    # Axis lines and names: vertical name above its axis, horizontal name to
    # the right of its own, as requested.
    d.line([(x0, y0), (x0, y1)], fill=AXIS, width=2)
    d.line([(x0, y1), (x1, y1)], fill=AXIS, width=2)
    # Vertical axis name above its axis, clear of the tile's name chip.
    d.text((x0 - f_ax.size * 0.35, y0 - gutter_t + 4), axis_name, font=f_ax, fill=AXIS)

    for hv in _nice_ticks(*h_range):
        col = gutter_l + h_to_col(hv)
        if not (x0 - 1 <= col <= x1 + 1):
            continue
        d.line([(col, y1), (col, y1 + 6)], fill=AXIS, width=1)
        t = f"{hv:g}"
        d.text((col - d.textlength(t, font=f) / 2, y1 + 9), t, font=f, fill=TEXT)
    d.text((x1 + 5, y1 - f_ax.size * 0.55), h_axis_name, font=f_ax, fill=AXIS)

    return np.asarray(canvas)


def ortho_views(mesh, plane, offset=None, out_path=None, cfg: RenderConfig = None,
                caption=None, marks=None, layers=None, pair=None, residual=None):
    """Stage 3b: the four views that make a section offset choosable.

    For a section on XY (normal Z) these are the views from +X, -X, +Y and -Y --
    every direction in which the plane is edge-on and therefore appears as a
    straight level line across the part's silhouette. The section normal is put
    on screen-up so that line is horizontal in all four tiles regardless of
    which plane family is active, which keeps the "raise/lower the level"
    interaction identical for XY, YZ and XZ.

    `marks` are feature levels from the sweep -- the offsets at which the
    cross-section changes. Drawing them here turns choosing an offset from
    reading a number off a ruler into picking one of a few labelled bands.
    """
    from .section import PLANES

    cfg = cfg or RenderConfig()
    spec = PLANES[plane]
    poly = pv.wrap(mesh)

    a = spec["axis"]
    up = np.zeros(3)
    up[a] = 1.0

    views = []
    for ax in (spec["u"], spec["v"]):
        for sign in (1.0, -1.0):
            d = np.zeros(3)
            d[ax] = sign
            # Short label: the name chip is drawn over the tile's top-left
            # corner, which is exactly where the vertical axis name goes. A
            # wide "from +X" chip swallows it.
            views.append((f"{'+' if sign > 0 else '-'}{'XYZ'[ax]}", d))

    lo, hi = mesh.bounds
    if layers:
        ms = [m for m, _, _ in layers if m is not None and len(m.vertices)]
        if ms:
            lo = np.min([m.bounds[0] for m in ms] + [lo], axis=0)
            hi = np.max([m.bounds[1] for m in ms] + [hi], axis=0)
    centre = (lo + hi) / 2

    half_v = (hi[a] - lo[a]) / 2.0
    half_h = max(hi[spec["u"]] - lo[spec["u"]], hi[spec["v"]] - lo[spec["v"]]) / 2.0
    margin = 1.10          # the scale now lives outside the render, so less room needed
    aspect = float(np.clip(half_h / max(half_v, 1e-9), 0.65, 2.4))
    img_h = int(min(cfg.tile, 1_200_000 / max(cfg.tile * aspect, 1)))
    img_w = int(img_h * aspect)
    scale = max(half_v, half_h / aspect) * margin
    ppu = img_h / (2.0 * scale)          # pixels per mm, both axes (parallel proj)

    images, labels = [], []
    for label, d in views:
        if pair is not None:
            # Same convention as the projections: target green, reconstruction
            # red, both depth-shaded so a coaxial step never vanishes.
            tgt, blt = pair
            dv = -np.asarray(d, float)
            dv = dv / np.linalg.norm(dv)
            allc = np.array([[x, y, z] for x in (lo[0], hi[0])
                             for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            proj = allc @ dv
            drange = (float(proj.min()), float(proj.max()))
            size = (img_w, img_h)
            if residual is not None:
                shot = _residual_tile(residual, d, up, centre, scale, size, cfg, drange)
            else:
                it = _solo_shot(tgt, d, up, centre, scale, size, cfg, drange)
                ib = (_solo_shot(blt, d, up, centre, scale, size, cfg, drange)
                      if blt is not None and len(blt.vertices) else np.zeros_like(it))
                shot, _, _ = channel_compose(ib, it)
            images.append(_decorate_ortho(
                shot, "XYZ"[a], "XYZ"[int(np.argmax(np.abs(np.cross(dv, up))))],
                (lambda v, _c=float(centre[a]): img_h / 2.0 - (v - _c) * ppu),
                (lambda x, _j=int(np.argmax(np.abs(np.cross(dv, up)))),
                        _s=float(np.sign(np.cross(dv, up)[int(np.argmax(np.abs(np.cross(dv, up))))])):
                 img_w / 2.0 + (x - float(centre[_j])) * _s * ppu),
                (float(centre[a]) - scale, float(centre[a]) + scale),
                tuple(sorted((float(centre[int(np.argmax(np.abs(np.cross(dv, up))))])
                              - (img_w / 2.0) / ppu,
                              float(centre[int(np.argmax(np.abs(np.cross(dv, up))))])
                              + (img_w / 2.0) / ppu))),
                offset, marks, dark=True))
            labels.append(label)
            continue

        p = pv.Plotter(off_screen=True, window_size=(img_w, img_h))
        p.background_color = cfg.background
        if layers:
            # Overlay mode: several solids, drawn in the order given. Depth
            # peeling so the translucent layer composites correctly whatever
            # order the actors went in.
            p.enable_depth_peeling(number_of_peels=12, occlusion_ratio=0.0)
            for mesh_i, colour, op in layers:
                if mesh_i is None or not len(mesh_i.vertices):
                    continue
                p.add_mesh(pv.wrap(mesh_i), color=colour, opacity=op,
                           smooth_shading=True, split_sharp_edges=True,
                           feature_angle=cfg.feature_angle, specular=0.12,
                           ambient=0.32, diffuse=0.74)
        else:
            p.add_mesh(poly, **dict(CLAY, feature_angle=cfg.feature_angle))
            if cfg.feature_edges:
                edges = _feature_edges(poly, cfg.feature_angle)
                if edges.n_cells:
                    act = p.add_mesh(edges, color=cfg.edge_color,
                                     line_width=cfg.edge_width, lighting=False)
                    mp = act.GetMapper()
                    mp.SetResolveCoincidentTopologyToPolygonOffset()
                    mp.SetRelativeCoincidentTopologyLineOffsetParameters(-1.0, -6.0)

        if offset is not None:
            _add_section_plane(p, mesh, plane, offset)
        _add_lighting(p)
        p.enable_anti_aliasing("fxaa")

        p.camera.SetParallelProjection(True)
        p.camera_position = [tuple(d), (0, 0, 0), tuple(up)]
        p.reset_camera()
        p.camera.SetFocalPoint(*centre)
        p.camera.SetParallelScale(scale)
        shot = p.screenshot(return_img=True)
        p.close()

        # Which world axis runs left-to-right on screen, and with what sign.
        view = -d / np.linalg.norm(d)
        right = np.cross(view, up)
        right /= np.linalg.norm(right)
        j = int(np.argmax(np.abs(right)))
        sgn = float(np.sign(right[j]))

        def v_to_row(v, _c=float(centre[a])):
            return img_h / 2.0 - (v - _c) * ppu

        def h_to_col(x, _c=float(centre[j]), _s=sgn):
            return img_w / 2.0 + (x - _c) * _s * ppu

        h_edge = (img_w / 2.0) / ppu
        h_range = tuple(sorted((float(centre[j]) - h_edge, float(centre[j]) + h_edge)))
        v_range = (float(centre[a]) - scale, float(centre[a]) + scale)

        images.append(_decorate_ortho(
            shot, "XYZ"[a], "XYZ"[j], v_to_row, h_to_col, v_range, h_range,
            offset, marks))
        labels.append(label)

    if caption is None:
        n_marks = len(marks or [])
        where = (f"at {'XYZ'[a]} = {offset:.3f} mm   " if offset is not None
                 else "choosing the offset   ")
        legend = ("orange = chosen level; " if offset is not None else "")
        # Say what a band IS. A shaded stripe nobody can interpret is
        # decoration, and the whole reason for drawing uncertainty rather than
        # only printing it is that the eye reads the picture first.
        widest = max((float(m.get("uncertainty", 0.0) or 0.0)
                      for m in (marks or [])), default=0.0)
        # Its own line. Appended to the legend it ran past the image width and
        # was cut mid-word, and a truncated legend is worse than none.
        band = (f"\na level's shaded band is how well it is LOCATED: widest "
                f"here +/-{widest:.3f} mm. No visible band = exact to under a "
                f"pixel." if widest > 0 else "")
        caption = (f"section plane {plane} (normal {'XYZ'[a]}) {where}"
                   f"range [{lo[a]:.3f}, {hi[a]:.3f}]\n"
                   f"tiles labelled by the direction the camera looks from; "
                   f"{legend}blue = the {n_marks} feature level(s){band}")
        if residual is not None:
            caption += "\n" + (RESIDUAL_LEGEND if residual.context is not None
                               else RESIDUAL_SOLO_LEGEND)
    return composite(images, labels, caption, out_path)
def overview(mesh, out_path, cfg: RenderConfig = None, caption=None,
             plane=None, offset=None):
    """Stage 1 entry point. `mesh` is a trimesh.Trimesh."""
    cfg = cfg or RenderConfig()
    poly = pv.wrap(mesh)

    lo, hi = mesh.bounds
    if caption is None:
        ext = hi - lo
        caption = (
            f"bbox {ext[0]:.2f} x {ext[1]:.2f} x {ext[2]:.2f} mm   "
            f"origin ({lo[0]:.2f}, {lo[1]:.2f}, {lo[2]:.2f})   "
            f"vol {mesh.volume:.1f} mm^3   {len(mesh.faces)} tris\n"
            "orthographic; each view labelled by the octant the camera sits in"
        )

    # Shared bounds so all four tiles are graduated identically.
    bounds = (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
    images, labels = [], []
    for label, direction in cfg.views:
        images.append(render_view(poly, direction, cfg, bounds=bounds,
                                  mesh=mesh, plane=plane, offset=offset))
        labels.append(label)

    return composite(images, labels, caption, out_path)


# --------------------------------------------------- level views on clay panes

# On a pane that already carries a blue grid, a blue feature level is a grid
# line with a label. The rules therefore move to violet; ends stay green and the
# chosen level stays amber, so the two things a reader already knows the colour
# of do not move.
PANE_END = (14, 122, 75)
PANE_MARK = (124, 45, 178)
PANE_LEVEL = (194, 118, 26)


def _pane_font(size):
    for name in ("/System/Library/Fonts/SFNSMono.ttf",
                 "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            from PIL import ImageFont
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    from PIL import ImageFont
    return ImageFont.load_default()


def _draw_levels_on_pane(img, frame, marks, offset):
    """Draw the feature levels across one orthographic pane.

    The pane already states its own axes, grid and title; all this adds is the
    levels, and it adds them at rows taken from the pane's own frame rather than
    from a second guess at the layout -- see orthographic_views.pane_frames.
    """
    from PIL import Image, ImageDraw

    from orthographic_views import frame_row

    pil = Image.fromarray(img).convert("RGB")
    d = ImageDraw.Draw(pil, "RGBA")
    H = frame["height"]
    x0, y0b, x1, y1b = frame["rect"]
    top, bot = H - y1b, H - y0b                  # the plot, top-down

    f = _pane_font(13)
    rows = [(frame_row(frame, m["value"]), m) for m in (marks or [])]
    rows = [(r, m) for r, m in rows if top - 2 <= r <= bot + 2]

    # Same thinning rule as the older elevations: every rule is drawn, only the
    # text is suppressed when two would overprint, and ends win the ties.
    kept, hidden = [], 0
    for row, m in sorted(rows, key=lambda rm: (rm[1].get("kind") != "end", rm[0])):
        if any(abs(row - r2) < f.size * 1.3 for r2, _ in kept):
            hidden += 1
            continue
        kept.append((row, m))
    labelled = {id(m) for _, m in kept}

    n_bounds = sum(1 for _, m in rows if m.get("kind") != "end")
    width = 3 if n_bounds <= 2 else max(1, int(round(3.0 - 0.34 * (n_bounds - 2))))
    alpha = 255 if n_bounds <= 2 else max(90, 255 - 18 * (n_bounds - 2))

    # Uncertainty first, under the rules: a band is where the level might be,
    # and a rule drawn under its own band would read as the band's edge.
    for _row, m in rows:
        unc = float(m.get("uncertainty", 0.0) or 0.0)
        if unc <= 0:
            continue
        a, b = frame_row(frame, m["value"] + unc), frame_row(frame, m["value"] - unc)
        lo_r, hi_r = max(min(a, b), top), min(max(a, b), bot)
        if hi_r - lo_r < 1.6:
            continue                    # sub-pixel: the hairline already says exact
        base = PANE_END if m.get("kind") == "end" else PANE_MARK
        d.rectangle([x0, lo_r, x1, hi_r], fill=base + (56,))

    def chip(row, text, colour):
        tw = d.textlength(text, font=f)
        ty = row + 4 if row - top < f.size * 2.2 else row - f.size - 5
        d.rectangle([x1 - tw - 14, ty - 2, x1 - 2, ty + f.size + 3], fill=colour)
        d.text((x1 - tw - 8, ty), text, font=f, fill=(255, 255, 255, 255))

    for row, m in rows:
        end = m.get("kind") == "end"
        colour = PANE_END if end else PANE_MARK[:3] + (alpha,)
        if end:
            # A white halo so the rule survives crossing the part, which on a
            # light clay render it otherwise does not.
            d.line([(x0, row), (x1, row)], fill=(255, 255, 255, 210), width=7)
        d.line([(x0, row), (x1, row)], fill=colour, width=4 if end else width)
        if id(m) in labelled:
            # The same four decimals the level table prints, so the picture
            # and the text never show one level as two different numbers.
            shown = f"{float(m['value']):.4f}".rstrip("0").rstrip(".")
            chip(row, f"{m.get('name', '?')} {shown}",
                 PANE_END if end else PANE_MARK)

    if offset is not None:
        row = frame_row(frame, offset)
        if top - 2 <= row <= bot + 2:
            d.line([(x0, row), (x1, row)], fill=(255, 255, 255, 210), width=7)
            d.line([(x0, row), (x1, row)], fill=PANE_LEVEL, width=3)
            chip(row, f"{'XYZ'[frame['kv']]} = {offset:.4g}", PANE_LEVEL)

    return np.asarray(pil), hidden


def ortho_level_views(mesh, plane, offset=None, out_path=None, marks=None,
                      tile=440, caption=None, edges=False, window="bounds"):
    """Stage 3b on the clay panes: the four edge-on views, levels drawn on them.

    The four views are chosen exactly as before -- the directions in which the
    section plane is edge-on, so a level is a horizontal line -- but the panes
    come from `visualize_orthographic` rather than from a depth-shaded pass on
    black. Passing the section normal as that module's `up` axis is what selects
    them: its own layout puts `up` on screen-up and gives the four side panes to
    the two remaining axes, which are precisely the four wanted here. The two it
    leaves out are +/- the normal, the pair that cannot show a level at all.

    Each pane arrives already carrying its own numbered grid, so nothing is
    added around it -- only the levels, over it.

    `window="bounds"` rather than that module's default: this stage exists to
    resolve a position along the normal, and the default window runs each axis
    from the origin, so a part sitting away from it spends the gap on empty
    frame -- 66% of the normal axis on the first part this was tried on. It also
    means a part at a negative coordinate is drawn at all, which the default
    would leave outside the frame while the level rules pointed at nothing.
    """
    from orthographic_views import pane_frames, visualize_orthographic

    from .section import PLANES

    spec = PLANES[plane]
    a = spec["axis"]
    axis_name = "XYZ"[a]

    panes, titles = visualize_orthographic(mesh, mosaic=False, up=axis_name,
                                           max_side=tile, edges=edges,
                                           window=window)
    frames = pane_frames(mesh, up=axis_name, max_side=tile, window=window)

    keys = [f"{'+' if s > 0 else '-'}{'XYZ'[ax]}"
            for ax in (spec["u"], spec["v"]) for s in (1, -1)]

    tiles, hidden_total = [], 0
    for key in keys:
        frame = frames[key]
        # The pane's vertical axis must be the section normal, or a level is not
        # a horizontal line and this whole stage means nothing. It is, by
        # construction -- assert it rather than trust the construction.
        assert frame["kv"] == a, f"{key} shows {'XYZ'[frame['kv']]} up, not {axis_name}"
        img, hidden = _draw_levels_on_pane(panes[key], frame, marks, offset)
        hidden_total += hidden
        tiles.append(img)

    if caption is None:
        lo, hi = mesh.bounds[:, a]
        where = (f"at {axis_name} = {offset:.4g}   " if offset is not None
                 else "choosing the offset   ")
        widest = max((float(m.get("uncertainty", 0.0) or 0.0)
                      for m in (marks or [])), default=0.0)
        band = (f"\na level's shaded band is how well it is LOCATED: widest here "
                f"+/-{widest:g}. No visible band = exact to under a pixel."
                if widest > 0 else "")
        thinned = (f"   ({hidden_total} label(s) suppressed where rules crowd; "
                   f"every rule is drawn)" if hidden_total else "")
        caption = (f"section plane {plane} (normal {axis_name}) {where}"
                   f"range [{lo:g}, {hi:g}]\n"
                   f"each tile carries its own axes and grid in model units; "
                   f"green = part ends, violet = feature levels"
                   f"{', orange = chosen level' if offset is not None else ''}"
                   f"{band}{thinned}")
    return _tile_panes(tiles, caption, out_path)


def _tile_panes(tiles, caption, out_path, pad=12):
    """2x2 of panes that already carry their own titles, plus a caption strip.

    Deliberately not `composite`: that draws a dark name chip into each tile's
    top-left corner, which is where these panes print their own title.
    """
    from PIL import Image, ImageDraw

    f = _pane_font(20)
    rows = [tiles[0:2], tiles[2:4]]
    row_w = [sum(t.shape[1] for t in r) + pad * (len(r) + 1) for r in rows]
    row_h = [max(t.shape[0] for t in r) for r in rows]
    W = max(row_w)

    lines = _wrap_caption(caption, f, W - pad * 2, ImageDraw.Draw(
        Image.new("RGB", (8, 8))))
    header = 16 + 24 * len(lines)
    H = header + sum(row_h) + pad * (len(rows) + 1)

    canvas = Image.new("RGB", (W, H), "#f4f5f7")
    draw = ImageDraw.Draw(canvas)
    for i, line in enumerate(lines):
        draw.text((pad, 8 + i * 24), line, font=f, fill="#2b3038")

    y = header + pad
    for r, h in zip(rows, row_h):
        x = pad
        for t in r:
            canvas.paste(Image.fromarray(t), (x, y))
            draw.rectangle([x, y, x + t.shape[1] - 1, y + t.shape[0] - 1],
                           outline="#c9ced6", width=1)
            x += t.shape[1] + pad
        y += h + pad

    canvas.save(out_path)
    return out_path



def comparison_sheets(target, built, out_dir, kind, tile=440, ortho_up="Y",
                      window="bounds"):
    """The two deviation sheets for one view family, written to `out_dir`.

    `kind` is "iso" or "ortho"; returns ([(path, "target"), (path, "predict")],
    thresholds) -- the band distances in model units, which is the only form in
    which "slight" and "large" mean anything to a caller working in mm.

    Two images rather than one, because the measurement is one-sided. Every
    triangle is coloured by its distance to the OTHER mesh's surface, so
    geometry the prediction left out has no triangle of its own to paint and
    appears only on the target sheet, while geometry it invented appears only on
    the prediction sheet. One sheet would silently answer half the question.

    Both are drawn to one ruler -- one framing, one window, one pair of
    thresholds, all measured over the two meshes together -- so the sheets can
    be flipped between and the same physical error is the same colour and lands
    in the same place on each.
    """
    from pathlib import Path

    from comparison import FRACTIONS, comparative_mesh_vis, max_axis_span
    from isometric_views import write_png

    sheets = comparative_mesh_vis(target, built, max_side=tile, up="Z",
                                  ortho_up=ortho_up, window=window)
    span = max_axis_span([pv.wrap(target), pv.wrap(built)])
    out = []
    for name in ("target", "predict"):
        path = Path(out_dir) / f"{name}_{kind}.png"
        write_png(str(path), sheets[f"{name}_{kind}"])
        out.append((path, name))
    return out, [f * span for f in FRACTIONS]


# --------------------------------------------------------- surface segmentation

# One hue family per kind of surface, so a glance says "that band is a torus"
# before the label is read, and neighbours of one kind still differ in shade.
# Deliberately none of the target/reconstruction colours: green, red and yellow
# mean target, built and agreement everywhere else, and this picture is neither.
SURFACE_HUES = {"plane": 214, "cylinder": 28, "cone": 272, "sphere": 176,
                "torus": 322, "revolve": 110, "freeform": None}
_SHADES = ((0.52, 0.55), (0.40, 0.72), (0.60, 0.40), (0.34, 0.60), (0.56, 0.68))


def _surface_rgb(kind, k):
    import colorsys
    hue = SURFACE_HUES.get(kind)
    s, v = _SHADES[k % len(_SHADES)]
    if hue is None:
        g = 0.42 + 0.08 * (k % 3)
        return np.array([g, g, g])
    h = ((hue + (k % 4 - 1.5) * 7) % 360) / 360.0
    return np.array(colorsys.hsv_to_rgb(h, s, 0.55 + 0.35 * v))


def surface_views(mesh, seg, out_path, cfg: RenderConfig = None, named=None):
    """Four views from above and two from below, each surface in its own colour,
    the boundaries between surfaces drawn, and the named ones labelled where
    they can be seen.

    Labels are placed on a point OF the surface and hidden when that point is
    behind something, so a name in the picture is always on the face it names.
    """
    cfg = cfg or RenderConfig()
    named = set(named or [s.name for s in seg.surfaces])
    poly = pv.wrap(mesh)
    rgb = np.zeros((len(mesh.faces), 3))
    for k, s in enumerate(seg.surfaces):
        rgb[s.faces] = _surface_rgb(s.kind, k)
    poly.cell_data["rgb"] = (rgb * 255).astype(np.uint8)

    # Boundaries: mesh edges whose two faces belong to different surfaces.
    adj = mesh.face_adjacency
    cut = seg.labels[adj[:, 0]] != seg.labels[adj[:, 1]]
    edges = mesh.face_adjacency_edges[cut]
    lines = None
    if len(edges):
        cells = np.column_stack([np.full(len(edges), 2), edges]).ravel()
        lines = pv.PolyData(np.asarray(mesh.vertices, float), lines=cells)

    diag = float(np.linalg.norm(mesh.extents)) or 1.0
    pts, nrm, names = [], [], []
    tc = np.asarray(mesh.triangles_center)
    fn = np.asarray(mesh.face_normals)
    for s in seg.surfaces:
        if s.name not in named:
            continue
        c = tc[s.faces].mean(axis=0)
        f = s.faces[int(np.argmin(np.linalg.norm(tc[s.faces] - c, axis=1)))]
        pts.append(tc[f])
        nrm.append(fn[f])
        names.append(s.name)
    pts, nrm = np.array(pts).reshape(-1, 3), np.array(nrm).reshape(-1, 3)

    def shot(direction):
        p = pv.Plotter(off_screen=True, window_size=(cfg.tile, cfg.tile))
        p.background_color = cfg.background
        p.add_mesh(poly, scalars="rgb", rgb=True, smooth_shading=False,
                   ambient=0.35, diffuse=0.7, specular=0.05)
        if lines is not None:
            actor = p.add_mesh(lines, color="#15181c", line_width=1.6,
                               lighting=False)
            mapper = actor.GetMapper()
            mapper.SetResolveCoincidentTopologyToPolygonOffset()
            mapper.SetRelativeCoincidentTopologyLineOffsetParameters(-1.0, -6.0)
        if cfg.lights:
            _add_lighting(p)
        p.camera.SetParallelProjection(True)
        _frame_bbox(p, mesh, direction, margin=1.06)
        img = p.screenshot(return_img=True)
        cam = p.camera
        frame = (np.array(cam.position), np.array(cam.focal_point),
                 np.array(cam.up), float(cam.parallel_scale))
        p.close()
        img = _label_surfaces(img, mesh, pts, nrm, names, frame, diag)
        if cfg.gnomon:
            img = draw_gnomon(img, direction)
        return img

    above = [(shot(d), lab) for lab, d in cfg.views]
    below = [(shot(d), lab) for lab, d in VIEWS_UNDER[:2]]
    caption = ("faces: each named surface in its own colour -- blues planes, "
               "oranges cylinders, violets cones, teals spheres, pinks tori, "
               "greens revolves, greys freeform\nblack lines are where one surface ends and the "
               "next begins; labels are the F names in the text")
    return composite_groups([("from above", above, 4), ("from below", below, 2)],
                            caption, out_path)


def _frame_bbox(p, mesh, direction, margin=1.06):
    """Fill the tile with the part: frame the projected bounding box, not the
    bounding sphere, which leaves a long part a third of the width."""
    lo, hi = np.asarray(mesh.bounds[0], float), np.asarray(mesh.bounds[1], float)
    centre = (lo + hi) / 2
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    up = np.array([0.0, 0.0, 1.0])
    if abs(d @ up) > 0.999:
        up = np.array([0.0, 1.0, 0.0])
    r = np.cross(-d, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, -d)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                        for z in (lo[2], hi[2])]) - centre
    half = max(float(np.abs(corners @ r).max()), float(np.abs(corners @ u).max()))
    diag = float(np.linalg.norm(hi - lo)) or 1.0
    p.camera_position = [tuple(centre + d * diag * 4.0), tuple(centre), tuple(up)]
    p.camera.parallel_scale = half * margin
    p.reset_camera_clipping_range()


def _label_font(size):
    from PIL import ImageFont
    cands = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
             "/System/Library/Fonts/Supplemental/Arial Bold.ttf"]
    try:
        from matplotlib import font_manager
        cands.append(font_manager.findfont("DejaVu Sans:bold"))
    except Exception:                                              # noqa: BLE001
        pass
    for c in cands:
        try:
            return ImageFont.truetype(c, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _label_surfaces(img, mesh, pts, nrm, names, frame, diag):
    """Write each surface's name on the picture where that surface is seen.

    Projected by hand from the camera's own frame and drawn after the render,
    rather than left to VTK's label actor, which drew nothing at all under the
    off-screen OSMesa build this runs on. A name whose point is hidden behind
    other geometry -- a ray from it towards the camera hits the part -- is not
    drawn in that view, so a label is always on the face it names.
    """
    if not len(pts):
        return img
    from PIL import Image, ImageDraw
    pos, focal, up, scale = frame
    f = focal - pos
    f /= np.linalg.norm(f)
    u = up - (up @ f) * f
    u /= np.linalg.norm(u)
    r = np.cross(f, u)
    H, W = img.shape[:2]
    lift = pts + nrm * diag * 1e-3 - f * diag * 1e-3
    try:
        hidden = mesh.ray.intersects_any(lift, np.repeat(-f[None, :], len(lift), 0))
    except Exception:                                              # noqa: BLE001
        hidden = np.zeros(len(lift), bool)
    facing = nrm @ f < 0
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    fnt = _label_font(max(11, H // 34))
    for P, name, hid, face in zip(pts, names, hidden, facing, strict=True):
        if hid or not face:
            continue
        x = W / 2 + ((P - focal) @ r) / scale * (H / 2)
        y = H / 2 - ((P - focal) @ u) / scale * (H / 2)
        if not (0 <= x < W and 0 <= y < H):
            continue
        tw = draw.textlength(name, font=fnt)
        th = fnt.size if hasattr(fnt, "size") else 11
        box = [x - tw / 2 - 3, y - th / 2 - 2, x + tw / 2 + 3, y + th / 2 + 3]
        draw.rectangle(box, fill=(32, 36, 42))
        draw.text((x - tw / 2, y - th / 2), name, font=fnt, fill=(255, 255, 255))
    return np.asarray(im)


def section_diff_plot(target_polys, built_polys, regions, u_name, v_name, title,
                      out_path, fill=False):
    """`section --built`: the target's section against the reconstruction's at
    the same cut. Outlines always (target solid, reconstruction dashed red);
    with `fill` the difference is painted -- green = in the target only
    (MISSING), red = in the reconstruction only (EXTRA), grey = both -- the
    colours `overview --built` and `residual` use. Each listed region carries
    its name (M1, X1, ...) so the picture and the table point at the same
    patch."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath
    from shapely.ops import unary_union

    def paint(ax, geom, colour, alpha):
        for g in getattr(geom, "geoms", [geom]):
            if g.is_empty or g.geom_type != "Polygon" or g.area <= 0:
                continue
            verts, codes = [], []
            for ring in [g.exterior, *g.interiors]:
                c = np.asarray(ring.coords)
                verts += c.tolist()
                codes += [MPath.MOVETO] + [MPath.LINETO] * (len(c) - 2) + [MPath.CLOSEPOLY]
            ax.add_patch(PathPatch(MPath(verts, codes), facecolor=colour,
                                   edgecolor="none", alpha=alpha))

    def outline(ax, polys, **kw):
        for p in polys:
            for g in getattr(p, "geoms", [p]):
                for ring in [g.exterior, *g.interiors]:
                    c = np.asarray(ring.coords)
                    ax.plot(c[:, 0], c[:, 1], **kw)

    tgt = unary_union(target_polys) if target_polys else None
    got = unary_union(built_polys) if built_polys else None
    fig, ax = plt.subplots(figsize=(9.5, 9.5), dpi=110)
    if fill:
        if tgt is not None and got is not None:
            paint(ax, tgt.intersection(got), "#b8bcc4", 1.0)
        for _name, kind, geom, _ in regions:
            paint(ax, geom, "#1a9e3a" if kind == "missing" else "#c8281e", 0.85)
    outline(ax, target_polys, color="#12304f", lw=1.3)
    outline(ax, built_polys, color="#c8281e", lw=1.0, ls="--")
    for name, kind, geom, _ in regions:
        c = geom.representative_point()
        ax.annotate(name, (c.x, c.y), fontsize=9, ha="center", va="center",
                    color="white", bbox=dict(boxstyle="round,pad=0.2", lw=0,
                                             fc="#1a9e3a" if kind == "missing" else "#c8281e"))
    ax.set_aspect("equal")
    ax.grid(True, lw=0.3)
    ax.set_xlabel(f"{u_name} (mm)")
    ax.set_ylabel(f"{v_name} (mm)")
    legend = ("green = MISSING (target only), red = EXTRA (built only), grey = both"
              if fill else "target solid, built dashed red; --fill paints the difference")
    ax.set_title(title + "\n" + legend, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
