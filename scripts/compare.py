"""Score and visualise a reconstruction against its ground-truth program.

Three panels, one camera: the target, the prediction, and the prediction
coloured by signed distance to the target. The third panel is the one that
matters -- a volumetric score can look excellent while a feature sits in
completely the wrong place, so the deviation map is what makes a "0.99" either
trustworthy or a warning.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np


def run_program(path):
    import cadquery as cq
    ns = {"show_object": lambda o, *a, **k: None, "debug": lambda *a, **k: None}
    exec(compile(Path(path).read_text(), str(path), "exec"), ns)
    for name in ("result", "r", "part", "shape"):
        v = ns.get(name)
        if isinstance(v, cq.Workplane | cq.Shape | cq.Assembly):
            return v
    raise RuntimeError(f"no result in {path}")


MESH_SUFFIXES = (".stl", ".obj", ".ply", ".off", ".glb", ".gltf", ".3mf")


def declared_frame(path):
    """The `_FRAME` a program declares (4x4), or None.

    `sketch` writes a program in the frame `ingest` put the target in, and says
    so in `_FRAME`. Scoring it against the mesh on disk without undoing that
    charges the program for a move the harness made -- export_stl.py undoes
    it, and so does this file.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from mesh2cad.frame import program_frame
    T, _how = program_frame(path)
    return T


def load_target(path):
    """A target is either a program that builds it or a mesh already on disk.

    Agent runs are scored against a mesh the model reconstructed from, and for
    most parts no ground-truth program exists at all.
    """
    p = Path(path)
    if p.suffix.lower() in MESH_SUFFIXES:
        import trimesh
        m = trimesh.load(p, force="mesh")
        m.merge_vertices()
        return m
    return program_mesh(p)


def program_mesh(path):
    """A program's solid as `sketch --check` and export_stl.py mesh it: each
    feature meshed and the features combined -- mesh2cad.build's load_program --
    not the compound tessellated in one piece, which can count overlapping
    features twice or come out open. Its notes go to stderr: stdout is this
    script's JSON."""
    import contextlib
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from mesh2cad.build import load_program
    with contextlib.redirect_stdout(sys.stderr):
        return load_program(path)


def solid_volume(path):
    """The volume of the program's own solid, its solids fused (a compound of
    overlapping, unfused features would count the overlap twice), or None
    when it does not build."""
    try:
        wp = run_program(path)
        solids = wp.solids().vals() if hasattr(wp, "solids") else []
        if len(solids) > 1:
            return float(solids[0].fuse(*solids[1:]).Volume())
        v = wp.val() if hasattr(wp, "val") else wp
        return float(v.Volume())
    except Exception:                                              # noqa: BLE001
        return None


def oriented(mesh, label, notes):
    """`mesh` with outward normals and a positive volume, said when it had to
    be turned."""
    import trimesh
    if mesh.is_watertight and mesh.volume < 0:
        before = float(mesh.volume)
        mesh.invert()
        notes.append(f"the {label} mesh was inside out (volume {before:.6g}); turned "
                     f"outward before scoring")
    elif not mesh.is_watertight:
        before = float(mesh.volume)
        trimesh.repair.fix_normals(mesh)
        if mesh.volume < 0:
            mesh.invert()
        if abs(mesh.volume - before) > 1e-6 * max(abs(before), 1):
            notes.append(f"the {label} mesh is open; its normals were made consistent and "
                         f"outward (volume {before:.6g} -> {mesh.volume:.6g})")
    return mesh


def tessellate(wp, tol=0.01):
    import cadquery as cq
    import trimesh
    shape = wp.toCompound() if hasattr(wp, "toCompound") else (
        wp.val() if hasattr(wp, "val") else wp)
    f = tempfile.mktemp(suffix=".stl")
    cq.exporters.export(shape, f, tolerance=tol, angularTolerance=0.1)
    m = trimesh.load(f, force="mesh")
    os.unlink(f)
    m.merge_vertices()
    return m


# Cells along the part's longest side, to start with. Column filling costs one
# line per column, not one ray per point, so the grid can start fine and refine
# while it misjudges the target's own volume by more than GRID_ERR_PCT -- thin
# sheet parts need it.
VOXEL_CELLS = 192
VOXEL_CELLS_MAX = 768
GRID_ERR_PCT = 1.0
# Irregular sub-cell offsets, so no column or cell centre lies on an
# axis-aligned face.
JITTER = (0.137, 0.219, 0.311)


def _lattice(lo, hi, pitch, jitter):
    return [np.arange(lo[i] + pitch * jitter[i], hi[i], pitch) for i in range(3)]


def _fill_along(mesh, axes, k):
    """Occupancy of the lattice by casting one line per column along axis k.

    Every triangle is projected onto the plane across k; each column centre it
    covers gets the triangle's height there as a crossing, and a cell is inside
    when an odd number of crossings lie below its centre. One pass over the
    triangles replaces per-point ray casts (trimesh `contains`, no embree here),
    which are far too slow on meshes with hundreds of thousands of faces.
    """
    i, j = [a for a in range(3) if a != k]
    ai, aj, ak = axes[i], axes[j], axes[k]
    tri = np.asarray(mesh.triangles, float)
    P, Q, Z = tri[:, :, i], tri[:, :, j], tri[:, :, k]
    # edge functions of the projected triangle; drop the edge-on ones
    area = (P[:, 1] - P[:, 0]) * (Q[:, 2] - Q[:, 0]) - (P[:, 2] - P[:, 0]) * (Q[:, 1] - Q[:, 0])
    keep = np.abs(area) > 1e-14 * max(1.0, float(np.abs(area).max()))
    P, Q, Z, area = P[keep], Q[keep], Z[keep], area[keep]
    pi0, pq0 = ai[0], aj[0]
    pitch = ai[1] - ai[0] if len(ai) > 1 else 1.0
    c0 = np.ceil((P.min(1) - pi0) / pitch).astype(int).clip(0, len(ai))
    c1 = np.floor((P.max(1) - pi0) / pitch).astype(int).clip(-1, len(ai) - 1)
    r0 = np.ceil((Q.min(1) - pq0) / pitch).astype(int).clip(0, len(aj))
    r1 = np.floor((Q.max(1) - pq0) / pitch).astype(int).clip(-1, len(aj) - 1)
    ni, nj = np.maximum(c1 - c0 + 1, 0), np.maximum(r1 - r0 + 1, 0)
    n = ni * nj
    tid = np.repeat(np.arange(len(n)), n)
    off = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
    ci = c0[tid] + off // np.maximum(nj[tid], 1)
    cj = r0[tid] + off % np.maximum(nj[tid], 1)
    x, y = ai[ci], aj[cj]
    # barycentric weights of the column centre in its triangle
    p, q, z, ar = P[tid], Q[tid], Z[tid], area[tid]
    w0 = ((p[:, 1] - x) * (q[:, 2] - y) - (p[:, 2] - x) * (q[:, 1] - y)) / ar
    w1 = ((p[:, 2] - x) * (q[:, 0] - y) - (p[:, 0] - x) * (q[:, 2] - y)) / ar
    w2 = 1.0 - w0 - w1
    hit = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
    col = (ci * len(aj) + cj)[hit]
    zh = (w0 * z[:, 0] + w1 * z[:, 1] + w2 * z[:, 2])[hit]
    # cells below how many crossings, per column: one sorted key per crossing
    span = float(ak[-1] - ak[0]) + 4 * pitch + 1.0
    key = np.sort(col * span + (zh - ak[0] + pitch))
    ncol = len(ai) * len(aj)
    zk = ak - ak[0] + pitch
    inside = np.empty((ncol, len(ak)), dtype=bool)
    step = max(1, 4_000_000 // max(len(ak), 1))     # bounds the int64 temporaries
    for c in range(0, ncol, step):
        cols = np.arange(c, min(c + step, ncol))
        below = np.searchsorted(key, cols[:, None] * span + zk[None, :])
        start = np.searchsorted(key, cols * span)
        inside[c:c + len(cols)] = (below - start[:, None]) % 2 == 1
    inside = inside.reshape(len(ai), len(aj), len(ak))
    order = np.argsort([i, j, k])
    return np.transpose(inside, order)


def occupancy(mesh, axes):
    """Majority of the three column fills. On a closed mesh all three agree;
    where a mesh is open or self-intersecting one direction's parity breaks
    and the other two outvote it."""
    votes = sum(_fill_along(mesh, axes, k).astype(np.uint8) for k in range(3))
    return votes >= 2


def to_surface(pts, surface):
    """Exact unsigned distance from each point to a mesh surface (VTK)."""
    import pyvista as pv
    d = pv.PolyData(np.asarray(pts, float)).compute_implicit_distance(surface)
    return np.abs(np.asarray(d["implicit_distance"], float))


def _stats(d, prefix):
    return {f"{prefix}_max": float(d.max()), f"{prefix}_mean": float(d.mean()),
            f"{prefix}_rms": float(np.sqrt((d ** 2).mean())),
            f"{prefix}_p95": float(np.percentile(d, 95))}


def _grid_err(occ, pitch, mesh):
    """|voxel volume - exact volume| of a closed mesh, in % (nan if open)."""
    if not mesh.is_watertight:
        return float("nan")
    return (abs(float(occ.sum()) * pitch ** 3 - float(mesh.volume))
            / max(abs(float(mesh.volume)), 1e-12) * 100.0)


def metrics(gt, rec, cells=VOXEL_CELLS, n_samples=20000, seed=0):
    import pyvista as pv
    import trimesh

    # Occupancy on ONE shared lattice spanning both parts.
    #
    # Voxelising each mesh separately and comparing sparse_indices is wrong:
    # each VoxelGrid carries its own origin, so the comparison is of relative
    # shape and is completely blind to translation -- a purely translated copy
    # would score a perfect 1.0.
    while True:
        pitch = float(max(gt.extents) / cells)
        lo = np.minimum(gt.bounds[0], rec.bounds[0]) - pitch
        hi = np.maximum(gt.bounds[1], rec.bounds[1]) + pitch
        axes = _lattice(lo, hi, pitch, JITTER)
        og = occupancy(gt, axes)
        grid_err = _grid_err(og, pitch, gt)
        if not grid_err > GRID_ERR_PCT or cells >= VOXEL_CELLS_MAX:
            break
        cells *= 2
    orr = occupancy(rec, axes)
    inter = int(np.count_nonzero(og & orr))
    union = int(np.count_nonzero(og | orr))
    iou = inter / union if union else 0.0

    # grid_err: how well the grid measures the target itself (voxel volume
    # against the exact volume, closed targets only). It understates IoU noise
    # -- errors cancel within one solid but not between two -- so read IoU
    # differences of about a cell layer as ties.

    # Distances both ways, from area-weighted samples to the other SURFACE.
    # One-way (prediction -> target) cannot see geometry the prediction left out.
    rng = np.random.default_rng(seed)
    pg, _ = trimesh.sample.sample_surface(gt, n_samples, seed=int(rng.integers(1 << 30)))
    pr, _ = trimesh.sample.sample_surface(rec, n_samples, seed=int(rng.integers(1 << 30)))
    sg, sr = pv.wrap(gt), pv.wrap(rec)
    d_rg, d_gr = to_surface(pr, sg), to_surface(pg, sr)

    # Signed distance of the prediction's vertices, for the deviation panel:
    # negative inside the target, positive outside (VTK, so it no longer
    # costs a trimesh proximity query per vertex).
    sd = np.asarray(pv.PolyData(np.asarray(rec.vertices, float))
                    .compute_implicit_distance(sg)["implicit_distance"], float)
    sd = np.nan_to_num(sd)

    bbox_shift = np.asarray(rec.bounds[0], float) - np.asarray(gt.bounds[0], float)
    surf = surface_metrics(gt, rec)
    diag = float(np.linalg.norm(gt.bounds[1] - gt.bounds[0]))
    cd = float(d_rg.mean() + d_gr.mean())

    return {
        "voxel_iou": iou,
        "voxel_grid_err_pct": grid_err,
        "voxel_cells": int(cells),
        "voxel_pitch": pitch,
        "n_grid": int(og.size),
        # Reported explicitly: a rigid offset is invisible in a shape-only
        # comparison and is exactly the error class that slipped through before.
        "bbox_shift": [round(float(x), 4) for x in bbox_shift],
        "bbox_shift_mag": round(float(np.linalg.norm(bbox_shift)), 4),
        "vol_gt": float(gt.volume),
        "vol_rec": float(rec.volume),
        "vol_err_pct": float(abs(rec.volume - gt.volume) / max(gt.volume, 1e-9) * 100),
        # two-sided: the larger of the two directions
        "max_dev": float(max(d_rg.max(), d_gr.max())),
        "mean_dev": float(0.5 * (d_rg.mean() + d_gr.mean())),
        "rms_dev": float(np.sqrt(0.5 * ((d_rg ** 2).mean() + (d_gr ** 2).mean()))),
        **_stats(d_rg, "dev_pred_to_gt"),     # invented geometry
        **_stats(d_gr, "dev_gt_to_pred"),     # missing geometry
        "signed": sd,
        **surf,
        # point-to-surface Chamfer / Hausdorff, replacing the sample-to-sample
        # ones from surface_metrics (kept there as *_sample_*): those sit at
        # the sampling floor and cannot rank close reconstructions.
        "cd_mm": cd,
        "cd_norm_x1000": cd / diag * 1000.0,
        "hd_mm": float(max(d_rg.max(), d_gr.max())),
        "hd_norm_x1000": float(max(d_rg.max(), d_gr.max())) / diag * 1000.0,
        "hd95_mm": float(max(np.percentile(d_rg, 95), np.percentile(d_gr, 95))),
        "n_dist_samples": n_samples,
    }


def surface_metrics(gt, rec, n_samples=8192, feature_angle=30.0, seed=0):
    """The point-set metrics standard in the CAD-reconstruction literature.

    Chamfer and Hausdorff are reported both in millimetres and normalised by the
    ground truth's bounding-box diagonal, because papers differ on whether they
    normalise and an unlabelled CD is not comparable across either convention.

    Edge Chamfer Distance is computed over crease edges only. It is the metric
    that actually distinguishes a CAD reconstruction from a smooth surface fit:
    two shapes can agree closely on surface samples while putting the sharp
    edges -- the things a CAD program is made of -- in different places.
    """
    import trimesh
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    pg, ig = trimesh.sample.sample_surface(gt, n_samples, seed=int(rng.integers(1 << 30)))
    pr, ir = trimesh.sample.sample_surface(rec, n_samples, seed=int(rng.integers(1 << 30)))

    tg, tr = cKDTree(pg), cKDTree(pr)
    d_rg, idx_rg = tg.query(pr)   # prediction -> ground truth
    d_gr, idx_gr = tr.query(pg)   # ground truth -> prediction

    diag = float(np.linalg.norm(gt.bounds[1] - gt.bounds[0]))

    # Normal consistency: how well the surfaces agree in orientation, not just
    # position. Catches a reconstruction that is close but inside-out or offset
    # across a thin wall.
    ng = gt.face_normals[ig]
    nr = rec.face_normals[ir]
    nc = 0.5 * (float(np.abs((nr * ng[idx_rg]).sum(1)).mean())
                + float(np.abs((ng * nr[idx_gr]).sum(1)).mean()))

    def edge_points(mesh, n):
        """Sample points along crease edges (dihedral angle > feature_angle)."""
        adj = mesh.face_adjacency
        if len(adj) == 0:
            return np.zeros((0, 3))
        ang = mesh.face_adjacency_angles
        sharp = mesh.face_adjacency_edges[ang > np.radians(feature_angle)]
        if len(sharp) == 0:
            return np.zeros((0, 3))
        a = mesh.vertices[sharp[:, 0]]
        b = mesh.vertices[sharp[:, 1]]
        lens = np.linalg.norm(b - a, axis=1)
        total = lens.sum()
        if total <= 0:
            return np.zeros((0, 3))
        # Distribute samples proportionally to edge length.
        counts = np.maximum(1, (lens / total * n).astype(int))
        segs = np.repeat(np.arange(len(sharp)), counts)
        t = np.random.default_rng(0).random(len(segs))[:, None]
        return a[segs] + (b[segs] - a[segs]) * t

    eg, er = edge_points(gt, n_samples), edge_points(rec, n_samples)
    if len(eg) and len(er):
        de_rg, _ = cKDTree(eg).query(er)
        de_gr, _ = cKDTree(er).query(eg)
        ecd = float(de_rg.mean() + de_gr.mean())
        ehd = float(max(de_rg.max(), de_gr.max()))
    else:
        ecd = ehd = float("nan")

    cd = float(d_rg.mean() + d_gr.mean())
    hd = float(max(d_rg.max(), d_gr.max()))
    hd95 = float(max(np.percentile(d_rg, 95), np.percentile(d_gr, 95)))

    return {
        "cd_sample_mm": cd,
        "hd_sample_mm": hd,
        "hd95_sample_mm": hd95,
        "ecd_mm": ecd,
        "ecd_norm_x1000": ecd / diag * 1000.0 if ecd == ecd else float("nan"),
        "ehd_mm": ehd,
        "normal_consistency": nc,
        "gt_bbox_diag_mm": diag,
        "n_samples": n_samples,
    }


def render(gt, rec, sd, out_path, tile=720, title=""):
    import pyvista as pv
    from PIL import Image, ImageDraw, ImageFont

    pg, pr = pv.wrap(gt), pv.wrap(rec)
    lo = np.minimum(gt.bounds[0], rec.bounds[0])
    hi = np.maximum(gt.bounds[1], rec.bounds[1])
    centre = (lo + hi) / 2

    # Look roughly across the longest axis, not down it. A fixed isometric
    # foreshortens a 91mm rod into a disc, which is the least informative view
    # available of the part we are trying to show.
    longest = int(np.argmax(hi - lo))
    d = np.array({0: (0.35, 1.0, 0.70), 1: (1.0, 0.35, 0.70),
                  2: (1.0, 0.80, 0.38)}[longest], dtype=float)
    up = np.array([0.0, 0.0, 1.0])
    if longest == 2:
        up = np.array([0.0, 0.0, 1.0])

    # Fit the projected bounding box rather than the bounding sphere, so the
    # part actually fills the panel.
    view = d / np.linalg.norm(d)
    right = np.cross(view, up)
    right /= np.linalg.norm(right)
    up_actual = np.cross(right, view)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) - centre
    half_r = float(np.abs(corners @ right).max())
    half_u = float(np.abs(corners @ up_actual).max())
    scale = max(half_u, half_r) * 1.10

    clay = dict(smooth_shading=True, split_sharp_edges=True, feature_angle=30.0,
                specular=0.16, specular_power=22, ambient=0.30, diffuse=0.76)

    # How far to stand the camera off. Under parallel projection the distance
    # changes nothing about the picture, only the clipping range, which
    # reset_camera recomputes anyway.
    standoff = float(np.linalg.norm(hi - lo)) + 1.0

    def shot(mesh_pv, **kw):
        p = pv.Plotter(off_screen=True, window_size=(tile, tile))
        p.background_color = "white"
        p.add_mesh(mesh_pv, **kw)
        p.remove_all_lights()
        for pos, inten, col in (((-1, -0.6, 1.2), 0.85, "#ffffff"),
                                ((1, 0.4, 0.35), 0.42, "#fff2e0"),
                                ((0.2, 1.0, -0.6), 0.28, "#dce6ff")):
            L = pv.Light(position=pos, light_type="camera light", intensity=inten)
            L.diffuse_color = col
            p.add_light(L)
        p.enable_anti_aliasing("fxaa")
        p.camera.SetParallelProjection(True)
        # A camera POSITION, not a direction. `view` on its own points out of
        # the origin, so the direction of projection came out as
        # centre - view -- which is the direction of the part's own offset from
        # the origin whenever that offset is large. A hex standoff spanning
        # Y = -61.5..12.3 was therefore photographed straight down its length,
        # exactly the view the code above goes to some trouble to avoid, and
        # the three panels showed a hexagon instead of a rod.
        p.camera_position = [tuple(centre + view * standoff), tuple(centre),
                             tuple(up)]
        p.reset_camera()
        p.camera.SetFocalPoint(*centre)
        p.camera.SetParallelScale(scale)
        img = p.screenshot(return_img=True)
        p.close()
        return img

    a = shot(pg, color="#8fa9c4", **clay)
    b = shot(pr, color="#e0a458", **clay)

    # Symmetric range so zero is the neutral midpoint, with a floor at the
    # tessellation noise level. Without the floor an exact reconstruction
    # autoscales 0.003mm of chord error into vivid red patches that read as real
    # error -- the scale must not make noise look like a defect.
    NOISE_FLOOR_MM = 0.05
    lim = max(float(np.abs(sd).max()), NOISE_FLOOR_MM)
    pr_c = pr.copy()
    pr_c["deviation"] = sd
    c = shot(pr_c, scalars="deviation", cmap="coolwarm", clim=(-lim, lim),
             show_scalar_bar=False, **clay)

    pad, header, barh = 14, 62, 54
    h, w = a.shape[:2]
    W, H = w * 3 + pad * 4, h + pad * 2 + header + barh
    canvas = Image.new("RGB", (W, H), "#f6f7f9")
    draw = ImageDraw.Draw(canvas)

    def font(sz):
        for n in ("/System/Library/Fonts/SFNSMono.ttf",
                  "/System/Library/Fonts/Supplemental/Arial.ttf"):
            try:
                return ImageFont.truetype(n, sz)
            except OSError:
                continue
        return ImageFont.load_default()

    f_t, f_l = font(26), font(20)
    draw.text((pad + 4, 16), title, font=f_t, fill="#1d2126")

    for i, (img, label, col) in enumerate([
            (a, "TARGET  (ground truth)", "#4a6b8a"),
            (b, "PREDICTED  (agent)", "#a06a1f"),
            (c, "DEVIATION  (predicted, coloured by signed distance)", "#7a3b6b")]):
        x = pad + i * (w + pad)
        y = header
        canvas.paste(Image.fromarray(img), (x, y))
        draw.rectangle([x, y, x + w - 1, y + h - 1], outline="#d3d8df", width=1)
        tw = draw.textlength(label, font=f_l)
        draw.rectangle([x + 8, y + 8, x + 22 + tw, y + 42], fill=col)
        draw.text((x + 15, y + 13), label, font=f_l, fill="#ffffff")

    # Colour bar under the deviation panel.
    bx, by = pad + 2 * (w + pad), header + h + 12
    bw, bh = w, 18
    # matplotlib 3.9 removed cm.get_cmap; the registry is the supported route.
    import matplotlib
    grad = matplotlib.colormaps["coolwarm"](np.linspace(0, 1, bw))[:, :3]
    bar = np.repeat((grad * 255).astype(np.uint8)[None, :, :], bh, axis=0)
    canvas.paste(Image.fromarray(bar), (bx, by))
    draw.rectangle([bx, by, bx + bw - 1, by + bh - 1], outline="#b8bec7", width=1)
    for frac, txt in ((0.0, f"-{lim:.3f} mm"), (0.5, "0"), (1.0, f"+{lim:.3f} mm")):
        t = f"{txt}"
        tw = draw.textlength(t, font=f_l)
        draw.text((bx + frac * bw - tw * frac, by + bh + 4), t, font=f_l, fill="#4a5058")

    canvas.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True,
                    help="the target: a mesh, or a program that builds one")
    ap.add_argument("--rec", required=True,
                    help="the reconstruction: a program, or a mesh already on disk")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--tile", type=int, default=720)
    args = ap.parse_args()

    notes = []
    gt = oriented(load_target(args.gt), "target", notes)
    rec = oriented(load_target(args.rec), "reconstruction", notes)
    # A program scored against a mesh goes back into the mesh's frame first.
    frame = None
    for mesh, path, other in ((rec, args.rec, args.gt), (gt, args.gt, args.rec)):
        if Path(path).suffix.lower() not in MESH_SUFFIXES and \
                Path(other).suffix.lower() in MESH_SUFFIXES:
            T = declared_frame(path)
            if T is not None:
                mesh.apply_transform(np.linalg.inv(T))
                frame = f"undid the _FRAME declared in {Path(path).name}"
    m = metrics(gt, rec)
    if frame:
        m["frame"] = frame
    if Path(args.rec).suffix.lower() not in MESH_SUFFIXES:
        # The volume scored beside the program's own solid's, so a scored mesh that
        # is not the delivered solid is visible.
        v = solid_volume(args.rec)
        T = declared_frame(args.rec)
        if v is not None and T is not None:
            v /= abs(float(np.linalg.det(np.asarray(T, float)[:3, :3])))
        m["vol_solid"] = v
        if v and abs(float(rec.volume) - v) > 0.05 * abs(v):
            # What is scored is what export_stl.py delivers: the features meshed
            # and combined. A B-rep that reads otherwise is broken -- unfused
            # overlaps counted twice, inside out, a feature lost in the fuse --
            # and a score of it would not be the delivered part.
            m["volume_mismatch"] = (f"the program's B-rep solid reads {v:.6g} mm^3, its "
                                    f"features meshed and combined (scored here, and what "
                                    f"export_stl.py writes) {float(rec.volume):.6g} "
                                    f"({(v / float(rec.volume) - 1):+.0%}): the B-rep is "
                                    f"unfused, inside out or lost a feature in the fuse")
    if notes:
        m["notes"] = notes
    sd = m.pop("signed")
    render(gt, rec, sd, args.out, tile=args.tile, title=args.title)

    m["image"] = args.out
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
