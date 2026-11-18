"""Stage 0: ingest, repair, and align.

Alignment is not cosmetic. The whole section-driven approach assumes the part's
natural planes are XY / YZ / ZX; if the part sits 7 degrees off-axis, every
section is an oblique cut through features that should have been rectangles, and
the model will spend a dozen turns discovering that the geometry is "weird"
rather than that the frame is wrong. So we find the part's own frame up front and
report the rotation we applied.

We also classify the part's construction regime here. Mechanical parts are
gears, springs and threaded fasteners as much as they are brackets, and a
helical spring's XY section is a scatter of disconnected ellipses that says
almost nothing. Better to detect that at ingest and route to a different
strategy than to hand the model a useless section and let it work out why.
"""

import hashlib
import os
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import trimesh


@dataclass
class Regime:
    """What construction strategy this part most likely wants."""
    kind: str                 # prismatic | revolve | helical | patterned | freeform
    confidence: float
    evidence: dict

    def as_dict(self):
        return asdict(self)


def load(path, repair=True):
    mesh = trimesh.load(path, force="mesh", process=False)
    if repair:
        # STL is a triangle soup, so an unmerged mesh always reports
        # non-watertight. Merge first, then drop the zero-area triangles that
        # CadQuery's tessellator leaves at sphere poles -- on most CAD exports
        # that pair of steps takes the mesh from "leaking" to watertight, and
        # the holes were never real.
        mesh.merge_vertices()
        areas = mesh.area_faces
        degenerate = areas <= 1e-12
        if degenerate.any():
            mesh.update_faces(~degenerate)
            mesh.remove_unreferenced_vertices()
            mesh.merge_vertices()
        if hasattr(mesh, "update_faces") and hasattr(mesh, "unique_faces"):
            mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals()
    return mesh


def normal_clusters(mesh, min_area_frac=0.005):
    """Area-weighted histogram of face normals, merged into clusters.

    Area weighting matters: a cylinder contributes hundreds of tiny facets whose
    normals fan out, while the one flat face we care about may be a single
    triangle pair. Weighting by area keeps the flats dominant.
    """
    normals = mesh.face_normals
    areas = mesh.area_faces
    total = areas.sum()

    # Fold antipodal normals together -- a slab's two faces define one direction,
    # so flip every normal into the hemisphere with a non-negative first
    # significant component.
    folded = normals.copy()
    lead = folded[np.arange(len(folded)), np.argmax(np.abs(folded), axis=1)]
    folded[lead < 0] *= -1

    clusters = []
    remaining = np.ones(len(normals), bool)
    order = np.argsort(-areas)
    for idx in order:
        if not remaining[idx]:
            continue
        n = folded[idx]
        same = remaining & (np.abs(folded @ n) > np.cos(np.radians(3.0)))
        area = float(areas[same].sum())
        remaining &= ~same
        if area / total >= min_area_frac:
            clusters.append({"normal": n.tolist(), "area": area, "frac": area / total})
        if not remaining.any():
            break
    clusters.sort(key=lambda c: -c["area"])
    return clusters


def align_to_axes(mesh, angle_tol=25.0, min_secondary_frac=0.05):
    """Rotate so the part's dominant flat directions land on X/Y/Z.

    Returns (mesh, 4x4 transform, report). Identity when the part is already
    aligned, which is the common case for CAD exports and therefore a useful
    sanity check on this function.
    """
    clusters = normal_clusters(mesh)
    report = {"n_clusters": len(clusters), "applied": False, "angle_deg": 0.0}

    if not clusters:
        return mesh, np.eye(4), report

    primary = np.array(clusters[0]["normal"], float)
    primary /= np.linalg.norm(primary)

    # Secondary must be near-perpendicular to the primary AND carry real area.
    #
    # The area floor is the important part. A tessellated cylinder contributes a
    # fan of facet normals all perpendicular to its axis, each with a tiny and
    # nearly identical area share. Without a floor, the first of those facets is
    # picked as a frame axis and the whole part is rotated by whatever arbitrary
    # angle that facet happens to sit at -- a rotated bolt circle that still
    # scores high IoU, a wrong answer that no downstream check would catch.
    candidates = []
    for c in clusters[1:]:
        v = np.array(c["normal"], float)
        v /= np.linalg.norm(v)
        if abs(v @ primary) < np.cos(np.radians(90 - 8.0)):
            candidates.append((v, c["frac"]))

    secondary = None
    for v, frac in candidates:
        if frac >= min_secondary_frac:
            secondary = v
            break

    if secondary is None:
        report["skipped_reason"] = (
            "no secondary planar cluster above "
            f"{min_secondary_frac:.0%} of area"
            + (f" (best candidate {max(f for _, f in candidates):.3%})"
               if candidates else " (no perpendicular clusters)"))
        return mesh, np.eye(4), report

    # A run of near-equal-area perpendicular clusters is a tessellated round
    # face, not a set of design faces; refuse to take a frame from it.
    fan = [f for _, f in candidates if abs(f - candidates[0][1]) <= 0.1 * candidates[0][1]]
    if len(fan) >= 4:
        report["skipped_reason"] = (
            f"{len(fan)} perpendicular clusters of near-equal area "
            f"({candidates[0][1]:.3%}) -- looks like a tessellated cylinder, not planar faces")
        return mesh, np.eye(4), report

    # Snap the primary to whichever axis it is already closest to, so an
    # already-aligned part maps to the identity instead of a 90 deg shuffle.
    axes = np.eye(3)
    tgt_p = axes[np.argmax(np.abs(primary @ axes.T))] * np.sign(primary @ axes[np.argmax(np.abs(primary @ axes.T))])
    rest = [a for a in axes if abs(a @ tgt_p) < 0.5]
    tgt_s = max(rest, key=lambda a: abs(secondary @ a))
    tgt_s = tgt_s * np.sign(secondary @ tgt_s)

    src = np.column_stack([primary, secondary - (secondary @ primary) * primary, np.zeros(3)])
    src[:, 1] /= np.linalg.norm(src[:, 1])
    src[:, 2] = np.cross(src[:, 0], src[:, 1])

    dst = np.column_stack([tgt_p, tgt_s - (tgt_s @ tgt_p) * tgt_p, np.zeros(3)])
    dst[:, 1] /= np.linalg.norm(dst[:, 1])
    dst[:, 2] = np.cross(dst[:, 0], dst[:, 1])

    R = dst @ src.T
    angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    report["angle_deg"] = float(angle)

    if angle < 0.5 or angle > angle_tol:
        # Either already aligned, or the "frame" we found is not credible.
        report["skipped_reason"] = "already aligned" if angle < 0.5 else "rotation exceeds tolerance"
        return mesh, np.eye(4), report

    T = np.eye(4)
    T[:3, :3] = R
    out = mesh.copy()
    out.apply_transform(T)
    report["applied"] = True
    return out, T, report


def detect_helix(mesh, axis_idx, n=64, min_turns=1.5, min_r2=0.97):
    """Test whether the part is a tube swept along a helix about `axis_idx`.

    The signature we look for is direct: slice perpendicular to the candidate
    axis, and track the angular position of the cross-section's centroid as the
    slice height rises. A helix advances that angle linearly with height, one
    full turn per pitch, so an unwrapped angle-vs-height regression is a straight
    line -- and its slope hands us the pitch for free.

    Restricting to single-loop slices is what makes this robust: a spring's
    ground flat ends produce multi-loop slices, and skipping them removes the
    end effects without any special-casing.
    """
    from .section import PLANES, sweep

    plane = {2: "XY", 0: "YZ", 1: "XZ"}[axis_idx]
    spec = PLANES[plane]
    centre = mesh.bounds.mean(axis=0)
    cu, cv = centre[spec["u"]], centre[spec["v"]]

    try:
        # on_fail=raise: a broken plane fails fast here; skipping or rebuilding
        # its cuts makes ingest several times slower.
        slices = sweep(mesh, plane, n=n, keep_polygons=True, on_fail="raise")
    except Exception:
        return {"is_helix": False, "reason": "sweep failed"}

    heights, angles, radii = [], [], []
    for s in slices:
        if s.n_loops != 1 or not s.polygons:
            continue
        c = s.polygons[0].centroid
        angles.append(np.arctan2(c.y - cv, c.x - cu))
        # How far the swept profile's centre sits from the axis -- the coil
        # radius, which is what a sweep needs and what the pitch alone cannot
        # give. Measured here because this is the one place the centroids are
        # already in hand; it costs nothing and was simply never kept.
        radii.append(float(np.hypot(c.x - cu, c.y - cv)))
        heights.append(s.offset)

    loops = "single"
    if len(heights) < 12:
        # A thread exported without its core thins to nothing at its root, and
        # its slices come apart into several loops, so few of them are single and
        # the helix cannot be refitted where the loop says to work. Then the
        # largest loop of each slice is followed.
        heights, angles, radii = [], [], []
        for s in slices:
            if not s.polygons:
                continue
            c = max(s.polygons, key=lambda p: p.area).centroid
            angles.append(np.arctan2(c.y - cv, c.x - cu))
            radii.append(float(np.hypot(c.x - cu, c.y - cv)))
            heights.append(s.offset)
        loops = "largest of each slice"
    if len(heights) < 12:
        return {"is_helix": False, "reason": f"only {len(heights)} slices with a loop"}

    heights = np.asarray(heights)
    theta = np.unwrap(np.asarray(angles))
    turns = float(abs(theta[-1] - theta[0]) / (2 * np.pi))

    def fit(h, t):
        slope, intercept = np.polyfit(h, t, 1)
        pred = slope * h + intercept
        ss_res = float(((t - pred) ** 2).sum())
        ss_tot = float(((t - t.mean()) ** 2).sum())
        return slope, (1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0)

    slope_all, r2_all = fit(heights, theta)

    # Real springs have closed/ground end turns at a much tighter pitch than the
    # body. A single regression over the whole part blends the two and reports
    # a pitch that matches neither, so fit the central portion for the body pitch
    # and keep the global fit only as the "is it a helix at all" evidence.
    core = slice(int(len(heights) * 0.15), int(np.ceil(len(heights) * 0.85)))
    if core.stop - core.start >= 8:
        slope, r2_core = fit(heights[core], theta[core])
        intercept = float(np.polyfit(heights[core], theta[core], 1)[1])
    else:
        slope, r2_core = slope_all, r2_all
        intercept = float(np.polyfit(heights, theta, 1)[1])

    pitch = float(2 * np.pi / slope) if abs(slope) > 1e-9 else None
    pitch_all = float(2 * np.pi / slope_all) if abs(slope_all) > 1e-9 else None
    ends_differ = (
        pitch is not None and pitch_all is not None
        and abs(pitch - pitch_all) / abs(pitch) > 0.02
    )

    ok = r2_all >= min_r2 and turns >= min_turns and abs(slope) > 1e-6
    # The helical stretch: every slice whose largest loop sits where the fitted
    # line says, multi-loop ones too. Its length over the pitch is the turn
    # count -- the single-loop slices alone skip ends by design and undercount
    # the turns -- and its ends are where the helix runs.
    lo_a, hi_a = float(mesh.bounds[0][axis_idx]), float(mesh.bounds[1][axis_idx])
    turns_long = turns
    if pitch:
        on = []
        for s_ in slices:
            if not s_.polygons:
                continue
            c = max(s_.polygons, key=lambda p_: p_.area).centroid
            miss = np.arctan2(c.y - cv, c.x - cu) - (slope * s_.offset + intercept)
            if abs((miss + np.pi) % (2 * np.pi) - np.pi) < 0.35:
                on.append(s_.offset)
        if len(on) >= 2:
            lo_a, hi_a = float(min(on)), float(max(on))
            turns_long = (hi_a - lo_a) / abs(pitch)
    axis_at = centre.copy()
    axis_at[axis_idx] = 0.0
    return {
        "is_helix": bool(ok),
        "axis": "XYZ"[axis_idx],
        "turns": round(turns_long, 3),
        "turns_tracked": round(turns, 3),
        "loops": loops,
        "centre": [round(float(v), 4) for v in axis_at],
        "extent": [round(lo_a, 4), round(hi_a, 4)],
        "r2": round(float(r2_all), 4),
        "r2_body": round(float(r2_core), 4),
        "pitch": round(pitch, 4) if pitch is not None else None,
        "pitch_whole_part": round(pitch_all, 4) if pitch_all is not None else None,
        # Median, not mean: a ground end turn sits at a different radius and
        # would drag an average off the body it is not part of.
        "radius": round(float(np.median(radii)), 4),
        "radius_spread": round(float(np.percentile(radii, 90)
                                     - np.percentile(radii, 10)), 4),
        "closed_ends": bool(ends_differ),
        # The angle is measured from u towards v, which turns about +axis only
        # when u x v = +axis: X x Y = Z and Y x Z = X, but X x Z = -Y. Read
        # plainly, every helix along Y would come out the other hand.
        "handedness": "right" if slope * (-1.0 if plane == "XZ" else 1.0) > 0 else "left",
        "n_slices_used": len(heights),
    }


# A thread is found where the outline of a cut across the axis has a
# bump that turns with the cut: the first harmonic of the outline's radius
# round the axis, which a hex or a bore does not have (their outlines repeat
# every 60 deg, or not at all), at least this share of the mean radius, in a
# run of cuts whose phase follows one line this well. Fine threads have bumps
# of only 1-2% of their radius, so the share is low; the phase line
# (THREAD_R2) is what keeps a part with no thread out.
THREAD_AMP = 0.01
THREAD_R2 = 0.995
THREAD_SLICES = 160
THREAD_MAX_FACES = 1_500_000


def _outline_harmonic(polys, cu, cv, bins=360):
    """(phase, amplitude / mean radius, mean radius) of the outline's radius
    about (cu, cv), its largest at each degree, or None if the outline does
    not go all the way round."""
    import shapely
    r = np.zeros(bins)
    def parts(gs):
        for g in gs:
            if hasattr(g, "geoms"):
                yield from parts(g.geoms)
            elif g.geom_type == "Polygon" and not g.is_empty:
                yield g
    for g in parts(polys):
        g = shapely.segmentize(g, max(g.length / 4000.0, 1e-4))
        for ring in (r for q in parts([g]) for r in [q.exterior, *q.interiors]):
            q = np.asarray(ring.coords) - (cu, cv)
            k = (np.degrees(np.arctan2(q[:, 1], q[:, 0])) % 360.0 * bins / 360.0).astype(int) % bins
            np.maximum.at(r, k, np.hypot(q[:, 0], q[:, 1]))
    ok = r > 0
    if ok.sum() < 0.85 * bins:
        return None
    ang = np.radians((np.arange(bins) + 0.5) * 360.0 / bins)
    mean = float(r[ok].mean()) or 1.0
    # sum r e^{+i phi}: its angle is where the bump is (e^{-i phi} gives minus it)
    c1 = (r[ok] * np.exp(1j * ang[ok])).sum() / ok.sum()
    return float(np.angle(c1)), float(abs(c1)) / mean, mean


def detect_threads(mesh, axis_idx, n=THREAD_SLICES, min_turns=1.5):
    """Helices on stretches of the part about `axis_idx`: a thread on a hex
    nozzle or a stud, which the centroid test of the whole part cannot see --
    the hex or the bore holds every centroid on the axis.

    The cut's outline has a bump where the thread crosses it, and the bump
    turns one way round the axis as the cut rises, a turn per pitch: the
    phase of the outline radius's first harmonic is a straight line in the
    height over the thread and nowhere else. Each run of cuts that has such
    a bump is fitted on its own, about the middle of its own cuts, and kept
    when the line holds (THREAD_R2) for at least `min_turns` turns."""
    from shapely.ops import unary_union

    from .section import PLANES, sweep

    if len(mesh.faces) > THREAD_MAX_FACES:
        return []
    plane = {2: "XY", 0: "YZ", 1: "XZ"}[axis_idx]
    spec = PLANES[plane]
    centre = mesh.bounds.mean(axis=0)
    try:
        slices = sweep(mesh, plane, n=n, keep_polygons=True, on_fail="skip")
    except Exception:                                              # noqa: BLE001
        return []
    rows = [(s.offset, s.polygons) for s in slices if s.polygons]
    if len(rows) < n // 2:
        return []
    cu, cv = centre[spec["u"]], centre[spec["v"]]
    marks = []
    for _, polys in rows:
        h = _outline_harmonic(polys, cu, cv)
        marks.append(h is not None and h[1] >= THREAD_AMP)
    idx = [i for i, m in enumerate(marks) if m]
    runs, cur = [], []
    for i in idx:
        if cur and i - cur[-1] > 2:
            runs.append(cur)
            cur = []
        cur.append(i)
    if cur:
        runs.append(cur)
    out = []
    for run in runs:
        if len(run) < 8:
            continue
        # about this stretch's own axis: the middle of its cuts
        cs = [unary_union(rows[i][1]).centroid for i in run]
        au, av = float(np.median([c.x for c in cs])), float(np.median([c.y for c in cs]))
        got = [(rows[i][0], _outline_harmonic(rows[i][1], au, av)) for i in run]
        got = [(h, g) for h, g in got if g is not None]
        if len(got) < 8:
            continue
        hs = np.array([h for h, _ in got])
        th = np.unwrap(np.array([g[0] for _, g in got]))
        slope, icpt = np.polyfit(hs, th, 1)
        pred = slope * hs + icpt
        ss = float(((th - th.mean()) ** 2).sum())
        r2 = 1.0 - float(((th - pred) ** 2).sum()) / ss if ss > 1e-12 else 0.0
        if abs(slope) < 1e-9 or r2 < THREAD_R2:
            continue
        pitch = float(2 * np.pi / abs(slope))
        lo, hi = float(hs.min()), float(hs.max())
        turns = (hi - lo) / pitch
        if turns < min_turns:
            continue
        axis_at = centre.copy()
        axis_at[spec["u"]], axis_at[spec["v"]], axis_at[axis_idx] = au, av, 0.0
        radii = [g[2] for _, g in got]
        out.append({
            "is_helix": True, "kind": "thread",
            "axis": "XYZ"[axis_idx],
            "turns": round(turns, 3), "turns_tracked": round(turns, 3),
            "loops": "thread",
            "centre": [round(float(v), 4) for v in axis_at],
            "extent": [round(lo, 4), round(hi, 4)],
            "r2": round(r2, 4), "r2_body": round(r2, 4),
            "pitch": round(pitch, 4), "pitch_whole_part": round(pitch, 4),
            "radius": round(float(np.median(radii)), 4),
            "radius_spread": round(float(np.percentile(radii, 90) - np.percentile(radii, 10)), 4),
            "closed_ends": False,
            # as detect_helix: the angle runs from u to v, about -Y on XZ
            "handedness": "right" if slope * (-1.0 if plane == "XZ" else 1.0) > 0 else "left",
            "n_slices_used": len(got),
        })
    return out


# The revolve probe read slab by slab along its axis: enough samples
# that each of the slabs has about a hundred, and a slab is round when that
# share of its samples lands back on the surface under both test rotations.
LOCAL_SAMPLES = 2000
LOCAL_SLABS = 20
ROUND_SLAB = 0.95


def revolve_bands(mesh, axis, n=LOCAL_SAMPLES, seed=None):
    """Where along `axis` the part is a solid of revolution about it.

    The probe's one number is the share of the WHOLE surface that survives the
    test rotations, and read as a verdict it fails a turned sleeve with a saddle
    cut across one end, though nine tenths of its length are exactly round.
    Scored slab by slab, such a sleeve reads 1.000 everywhere but the saddle.

    Returns runs along the axis, [{"round", "from", "to", "score", "frac"}],
    the edge between a round run and the next taken from the farthest sample
    that missed, not the slab grid; [] when the surface cannot be sampled.
    """
    try:
        pts, _ = trimesh.sample.sample_surface(
            mesh, n, seed=REVOLVE_SEED if seed is None else seed)
    except Exception:                                # noqa: BLE001
        return []
    pts = np.asarray(pts, float)
    centre = mesh.bounds.mean(axis=0)
    tol = REVOLVE_TOL_FRAC * float(np.linalg.norm(mesh.extents))
    u, v = [i for i in range(3) if i != axis]
    hit = np.ones(len(pts), bool)
    for deg in REVOLVE_TEST_ANGLES:
        a = np.radians(deg)
        c, s_ = np.cos(a), np.sin(a)
        q = pts - centre
        moved = q.copy()
        moved[:, u] = c * q[:, u] - s_ * q[:, v]
        moved[:, v] = s_ * q[:, u] + c * q[:, v]
        try:
            _, dist, _ = mesh.nearest.on_surface(moved + centre)
        except Exception:                            # noqa: BLE001
            return []
        hit &= np.asarray(dist) <= tol
    lo, hi = (float(x) for x in mesh.bounds[:, axis])
    if hi <= lo:
        return []
    z = pts[:, axis]
    slab = np.clip(((z - lo) / (hi - lo) * LOCAL_SLABS).astype(int), 0, LOCAL_SLABS - 1)
    state = []
    for k in range(LOCAL_SLABS):
        sel = slab == k
        state.append(bool(hit[sel].mean() >= ROUND_SLAB) if sel.any()
                     else (state[-1] if state else True))
    runs = []
    for k, st in enumerate(state):
        if runs and runs[-1][0] == st:
            runs[-1][2] = k
        else:
            runs.append([st, k, k])
    step = (hi - lo) / LOCAL_SLABS
    out = []
    for st, k0, k1 in runs:
        a, b = lo + k0 * step, lo + (k1 + 1) * step
        if not st:
            # The misses say where the departure really is; the grid only
            # says which slabs it touched.
            miss = z[(slab >= k0) & (slab <= k1) & ~hit]
            if len(miss):
                a = lo if k0 == 0 else float(miss.min())
                b = hi if k1 == LOCAL_SLABS - 1 else float(miss.max())
        out.append({"round": st, "from": a, "to": b})
    for i in range(1, len(out)):                     # round runs meet the misses
        if out[i]["round"]:
            out[i]["from"] = out[i - 1]["to"]
        else:
            out[i - 1]["to"] = out[i]["from"]
    for i, r in enumerate(out):
        # Scored clear of an inner edge: a face lying on it (the floor of the
        # cut that ends there) belongs to the run that is not round.
        a = r["from"] + (tol if i > 0 else 0.0)
        b = r["to"] - (tol if i < len(out) - 1 else 0.0)
        sel = (z >= a) & (z <= b)
        r["score"] = float(hit[sel].mean()) if sel.any() else 0.0
        r["frac"] = (r["to"] - r["from"]) / (hi - lo)
    return out


def classify(mesh, clusters=None):
    """Guess the construction regime from cheap global statistics."""
    clusters = clusters if clusters is not None else normal_clusters(mesh)
    flat_frac = sum(c["frac"] for c in clusters)
    axis_frac = sum(
        c["frac"] for c in clusters
        if np.max(np.abs(c["normal"])) > np.cos(np.radians(3.0))
    )

    ev = {"flat_area_frac": round(flat_frac, 3),
          "axis_aligned_flat_frac": round(axis_frac, 3),
          "n_flat_clusters": len(clusters)}

    # Rotational symmetry probe. See _revolve_probe: a radial-profile test scores
    # a hex nut and a rectangular C-channel over the "solid of revolution"
    # threshold, because it measures whether the mesh VERTICES sit at a constant
    # radius -- which every regular polygon trivially satisfies.
    best_axis, best_score = _revolve_probe(mesh)
    ev["revolve_axis"] = "XYZ"[best_axis] if best_axis is not None else None
    ev["revolve_score"] = round(best_score, 3)

    # Check helix before revolve: a coil spring scores high on the radial-profile
    # probe too (its wire really does sit at a near-constant radius), so a plain
    # revolve test claims it and produces a sketch that is nonsense. The helix
    # test is the more specific hypothesis and must get first refusal.
    helix = detect_helix(mesh, best_axis) if best_axis is not None else \
        {"is_helix": False, "reason": "no axis"}
    if not helix.get("is_helix"):
        # A helicoid ribbon with no core is round about no axis, and the
        # probe's best one is across it; the helix is about another.
        for k in (k for k in range(3) if k != best_axis):
            h = detect_helix(mesh, k)
            if h.get("is_helix") and h["r2"] > helix.get("r2", -1.0):
                helix = h
    helices = [helix] if helix.get("is_helix") else []
    # Per body as well: a helix that shares the mesh with something
    # else -- a worm beside its wheel -- has its own axis, which the probe
    # of the whole part does not pick. The six largest bodies of at least 0.5%
    # of the volume are tried about their own best axes (a wire coil can be a
    # small share of a mesh it shares with a wheel).
    try:
        bodies = mesh.split(only_watertight=False) if len(mesh.faces) < 2_000_000 else []
    except Exception:                                              # noqa: BLE001
        bodies = []
    if len(bodies) > 1:
        total = sum(abs(float(b.volume)) for b in bodies) or 1.0
        for b in sorted(bodies, key=lambda b: -abs(float(b.volume)))[:6]:
            if abs(float(b.volume)) < 0.005 * total:
                continue
            b_axis, _ = _revolve_probe(b)
            if b_axis is None:
                continue
            h = detect_helix(b, b_axis)
            if h.get("is_helix") and not any(
                    g["axis"] == h["axis"] and np.allclose(g["centre"], h["centre"], atol=1e-3 * float(
                        np.max(mesh.extents))) for g in helices):
                helices.append(h)
    if not helices:
        # A thread on a part that is not itself a coil (a hex nozzle, a
        # stud), on the stretch where it runs.
        order = [best_axis] if best_axis is not None else []
        for k in order + [k for k in range(3) if k not in order]:
            for h in detect_threads(mesh, k):
                if not any(g["axis"] == h["axis"] and abs(g["extent"][0] - h["extent"][0])
                           < abs(h["pitch"]) for g in helices):
                    helices.append(h)
        helices.sort(key=lambda h: -h["turns"])
    ev["helix"] = helices[0] if helices else helix
    ev["helices"] = helices
    if helix.get("is_helix"):
        return Regime("helical", min(helix["r2"], 0.95), ev)

    # No flat-area guard: the probe fails polygonal prisms outright, and a
    # guard on large flat faces would be wrong the other way -- a bucket or a
    # dome cap scores 1.000 and would be labelled "patterned" because it has a
    # large flat face.
    if best_score > 0.93:
        return Regime("revolve", min(best_score, 0.95), ev)
    if axis_frac > 0.55:
        return Regime("prismatic", min(axis_frac, 0.95), ev)
    if flat_frac > 0.4:
        return Regime("patterned", 0.5, ev)
    return Regime("freeform", 0.4, ev)


def _ingest_cache_path(path):
    """Where Stage 0 of this exact file, by this exact code, is kept, or None.

    The key is the mesh file's bytes and this module's source: a changed mesh or
    a changed Stage 0 can never be served an old answer. $M2C_INGEST_CACHE=0
    turns the cache off.
    """
    if os.environ.get("M2C_INGEST_CACHE", "1") == "0":
        return None
    try:
        p = Path(path).resolve()
        h = hashlib.sha1(Path(__file__).read_bytes())
        h.update(p.read_bytes())
        # A residual export's record changes how it is ingested.
        rec = p.with_name(p.name + ".source.json")
        if rec.is_file():
            h.update(rec.read_bytes())
        return p.parent / ".m2c" / "cache" / f"ingest_{h.hexdigest()[:20]}.pkl"
    except OSError:
        return None


def ingest(path):
    """Full Stage 0. Returns (mesh, info dict).

    Cached on disk next to the mesh. Every command ingests, and on a 670k-face
    worm-gear pair Stage 0 alone was 80-110 s of each call -- most of it the
    revolve probe's closest-point queries -- recomputing, bit for bit, what the
    previous call had already found. The cached mesh is the aligned vertices
    and faces as Stage 0 left them; nothing is approximated. A cache that
    cannot be read or written is skipped, never trusted and never fatal.
    """
    cache = _ingest_cache_path(path)
    if cache is not None:
        try:
            with open(cache, "rb") as fh:
                verts, faces, info = pickle.load(fh)
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            _mark_zipped(mesh, info)
            if info.get("repair"):
                _say_open(info["repair"], ("repair", len(faces)))
            if info.get("open"):
                _say_open(info["open"], len(faces))
            return mesh, info
        except Exception:                                          # noqa: BLE001
            pass
    mesh, info = _ingest(path)
    if cache is not None:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(f".{os.getpid()}.tmp")
            with open(tmp, "wb") as fh:
                pickle.dump((np.asarray(mesh.vertices), np.asarray(mesh.faces), info), fh,
                            protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(cache)
        except Exception:                                          # noqa: BLE001
            pass
    return mesh, info


def _mark_zipped(mesh, info):
    """Put back on a cached mesh what `zip_cracks` marked on it (see there)."""
    if info.get("zipped_faces") is not None:
        mesh.metadata["zipped_faces"] = np.asarray(info["zipped_faces"], int)
        mesh.metadata["zip_move"] = float(info.get("zip_move") or 0.0)


def _say_open(note, key):
    """The open-target note, once per mesh per process, as `_open_note` does."""
    if ("cached", key) not in _OPEN_NOTED:
        _OPEN_NOTED.add(("cached", key))
        print(note, file=sys.stderr)


# A part whose longest side is under this is scaled up at ingest by the power of
# ten that brings it to at least this size. Every tolerance, offset and grid in
# the harness is in absolute units -- the 4-decimal emit grid, the 0.02 mm step
# into the solid, a 0.01 mm fit-tolerance entry, the 0.01 mm tessellation of
# every program -- and many parts are under 2 units across: at 0.04 across the
# volume prints 0.000, the outer ring fits as one line and the step into the
# solid is half the part. A power of ten keeps every printed digit the same
# digit; the program carries the scale in _FRAME and export_stl / compare undo
# it.
MM_SCALE_MIN = 10.0


def mm_scale(mesh):
    """The power of ten `ingest` scales this mesh by: 1 for a part at least
    MM_SCALE_MIN across, else the smallest 10**k that brings it there."""
    e = float(np.max(mesh.extents)) if len(mesh.vertices) else 0.0
    if not np.isfinite(e) or e <= 0 or e >= MM_SCALE_MIN:
        return 1.0
    return float(10.0 ** int(np.ceil(np.log10(MM_SCALE_MIN / e) - 1e-9)))


# How far apart the two sides of a crack may be and still be zipped: under
# ZIP_EDGE_FRAC of the edge they close onto, and never more than ZIP_DIAG_FRAC
# of the part's diagonal. A CAD export that meshes each face on its own leaves
# the two meshings of a shared curve apart by the chord sagitta -- a few
# thousandths of a mm typically, a few tenths at the worst -- while a real
# face is at least an edge length wide.
ZIP_EDGE_FRAC = 0.3
ZIP_DIAG_FRAC = 2e-3


def _boundary_halfedges(faces):
    """Each edge used by one triangle only, as (a, b) in that triangle's winding,
    the triangle's row, and how many edges more than two triangles share."""
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    rows = np.tile(np.arange(len(faces)), 3)
    _, inv, cnt = np.unique(np.sort(e, axis=1), axis=0, return_inverse=True,
                            return_counts=True)
    one = cnt[inv.ravel()] == 1
    return e[one], rows[one], int((cnt > 2).sum())


def zip_cracks(mesh):
    """Close the cracks between faces that were meshed apart.

    Every boundary vertex is matched to the nearest boundary edge it is not on:
    one lying along that edge splits it there (the T-junction the two meshings
    left), one at its end is merged with that end. When both sides of a
    shared curve have done that they carry the same vertices, and the curve is
    two triangles' edges again.

    The two sides are a crack's width apart, so one of them bends by that
    much, and it must be the side with the smaller triangles: a triangle can
    run the whole length of a face, and fanning a bore's long strips out to
    vertices a crack's width off would leave every section across the bore a
    zigzag the arc fit breaks into pieces. So a vertex whose own edges are
    shorter than the split triangle is tall is moved onto the edge; otherwise it
    stays and the split triangle reaches out to it. The note says how far the
    farthest vertex went. Nothing is filled: a hole that
    is really missing a face stays open.

    Returns (mesh, report): the zipped mesh, or the mesh as given when zipping
    left it no closer to closed, and what was done, for the note.
    """
    from scipy.spatial import cKDTree

    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces, int)
    E, rows, shared0 = _boundary_halfedges(F)
    report = {"boundary_before": len(E), "shared_before": shared0}
    if not len(E):
        return mesh, None
    cap = ZIP_DIAG_FRAC * float(np.linalg.norm(mesh.extents))
    A, B = V[E[:, 0]], V[E[:, 1]]
    L = np.linalg.norm(B - A, axis=1)
    near = cKDTree((A + B) / 2)
    # How far each vertex's own triangles reach, and how tall each open edge's
    # triangle stands off it: which side of a crack bends.
    ue = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    reach = np.zeros(len(V))
    np.maximum.at(reach, ue.ravel(), np.repeat(np.linalg.norm(V[ue[:, 0]] - V[ue[:, 1]],
                                                              axis=1), 2))
    third = F[rows].sum(axis=1) - E.sum(axis=1)
    tall = np.linalg.norm(np.cross(B - A, V[third] - A), axis=1) / np.maximum(L, 1e-30)
    splits, merges = {}, []
    moved = V.copy()
    for v in np.unique(E):
        best = None
        for j in near.query_ball_point(V[v], r=L.max() / 2 + cap):
            if v in F[rows[j]]:
                continue
            ab = B[j] - A[j]
            t = float(np.dot(V[v] - A[j], ab) / np.dot(ab, ab))
            d = float(np.linalg.norm(A[j] + min(max(t, 0.0), 1.0) * ab - V[v]))
            if best is None or d < best[0]:
                best = (d, j, t)
        if best is None or best[0] >= min(ZIP_EDGE_FRAC * L[best[1]], cap):
            continue
        d, j, t = best
        if 1e-6 < t < 1 - 1e-6:
            splits.setdefault(j, []).append((t, v))
            if reach[v] < tall[j]:
                moved[v] = A[j] + t * (B[j] - A[j])
        else:
            merges.append((v, E[j, 0] if t <= 0.5 else E[j, 1], d))
    keep = np.ones(len(F), bool)
    fans = []
    # Which triangles the zip makes or bends, carried with the mesh: they sit a
    # crack's width off their neighbours, and segment.derive_tolerance must not
    # read that as the tessellation's own noise.
    touched = np.zeros(len(V), bool)
    for j, on in splits.items():
        f = rows[j]
        if not keep[f]:
            continue            # a triangle with two open edges: split one
        a, b = E[j]
        c = next(x for x in F[f] if x not in (a, b))
        chain = [a] + [v for _, v in sorted(on)] + [b]
        keep[f] = False
        fans += [[chain[i], chain[i + 1], c] for i in range(len(chain) - 1)]
    F = np.vstack([F[keep], np.asarray(fans, int).reshape(-1, 3)])
    zipped = np.concatenate([np.zeros(int(keep.sum()), bool), np.ones(len(fans), bool)])
    for a, b, _ in merges:
        touched[a] = touched[b] = True
    if merges:
        root = np.arange(len(V))

        def find(x):
            while root[x] != x:
                root[x] = root[root[x]]
                x = root[x]
            return x
        for a, b, _ in merges:
            ra, rb = find(a), find(b)
            if ra != rb:
                root[max(ra, rb)] = min(ra, rb)
        F = np.vectorize(find)(F)
        ok = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
        F, zipped = F[ok], zipped[ok]
        _, first = np.unique(np.sort(F, axis=1), axis=0, return_index=True)
        first = np.sort(first)
        F, zipped = F[first], zipped[first]
        touched |= root != np.arange(len(V))
    E2, _, shared = _boundary_halfedges(F)
    report.update(boundary_after=len(E2), shared_after=shared,
                  splits=sum(len(on) for on in splits.values()), merges=len(merges),
                  move_max=max(float(np.max(np.linalg.norm(moved - V, axis=1))),
                               max((d for *_, d in merges), default=0.0)))
    if len(E2) >= len(E) or shared > shared0:
        report["kept"] = False
        return mesh, report
    touched |= np.any(moved != V, axis=1)
    zipped |= touched[F].any(axis=1)
    out = trimesh.Trimesh(vertices=moved, faces=F, process=False)
    out.remove_unreferenced_vertices()
    out.fix_normals()
    out.metadata["zipped_faces"] = np.flatnonzero(zipped)
    out.metadata["zip_move"] = float(report["move_max"])
    report["kept"] = True
    report["watertight"] = bool(out.is_watertight)
    return out, report


def _repair_note(rep):
    """What `zip_cracks` did to the target, said on every command."""
    if not rep:
        return ""
    was = f"{rep['boundary_before']} boundary edges"
    if not rep.get("kept"):
        return (f"note: ingest tried to zip the target's {was} and left it as it was "
                f"(zipping would leave {rep['boundary_after']} open and "
                f"{rep['shared_after']} shared by more than two triangles)")
    now = ("it is now watertight" if rep.get("watertight") else
           f"{rep['boundary_after']} boundary edges are left")
    return (f"note: the target's {was} were cracks between faces meshed apart; "
            f"ingest zipped them ({rep['splits']} T-junctions split and "
            f"{rep['merges']} vertex pairs merged; no vertex moved more than "
            f"{rep['move_max']:.4f}) and {now}")


def _ingest(path):
    from mesh2cad.frame import export_source
    mesh = load(path)
    # A residual export is already in its target's frame (the export records
    # it beside itself). Aligned or scaled again, its coordinates would no
    # longer be the target's, and a cutter sketched on it would land a power
    # of ten away from the program.
    up = export_source(path)
    if up is not None:
        T, align_report = np.eye(4), {"applied": False, "export_of": up["named"]}
    else:
        mesh, T, align_report = align_to_axes(mesh)
    # After the alignment, so the transform is the one frame.py takes from the
    # raw target; zipping only adds edges between vertices already there.
    repair = None
    if not mesh.is_watertight:
        mesh, repair = zip_cracks(mesh)
    repair_note = _repair_note(repair)
    if repair_note:
        print(repair_note, file=sys.stderr)
    scale = 1.0 if up is not None else mm_scale(mesh)
    if scale != 1.0:
        S = np.diag([scale, scale, scale, 1.0])
        mesh.apply_transform(S)
        T = S @ np.asarray(T, float)
        if "zip_move" in mesh.metadata:
            mesh.metadata["zip_move"] *= scale
    clusters = normal_clusters(mesh)
    regime = classify(mesh, clusters)

    open_note = _open_note(mesh)
    rax = regime.evidence.get("revolve_axis")
    bands = revolve_bands(mesh, "XYZ".index(rax)) if rax else []
    lo, hi = mesh.bounds
    info = {
        "bbox_min": lo.tolist(),
        "bbox_max": hi.tolist(),
        "extents": (hi - lo).tolist(),
        "volume": float(mesh.volume),
        "area": float(mesh.area),
        "n_faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "open": open_note,
        "repair": repair_note,
        "zipped_faces": (mesh.metadata["zipped_faces"].tolist()
                         if "zipped_faces" in mesh.metadata else None),
        "zip_move": mesh.metadata.get("zip_move"),
        "alignment": align_report,
        "scale": scale,
        "transform": np.asarray(T, float).tolist(),
        "regime": regime.as_dict(),
        "revolve_bands": bands,
        "flat_clusters": [
            {"normal": [round(x, 4) for x in c["normal"]], "area_frac": round(c["frac"], 4)}
            for c in clusters[:8]
        ],
    }
    return mesh, info


# Paths already warned about, so a command that ingests twice says it once.
_OPEN_NOTED = set()


def _open_note(mesh):
    """Say on every command that the target is not a closed solid.

    Without it the only sign is `watertight False` in `overview`, and every
    later section, band table and body count on the broken region comes back
    looking normal. Returns the note, or "" for a closed target.
    """
    if mesh.is_watertight:
        return ""
    _, uses = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    boundary, shared = int((uses == 1).sum()), int((uses > 2).sum())
    parts = []
    if boundary:
        parts.append(f"{boundary} boundary edges")
    if shared:
        parts.append(f"{shared} edges shared by more than two triangles")
    if not parts:
        parts.append("inconsistent winding" if not mesh.is_winding_consistent
                     else "no enclosed volume")
    note = (f"note: the target is not watertight ({', '.join(parts)}); "
            f"section areas, band tables and body counts near those edges can "
            f"be wrong")
    key = (len(mesh.faces), boundary, shared)
    if key not in _OPEN_NOTED:
        _OPEN_NOTED.add(key)
        print(note, file=sys.stderr)
    return note


# Angles chosen because they divide nothing: not 360/n for any small n, so a
# hexagon, a square or a five-spoke wheel all fail them while a body of
# revolution survives any angle at all.
REVOLVE_TEST_ANGLES = (17.0, 43.0)
REVOLVE_SAMPLES = 600
REVOLVE_SEED = 20260812
REVOLVE_TOL_FRAC = 0.002


def _revolve_probe(mesh, n=REVOLVE_SAMPLES, seed=REVOLVE_SEED):
    """Which axis, if any, the part is a solid of revolution about.

    A body of revolution is invariant under EVERY rotation about its axis, so
    rotate the sampled SURFACE by angles that are not a symmetry of anything
    (17 and 43 degrees) and count how much lands back on the surface. The score
    is the worst of those angles, so a part that merely has n-fold symmetry --
    a hex nut, a five-spoke wheel -- cannot pass.

    Surface samples rather than vertices, because vertex density follows the
    tessellation rather than the shape.
    """
    import trimesh

    try:
        pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    except TypeError:                                # older trimesh
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            pts, _ = trimesh.sample.sample_surface(mesh, n)
        finally:
            np.random.set_state(state)
    except Exception:                                # noqa: BLE001
        return None, 0.0

    pts = np.asarray(pts, float)
    centre = mesh.bounds.mean(axis=0)
    tol = REVOLVE_TOL_FRAC * float(np.linalg.norm(mesh.extents))

    best_axis, best_score = None, 0.0
    for ax in range(3):
        u, v = [i for i in range(3) if i != ax]
        worst = 1.0
        for deg in REVOLVE_TEST_ANGLES:
            a = np.radians(deg)
            c, s_ = np.cos(a), np.sin(a)
            q = pts - centre
            moved = q.copy()
            moved[:, u] = c * q[:, u] - s_ * q[:, v]
            moved[:, v] = s_ * q[:, u] + c * q[:, v]
            try:
                _, dist, _ = mesh.nearest.on_surface(moved + centre)
            except Exception:                        # noqa: BLE001
                return None, 0.0
            worst = min(worst, float(np.mean(np.asarray(dist) <= tol)))
        if worst > best_score:
            best_axis, best_score = ax, worst
    return best_axis, best_score
