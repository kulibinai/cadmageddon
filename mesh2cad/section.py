"""Stages 3-4: where to cut, and what the cut looks like.

The expensive part of a human-in-the-loop section workflow is the search: pick an
offset, look, it is 2mm into a fillet, pick again. Each iteration is a render and
a round trip. But the search is cheap to do exhaustively in advance -- one sweep
of ~200 slices costs a couple of seconds -- and the result is far more
informative than any single slice.

Sweeping gives us, for free, the thing the model actually needs next: the
*plateaus*. A run of offsets over which the cross-section does not change is a
prismatic region, its midpoint is the canonical section to sketch, and its width
is the extrusion length. A transition is a feature boundary. So instead of the
model hunting for a good offset, we hand it a short table of candidate offsets
each already annotated with the extrusion depth it implies.
"""

import os
import sys
from dataclasses import dataclass

import numpy as np

# Which world axes map to the 2D sketch axes, per plane. trimesh builds its own
# planar frame; we re-express results in these conventions so that a coordinate
# the model reads off a section means the same thing as a coordinate in the
# overview render.
# Normals and axis pairs match CadQuery's own named workplanes exactly -- note
# that CadQuery's "XZ" plane has normal -Y, not +Y, because (X, Z, -Y) is the
# right-handed triple. Getting this wrong mirrors every XZ sketch we emit.
PLANES = {
    "XY": {"normal": np.array([0.0, 0.0, 1.0]), "axis": 2, "u": 0, "v": 1, "u_name": "X", "v_name": "Y"},
    "YZ": {"normal": np.array([1.0, 0.0, 0.0]), "axis": 0, "u": 1, "v": 2, "u_name": "Y", "v_name": "Z"},
    "XZ": {"normal": np.array([0.0, -1.0, 0.0]), "axis": 1, "u": 0, "v": 2, "u_name": "X", "v_name": "Z"},
}


@dataclass
class Slice:
    offset: float
    n_loops: int
    area: float
    perimeter: float
    polygons: list = None   # shapely Polygons in (u, v) coordinates
    # Counted separately from n_loops because n_loops counts connected
    # polygons, so a disc with nine holes through it is still "1 loop" -- a
    # summary that reads as "no holes here" and nearly caused a reconstruction
    # to omit every hole in the part.
    n_holes: int = 0
    # How far the section reaches along each sketch axis. Free to compute --
    # the polygons are already in hand here -- and it is the signal a torus's
    # equator actually shows up in: its area barely moves there while its
    # width is at a clean maximum.
    extent_u: float = 0.0
    extent_v: float = 0.0

    def signature(self):
        return (self.n_loops, self.n_holes, round(self.area, 4), round(self.perimeter, 4))


@dataclass
class Plateau:
    """A run of offsets whose cross-section is effectively constant."""
    start: float
    end: float
    rep_offset: float       # canonical offset to sketch at
    n_loops: int
    area: float
    n_holes: int = 0

    @property
    def depth(self):
        """Candidate extrusion length for the sketch taken at rep_offset."""
        return self.end - self.start


def _plane_transform(path2d):
    """The 2D -> 3D transform for a trimesh Path2D.

    trimesh stores `to_3D` (planar -> world). Older versions stored the inverse
    under `to_2D`, so accept either and always return the planar -> world form.
    """
    md = path2d.metadata
    if "to_3D" in md:
        return np.asarray(md["to_3D"])
    if "to_2D" in md:
        return np.linalg.inv(np.asarray(md["to_2D"]))
    raise KeyError("Path2D carries neither to_3D nor to_2D")


def _to_uv(polygon, spec, to_3d):
    """Re-express a trimesh planar polygon in our (u, v) world-axis convention.

    Round-tripping through world coordinates rather than trusting trimesh's own
    planar frame is what makes a coordinate read off a section mean the same
    thing as a coordinate in the overview render.
    """
    from shapely.geometry import Polygon

    def conv(coords):
        pts = np.asarray(coords)
        xyz = np.column_stack([pts[:, 0], pts[:, 1], np.zeros(len(pts)), np.ones(len(pts))])
        world = (to_3d @ xyz.T).T[:, :3]
        return np.column_stack([world[:, spec["u"]], world[:, spec["v"]]])

    return Polygon(conv(polygon.exterior.coords),
                   [conv(r.coords) for r in polygon.interiors])


# (mesh, plane) pairs on which trimesh has failed to close a cut once in this
# process. trimesh spends its buffering repairs before it gives up -- most of a
# long sweep on a mesh with touching bodies -- so once a plane is known broken
# a cut with an invalid loop is given up on without that attempt. A plane that
# never fails is untouched.
_BROKEN = set()
_NOTED = set()


class CutError(ValueError):
    """A cut of the target that doesn't close into polygons."""


def _note_once(key, text):
    if key not in _NOTED:
        _NOTED.add(key)
        print(text, file=sys.stderr)


def _branches(sec):
    """True when more than two segments of the cut meet at one point.

    A closed solid cuts into loops, where every point joins exactly two
    segments. More than two means two surfaces touch there -- a bore wall and
    the bush pressed into it, pins in their holes -- and trimesh then either
    can't close the loops or joins them in the wrong order (a fan across the
    bore). The test costs a few ms against ~70 ms for building the loops. A cut
    lying in a flat face has such points too, so it is only asked on a plane
    already seen to fail.
    """
    ends = [e.end_points for e in sec.entities if len(e.end_points)]
    if not ends:
        return False
    return bool((np.bincount(np.concatenate(ends)) > 2).any())


def _cut_polygons(sec, plane, key, on_fail="refuse"):
    """The cut's filled polygons, as trimesh builds them.

    trimesh raises "unable to recover polygon!" when a loop it found cannot be
    made valid -- doubled walls, edges shared by 3-4 triangles, touching bodies
    -- and uncaught that would end the command with a traceback on every call
    that touched the plane, before it printed anything.

    on_fail says what happens instead:
      raise  -- trimesh's ValueError, for callers that already handle it;
      skip   -- None, for the level sweep: one cut fewer out of hundreds, and
                the flat-face levels don't depend on it;
      refuse -- CutError with a message the command prints, for a cut a
                profile would be read from. Rebuilding the loop instead
                (shapely make_valid) is not done: slow and approximate.
    """
    if key in _BROKEN and on_fail != "raise" and _branches(sec):
        # Checked before trimesh builds any loop: that build is where a broken
        # cut spends its time.
        # Only once the plane has failed: a cut lying IN a flat face also has
        # such points and closes fine, and refinement probes cut exactly there.
        if on_fail == "skip":
            return None
        raise CutError(plane)
    if key not in _BROKEN or on_fail == "raise":
        try:
            return list(sec.polygons_full)
        except ValueError:
            if on_fail == "raise":
                raise
            _BROKEN.add(key)
    from shapely.geometry import Polygon

    closed, enclosure = sec.polygons_closed, sec.enclosure_directed
    out = []
    for root in sec.root:
        if closed[root] is None:
            continue
        holes = [np.array(closed[c].exterior.coords)[::-1]
                 for c in enclosure[root].keys() if closed[c] is not None]
        poly = Polygon(shell=closed[root].exterior, holes=holes)
        if not poly.is_valid:
            if on_fail == "skip":
                return None
            raise CutError(plane)
        out.append(poly)
    return out


def sweep(mesh, plane, n=200, keep_polygons=False, on_fail="skip"):
    """Slice the mesh at n offsets across its extent along the plane normal.

    A cut trimesh can't close is left out and counted (see _cut_polygons)."""
    spec = PLANES[plane]
    lo, hi = mesh.bounds[:, spec["axis"]]
    span = hi - lo
    # Inset so the first and last slice are not degenerate coincident-face cuts.
    eps = span * 1e-3
    offsets = np.linspace(lo + eps, hi - eps, n)

    origin = mesh.bounds.mean(axis=0).copy()
    origin[spec["axis"]] = 0.0

    # section_multiplane measures `heights` along the plane normal, not along the
    # world axis. XZ's normal is -Y, so a world offset of -183 would be placed at
    # y = +183 -- outside the mesh, and every slice comes back empty. Convert
    # world offset to signed height along the normal.
    sense = float(spec["normal"][spec["axis"]])
    sections = mesh.section_multiplane(
        plane_origin=origin, plane_normal=spec["normal"], heights=offsets * sense
    )

    out, skipped = [], 0
    for off, sec in zip(offsets, sections):
        if sec is None:
            out.append(Slice(float(off), 0, 0.0, 0.0, [] if keep_polygons else None))
            continue
        polys = _cut_polygons(sec, plane, (id(mesh), plane), on_fail)
        if polys is None:
            skipped += 1
            continue
        area = float(sum(p.area for p in polys))
        perim = float(sum(p.length for p in polys))
        holes = int(sum(len(p.interiors) for p in polys))
        kept = None
        if keep_polygons:
            kept = [_to_uv(p, spec, _plane_transform(sec)) for p in polys]
        # trimesh's planar frame is not our (u, v), but an EXTENT is invariant
        # to which of the two in-plane axes is which up to a swap, and both are
        # kept, so nothing here depends on getting that mapping right.
        bb = np.array([p.bounds for p in polys], float) if polys else None
        eu = float(bb[:, 2].max() - bb[:, 0].min()) if bb is not None else 0.0
        ev = float(bb[:, 3].max() - bb[:, 1].min()) if bb is not None else 0.0
        out.append(Slice(float(off), len(polys), area, perim, kept, n_holes=holes,
                         extent_u=eu, extent_v=ev))
    if skipped:
        axis = "XYZ"[spec["axis"]]
        _note_once(("skipped", id(mesh), plane, n),
                   f"note: {skipped} of {n} {plane} cuts along {axis} were left "
                   f"out of the level sweep: the target's mesh crosses itself or "
                   f"has doubled walls there, so they don't close into polygons. "
                   f"Sweep levels and band areas on {plane} come from the other "
                   f"{n - skipped}; levels from flat faces are unaffected.")
    return out


def find_plateaus(slices, area_tol=0.02, min_width_frac=0.05):
    """Group the sweep into runs of effectively-constant cross-section.

    area_tol is relative: a run survives while every slice's area stays within
    area_tol of the run's median. Relative rather than absolute because the same
    harness has to handle a 3mm grub screw and a 300mm flange.
    """
    if not slices:
        return []
    span = slices[-1].offset - slices[0].offset
    min_width = span * min_width_frac

    plateaus, run = [], []

    def flush():
        if len(run) < 2:
            return
        width = run[-1].offset - run[0].offset
        if width < min_width:
            return
        areas = [s.area for s in run]
        mid = run[len(run) // 2]
        plateaus.append(Plateau(
            start=run[0].offset, end=run[-1].offset, rep_offset=mid.offset,
            n_loops=mid.n_loops, area=float(np.median(areas)),
            n_holes=mid.n_holes,
        ))

    for s in slices:
        if s.n_loops == 0:
            flush()
            run = []
            continue
        if not run:
            run = [s]
            continue
        ref = float(np.median([r.area for r in run]))
        same_topology = (s.n_loops, s.n_holes) == (run[-1].n_loops, run[-1].n_holes)
        same_area = abs(s.area - ref) <= area_tol * max(ref, 1e-9)
        if same_topology and same_area:
            run.append(s)
        else:
            flush()
            run = [s]
    flush()
    return plateaus


# How precisely a bracketed transition is pinned down, in mm. A sweep locates a
# boundary to its own step and no better; bisecting the bracket costs about
# eight extra slices and locates it to here regardless of how coarse the sweep
# was. That trade is the whole point: sampling finely enough to get 0.001 mm out
# of the sweep alone would cost thousands of slices.
TRANSITION_PRECISION = 1e-3
TRANSITION_MAX_ITERS = 24


def _signature(mesh, plane, offset):
    """(n_loops, n_holes, area) of the section at one offset."""
    polys = section_at(mesh, plane, float(offset))
    return (len(polys),
            int(sum(len(p.interiors) for p in polys)),
            float(sum(p.area for p in polys)))


def _same_side(sig, ref, rel_jump):
    if (sig[0], sig[1]) != (ref[0], ref[1]):
        return False
    return abs(sig[2] - ref[2]) / max(ref[2], 1e-9) <= rel_jump


def refine_transition(mesh, plane, a, b, rel_jump=0.05,
                      precision=TRANSITION_PRECISION,
                      max_iters=TRANSITION_MAX_ITERS):
    """Bisect a bracketed transition down to `precision`.

    Returns (offset, half_width, iterations, still_a_step). `still_a_step` is
    the honest part: a real step keeps its full jump however narrow the bracket
    gets, while a cone's area ramps, and its "transition" is an artefact of
    where the samples happened to fall. Narrowing tells the two apart, and the
    caller is told which it got rather than being handed the same number twice.
    """
    sa, sb = _signature(mesh, plane, a), _signature(mesh, plane, b)
    if _same_side(sb, sa, rel_jump):
        return None
    it = 0
    while (b - a) > precision and it < max_iters:
        m = 0.5 * (a + b)
        if _same_side(_signature(mesh, plane, m), sa, rel_jump):
            a, sa = m, _signature(mesh, plane, m)
        else:
            b, sb = m, _signature(mesh, plane, m)
        it += 1
    ref = max(sa[2], 1e-9)
    still_a_step = ((sa[0], sa[1]) != (sb[0], sb[1])
                    or abs(sb[2] - sa[2]) / ref > rel_jump)
    return 0.5 * (a + b), 0.5 * (b - a), it, still_a_step


def transitions(slices, rel_jump=0.05, mesh=None, plane=None,
                precision=TRANSITION_PRECISION):
    """Offsets where the cross-section changes -- i.e. feature boundaries.

    Pass `mesh` and `plane` to have each bracket bisected onto the actual
    transition. Without them the offset is the midpoint of two samples and is
    only as good as the sweep, and the reconstruction inherits every bit of the
    error (a visibly thicker end cap, say).

    Every entry says which it is. `located` is "measured" or "sampled", and
    `uncertainty` is the half-width the offset is known to -- so a level that
    was pinned down and a level that was guessed at can be told apart by
    something other than tone of voice.
    """
    out = []
    for a, b in zip(slices, slices[1:]):
        if (a.n_loops, a.n_holes) != (b.n_loops, b.n_holes):
            t = {"offset": (a.offset + b.offset) / 2, "kind": "topology",
                 "detail": f"{a.n_loops} -> {b.n_loops} loops, "
                           f"{a.n_holes} -> {b.n_holes} holes"}
        else:
            ref = max(a.area, 1e-9)
            if abs(b.area - a.area) / ref <= rel_jump:
                continue
            t = {"offset": (a.offset + b.offset) / 2, "kind": "area",
                 "detail": f"{a.area:.2f} -> {b.area:.2f} mm^2"}
        t["bracket"] = [float(a.offset), float(b.offset)]
        t["located"] = "sampled"
        t["uncertainty"] = float(b.offset - a.offset) / 2.0

        if mesh is not None and plane is not None:
            try:
                got = refine_transition(mesh, plane, float(a.offset),
                                        float(b.offset), rel_jump=rel_jump,
                                        precision=precision)
            except CutError:
                got = None      # a probe cut didn't close: keep the sample
            if got is not None:
                off, half, _, still_a_step = got
                if still_a_step:
                    t["offset"], t["uncertainty"] = off, half
                    t["located"] = "measured"
                else:
                    # The jump decayed as the bracket narrowed: the section is
                    # ramping, not stepping. Keep the sample midpoint and its
                    # honest half-step uncertainty, and say why.
                    t["located"] = "sampled (section ramps, no step to refine)"
        out.append(t)
    return out


def spectrum(mesh, plane, n=200, slices=None):
    """Stage 3. Everything the model needs to choose an offset in one call.

    `slices` lets a caller that has already swept hand the result in. The sweep
    is the expensive part of this and feature_levels needs it too; sweeping
    twice per call was pure waste.
    """
    if slices is None:
        slices = sweep(mesh, plane, n=n)
    plats = find_plateaus(slices)
    spec = PLANES[plane]
    lo, hi = mesh.bounds[:, spec["axis"]]

    # Rank candidates by how much of the part each plateau accounts for: a wide
    # plateau with a large section is the part's primary prismatic body, and is
    # almost always the right place to start the reconstruction.
    cands = sorted(plats, key=lambda p: -(p.depth * max(p.area, 0.0)))
    return {
        "plane": plane,
        "normal_axis": "XYZ"[spec["axis"]],
        "sketch_axes": (spec["u_name"], spec["v_name"]),
        "range": [float(lo), float(hi)],
        "n_samples": n,
        # Disclosed because it is otherwise indistinguishable from a real
        # feature: the sweep insets its first and last sample so they are not
        # degenerate coincident-face cuts, which makes every plateau read a
        # touch shorter than the part's true extent at both ends.
        "end_inset": round(float((hi - lo) * 1e-3), 6),
        "candidates": [
            {
                "id": f"c{i}",
                "offset": round(p.rep_offset, 4),
                "span": [round(p.start, 4), round(p.end, 4)],
                "depth": round(p.depth, 4),
                "n_loops": p.n_loops,
                "n_holes": p.n_holes,
                "area": round(p.area, 3),
            }
            for i, p in enumerate(cands[:8])
        ],
        # NOT truncated. Keeping only the first N, on a part whose area ramps
        # continuously -- a spoke is a wedge, so nearly every sample clears the 5%
        # jump test -- would silently discard everything above the Nth: half of a
        # symmetric part, inherited by `feature_levels`, with the picture showing
        # boundaries all on one side and no reason why. Cap the display, never the
        # data.
        "transitions": [
            {"offset": round(t["offset"], 4), "kind": t["kind"],
             "detail": t["detail"], "located": t["located"],
             "uncertainty": round(t["uncertainty"], 4)}
            for t in transitions(slices, mesh=mesh, plane=plane)
        ],
        "profile": [
            {"offset": round(s.offset, 3), "n": s.n_loops, "holes": s.n_holes,
             "area": round(s.area, 2)}
            for s in slices
        ],
    }


def plane_survey(mesh, n=120):
    """Stage 2. Score all three planes so the model can pick one in one shot.

    The useful signal is what fraction of the part's extent the single best
    plateau covers. A part that is a prism along Y scores ~100% on XZ and much
    less on the other two, which is precisely the "which plane should I cut on"
    question -- and it is cheap enough to just answer rather than ask.
    """
    out = []
    for plane in ("XY", "YZ", "XZ"):
        spec = PLANES[plane]
        lo, hi = mesh.bounds[:, spec["axis"]]
        span = float(hi - lo)
        try:
            slices = sweep(mesh, plane, n=n)
        except Exception as e:
            out.append({"plane": plane, "error": f"{type(e).__name__}: {e}"})
            continue
        plats = find_plateaus(slices)
        best = max(plats, key=lambda p: p.depth * max(p.area, 0.0), default=None)
        out.append({
            "plane": plane,
            "normal_axis": "XYZ"[spec["axis"]],
            "sketch_axes": (spec["u_name"], spec["v_name"]),
            "extent": round(span, 3),
            "n_plateaus": len(plats),
            "n_transitions": len(transitions(slices)),
            "best_depth": round(best.depth, 3) if best else 0.0,
            "prismaticity": round(best.depth / span, 3) if best and span > 0 else 0.0,
            "best_loops": best.n_loops if best else 0,
            "best_holes": best.n_holes if best else 0,
            "best_area": round(best.area, 2) if best else 0.0,
            "best_offset": round(best.rep_offset, 3) if best else None,
        })
    ranked = sorted([o for o in out if "error" not in o],
                    key=lambda o: -o["prismaticity"])
    return {"planes": out,
            "recommended": ranked[0]["plane"] if ranked else None,
            "reason": (f"{ranked[0]['plane']} holds a constant cross-section over "
                       f"{ranked[0]['prismaticity']:.0%} of the part's "
                       f"{ranked[0]['normal_axis']} extent") if ranked else None}


def project_slab(mesh, plane, a, b):
    """Every bit of material between levels `a` and `b` along the plane's
    normal, projected onto the plane: its outline as (u, v) polygons, holes
    where nothing lies anywhere in the slab.

    A section is the material at ONE level. A joggled or bent plate is not
    flat at any level, so a cut catches part of it and can miss a whole side
    plate, where a projection of the slab onto the plane gives the full
    outline -- tab, hole and slot. The projection is the union of the slab's
    triangles, laid flat.
    """
    import shapely
    import trimesh

    spec = PLANES[plane]
    k = spec["axis"]
    lo, hi = sorted((float(a), float(b)))
    n = np.zeros(3)
    n[k] = 1.0
    m = mesh
    for normal, at in ((n, lo), (-n, hi)):
        o = np.zeros(3)
        o[k] = at
        m = trimesh.intersections.slice_mesh_plane(m, plane_normal=normal, plane_origin=o)
        if m is None or not len(m.faces):
            return []
    tri = np.asarray(m.triangles)[:, :, [spec["u"], spec["v"]]]
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    size = float(np.max(mesh.extents))
    tri = tri[area > (1e-9 * size) ** 2]
    if not len(tri):
        return []
    region = shapely.union_all(shapely.polygons(tri))
    # Triangles that share an edge meet to the last bit or not at all; close
    # the hairline seams they leave without moving the outline.
    eps = 1e-6 * size
    region = region.buffer(eps, join_style="mitre").buffer(-eps, join_style="mitre")
    return [g for g in getattr(region, "geoms", [region])
            if g.geom_type == "Polygon" and g.area > (1e-6 * size) ** 2]


def section_at(mesh, plane, offset):
    """Extract the single cross-section at a chosen offset, in (u, v) coords."""
    spec = PLANES[plane]
    origin = mesh.bounds.mean(axis=0).copy()
    origin[spec["axis"]] = 0.0
    sense = float(spec["normal"][spec["axis"]])
    sections = mesh.section_multiplane(
        plane_origin=origin, plane_normal=spec["normal"], heights=[offset * sense]
    )
    sec = sections[0]
    if sec is None:
        return []
    try:
        polys = _cut_polygons(sec, plane, (id(mesh), plane))
    except CutError:
        axis = "XYZ"[spec["axis"]]
        raise CutError(
            f"refused: the {plane} cut at {axis} = {offset:.4f} doesn't close "
            f"into polygons -- the target's mesh crosses itself or has doubled "
            f"walls there. Cut at another level or on another plane.") from None
    return [_to_uv(p, spec, _plane_transform(sec)) for p in polys]


# Where the level table comes from. `both` is faces first with the sweep filling
# in what no face explains; `faces` and `sweep` are each alone, for comparing
# them. The environment variable sets the default, so an ablation can switch
# every call of a run without the agent having to type a flag.
LEVEL_SOURCES = ("both", "faces", "sweep")
LEVELS_ENV = "MESH2CAD_LEVELS"
# Seconds the level table may spend finding round surfaces. It is built on
# every `section`, `ortho` and `sketch`; `faces` allows itself ten times as
# long, and a segmentation is cached for whichever call may use it.
LEVELS_SEGMENT_BUDGET = 90.0


def level_source(source=None):
    got = source or os.environ.get(LEVELS_ENV) or "both"
    if got not in LEVEL_SOURCES:
        raise SystemExit(f"--levels {got!r}: use one of {', '.join(LEVEL_SOURCES)}")
    return got


def feature_levels(mesh, plane, samples, source=None, cache_dir=None):
    """The levels along the plane normal, named.

    Shared by `ortho`, `section` and `sketch` so the picture, the measurement
    and the printed table can never disagree about where a level is.

    Faces first (see mesh2cad.face_levels): a flat face perpendicular to the
    axis is a level to float precision, and a surface lying across the axis is
    not a level at all, however much it changes the area of a cut. The sweep
    still runs -- the band table is read off it, and it is an independent check
    -- and each of its levels either lands on a face level, falls where some
    face bends through the cut and is explained by it, or is kept and labelled
    as seen by the sweep alone. Nothing it finds is dropped without a count.
    """
    from mesh2cad.budget import stage
    stage(f"finding the levels across {plane} ({samples} cuts)")
    source = level_source(source)
    marks, s, tol = _sweep_marks_cached(mesh, plane, samples, cache_dir)
    if source != "sweep":
        got = _with_faces(mesh, plane, samples, marks, s, tol, source, cache_dir)
        if got is not None:
            marks = got
    _name_levels(marks)
    return marks, s


def _name_levels(marks):
    """The model never handles a raw number: a level is a thing the model
    refers to, and without a name the only way to refer to one is to copy its
    float off a plot -- and a mistyped or unconverged digit builds every feature
    on that level wrong without any warning. Ends are E, mirrors M, turning
    points T, taper and blend ends K, everything else B, each numbered from the
    low end so the ordering is readable."""
    marks.sort(key=lambda m: m["value"])
    count = {"E": 0, "M": 0, "T": 0, "K": 0, "B": 0}
    prefix = {"end": "E", "mirror": "M", "turning": "T", "taper": "K"}
    for m in marks:
        pre = prefix.get(m["kind"], "B")
        m["name"] = f"{pre}{count[pre]}"
        count[pre] += 1


_SWEEP_MEMO = {}


def _sweep_marks_cached(mesh, plane, samples, cache_dir=None):
    """`_sweep_marks`, kept between calls like the segmentation.

    Every `ortho`, `section` and `sketch` builds the level table, and the sweep
    under it -- `samples` cuts and the bisection that refines each step -- came
    out the same on every call: 90-130 s of each one on a 670k-face mesh. The
    key is the mesh's own vertices and faces, the plane, the sample count and
    the source of the code that computes it, so a different part, plane or
    version is never served an old table, and the next command -- a new
    process -- computes the same key and finds it. Each caller gets its own copy (the
    level table is annotated after this), and a cache that cannot be read or
    written is skipped, never trusted and never fatal.
    """
    import hashlib
    import pickle
    from pathlib import Path
    # Only the real sweep is kept. A replaced one (a test adding a level the
    # faces contradict) is run every time and never stored. The key used to
    # carry id(_sweep_marks) for that -- a memory address, different in every
    # process -- so each command wrote a table that no later command could find,
    # and the disk cache saved nothing between calls.
    if (_sweep_marks.__module__, _sweep_marks.__qualname__) != (__name__, "_sweep_marks"):
        return _sweep_marks(mesh, plane, samples)
    h = hashlib.sha1()
    here = Path(__file__).resolve().parent
    for src in ("section.py", "levels.py"):
        try:
            h.update((here / src).read_bytes())
        except OSError:
            pass
    h.update(f"{plane};{samples};".encode())
    h.update(np.ascontiguousarray(mesh.vertices, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(mesh.faces, dtype=np.int64).tobytes())
    key = h.hexdigest()[:20]
    blob = _SWEEP_MEMO.get(key)
    path = Path(cache_dir) / f"sweep_{key}.pkl" if cache_dir is not None else None
    if blob is None and path is not None:
        try:
            blob = path.read_bytes()
            pickle.loads(blob)
        except Exception:                                          # noqa: BLE001
            blob = None
    if blob is None:
        blob = pickle.dumps(_sweep_marks(mesh, plane, samples),
                            protocol=pickle.HIGHEST_PROTOCOL)
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(f".{os.getpid()}.tmp")
                tmp.write_bytes(blob)
                tmp.replace(path)
            except Exception:                                      # noqa: BLE001
                pass
    _SWEEP_MEMO[key] = blob
    return pickle.loads(blob)


def _sweep_marks(mesh, plane, samples):
    """The sweep's levels: transitions, plateau edges, mirrors, turning points
    and taper ends, with the part's two ends -- unnamed."""
    slices = sweep(mesh, plane, n=samples)
    s = spectrum(mesh, plane, n=samples, slices=slices)
    lo_a, hi_a = s["range"]
    span = hi_a - lo_a

    raw = [{"value": t["offset"], "kind": "boundary",
            "located": t["located"], "uncertainty": t["uncertainty"]}
           for t in s["transitions"]]
    for c in s["candidates"]:
        raw.append({"value": c["span"][0], "kind": "edge"})
        raw.append({"value": c["span"][1], "kind": "edge"})
    raw.sort(key=lambda m: m["value"])

    # One physical boundary is reported three times -- as the end of one
    # plateau, the transition itself, and the start of the next -- all within a
    # sweep step of each other. Drawn separately they overprint into a blur, so
    # collapse anything inside a couple of sweep steps into a single level.
    tol = max(span / samples * 2.5, span * 0.004)
    def _group_value(grp):
        """One measurement beats any number of guesses about the same place.

        A group is usually a measured transition flanked by the two plateau
        edges that reported the same boundary, and averaging the three drags
        the answer away from the one member that actually knows, which the
        reconstruction would wear as a thicker or thinner wall.
        """
        exact = [x for x in grp["members"] if x.get("located") == "measured"]
        if exact:
            return min(exact, key=lambda x: x.get("uncertainty", 0.0))["value"]
        return sum(x["value"] for x in grp["members"]) / len(grp["members"])

    merged = []
    for m in raw:
        if merged and abs(m["value"] - merged[-1]["value"]) <= tol:
            grp = merged[-1]
            grp["members"].append(m)
            grp["value"] = _group_value(grp)
            if m["kind"] == "boundary":
                grp["kind"] = "boundary"
        else:
            merged.append({"value": m["value"], "kind": m["kind"], "members": [m]})

    marks = []
    for g in merged:
        at_end = min(abs(g["value"] - lo_a), abs(g["value"] - hi_a)) <= tol
        # A transition is measured: the cross-section demonstrably changed
        # there. A plateau edge is inferred: find_plateaus groups greedily in
        # sweep order, so where it decides a run stops is direction-dependent
        # and does not mirror even when the part does. Labelling both "prism
        # boundary" would give the guess the same standing as the measurement.
        from_transition = any(x["kind"] == "boundary" for x in g["members"])
        measured = [x for x in g["members"] if x.get("located") == "measured"]
        if at_end:
            kind, label = "end", "part end"
        elif measured:
            kind, label = "boundary", "prism boundary"
        elif from_transition:
            kind, label = "boundary", "prism boundary (located by sampling only)"
        else:
            kind, label = "boundary", "plateau edge"
        # Four places, not three: a bisected boundary is known to 0.001 mm and
        # rounding it to 0.001 throws away the last thing bisection bought.
        marks.append({"value": round(g["value"], 4), "kind": kind, "label": label,
                      "uncertainty": (min(x["uncertainty"] for x in measured)
                                      if measured else
                                      max((x.get("uncertainty", tol)
                                           for x in g["members"]), default=tol))})

    # The two real ends of the part, always. Previously an end appeared only if
    # a plateau happened to reach it, so on a part with no plateau near an
    # extreme -- a sphere, say -- that end silently vanished from both the list
    # and the picture. The extremes come from the bbox, not the sweep, so they
    # are exact rather than inset.
    spec = PLANES[plane]
    true_lo, true_hi = (float(v) for v in mesh.bounds[:, spec["axis"]])
    for v in (true_lo, true_hi):
        near = [m for m in marks if abs(v - m["value"]) <= tol and m["kind"] == "end"]
        if near:
            # Snap to the exact value rather than keeping the sweep's. A
            # plateau edge is inset by construction, so a mark labelled
            # "part end" sitting 0.2 mm inside the real face is simply wrong --
            # and it is wrong in the one place a reader is most entitled to
            # trust the number, since the bounding box is exact.
            for m in near:
                m["value"] = round(v, 4)
        else:
            marks.append({"value": round(v, 4), "kind": "end", "label": "part end"})
    marks.sort(key=lambda m: m["value"])
    extra, s["level_notes"] = _symmetry_marks(mesh, plane, slices, marks, tol,
                                              marks_ref=marks)
    marks += extra
    marks.sort(key=lambda m: m["value"])
    return marks, s, tol


# A flat face tilted less than this off its nearest axis is read as that axis's
# face with mesh noise in it, not as a slanted one worth a frame.
SLANT_MIN_DEG = 1.0


def _slanted_planes(seg, k):
    """[(surface, degrees off its nearest axis)], largest first: the flat faces
    lying on no axis that cross axis k obliquely. One that runs ALONG k is not
    listed -- a section across k shows it as a straight edge, and an extrude
    along k builds it."""
    out = []
    for s_ in seg.surfaces:
        if s_.kind != "plane":
            continue
        n = np.asarray(s_.params["normal"], float)
        n = n / np.linalg.norm(n)
        off = float(np.degrees(np.arccos(min(1.0, np.abs(n).max()))))
        if off > SLANT_MIN_DEG and abs(n[k]) > np.sin(np.radians(SLANT_MIN_DEG)):
            out.append((s_, off))
    out.sort(key=lambda t: -t[0].area)
    return out


def _with_faces(mesh, plane, samples, sweep_marks, s, sweep_tol, source,
                cache_dir):
    """Face levels, and the sweep's reconciled with them. None means: keep the
    sweep's table, with the reason already in `s["level_notes"]`."""
    from mesh2cad.face_levels import (
        anchored_slopes,
        candidates,
        explained_by,
        merge,
        seams,
    )
    from mesh2cad.segment import segment_cached

    spec = PLANES[plane]
    k = spec["axis"]
    ax = "XYZ"[k]
    try:
        seg = segment_cached(mesh, cache_dir=cache_dir,
                             budget=LEVELS_SEGMENT_BUDGET)
    except Exception as exc:                                       # noqa: BLE001
        s["level_notes"] = [f"levels are the sweep's alone: the segmentation "
                            f"failed ({type(exc).__name__}: {exc})"] + \
            list(s.get("level_notes") or [])
        return None
    if any("does not look like an exact CAD tessellation" in n for n in seg.notes):
        s["level_notes"] = ["levels are the sweep's alone: this mesh is not an "
                            "exact tessellation, so its faces cannot be read "
                            "off it (see `faces`)"] + list(s.get("level_notes") or [])
        return None

    cands, ramps, across = candidates(seg, mesh, k)
    cands, loose = anchored_slopes(cands, seg)
    face, n_seams, seam_faces = seams(merge(cands, seg, k), seg, mesh, k)

    # The two ends of the part are the bounding box, exact, always; a face
    # level there IS the end, and says which face.
    lo, hi = (float(v) for v in mesh.bounds[:, k])
    near_tol = max(8.0 * seg.tol, 1e-9)
    ends = []
    for v in (lo, hi):
        at = [m for m in face if abs(m["value"] - v) <= near_tol + m["uncertainty"]]
        label = "part end" + (f" -- {at[0]['label']}" if at else "")
        ends.append({"value": v, "kind": "end", "label": label,
                     "uncertainty": 0.0, "faces": at[0]["faces"] if at else [],
                     "source": "faces", "refined": True})
        for m in at:
            face.remove(m)
    marks = ends + face

    notes = []
    if seg.partial:
        # The cap and what it left, not just that it hit one.
        from mesh2cad.contrib.instruments.faces import FACES_SEGMENT_BUDGET
        untried = getattr(seg, "untried", 0)
        notes.append(
            f"the search for round faces stopped at its {seg.budget:.0f} s cap with "
            f"{untried} candidate facet patch(es) untried "
            f"({getattr(seg, 'untried_area', 0.0):.1%} of the surface): ends of round "
            f"faces there may be missing from this table. Levels the sweep sees "
            f"where no face was found are kept; `faces` searches longer "
            f"(up to {FACES_SEGMENT_BUDGET:.0f} s)")
    if n_seams:
        notes.append(
            f"{n_seams} join(s) where one curved face runs smoothly into the "
            f"next ({', '.join(seam_faces[:10])}{' ...' if len(seam_faces) > 10 else ''}) "
            f"are not listed: they are seams inside one profile drawn as several "
            f"arcs, not places anything starts or stops")
    if loose:
        notes.append(
            f"{loose} sloped face(s) meet only other sloped faces along {ax} -- "
            f"the corners of a profile drawn along some other axis, like a "
            f"gear's tooth flanks -- and their ends are not listed")
    names = [seg.surfaces[i].name for i in across]
    if names:
        kinds = {}
        for i in across:
            kinds[seg.surfaces[i].kind] = kinds.get(seg.surfaces[i].kind, 0) + 1
        notes.append(
            f"{len(names)} surface(s) run across {ax} and give it no level -- "
            + ", ".join(f"{n} {kd}" for kd, n in sorted(kinds.items()))
            + ": " + ", ".join(names[:12]) + (" ..." if len(names) > 12 else ""))
    slant = _slanted_planes(seg, k)
    if slant:
        notes.append(
            f"{len(slant)} flat face(s) cross {ax} at a slant -- "
            + ", ".join(f"{s_.name} ({d:.0f} deg off its axis)" for s_, d in slant[:8])
            + (" ..." if len(slant) > 8 else "")
            + ": a plate, tab or cut set at that angle, which no level and no "
              f"{plane} section gives whole. `sketch --face F#` builds on one in "
              "its own frame (its outline, --to the parallel face); --frame F# "
              "lays it flat for ortho and section")

    if source == "faces":
        s["level_notes"] = notes
        return marks

    # The sweep's own levels, against the faces'.
    step = (hi - lo) / max(1, samples)
    hidden, why, kept, flat = 0, set(), [], 0
    for m in sweep_marks:
        if m["kind"] == "end":
            continue
        reach = max(sweep_tol, float(m.get("uncertainty") or 0.0))
        if any(abs(f["value"] - m["value"]) <= reach + f["uncertainty"]
               for f in marks):
            continue                        # the face level is the same place
        bent = explained_by(m["value"], ramps, max(2.0 * step, near_tol))
        if bent:
            hidden += 1
            why.update(seg.surfaces[i].name for i in bent)
            continue
        if m["kind"] != "mirror" and not seg.partial:
            # Nothing bends here and nothing ends here: every face at this
            # height is a wall along the axis. The cut cannot change, so what
            # the sweep saw is noise in its sampling -- or a broken cut, which
            # is how an open mesh once got a level called B0 at 56% of its
            # height and was built to it. A mirror plane is a different claim,
            # about symmetry, and stays.
            flat += 1
            continue
        m = dict(m)
        m["label"] = m["label"] + " -- sweep only: no face ends here"
        m["source"] = "sweep"
        kept.append(m)
    if hidden:
        w = sorted(why, key=lambda n: int(n[1:]))
        notes.append(
            f"{hidden} level(s) the sweep saw lie where a face bends through the "
            f"cut ({', '.join(w[:10])}{' ...' if len(w) > 10 else ''}): the "
            f"section changes there because the face curves, not because "
            f"anything stops. Not listed; `--levels sweep` lists them")
    if flat:
        notes.append(
            f"{flat} level(s) the sweep saw lie where every face runs straight "
            f"along {ax} -- the cut cannot change there, so what it saw is "
            f"sampling noise or a broken cut. Not listed; `--levels sweep` "
            f"lists them")
    if kept:
        notes.append(f"{len(kept)} level(s) come from the sweep alone -- no face "
                     f"ends there; treat them as the sweep always was: a change "
                     f"in the cut, located by sampling")
        # The sweep's notes are about what IT could not see; they still hold
        # for the levels it alone contributes.
        notes += list(s.get("level_notes") or [])
    s["level_notes"] = notes
    return marks + kept


# How many of each to keep. A bellows genuinely has a mirror plane per
# corrugation, and printing twelve of them buries the level table it is meant
# to help (R4). The count dropped is always reported (R2).
MAX_MIRRORS = 3
MAX_TURNING = 3
MAX_TAPERS = 4
# Refinement costs sections, so only the strongest few earn it. The rest keep
# their sampled value and say so.
MAX_REFINE = 2


def _symmetry_marks(mesh, plane, slices, existing, tol, marks_ref=None):
    """Mirror planes and turning points, as levels the model can name.

    These are the levels the sweep could never produce. It looks for value
    jumps; a torus's equator is not a jump in anything, so without these it
    would have to be typed in as a raw number.

    Anything landing on a level that already has a name is dropped: the model
    can say `B7`, and a second name for the same plane is cost without
    information.
    """
    from mesh2cad import levels as lv

    found, notes = [], []
    # Checked against everything accepted SO FAR, not just against the levels
    # that existed before. A torus's equator is both a mirror plane and the
    # turning point of its width -- the two detectors agree to four decimals,
    # which is reassuring and is still only one place. Mirrors are considered
    # first because they are the better-conditioned measurement.
    marks_ref = existing if marks_ref is None else marks_ref
    seen = list(existing)

    mirrors = sorted(lv.mirror_levels(slices), key=lambda m: -m["half_width"])
    turning = sorted(lv.extremum_levels(slices), key=lambda t: -t["prominence"])
    tapers = sorted(lv.taper_levels(slices), key=lambda t: -t["strength"])
    for kind, items, cap in (("mirror", mirrors, MAX_MIRRORS),
                             ("taper", tapers, MAX_TAPERS),
                             ("turning", turning, MAX_TURNING)):
        kept, dropped = items[:cap], max(0, len(items) - cap)
        for rank, item in enumerate(kept):
            # Listed at sweep resolution and refined only when the model names
            # one. Refining every candidate here cost about a second per level
            # on every call that builds a level table -- `section`, `ortho` and
            # every `sketch` -- whether or not these levels were ever used,
            # which nearly doubled the cost of the table. Detection is nearly
            # free; the measurement happens at the point of use. See
            # refine_named_level in the CLI.
            del rank
            value, unc = item["offset"], item["uncertainty"]
            # One measurement beats any number of guesses about the same place
            # -- the same rule that stops a bisected transition being averaged
            # with the plateau edges around it. A plateau edge IS a guess: a
            # greedy scan's opinion about where a run stopped drifting. When a
            # measured level lands on one, the guess is replaced rather than the
            # measurement dropped.
            clash = [m for m in seen if abs(value - m["value"]) <= tol]
            guesses = [m for m in clash if "plateau edge" in m.get("label", "")]
            if clash and len(guesses) != len(clash):
                continue          # coincides with something already measured
            for g in guesses:
                if g in marks_ref:
                    marks_ref.remove(g)
                if g in seen:
                    seen.remove(g)
            label = ("mirror plane" if kind == "mirror" else
                     item["kind"] if kind == "taper" else
                     f"turning point, {item['kind']}")
            label += " (sweep resolution; refined when you name it)"
            entry = {"value": round(float(value), 4), "kind": kind,
                     "label": label, "uncertainty": float(unc)}
            found.append(entry)
            seen.append(entry)
        if dropped:
            notes.append(f"{dropped} further {kind} level(s) were found and are "
                         f"NOT listed; they rank below the ones shown. A "
                         f"corrugated part has one per corrugation.")
    # A silence has to mean "nothing there", not "could not look".
    blind = lv.taper_blind_radius(slices)
    if blind is not None:
        # Concrete, because "raise --samples" is not a command. This note fired
        # three times in one run at 0.877 mm while the taper it was hiding sat
        # 0.34 mm from a jump, and was ignored every time -- the reader had no
        # way to know WHICH sampling would help, and guessing costs a call.
        n_now = len(slices)
        n_more = n_now * 4
        notes.append(f"a taper end within {blind:.3f} mm of a value jump or of "
                     f"the part's end cannot be seen at this sampling. If this "
                     f"part has a chamfer or a draft near a step, re-run with "
                     f"`--samples {n_more}`, which would shrink that radius to "
                     f"{blind / 4:.3f} mm.")
    return found, notes
