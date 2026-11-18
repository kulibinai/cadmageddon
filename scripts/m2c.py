"""m2c -- the agent-facing CLI for the mesh2cad stages.

This is a stand-in for the eventual MCP server: same stage boundaries, same
outputs, but driven from a shell so it can be exercised by an agent today. Every
command prints a compact text block (the part the model reasons over) and, where
useful, writes a PNG (the part it looks at).

Stage 4 -- decomposing a section profile into named lines and arcs -- does not
exist yet, so `section` emits the raw polyline plus a crude corner list and says
so. That gap is deliberate here: it is the thing we are trying to find out about.
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mesh2cad import budget as _tb
from mesh2cad import contract, journal, readback, registry
from mesh2cad.emit import (
    EMIT_DECIMALS,
    EmitError,
    build_extrude,
    build_helical_sweep,
    build_loft,
    build_loft_through,
    build_revolve,
    build_revolve_profile,
)
from mesh2cad.fit import (
    blend_ring,
    conform_rings,
    decompose_polygon,
    detect_arrays,
    detect_patterns,
    grow_edges,
    half_arc_options,
    plot_series,
    ring_polyline,
    ring_thinness,
)
from mesh2cad.frame import (
    FRAME_BLOCK,
    declared_frame,
    export_source,
    frame_block,
    ingest_frame,
    same_frame,
)
from mesh2cad.frame import describe_frame as describe_program_frame
from mesh2cad.normalize import classify, ingest
from mesh2cad.program import (  # noqa: E402
    GLUES,
    ProgramError,
    delta_line,
    describe,
    drop_feature,
    intersect_into,
    join_program,
    keep_current,
    last_residual,
    last_volume,
    record_residual,
    replace_feature,
    residual_totals,
    sliver_block,
    snapshot,
    split_program,
    summary_line,
    undo,
    without_slivers,
)
from mesh2cad.render import (
    RenderConfig,
    comparison_sheets,
    ortho_level_views,
    overlay_views,
    section_diff_plot,
)
from mesh2cad.ring_groups import GROUP_MIN_PRIMITIVES, run_line  # noqa: E402
from mesh2cad.ring_groups import group as group_ring  # noqa: E402
from mesh2cad.section import (
    LEVEL_SOURCES,
    PLANES,
    CutError,
    feature_levels,
    plane_survey,
    project_slab,
    section_at,
    spectrum,
)
from mesh2cad.summary import recall as recall_output

DEFAULT_OUT = "out/agent"
# Tile for the clay elevation panes (`ortho`, and `section`'s companion view).
# Bigger than the other stages' tiles and it has to be: those panes share ONE
# pixels-per-unit across all six, set by the part's LARGEST dimension, so a part
# much longer than it is deep gets very few pixels on its normal axis -- the one
# axis this picture exists to resolve. At 560 a seven-level part landed its
# levels 14 px apart, closer than the rules are wide; 900 buys 25.
ORTHO_TILE = 900
# The STL tolerance the scorer uses (CADENA's evaluation, build_mesh). The
# health check meshes at the same number, so "passes here" means "scoreable".
STL_TOL = 0.01


def out_dir(args):
    """Per-mesh output directory.

    The default used to be one fixed path shared by every invocation, so a run
    on part B would read back part A's leftover overview.png. That actually
    happened: an agent read a stale render of an unrelated part, believed it,
    and burned a round trip before noticing. Namespacing by the mesh removes the
    failure mode entirely.
    """
    d = Path(args.out)
    if args.out == DEFAULT_OUT:
        d = d / Path(args.mesh).resolve().parent.name
    d.mkdir(parents=True, exist_ok=True)
    return d


PLANE_HELP = ("the section plane; the levels are measured along its normal "
              "(XY -> Z, YZ -> X, XZ -> Y)")
LEVELS_HELP = ("where the level table comes from. faces: the flat faces across "
               "the axis and the ends of round ones -- exact, and a surface "
               "lying across the axis gives no level. sweep: where the area of "
               "a stack of cuts changes. both (the default): faces, plus any "
               "sweep level no face explains, labelled as such. The default "
               "can be set with $MESH2CAD_LEVELS")


def _cache_dir(args):
    """Where a segmentation is kept between calls: next to the renders."""
    return out_dir(args) / ".m2c" / "cache"


def _levels(args, mesh, plane=None):
    return feature_levels(mesh, plane or args.plane, args.samples,
                          source=getattr(args, "levels", None),
                          cache_dir=_cache_dir(args))


MERIDIAN_HELP = ("the profile about an axis at EVERY angle instead of one cut: "
                 "a round face F# or hole stack C# from `faces` (its axis, which "
                 "must be X, Y or Z), or a line like Y=0,Z=0. 72 half-plane "
                 "cuts through the axis give two profiles: the envelope (material "
                 "at ANY angle -- holes filled, blades and teeth as full rings) and "
                 "the core (material at EVERY angle -- blades and teeth gone). Use it "
                 "when a cut through the axis crosses holes, blades or teeth. With "
                 "sketch: --op revolve --use <its rings>")


PROJECT_HELP = ("project everything between --offset (sketch: --at) and this "
                "level onto the plane instead of cutting at one level: the "
                "outline of all that material, as named rings. A joggled or "
                "bent plate is flat at no level, so a cut shows only part of "
                "it; its projection is the whole outline")


def _project_polys(args, mesh, level, marks):
    """(polys, other level, note) for --project, or a loud refusal."""
    other = resolve_level(args.project, marks, "--project", mesh, args.plane)
    if abs(other - level) < 1e-9:
        raise SystemExit(f"--project {args.project} is the level itself: there is "
                         f"no material between a level and itself to project")
    polys = project_slab(mesh, args.plane, level, other)
    if not polys:
        raise SystemExit(f"--project: no material between {level:.4f} and "
                         f"{other:.4f} to project")
    ax = "XYZ"[PLANES[args.plane]["axis"]]
    return polys, other, (f"PROJECTED: everything from {ax}={min(level, other):.4f} to "
                          f"{ax}={max(level, other):.4f} laid flat on the plane -- "
                          f"its outline, not a cut at one level")


def _dense(poly, parts=64):
    """`poly` with a point every 1/parts of its larger extent along each edge."""
    b = poly.bounds
    return poly.segmentize(max(b[2] - b[0], b[3] - b[1]) / parts)


# A meridian polygon with more vertices than this is a trace, not a profile:
# the core of a tapered thread can come back with thousands of primitives, and
# the revolve dies in RecursionError.
MERIDIAN_MAX_VERTICES = 240
# ...and it is simplified by no more than this share of its size to get there.
MERIDIAN_MAX_SIMPLIFY = 0.01


def _vertices(q):
    return sum(len(r.coords) for g in getattr(q, "geoms", [q])
               for r in [g.exterior, *g.interiors])


def _capped(polys, what):
    """`polys` with each one traced at too many vertices simplified -- at the
    least tolerance that brings it under MERIDIAN_MAX_VERTICES, at most
    MERIDIAN_MAX_SIMPLIFY of its size -- and a note for each. A threaded
    stretch is why: the core is the least radius over 72+ cuts, a thread's
    root scalloped a little at every cut angle, and simplified it is the root
    line or cone it is."""
    out, notes = [], []
    for q in polys:
        n0 = _vertices(q)
        if n0 <= MERIDIAN_MAX_VERTICES:
            out.append(q)
            continue
        x0, y0, x1, y1 = q.bounds
        size = max(x1 - x0, y1 - y0) or 1.0
        tol, got = 1e-4 * size, q
        while tol <= MERIDIAN_MAX_SIMPLIFY * size:
            got = q.simplify(tol, preserve_topology=True)
            if _vertices(got) <= MERIDIAN_MAX_VERTICES:
                break
            tol *= 2.0
        out.append(got)
        notes.append(f"the {what} profile was traced at {n0} vertices (the "
                     f"cuts' {'intersection' if what == 'core' else 'union'} over a "
                     f"thread or a knurl follows every turn); simplified by "
                     f"{min(tol, MERIDIAN_MAX_SIMPLIFY * size):.4f} mm "
                     f"({min(tol, MERIDIAN_MAX_SIMPLIFY * size) / size:.2%} of its size) to "
                     f"{_vertices(got)} before it is fitted -- a threaded stretch's root "
                     f"(core) or crest (envelope) line")
    return out, notes


def _meridian(args, mesh):
    """Both meridian profiles about --meridian's axis, as named rings."""
    from mesh2cad.meridian import MeridianError, axis_from, profiles, to_plane
    from mesh2cad.segment import segment_cached

    try:
        k, centre, what = axis_from(
            args.meridian, lambda: segment_cached(mesh, cache_dir=_cache_dir(args)))
        env, core, notes = profiles(mesh, k, centre)
    except MeridianError as exc:
        raise SystemExit(f"--meridian {args.meridian}: {exc}") from None
    plane, level, env_p, axis = to_plane(env, k, centre)
    _, _, core_p, _ = to_plane(core, k, centre)
    # A meridian polygon carries only the corners of a straight run, and the
    # fitter, given a few concyclic corners, fits one arc through them: a
    # stepped cylinder's profile came back with an arc from (20, 0) through
    # (-1, 1) to (0, 22), across the axis, and the revolve never built. Points
    # along every edge let it see the lines.
    env_p, env_n = _capped(env_p, "envelope")
    core_p, core_n = _capped(core_p, "core")
    notes = list(notes) + env_n + core_n
    env_p = [_dense(q) for q in env_p]
    core_p = [_dense(q) for q in core_p]
    rings_env, ri, ni = _rings_from_polys(env_p, args.tol, args.grid)
    rings_core, _, _ = _rings_from_polys(core_p, args.tol, args.grid,
                                         ring_start=ri, node_start=ni)
    t = 0 if k == 0 else 1
    _snap_to_levels(rings_env + rings_core, t, _flat_levels(env_p + core_p, t))
    return {"k": k, "centre": centre, "what": what, "notes": notes, "plane": plane,
            "level": level, "axis": axis, "env": (env_p, rings_env),
            "core": (core_p, rings_core)}


# A meridian point this near a level along the axis is on it. A fitted line
# that ends a ten-thousandth below the part's end leaves a thin lip there that
# a cut from the end does not reach, bridging the cut's mouth -- a handle.
MERIDIAN_SNAP = 1e-3


def _flat_levels(polys, t):
    """The levels along the axis (coordinate `t` of the plane) where the
    meridian polygons run square to it -- a face across the axis -- and their
    two ends."""
    out = set()
    for poly in polys:
        for ring in [poly.exterior, *poly.interiors]:
            c = np.asarray(ring.coords, float)
            for a, b in zip(c[:-1], c[1:], strict=True):
                if abs(a[t] - b[t]) <= 1e-9 and abs(a[1 - t] - b[1 - t]) > 10 * MERIDIAN_SNAP:
                    out.add(round(float(a[t]), 6))
        lo, hi = poly.bounds[t], poly.bounds[t + 2]
        out.update((round(float(lo), 6), round(float(hi), 6)))
    return sorted(out)


def _snap_to_levels(rings, t, levels):
    """Every end point of the rings' segments (and every node) whose coordinate
    `t` is within MERIDIAN_SNAP of one of `levels`, put on it."""
    if not levels:
        return
    lv = np.asarray(levels, float)

    def put(p):
        i = int(np.argmin(np.abs(lv - p[t])))
        if 0.0 < abs(lv[i] - p[t]) <= MERIDIAN_SNAP:
            p = list(p)
            p[t] = float(lv[i])
        return p
    for r in rings:
        for n in r["nodes"]:
            n["uv"] = put(n["uv"])
        for sg in r["segments"]:
            for key in ("from_xy", "to_xy"):
                if key in sg:
                    sg[key] = put(sg[key])


# A ring this long gets a one-line shape summary above its primitives:
# under it the list is already read at a glance.
SHAPE_SUMMARY_MIN = 8
# Values within this fraction of a group's smallest are one size in the summary.
SHAPE_GROUP_FRAC = 0.15


def _size_groups(values, fmt, prefix=""):
    """`values` grouped into sizes, commonest first: "R2.74-3.10 x14"."""
    vals = sorted(values)
    groups = []
    for v in vals:
        if groups and v <= groups[-1][0] * (1 + SHAPE_GROUP_FRAC):
            groups[-1].append(v)
        else:
            groups.append([v])
    groups.sort(key=lambda g: (-len(g), g[0]))
    out = []
    for g in groups[:5]:
        size = (fmt.format(g[0]) if g[-1] - g[0] < 5e-4
                else f"{fmt.format(g[0])}-{fmt.format(g[-1])}")
        out.append(f"{prefix}{size} x{len(g)}")
    if len(groups) > 5:
        out.append(f"{sum(len(g) for g in groups[5:])} in {len(groups) - 5} other sizes")
    return ", ".join(out)


def _ring_lines(r, detail=False):
    """A ring as `section` prints it: the header, a shape summary when the ring
    is long, and one line per primitive with its end points written out -- not a
    separate block of N# lines that nothing takes as input.

    A long ring prints its repeats as groups (mesh2cad/ring_groups.py) unless
    `detail`, which is `--detail R#`."""
    out = [f"\n--- ring {r.get('name', '?')} ({r['role']}): "
           f"{r['n_primitives']} primitives "
           f"({r['n_lines']} lines, {r['n_arcs']} arcs), max fit residual "
           f"{r['max_residual']:.5f} mm"]
    segs = r["segments"]
    if len(segs) >= SHAPE_SUMMARY_MIN:
        lines = [sg["length"] for sg in segs if sg["kind"] == "line"]
        arcs = [sg["radius"] for sg in segs if sg["kind"] == "arc"]
        parts = []
        if lines:
            parts.append(f"lines {_size_groups(lines, '{:.4f}')}")
        if arcs:
            parts.append(f"arcs {_size_groups(arcs, '{:.4g}', 'R')}")
        out.append(f"    shape: {'; '.join(parts)}")
    at = {n["name"]: n["uv"] for n in r["nodes"]}

    def pt(name):
        u, v = at.get(name, (float("nan"), float("nan")))
        return f"({u:10.4f}, {v:10.4f})"

    def line(sg):
        if sg["kind"] == "line":
            return (f"    {sg['name']:<8s} line {pt(sg['from'])} -> {pt(sg['to'])}  "
                    f"len={sg['length']:9.4f}  res={sg['residual']:.5f}")
        return (f"    {sg['name']:<8s} arc  {pt(sg['from'])} -> {pt(sg['to'])}  "
                f"R={sg['radius']:9.4f} centre=({sg['centre'][0]:.4f}, "
                f"{sg['centre'][1]:.4f}) sweep={sg['sweep_deg']:7.2f}deg "
                f"{'CCW' if sg.get('ccw') else 'CW'} res={sg['residual']:.5f}")
    if detail or len(segs) < GROUP_MIN_PRIMITIVES:
        return out + [line(sg) for sg in segs]
    runs, listed, unlisted = group_ring(segs)
    if not runs and not unlisted:
        return out + [line(sg) for sg in segs]
    rows = [(i, run_line(segs, (i, p, k), at)) for i, p, k in runs]
    rows += [(i, line(segs[i])) for i in listed]
    covered = sum(p * k for _, p, k in runs)
    out.append(f"    grouped: {len(runs)} repeat(s) cover {covered} of {len(segs)} "
               f"primitives; `--detail {r.get('name', '?')}` lists every one")
    out += [text for _, text in sorted(rows, key=lambda row: row[0])]
    if unlisted:
        out.append(f"    [{unlisted} shorter primitives of this ring are not listed: "
                   f"they fit no repeat and only the {len(listed)} longest are "
                   f"shown. `--detail {r.get('name', '?')}` lists them]")
    return out


def _detailed(args, r):
    return r.get("name") in (getattr(args, "detail", None) or [])


def _print_rings(rings, args):
    for r in rings:
        print("\n".join(_ring_lines(r, _detailed(args, r))))


def _detail_note(args, rings):
    """--detail on a ring this section does not have says so."""
    names = [r.get("name") for r in rings]
    for name in getattr(args, "detail", None) or []:
        if name not in names:
            print(f"note: --detail {name}: no such ring in this section "
                  f"(it has {', '.join(n for n in names if n) or 'none'})")


def _section_meridian(args, mesh):
    from mesh2cad.meridian import ANGLES

    m = _meridian(args, mesh)
    spec = PLANES[m["plane"]]
    ax = "XYZ"[m["k"]]
    d = out_dir(args)
    tag = "".join(c if c.isalnum() else "_" for c in args.meridian)
    png = d / f"meridian_{tag}.png"
    series = [{"rings": m["env"][1], "polys": m["env"][0],
               "label": "envelope: material at ANY angle",
               "colours": ("#1f5fa8", "#1f5fa8", "#12304f"), "style": "-", "lw": 2.6,
               "label_offset": (6, 6)}]
    if m["core"][1]:
        series.append({"rings": m["core"][1], "polys": m["core"][0],
                       "label": "core: material at EVERY angle",
                       "colours": ("#b3341f", "#b3341f", "#5a1a10"), "style": "--",
                       "lw": 1.8, "label_offset": (6, -14)})
    plot_series(series, spec["u_name"], spec["v_name"],
                f"meridian about {m['what']} ({ax}), drawn in {m['plane']} at "
                f"{'XYZ'[spec['axis']]}={m['level']:.3f}", png)
    print(f"IMAGE: {png}")
    print(f"meridian about {m['what']}: the part cut by {ANGLES} half-planes "
          f"through the axis (one every {360 // ANGLES} deg), drawn as the upper "
          f"half of plane {m['plane']} at {'XYZ'[spec['axis']]} = {m['level']:.4f}; "
          f"in-plane axes ({spec['u_name']}, {spec['v_name']}), the axis is the "
          f"line {m['axis'][0]} -> {m['axis'][1]}")
    for n in m["notes"]:
        print(f"note: {n}")
    print("\n=== envelope: material at ANY angle (holes filled; blades, teeth and "
          "bosses as full rings) ===")
    _print_rings(m["env"][1], args)
    if m["core"][1]:
        print("\n=== core: material at EVERY angle (blades, teeth and bosses gone; "
              "holes leave a gap) ===")
        _print_rings(m["core"][1], args)
    else:
        print("\n(no core: no material is there at every angle)")
    cuts = getattr(args, "_cuts", None)
    if cuts:
        # `section --meridian --split` must print the cut profile, so the cut
        # can be looked at.
        for label, (raw, rs) in (("envelope", m["env"]), ("core", m["core"])):
            outer = [r for r in rs if r["role"] == "outer"]
            if not outer:
                continue
            try:
                o, h, desc = _meridian_cut(args, mesh, spec, outer,
                                           [r for r in rs if r["role"] == "hole"], raw)
            except SystemExit as exc:
                print(f"\n=== {label} cut: {exc} ===")
                continue
            print(f"\n=== {label} cut at {desc} ===")
            _print_rings(o + h, args)
    _detail_note(args, m["env"][1] + m["core"][1])
    first = m["env"][1][0]["name"] if m["env"][1] else "R0"
    split = "".join(f' --split "{e}" --keep {k}' for e, k in (cuts or []))
    print(f"\nto build one: sketch --meridian {args.meridian} --use {first}{split} "
          f"--op revolve; then cut what is not round from it (or add to the core)")


# The cut through a revolve axis is called short of the profile when either
# half misses more than this share of the meridian envelope's area.
MERIDIAN_HINT_SHORTFALL = 0.02
# Only an axis carrying this share of the part's area is a revolve axis worth
# the hint (and the 72 cuts it costs); a small boss's axis is not.
MERIDIAN_HINT_MIN_AREA = 0.10
# ...or one the overview's revolve probe (normalize._revolve_probe) finds the
# part this close to round about: a drum whose faces the segmentation does not
# recognise as round (mostly freeform at the section's budget) carries little
# area on its axis, and would get no hint on exactly the part it is for.
MERIDIAN_HINT_PROBE = 0.6


def _meridian_hint(args, mesh, plane, level):
    """One line, or None: this section passes through a revolve axis but is
    not its profile, because something not round about that axis crosses it.

    A model rarely thinks of the meridian on its own when a cut through a
    revolve's axis is crossed by teeth, blades or bores. The section is where it
    decides, so that is where to say it.
    """
    from mesh2cad.meridian import _others, shortfall
    from mesh2cad.section import LEVELS_SEGMENT_BUDGET
    from mesh2cad.segment import coaxial_groups, principal_of, segment_cached

    kp = PLANES[plane]["axis"]
    try:
        seg = segment_cached(mesh, cache_dir=_cache_dir(args),
                             budget=LEVELS_SEGMENT_BUDGET)
    except Exception:                                              # noqa: BLE001
        return None
    span = float(np.ptp(mesh.bounds[:, kp])) or 1.0
    best = None
    for g in coaxial_groups(seg, min_size=1):
        s0 = seg.surfaces[g[0]]
        ax = principal_of(s0.params["axis"], tol_rad=1e-4)
        if ax is None or "XYZ".index(ax) == kp:
            continue
        if abs(float(s0.params["point"][kp]) - level) > max(1e-4 * span, 10 * seg.tol):
            continue
        area = sum(seg.surfaces[i].area for i in g)
        if best is None or area > best[0]:
            best = (area, "XYZ".index(ax), np.asarray(s0.params["point"], float), g)
    if best is None:
        return None
    # The probe's axis is a world one; in a frame it says nothing.
    regime = None if getattr(args, "_frame_info", None) else getattr(args, "_regime", None)
    ev = (regime or {}).get("evidence", {})
    probed = (ev.get("revolve_axis") == "XYZ"[best[1]]
              and (ev.get("revolve_score") or 0.0) >= MERIDIAN_HINT_PROBE)
    if best[0] < MERIDIAN_HINT_MIN_AREA * (seg.area or 1.0) and not probed:
        return None
    _area, k, p, g = best
    i, j = _others(k)
    spec = PLANES[plane]
    side = spec["u"] if spec["u"] != k else spec["v"]
    miss = shortfall(mesh, k, (float(p[i]), float(p[j])), side)
    if not miss or max(miss) < MERIDIAN_HINT_SHORTFALL:
        return None
    rounds = [seg.surfaces[i_] for i_ in g if "axis" in seg.surfaces[i_].params]
    name = max(rounds, key=lambda s: s.area).name
    return (f"note: this cut goes through the {'XYZ'[k]} axis of {name} but is short "
            f"of the profile by {100 * max(miss):.0f}% -- holes, blades or teeth "
            f"cross it, so these rings are not what was revolved. `section "
            f"--meridian {name}` gives the profile at every angle")


def _meridian_cut(args, mesh, spec, outer, holes, raw):
    """(outer, holes, desc): the meridian rings named, cut by every --split,
    with their roles kept and the meridian's names, as `sketch --split` keeps
    them. A cut that keeps nothing is refused, saying why.

    The cut is made on the meridian's own polygons (`raw`), and the kept
    pieces fitted again, as `split_profile` does for a section: cutting the
    fitted rings would carry a fitted arc that strays past the axis into the
    piece, and the revolve would not build."""
    polys = []
    for o in outer:
        g = readback.shape(o)
        if g is None or not raw:
            continue
        best = max(raw, key=lambda p_: g.intersection(p_).area / max(g.union(p_).area, 1e-12))
        if all(best is not q for q in polys):
            polys.append(best)
    rings = outer + holes
    cuts = [(resolve_split_expr(e, rings, mesh=mesh, args=args, spec=spec), k)
            for e, k in args._cuts]
    _, kept, _, desc, clip = split_profile(polys, spec, cuts[0][0], cuts[0][1], args.tol,
                                           args.grid, more=cuts[1:])
    _source_names(kept, rings, clip)
    out = [r for r in kept if r["role"] == "outer"]
    # the polygons bring every hole they have; keep the ones --use named
    kept_holes = [k for k in kept if k["role"] == "hole"
                  and any(_same_piece(k, h, clip) for h in holes)]
    print(f"meridian cut at {desc}: keeps {readback.listing(out + kept_holes)}")
    return out, kept_holes, desc


def _sketch_meridian(args, mesh):
    from mesh2cad.meridian import revolve_code

    if args.op != "revolve":
        raise SystemExit(f"--meridian builds a revolve about its axis: give --op "
                         f"revolve (got --op {args.op})")
    m = _meridian(args, mesh)
    rings = m["env"][1] + m["core"][1]
    if not args.use:
        args.use = ",".join(r["name"] for r in m["env"][1])
        print(f"--meridian without --use: the envelope, rings {args.use}")
    picked = [_resolve(n, rings, {}) for n in args.use.split(",") if n.strip()]
    chosen = []
    for kind, r in picked:
        if kind == "ring" and all(r is not c for c in chosen):
            chosen.append(r)
    outer = [r for r in chosen if r["role"] == "outer"]
    holes = [r for r in chosen if r["role"] == "hole"]
    if not outer:
        raise SystemExit("--use named no outer ring of the meridian")
    core_ids = {id(r) for r in m["core"][1]}
    args.plane = m["plane"]
    spec = PLANES[args.plane]
    ax = "XYZ"[spec["axis"]]
    named = ", ".join(f"{r['name']} ({'core' if id(r) in core_ids else 'envelope'}, "
                      f"{r['role']})" for r in chosen)
    cut = ""
    if getattr(args, "_cuts", None):
        # --split cuts the half profile across (or along) the axis -- the hex
        # head off a bolt's shank -- and is read before this path returns, so
        # the whole envelope is not revolved instead.
        outer, holes, desc = _meridian_cut(args, mesh, spec, outer, holes,
                                           m["env"][0] + m["core"][0])
        cut = f", cut at {desc}"
    if getattr(args, "axis", None):
        # R2: read and dropped like --split was (286, 284 gave both).
        print(f"note: --axis {args.axis.strip()} is not used: --meridian revolves about "
              f"its own axis, {m['what']}")
    angle = args.angle
    to = getattr(args, "to", None)
    if to is not None:
        if abs(args.angle - 360.0) > 1e-9:
            raise SystemExit(f"--to {to} and --angle {args.angle:g} both say where the "
                             f"revolve ends; give one")
        angle, end_note = _revolve_end(args, mesh, spec, m["level"], m["axis"], m["what"],
                                       outer, holes, str(to).strip())
        print(end_note)
    code = revolve_code(args.plane, m["level"], outer, holes, m["axis"], angle)
    detail = (f"{angle:g} deg about {m['what']}{cut}, meridian rings " + named)
    _finish_sketch(args, code, ax, m["level"], m["level"], None, detail, "revolve",
                   profiles=f"{sum(r['n_primitives'] for r in chosen)} primitives "
                            f"of the meridian")
SHARP_HELP = ("with --face: where a chamfer, a round or a turned end (a face "
              "leaning across the outline, not a wall) has cut the face's edge "
              "back, carry the walls on either side -- and the walls those faces "
              "meet beyond it -- on to their corners instead: the sharp block, to "
              "cut the blend from with --op chamfer / --op fillet")


FRAME_HELP = ("work across an inclined face or axis: a flat face F# (across its "
              "normal), a round face F# or a hole stack C# (across its axis), "
              "as `faces` names them. The target is turned by the smallest "
              "rotation that lays that direction on X, Y or Z; every level, "
              "ring and coordinate is then in that frame, and a sketch is built "
              "there and turned back when it is written. Give the same --frame "
              "to ortho, section and sketch. --face on an inclined face and "
              "--op hole on an inclined stack set it themselves")


def _implied_frame(args, seg):
    """The frame an inclined --face or --op hole stack names by itself, or None.

    Not a choice the tool makes: an inclined flat face has one plane it can be
    sketched on, and a hole stack one axis it can be cut along.
    """
    from mesh2cad.frames import inclined
    from mesh2cad.holes import stack_names

    face = getattr(args, "face", None)
    if face:
        s = seg.by_name(face.upper())
        if s is not None and s.kind == "plane" and inclined(s.params["normal"]):
            return face.upper()
        return None
    if getattr(args, "op", None) == "hole" and getattr(args, "use", None):
        stacks = dict(stack_names(seg))
        names = [n.strip().upper() for n in args.use.split(",") if n.strip()]
        axes = [(n, np.asarray(seg.surfaces[stacks[n][0]].params["axis"], float))
                for n in names if n in stacks]
        tilted = [(n, a) for n, a in axes if inclined(a)]
        if not tilted:
            return None
        n0, a0 = tilted[0]
        if any(abs(float(a @ a0)) < 1 - 1e-6 for _, a in axes):
            return None          # several axes: _sketch_hole frames each one
        return n0
    return None


def _apply_frame(args, mesh):
    """The mesh to work on: the target, or the target turned into --frame's
    frame (or the one an inclined --face / hole stack implies).

    The frame is kept on args for `_finish_sketch`, which turns the feature
    back. A direction already on an axis needs no frame and gets none, said.
    """
    from mesh2cad.frames import (
        PLANE_ACROSS,
        FrameError,
        describe_frame,
        direction_of,
        frame_for,
        inclined,
        turn_mesh,
    )
    from mesh2cad.segment import frame_source, segment_cached

    args._frame_info = None
    name = getattr(args, "frame", None)
    if not name and not getattr(args, "face", None) and getattr(args, "op", None) != "hole":
        return mesh
    seg = segment_cached(mesh, cache_dir=_cache_dir(args))
    implied = None
    if not name:
        implied = name = _implied_frame(args, seg)
        if not name:
            # A flat face on an axis has one plane it can be sketched on, as
            # an inclined one has: take it rather than refusing for --plane.
            s = seg.by_name(args.face.upper()) if getattr(args, "face", None) else None
            if (s is not None and s.kind == "plane" and hasattr(args, "plane")
                    and args.plane is None):
                k = int(np.argmax(np.abs(np.asarray(s.params["normal"], float))))
                args.plane = PLANE_ACROSS[k]
                print(f"note: --face {s.name} lies across {'XYZ'[k]}, so the "
                      f"plane is {args.plane}")
            elif (s is not None and s.kind != "plane" and hasattr(args, "plane")
                    and args.plane is None):
                # A cone or a cylinder named in --face needs no --plane: refusing it
                # for a missing one would be an error no plane could mend.
                about = (f" about {'XYZ'[int(np.argmax(np.abs(s.params['axis'])))]}"
                         if "axis" in s.params else "")
                raise SystemExit(
                    f"--face {s.name} is a {s.kind}{about}, not a flat face: --face takes "
                    f"a flat face, whose outline is the profile (`faces` lists them as "
                    f"plane). A round face is turned or bored: --op revolve --axis "
                    f"{s.name} turns a profile about it, --op hole makes a bore")
            return mesh
    name = name.strip().upper()
    try:
        d, what = direction_of(seg, name)
    except FrameError as exc:
        raise SystemExit(f"--frame {name}: {exc}") from None
    fr = frame_for(d)
    across = PLANE_ACROSS[fr["k"]]
    if not inclined(d):
        print(f"note: --frame {name}: {what} already lies on "
              f"{'XYZ'[fr['k']]}, so no frame is needed; working in the target's "
              f"own frame (the plane across it is {across})")
        if hasattr(args, "plane") and args.plane is None:
            args.plane = across
        return mesh
    if getattr(args, "face", None) and getattr(args, "plane", None) not in (None, across):
        raise SystemExit(f"--face {args.face}: in the frame of {name} the face "
                         f"lies across {'XYZ'[fr['k']]}, so its plane is {across}; "
                         f"give --plane {across} or leave --plane out")
    if hasattr(args, "plane") and args.plane is None:
        args.plane = across
    turned = turn_mesh(mesh, fr["R"])
    frame_source(turned, mesh, fr["R"])
    args._frame_info = (name, fr, what)
    print(("note: " if implied else "") + describe_frame(fr, name, what)
          + (f" (set by {'--face' if getattr(args, 'face', None) else '--op hole'} "
             f"{name}, which is inclined)" if implied else ""))
    return turned


# A wall of a prism along the sketch axis has its normal across that axis: a
# neighbour across the face's outline whose normal leans more than this (sine,
# about 6 deg) out of the plane is a chamfer, a round or a taper, not a wall.
FACE_LEAN_SIN = 0.1
# The outline is said to lean when this share of its length does.
FACE_LEAN_SHARE = 0.02
# A plane (normal) or cylinder (axis) within this sine of square to the sketch
# axis is a wall that --sharp may carry to a corner.
FACE_WALL_SIN = 0.035


def _wall_curve(mesh, seg, i, k, u, v):
    """Surface i as the curve it draws in the sketch plane when it is a wall
    of a prism along axis k: ("line", point, unit direction) or ("circle",
    centre, radius); None for anything else."""
    s = seg.surfaces[i]
    P = np.asarray(mesh.vertices, float)[np.unique(mesh.faces[s.faces])][:, [u, v]]
    if s.kind == "plane":
        n = np.asarray(s.params["normal"], float)
        n = n / np.linalg.norm(n)
        if abs(n[k]) > FACE_WALL_SIN:
            return None
        n2 = n[[u, v]] / np.linalg.norm(n[[u, v]])
        return ("line", P.mean(axis=0), np.array([-n2[1], n2[0]]))
    if s.kind == "cylinder":
        a = np.asarray(s.params["axis"], float)
        if abs(a[k]) / np.linalg.norm(a) < 1.0 - FACE_WALL_SIN ** 2:
            return None
        c = np.asarray(s.params["point"], float)[[u, v]]
        return ("circle", c, float(np.median(np.linalg.norm(P - c, axis=1))))
    return None


def _curve_dist(C, P):
    """Distance of points P (n, 2) from curve C."""
    if C[0] == "line":
        n = np.array([-C[2][1], C[2][0]])
        return np.abs((P - C[1]) @ n)
    return np.abs(np.linalg.norm(P - C[1], axis=1) - C[2])


def _curves_meet(A, B):
    """Where two wall curves cross, as a list of points."""
    if A[0] == "circle" and B[0] == "line":
        A, B = B, A
    if A[0] == "line" and B[0] == "line":
        M = np.array([A[2], -B[2]]).T
        if abs(np.linalg.det(M)) < 1e-9:
            return []
        t = np.linalg.solve(M, B[1] - A[1])[0]
        return [A[1] + t * A[2]]
    if A[0] == "line":
        f = A[1] - B[1]
        b, c = f @ A[2], f @ f - B[2] ** 2
        disc = b * b - c
        if disc < 0:
            return []
        return [A[1] + (-b + sg * np.sqrt(disc)) * A[2] for sg in (-1.0, 1.0)]
    d = B[1] - A[1]
    dd = float(np.linalg.norm(d))
    if dd < 1e-9 or dd > A[2] + B[2] or dd < abs(A[2] - B[2]):
        return []
    a = (A[2] ** 2 - B[2] ** 2 + dd * dd) / (2 * dd)
    h = np.sqrt(max(A[2] ** 2 - a * a, 0.0))
    m, perp = A[1] + a * d / dd, np.array([-d[1], d[0]]) / dd
    return [m + h * perp, m - h * perp]


def _along_curve(C, p, q, near):
    """Points from p (excluded) to q along wall curve C; round a circle, the
    way that passes nearer `near`."""
    if C[0] == "line":
        return [np.asarray(q, float)]
    c, r = C[1], C[2]
    a0 = np.arctan2(*(p - c)[::-1])
    a1 = np.arctan2(*(q - c)[::-1])
    ccw = (a1 - a0) % (2 * np.pi)
    best = None
    for sweep in (ccw, ccw - 2 * np.pi):
        mid = c + r * np.array([np.cos(a0 + sweep / 2), np.sin(a0 + sweep / 2)])
        d = float(np.linalg.norm(mid - near))
        if best is None or d < best[0]:
            best = (d, sweep)
    sweep = best[1]
    n = max(2, int(abs(sweep) / np.radians(3.0)))
    t = a0 + sweep * np.linspace(0.0, 1.0, n + 1)[1:]
    return [c + r * np.array([np.cos(x), np.sin(x)]) for x in t]


def _outline_lean(mesh, seg, idx, loop, by_edge, mine, k):
    """Per edge of a face's boundary loop: (neighbour surface, leans, length,
    angle to the face in degrees)."""
    V = np.asarray(mesh.vertices, float)
    N = np.asarray(mesh.face_normals, float)
    labels = np.asarray(seg.labels)
    out = []
    for a, b in zip(loop, loop[1:] + loop[:1], strict=True):
        nb = [f for f in by_edge.get((min(a, b), max(a, b)), []) if f not in mine]
        if not nb:
            out.append((-1, False, 0.0, 90.0))
            continue
        c = min(abs(float(N[nb[0], k])), 1.0)
        out.append((int(labels[nb[0]]), c > FACE_LEAN_SIN,
                    float(np.linalg.norm(V[a] - V[b])), float(np.degrees(np.arccos(c)))))
    return out


def _sharp_loop(mesh, seg, idx, P2, lean, k, u, v):
    """The loop with each stretch that leaning faces cut back replaced by the
    walls on either side of it -- and the walls those faces meet beyond it --
    carried on to their corners. Returns (points, wall names, stretches
    closed, stretches left as they were, the leaning faces of the closed
    ones)."""
    n = len(P2)
    flags = [x[1] for x in lean]
    labs = [x[0] for x in lean]
    curves = {}

    def curve(i):
        if i not in curves:
            curves[i] = _wall_curve(mesh, seg, i, k, u, v) if i >= 0 else None
        return curves[i]

    nb = {}
    for j in seg.joints:
        nb.setdefault(j.a, set()).add(j.b)
        nb.setdefault(j.b, set()).add(j.a)

    def beyond(leaners):
        # The walls the leaning faces meet, or meet through one more face that
        # is no wall: a plate end can be freeform pieces whose end wall joins
        # only the cones behind them.
        near = {j for s in leaners for j in nb.get(s, ()) if j != idx and j not in leaners}
        via = {j for s in near if curve(s) is None for j in nb.get(s, ())
               if j != idx and j not in leaners}
        return {j for j in near | via if curve(j) is not None}

    if not any(flags):
        return P2, [], 0, 0, set()
    # A stretch can end on faces that stand upright but are no plane or
    # cylinder -- freeform pieces round a plate end -- and there is no wall
    # there to carry on. Those edges go with the stretch next to them, so it
    # runs on to the nearest wall either side.
    leaning = flags
    flags = _with_neighbours(leaning, [curve(lb) is not None for lb in labs])
    bent = {lb for lb, f in zip(labs, leaning, strict=True) if f}
    if all(flags):
        ring = _sharp_ring(P2, beyond(set(labs)), curve)
        if ring is None:
            return P2, [], 0, 1, set()
        return ring[0], [seg.surfaces[w].name for w in sorted(ring[1])], 1, 0, bent
    # Edge e runs from vertex e to vertex e + 1. Start on an edge along a wall.
    start = next(i for i in range(n) if not flags[i])
    order = [(start + i) % n for i in range(n)]
    pts, used, cut_by, closed, left, r = [P2[order[0]]], set(), set(), 0, 0, 0
    while r < n:
        if not flags[order[r]]:
            if r + 1 < n:
                pts.append(P2[order[r + 1]])
            r += 1
            continue
        q = r
        while q < n and flags[order[q]]:
            q += 1
        run_v = np.asarray([P2[order[x % n]] for x in range(r, q + 1)])
        leaners = {labs[order[x]] for x in range(r, q)}
        seq, path = _close_stretch(run_v, labs[order[r - 1]], labs[order[q % n]], leaners,
                                   beyond, curve)
        if path is not None:
            pts += path[:-1]
            used.update(seq)
            cut_by.update(leaners & bent)
            closed += 1
        else:
            pts += list(run_v[1:-1])
            left += 1
        if q < n:
            pts.append(run_v[-1])
        r = q
    names = [seg.surfaces[w].name for w in sorted(used)]
    return np.asarray(pts), names, closed, left, cut_by


def _with_neighbours(leaning, wall):
    """Per edge: in a stretch to sharpen -- leaning, or one of a run of
    edges on no wall that touches a leaning one (round the loop)."""
    n = len(leaning)
    out = list(leaning)
    soft = [a or not b for a, b in zip(leaning, wall, strict=True)]
    if all(soft):
        return [True] * n if any(leaning) else out
    start = next(i for i in range(n) if not soft[i])
    i = 0
    while i < n:
        j = i
        while j < n and soft[(start + j) % n]:
            j += 1
        run = [(start + x) % n for x in range(i, j)]
        if any(leaning[x] for x in run):
            for x in run:
                out[x] = True
        i = j + 1
    return out


def _sharp_ring(P2, walls, curve):
    """A loop that leans all the way round (a block chamfered along its whole
    edge, a hole's mouth): the walls those faces meet, in order round the loop
    and carried to their corners -- (points, walls), or None."""
    from shapely.geometry import LinearRing

    span = float(np.ptp(P2, axis=0).max())
    near = []
    for w in walls:
        d = _curve_dist(curve(w), P2)
        if d.min() <= 0.25 * span:
            near.append((int(d.argmin()), w))
    seq = [w for _, w in sorted(near)]
    if not seq:
        return None
    if len(seq) == 1:
        C = curve(seq[0])
        if C[0] != "circle":
            return None
        t = np.linspace(0.0, 2 * np.pi, 121)[:-1]
        ring = C[1] + C[2] * np.stack([np.cos(t), np.sin(t)], axis=1)
    else:
        corners = []
        for A, B in zip(seq, seq[1:] + seq[:1], strict=True):
            meet = _curves_meet(curve(A), curve(B))
            if not meet:
                return None
            corners.append(min(meet, key=lambda p: float(np.min(np.linalg.norm(P2 - p, axis=1)))))
        pts = []
        for x, w in enumerate(seq[1:] + seq[:1]):
            a, b = corners[x], corners[(x + 1) % len(seq)]
            near_w = P2[int(_curve_dist(curve(w), P2).argmin())]
            pts += _along_curve(curve(w), a, b, near_w)
        ring = np.asarray(pts)
    if len(ring) < 3 or not LinearRing(ring).is_valid:
        return None
    if LinearRing(ring).is_ccw != LinearRing(P2).is_ccw:
        ring = ring[::-1]
    return ring, set(seq)


def _close_stretch(run_v, prev_l, next_l, leaners, beyond, curve):
    """The walls, in order, that replace one leaning stretch of an outline, and
    the points along them from its first vertex to its last; ([], None) when
    they do not meet near it."""
    s_pt, e_pt = run_v[0], run_v[-1]
    bound = max(0.5 * float(np.linalg.norm(e_pt - s_pt)), 1e-6)
    mids = []
    for w in beyond(leaners) - {prev_l, next_l}:
        d = _curve_dist(curve(w), run_v)
        if d.min() <= bound:
            mids.append((int(d.argmin()), w))
    seq = [prev_l] + [w for _, w in sorted(mids)] + [next_l]
    seq = [w for x, w in enumerate(seq) if x == 0 or w != seq[x - 1]]
    if any(curve(w) is None for w in seq):
        return [], None
    near = run_v[len(run_v) // 2]
    path, cur = [], s_pt
    for A, B in zip(seq, seq[1:], strict=False):
        meet = _curves_meet(curve(A), curve(B))
        if not meet:
            return [], None
        X = min(meet, key=lambda p: float(np.min(np.linalg.norm(run_v - p, axis=1))))
        if float(np.min(np.linalg.norm(run_v - X, axis=1))) > bound:
            return [], None
        path += _along_curve(curve(A), cur, X, near)
        cur = X
    path += _along_curve(curve(seq[-1]), cur, e_pt, near)
    return seq, path


# A plane section is said to cross leaning faces when this share of its outline
# lies on them (a face's own outline is held to FACE_LEAN_SHARE).
SECTION_LEAN_SHARE = 0.05


def _section_lean(args, mesh, plane, level):
    """A WARNING, or None: this cut's outline runs along faces that lean across
    the sketch axis -- a chamfer, a turned end, a draft -- so the profile here
    is not the same a little way along it, and a prism of it is short or proud
    elsewhere."""
    try:
        import trimesh

        from mesh2cad.segment import segment_cached

        polys = section_at(mesh, plane, level)
        if not polys:
            return None
        spec = PLANES[plane]
        k, u, v = spec["axis"], spec["u"], spec["v"]
        mids, lens = [], []
        for p in polys:
            for ring in (p.exterior, *p.interiors):
                c = np.asarray(ring.coords)
                mids.append((c[:-1] + c[1:]) / 2.0)
                lens.append(np.linalg.norm(np.diff(c, axis=0), axis=1))
        mid, length = np.concatenate(mids), np.concatenate(lens)
        P = np.zeros((len(mid), 3))
        P[:, u], P[:, v], P[:, k] = mid[:, 0], mid[:, 1], level
        _c, _d, tri = trimesh.proximity.closest_point(mesh, P)
        nk = np.abs(np.asarray(mesh.face_normals)[tri][:, k])
        lean = nk > FACE_LEAN_SIN
        share = float(length[lean].sum() / max(length.sum(), 1e-12))
        if not lean.any():
            return None
        blend = _section_through_blend(args, mesh, plane, level, mid[lean], tri[lean])
        if share < SECTION_LEAN_SHARE:
            return blend
        from mesh2cad.section import LEVELS_SEGMENT_BUDGET
        seg = segment_cached(mesh, cache_dir=_cache_dir(args), budget=LEVELS_SEGMENT_BUDGET)
        labels = np.asarray(seg.labels)[tri[lean]]
        by = {}
        for lb, ln in zip(labels, length[lean], strict=True):
            if lb >= 0:
                by[int(lb)] = by.get(int(lb), 0.0) + float(ln)
        S = seg.surfaces
        order = sorted(by, key=lambda i: -by[i])
        listing = (", ".join(f"{S[i].kind} {S[i].name}" for i in order[:4])
                   + (f" and {len(order) - 4} more" if len(order) > 4 else ""))
        ang = np.degrees(np.arccos(np.clip(nk[lean], 0, 1)))
        ax = "XYZ"[k]
        return (f"WARNING: {100 * share:.0f}% of this cut's outline ({length[lean].sum():.1f} of "
                f"{length.sum():.1f} mm) lies on {listing}, which lean across {ax} "
                f"({ang.min():.0f}-{ang.max():.0f} deg off it; a wall is 90): at {ax}={level:.4f} the "
                f"profile is where they cut it back, and it changes along {ax}, so a prism of it is "
                f"short or proud at other levels. If they are chamfers or turned ends, take the "
                f"profile at a level clear of them, or `sketch --face F# --sharp` on a flat face, "
                f"and cut them after with `--op chamfer`. If the side tapers, it is a loft"
                + ("\n" + blend if blend else ""))
    except Exception:                                              # noqa: BLE001
        return None


# A cut that crosses a round blend face (part of a cylinder, cone or torus, not
# a whole hole) is warned whatever share of the outline lies on it, when the
# face spreads this far across the outline: the wall a prism of this profile
# gets is where the cut meets the face, not where the face starts or ends --
# even when the blend is a tiny share of a long outline, under
# SECTION_LEAN_SHARE.
SECTION_BLEND_SPREAD = 0.25     # mm, and ...
SECTION_BLEND_REL = 0.002       # ... at least this share of the diagonal


def _section_through_blend(args, mesh, plane, level, pts, tri):
    """A WARNING, or None: the cut at `level` crosses round blend faces that
    spread across its outline (see SECTION_BLEND_SPREAD). `pts` are outline
    points on leaning facets `tri`, in the plane's (u, v)."""
    from mesh2cad.blends import MAX_SWEEP
    from mesh2cad.section import LEVELS_SEGMENT_BUDGET
    from mesh2cad.segment import segment_cached
    seg = segment_cached(mesh, cache_dir=_cache_dir(args), budget=LEVELS_SEGMENT_BUDGET)
    spec = PLANES[plane]
    k, u, v = spec["axis"], spec["u"], spec["v"]
    labels = np.asarray(seg.labels)[tri]
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces)
    floor = max(SECTION_BLEND_SPREAD, SECTION_BLEND_REL * float(np.linalg.norm(mesh.extents)))
    hits = []
    for lb in np.unique(labels[labels >= 0]):
        s = seg.surfaces[int(lb)]
        if s.kind not in ("cylinder", "cone", "torus") or s.extent.get("sweep", 360.0) > MAX_SWEEP:
            continue
        on = pts[labels == lb]
        W = V[np.unique(F[s.faces])]
        flat = np.c_[W[:, u], W[:, v]]
        spread = float(np.min(np.linalg.norm(flat[:, None, :] - on[None, :, :], axis=2), axis=1).max())
        if spread >= floor:
            size = (f"R{s.params['radius']:.4g}" if s.kind == "cylinder" else
                    f"r{s.params['minor']:.4g}" if s.kind == "torus" else f"{s.params.get('half_angle', 0):.0f} deg")
            hits.append((spread, s, size, float(W[:, k].min()), float(W[:, k].max())))
    if not hits:
        return None
    hits.sort(key=lambda h: -h[0])
    ax = "XYZ"[k]
    listing = "; ".join(f"{s.kind} {s.name} ({size}) runs {ax}={lo:.4g} to {hi:.4g} and spreads "
                        f"{spread:.3g} mm across the outline" for spread, s, size, lo, hi in hits[:3])
    names = ",".join(s.name for _sp, s, *_ in hits[:3])
    return (f"WARNING: this cut at {ax}={level:.4f} crosses round blend faces: {listing}. The profile has "
            f"each wall where the cut meets the face, so a prism of it is that far proud or short at "
            f"other levels of the face. Take the profile clear of them (past the {ax} range above), or "
            f"`sketch --face F# --sharp`, and add or cut them after with `--op fillet --use {names}`")


def _lean_notes(mesh, seg, name, ax, leans, sharps, use_sharp, raw):
    """What --face says about faces leaning across its outline, and what
    --sharp did about them. The loops come outer first.

    A block extruded from its bottom face, whose outline turned-end cones had
    already cut back, carries that edge up its whole height and comes out short;
    nothing would say those faces were not walls, or that the block wanted
    building sharp and the cones cutting from it."""
    from shapely.geometry import Polygon

    from mesh2cad import blends

    total = sum(x[2] for ln in leans for x in ln)
    by = {}
    for ln in leans:
        for lab, leaning, length, ang in ln:
            if leaning and lab >= 0:
                b = by.setdefault(lab, [0.0, []])
                b[0] += length
                b[1].append(ang)
    leaning = sum(b[0] for b in by.values())
    if total <= 0 or leaning < FACE_LEAN_SHARE * total:
        return ([f"--sharp: nothing to carry -- every edge of {name}'s outline is on "
                 f"a wall"] if use_sharp else [])
    S = seg.surfaces
    order = sorted(by, key=lambda i: -by[i][0])
    listing = (", ".join(f"{S[i].kind} {S[i].name}" for i in order[:4])
               + (f" and {len(order) - 4} more" if len(order) > 4 else ""))
    angs = [float(np.median(by[i][1])) for i in order]
    # One name per surface: --op chamfer / fillet takes the faces on it too. A
    # hole's mouth chamfer (concave, on the hole's axis) is the hole's: `--op
    # hole` cuts it with the bore. Turned ends can share the hole's axis too,
    # but are convex: chamfers of the block.
    from mesh2cad.holes import stack_names
    owned = {i: c for c, g in stack_names(seg) for i in g if S[i].sense == "concave"}
    # Advice only for the faces whose stretch --sharp closes, between walls: a
    # face that leans with no wall beyond it is the part's own body -- a flat
    # milled on a shaft, whose "round" is the shaft.
    cut_by = set().union(*(sh[4] for sh in sharps))
    # A freeform piece is no blend the op can take, but the cone or round it is
    # a piece of is, so the advice names that surface.
    k = "XYZ".index(ax)
    nb = {}
    for j in seg.joints:
        nb.setdefault(j.a, set()).add(j.b)
        nb.setdefault(j.b, set()).add(j.a)

    def rounds_behind(i):
        out = []
        for j in sorted(nb.get(i, ())):
            t = S[j]
            if t.kind in ("cone", "torus") or (
                    t.kind == "cylinder" and abs(float(np.asarray(t.params["axis"])[k]))
                    < 1.0 - FACE_WALL_SIN ** 2):
                out.append(j)
        return out

    pool = []
    for i in order:
        if i in cut_by:
            pool += [i] if S[i].kind != "freeform" else [j for j in rounds_behind(i) if j not in pool]
    cmds, covered = {}, set()
    for i in pool:
        s = S[i]
        op = ("chamfer" if s.kind in ("cone", "plane") else
              "fillet" if s.kind in ("cylinder", "torus", "sphere") else None)
        if i in owned:
            if owned[i] not in cmds.setdefault("hole", []):
                cmds["hole"].append(owned[i])
            continue
        if op is None or i in covered:
            continue
        try:
            grp = blends.same_surface(mesh, seg, s)
        except Exception:                                          # noqa: BLE001
            grp = [s]
        covered.update(j for j, t in enumerate(S) if any(t is g for g in grp))
        cmds.setdefault(op, []).append(s.name)
    how = (" and ".join(f"`sketch --op {op} --use {','.join(ns)} --base PROG`"
                        for op, ns in cmds.items())
           or "`--op chamfer` / `--op fillet` on them")
    walls = sorted({w for sh in sharps for w in sh[1]}, key=lambda w: int(w[1:]))
    closed, left = sum(sh[2] for sh in sharps), sum(sh[3] for sh in sharps)
    spread = "" if max(angs) - min(angs) < 1 else f"-{max(angs):.0f}"
    head = (f"{100 * leaning / total:.0f}% of {name}'s outline ({leaning:.1f} of "
            f"{total:.1f} mm) is where {listing} meet it at {min(angs):.0f}{spread} "
            f"deg, not 90 as a wall does")
    if not closed:
        if use_sharp:
            return [f"--sharp: {head}, but the walls either side of them do not meet "
                    f"near them, so the outline is left as the face has it"]
        # No word of --sharp here: named in the note, the model would try it on
        # an outline it cannot change.
        return [f"note: {head}, and no walls meet beyond them: along them the part is "
                f"not a prism across {ax} (a flat on a round body, a bore or a round "
                f"crossing the face), and this outline is the part only at {name} itself"]
    if use_sharp:
        gain = [Polygon(sh[0]).area - Polygon(r).area for sh, r in zip(sharps, raw, strict=True)]
        d_area = gain[0] - sum(gain[1:])
        return [f"sharp: {name}'s outline with {', '.join(walls)} carried on to their "
                f"corners where {listing} cut it back ({d_area:+.1f} mm^2 of profile)"
                + (f"; {left} stretch(es) left as the face has them -- the walls either "
                   f"side do not meet near them" if left else "")
                + f". The block is sharp there: cut them from it after, {how}"]
    return [f"WARNING: {head}. They lean across {ax}: there the outline is the edge "
            f"they have already cut back, and a prism of it carries that edge through "
            f"every level, leaving out what they leave standing further from {name}. "
            f"If they are chamfers or rounds, build the block sharp and cut them after: "
            f"`--face {name} --sharp` carries {', '.join(walls)} on to their corners, "
            f"then {how}"
            + (f" ({left} stretch(es) join no walls and stay as the face has them)"
               if left else "")
            + ". If the whole side tapers, it is a loft, not a prism"]


def _face_profile(mesh, args, plane, name):
    """The profile a flat face IS, for --face NAME.

    Returns (polygons in (u, v), level, side, notes). A cut a little way inside
    the solid measures the prism beside the face; the face's own boundary is
    that prism's outline AT the face, which for a taper is the thing a cut can
    only approximate.

    Not every loop of the face is a hole in that prism. A loop whose walls go
    INTO the material is -- a bore, a pocket. A loop whose walls rise out of the
    face on the other side is the footprint of something standing on it, and
    the prism under the face runs straight through there; it is filled, and
    said so.
    """
    from shapely.geometry import Polygon

    from mesh2cad.segment import principal_of, segment_cached

    seg = segment_cached(mesh, cache_dir=_cache_dir(args))
    _tb.stage(f"reading flat face {name}'s outline and the cut beside it")
    surf = seg.by_name(name)
    spec = PLANES[plane]
    k = spec["axis"]
    ax = "XYZ"[k]
    if surf is None:
        raise SystemExit(f"--face {name}: no such surface. `faces` lists them; "
                         f"this part has F0..F{len(seg.surfaces) - 1}")
    if surf.kind != "plane" or principal_of(surf.params["normal"]) != ax:
        what = (surf.kind if surf.kind != "plane" else
                "a flat face not perpendicular to " + ax)
        raise SystemExit(f"--face {name}: it is {what}; --face takes a flat face "
                         f"perpendicular to {ax}, the normal of --plane {plane}")
    V = np.asarray(mesh.vertices, float)
    F = mesh.faces[surf.faces]
    level = float(V[np.unique(F), k].mean())
    facing = 1.0 if surf.params["normal"][k] > 0 else -1.0
    # Material is on the side the face does NOT face.
    side = "below" if facing > 0 else "above"

    # Directed boundary edges of the face, chained into loops.
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    key = np.sort(e, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    border = e[cnt[inv.ravel()] == 1]
    nxt = {int(a): int(b) for a, b in border}
    loops, seen = [], set()
    for a, _ in border:
        a = int(a)
        if a in seen:
            continue
        loop, cur = [a], nxt.get(a)
        seen.add(a)
        while cur is not None and cur != a and cur not in seen:
            loop.append(cur)
            seen.add(cur)
            cur = nxt.get(cur)
        if len(loop) >= 3:
            loops.append(loop)
    if not loops:
        raise SystemExit(f"--face {name}: it has no boundary to take a profile from")

    # Which side the neighbour across each loop lies on.
    mine = set(int(f) for f in surf.faces)
    by_edge = {}
    for f_i, (a, b) in zip(mesh.edges_face, mesh.edges_sorted, strict=True):
        by_edge.setdefault((int(a), int(b)), []).append(int(f_i))
    tc = np.asarray(mesh.triangles_center)

    def wall_side(loop):
        ds = []
        for a, b in zip(loop, loop[1:] + loop[:1], strict=True):
            for f_i in by_edge.get((min(a, b), max(a, b)), []):
                if f_i not in mine:
                    ds.append(tc[f_i, k] - level)
        return float(np.median(ds)) if ds else 0.0

    u, v = spec["u"], spec["v"]
    idx = next(i for i, s in enumerate(seg.surfaces) if s is surf)
    leans = [_outline_lean(mesh, seg, idx, lp, by_edge, mine, k) for lp in loops]
    sharps = [_sharp_loop(mesh, seg, idx, V[lp][:, [u, v]], ln, k, u, v)
              for lp, ln in zip(loops, leans, strict=True)]
    use_sharp = bool(getattr(args, "sharp", False))
    rings = [(Polygon(sh[0] if use_sharp else V[lp][:, [u, v]]), wall_side(lp), i)
             for i, (lp, sh) in enumerate(zip(loops, sharps, strict=True))]
    rings.sort(key=lambda r: -abs(r[0].area))
    outer, outer_side, outer_i = rings[0]
    material = -facing                       # sign of the material side
    holes, filled, in_profile = [], 0, [outer_i]
    for poly, ws, i in rings[1:]:
        if ws * material > 0:
            holes.append(poly.exterior.coords)
            in_profile.append(i)
        else:
            filled += 1
    notes = [f"profile of flat face {name}: {ax} = {level:.4f}, facing "
             f"{'+' if facing > 0 else '-'}{ax}, material {side} it"]
    notes += _lean_notes(mesh, seg, name, ax, [leans[i] for i in in_profile],
                         [sharps[i] for i in in_profile], use_sharp,
                         [V[loops[i]][:, [u, v]] for i in in_profile])
    if filled:
        notes.append(f"{filled} loop(s) of {name} are the footprint of something "
                     f"standing on it (its walls rise away from the material); "
                     f"filled, because the prism under the face runs through "
                     f"there")
    if outer_side * material < 0:
        notes.append(f"the walls round {name}'s outer edge rise AWAY from the "
                     f"material: it is the floor of a pocket or a step down, and "
                     f"this outline is the pocket's, not the section of the "
                     f"solid under it")
    poly = Polygon(outer.exterior.coords, holes)
    if not poly.is_valid:
        poly = poly.buffer(0)
    polys = list(getattr(poly, "geoms", [poly]))

    # The face's outline is the prism's section only if nothing passes through
    # the parts of it that were filled or lie outside it. A bore running up
    # through a boss that hangs under the face is not a loop of the face at
    # all, and the outline would build the block beside it solid. So compare
    # with a real cut a hair inside the material, and say so when they differ.
    span = float(np.ptp(V[:, k])) or 1.0
    eps = max(1e-4 * span, 10.0 * seg.tol)
    cut = section_at(mesh, plane, level + (-eps if side == "below" else eps))
    a_face = float(sum(p_.area for p_ in polys))
    # Only the cut INSIDE the outline says anything about the prism under the
    # face. Material the same plane meets elsewhere belongs to other features;
    # counted, it made every inclined face whose plane crosses the rest of the
    # part (a tab rising from a block, seen in its own frame) look wrong.
    from shapely.ops import unary_union
    a_cut = (float(unary_union(cut).intersection(unary_union(polys)).area)
             if cut else 0.0)
    # A sharp outline is wider than any cut by design: the check is the face's.
    sharpened = use_sharp and any(sharps[i][2] for i in in_profile)
    if use_sharp and not sharpened:
        # A --sharp that changed nothing would build again the cut the
        # model had just undone.
        said = next((n for n in notes if n.startswith("--sharp")), "--sharp: nothing to carry")
        raise SystemExit(f"{said} -- nothing built: without --sharp the outline is the same")
    if not sharpened and abs(a_face - a_cut) > 0.01 * max(a_face, a_cut, 1e-12):
        name_hint = next((m["name"] for m in _levels(args, mesh, plane)[0]
                          if abs(m["value"] - level) < 1e-9), f"{level:.4f}")
        notes.append(
            f"WARNING: the section a hair inside the material covers "
            f"{a_cut:.4f} of {name}'s outline {a_face:.4f} -- something passes "
            f"through where the face is not (a bore under a boss, a slot beside "
            f"a step), so this outline is NOT the prism's section. To cut "
            f"instead: --at {name_hint} --side {side}")
    return polys, level, side, notes


DIFF_LISTED = 8


def _ring_points(ring, per_arc=24):
    """A ring's outline as points, arcs sampled along their sweep -- a ring of
    one full-circle arc is a circle, not the one node it starts and ends on."""
    pts = []
    for sg in ring["segments"]:
        a = np.asarray(sg["from_xy"], float)
        if sg["kind"] == "arc":
            c = np.asarray(sg["centre"], float)
            t0 = np.arctan2(*(a - c)[::-1])
            sweep = np.radians(sg["sweep_deg"]) * (1.0 if sg["ccw"] else -1.0)
            t = t0 + sweep * np.linspace(0.0, 1.0, per_arc, endpoint=False)
            pts.extend(zip(c[0] + sg["radius"] * np.cos(t), c[1] + sg["radius"] * np.sin(t),
                           strict=True))
        else:
            pts.append(tuple(a))
    pts.append(pts[0])
    return pts


def _built_overlay(args, spec, ax, level, series, printed, d, mesh):
    """`section --built PROG [--fill]`: the program's section at the same cut
    as the target's, drawn over it, and the difference listed region by region.
    Returns the lines to print.

    A region thinner on average (2 x area / perimeter) than `residual`'s film
    limit is the two tessellations of one surface, counted and not listed --
    the same line `residual` draws in 3D. A floor tied to the section's size
    would hide thin skins on a thin wall, which can be the whole fault."""
    from shapely.geometry import LineString
    from shapely.ops import unary_union

    from mesh2cad import leftover
    from mesh2cad.build import load_program

    if getattr(args, "_frame_info", None) or getattr(args, "face", None):
        return ["note: --built is not drawn for a section taken on a face or in a "
                "--frame: the program is not turned into that frame here. Name the "
                "level instead (--offset) to compare"]
    try:
        built = load_program(args.built)
    except Exception as exc:                                       # noqa: BLE001
        return [f"note: --built {args.built} did not build ({type(exc).__name__}: "
                f"{exc}); nothing to compare"]
    lines = []
    for s, (side, rings, cut) in zip(series, printed, strict=True):
        tgt = unary_union(s["polys"]) if s["polys"] else None
        got_polys = section_at(built, args.plane, cut)
        got = unary_union(got_polys) if got_polys else None
        where = f"{ax}={cut:.4f}" + (f" ({side} the boundary)" if side else "")
        if tgt is None:
            lines.append(f"\nbuilt vs target at {where}: the target has no section here")
            continue
        missing = tgt.difference(got) if got is not None else tgt
        extra = got.difference(tgt) if got is not None else None
        thin = leftover.film_limit(float(np.linalg.norm(mesh.extents)), STL_TOL)
        regions, slivers = [], [0, 0.0]
        rlines = [(r.get("name", "?"), LineString(_ring_points(r)))
                  for r in rings if r.get("segments")]
        for kind, geom in (("missing", missing), ("extra", extra)):
            if geom is None:
                continue
            for g in sorted(getattr(geom, "geoms", [geom]), key=lambda g: -g.area):
                if g.is_empty or g.geom_type != "Polygon" or g.area <= 0:
                    continue
                if 2.0 * g.area / max(g.length, 1e-12) < thin:
                    slivers[0] += 1
                    slivers[1] += g.area
                    continue
                on = min(rlines, key=lambda rl: rl[1].distance(g))[0] if rlines else "?"
                regions.append((kind, g, on))
        named, counts = [], {"missing": 0, "extra": 0}
        for kind, g, on in regions:
            counts[kind] += 1
            named.append((f"{'M' if kind == 'missing' else 'X'}{counts[kind]}", kind, g, on))
        both_a = tgt.intersection(got).area if got is not None else 0.0
        lines.append(f"\nbuilt vs target at {where}: target {tgt.area:.4f} mm^2, built "
                     f"{0.0 if got is None else got.area:.4f}, both {both_a:.4f}; MISSING "
                     f"{missing.area:.4f} in {counts['missing']} region(s), EXTRA "
                     f"{0.0 if extra is None else extra.area:.4f} in {counts['extra']}")
        if not named:
            lines.append("  no region above the sliver floor either way: the two "
                         "sections agree here")
        for name, kind, g, on in named[:DIFF_LISTED]:
            c = g.centroid
            lines.append(f"  {name:<4} {kind:<7} {g.area:12.4f} mm^2  centre "
                         f"({c.x:9.4f}, {c.y:9.4f})  size {g.bounds[2] - g.bounds[0]:.4f} x "
                         f"{g.bounds[3] - g.bounds[1]:.4f}  on ring {on}")
        if len(named) > DIFF_LISTED:
            rest = named[DIFF_LISTED:]
            lines.append(f"  ... and {len(rest)} smaller, "
                         f"{sum(g.area for _, _, g, _ in rest):.4f} mm^2 in all")
        if slivers[0]:
            lines.append(f"  also {slivers[0]} sliver(s) thinner than {thin:.4g} mm, "
                         f"{slivers[1]:.4f} mm^2 in all -- the two tessellations, not "
                         f"features")
        png = d / (f"section_{args.plane}_{level:+.3f}_built{'_' + side if side else ''}"
                   f"{'_fill' if args.fill else ''}.png")
        section_diff_plot(s["polys"], got_polys, [(n, k, g, o) for n, k, g, o in named],
                          spec["u_name"], spec["v_name"],
                          f"section {args.plane} at {where}: target vs {Path(args.built).name}",
                          png, fill=args.fill)
        lines.insert(0, f"IMAGE: {png}")
    return lines


def _rings_from_polys(polys, tol, grid, ring_start=0, node_start=0):
    rings, ni, ri = [], node_start, ring_start
    for pi, poly in enumerate(polys):
        rs = decompose_polygon(poly, tol=tol, grid=grid, node_start=ni, ring_start=ri)
        for r in rs:
            r["poly_index"] = pi
        rings.extend(rs)
        ni += sum(len(r["nodes"]) for r in rs)
        ri += len(rs)
    return rings, ri, ni


def _load_built(path):
    """Execute a CadQuery program and tessellate it, or None."""
    import os
    import tempfile

    import cadquery as cq
    import trimesh
    ns = {"show_object": lambda o, *a, **k: None}
    exec(compile(Path(path).read_text(), str(path), "exec"), ns)
    obj = ns.get("result")
    shape = obj.toCompound() if hasattr(obj, "toCompound") else obj.val()
    f = tempfile.mktemp(suffix=".stl")
    cq.exporters.export(shape, f, tolerance=0.01, angularTolerance=0.1)
    m = trimesh.load(f, force="mesh")
    os.unlink(f)
    m.merge_vertices()
    return m


def cmd_overview(args):
    mesh, info = ingest(args.mesh)
    reg = info["regime"]
    d = out_dir(args)
    built = _load_built(args.built) if getattr(args, "built", None) else None

    # Two renders rather than one channel-composited sheet: the four isometric
    # views and the six orthographic ones, each drawn by its own module. Both
    # are Z-up, because the rest of the harness is -- `visualize_orthographic`
    # defaults to Y-up, which would title the +Y pane "FRONT" and the +Z pane
    # "RIGHT" on a part whose levels are all measured along Z.
    from isometric_views import visualize_isometric, write_png
    from orthographic_views import visualize_orthographic

    iso_png, ortho_png = d / "iso.png", d / "ortho.png"
    write_png(str(iso_png),
              visualize_isometric(mesh, mosaic=True, up="Z", max_side=args.tile))
    # window="bounds", not that module's [0, max] default. A part is not
    # obliged to sit in the positive octant and most do not: a tube spanning
    # X[-60.7, 92.3], Y and Z [-60, 60] rendered as a QUARTER of its own
    # annulus, silently, with the axes reading 0..60 as though that were the
    # part. The default says where a part sits relative to the origin, which is
    # worth something -- but not the part itself.
    ortho_img, _titles = visualize_orthographic(mesh, mosaic=True, up="Z",
                                                max_side=args.tile,
                                                window="bounds")
    write_png(str(ortho_png), ortho_img)

    lo, hi = info["bbox_min"], info["bbox_max"]
    print(f"IMAGE: {iso_png}")
    print(f"IMAGE: {ortho_png}")
    if built is not None:
        # Neither of these two renders takes a second solid, so --built cannot
        # be shown here at all. Saying so, rather than quietly drawing the
        # target and letting it be read as the comparison that was asked for.
        print(f"note: both renders show the TARGET only -- they draw one solid, "
              f"so the {len(built.faces)}-triangle reconstruction in "
              f"{args.built} is not in them. Use `progress --built` for the "
              f"target-against-reconstruction comparison.")
    print(f"bbox_min   ({lo[0]:.3f}, {lo[1]:.3f}, {lo[2]:.3f})")
    print(f"bbox_max   ({hi[0]:.3f}, {hi[1]:.3f}, {hi[2]:.3f})")
    print(f"extents    {info['extents'][0]:.3f} x {info['extents'][1]:.3f} x {info['extents'][2]:.3f} mm")
    print(f"volume     {info['volume']:.3f} mm^3")
    print(f"watertight {info['watertight']}")
    print(_bodies_line(mesh))
    # R2. `ingest` rotates the part so its own faces land on XY/YZ/XZ, which is
    # what makes an axis-aligned sketch possible at all. Every number below,
    # every level, every ring name is in the ROTATED frame, and a program written
    # against them is too -- so it is said here, or a scorer comparing the
    # program to the un-rotated STL records a correct reconstruction as wrong.
    al = info["alignment"]
    if al.get("applied"):
        print(f"aligned    the target was ROTATED {al['angle_deg']:.3f} deg to put "
              f"its faces on the axes;\n"
              f"           everything below is in that rotated frame, and so is "
              f"any program you write.\n"
              f"           The emitted program records the rotation as _FRAME.")
    if al.get("export_of"):
        print(f"frame      a residual export of {al['export_of']}: kept in that target's "
              f"frame, neither turned nor scaled again,\n"
              f"           so its coordinates are the target's and a sketch on it is "
              f"written there")
    if info.get("scale", 1.0) != 1.0:
        # R2: the same kind of move as the rotation, and said the same way.
        print(f"scaled     the target is {max(info['extents']) / info['scale']:.4g} "
              f"across, far below millimetre size, so it was SCALED by "
              f"{info['scale']:g};\n"
              f"           every number below is in that scale, and so is any "
              f"program you write.\n"
              f"           The emitted program records the scale in _FRAME, and "
              f"export / scoring undo it.")
    print(f"regime     {reg['kind']} (confidence {reg['confidence']:.2f}) -- one label "
          f"for the whole part; the revolve and prism lines below say where along "
          f"each axis it holds and where not")
    h = reg["evidence"].get("helix") or {}
    extra = (reg["evidence"].get("helices") or [])[1:]
    if h.get("is_helix"):
        # A3: the fit gets a NAME, and the name is what `sketch --along` takes.
        # It used to print `pitch=8.26`, which left the model nothing to do but
        # type 8.26 back -- the one habit the naming grammar exists to prevent.
        # The numbers stay on the line as evidence of what was measured and how
        # well; they are not what the model passes on.
        print(f"helix      H0  axis={h['axis']}  {h['turns']:g} turns  "
              f"{h['handedness']}-handed  (fit r2 {h['r2']:.4f})")
        if h.get("centre"):
            c, e = h["centre"], h["extent"]
            print(f"           axis through ({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}), from "
                  f"{h['axis']}={e[0]:.3f} to {e[1]:.3f}"
                  + ("; a thread on the part, fitted on that stretch where its cuts' "
                     "outline turns" if h.get("loops") == "thread" else
                     "; followed by the largest loop of each slice"
                     if h.get("loops") != "single" else ""))
        print("           sweep a profile along it with "
              "`sketch ... --op sweep --along H0 --use <ring>`")
        for k, g in enumerate(extra, 1):
            c, e = g["centre"], g["extent"]
            print(f"helix      H{k}  axis={g['axis']}  {g['turns']:g} turns  "
                  f"{g['handedness']}-handed  (fit r2 {g['r2']:.4f}), axis through "
                  f"({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}), from {g['axis']}={e[0]:.3f} to "
                  f"{e[1]:.3f} -- "
                  + ("another threaded stretch" if g.get("kind") == "thread"
                     else "another body of the mesh"))
        if h.get("closed_ends"):
            # Two pitches were measured, so two names exist; saying only "H0"
            # would hide that the ends are not the body.
            print(f"           H0@body and H0@ends differ -- the end turns are "
                  f"wound tighter than the body ({h['pitch_whole_part']:g} mm "
                  f"over the whole part against {h['pitch']:g} mm through the "
                  f"middle). H0 alone means H0@body.")
    # Rotational-symmetry probe. Printed always, not only when the classifier
    # commits to "revolve": a high score is the signal to section on a plane
    # CONTAINING the axis and revolve the profile, rather than hunt for prisms
    # along it -- and a revolve has no constant cross-section for `survey` to
    # find, so without this the sweep machinery leads nowhere.
    ev = reg["evidence"]
    if ev.get("revolve_axis"):
        ax = ev["revolve_axis"]
        print(f"revolve    axis={ax} score={ev['revolve_score']:.3f} over the whole "
              f"surface (the share that lands back on itself turned about {ax})")
        for line in _revolve_band_lines(info.get("revolve_bands") or [], ax):
            print(line)
    for line in _prism_lines(mesh):
        print(line)
    print("flat face clusters (normal, area fraction):")
    for c in info["flat_clusters"][:6]:
        print(f"    {tuple(c['normal'])}  {c['area_frac']:.3f}")


def _bodies_line(mesh):
    """How many bodies the target is, with each one's volume and genus: an
    assembly of several parts, or a worm beside its wheel, should be seen at
    once, not in a `--check` line halfway through the build."""
    if not mesh.is_watertight:
        n, how = _target_bodies(mesh)
        return f"bodies     {n} piece(s){how}"
    parts = sorted((b for b in mesh.split() if b.volume > 0), key=lambda b: -b.volume)
    shown = ", ".join(f"{b.volume:.1f} mm^3 genus {max(0, (2 - int(b.euler_number)) // 2)}"
                      for b in parts[:6])
    more = f", and {len(parts) - 6} more" if len(parts) > 6 else ""
    return (f"bodies     {len(parts)} ({shown}{more}); genus = the number of "
            f"through-holes or handles")


def _revolve_band_lines(bands, ax):
    """Where along its axis the part is round: one global score read as a
    verdict fails a turned sleeve with a saddle covering a tenth of its
    length."""
    if not bands:
        return []
    if len(bands) == 1:
        b = bands[0]
        return [f"           {'round' if b['round'] else 'not round'} along the whole "
                f"length about {ax} (score {b['score']:.2f})"]
    out = []
    for b in bands:
        where = f"{ax}={b['from']:.3f}..{b['to']:.3f} ({b['frac']:.0%} of the length)"
        if b["round"]:
            out.append(f"           round about {ax} over {where}, score {b['score']:.2f}")
        else:
            out.append(f"           not round over {where}, score {b['score']:.2f} -- "
                       f"something there is not turned: a cut across the axis, a "
                       f"flat, holes off the axis")
    return out


# Slices per axis for the prism lines, and the shortest run they list as a
# share of the axis: coarse on purpose, `spectrum` has the fine sweep.
PRISM_SLICES = 40
PRISM_MIN_FRAC = 0.1


def _prism_lines(mesh):
    """Where along each axis the cross-section stays the same: the stretches a
    single extrude would build, with the share of the length they cover --
    `regime prismatic 0.61` says neither where nor along what.

    Neighbouring runs with the same loops and holes, areas within a tenth and
    only a slice or two between them are one stretch whose area drifts: a
    drafted wall would otherwise come out as several "constant" runs a few
    percent apart, which reads as steps that are not there. They are printed as
    one tapering stretch, which is what `--taper` or a loft builds.
    """
    from mesh2cad.section import PLANES, find_plateaus, sweep
    out = []
    for plane in ("XY", "YZ", "XZ"):
        k = PLANES[plane]["axis"]
        lo, hi = (float(x) for x in mesh.bounds[:, k])
        if hi <= lo:
            continue
        try:
            plats = sorted(find_plateaus(sweep(mesh, plane, n=PRISM_SLICES)),
                           key=lambda p: p.start)
        except Exception:                                          # noqa: BLE001
            continue
        step = (hi - lo) / PRISM_SLICES
        groups = []
        for p in plats:
            g = groups[-1] if groups else None
            if (g and p.start - g[-1].end <= 2.5 * step
                    and (p.n_loops, p.n_holes) == (g[-1].n_loops, g[-1].n_holes)
                    and abs(p.area - g[-1].area) <= 0.1 * max(p.area, g[-1].area)):
                g.append(p)
            else:
                groups.append([p])
        groups = [g for g in groups if g[-1].end - g[0].start >= PRISM_MIN_FRAC * (hi - lo)]
        if not groups:
            continue
        cover = sum(g[-1].end - g[0].start for g in groups) / (hi - lo)
        ax = "XYZ"[k]
        spans = []
        for g in groups[:4]:
            what = f"{g[0].n_loops} loop(s), {g[0].n_holes} hole(s)"
            if len(g) == 1:
                spans.append(f"{ax}={g[0].start:.3f}..{g[0].end:.3f} the same ({what})")
            else:
                # A taper runs one way; a waist (a throated wheel) goes
                # down and back up and is only "varying".
                areas = np.array([p.area for p in g])
                steps = np.diff(areas)
                one_way = bool((steps >= 0).all() or (steps <= 0).all())
                if one_way and abs(areas[-1] - areas[0]) > 0.02 * areas.max():
                    shape = f"tapering, area {areas[0]:.1f} -> {areas[-1]:.1f}"
                else:
                    shape = f"varying, area {areas.min():.1f}..{areas.max():.1f}"
                spans.append(f"{ax}={g[0].start:.3f}..{g[-1].end:.3f} {shape} ({what})")
        more = f" and {len(groups) - 4} more" if len(groups) > 4 else ""
        out.append(f"prism      along {ax} ({plane} sections): {'; '.join(spans)}{more}"
                   f" -- {cover:.0%} of the length")
    return out


def cmd_survey(args):
    mesh, _ = ingest(args.mesh)
    s = plane_survey(mesh)
    print(f"{'plane':6s} {'normal':6s} {'sketch':8s} {'extent':>9s} {'prismatic':>10s} "
          f"{'depth':>9s} {'loops':>5s} {'area':>10s} {'offset':>9s}")
    for p in s["planes"]:
        if "error" in p:
            print(f"{p['plane']:6s} ERROR {p['error']}")
            continue
        print(f"{p['plane']:6s} {p['normal_axis']:6s} {'/'.join(p['sketch_axes']):8s} "
              f"{p['extent']:9.3f} {p['prismaticity']:9.0%} {p['best_depth']:9.3f} "
              f"{p['best_loops']:5d} {p['best_area']:10.2f} "
              f"{p['best_offset'] if p['best_offset'] is not None else float('nan'):9.3f}")
    print(f"\nrecommended: {s['recommended']}  ({s['reason']})")


def cmd_spectrum(args):
    mesh, _ = ingest(args.mesh)
    s = spectrum(mesh, args.plane, n=args.samples)
    print(f"plane {s['plane']}   normal axis {s['normal_axis']}   "
          f"sketch axes {s['sketch_axes'][0]}/{s['sketch_axes'][1]}   "
          f"offset range [{s['range'][0]:.3f}, {s['range'][1]:.3f}]")
    step = (s["range"][1] - s["range"][0]) / max(1, args.samples)
    print(f"sweep step {step:.4f} mm at --samples {args.samples}: a feature "
          f"thinner than about {2 * step:.4f} mm produces no level at all and "
          f"is invisible here. Raise --samples if the part may have one.")
    print(f"end inset: first/last sample are {s['end_inset']:.4f} mm inside the "
          f"extremes (avoids degenerate coincident-face cuts), so depths read "
          f"about {2 * s['end_inset']:.4f} mm short of the true extent")
    print("\ncandidate offsets (constant cross-section runs):")
    print(f"  {'id':4s} {'offset':>9s} {'span':>22s} {'depth':>9s} {'loops':>6s} "
          f"{'holes':>6s} {'area':>10s}")
    for c in s["candidates"]:
        print(f"  {c['id']:4s} {c['offset']:9.3f} "
              f"[{c['span'][0]:9.3f},{c['span'][1]:9.3f}] {c['depth']:9.3f} "
              f"{c['n_loops']:6d} {c['n_holes']:6d} {c['area']:10.2f}")
    trs = s["transitions"]
    shown = trs[:12]
    print(f"\ntransitions (cross-section changes) -- {len(trs)} total:")
    for t in shown:
        print(f"  {t['offset']:9.3f}  {t['kind']:8s}  {t['detail']}")
    if len(trs) > len(shown):
        # Say what is being withheld and where it is, rather than ending the
        # list and letting it read as complete.
        rest = trs[len(shown):]
        print(f"  ... {len(rest)} more, spanning {rest[0]['offset']:.3f} to "
              f"{rest[-1]['offset']:.3f} (the levels themselves are all in "
              f"`ortho`, which merges them)")

    # ASCII area profile: cheaper than an image and enough to see the shape.
    prof = s["profile"]
    if prof:
        amax = max(p["area"] for p in prof) or 1.0
        print(f"\narea profile along {s['normal_axis']} (each row = one sample, bar = area/{amax:.1f}):")
        step = max(1, len(prof) // 32)
        rows = prof[::step]
        turns = _turns_along(mesh, args.plane, [p["offset"] for p in rows])
        for p, t in zip(rows, turns, strict=True):
            bar = "#" * int(round(40 * p["area"] / amax))
            # Holes are printed alongside loops: "loops" counts connected
            # polygons, so an annulus is 1 loop and reads as solid without this.
            print(f"  {p['offset']:9.3f} |{bar:<40s}| {p['area']:9.2f} "
                  f"({p['n']} loops, {p['holes']} holes)"
                  + (f"  turn {t:+6.1f}" if t is not None else ""))
        twist = _twist_line(args, mesh, [p["offset"] for p in rows], turns)
        if twist:
            print("\n" + twist)




def _turns_along(mesh, plane, offsets):
    """The long axis's turn (deg, from the first row that has one) of the
    largest section at each offset, unwrapped; None where it has no long axis
    (a turn column beside `spectrum`'s area profile)."""
    out, prev, total = [], None, 0.0
    for at in offsets:
        cut = [p for p in section_at(mesh, plane, float(at)) if p.area > 0]
        big = max(cut, key=lambda p: p.area) if cut else None
        ang = _long_axis(big) if big is not None else None
        if ang is None:
            out.append(None)
            prev = None
            continue
        # a section that changed its size by more than a quarter is another
        # shape (a hole, an end), not the same one turned: count from 0 again
        if prev is not None and 0.8 <= big.area / prev[1] <= 1.25:
            total += ((ang - prev[0]) + 90.0) % 180.0 - 90.0
        else:
            total = 0.0
        prev = (ang, big.area)
        out.append(total)
    return out


def _twist_line(args, mesh, offsets, turns):
    """One line naming a band where the section keeps its shape while it turns,
    with the T levels that end the twist: a model can see "a twisted plate" and
    still loft from the part's ends through it."""
    from shapely import affinity
    k = [i for i, t in enumerate(turns) if t is not None]
    moving = [i for a, i in zip(k, k[1:], strict=False)
              if abs(turns[i] - turns[a]) >= 1.0]
    if not moving:
        return None
    # the longest run of rows turning one way
    runs, cur = [], [moving[0]]
    for i in moving[1:]:
        same = np.sign(turns[i] - turns[i - 1]) == np.sign(turns[cur[-1]] - turns[cur[-1] - 1])
        if i == cur[-1] + 1 and same:
            cur.append(i)
        else:
            runs.append(cur)
            cur = [i]
    runs.append(cur)
    run = max(runs, key=len)
    a, b = run[0] - 1, run[-1]
    total = turns[b] - turns[a]
    if abs(total) < 2 * TWIST_MIN:
        return None
    lo, hi = float(offsets[a]), float(offsets[b])
    try:
        marks, _ = _levels(args, mesh)
    except Exception:                                              # noqa: BLE001
        marks = []
    step = abs(offsets[1] - offsets[0]) if len(offsets) > 1 else 0.0

    def cut(at):
        got = [p for p in section_at(mesh, args.plane, float(at)) if p.area > 0]
        return max(got, key=lambda p: p.area) if got else None

    def name(at, toward):
        # a named level within a row of the band's end that cuts the part
        near = [m for m in marks if abs(m["value"] - at) <= step and m.get("name")
                and cut(m["value"]) is not None]
        if not near:
            return f"{at:.4f}", at
        m = min(near, key=lambda m: abs(m["value"] - toward))
        return m["name"], float(m["value"])
    (na, va), (nb, vb) = name(lo, lo - step), name(hi, hi + step)
    p0, p1 = cut(va), cut(vb)
    if p0 is None or p1 is None:
        return None
    t = np.radians(total)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    c0, c1 = np.array(p0.centroid.coords[0]), np.array(p1.centroid.coords[0])
    c = np.linalg.solve(np.eye(2) - rot, c1 - rot @ c0) if abs(total) > 1e-6 else c0
    turned = affinity.rotate(p0, total, origin=tuple(c))
    cover = turned.intersection(p1).area / max(turned.union(p1).area, 1e-12)
    if cover < TWIST_SAME:
        return None
    mid = [p for p in section_at(mesh, args.plane, 0.5 * (va + vb)) if p.area > 0]
    size = float(np.sqrt(max(mid, key=lambda p: p.area).area / p0.area)) if mid else 1.0
    _, rings, _, _ = _rings_at(mesh, args.plane, va, None, None)
    ring = next((r for r in rings if r.get("role") == "outer"
                 and _ring_polygon(r).buffer(0).contains(p0.representative_point())), None)
    ax = "XYZ"[PLANES[args.plane]["axis"]]
    return (f"twist: the section turns {total:+.1f} deg about {ax} between {na} "
            f"({ax}={va:.4f}) and {nb} ({ax}={vb:.4f}), keeping its shape (turned, the "
            f"near ring covers the far one {cover:.2f}; mid-way it is {size:.2f} of the ends' "
            f"size). Build it: sketch {args.mesh} --plane {args.plane} --at {na} --use "
            f"{ring['name'] if ring else 'R#'} --op loft --to {nb}")


def refine_named_level(mark, mesh, plane):
    """Pin down a mirror or turning-point level, at the moment it is used.

    Detection is nearly free; refining against real sections is not -- about a
    second per level. Paying that on every `section`, `ortho` and `sketch` call
    nearly doubled the cost of the level table whether or not anyone used these
    levels, which is A4's whole complaint. So the table lists them at sweep
    resolution, says so, and the work happens when the model actually names one.

    What it moves is printed, because the value used is then not the value that
    was shown, and a tool that quietly hands back a different number than it
    displayed is exactly R2's failure mode.
    """
    from mesh2cad import levels as lv
    from mesh2cad.section import sweep

    if mark.get("kind") not in ("mirror", "turning", "taper") or mesh is None:
        return mark["value"]
    if mark.get("refined"):
        return mark["value"]
    slices = sweep(mesh, plane, n=mark.get("samples", 600))
    scale = np.array([float(np.ptp([g(s) for s in slices])) for g in
                      (lambda s: s.area, lambda s: s.perimeter,
                       lambda s: s.extent_u, lambda s: s.extent_v)])
    step = float(slices[1].offset - slices[0].offset)
    detect = {"mirror": lv.mirror_levels, "turning": lv.extremum_levels,
              "taper": lv.taper_levels}[mark["kind"]]
    src = [c for c in detect(slices)
           if abs(c["offset"] - mark["value"]) <= 2.0 * step]
    if not src:
        return mark["value"]
    item = src[0]
    try:
        if mark["kind"] == "mirror":
            got = lv.refine_mirror(mesh, plane, item["offset"],
                                   item["half_width"], scale, step)
        elif mark["kind"] == "taper":
            got = lv.refine_taper(mesh, plane, item["offset"],
                                  item["half_width"], item["channel"])
        else:
            got = lv.refine_extremum(mesh, plane, item["offset"],
                                     item["bracket"], item["channel"],
                                     item["sense"])
    except CutError:
        got = None      # a probe cut didn't close: the note below says so
    if got is None:
        print(f"note: {mark['name']} could not be refined against real sections; "
              f"using the sampled value {mark['value']:.4f} (+/-{step:.4f})")
        return mark["value"]
    value, unc = got[0], got[1]
    print(f"note: {mark['name']} refined from the sampled {mark['value']:.4f} to "
          f"{value:.4f} (+/-{unc:.4f}) by measuring real sections")
    mark["value"], mark["uncertainty"], mark["refined"] = value, unc, True
    return value


def _pin_level(name, want, by_name, marks, what, value_of):
    """Resolve `NAME@VALUE` -- the name, checked against the place it meant.

    Level names are POSITIONAL: `B0`, `B1`, ... are numbered along the axis, so
    adding or removing a detector renumbers them. Turning on a new detector can
    replace plateau edges with other levels and shift every level above them
    down by one. A recipe saying `--to B1` then aims somewhere else, runs
    without error, and builds the wrong part.

    That is A3 failing in the one way A3 exists to prevent: the name was meant
    to be the thing that could not be silently wrong, and a raw number would
    have been safer, because a number at least stays put.

    So a name can carry the place it meant. `B1@10.0` resolves B1 and checks it
    is still at 10.0; if the table has been renumbered it looks for the level
    that IS at 10.0 and uses that, saying so. Only if neither holds does it
    refuse. The value is a checksum, not a measurement -- nothing is computed
    from it, and whatever precision the level has is still the level's.
    """
    tol_default = 1e-3
    m = by_name.get(name)
    if m is not None:
        # Against the TABLE value first, before refining. A pin identifies
        # which level is meant; refinement then improves where it is. Checking
        # the refined value against a pin copied from the table rejects the
        # very level it names -- M0 is listed at 25.3077 and refines to 25.1601,
        # so `M0@25.3077` refused itself.
        table_v = float(m["value"])
        unc = max(float(m.get("uncertainty", 0.0)), tol_default)
        if abs(table_v - want) <= max(unc, tol_default):
            return value_of(m)
        # A pin copied from a REFINED reading is equally valid, and costs a
        # refinement to check -- but only when the cheap check has failed.
        v = value_of(m)
        if abs(v - want) <= max(float(m.get("uncertainty", 0.0)), tol_default):
            return v
    # The name has moved. Is the PLACE still here under another name?
    here = [x for x in marks
            if x.get("name")
            and abs(float(x["value"]) - want)
            <= max(float(x.get("uncertainty", 0.0)), tol_default)]
    where_now = f" (it is at {m['value']:.4f})" if m is not None else " (it is gone)"
    if here:
        best = min(here, key=lambda x: abs(float(x["value"]) - want))
        print(f"note: {what} {name}@{want:g} -- {name} is no longer at "
              f"{want:g}{where_now}; the level there is now {best['name']} "
              f"({best['label']}), and that is what was used. Level names are "
              f"positional and this table has been renumbered.",
              file=sys.stderr)
        return value_of(best)
    raise SystemExit(
        f"{what}: {name}@{want:g} -- {name} is not at {want:g}{where_now}, "
        f"and no level is. Known: "
        + ", ".join(f"{n}@{by_name[n]['value']:g}"
                    for n in sorted(by_name, key=_level_key)))


def resolve_level(expr, marks, what="--at", mesh=None, plane=None):
    """A level name, `NAME@VALUE`, mid() of two, or a raw number.

    Names are the point: the model never handles a raw number. A bare number still works -- the model
    must be able to ask for a place no detector found -- but it is answered with
    the nearest named level, so the habit is visible rather than silent.

    `NAME@VALUE` pins a name to the place it meant, which is what makes a
    recorded recipe survive a change to the detectors. See `_pin_level`.
    """
    if expr is None:
        return None
    text = str(expr).strip()
    by_name = {m["name"]: m for m in marks if m.get("name")}

    def _value(m):
        return refine_named_level(m, mesh, plane)

    pinned = re.match(r"^([A-Za-z]+\d+)@(-?[\d.]+(?:[eE]-?\d+)?)$", text)
    if pinned:
        return _pin_level(pinned.group(1), float(pinned.group(2)),
                          by_name, marks, what, _value)

    mid = re.match(r"^mid\(\s*(\w+)\s*,\s*(\w+)\s*\)$", text)
    if mid:
        a, b = mid.group(1), mid.group(2)
        for n in (a, b):
            if n not in by_name:
                raise SystemExit(f"{what}: no level named {n!r}. Known: "
                                 + ", ".join(sorted(by_name, key=_level_key)))
        return (_value(by_name[a]) + _value(by_name[b])) / 2.0

    if text in by_name:
        _printed_elsewhere(text, by_name[text], plane, what)
        return _value(by_name[text])

    if re.match(r"^[A-Z]+\d+$", text):
        m, rec = readback.printed_level(text, plane)
        if m is not None:
            # A name read off `ortho --levels sweep` must stay good on the default
            # table of the next command: a name the tool printed for this plane
            # stays good there, and says where it is from.
            print(f"note: {what} {text} is not in this level table; the last level "
                  f"table printed for {plane} ({rec['source']} levels) named {text} = "
                  f"{m['value']:.4f} ({m['label']}), and that is used",
                  file=sys.stderr)
            return float(m["value"])

    if re.match(r"^[Ff]\d+$", text) and mesh is not None and plane:
        return _flat_face_level(text.upper(), marks, what, mesh, plane, _value)

    if re.match(r"^-?[\d.]+([eE]-?\d+)?$", text):
        v = float(text)
        if by_name:
            near = min(by_name.values(), key=lambda m: abs(m["value"] - v))
            if abs(near["value"] - v) < 1e-9:
                return v
            print(f"note: {what} {v:g} is a raw number; the nearest named level "
                  f"is {near['name']} at {near['value']:.4f} ({near['label']}). "
                  f"Naming it carries whatever precision the level has.",
                  file=sys.stderr)
        return v

    other = [(pl, m) for pl, m in readback.planes_naming(text) if pl != plane]
    raise SystemExit(f"{what}: {text!r} is not a level name, mid(A,B), or a "
                     f"number. Known levels: "
                     + ", ".join(f"{n}={by_name[n]['value']:.4f}"
                                 for n in sorted(by_name, key=_level_key))
                     + "".join(f". {text} was printed for plane {pl} ({m['value']:.4f}, "
                               f"along {'XYZ'[PLANES[pl]['axis']]}), not for {plane}: a "
                               f"level there is not a level here" for pl, m in other[:2]))


def _printed_elsewhere(name, mark, plane, what):
    """Warn when the last level table printed for this plane gave `name`
    another place: two tables (faces / sweep) number their levels apart."""
    m, rec = readback.printed_level(name, plane)
    if m is None or m.get("label") == mark.get("label") and abs(m["value"] - mark["value"]) < 1e-9:
        return
    tol = max(m.get("uncertainty", 0.0), float(mark.get("uncertainty", 0.0)), 1e-3)
    if abs(m["value"] - float(mark["value"])) > tol:
        print(f"WARNING: {what} {name} is {float(mark['value']):.4f} in this level table, "
              f"but the last one printed for {plane} ({rec['source']} levels) had {name} "
              f"at {m['value']:.4f} ({m['label']}); give the same --levels as that "
              f"table if that is the one meant", file=sys.stderr)


# A flat face this close to across the axis gives a level along it.
FACE_ACROSS_DEG = 1.0


def _flat_face_level(name, marks, what, mesh, plane, value_of):
    """Where a flat face F# lies across the plane's axis, for --at / --to /
    --offset F#: the level table's own level for that face when it has one.

    "--to the parallel face" is how a plate is described, so `--to F5` must
    work: faces have names too.
    """
    from mesh2cad.section import LEVELS_SEGMENT_BUDGET
    from mesh2cad.segment import principal_of, segment_cached

    k = PLANES[plane]["axis"]
    ax = "XYZ"[k]
    seg = segment_cached(mesh, budget=LEVELS_SEGMENT_BUDGET)
    s = seg.by_name(name)
    if s is None:
        raise SystemExit(f"{what} {name}: no such surface; `faces` lists "
                         f"F0..F{len(seg.surfaces) - 1}")
    off = None
    if s.kind == "plane":
        n = np.asarray(s.params["normal"], float)
        off = float(np.degrees(np.arccos(min(1.0, abs(n[k]) / (np.linalg.norm(n) or 1.0)))))
    # The far side of a sheet leg, in the frame its near side implies, is
    # a hair off across the axis (the two were never cut exactly parallel),
    # and a tolerance of 1e-6 rad would refuse it: "not across Y".
    if s.kind != "plane" or (principal_of(s.params["normal"]) != ax
                             and off > FACE_ACROSS_DEG):
        raise SystemExit(f"{what} {name}: {s.name} is "
                         + (f"a {s.kind}" if s.kind != "plane"
                            else f"a flat face {off:.2f} deg off across {ax}")
                         + f", so it gives no level along {ax}; name a flat face "
                           f"across {ax}, or a level")
    V = np.asarray(mesh.vertices, float)
    at = float(V[np.unique(mesh.faces[s.faces]), k].mean())
    if principal_of(s.params["normal"]) != ax:
        print(f"note: {what} {name} is {off:.3f} deg off across {ax}; its mean level "
              f"{ax} = {at:.4f} is used")
    near = [m for m in marks if m.get("name") and abs(m["value"] - at) < 1e-6]
    if near:
        print(f"note: {what} {name} is level {near[0]['name']} ({ax} = "
              f"{near[0]['value']:.4f})")
        return value_of(near[0])
    print(f"note: {what} {name}: flat face at {ax} = {at:.4f}")
    return at


def _level_key(name):
    return (name[0], int(name[1:]))


def measure_offset(mesh, plane, requested, marks, samples, side=None):
    """(sketch level, level actually cut). The sketch level is never moved.

    A sketch belongs on a real face, and a face is exactly where a section
    degenerates: the plane is coincident with the geometry it is meant to cut.
    So the sketch plane stays exactly where it was asked for -- it is what the
    generated CadQuery workplane must sit on -- and a little way into the solid
    is where the profile is actually measured.

    This used to also SNAP: a requested offset within 1% of the part's span of
    a known level was replaced by that level. That broke the rule that design decisions belong to the model -- the tool
    moving the plane the model named -- and it had already caused real damage,
    measuring a profile 1.8 mm from where it was asked for because the snap
    landed on a boundary contributed by geometry a `--split` had just discarded.
    It is gone. Levels are all named now, so a model that wants a level says its
    name and gets it exactly; a model that gives a number gets that number, with
    the nearest named level printed as evidence rather than substituted for it.

    What survives is the step INTO the solid, which is not a design decision at
    all: it is where a measurement can be taken without degenerating, and the
    caller never sees it as anything but provenance.
    """
    spec = PLANES[plane]
    lo, hi = (float(v) for v in mesh.bounds[:, spec["axis"]])
    span = hi - lo

    resolution = span / max(1, samples)
    # A bounding-box face is exact; a level carries whatever its own detection
    # earned. Both are needed below, because how far we must step into the
    # solid depends entirely on how well we know where its face is.
    targets = [{"value": lo, "what": "part face", "uncertainty": 0.0},
               {"value": hi, "what": "part face", "uncertainty": 0.0}]
    # Mirror, turning-point and taper levels are snap targets too. Otherwise a
    # section asked for exactly at a taper end is measured exactly ON it,
    # catching the geometry mid-transition -- fin corners half-rounded -- so the
    # two loft profiles have different primitive counts and CadQuery answers
    # `BRep_API: command not done`. A named level is a face like any other, and
    # a face is exactly where a section degenerates.
    _WHAT = {"boundary": "prism boundary", "mirror": "mirror plane",
             "turning": "turning point", "taper": "taper end"}
    targets += [{"value": m["value"], "what": _WHAT[m["kind"]],
                 "uncertainty": float(m.get("uncertainty", resolution))}
                for m in marks if m["kind"] in _WHAT]

    # Which known level this IS, if any -- for its uncertainty and its label,
    # never to move the plane onto it. "Is" means within a hair, not "nearby".
    level = float(requested)
    near = min(targets, key=lambda t: abs(t["value"] - level))
    identified = abs(near["value"] - level) <= max(
        float(near.get("uncertainty", 0.0)), 1e-9)
    if not identified:
        # A place the model asked for that is not one of the levels we know.
        # It gets measured a little inside, exactly like any other, but there is
        # no level metadata to draw a step from -- so use the sweep resolution.
        near = {"value": level, "what": None, "uncertainty": resolution}

    # The step has to clear the level's OWN uncertainty. Step less than that and
    # you land inside the transition the step exists to escape, measuring
    # geometry mid-change -- a straight edge fitted as a huge arc, a volume off
    # by orders of magnitude.
    #
    # But not much more: a bisected boundary is known to a thousandth of a mm,
    # and stepping far past it to be safe is no longer caution, it is just
    # measuring the wrong place -- which for a prism costs nothing and for a LOFT
    # is a systematic error, because the profile it measures is the tapered
    # section part of the way along instead of the one at the named level.
    step = max(0.02, span * 0.0005, 3.0 * float(near.get("uncertainty", resolution)))

    # Step towards whichever side has more room, so we land inside a prism
    # rather than straight back onto the neighbouring boundary. Outside the
    # part there is no room at all -- a missing neighbour means zero, not an
    # invented one, or an end face steps out into empty space and the section
    # comes back empty.
    # "Everything that is not this level", and nothing stronger. Dropping any
    # neighbour within snap_tol would make a close boundary invisible to the
    # room calculation, and the step would walk straight past it into the next
    # feature -- a section measuring a solid foot while claiming to measure the
    # tapered tube above it. A neighbour is a neighbour at any distance; only
    # the level itself is excluded.
    same = max(1e-6, 0.5 * float(near.get("uncertainty", 0.0)))
    others = sorted(t["value"] for t in targets if abs(t["value"] - level) > same)
    nearer_below = [v for v in others if v < level]
    nearer_above = [v for v in others if v > level]
    room_below = level - max(nearer_below) if nearer_below else 0.0
    room_above = min(nearer_above) - level if nearer_above else 0.0
    if max(room_below, room_above) <= 0:
        return level, level, near["what"]

    # A boundary has a prism on each side and they are different sections. The
    # default picks the roomier side, which is a guess about intent -- so the
    # caller can name the side, and either way we report which one was taken.
    if side == "above":
        direction, room = 1.0, room_above
    elif side == "below":
        direction, room = -1.0, room_below
    else:
        direction = 1.0 if room_above >= room_below else -1.0
        room = room_above if direction > 0 else room_below
    if room <= 0:
        return level, level, near["what"]
    reach = min(step, room * 0.25)
    cut = level + direction * reach
    what = near["what"]
    # Room ran out before the level's own error bar did. Say so: the profile is
    # being measured where the cross-section may still be changing, and every
    # number that follows inherits that. Better a warned measurement than a
    # confident wrong one.
    #
    # Measured against the LEVEL's uncertainty, not the sweep's resolution. A
    # bisected boundary is known to 0.001 mm and is deliberately stepped past
    # by only 0.02, so comparing that to a 0.08 mm sweep step fired this
    # warning on almost every call -- and a warning that is always on is a
    # warning nobody reads.
    doubt = float(near.get("uncertainty", resolution))
    if reach < 2.0 * doubt:
        what = (what or "unnamed level")
        what += (f" [WARNING: only {reach:.4f} mm of room, and this level is "
                 f"itself only known to +/-{doubt:.4f} mm -- the profile is "
                 f"measured inside that band and may be mid-change; raise "
                 f"--samples or pick a level with more room]")
    return level, cut, what


def resolve_between(args, what):
    """Offset midway between two levels, e.g. --between -11.063,11.063.

    Reading two numbers off the picture and averaging them by hand is exactly
    the transcription step the rest of this CLI exists to remove.
    """
    raw = getattr(args, "between", None)
    if not raw:
        return None
    try:
        a, b = (float(x) for x in raw.replace(" ", "").split(","))
    except ValueError:
        raise SystemExit("--between wants two numbers, e.g. --between -11.063,11.063")
    mid = (a + b) / 2.0
    print(f"{what}: midway between {a:.4f} and {b:.4f} -> {mid:.4f} "
          f"(gap {abs(b - a):.4f} mm)")
    return mid


def cmd_ortho(args):
    mesh, _ = ingest(args.mesh)
    mesh = _apply_frame(args, mesh)
    if not args.plane:
        raise SystemExit("m2c ortho: error: the following arguments are required: "
                         "--plane (or --frame F#, which sets it)")
    marks_for_offset, spec_levels = _levels(args, mesh)
    mid = resolve_between(args, "ortho")
    if mid is not None:
        args.offset = mid
    elif args.offset is not None:
        args.offset = resolve_level(args.offset, marks_for_offset, "--offset",
                                mesh, args.plane)
    d = out_dir(args)
    png = (d / f"ortho_{args.plane}_{args.offset:+.3f}.png" if args.offset is not None
           else d / f"ortho_{args.plane}.png")

    marks = marks_for_offset

    # Grey clay on a gridded pane, not the green-on-black depth pass. The colour
    # convention it used to follow -- target green, reconstruction red -- was
    # there so a part did not change meaning between two commands of one
    # session; `overview` now renders the same grey clay, so matching it is what
    # keeps that promise rather than breaking it. The cost is `--built`, which
    # these panes cannot draw at all.
    built = _load_built(args.built) if getattr(args, "built", None) else None

    # Boundary detection is a heuristic and on a curved part it fires
    # constantly. When that crowd genuinely obscures the geometry, the caller
    # can drop it -- but the part ends stay, because those are exact.
    shown = marks
    if getattr(args, "no_boundaries", False):
        shown = [m for m in marks if m.get("kind") == "end"]

    ortho_level_views(mesh, args.plane, args.offset, png, marks=shown,
                      tile=args.tile)

    spec = PLANES[args.plane]
    lo, hi = mesh.bounds[:, spec["axis"]]
    print(f"IMAGE: {png}")
    if built is not None:
        print(f"note: --built is not drawn here -- these panes render one solid. "
              f"Use `progress --built {args.built} --views ortho --plane "
              f"{args.plane}` for the comparison.")
    readback.save_levels(out_dir(args), args.mesh, args.plane,
                         getattr(args, "levels", None), marks, _frame_key(args))
    print(f"feature levels marked ({len(shown)} of {len(marks)}):")
    for m in shown:
        print(f"    {m.get('name', '??'):<4} {m['value']:9.3f}  {m['label']}")
    print("    name a level in --at / --to / --offset; mid(B2,B5) also works")
    for note in spec_levels.get("level_notes") or []:
        print(f"    note: {note}")
    _spec = PLANES[args.plane]
    _lo, _hi = (float(v) for v in mesh.bounds[:, _spec["axis"]])
    _step = (_hi - _lo) / max(1, args.samples)
    if any(m.get("source") != "faces" for m in marks):
        print(f"    sweep step {_step:.4f} mm: a feature thinner than about "
              f"{2 * _step:.4f} mm produces no sweep level and is invisible to it")
    else:
        print("    every level is a face: exact to the tolerance `faces` prints, "
              "whatever --samples is")
    if len(shown) < len(marks):
        hidden = [m for m in marks if m not in shown]
        print(f"    ({len(hidden)} prism boundaries hidden by --no-boundaries: "
              + ", ".join(f"{m['value']:.3f}" for m in hidden) + ")")
    ax = "XYZ"[spec["axis"]]
    if args.offset is None:
        print(f"section plane {args.plane}, normal axis {ax}; no offset chosen yet")
    else:
        print(f"section plane {args.plane}, normal axis {ax}, "
              f"level {ax} = {args.offset:.3f} mm")
    print(f"valid offset range: [{lo:.3f}, {hi:.3f}]")
    print(f"the four views are the directions in which this plane is edge-on, so a "
          f"section level appears as a horizontal line; screen-up is {ax}")


def _rings_at(mesh, plane, cut_at, tol, grid, ring_start=0, node_start=0):
    _tb.stage(f"cutting {plane} at {cut_at:.4f} and fitting its rings")
    polys = section_at(mesh, plane, cut_at)
    rings, ni, ri = [], node_start, ring_start
    for pi, poly in enumerate(polys):
        rs = decompose_polygon(poly, tol=tol, grid=grid, node_start=ni, ring_start=ri)
        for r in rs:
            r["poly_index"] = pi
        rings.extend(rs)
        ni += sum(len(r["nodes"]) for r in rs)
        ri += len(rs)
    return polys, rings, ri, ni


def cmd_section(args):
    _split_cuts(args)
    mesh, info = ingest(args.mesh)
    args._regime = info.get("regime")
    mesh = _apply_frame(args, mesh)
    if getattr(args, "meridian", None):
        return _section_meridian(args, mesh)
    if not args.plane:
        raise SystemExit("m2c section: error: the following arguments are "
                         "required: --plane (or --meridian; --frame or an "
                         "inclined --face sets it)")
    spec = PLANES[args.plane]
    ax = "XYZ"[spec["axis"]]

    marks, spec_out = _levels(args, mesh)
    face_notes = []
    if getattr(args, "sharp", False) and not getattr(args, "face", None):
        raise SystemExit("--sharp works on a face's outline: give --face")
    if getattr(args, "face", None):
        face_polys, face_level, _face_side, face_notes = _face_profile(
            mesh, args, args.plane, args.face)
        args.offset = face_level
        sketch_level = cut_at = face_level
        snapped_to = f"flat face {args.face}"
    else:
        face_polys = None
        mid = resolve_between(args, "section")
        if mid is not None:
            args.offset = mid
        else:
            if args.offset is None:
                raise SystemExit("section needs --offset (a level name) or "
                                 "--face (a flat face from `faces`)")
            args._offset_text = str(args.offset)
            args.offset = resolve_level(args.offset, marks, "--offset", mesh,
                                        args.plane)
        sketch_level, cut_at, snapped_to = measure_offset(
            mesh, args.plane, args.offset, marks, args.samples, side=args.side)
    if getattr(args, "project", None):
        if face_polys is not None:
            raise SystemExit("--project takes --offset, not --face: a face is "
                             "already its whole outline")
        if getattr(args, "built", None):
            raise SystemExit("--built is not drawn with --project: the projection "
                             "is not a cut, so the program has no section to set "
                             "against it. Compare a plain section instead")
        face_polys, _other, proj_note = _project_polys(args, mesh, float(args.offset), marks)
        sketch_level = cut_at = float(args.offset)
        snapped_to = None
        face_notes = [proj_note]

    # There used to be a block here undoing the snap for `--split`, because a
    # split selects its own geometry and being pulled onto a boundary
    # contributed by bodies the split discards made `section` and `sketch`
    # disagree about which profile they were looking at. Snapping is gone, so
    # there is nothing left to undo -- and deleting the workaround rather than
    # keeping it is the point: it was only ever compensating for the tool
    # moving a plane the model had named.

    # At a boundary the cross-section is discontinuous, so unless a side was
    # named, show BOTH prisms in one frame -- how the smaller sits inside the
    # larger is the thing being decided here, and it is invisible in either
    # section alone.
    both = snapped_to == "prism boundary" and args.side is None
    series, printed = [], []
    if both:
        _, cut_up, _ = measure_offset(mesh, args.plane, args.offset, marks,
                                   args.samples, side="above")
        _, cut_dn, _ = measure_offset(mesh, args.plane, args.offset, marks,
                                   args.samples, side="below")
        polys_u, rings_u, ri, ni = _rings_at(mesh, args.plane, cut_up, args.tol, args.grid)
        polys_d, rings_d, _, _ = _rings_at(mesh, args.plane, cut_dn, args.tol,
                                           args.grid, ring_start=ri, node_start=ni)
        series = [
            {"rings": rings_u, "polys": polys_u,
             "label": f"above  {ax} > {sketch_level:.3f}  (cut at {cut_up:.3f})",
             "colours": ("#1f5fa8", "#1f5fa8", "#12304f"),
             "style": "-", "lw": 3.0, "label_offset": (6, 6)},
            {"rings": rings_d, "polys": polys_d,
             "label": f"below  {ax} < {sketch_level:.3f}  (cut at {cut_dn:.3f})",
             # Violet, not red. Red already means "reconstruction" in the
             # projection views, and reusing it here for "below the boundary"
             # made one colour carry two unrelated meanings in the same session.
             "colours": ("#7a3b9e", "#7a3b9e", "#43205a"),
             "style": "--", "lw": 1.9, "label_offset": (6, -14)},
        ]
        printed = [("above", rings_u, cut_up), ("below", rings_d, cut_dn)]
        polys, rings = polys_u + polys_d, rings_u + rings_d
    elif face_polys is not None:
        polys = face_polys
        rings, _, _ = _rings_from_polys(polys, args.tol, args.grid)
        series = [{"rings": rings, "polys": polys,
                   "label": f"projection to {args.project}" if getattr(args, "project", None)
                   else f"face {args.face}",
                   "colours": ("#1f5fa8", "#b3341f", "#12304f")}]
        printed = [("", rings, cut_at)]
    else:
        polys, rings, _, _ = _rings_at(mesh, args.plane, cut_at, args.tol, args.grid)
        if not polys:
            print(f"empty section at {args.plane} offset {cut_at}")
            return
        series = [{"rings": rings, "polys": polys, "label": "section",
                   "colours": ("#1f5fa8", "#b3341f", "#12304f")}]
        printed = [("", rings, cut_at)]

    d = out_dir(args)

    if args.split:
        # A split is its own step: cut, then LOOK at what came out and let it be
        # named, before deciding what 3D operation it feeds. Going straight from
        # the cut to a revolve hides the profile that everything downstream
        # depends on.
        if getattr(args, "built", None):
            print("note: --built is not drawn with --split: the split shows the "
                  "kept profile; run the same section without --split to compare")
        if face_polys is None and abs(cut_at - float(args.offset)) > 1e-9:
            # The level as given, as `sketch --split` takes it: the pieces are
            # named here for sketch to use, and a step into the solid could give a
            # different set of pieces than sketch finds cutting on the level.
            exact = section_at(mesh, args.plane, float(args.offset))
            if exact:
                cut_at = sketch_level = float(args.offset)
                polys = exact
                rings, _, _ = _rings_from_polys(polys, args.tol, args.grid)
                print(f"note: --split cuts at {ax}={cut_at:.4f} as given (as sketch --split "
                      f"does), not a hair inside, so these names are the ones sketch takes")
            else:
                # On a face the exact cut is empty; both commands then cut where the
                # section was measured.
                polys = section_at(mesh, args.plane, cut_at)
                rings, _, _ = _rings_from_polys(polys, args.tol, args.grid)
                print(f"note: the section exactly at {ax}={float(args.offset):.4f} is empty "
                      f"(a face lies on it), so --split cuts where it was measured, "
                      f"{ax}={cut_at:.4f} (as sketch --split does)")
        only = _split_selection(args.use, rings, polys)
        cuts = [(resolve_split_expr(e, rings, mesh=mesh, args=args, spec=spec), k)
                for e, k in args._cuts]
        args.split = cuts[0][0]
        geoms, hrings, axis, desc, clip = split_profile(
            polys, spec, args.split, args.keep, args.tol, args.grid, only=only,
            more=cuts[1:])
        _source_names(hrings, rings, clip)

        png = d / f"section_{args.plane}_{sketch_level:+.3f}_split.png"
        title = (f"section {args.plane} at {ax}={sketch_level:.3f} mm, split at "
                 f"{args.split.strip()} keeping {desc}   "
                 f"(axes {spec['u_name']}/{spec['v_name']}, fit tol {args.tol} mm)")
        plot_series([{"rings": hrings, "polys": geoms, "label": f"kept {desc}",
                      "colours": ("#1f5fa8", "#b3341f", "#12304f"),
                      "style": "-", "lw": 2.4, "label_offset": (6, 6)}],
                    spec["u_name"], spec["v_name"], title, png)
        print(f"IMAGE: {png}")
        print(f"split at {args.split.strip()}, kept {desc}: "
              f"{len(geoms)} piece(s), {sum(r['n_primitives'] for r in hrings)} primitives")
        print(f"revolve axis if you revolve this: {axis[0]} -> {axis[1]}")
        for r in hrings:
            print("\n".join(_ring_lines(r, _detailed(args, r))))
        _detail_note(args, hrings)
        typed = getattr(args, "_offset_text", "")
        at = (typed if typed and not re.fullmatch(r"-?[\d.]+(?:e-?\d+)?", typed.strip())
              else _name_in_hint(sketch_level, spec["axis"], mesh, args))
        print(f"\nto build it:  sketch --plane {args.plane} --at {at} "
              + " ".join(f"--split \"{_split_hint(e, mesh, args)}\" --keep {k}"
                         for e, k in cuts)
              + " --op revolve")
        return

    png = d / f"section_{args.plane}_{sketch_level:+.3f}.png"
    title = (f"section {args.plane} at {ax}={sketch_level:.3f} mm   "
             f"(sketch axes {spec['u_name']}/{spec['v_name']}, "
             f"fit tol {rings[0].get('tol', args.tol) if rings else args.tol} mm)")
    plot_series(series, spec["u_name"], spec["v_name"], title, png)

    # The same clay panes `ortho` draws. This companion elevation answers the
    # same question -- where along the normal the cut is -- and was the last
    # place still drawing it green-on-black, so one session showed a level two
    # different ways depending on which command asked.
    ortho_png = d / f"ortho_{args.plane}_{sketch_level:+.3f}.png"
    ortho_level_views(mesh, args.plane, sketch_level, ortho_png, marks=marks,
                      tile=ORTHO_TILE)

    print(f"IMAGE: {png}")
    print(f"IMAGE: {ortho_png}")
    built_lines = (_built_overlay(args, spec, ax, sketch_level, series, printed, d, mesh)
                   if getattr(args, "built", None) else [])
    if getattr(args, "fill", False) and not getattr(args, "built", None):
        built_lines = ["note: --fill paints the difference to a --built program; "
                       "without --built there is nothing to fill"]
    print(f"section on {args.plane}; in-plane axes are ({spec['u_name']}, {spec['v_name']})")
    print(f"SKETCH PLANE: {ax} = {sketch_level:.4f}"
          + (f"   (snapped to the {snapped_to}; put the CadQuery workplane here)"
             if snapped_to else ""))
    if both:
        print("this level is a boundary: TWO different prisms meet here, and both "
              "sections are shown.")
    for note in face_notes:
        print(note)
    for line in built_lines:
        print(line)
    strips, centres = [], {}
    for line in _strip_lines(rings, polys if face_polys is None and not both else (), strips,
                             lambda n, c: _sheet_recipe(mesh, args.plane, sketch_level, n, c,
                                                        marks, args.mesh), centres):
        print(line)
    if face_polys is None and not both and strips:
        tubes, tube_recs = ([], []) if len(centres) < 2 else _tube_paths(
            args, mesh, polys, centres, len(rings), float(cut_at))
        for line in tubes:
            print(line)
        # --along W# is the path where it was printed, from any plane.
        readback.save_paths(out_dir(args), args.mesh,
                            [{"name": "W" + n[1:], "source": "section", "plane": args.plane,
                              "level": float(cut_at), "ring": n} for n in strips] + tube_recs,
                            _frame_key(args))
    if getattr(args, "law", None):
        other = resolve_level(args.law, marks, "--law", mesh, args.plane)
        chosen = ([r for r in rings if r.get("name") in
                   {n.strip() for n in args.use.split(",")}] if args.use else rings[:6])
        if not args.use and len(rings) > 6:
            print(f"note: --law fits the first 6 of {len(rings)} rings; name others "
                  f"with --use")
        for r in chosen:
            fits, zs, lw = _law_of(mesh, args.plane, r, sketch_level, other, ax)
            for line in _law_lines(r, fits, zs, lw, ax, sketch_level, other):
                print(line)
    if face_polys is None and not both:
        hint = _meridian_hint(args, mesh, args.plane, sketch_level)
        if hint:
            print(hint)
        lean = _section_lean(args, mesh, args.plane, cut_at)
        if lean:
            print(lean)

    all_arc_radii, halves = [], []
    readback.save_section(d, args.mesh, args.plane, sketch_level,
                          [r for _, rgroup, _ in printed for r in rgroup], _frame_key(args))
    for side, rgroup, cut in printed:
        if side:
            print(f"\n=== {side} the boundary ({ax} "
                  f"{'>' if side == 'above' else '<'} {sketch_level:.3f}, "
                  f"measured at {cut:.4f}) ===")
        for r in rgroup:
            print("\n".join(_ring_lines(r, _detailed(args, r))))
            for sg in r["segments"]:
                if sg["kind"] == "arc":
                    all_arc_radii.append(sg["radius"])
                halves.extend(half_arc_options(sg, spec["u_name"], spec["v_name"]))
    _detail_note(args, [r for _, rgroup, _ in printed for r in rgroup])

    if halves:
        print("\nhalf-arc options (a full circle split on a diameter; for revolves):")
        for h in halves:
            print(f"  {h['name']:<12s} / {h['alias']:<14s} keeps {h['keeps']:<20s} "
                  f"from ({h['from'][0]:8.3f},{h['from'][1]:8.3f}) "
                  f"to ({h['to'][0]:8.3f},{h['to'][1]:8.3f})")

    print("\nlevels on this axis (name them in --at / --to; append @value -- "
          "B1@10.0 -- to pin a name against renumbering):")
    readback.save_levels(d, args.mesh, args.plane, getattr(args, "levels", None), marks,
                         _frame_key(args))
    for m in marks:
        # Four places for a level known to better than a thousandth, three
        # otherwise. Printing 25.1601 as 25.160 throws away the digit that
        # bisection and mirror-refinement exist to earn.
        unc = float(m.get("uncertainty", 1.0))
        val = f"{m['value']:9.4f}" if unc < 5e-4 else f"{m['value']:9.3f} "
        print(f"    {m.get('name', '??'):<4} {val}  {m['label']}")
    for note in (spec_out.get("level_notes") or []):
        print(f"    note: {note}")

    # The levels partition the axis into BANDS, and a band is a feature. This
    # is already known -- it is one lookup per band in the sweep that was just
    # done -- and not printing it cost a connecting rod its reconstruction: the
    # model spent ten of fourteen calls sectioning at raw offsets to find out
    # what the bands were, then had two calls left to build five of them and
    # shipped 38% of the volume.
    #
    # Evidence, not a verdict: it says what the cross-section IS in each
    # band, not which to build, in what order, or with what operation. A band
    # is not always one feature and the table does not claim it is.
    bands = _bands(spec_out, marks)
    if bands:
        # With levels read off faces, a band is no longer constant by
        # construction: a cylinder lying across the axis changes the cut all
        # the way through the band it sits in, and says so here rather than
        # letting one sample stand for the band.
        print("\nbands between those levels (the section in the middle of each; "
              "one that changes inside is marked):")
        for lo, hi, loops, holes, area, amin, amax in bands:
            varies = amax - amin > 0.02 * max(amax, 1e-12)
            area_txt = (f"area {amin:.2f}..{amax:.2f} -- CHANGES inside the band"
                        if varies else f"area {area:9.2f}")
            print(f"    {lo:<4s} -> {hi:<4s}  {(hi_v(marks, hi) - hi_v(marks, lo)):8.3f} mm  "
                  f"{loops} loop(s), {holes} hole(s), {area_txt}")

    pats = detect_patterns(rings)
    if pats:
        print("\npatterns (equal-radius circular holes, grouped):")
        for g in pats:
            line = (f"  {g['count']}x circle R={g['hole_radius']:.4f} "
                    f"(d={g['hole_diameter']:.4f})")
            if g.get("pattern") == "circular":
                line += (f"  -> pitch circle R={g['pitch_circle_radius']:.4f} about "
                         f"({g['pitch_circle_centre'][0]:.4f}, "
                         f"{g['pitch_circle_centre'][1]:.4f}), start "
                         f"{g['start_angle_deg']:.3f}deg, spacing {g['spacing_deg']:.3f}deg")
            print(line + f"\n       members: {', '.join(g['names'])}")

    arrays = detect_arrays(rings)
    if arrays:
        spec = PLANES[args.plane]
        sk = (spec["u_name"], spec["v_name"])
        print("\narrays (congruent loops at constant pitch):")
        for i, g in enumerate(arrays):
            name = f"P{i}"
            line = (f"  {name}  {g['count']}x loop {g['size'][0]:.4f} x "
                    f"{g['size'][1]:.4f}")
            if g["even"]:
                line += (f"  -> pitch {g['pitch']:.4f} mm along "
                         f"{sk[g['axis']]}, from {g['from']:.4f} to {g['to']:.4f}")
            else:
                line += "  -> collinear but unevenly spaced"
            print(line)
            print(f"       members: {', '.join(g['names'])}")
            if g["odd_members"]:
                print(f"       NOT UNIFORM: {', '.join(g['odd_members'])} differ "
                      f"in shape from the rest -- the array has exceptions, and "
                      f"building it as one repeat will lose them")
            print(f"       name {name} in --use to take the whole array")

    tols = sorted({r.get("tol") for r in rings if r.get("tol")})
    print("\nfit tolerance: " + (", ".join(f"{t:g}" for t in tols) if tols
          else str(args.tol)) + " mm (derived from the mesh sampling"
          + (" unless --tol was given)" if args.tol is None else ", overridden by --tol)"))
    if all_arc_radii:
        print(f"smallest arc radius found: {min(all_arc_radii):.4f} mm "
              f"({len(all_arc_radii)} arcs total)")
    else:
        print("no arcs anywhere in this section: every segment fits a straight line "
              f"to within {args.tol} mm, so there are no fillets above that tolerance.")


def _section_at_level(mesh, plane, at, samples, tol, grid, side=None,
                      source=None, cache_dir=None, face=None):
    """Everything `sketch` needs about one level: rings, halves, and the levels.

    At a boundary with no side named, BOTH prisms are decomposed with continuous
    ring numbering -- exactly as `section` prints them. Otherwise the model reads
    `A2_0` off the section table and finds it missing here, because the default
    side happened to be the other one.
    """
    marks, _ = feature_levels(mesh, plane, samples, source=source,
                              cache_dir=cache_dir)
    spec = PLANES[plane]
    if face is not None:
        # The profile is the face itself, measured ON it rather than a little
        # inside the solid, and the solid is on the side it does not face.
        polys, level, fside, _notes = face
        rings, _, _ = _rings_from_polys(polys, tol, grid)
        for r in rings:
            r["side"] = fside
        halves = {}
        for r in rings:
            for s_ in r["segments"]:
                for h in half_arc_options(s_, spec["u_name"], spec["v_name"]):
                    h["side"] = r["side"]
                    halves[h["name"]] = h
                    halves[h["alias"]] = h
        return rings, halves, marks, level, level, "flat face"
    sketch_level, cut_at, snapped = measure_offset(mesh, plane, at, marks, samples,
                                                side=side)

    both = snapped == "prism boundary" and side is None
    rings = []
    if both:
        _, cut_up, _ = measure_offset(mesh, plane, at, marks, samples, side="above")
        _, cut_dn, _ = measure_offset(mesh, plane, at, marks, samples, side="below")
        _, ru, ri, ni = _rings_at(mesh, plane, cut_up, tol, grid)
        _, rd, _, _ = _rings_at(mesh, plane, cut_dn, tol, grid,
                                ring_start=ri, node_start=ni)
        for r in ru:
            r["side"] = "above"
        for r in rd:
            r["side"] = "below"
        rings = ru + rd
    else:
        _, rings, _, _ = _rings_at(mesh, plane, cut_at, tol, grid)
        for r in rings:
            r["side"] = side or ("above" if cut_at > sketch_level else "below")

    halves = {}
    for r in rings:
        for s_ in r["segments"]:
            for h in half_arc_options(s_, spec["u_name"], spec["v_name"]):
                h["side"] = r["side"]
                halves[h["name"]] = h
                halves[h["alias"]] = h
    return rings, halves, marks, sketch_level, cut_at, snapped


def _array_groups(rings):
    """P0, P1 ... -> the rings of each detected array, in the printed order."""
    groups = {}
    for i, g in enumerate(detect_arrays(rings)):
        wanted = set(g["names"])
        members = [r for r in rings
                   if any(s_["name"] in wanted for s_ in r["segments"])]
        groups[f"P{i}"] = members
    return groups


def _resolve(name, rings, halves, arrays=None):
    """Map a name the model typed onto a ring, a half arc, or a whole array."""
    name = name.strip()
    if arrays and name in arrays:
        # The point of the array name: seven fins cost one word, and the model
        # never has to transcribe six more that it could get wrong.
        return ("array", arrays[name])
    for r in rings:
        if r.get("name") == name:
            return ("ring", r)
    if name in halves:
        return ("half", halves[name])
    for r in rings:
        for s_ in r["segments"]:
            if s_["name"] == name:
                return ("ring", r)
    # Unique prefix, so "A0" resolves when there is exactly one A0_*.
    hits = [(r, s_) for r in rings for s_ in r["segments"]
            if s_["name"].startswith(name)]
    if len(hits) == 1:
        return ("ring", hits[0][0])
    if len(hits) > 1:
        raise SystemExit(f"'{name}' is ambiguous: matches "
                         + ", ".join(sorted(s_['name'] for _, s_ in hits)))
    raise SystemExit(f"'{name}' matches nothing in this section. The rings here: "
                     + readback.listing(rings))


def _frame_key(args):
    """The --frame (or face-implied frame) a command measured in, or ""."""
    fi = getattr(args, "_frame_info", None)
    return fi[0] if fi else ""


def _resolve_named(name, rings, halves, arrays, args, ax, level):
    """`_resolve`, knowing which ring the model last read under this name on
    this plane.

    A name that is not in this section resolves to the ring here that overlaps
    the ring it was read as -- the same ring at another level -- and says so.
    A name that is here, but was another ring when it was read, is warned
    about, so a pin is never built for the body the model meant."""
    rec = readback.last_section(out_dir(args), args.mesh, args.plane, _frame_key(args))
    seen = rec if rec and abs(float(rec["level"]) - level) > 1e-6 else None
    where = f"your last section on {args.plane} ({ax} = {seen['level']:.4f})" if seen else ""
    try:
        kind, obj = _resolve(name, rings, halves, arrays)
    except SystemExit as exc:
        was = readback.ring_of(seen, name.strip())
        if was is None or "matches nothing" not in str(exc):
            raise
        got, score = readback.best_match(was, rings)
        if got is None or score <= readback.SAME_RING:
            near = (f"; the nearest of its role, {readback.describe(got)}, overlaps it "
                    f"{score:.2f}" if got is not None else "")
            raise SystemExit(f"{exc}. {name.strip()} was {readback.describe(was)} in "
                             f"{where}, and no ring here overlaps it by more than half"
                             f"{near}: name one of the rings here") from None
        print(f"note: {name.strip()} is not in the section at {ax} = {level:.4f}; in "
              f"{where} it was {readback.describe(was)}, which is {got['name']} here "
              f"({readback.describe(got)}, overlap {score:.2f}) -- {got['name']} is used")
        return "ring", got
    if seen is not None and kind == "ring":
        was = readback.ring_of(seen, name.strip())
        if was is not None and readback.overlap(was, obj) <= readback.SAME_RING:
            got, score = readback.best_match(was, rings)
            alt = (f"; the ring here that overlaps that one is {got['name']} "
                   f"(overlap {score:.2f}) -- name {got['name']} if that is the one meant"
                   if got is not None and score > readback.SAME_RING and got is not obj
                   else "")
            print(f"WARNING: {name.strip()} here ({ax} = {level:.4f}) is in "
                  f"{readback.describe(obj)}; in {where} it was in "
                  f"{readback.describe(was)}{alt}")
    return kind, obj


def _node_xy(node, rings):
    """The in-plane coordinates of a named node, or a loud failure."""
    for r in rings:
        for s_ in r["segments"]:
            if s_.get("from") == node:
                return s_["from_xy"]
    known = sorted({s_.get("from") for r in rings
                    for s_ in r["segments"] if s_.get("from")},
                   key=lambda n: int(n[1:]))
    raise SystemExit(f"--split: no node named {node!r} in this section. "
                     f"Known nodes: {', '.join(known)}")


def _zero(v):
    """`v` with a -0 that only rounding made written as 0."""
    v = round(float(v), 6)
    return 0.0 if v == 0 else v


def _split_hint(split, mesh, args):
    """"X=B1" for a split line at a level, so the hint can be run as printed."""
    m = re.fullmatch(r"\s*([XYZ])\s*=\s*(-?[\d.]+(?:e-?\d+)?)\s*", split)
    if not m:
        return split.strip()
    k = "XYZ".index(m.group(1))
    return f"{m.group(1)}={_name_in_hint(float(m.group(2)), k, mesh, args)}"


def _named_plane(name, mesh, args):
    """(world axis, coordinate, label) of a plane named S# (a mirror plane
    `symmetry` names) or F# (a flat face `faces` names, square to X, Y or Z),
    or None when `name` is neither."""
    name = name.strip()
    if re.fullmatch(r"S\d+", name):
        from mesh2cad.contrib.instruments.symmetry import candidates
        named, _near, _tol, _n = candidates(mesh)
        k = int(name[1:])
        if k >= len(named):
            raise SystemExit(f"{name}: `symmetry` names S0..S{len(named) - 1} here"
                             if named else f"{name}: `symmetry` finds nothing here")
        c = named[k]
        if c["kind"] != "mirror plane":
            raise SystemExit(f"{name} is a {c['kind']} ({c['about']}), not a plane")
        return c["axis"], c["coord"], f"{name} ({c['about']})"
    if re.fullmatch(r"F\d+", name):
        seg = _faces_of(mesh, args)
        f = seg.by_name(name)
        if f is None:
            raise SystemExit(f"{name}: no such face; `faces` names F0..F{len(seg.surfaces) - 1}")
        if f.kind != "plane":
            return None
        n = np.asarray(f.params["normal"], float)
        k = int(np.argmax(np.abs(n)))
        if abs(n[k]) < 0.999:
            raise SystemExit(f"{name} is a plane at a slant, not square to X, Y or Z")
        coord = float(f.params["offset"]) * float(np.sign(n[k]))
        return k, coord, f"{name} ({'XYZ'[k]} = {coord:.4f})"
    return None


def _named_axis(name, mesh, args):
    """(world axis k, the point it passes through) of an axis named S# (a
    rotation axis `symmetry` names), F# (a round face's axis, along X, Y or
    Z) or H# (a helix `overview` fitted), or None when `name` is none of them."""
    name = name.strip()
    if re.fullmatch(r"H\d+", name):
        h = _helix_fit(mesh, name)
        if not h.get("is_helix") or not h.get("centre"):
            raise SystemExit(f"{name}: no helix of that name here"
                             + (f" ({h.get('reason')})" if h.get("reason") else ""))
        return "XYZ".index(h["axis"]), np.asarray(h["centre"], float)
    if re.fullmatch(r"S\d+", name):
        from mesh2cad.contrib.instruments.symmetry import candidates
        named, _near, _tol, _n = candidates(mesh)
        k = int(name[1:])
        if k >= len(named):
            raise SystemExit(f"{name}: `symmetry` names S0..S{len(named) - 1} here"
                             if named else f"{name}: `symmetry` finds nothing here")
        c = named[k]
        if c["kind"] != "rotation axis" or "axis" not in c:
            raise SystemExit(f"{name} is a {c['kind']} ({c['about']}), not an axis")
        return c["axis"], np.asarray(c["centre"], float)
    if re.fullmatch(r"F\d+", name):
        seg = _faces_of(mesh, args)
        f = seg.by_name(name)
        if f is None:
            raise SystemExit(f"{name}: no such face; `faces` names F0..F{len(seg.surfaces) - 1}")
        if "axis" not in (f.params or {}):
            raise SystemExit(f"{name} is a {f.kind}: it has no axis")
        a = np.asarray(f.params["axis"], float)
        k = int(np.argmax(np.abs(a)))
        if abs(a[k]) / np.linalg.norm(a) < 0.999:
            raise SystemExit(f"{name}'s axis is not along X, Y or Z")
        return k, np.asarray(f.params["point"], float)
    return None


def _faces_of(mesh, args):
    """The target's segmentation, as `faces` and the level table cache it."""
    from mesh2cad.section import LEVELS_SEGMENT_BUDGET
    from mesh2cad.segment import segment_cached
    return segment_cached(mesh, cache_dir=_cache_dir(args), budget=LEVELS_SEGMENT_BUDGET)


def _level_value(text, k, mesh, args, what):
    """A coordinate along world axis k written as a level -- E0, B3, M0,
    mid(B1,B2) -- or a number, as --at takes it."""
    plane = next(p for p, sp in PLANES.items() if sp["axis"] == k)
    marks, _ = feature_levels(mesh, plane, getattr(args, "samples", 900) or 900,
                              cache_dir=_cache_dir(args))
    return resolve_level(text, marks, what, mesh, plane)


def _name_in_hint(value, k, mesh, args):
    """`B1` when a level along axis k sits at `value`, else the number: the
    commands the tool prints for the model to run carry names too."""
    try:
        plane = next(p for p, sp in PLANES.items() if sp["axis"] == k)
        marks, _ = feature_levels(mesh, plane, getattr(args, "samples", 900) or 900,
                                  cache_dir=_cache_dir(args))
    except Exception:                                              # noqa: BLE001
        return f"{_zero(value):g}"
    hit = [m for m in marks if m.get("name") and abs(float(m["value"]) - value) < 1e-6]
    return hit[0]["name"] if hit else f"{_zero(value):g}"


def resolve_split_expr(expr, rings, mesh=None, args=None, spec=None):
    """Let the split line be named: "X=N2" means "the X of node N2".

    39.8385 is the X of a node the section already printed and named. Making the
    model retype it is the one failure mode this whole design exists to remove --
    a mistyped digit here silently moves the cut line and every dimension
    downstream of it. Naming the node cannot go wrong: it resolves or it fails
    loudly.
    """
    # A plane named whole (S0, F12) -- the line where it crosses this plane --
    # or an axis lying in it (H0, a round face F#): the line it runs along.
    if mesh is not None and spec is not None and re.fullmatch(r"\s*[SFH]\d+\s*", expr):
        got = None if expr.strip()[0] == "H" else _named_plane(expr, mesh, args)
        if got is None:
            k, pt = _named_axis(expr, mesh, args)
            if k == spec["axis"]:
                raise SystemExit(f"--split {expr.strip()}: that axis is square to the "
                                 f"sketch plane, so it is no line in it")
            o = next(i for i in range(3) if i not in (k, spec["axis"]))
            return f"{'XYZ'[o]}={_zero(pt[o]):.6f}"
        k, coord, _label = got
        if k == spec["axis"]:
            raise SystemExit(f"--split {expr.strip()}: that plane is the sketch plane "
                             f"itself, not a line across it")
        return f"{'XYZ'[k]}={_zero(coord):.6f}"
    # "X=M0", "X=mid(B1,B2)": a level along that world axis, the way
    # --at names one. Node names (N2) and mid(N3,N9) keep their meaning.
    lv = re.match(r"\s*([XYZxyz])\s*=\s*(.+?)\s*$", expr)
    if (lv and mesh is not None and not re.fullmatch(r"-?[\d.]+(?:e-?\d+)?", lv.group(2))
            and not re.fullmatch(r"N\d+|mid\(\s*N\d+\s*,\s*N\d+\s*\)", lv.group(2))):
        k = "XYZ".index(lv.group(1).upper())
        return f"{'XYZ'[k]}={_zero(_level_value(lv.group(2), k, mesh, args, '--split')):.6f}"
    # "Y=mid(N3,N9)": the axis of a cylinder is halfway between its two walls,
    # and both walls are already named. Averaging them by hand is the same
    # retyping risk as any other coordinate, one step further removed.
    mid = re.match(r"\s*([A-Za-z])\s*=\s*mid\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*$",
                   expr)
    if mid:
        ax_name = mid.group(1).upper()
        idx = 0 if ax_name == "X" else 1
        vals = [_node_xy(n, rings)[idx] for n in (mid.group(2), mid.group(3))]
        return f"{ax_name}={sum(vals) / 2.0:.6f}"

    m = re.match(r"\s*([A-Za-z])\s*=\s*([A-Za-z]\w*)\s*$", expr)
    if not m:
        return expr
    ax_name, node = m.group(1).upper(), m.group(2)
    pt = _node_xy(node, rings)
    idx = 0 if ax_name == "X" else 1
    # The node's coordinates are in the section's own in-plane axes, in the
    # order the section printed them, so index by position not by world axis.
    return f"{ax_name}={pt[idx]:.6f}"


def _split_selection(use, rings, polys):
    """Which of `polys` `--split --use NAMES` cuts, or None for all of them.

    The names resolve exactly as plain --use resolves them -- a ring (R1), a
    primitive (L1_0), a unique prefix (A0) or an array (P0) -- and each selects
    the polygon its ring lies on. Matching primitive names only would make the
    ring name that `section` prints and every other --use accepts select
    nothing. A name that matches nothing still stops the command, and says
    which name.

    The polygon is found by where the ring IS, not by its index: sketch names
    rings at the measured (possibly two-sided) level but cuts the section taken
    at --at itself, so an index into one list need not point into the other.
    """
    if not use or not polys:
        return None
    from shapely.geometry import Point

    arrays = _array_groups(rings)
    picked = []
    for n in (n for n in use.split(",") if n.strip()):
        kind, obj = _resolve(n, rings, {}, arrays)
        picked.extend(obj if kind == "array" else [obj])
    only = set()
    for r in picked:
        pts = [Point(s_["from_xy"]) for s_ in r["segments"]]
        only.add(min(range(len(polys)),
                     key=lambda i: sum(polys[i].boundary.distance(q) for q in pts)))
    return only


SPLIT_SLIVER_MM = 1e-3


SIDED = re.compile(r"^\s*([A-Za-z])\s*(>=|<=)\s*(.+?)\s*$")


def _split_cuts(args):
    """`args._cuts`, the [(expr, keep)] of every --split, and
    args.split / args.keep the first of them, as the rest of the code reads
    them. One --keep per --split, or one for a single cut; a cut written with
    its side ("X>=0,Z<=31.75") carries its own."""
    exprs = args.split if isinstance(args.split, list) else ([args.split] if args.split else [])
    keeps = args.keep if isinstance(args.keep, list) else ([args.keep] if args.keep else [])
    cuts, plain = [], 0
    for e in exprs:
        parts = [q for q in re.split(r",(?![^()]*\))", e) if q.strip()]
        sided = [SIDED.match(q) for q in parts]
        if all(sided):
            cuts.extend((f"{m.group(1)}={m.group(3)}", "+" if m.group(2) == ">=" else "-")
                        for m in sided)
        elif any(sided):
            raise SystemExit(f'--split {e!r}: write every cut in it with its side '
                             f'("X>=0,Z<=31.75"), or one cut "X=0" with --keep')
        else:
            cuts.append((e, None))
            plain += 1
    if keeps and not plain:
        raise SystemExit("--keep: the --split already says which side of each cut to keep")
    if keeps and len(keeps) != plain:
        raise SystemExit(f"--keep: {len(keeps)} given for {plain} --split; give one --keep "
                         f"per --split, in the same order")
    it = iter(keeps or ["+"] * plain)
    args._cuts = [(e, k if k is not None else next(it)) for e, k in cuts]
    args.split, args.keep = (args._cuts[0] if args._cuts else (None, "+"))
    return args._cuts


def _kept_nothing(expr, keep, chosen, which, val, spec, pieces, selected):
    """Why a --split kept nothing, said -- "kept nothing" alone costs the
    model calls to work out."""
    from shapely.ops import unary_union
    if pieces:
        return (f'--split "{expr}" --keep {keep} kept only {len(pieces)} sliver(s) '
                f"thinner than {SPLIT_SLIVER_MM} mm: there is no profile on that side of "
                f"the cut")
    name = spec["u_name"] if which == 0 else spec["v_name"]
    b = unary_union(chosen).bounds
    lo, hi = (b[0], b[2]) if which == 0 else (b[1], b[3])
    other = "-" if keep == "+" else "+"
    what = "the rings --use names span" if selected else "the section spans"
    return (f'--split "{expr}" --keep {keep} kept nothing: {what} {name} {lo:.4f}..{hi:.4f}, '
            f"all on the {other} side of {name}={val:g}. Keep {other}, or cut at a {name} "
            f"inside that span")


def _split_needs_rings(polys, ax, at):
    """Refuse a --split of a section with no closed ring, saying why (an open
    target)."""
    if not polys:
        raise SystemExit(f"--split: the section at {ax}={at:.4f} has no closed ring -- the "
                         f"target does not close at this level (an open or non-manifold "
                         f"stretch), so there is nothing to cut. Section at another level")


def _rename_ring(ring, idx):
    """Ring R{idx}, its primitives L{idx}_k / A{idx}_k."""
    ring["name"] = f"R{idx}"
    for sg in ring.get("segments", []):
        sg["name"] = re.sub(r"^([A-Za-z])\d+_", rf"\g<1>{idx}_", sg["name"])


def _source_names(kept, rings, clip=None):
    """Name each ring a --split kept after the ring of the section it was cut
    from. Numbered afresh from R0, R1 of `section --split` would be another ring
    than R1 of `section`, and a name read off one would pick a different ring in
    the other. A ring two pieces were cut from names the first; the second is
    numbered past the section's rings.

    Matched by overlap over union with the source ring as the cut leaves it
    (`clip`), not by how much of the piece lies inside it: an outline that
    contains a pipe would claim the pipe's piece, and at a boundary, where
    `section` prints both sides, the same ring twice tied -- the lower name,
    the one printed first, is taken."""
    top = max((int(r["name"][1:]) for r in rings if r.get("name", "")[1:].isdigit()), default=-1)
    used = set()
    for k in kept:
        g, best, share = readback.shape(k), None, 0.0
        scored = []
        for i, r in enumerate(rings):
            src = readback.shape(r) if r.get("role") == k.get("role") and r.get("name") else None
            if g is None or src is None:
                continue
            if clip is not None:
                src = src.intersection(clip)
            union = g.union(src).area
            scored.append((round(g.intersection(src).area / union, 3) if union > 0 else 0.0,
                           -i, r))
        if scored:
            share, _, best = max(scored, key=lambda t: t[:2])
        if best is not None and share > 0.5 and best["name"] not in used:
            idx = int(best["name"][1:])
        else:
            top += 1
            idx = top
        used.add(f"R{idx}")
        _rename_ring(k, idx)
    return kept


def _same_piece(kept, ring, clip):
    """Whether `kept` is what the cut left of `ring`: the same role, and
    overlapping it, as clipped, by more than half of their union."""
    if kept.get("role") != ring.get("role"):
        return False
    g, src = readback.shape(kept), readback.shape(ring)
    if g is None or src is None:
        return False
    if clip is not None:
        src = src.intersection(clip)
    union = g.union(src).area
    return union > 0 and g.intersection(src).area / union > 0.5


def _split_use(args, kept, rings, desc, ax, level, clip=None):
    """The rings of what a --split kept that --use names, by the section's own
    names, or all of them. A name cut away entirely, or no ring of the section
    at all, is refused with what is there -- never a fallback to the whole kept
    piece."""
    if not args.use:
        return kept
    arrays = _array_groups(kept)
    out = []
    for n in (n.strip() for n in args.use.split(",") if n.strip()):
        try:
            kind, obj = _resolve(n, kept, {}, arrays)
        except SystemExit:
            kind0, obj0 = _resolve_named(n, rings, {}, _array_groups(rings), args, ax, level)
            src = obj0 if kind0 == "array" else [obj0]
            # By place, not by name: at a boundary `section` prints one ring
            # under two names (R1 above, R3 below), and a kept ring carries one.
            hits = [k for k in kept if any(_same_piece(k, r, clip) for r in src)]
            if not hits:
                raise SystemExit(f"--use {n}: the cut {desc} keeps nothing of "
                                 f"{readback.describe(src[0])}. It keeps: "
                                 f"{readback.listing(kept)}") from None
            kind, obj = "array", hits
        if kind == "half":
            raise SystemExit(f"--use {n}: a half arc is not a ring of the cut; under "
                             f"--split, name rings or their primitives")
        for r in (obj if kind == "array" else [obj]):
            if all(r is not o for o in out):
                out.append(r)
    return out


def _axis_crosses(geoms, axis):
    """Whether the in-plane line through the two `axis` points runs through
    the inside of any of `geoms` (along an edge is fine)."""
    from shapely.geometry import LineString

    (x0, y0), (x1, y1) = axis
    d = np.array([x1 - x0, y1 - y0], float)
    d /= np.linalg.norm(d) or 1.0
    far = 10.0 * max(max(abs(c) for c in g.bounds) for g in geoms) + 10.0
    line = LineString([(x0 - far * d[0], y0 - far * d[1]), (x0 + far * d[0], y0 + far * d[1])])
    return any(g.buffer(-1e-6).intersection(line).length > 0 for g in geoms)


def split_profile(polys, spec, expr, keep, tol, grid, only=None, more=()):
    """Cut the section at a line, keep one side, and re-fit what remains.

    Done on the section POLYGONS, not on the fitted primitives: clipping a
    polygon with a half-plane is exact and cannot leave a gap at the join, and
    re-running the same decomposition afterwards recovers the cut line as an
    ordinary straight edge. The profile that comes out is therefore closed by
    construction and carries the same kind of names as any other section --
    which is the point, because the next step has to be able to refer to it.
    """
    from shapely.geometry import box
    from shapely.ops import unary_union

    m = re.match(r"\s*([A-Za-z])\s*=\s*(-?[\d.]+)\s*$", expr)
    if not m:
        raise SystemExit(f'--split wants "AXIS=VALUE", e.g. --split "X=0"; got {expr!r}')
    ax_name, val = m.group(1).upper(), float(m.group(2))
    if ax_name == spec["u_name"]:
        which = 0
    elif ax_name == spec["v_name"]:
        which = 1
    else:
        raise SystemExit(f"--split axis {ax_name} is not in this plane "
                         f"({spec['u_name']}, {spec['v_name']})")
    if keep not in ("+", "-"):
        raise SystemExit('--keep must be "+" or "-"')

    chosen = [p_ for i, p_ in enumerate(polys) if only is None or i in only]
    if not chosen:
        raise SystemExit("nothing selected to split")

    xs = [c for p_ in polys for c in p_.bounds[0::2]]
    ys = [c for p_ in polys for c in p_.bounds[1::2]]
    pad = max(max(xs) - min(xs), max(ys) - min(ys)) + 10.0
    lo_x, hi_x, lo_y, hi_y = min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad
    up = keep == "+"
    if which == 0:
        clip = box(val, lo_y, hi_x, hi_y) if up else box(lo_x, lo_y, val, hi_y)
        axis = ((val, 0.0), (val, 1.0))
    else:
        clip = box(lo_x, val, hi_x, hi_y) if up else box(lo_x, lo_y, hi_x, val)
        axis = ((0.0, val), (1.0, val))

    descs = [f"{ax_name}{'>=' if up else '<='}{val:g}"]
    for e2, k2 in more:
        # A second cut -- the band of a revolve, the foot an axis cut cannot
        # exclude -- kept on its own side.
        m2 = re.match(r"\s*([A-Za-z])\s*=\s*(-?[\d.]+)\s*$", e2)
        if not m2:
            raise SystemExit(f'--split wants "AXIS=VALUE", e.g. --split "X=0"; got {e2!r}')
        a2, v2 = m2.group(1).upper(), float(m2.group(2))
        if a2 not in (spec["u_name"], spec["v_name"]):
            raise SystemExit(f"--split axis {a2} is not in this plane "
                             f"({spec['u_name']}, {spec['v_name']})")
        up2 = k2 == "+"
        if a2 == spec["u_name"]:
            box2 = box(v2, lo_y, hi_x, hi_y) if up2 else box(lo_x, lo_y, v2, hi_y)
        else:
            box2 = box(lo_x, v2, hi_x, hi_y) if up2 else box(lo_x, lo_y, hi_x, v2)
        whole2 = unary_union(chosen)
        if whole2.intersection(box2).area <= 1e-9:
            raise SystemExit(_kept_nothing(e2, k2, chosen, 0 if a2 == spec["u_name"] else 1,
                                           v2, spec, [], only is not None))
        clip = clip.intersection(box2)
        descs.append(f"{a2}{'>=' if up2 else '<='}{v2:g}")

    kept = unary_union([p_.intersection(clip) for p_ in chosen])
    pieces = [g for g in getattr(kept, "geoms", [kept]) if g.area > 1e-9]
    # A piece thinner than SPLIT_SLIVER_MM is a thread, not a profile: a sliver
    # a hundred-thousandth thick fits to one primitive from a point back to
    # itself, and a revolve built from it breaks the program.
    geoms = [g for g in pieces if 2 * g.area / max(g.length, 1e-12) >= SPLIT_SLIVER_MM]
    if len(geoms) < len(pieces):
        print(f"note: --split left out {len(pieces) - len(geoms)} sliver(s) thinner "
              f"than {SPLIT_SLIVER_MM} mm")
    if not geoms:
        raise SystemExit(_kept_nothing(expr, keep, chosen, which, val, spec, pieces,
                                       only is not None))

    rings, ni, ri = [], 0, 0
    for pi, g in enumerate(geoms):
        # Tell the decomposition where the cut was, so it can absorb the stub the
        # cut leaves mid-chord and land the arc exactly on the line.
        rs = decompose_polygon(g, tol=tol, grid=grid, node_start=ni, ring_start=ri,
                               cut=(which, val))
        for r in rs:
            r["poly_index"] = pi
        rings.extend(rs)
        ni += sum(len(r["nodes"]) for r in rs)
        ri += len(rs)
    return geoms, rings, axis, ", ".join(descs), clip


# How far an axis may be from the sketch plane and still lie in it: the
# tolerance a named axis (F#, S#) is held to below, applied to a typed one too.
AXIS_IN_PLANE = 1e-3


def parse_axis(expr, spec, level, mesh=None, args=None):
    """A revolve axis named in world terms, e.g. "X=0,Y=0".

    Two coordinate constraints define a line. One of them must pin the section
    plane's own normal to the level we sketched at -- otherwise the named line
    does not lie in the profile's plane at all, and revolving about it is
    meaningless rather than merely wrong.

    Either value may be a level instead of a number ("Y=mid(B2,E1),Z=E0"), and
    the whole axis may be a name: a round face's axis (F#) or a rotation axis
    `symmetry` found (S#), rather than a value typed from the bounding box.
    """
    if mesh is not None and re.fullmatch(r"\s*[SFH]\d+\s*", expr):
        name = expr.strip()
        got = _named_axis(expr, mesh, args)
        k, pt = got
        free = [i for i in range(3) if i != k]
        expr = ",".join(f"{'XYZ'[i]}={_zero(pt[i]):.6f}" for i in free)
        normal = spec["axis"]
        if abs(pt[normal] - level) > AXIS_IN_PLANE and normal != k:
            raise SystemExit(f"--axis {name}: that axis passes "
                             f"{'XYZ'[normal]}={pt[normal]:.4f}, not the sketch plane "
                             f"at {level:g} -- section through it first")
        if normal != k:
            expr = ",".join(f"{'XYZ'[i]}={_zero(level if i == normal else pt[i]):.6f}"
                            for i in free)
    vals = {}
    # Commas inside mid(A,B) do not separate the two constraints.
    for q in (x.strip() for x in re.split(r",(?![^()]*\))", expr.replace(" ", ""))
              if x.strip()):
        m = re.match(r"^([A-Za-z])=(-?[\d.]+)$", q)
        n = re.match(r"^([XYZxyz])=(.+)$", q)
        if not m and n and mesh is not None:
            k = "XYZ".index(n.group(1).upper())
            vals[n.group(1).upper()] = _level_value(n.group(2), k, mesh, args, "--axis")
            continue
        if not m:
            raise SystemExit(f'--axis wants "X=0,Y=0", level names ("Y=B1,Z=E0"), '
                             f'or a name (F#, S#); could not read {q!r}')
        vals[m.group(1).upper()] = float(m.group(2))
    if len(vals) != 2:
        raise SystemExit('--axis needs exactly two constraints, e.g. "X=0,Y=0"')

    normal_name = "XYZ"[spec["axis"]]
    if normal_name not in vals:
        raise SystemExit(f'--axis must pin {normal_name} (the plane normal) to '
                         f'{level:g}, e.g. "{normal_name}={level:g},X=0"')
    off = abs(vals[normal_name] - level)
    if off > AXIS_IN_PLANE:
        raise SystemExit(f"--axis pins {normal_name}={vals[normal_name]:g} but the "
                         f"sketch plane is at {normal_name}={level:g}; the axis does "
                         f"not lie in the profile's plane")
    if off > 1e-9:
        # A level printed to four places and typed back is up to 5e-5 off the
        # value it names, and the revolve must not be refused for it.
        print(f"note: --axis pins {normal_name}={vals[normal_name]:g}, {off:.1e} from the "
              f"sketch plane {normal_name}={level:.6g}; taken to lie in it (within "
              f"{AXIS_IN_PLANE:g}, as a named axis is)")

    other = [k for k in vals if k != normal_name][0]
    v = vals[other]
    if other == spec["u_name"]:
        return ((v, 0.0), (v, 1.0))
    if other == spec["v_name"]:
        return ((0.0, v), (1.0, v))
    raise SystemExit(f"--axis constrains {other}, which is not in this plane")


def _helix_fit(mesh, name="H0"):
    """The helix `overview` fitted on this mesh, or why there is not one.

    Re-fitted rather than carried: `sketch` is a separate invocation and the
    classifier is cheap next to the sweep it is about to drive. Re-measuring is
    also the only way `--along H0` can mean the same path it meant when overview
    printed the name -- exactly the failure that `--samples` caused for level
    names, and not a mistake worth making twice.
    """
    reg = classify(mesh)
    k = int(name[1:] or 0) if name and name[0] == "H" and name[1:].isdigit() else 0
    hs = reg.evidence.get("helices") or []
    if k and k < len(hs):
        return hs[k]
    if k:
        return {"is_helix": False, "reason": f"there is no {name}: {len(hs)} helix(es) fitted"}
    return reg.evidence.get("helix") or {"is_helix": False,
                                         "reason": f"regime is {reg.kind}"}


# A stack that is not named but lies on a named one's axis within this share of
# the smaller of the two stacks' largest radii, and meets it along the axis, is
# the same hole split in two: a counterbore's short wall can fit its axis a
# fraction of a mm off the bore's, `faces` names it apart, and the model would
# leave the counterbores in, with a thin wall across every bore.
CONTINUE_OFFSET = 0.02


def _continuations(seg, names):
    """[(stack, the named stack it continues, axis offset)] for stacks not
    named that continue a named one along its axis."""
    from mesh2cad.holes import HoleError, pieces, stack_names

    def span(n):
        try:
            k, ps, _ = pieces(seg, [n])
        except HoleError:
            return None
        if not ps:
            return None
        return (k, np.asarray(ps[0][1], float), min(p[2] for p in ps),
                max(p[3] for p in ps), max(max(p[4], p[5]) for p in ps))

    named = [(n, span(n)) for n in names]
    out = []
    for other, _ in stack_names(seg):
        if other in names:
            continue
        so = span(other)
        if so is None:
            continue
        for n, sn in named:
            if sn is None or sn[0] != so[0]:
                continue
            off = float(np.linalg.norm(sn[1] - so[1]))
            gap = max(so[2] - sn[3], sn[2] - so[3])
            if off <= CONTINUE_OFFSET * min(sn[4], so[4]) and gap <= 1e-3:
                out.append((other, n, off))
                break
    return out


def _air_between(mesh, plane, a, b, profile):
    """Whether the target is air inside `profile` all the way from level a to b."""
    from shapely.ops import unary_union

    from mesh2cad.section import CutError
    if b - a <= 1e-3:
        return True                     # thinner than anything a part is made of
    for f in (0.25, 0.5, 0.75):
        try:
            cut = section_at(mesh, plane, a + f * (b - a))
        except CutError:
            return False
        if cut and unary_union(cut).intersection(profile).area >= 0.02 * profile.area:
            return False
    return True


def _bridge_gaps(mesh, plane, k, parts):
    """(parts, notes): a cylinder added across every gap along one hole's axis
    where the target is air -- a face of the hole `pieces` left out.

    A face of a bore step can be left out as "convex, or not a cylinder or
    cone", leaving a gap in the cutter, and every such bore would come out
    blocked by a disc. Its radius is the smaller of the two pieces either side,
    so the bridge never cuts wider than the hole. A gap where the target has
    material -- two holes on one axis with a wall between them -- is left alone.
    """
    from shapely.geometry import Point

    by_centre = {}
    for p in parts:
        by_centre.setdefault((round(p[1][0], 4), round(p[1][1], 4)), []).append(p)
    add, notes = [], []
    ax = "XYZ"[k]
    for ps in by_centre.values():
        ps = sorted(ps, key=lambda p: p[2])
        reach, r_end = ps[0][3], ps[0][5]
        for p in ps[1:]:
            if p[2] > reach + 1e-6:
                r = min(r_end, p[4])
                if r > 0 and _air_between(mesh, plane, reach, p[2], Point(p[1]).buffer(r, 32)):
                    add.append(("cylinder", p[1], reach, p[2], r, r))
                    notes.append(f"note: the hole on ({p[1][0]:.4f}, {p[1][1]:.4f}) has a "
                                 f"{p[2] - reach:.4f} mm gap between its pieces ({ax}={reach:.4f} "
                                 f"to {ax}={p[2]:.4f}) where the target is air -- a face of it "
                                 f"left out above -- so it is cut there at R {r:.4f}")
            if p[3] > reach:
                reach, r_end = p[3], p[5]
    return parts + add, notes


def _hole_group(args, mesh, seg, names):
    """The cutter for stacks along ONE axis of this mesh's frame.

    Returns (code, plane, axis letter, lowest level, detail).
    """
    from shapely.geometry import Point

    from mesh2cad.holes import (
        PLANE_OF_AXIS,
        HoleError,
        cutter_code,
        describe_parts,
        pieces,
    )

    names, of_stack = list(names), {}
    for other, of, off in _continuations(seg, names):
        print(f"note: {other} lies on the axis of {of} ({off:.3f} mm off it) and runs on "
              f"from it -- one hole the segmentation split in two -- so it is cut with it, "
              f"on {of}'s axis")
        names.append(other)
        of_stack[other] = of
    try:
        k, _, notes = pieces(seg, names)            # one axis, every name known
        per = {n: pieces(seg, [n])[1] for n in names}
    except HoleError as exc:
        raise SystemExit(f"--op hole: {exc}") from None
    # Every piece of a stack on ONE centre -- its faces' fitted axes differ in
    # the 4th decimal, which would split one hole into two for the run-on and
    # the gaps below -- and a stack split off a named one on that one's centre:
    # a fraction of a mm off, counterbore floors meet the bores' ends in near
    # coincident offset planes and OCC returns no solid for the whole cut.
    centre = {n: tuple(float(x) for x in np.mean([p[1] for p in ps], axis=0))
              for n, ps in per.items()}
    parts = [(p[0], centre[of_stack.get(n, n)]) + tuple(p[2:])
             for n in names for p in per[n]]
    for n in notes:
        print(n)
    plane = PLANE_OF_AXIS[k]
    ax = "XYZ"[k]
    parts, notes = _bridge_gaps(mesh, plane, k, parts)
    for n in notes:
        print(n)
    # Through: the outermost piece of each hole runs on past the face it
    # leaves by, when the target is open there (as for --combine cut). A cone
    # stays as measured and runs on as a cylinder of its end radius: moving
    # its end with both radii kept would change the cone's angle and leave a
    # wedge in every bore.
    by_centre = {}
    for idx, p in enumerate(parts):
        by_centre.setdefault((round(p[1][0], 4), round(p[1][1], 4)), []).append(idx)
    run_on = []
    for idxs in by_centre.values():
        low = min(idxs, key=lambda i: parts[i][2])
        high = max(idxs, key=lambda i: parts[i][3])
        kind, c, lo, hi, r0, r1 = parts[low]
        lo2, _, n_lo = _through_ends(mesh, plane, None, lo, hi, ends=(True, False),
                                     profile=Point(c).buffer(max(r0, 1e-6), 32))
        if abs(r0 - r1) < 1e-9:
            parts[low] = (kind, c, lo2, hi, r0, r1)
        elif lo2 < lo:
            run_on.append(("cylinder", c, lo2, lo, r0, r0))
        kind, c, lo, hi, r0, r1 = parts[high]
        _, hi2, n_hi = _through_ends(mesh, plane, None, lo, hi, ends=(False, True),
                                     profile=Point(c).buffer(max(r1, 1e-6), 32))
        if abs(r0 - r1) < 1e-9:
            parts[high] = (kind, c, lo, hi2, r0, r1)
        elif hi2 > hi:
            run_on.append(("cylinder", c, hi, hi2, r1, r1))
        for n in n_lo + n_hi:
            print(n)
    parts = parts + run_on
    try:
        code = cutter_code(k, parts)
    except HoleError as exc:
        raise SystemExit(f"--op hole: {exc}") from None
    lo = min(p[2] for p in parts)
    return code, plane, ax, lo, f"{', '.join(names)}: {describe_parts(parts, k)}"


def _axis_groups(seg, names):
    """The named stacks grouped by the line direction of their axes, in the
    order named: [(direction, [names])]."""
    from mesh2cad.holes import stack_names

    stacks = dict(stack_names(seg))
    groups = []
    for n in names:
        if n not in stacks:
            groups.append((None, [n]))          # pieces() says what is wrong
            continue
        a = np.asarray(seg.surfaces[stacks[n][0]].params["axis"], float)
        for d, g in groups:
            if d is not None and abs(float(a @ d)) > 1 - 1e-6:
                g.append(n)
                break
        else:
            groups.append((a / np.linalg.norm(a), [n]))
    return groups


def _sketch_hole(args, mesh):
    """`--op hole`: cut the named coaxial stacks (mesh2cad.holes).

    Stacks along several axes are one command and one feature: each axis's
    stacks are cut in that axis's own frame (none for X, Y, Z), turned back,
    and the cutters unioned, so countersunk holes on several axes (two on Y
    and Z, two on the 45 deg diagonals, say) are cut in one command.
    """
    from mesh2cad.frames import (
        back_code,
        bspline_code,
        frame_for,
        gather_code,
        inclined,
        turn_mesh,
    )
    from mesh2cad.segment import frame_source, segment_cached

    if not args.use:
        raise SystemExit("--op hole needs --use with stack names from `faces` "
                         "(C0, C1, ...)")
    if not args.base:
        raise SystemExit("--op hole cuts into a program: give --base PROG")
    ignored = [f for f in ("plane", "at", "to") if getattr(args, f, None) is not None]
    if ignored:
        print(f"note: --op hole takes its axis and depth from the stack; "
              f"{', '.join('--' + f for f in ignored)} not used")
    if getattr(args, "combine", "add") != "cut":
        print("note: --op hole always cuts (--combine cut)")
        args.combine = "cut"
    seg = segment_cached(mesh, cache_dir=_cache_dir(args))
    names = [n.strip().upper() for n in args.use.split(",") if n.strip()]
    groups = _axis_groups(seg, names)
    if len(groups) == 1 or getattr(args, "_frame_info", None):
        code, plane, ax, lo, detail = _hole_group(args, mesh, seg, names)
        if getattr(args, "_frame_info", None):
            code += bspline_code(code)      # turned back by _finish_sketch
        args.plane = plane
        _finish_sketch(args, code, ax, lo, lo, None, detail, "hole")
        return

    blocks, details, first, framed = [], [], None, False
    for i, (d, g) in enumerate(groups):
        if d is not None and inclined(d):
            fr = frame_for(d)
            turned = turn_mesh(mesh, fr["R"])
            frame_source(turned, mesh, fr["R"])
            print(f"note: {', '.join(g)} lie on an inclined axis: cut in its own "
                  f"frame (turned {fr['angle']:.4f} deg onto "
                  f"{'+' if fr['sign'] > 0 else '-'}{'XYZ'[fr['k']]}) and turned back")
            code, plane, ax, lo, detail = _hole_group(
                args, turned, segment_cached(turned, cache_dir=_cache_dir(args)), g)
            code += bspline_code(code) + back_code(fr, g[0])
            framed = True
            detail += f" (in the frame of {g[0]})"
        else:
            code, plane, ax, lo, detail = _hole_group(args, mesh, seg, g)
        if first is None:
            first = (plane, ax, lo)
        blocks.append(f"{code}\n_c{i} = result\n_c{i}_steps = list(_STEPS)")
        details.append(detail)
    n = len(blocks)
    # B-spline pieces are gathered, never fused (frames.bspline_code).
    code = ("\n".join(blocks)
            + (gather_code(n) if framed else
               "\nresult = _c0" + "".join(f".union(_c{i})" for i in range(1, n)))
            + "\n_STEPS = " + " + ".join(f"_c{i}_steps" for i in range(n)))
    args.plane, ax, lo = first
    _finish_sketch(args, code, ax, lo, lo, None,
                   " | ".join(details) + f" -- {n} axes, each in its own frame",
                   f"hole on {n} axes")


def _sketch_blend(args, mesh):
    """`--op fillet` / `--op chamfer`: the target's blend faces, as solids
    (mesh2cad.blends). Cut on a convex edge, added on a concave one."""
    from mesh2cad.blends import (
        PLANE_OF_AXIS,
        BlendError,
        code,
        describe,
        describe_trims,
        parts,
        trims,
    )
    from mesh2cad.segment import segment_cached

    op = args.op
    _tb.stage(f"finding the edges and faces of the {op} on {args.use}")
    if not args.use:
        raise SystemExit(f"--op {op} needs --use with face names from `faces` (F12, ...)")
    if not args.base:
        raise SystemExit(f"--op {op} rounds an edge of a program: give --base PROG")
    ignored = [f for f in ("plane", "at", "to") if getattr(args, f, None) is not None]
    if ignored:
        print(f"note: --op {op} takes its place and size from the faces; "
              f"{', '.join('--' + f for f in ignored)} not used")
    seg = segment_cached(mesh, cache_dir=_cache_dir(args))
    names = [n.strip().upper() for n in args.use.split(",") if n.strip()]
    try:
        specs = parts(mesh, seg, names, op)
    except BlendError as exc:
        raise SystemExit(f"--op {op}: {exc}") from None
    mode = specs[0]["mode"]
    for x in specs:
        if x.get("note"):
            print(x["note"])
    want = "cut" if mode == "cut" else "add"
    if (getattr(args, "combine", "add") or "add") != want:
        print(f"note: --op {op} on a {'convex' if mode == 'cut' else 'concave'} edge "
              f"{'cuts' if mode == 'cut' else 'adds'} (--combine {want})")
    args.combine = want
    first = specs[0]
    a = np.asarray(first["axis"], float)
    k = int(np.argmax(np.abs(a)))
    args.plane = PLANE_OF_AXIS[k]
    p0, lo, _hi = first["along"]
    level = float(p0[k] + lo * a[k])
    args._blend_before = _blend_volumes(args.base, code(specs))
    # A concave blend also cuts the build back on its open side, as a feature of
    # its own before the one that adds it (see blends.trims).
    pre = None
    back = trims(mesh, specs) if mode == "add" else []
    if back and _overlap(args.base, code(back)) <= 1e-3:
        back = []                           # the build has nothing on the open side
    if back and getattr(args, "replace", None) is not None:
        print(f"note: --op {op} with --replace adds the blend only; the build is not cut "
              f"back on its open side ({describe_trims(back)})")
    elif back:
        print(f"note: the build is also cut back to the blend on its open side, where the "
              f"target is air: {describe_trims(back)}")
        pre = (f"# sketch on {args.plane} at {'XYZ'[k]} = {level:.4f}\n"
               f"# profile measured at {'XYZ'[k]} = {level:.4f}\n"
               f"# {op} trim: {describe_trims(back)}\n"
               "# combine: cut (subtracted from everything before it)\n"
               + code(back) + "\n" + GLUES["cut"] + "\n_prev = result\n_PREV_STEPS = list(_STEPS)\n\n")
    _finish_sketch(args, code(specs), "XYZ"[k], level, level, None, describe(specs), op, pre=pre)


def _blend_volumes(base, blocks):
    """(the base program's volume, the blend solids' volume), or None -- read
    before the program is written over, for `_blend_effect`."""
    try:
        vols = []
        for src in (Path(base).read_text(encoding="utf-8"),
                    "import cadquery as cq\n" + blocks + "\nshow_object(result)\n"):
            ns = {"show_object": lambda o, *a, **k: None}
            exec(compile(src, "<blend>", "exec"), ns)
            vols.append(float(ns["result"].val().Volume()))
        return tuple(vols)
    except Exception:                                              # noqa: BLE001
        return None


def _overlap(base, blocks):
    """mm^3 of the base program's build inside the solids `blocks` make; 0 when
    it cannot be measured, so nothing is added on a guess."""
    try:
        shapes = []
        for src in (Path(base).read_text(encoding="utf-8"),
                    "import cadquery as cq\n" + blocks + "\nshow_object(result)\n"):
            ns = {"show_object": lambda o, *a, **k: None}
            exec(compile(src, "<overlap>", "exec"), ns)
            shapes.append(ns["result"])
        return float(shapes[0].intersect(shapes[1]).val().Volume())
    except Exception:                                              # noqa: BLE001
        return 0.0


def _blend_effect(args, volume):
    """A `check: blends` line when a fillet or chamfer changed the build by
    next to nothing: the corner it shapes is not in the build, and the model
    would otherwise blame the op."""
    before = getattr(args, "_blend_before", None)
    if not before:
        return None
    base, solids = before
    if solids <= 0 or abs(volume - base) > 0.01 * solids:
        return None
    cut = getattr(args, "combine", "cut") == "cut"
    return (f"the {args.op} solid{'s' if ',' in (args.use or '') else ''} ({solids:.4g} mm^3) changed the build "
            f"by {abs(volume - base):.3g} mm^3: the build has {'no material' if cut else 'material already'} "
            f"where {'it cuts' if cut else 'it adds'}, so that corner is shaped by an earlier feature -- "
            f"`residual` shows what is really {'extra' if cut else 'missing'} there")


def cmd_sketch(args):
    _split_cuts(args)
    mesh, _info = ingest(args.mesh)
    # The frame the profile was measured in travels with the program, because
    # the program is only a reconstruction of the input mesh together with it.
    _program_frame(args, _info, args.base)
    # `--check` compares the built body count against the target's rather than
    # against 1, so it needs the target.
    args._target_mesh = mesh
    if (getattr(args, "along", None) or "").strip().upper().startswith("W"):
        args.along = args.along.strip().upper()
        if not args.use and not getattr(args, "face", None) and _strip_here(args):
            # W# is the centreline of ring R#: naming the path names the ring.
            args.use = "R" + args.along[1:]
            args._use_from_path = True
    if getattr(args, "law", False) and args.op != "loft":
        raise SystemExit(f"--law builds a loft (--op loft); --op {args.op} has no "
                         f"law to follow")
    if getattr(args, "taper", False) and args.op != "extrude":
        raise SystemExit(f"--taper drafts an extrude; --op {args.op} has no draft "
                         f"to give it (a loft between two measured profiles is "
                         f"its own taper)")
    # After _target_mesh: --check measures the program, which is always in the
    # target's frame, against the target -- never against the turned copy.
    mesh = _apply_frame(args, mesh)
    if getattr(args, "meridian", None):
        return _sketch_meridian(args, mesh)
    if args.op == "hole":
        return _sketch_hole(args, mesh)
    _check_to_profile(args)
    wire = _wire_path(args)
    if wire is not None:
        if getattr(args, "to_profile", None):
            raise SystemExit(f"--to-profile: {args.along} is a wire, round all the way along. "
                             f"Sweep it with no second profile")
        return _sketch_wire(args, wire)
    if args.op in ("fillet", "chamfer"):
        return _sketch_blend(args, mesh)
    _sweep_face_plane(args, mesh)
    if not args.plane:
        raise SystemExit("m2c sketch: error: the following arguments are "
                         "required: --plane (or --frame, which sets it)")
    spec = PLANES[args.plane]
    ax = "XYZ"[spec["axis"]]
    sense = float(spec["normal"][spec["axis"]])

    # Levels are named, so --at and --to take names, not raw numbers. Resolved
    # here, before anything is measured, so the rest of the command sees plain
    # numbers and nothing downstream has to know about naming.
    _marks, _ = _levels(args, mesh)
    face = None
    if getattr(args, "sharp", False) and not getattr(args, "face", None):
        raise SystemExit("--sharp works on a face's outline: give --face")
    if getattr(args, "face", None):
        face = _face_profile(mesh, args, args.plane, args.face)
        for note in face[3]:
            print(note)
        if args.at is None:
            args.at = face[1]
    if args.at is None:
        raise SystemExit("sketch needs --at (a level name) or --face (a flat "
                         "face from `faces`)")
    args.at = resolve_level(args.at, _marks, "--at", mesh, args.plane)
    args._marks = _marks
    if face is not None and abs(float(args.at) - face[1]) > 1e-9:
        raise SystemExit(f"--at {args.at:.4f} is not where face {args.face} is "
                         f"({face[1]:.4f}); give one or the other")
    # No `--to boundary` / `next` / `end` any more. Those made the TOOL pick
    # which level to stop at -- and a tool that decides is a bug,
    # however sensible its decision. Every level they could have chosen now has
    # a name, so `--to E1` and `--to B3` say the same thing and the choice is
    # the model's. They were also quietly lossy: "the nearest boundary in the
    # extrusion direction" is a different level depending on which side the
    # snap happened to measure from.
    if args.to is None and args.op in ("extrude", "loft"):
        raise SystemExit(
            f"--op {args.op} needs --to: the level the feature stops at. Name "
            f"one of "
            + ", ".join(f"{m['name']}={m['value']:.4f}"
                        for m in sorted(_marks, key=lambda m: _level_key(m["name"])))
            + ", or mid(A,B) of two of them, or a number.")
    # `--to B1,B2,B3` lofts through every level named, in the order named. Each
    # resolves the same way a single --to does, so a list carries the same
    # precision as the names in it and there is no second syntax to learn.
    # A revolve ends at a face it turns onto, not at a level along the plane
    # normal: resolved as one, --to F14 would become a number nothing uses.
    args._revolve_to = None
    if args.op == "revolve" and args.to is not None:
        args._revolve_to, args.to = str(args.to).strip(), None
    args._loft_levels = None
    if args.op == "loft" and _split_levels(args.to) is not None:
        names = _split_levels(args.to)
        args._loft_levels = [resolve_level(n, _marks, "--to", mesh, args.plane)
                             for n in names]
        if len(args._loft_levels) != len(set(args._loft_levels)):
            raise SystemExit(f"--to {args.to} names the same level twice; a "
                             f"loft through a level and itself has no thickness")
        args.to = args._loft_levels[-1]
    else:
        args.to = resolve_level(args.to, _marks, "--to", mesh, args.plane)

    if getattr(args, "project", None):
        if face is not None or args.split:
            raise SystemExit("--project takes --at and --use: not --face (already "
                             "an outline) or --split")
        polys, _other, proj_note = _project_polys(args, mesh, float(args.at), _marks)
        print(proj_note)
        face = (polys, float(args.at),
                args.side or ("above" if _other > float(args.at) else "below"), [])
    rings, halves, marks, level, cut_at, snapped = _section_at_level(
        mesh, args.plane, args.at, args.samples, args.tol, args.grid, side=args.side,
        source=getattr(args, "levels", None), cache_dir=_cache_dir(args), face=face)
    if getattr(args, "project", None):
        snapped = f"projected to {args.project}"
    if face is None and not args.split and args.op == "extrude":
        # a prism's warning: a sweep's or a revolve's profile leans by design
        lean = _section_lean(args, mesh, args.plane, cut_at)
        if lean:
            print(lean)

    lo_b, hi_b = (float(v) for v in mesh.bounds[:, spec["axis"]])

    if args.split:
        # The split says which geometry this sketch is about, and the snap has
        # to agree with it. Snapping consults feature_levels of the WHOLE mesh,
        # so on a residual it can pull --at onto a boundary contributed by bodies
        # the split discards, and measure the profile inside a transition.
        # Cutting out a profile means the plane was chosen deliberately; take it
        # literally.
        if snapped:
            print(f"note: --at {args.at:.4f} would have snapped to "
                  f"{ax}={level:.4f} ({snapped}), but --split selects its own "
                  f"geometry, so the level is taken as given")
        measured = cut_at
        level = cut_at = float(args.at)
        snapped = None
        polys = section_at(mesh, args.plane, cut_at)
        if not polys and abs(measured - cut_at) > 1e-9:
            # The level is a face, so the cut exactly on it is empty, while
            # `section` -- and the names the model read -- measured it a hair
            # inside. Cut there too.
            polys = section_at(mesh, args.plane, measured)
            if polys:
                print(f"note: the section exactly at {ax}={cut_at:.4f} is empty (a face "
                      f"lies on it), so --split cuts where the section was measured, "
                      f"{ax}={measured:.4f}, as `section` shows it")
                cut_at = measured
        _split_needs_rings(polys, ax, cut_at)
        # The names come from this exact cut, as `section --split` takes
        # them. Matched against rings measured at the snapped level instead,
        # the kept pieces could get other names than section printed, and a
        # sweep would take a ring the model didn't name.
        rings, _, _ = _rings_from_polys(polys, args.tol, args.grid)
        cuts = [(resolve_split_expr(e, rings, mesh=mesh, args=args, spec=spec), k)
                for e, k in args._cuts]
        args.split = cuts[0][0]
        if args.op not in ("extrude", "revolve", "sweep"):
            # R2: a loft of the kept half must not be built as a 360 deg revolve
            # about the cut line.
            raise SystemExit(f"--split with --op {args.op}: the cut keeps one section, and "
                             f"--split builds an extrude, a revolve or a sweep of it. A loft "
                             f"joins rings measured at two levels: name them with --use and "
                             f"no --split (--use R0 --op loft --to B1)")
        # The whole section is cut, as `section --split` cuts it, and --use picks
        # from what is kept, by the section's names.
        geoms, hr, cut_axis, desc, clip = split_profile(
            polys, spec, args.split, args.keep, args.tol, args.grid, more=cuts[1:])
        _source_names(hr, rings, clip)
        hr = _split_use(args, hr, rings, desc, ax, level, clip)
        outer = [r for r in hr if r["role"] == "outer"]
        holes = [r for r in hr if r["role"] == "hole"]
        if not outer and holes:
            if (getattr(args, "combine", "add") or "add") == "add":
                raise SystemExit(f"--use {args.use} names only holes of the cut; a hole "
                                 f"is a cutter's outline (--combine cut), or name the "
                                 f"outline it sits in as well")
            outer, holes = holes, []
            print(f"cutter: hole ring(s) {', '.join(r['name'] for r in outer)} used as "
                  f"the cut tool's outline (filled)")
        if holes:
            print(f"note: --split keeps {', '.join(r['name'] for r in holes)} as "
                  f"hole{'s' if len(holes) > 1 else ''} of the kept profile")

        if getattr(args, "grow", None):
            hr = grow_edges(hr, args.grow)
            print(f"grew {args.grow} -- the join now overlaps instead of "
                  f"sharing a face")

        if args.op == "extrude":
            # --split does not imply revolve. The cut is just a way of isolating a
            # profile, and a profile is a profile: a spoke is cut out of the section
            # exactly the same way and then extruded, not revolved. Passing
            # --op extrude and silently getting a revolve is the kind of
            # disagreement between what was asked and what was built that no
            # amount of looking at the render will explain.
            target = float(args.to)
            ends = _both_span(args, ax, level, target) if args.both else None
            if ends:
                print(ends[2])
                code, dist = build_extrude(args.plane, sense, ends[0], ends[1], outer, holes)
                span = (f"{ax}={min(ends[:2]):.4f} to {ax}={max(ends[:2]):.4f} "
                        f"(both sides of {ax}={level:.4f}, {abs(dist):.4f} mm total, "
                        f"far end on a named level)")
            elif args.both:
                code, dist = build_extrude(args.plane, sense, level, target, outer, holes,
                                           both=True)
                span = (f"{ax}={level - abs(dist):.4f} to "
                        f"{ax}={level + abs(dist):.4f} "
                        f"(both sides, {2 * abs(dist):.4f} mm total)")
            else:
                code, dist = build_extrude(args.plane, sense, level, target, outer, holes)
                span = (f"from {ax}={level:.4f} to {ax}={target:.4f} "
                        f"(depth {abs(dist):.4f} mm)")
            detail = (f"{span}, profile cut at {desc} "
                      f"({len(geoms)} piece(s), "
                      f"{sum(r['n_primitives'] for r in hr)} primitives)")
            _finish_sketch(args, code, ax, level, cut_at, snapped, detail,
                           "extrude",
                           profiles=f"{sum(r['n_primitives'] for r in hr)} "
                                    f"primitives cut at {desc}")
            return

        if args.op == "sweep":
            names = {r.get("name") for r in hr}
            pick = getattr(args, "profile", None)
            if not pick and len(hr) > 1:
                raise SystemExit(f"--split kept {len(hr)} pieces ({', '.join(sorted(names))}); "
                                 f"name the one to sweep with --profile (section --split "
                                 f"prints them)")
            ring = next((r for r in hr if r.get("name") == (pick or hr[0].get("name"))), None)
            if ring is None:
                raise SystemExit(f"--profile {pick}: the kept pieces are "
                                 f"{', '.join(sorted(names))}")
            if (args.along or "").startswith("W"):
                # A kept piece with a path `faces` named goes to the path sweep, not
                # the helix sweep.
                _sweep_along(args, mesh, hr, ring, [], spec, ax, sense, level, cut_at,
                             snapped)
                return
            _sweep_profile(args, mesh, ring, level, spec, ax, cut_at, snapped, marks)
            return

        axis, axis_note = cut_axis, f"the cut line {desc.split(', ')[0]}"
        if len(cuts) > 1 and not args.axis:
            print(f"note: no --axis: the revolve turns about the first cut's line, "
                  f"{desc.split(', ')[0]}")
        if args.axis:
            axis = parse_axis(args.axis, spec, level, mesh=mesh, args=args)
            axis_note = args.axis.strip()
            # Revolving about anything other than the line the profile was cut
            # on sweeps an open edge, because the closing edge only lies on the
            # cut. Say so rather than emitting a solid nobody can trust.
            # R2: an axis through the inside of the profile sweeps the profile
            # through itself, and OCC cannot build that at all.
            if _axis_crosses([g for g in (readback.shape(r) for r in outer + holes)
                              if g is not None], axis):
                u = spec["u_name"] if abs(axis[0][0] - axis[1][0]) < 1e-12 else spec["v_name"]
                at = axis[0][0] if u == spec["u_name"] else axis[0][1]
                raise SystemExit(
                    f"--axis {axis_note} runs through the inside of the profile cut "
                    f"at {desc}: a revolve about it sweeps the profile through itself "
                    f"and cannot be built. Cut the profile at the axis instead: "
                    f'--split "{u}={at:g}" with --keep + or -')
            if abs(axis[0][0] - cut_axis[0][0]) > 1e-6 or \
               abs(axis[0][1] - cut_axis[0][1]) > 1e-6:
                print(f"WARNING: axis {axis_note} is not the cut line {desc}; the "
                      f"profile is closed along the cut, so this revolve will not "
                      f"close on itself")

        angle = args.angle
        if getattr(args, "_revolve_to", None):
            angle, end_note = _revolve_end(args, mesh, spec, level, axis, axis_note, outer,
                                           holes, args._revolve_to)
            print(end_note)
        code, ang = build_revolve_profile(args.plane, sense, level, outer, holes, axis, angle)
        detail = (f"{ang:g} deg about {axis_note}, profile cut at {desc} "
                  f"({len(geoms)} piece(s), "
                  f"{sum(r['n_primitives'] for r in hr)} primitives)")
        for r in hr:
            print(f"  ring {r['role']}: {r['n_primitives']} prims "
                  f"({r['n_lines']} lines, {r['n_arcs']} arcs) "
                  f"max residual {r['max_residual']:.5f} mm")
        _finish_sketch(args, code, ax, level, cut_at, snapped, detail, "revolve",
                       profiles=f"{sum(r['n_primitives'] for r in hr)} primitives "
                                f"cut at {desc}")
        return

    if face is not None and not args.use and not args.split:
        # What --use's help has always said: the face IS the profile, so without
        # --use every ring of it is taken -- its outline and the holes in it.
        args.use = ",".join(r["name"] for r in rings if r.get("name"))
        print(f"--face {args.face} without --use: the whole face, rings {args.use}")
    if not args.use and not args.split:
        raise SystemExit("give --use with primitive names, or --split to cut the "
                         "whole section")
    arrays = _array_groups(rings)
    picked = ([_resolve_named(n, rings, halves, arrays, args, ax, level)
               for n in args.use.split(",") if n.strip()] if args.use else [])
    # An array name stands for its members, so flatten it back to rings here and
    # everything downstream -- sides, roles, emitted code -- is unchanged.
    chosen = []
    for kind, obj in picked:
        if kind == "array":
            chosen.extend(("ring", r) for r in obj)
        else:
            chosen.append((kind, obj))
    outer = [obj for kind, obj in chosen if kind == "ring" and obj["role"] == "outer"]
    holes = [obj for kind, obj in chosen if kind == "ring" and obj["role"] == "hole"]
    half_profiles = [obj for kind, obj in chosen if kind == "half"]
    # A hole in the section IS the cutter's outline: the cross-hole of a block
    # shows up as an inner ring of the block's cut, and that ring is what the
    # model names. Named alone under --combine cut/intersect, it is filled.
    if getattr(args, "combine", "add") != "add" and holes and not outer:
        outer, holes = holes, []
        for r in outer:
            r["was_role"] = r["role"]
            r["role"] = "outer"
        print(f"cutter: hole ring(s) {', '.join(r.get('name', '?') for r in outer)} "
              f"used as the {args.combine} tool's outline (filled)")
    elif args.op in ("extrude", "revolve") and holes and not outer and not half_profiles:
        # A pin standing in another body's bore, a boss seen from inside the
        # plate it stands on -- in this section its ring is a hole, and "needs
        # at least one outer profile" would be all the model gets. Named alone,
        # the opening is what gets built.
        outer, holes = holes, []
        for r in outer:
            r["was_role"] = r["role"]
            r["role"] = "outer"
        print(f"note: {', '.join(r.get('name', '?') for r in outer)} "
              f"{'is a hole' if len(outer) == 1 else 'are holes'} in this section, named "
              f"alone: the {args.op} fills {'it' if len(outer) == 1 else 'them'} -- the "
              f"solid that sits in that opening (a pin in a bore, a boss behind the "
              f"plate). To cut the opening instead: --combine cut")

    # `--grow` used to be read only inside the `--split` branch above. Off that
    # path the parser accepted it, nothing consulted it, and the command
    # reported success -- so the model saw its profile come back unchanged and
    # concluded the flag was "being silently ignored", which it was. It then
    # re-sent the same command sixteen more times. R2: a flag that cannot act
    # must say so, and this one can act here, because growing an edge by name
    # needs a named edge and not a cut.
    if getattr(args, "grow", None):
        if not (outer or holes):
            raise SystemExit("--grow needs a profile to act on; --use named no "
                             "rings, only half-profiles")
        grow_edges(outer + holes, args.grow)
        print(f"grew {args.grow} -- the join now overlaps instead of "
              f"sharing a face")

    lo, hi = (float(v) for v in mesh.bounds[:, spec["axis"]])

    # Which way does the solid lie from the sketch plane? Take it from the side
    # the chosen primitives came from -- naming A2_0 means "the prism below the
    # boundary", so extruding upwards would build the wrong body entirely.
    sides = {obj["side"] for kind, obj in chosen
             if kind == "ring" and obj.get("side")}
    sides |= {obj["side"] for kind, obj in chosen
              if kind == "half" and obj.get("side")}
    if len(sides) > 1:
        raise SystemExit(f"--use mixes primitives from both sides of the boundary "
                         f"({', '.join(sorted(sides))}); a sketch belongs to one prism")
    if sides:
        direction = 1.0 if sides.pop() == "above" else -1.0
    else:
        room_up = min([m["value"] for m in marks if m["value"] > level + 1e-6] + [hi]) - level
        room_dn = level - max([m["value"] for m in marks if m["value"] < level - 1e-6] + [lo])
        direction = 1.0 if room_up >= room_dn else -1.0

    # The default snap measured on whichever side had more room; if --use named
    # the other prism, that provenance line would name a level we never read.
    side_name = "above" if direction > 0 else "below"
    if face is None:
        _, cut_at, _ = measure_offset(mesh, args.plane, args.at, marks,
                                      args.samples, side=side_name)

    if args.op == "loft" and getattr(args, "law", False):
        from mesh2cad.laws import at, loft_code
        named = outer + holes
        if len(named) != 1 or half_profiles or getattr(args, "_loft_levels", None):
            raise SystemExit("--law lofts ONE ring between --at and one --to level: "
                             "name a single ring in --use (an opening as a cutter with "
                             "--combine cut)")
        target = float(args.to)
        fits, zs, lw = _law_of(mesh, args.plane, named[0], level, target, ax)
        for line in _law_lines(named[0], fits, zs, lw, ax, level, target):
            print(line)
        code = loft_code(args.plane, sense, [(level, at(fits, lw, level)),
                                             (target, at(fits, lw, target))])
        detail = (f"from {ax}={level:.4f} to {ax}={target:.4f}, {fits[0]['family']} "
                  f"law of ring {named[0].get('name', '?')} ({len(zs)} levels), ruled")
        _finish_sketch(args, code, ax, level, cut_at, snapped, detail, "loft",
                       profiles=f"law of {named[0].get('name', '?')}")
        return

    if args.op == "loft" and getattr(args, "_loft_levels", None):
        # Three or more levels: one loft through every profile named, rather
        # than a chain of two-level lofts, which is a different solid -- each
        # pair would get its own blend and the joins would be creases.
        #
        # Deliberately simpler than the two-level path below. That one
        # extrapolates both profiles out to their named levels, which is exact
        # for a ruled loft between two sections and meaningless for a section
        # that turns on the way. Here the profiles are used AS MEASURED, a
        # little inside each level, and that is stated.
        if not outer:
            raise SystemExit("loft needs at least one outer profile in --use")
        levels = args._loft_levels
        rings_at, holes_at, cuts_at = [outer], [holes], [cut_at]
        near_rings = rings
        for lv in levels:
            _, lv_cut, _ = measure_offset(mesh, args.plane, lv, marks,
                                          args.samples,
                                          side="below" if lv > level else "above")
            _, rings, _, _ = _rings_at(mesh, args.plane, lv_cut, args.tol, args.grid)
            # Matched to the profiles the model NAMED, not counted against
            # every loop present. Demanding equal loop counts is the bug the
            # two-level path already fixed: naming one boss and finding three
            # loops at the next level is a true sentence about the section and
            # a false one about the feature.
            here = _match_profiles(outer, rings, "outer", ax, lv, level, near_rings)
            here_h = _match_profiles(holes, rings, "hole", ax, lv, level, near_rings)
            rings_at.append(here)
            holes_at.append(here_h)
            cuts_at.append(lv_cut)
        code, rings_at, holes_at = _chained_loft(
            args, sense, [level] + levels, rings_at, holes_at, ax, cuts_at)
        measured = ", ".join(f"{ax}={c:.4f}" for c in cuts_at)
        print(f"profiles used AS MEASURED at {measured}; a loft through more "
              f"than two levels does not extrapolate, because the section is "
              f"not assumed to change linearly between them")
        detail = (f"through {len(levels) + 1} levels "
                  + " -> ".join(f"{ax}={v:.4f}" for v in [level] + levels)
                  + f", {'smooth' if args.smooth else 'ruled'} blend, "
                  + f"{len(outer)} profile(s)"
                  + (f" and {len(holes)} hole(s)" if holes else ""))
        _finish_sketch(args, code, ax, level, cut_at, snapped, detail, "loft",
                       profiles=f"{sum(r['n_primitives'] for r in rings_at[0])} "
                                f"primitives per level, {len(levels)} levels")
        return

    if args.op == "loft":
        target = float(args.to)
        if not outer:
            raise SystemExit("loft needs at least one outer profile in --use")

        # Measure the far profile the same way the near one was measured:
        # a little inside the level, on the side facing this feature.
        _, far_cut, _ = measure_offset(mesh, args.plane, target, marks, args.samples,
                                    side="below" if target > level else "above")
        _, far_rings, _, _ = _rings_at(mesh, args.plane, far_cut, args.tol, args.grid)
        # Holes loft too (paired among the far holes, as outers among the far
        # outers). Taking only the outer rings here filled every bore the loft
        # crossed, and said nothing about having done so.
        far = _match_profiles(outer, far_rings, "outer", ax, target, level, rings)
        far_h = _match_profiles(holes, far_rings, "hole", ax, target, level, rings)
        if _twist_loft(args, mesh, outer, holes, far, ax, sense, level, target, cut_at,
                       far_cut, snapped):
            return

        near, near_h = outer, holes
        near, far, turns = _align_starts(near, far, ax, target)
        near_h, far_h, _ = _align_starts(near_h, far_h, ax, target)
        _even_turn(mesh, args, outer, turns, rings, ax, level, target, cut_at, far_cut)

        # Both profiles were measured a little inside their levels, because a
        # section taken exactly on a face is degenerate. For a prism that is
        # free; for a taper it is the whole error. A ruled loft is linear, so
        # the two measurements give both ends exactly -- see fit.blend_ring.
        span = far_cut - cut_at
        blended = None
        if abs(span) > 1e-9:
            t_near = (level - cut_at) / span
            t_far = (target - cut_at) / span
            trial = [[blend_ring(a, b, t) for a, b in zip(A, B, strict=True)]
                     for A, B, t in ((near, far, t_near), (near, far, t_far),
                                     (near_h, far_h, t_near), (near_h, far_h, t_far))]
            if all(r is not None for group in trial for r in group):
                blended = trial
        if blended is not None:
            near, far, near_h, far_h = blended
            print(f"profiles extrapolated from {ax}={cut_at:.4f} and "
                  f"{ax}={far_cut:.4f} out to the named levels "
                  f"{ax}={level:.4f} and {ax}={target:.4f}, which is exact for "
                  f"a ruled loft")
            # Extrapolation can only be trusted as far as the taper is really
            # linear, and the level it is aimed at may be an inferred plateau
            # edge rather than the true end of the taper, and extrapolating to it
            # can put the part outside its own bounding box.
            #
            # The bounding box is not an estimate -- it is measured, and exact.
            # Nothing may be built outside it. Clamping to it is therefore using
            # a hard fact to bound a soft one, not fudging a number.
            clamped = _clamp_to_bbox(near + far + near_h + far_h, mesh, args.plane)
            if clamped:
                print(f"clamped {clamped} extrapolated vertex/vertices back onto "
                      f"the part's bounding box, which is exact; the taper was "
                      f"reaching past the end of the part")
        else:
            print(f"WARNING: could not pair the two profiles primitive for "
                  f"primitive, so they are used AS MEASURED, at "
                  f"{ax}={cut_at:.4f} and {ax}={far_cut:.4f} rather than at the "
                  f"levels named. The taper will be short by that much at each "
                  f"end.")

        (code, step), (near, far), (near_h, far_h) = _buildable_loft(
            lambda R, H: build_loft(args.plane, sense, level, target,
                                    R[0], R[1], H[0], H[1],
                                    ruled=not args.smooth),
            [near, far], [near_h, far_h], ax, [cut_at, far_cut],
            smooth=args.smooth)
        detail = (f"from {ax}={level:.4f} to {ax}={target:.4f} "
                  f"(depth {abs(step):.4f} mm), {len(near)} profile(s)"
                  + (f" and {len(near_h)} hole(s)" if near_h else "")
                  + f", {'smooth' if args.smooth else 'ruled'} blend"
                  + f", far profile measured at {ax}={far_cut:.4f}")
        _finish_sketch(args, code, ax, level, cut_at, snapped, detail, "loft",
                       profiles=f"{sum(r['n_primitives'] for r in near + near_h)} + "
                                f"{sum(r['n_primitives'] for r in far + far_h)} primitives")
        return

    if args.op == "sweep":
        if not outer:
            raise SystemExit("sweep needs one outer profile in --use -- the "
                             "cross-section that travels along the path")
        if len(outer) != 1:
            raise SystemExit(f"sweep takes exactly one profile; --use named "
                             f"{len(outer)}"
                             + (f". Two walls of a tube cut along it are not its profile: "
                                f"sweep its end face (--face F#, with the bore) or a "
                                f"section across it along the tube's own path, which "
                                f"`faces` lists as a pipe path W#. W# here: "
                                f"{_known_paths(args, mesh)}"
                                if (args.along or "").startswith("W") else ""))
        if (args.along or "").startswith("W"):
            _sweep_along(args, mesh, rings, outer[0], holes, spec, ax, sense, level,
                         cut_at, snapped)
            return
        if not re.fullmatch(r"H\d+", (args.along or "H0").split("@")[0]):
            raise SystemExit(f"unknown path {args.along!r}; the named paths are H0 "
                             f"(H1, ... on other bodies), the helices `overview` fits, "
                             f"and W#: {_known_paths(args, mesh)}")
        if getattr(args, "between", None) or not any(
                s_["kind"] == "arc" for s_ in outer[0]["segments"]) or \
                (args.along or "H0").split("@")[0] != "H0" or \
                _helix_fit(mesh).get("axis", "Z") != "Z":
            # The coil below is built about Z only; along X or Y, and for any
            # profile that is not a round wire, the general path.
            _sweep_profile(args, mesh, outer[0], level, spec, ax, cut_at, snapped, marks)
            return
        helix = _helix_fit(mesh)
        if not helix.get("is_helix"):
            raise SystemExit(
                "no helix was fitted on this mesh, so H0 does not exist here. "
                + (f"({helix.get('reason')})" if helix.get("reason") else ""))
        # H0@ends is the whole-part regression, which is the one that includes
        # the tighter end turns; H0 and H0@body are the central fit.
        want = (args.along or "H0").split("@")[1:] or ["body"]
        if want[0] not in ("body", "ends"):
            raise SystemExit(f"H0 has @body and @ends; {want[0]!r} is neither")
        pitch = (helix["pitch_whole_part"] if want[0] == "ends"
                 else helix["pitch"])
        # The wire's own radius and its distance from the axis, both read off
        # the ring the model named. The section fits that arc to microns; the
        # helix detector's slice centroids put the same coil radius 15% out,
        # so the better instrument wins even though both could answer.
        segs = outer[0]["segments"]
        arcs = [s for s in segs if s["kind"] == "arc"]
        if not arcs:
            raise SystemExit("sweep wants a round profile; the ring named in "
                             "--use has no arc to take a radius from")
        _one_circle(arcs, outer[0].get("name") or outer[0]["segments"][0]["name"])
        wire_r = float(np.mean([s["radius"] for s in arcs]))
        spec = PLANES[args.plane]
        axis_idx = "XYZ".index(helix["axis"])
        # A section's points are (u, v). Whichever of those runs along the helix
        # axis gives the ring's height; the other gives its distance from the
        # axis, which is the coil radius.
        i_along = 0 if spec["u"] == axis_idx else 1
        i_out = 1 - i_along
        world_out = spec["v"] if i_out else spec["u"]
        centre_out = float(mesh.bounds.mean(axis=0)[world_out])

        out = float(np.mean([s["centre"][i_out] for s in arcs]))
        along = float(np.mean([s["centre"][i_along] for s in arcs]))
        coil_r = abs(out - centre_out)
        # Height and base from the mesh, not from turns x pitch: the turn count
        # is fitted on single-loop slices, which deliberately skip the ground
        # ends, so the product overshot a 47 mm spring by 10 mm.
        base = float(mesh.bounds[0][axis_idx])
        height = float(mesh.bounds[1][axis_idx]) - base
        # Where the coil starts. The named ring sits at a known height on the
        # axis and a known angle about it, so it pins the turn it belongs to:
        # take off the rotation the helix has already accumulated by that
        # height, and what remains is the phase at the base.
        here = 0.0 if out >= centre_out else 180.0
        turned = 360.0 * (along - base) / pitch
        if helix.get("handedness", "right") != "right":
            turned = -turned
        phase = (here - turned) % 360.0
        code, height = build_helical_sweep(pitch, height, base, coil_r, wire_r,
                                           helix.get("handedness", "right"),
                                           phase)
        detail = (f"H0@{want[0]} about {helix['axis']}: pitch {pitch:g} mm, "
                  f"coil radius {coil_r:.4f}, wire radius {wire_r:.4f}, "
                  f"{height:.4f} mm tall, {helix['handedness']}-handed")
        _finish_sketch(args, code, ax, level, cut_at, snapped, detail, "sweep",
                       profiles=f"outer={[outer[0]['segments'][0]['name']]}")
        return

    if args.op == "extrude":
        target = float(args.to)
        if not outer:
            raise SystemExit("extrude needs at least one outer profile in --use")
        taper = None
        if getattr(args, "taper", False):
            taper = _taper_measure(args, mesh, spec, ax, outer, holes, level, cut_at,
                                   target, marks)
        # --both was wired into the --split path and silently dropped here, so
        # `--use R0 --both` built a one-sided prism and reported it as one. The
        # same shape of bug as --split quietly ignoring --op extrude: a flag
        # accepted, ignored, and not mentioned again.
        span = _both_span(args, ax, level, target) if args.both else None
        if getattr(args, "combine", "add") == "cut" and not taper:
            depth = abs(target - level)
            outer = _past_the_wall(args, outer, *(
                sorted(span[:2]) if span else (level - depth, level + depth) if args.both
                else sorted((level, target))))
        if span:
            start, stop, note = span
            print(note)
            code, dist = build_extrude(args.plane, sense, start, stop, outer, holes)
            detail = (f"{ax}={min(start, stop):.4f} to {ax}={max(start, stop):.4f} "
                      f"(both sides of {ax}={level:.4f}, {abs(dist):.4f} mm total, "
                      f"far end on a named level)")
        else:
            code, dist = build_extrude(args.plane, sense, level, target, outer, holes,
                                       both=args.both, taper=taper)
            detail = (f"{ax}={level - abs(dist):.4f} to {ax}={level + abs(dist):.4f} "
                      f"(both sides, {2 * abs(dist):.4f} mm total)" if args.both else
                      f"from {ax}={level:.4f} to {ax}={target:.4f}  "
                      f"(depth {abs(target - level):.4f} mm)"
                      + (f", drafted {taper:g} deg" if taper else ""))
        if getattr(args, "combine", "add") == "cut":
            a, b = (sorted(span[:2]) if span else
                    (level - abs(dist), level + abs(dist)) if args.both
                    else sorted((level, target)))
            # On a residual export everything but what is left is air, so beyond
            # the levels is judged on the target it came from -- else a cutter
            # runs past the levels into the wall.
            judge = None if getattr(args, "_frame_info", None) else getattr(
                args, "_check_mesh", None)
            a2, b2, notes = _through_ends(judge if judge is not None else mesh, args.plane,
                                          outer, a, b)
            if notes and taper:
                # A drafted cutter keeps its draft between the named levels; the
                # run past the open face is a straight prism of the start profile,
                # outside the part, unioned on. Rebuilding the whole cutter from
                # the new start would drop the taper.
                for n in notes:
                    print(n)
                lo_l, hi_l = sorted((level, target))
                ext = ((a2, lo_l) if a2 < lo_l - 1e-9 and abs(level - lo_l) < 1e-9 else
                       (hi_l, b2) if b2 > hi_l + 1e-9 and abs(level - hi_l) < 1e-9 else None)
                if (a2 < lo_l - 1e-9 and abs(target - lo_l) < 1e-9) or \
                        (b2 > hi_l + 1e-9 and abs(target - hi_l) < 1e-9):
                    print("note: --taper: the far end also runs out of the part; a "
                          "drafted cutter is not extended there, it stops at --to")
                if ext is not None:
                    start, stop = (ext if level == ext[1] else ext[::-1])
                    ecode, _ = build_extrude(args.plane, sense, start, stop, outer, holes)
                    code = (code + "\n_draft = result\n" + ecode
                            + "\nresult = _draft.union(result)\n_STEPS = [(\"add\", result)]")
                    detail += (f"; straight run past the open face {ax}={min(ext):.4f} to "
                               f"{ax}={max(ext):.4f} (through)")
            elif notes:
                for n in notes:
                    print(n)
                # one prism from the new start to the new end, sketched there
                start, stop = (a2, b2) if (args.both or target >= level) else (b2, a2)
                code, _ = build_extrude(args.plane, sense, start, stop, outer, holes)
                detail += f"; cutter runs {ax}={a2:.4f} to {ax}={b2:.4f} (through)"
    else:
        if outer or holes:
            if half_profiles:
                raise SystemExit("--use mixes half arcs and rings: a revolve takes one "
                                 "half arc (about its own chord) or closed rings with "
                                 "--axis, not both")
            return _revolve_rings(args, mesh, spec, ax, level, cut_at, snapped, sense,
                                  outer, holes)
        if len(half_profiles) != 1:
            raise SystemExit("revolve needs one half-arc profile (--use A0_0@upper), or "
                             "closed rings (--use R0,R1) with --axis")
        if getattr(args, "_revolve_to", None):
            print(f"note: --to {args._revolve_to} is not used: a half arc revolves "
                  f"--angle about its own chord; rings with --axis end at --to F#")
        # R2: it drops, it says so. This path revolves a half arc about its own
        # chord and has no way to honour --axis; accepting the flag and ignoring
        # it would give a revolve about something else and a residual the model
        # cannot explain.
        if args.axis:
            raise SystemExit(
                "a --use half-arc always revolves about its own chord, so "
                f"--axis {args.axis.strip()} cannot be honoured here. To revolve "
                "about a named axis, cut the profile with --split and pass "
                "--axis with it.")
        code, ang = build_revolve(args.plane, sense, level, half_profiles[0], args.angle)
        detail = f"{ang:g} deg about {half_profiles[0]['revolve_axis']}"

    _finish_sketch(args, code, ax, level, cut_at, snapped, detail, args.op,
                   profiles=(f"outer={[r['segments'][0]['name'] for r in outer]}  "
                             f"holes={[r['segments'][0]['name'] for r in holes]}  "
                             f"halves={[h['name'] for h in half_profiles]}"))


def _revolve_rings(args, mesh, spec, ax, level, cut_at, snapped, sense, outer, holes):
    """Closed rings -- outlines with their holes -- revolved about an axis in
    the sketch plane that crosses none of them, by --angle or to --to F#.

    The plain case is a half-profile ring on one side of the part's axis; the
    other is an annulus about an axis off the profile -- a pipe bend, a torus
    rim -- which cannot be built otherwise without filling its bore."""
    named = ", ".join(r["name"] for r in outer + holes)
    if not args.axis:
        raise SystemExit(f"--op revolve of rings ({named}) needs --axis: a round face "
                         f"F# or a symmetry axis S# that lies in this plane, or two "
                         f'coordinates ("{ax}={level:g},<{spec["u_name"]} or '
                         f'{spec["v_name"]}>=..."). A half arc (A0_0@upper) revolves '
                         f"about its own chord instead")
    if not outer:
        raise SystemExit(f"revolve needs an outline in --use; {named} "
                         f"{'is a hole' if len(holes) == 1 else 'are holes'} of the "
                         f"section, revolved and cut from the outline they sit in")
    axis = parse_axis(args.axis, spec, level, mesh=mesh, args=args)
    note = args.axis.strip()
    for r in outer + holes:
        g = readback.shape(r)
        if g is not None and _axis_crosses([g], axis):
            u = spec["u_name"] if abs(axis[0][0] - axis[1][0]) < 1e-12 else spec["v_name"]
            at = axis[0][0] if u == spec["u_name"] else axis[0][1]
            raise SystemExit(
                f"--axis {note} runs through ring {readback.describe(r)}: revolved "
                f"about it, the ring would sweep through itself and cannot be built. "
                f'Cut the section at the axis and keep one side: --split "{u}={at:g}" '
                f"--keep + or -")
    angle = args.angle
    if getattr(args, "_revolve_to", None):
        if abs(args.angle - 360.0) > 1e-9:
            raise SystemExit(f"--to {args._revolve_to} and --angle {args.angle:g} both say "
                             f"where the revolve ends; give one")
        angle, end_note = _revolve_end(args, mesh, spec, level, axis, note, outer, holes,
                                       args._revolve_to)
        print(end_note)
    code, ang = build_revolve_profile(args.plane, sense, level, outer, holes, axis, angle)
    detail = (f"{ang:g} deg about {note}, rings {named}"
              + (" (holes revolved and cut: kept as holes)" if holes else ""))
    _finish_sketch(args, code, ax, level, cut_at, snapped, detail, "revolve",
                   profiles=(f"outer={[r['name'] for r in outer]}  "
                             f"holes={[r['name'] for r in holes]}"))


REVOLVE_SAMPLES = 16      # points along a candidate sweep tested against the part


def _revolve_end(args, mesh, spec, level, axis, axis_note, outer, holes, to):
    """(signed angle, note) for a revolve that ends at face `to`: where the
    profile plane, turned about the axis, comes square to it -- a flat face's
    normal, or the axis of the cylinder or cone that carries on from the bend.

    Four turns do that (the face is met at two angles, either way round). The
    one taken is the longest whose sweep keeps the profile inside the part --
    measured, so the model never types the sign, which is easy to get wrong."""
    from shapely.ops import unary_union

    if not re.fullmatch(r"[Ff]\d+", to):
        raise SystemExit(f"--to {to}: a revolve ends at a face F# -- the flat end face, "
                         f"or the cylinder that carries on from the bend -- or give "
                         f"--angle in degrees")
    seg = _faces_of(mesh, args)
    f = seg.by_name(to.upper())
    if f is None:
        raise SystemExit(f"--to {to}: no such face; `faces` names F0..F{len(seg.surfaces) - 1}")
    if f.kind == "plane":
        t = np.asarray(f.params["normal"], float)
    elif f.kind in ("cylinder", "cone") and "axis" in f.params:
        t = np.asarray(f.params["axis"], float)
    else:
        raise SystemExit(f"--to {to} is a {f.kind}: it has no end plane for a revolve to "
                         f"stop on. Name the flat face it ends on or the cylinder after "
                         f"it, or give --angle")
    k = spec["axis"]
    iu, iv = "XYZ".index(spec["u_name"]), "XYZ".index(spec["v_name"])

    def world(uv):
        w = np.zeros(3)
        w[iu], w[iv], w[k] = uv[0], uv[1], level
        return w
    a0, a1 = world(axis[0]), world(axis[1])
    A = (a1 - a0) / np.linalg.norm(a1 - a0)
    n0 = np.zeros(3)
    n0[k] = 1.0
    t = t / np.linalg.norm(t)
    tp = t - (t @ A) * A
    if np.linalg.norm(tp) < 1e-6:
        raise SystemExit(f"--to {to}: {to} is parallel to the axis {axis_note}, so no turn "
                         f"about it comes square to {to}; give --angle")
    base = float(np.degrees(np.arctan2(tp @ np.cross(A, n0), tp @ n0))) % 180.0 or 180.0
    turns = [c for c in (base, base + 180.0, base - 180.0, base - 360.0)
             if 1e-6 < abs(c) < 360.0 - 1e-6]
    body = unary_union([readback.shape(r) for r in outer if readback.shape(r) is not None])
    cut = [readback.shape(r) for r in holes if readback.shape(r) is not None]
    if cut:
        body = body.difference(unary_union(cut))
    q = world(body.representative_point().coords[0])

    def turned(theta):
        th = np.radians(theta)
        v = q - a0
        return a0 + v * np.cos(th) + np.cross(A, v) * np.sin(th) + A * (A @ v) * (1 - np.cos(th))
    pts = np.array([turned(c * (i + 0.5) / REVOLVE_SAMPLES)
                    for c in turns for i in range(REVOLVE_SAMPLES)])
    inside = _inside_target(mesh, pts).reshape(len(turns), REVOLVE_SAMPLES).mean(axis=1)
    whole = [c for c, s in zip(turns, inside, strict=True) if s >= 0.95]
    others = ", ".join(f"{c:+.1f} deg {s:.0%} inside" for c, s in zip(turns, inside, strict=True))
    if not whole:
        raise SystemExit(f"--to {to}: no way round the axis {axis_note} to {to} keeps the "
                         f"profile inside the part ({others}); give --angle")
    best = max(whole, key=abs)
    return best, (f"--to {to}: the profile plane comes square to {to} ({f.kind}) at "
                  f"{best:+.3f} deg about {axis_note}, the longest turn that stays "
                  f"inside the part ({others})")


def _inside_target(mesh, pts):
    """Whether each point lies inside the target (by signed distance when the
    target is not closed, as `residual` measures one)."""
    if mesh.is_watertight:
        return np.asarray(mesh.contains(pts), float)
    import trimesh
    return (trimesh.proximity.signed_distance(mesh, pts) > 0).astype(float)


# How far a through-cut runs past the face it exits, as a share of the part's
# extent along the cut. Plates built slightly thicker than the target's, with
# slot cutters stopping exactly on the target's faces, leave a skin sealing
# every slot -- invisible to IoU and the residual, obvious in the picture.
THROUGH_MARGIN = 0.01


def _ring_polygon(ring):
    """A ring as a shapely polygon in (u, v), arcs sampled."""
    from shapely.geometry import Polygon

    from mesh2cad.fit import arc_points
    pts = []
    for prim in ring.get("_prims") or []:
        if prim.kind == "line":
            pts.append(tuple(prim.p0))
        else:
            pts.extend(map(tuple, arc_points(prim, 24)[:-1]))
    if len(pts) < 3:
        pts = [tuple(sg["from_xy"]) for sg in ring["segments"]]
    poly = Polygon(pts)
    return poly if poly.is_valid else poly.buffer(0)


# A prism cutter's edge within CUT_WALL_TAU of the build's wall, with air
# beyond, is carried CUT_WALL_PAST past it. Cutters come from measured faces
# and end on the wall within fit error, leaving skins that join a window's
# floor and ceiling (handles), needles (extra solids) or a comb.
CUT_WALL_TAU = 0.06
CUT_WALL_PAST = 0.08


def _past_the_wall(args, outer, lo, hi):
    """`outer` with every edge that lies on the --base build's wall (within
    CUT_WALL_TAU, the build's air beyond it) carried CUT_WALL_PAST past it, each
    ring still one outline, refitted so its arcs stay arcs. The output says how
    many of a ring's edges moved. Without a --base, in a --frame, or where the
    grown outline isn't one ring, the cutter is left as it was."""
    from shapely.geometry import LineString, Point, Polygon
    from shapely.ops import unary_union
    base = getattr(args, "base", None)
    if not base or not Path(base).is_file() or getattr(args, "_frame_info", None):
        return outer
    try:
        built = _load_built(base)
        body = unary_union(section_at(built, args.plane, 0.5 * (lo + hi))).buffer(0)
    except Exception:                                              # noqa: BLE001
        return outer
    if body.is_empty:
        return outer
    wall, out = body.boundary, []
    for r in outer:
        pts, owner = [], []
        segs = r.get("segments", [])
        for i, s in enumerate(segs):
            sub = [tuple(map(float, s["from_xy"]))]
            if s.get("kind") == "arc" and s.get("centre") is not None:
                c, rad = np.asarray(s["centre"], float), float(s["radius"])
                a0 = np.arctan2(sub[0][1] - c[1], sub[0][0] - c[0])
                deg = 360.0 if len(segs) == 1 else float(s.get("sweep_deg", 0.0))
                sweep = np.radians(deg) * (1.0 if s.get("ccw", True) else -1.0)
                k = max(4, int(np.ceil(abs(deg) / 2.5)))
                sub += [(float(c[0] + rad * np.cos(a0 + sweep * t)),
                         float(c[1] + rad * np.sin(a0 + sweep * t)))
                        for t in np.linspace(0.0, 1.0, k + 1)[1:-1]]
            pts += sub
            owner += [i] * len(sub)
        poly = Polygon(pts)
        if len(pts) < 3 or not poly.is_valid or poly.area <= 0:
            out.append(r)
            continue
        ccw = poly.exterior.is_ccw
        A = np.asarray(pts + [pts[0]], float)
        on, moved = [], set()
        for j, (a, b) in enumerate(zip(A[:-1], A[1:], strict=True)):
            d = b - a
            L = float(np.hypot(*d))
            if L < 1e-9:
                on.append(False)
                continue
            n = (np.array([d[1], -d[0]]) if ccw else np.array([-d[1], d[0]])) / L
            mid = 0.5 * (a + b)
            near = max(wall.distance(Point(*p)) for p in (a, mid, b)) <= CUT_WALL_TAU
            air = (not body.contains(Point(*(mid + n * CUT_WALL_TAU * 1.05)))
                   and not poly.contains(Point(*(mid + n * CUT_WALL_PAST))))
            on.append(bool(near and air))
            if on[-1]:
                moved.add(owner[j])
        if not moved:
            out.append(r)
            continue
        chains, cur = [], []
        for j, hit in enumerate(on):
            if hit:
                cur = cur or [tuple(A[j])]
                cur.append(tuple(A[j + 1]))
            elif cur:
                chains.append(cur)
                cur = []
        if cur:
            chains.append(cur)
        grown = poly
        for c in chains:
            grown = grown.union(LineString(c).buffer(
                -CUT_WALL_PAST if ccw else CUT_WALL_PAST, single_sided=True,
                join_style=2).buffer(1e-6, join_style=2))
        grown = grown.buffer(0)
        name = r.get("name", "?")
        if grown.geom_type != "Polygon" or len(grown.interiors):
            print(f"note: {name}: {len(moved)} of {len(r['segments'])} edges lie on the "
                  f"build's wall with air beyond, but carried past it the outline is no "
                  f"longer one ring, so the cutter is as measured")
            out.append(r)
            continue
        fit = decompose_polygon(grown.simplify(1e-4), tol=r.get("tol"), grid=args.grid)
        new = {**fit[0], "name": name, "role": "outer"}
        print(f"note: {name}: {len(moved)} of {len(r['segments'])} edges lie on the build's "
              f"wall (within {CUT_WALL_TAU:g} mm, air beyond): carried {CUT_WALL_PAST:g} mm "
              f"past it, still one outline, so no skin of the wall is left over the cut")
        out.append(new)
    return out


def _through_ends(mesh, plane, outer, lo, hi, profile=None, ends=(True, True)):
    """Run a cutter's ends past faces where the target is air beyond.

    For each end, look at the target just outside it, inside the cutter's
    outline. Air there means the hole exits the part at that face, so the
    cutter may run on -- up to THROUGH_MARGIN of the part's extent, and only as
    far as the target stays air. Material there (a blind hole, a pocket floor)
    leaves the end where it was named. Returns (lo, hi, notes).
    """
    from shapely.ops import unary_union

    from mesh2cad.section import CutError
    spec = PLANES[plane]
    k = spec["axis"]
    span = float(np.ptp(mesh.bounds[:, k])) or 1.0
    margin, eps = THROUGH_MARGIN * span, 1e-3 * span
    if profile is None:
        profile = unary_union([_ring_polygon(r) for r in outer])
    if profile.is_empty or profile.area <= 0:
        return lo, hi, []

    def air(at):
        try:
            cut = section_at(mesh, plane, at)
        except CutError:
            return False
        if not cut:
            return True
        return unary_union(cut).intersection(profile).area < 0.02 * profile.area

    notes, out = [], []
    ax = "XYZ"[k]
    for end, sign, check in ((lo, -1.0, ends[0]), (hi, 1.0, ends[1])):
        if not check:
            out.append(end)
            continue
        reach = 0.0
        for step in (eps, 0.25 * margin, 0.5 * margin, 0.75 * margin, margin):
            if not air(end + sign * step):
                break
            reach = step
        if reach >= 0.25 * margin:
            notes.append(f"through-cut: the target is open beyond {ax}={end:.4f} "
                         f"inside this outline, so the cutter runs on to "
                         f"{ax}={end + sign * reach:.4f} ({reach:.3f} mm past it), "
                         f"through any skin the build has there")
            end = end + sign * reach
        out.append(end)
    return out[0], out[1], notes


def parse_pattern(expr):
    """"5@Y=0,Z=0" -> (5, p0, p1): N copies evenly about that world axis.

    The `@` reads the same way as it does on a half arc (`A0_0@X+`): the thing,
    then the qualifier. The axis is written in world terms, exactly as --axis
    takes it, so one notation covers revolve and pattern rather than two.
    """
    m = re.match(r"\s*(\d+)\s*@\s*(.+?)\s*$", expr)
    if not m:
        raise SystemExit('--pattern wants "N@AXIS", e.g. --pattern "5@Y=0,Z=0"; '
                         f'got {expr!r}')
    n = int(m.group(1))
    if n < 2:
        raise SystemExit(f"--pattern needs at least 2 copies; got {n}")

    names = "XYZ"
    fixed = {}
    for part in m.group(2).split(","):
        k, _, v = part.partition("=")
        k = k.strip().upper()
        if k not in names:
            raise SystemExit(f"--pattern axis: {k!r} is not one of X, Y, Z")
        fixed[names.index(k)] = float(v)
    if len(fixed) != 2:
        raise SystemExit('--pattern axis needs two constraints, e.g. "Y=0,Z=0"')

    free = next(i for i in range(3) if i not in fixed)
    p0 = [0.0, 0.0, 0.0]
    for i, v in fixed.items():
        p0[i] = v
    p1 = list(p0)
    p1[free] += 1.0
    return n, tuple(p0), tuple(p1)


MIRROR_PLANES = {0: "YZ", 1: "XZ", 2: "XY"}


def mirror_code(k, coord):
    """The feature just built, unioned with its image in the plane normal to
    axis `k` at `coord`. A cutter is mirrored the same way, so a pair of holes
    is one cut, and a symmetric plate is built once per mirror pair."""
    base = [0.0, 0.0, 0.0]
    base[k] = round(float(coord), 6)
    plane = MIRROR_PLANES[k]
    return (f"\n_one = result\n"
            f"_one_steps = list(_STEPS)\n"
            f"result = result.union(_one.mirror({plane!r}, {tuple(base)!r}))\n"
            f"_STEPS = _STEPS + [(_op, _s.mirror({plane!r}, {tuple(base)!r})) "
            f"for _op, _s in _one_steps]")


def linear_code(n, step):
    """N copies of the feature just built, each `step` on from the last."""
    v = tuple(round(float(x), 6) for x in step)
    return (f"\n_one = result\n"
            f"_one_steps = list(_STEPS)\n"
            f"for _k in range(1, {n}):\n"
            f"    _v = ({v[0]} * _k, {v[1]} * _k, {v[2]} * _k)\n"
            f"    result = result.union(_one.translate(_v))\n"
            f"    _STEPS = _STEPS + [(_op, _s.translate(_v)) for _op, _s in _one_steps]")


def _levels_on(args, mesh, k):
    """The named levels along world axis `k`, in the names `section` prints."""
    plane = next(p for p, sp in PLANES.items() if sp["axis"] == k)
    marks, _ = feature_levels(mesh, plane, getattr(args, "samples", 900) or 900,
                              cache_dir=_cache_dir(args))
    return marks, plane


def _axis_and_rest(expr, default_axis, what):
    """("X", "B1..B4") from "X:B1..B4", or the default axis when none is named.
    "Z=0" and "Z:Z=0" are the plane Z = 0 (63: both refused, the mirror never
    applied)."""
    m = re.fullmatch(r"\s*([XYZxyz])\s*(?::\s*\1\s*)?=\s*(.+?)\s*", expr, re.I)
    if m:
        return "XYZ".index(m.group(1).upper()), m.group(2)
    m = re.fullmatch(r"\s*([XYZxyz])\s*:\s*(.+?)\s*", expr)
    if m:
        return "XYZ".index(m.group(1).upper()), m.group(2)
    if default_axis is None:
        raise SystemExit(f"{what} {expr!r}: name the axis, e.g. {what} \"X:{expr}\" "
                         f"-- this command has no sketch plane to take it from")
    return default_axis, expr.strip()


def _mirror_plane(args, mesh, expr, default_axis=None):
    """(axis index, coordinate, label) for `--mirror`: a mirror plane `symmetry`
    names (S#), or a level -- M#, any level, mid(A,B) -- on the sketch's axis
    or on the axis named before it ("Y:M0")."""
    if re.fullmatch(r"S\d+", expr.strip()):
        from mesh2cad.contrib.instruments.symmetry import candidates
        named, _near, _tol, _n = candidates(mesh)
        k = int(expr.strip()[1:])
        if k >= len(named):
            raise SystemExit(f"--mirror {expr}: `symmetry` names S0..S{len(named) - 1} "
                             f"on this part" if named else
                             f"--mirror {expr}: `symmetry` finds no mirror plane here")
        c = named[k]
        if c["kind"] != "mirror plane":
            raise SystemExit(f"--mirror {expr}: S{k} is a {c['kind']} ({c['about']}), "
                             f"not a mirror plane")
        return c["axis"], c["coord"], f"{expr.strip()} ({c['about']})"
    k, name = _axis_and_rest(expr, default_axis, "--mirror")
    marks, plane = _levels_on(args, mesh, k)
    coord = resolve_level(name, marks, "--mirror", mesh, plane)
    return k, coord, f"{name} ({'XYZ'[k]} = {coord:.4f})"


def _linear_step(args, mesh, expr, default_axis=None):
    """(n, step vector, label) for `--pattern "N@A..B"`: N copies, the last
    moved by the distance from level A to level B ("X:A..B" for another axis)."""
    m = re.fullmatch(r"\s*(\d+)\s*@\s*(.+?)\s*", expr)
    n = int(m.group(1))
    if n < 2:
        raise SystemExit(f"--pattern needs at least 2 copies; got {n}")
    k, rest = _axis_and_rest(m.group(2), default_axis, "--pattern")
    a, _, b = rest.partition("..")
    marks, plane = _levels_on(args, mesh, k)
    za = resolve_level(a.strip(), marks, "--pattern", mesh, plane)
    zb = resolve_level(b.strip(), marks, "--pattern", mesh, plane)
    step = [0.0, 0.0, 0.0]
    step[k] = (zb - za) / (n - 1)
    return n, step, (f"x{n} along {'XYZ'[k]} from {a.strip()} to {b.strip()} "
                     f"({step[k]:+.4f} each)")


def pattern_code(n, p0, p1):
    """Wrap the feature just built into N copies about an axis.

    Emitted as a loop rather than N unrolled unions. Five spokes written out
    five times is five chances for one of them to differ, and the loop says what
    the part actually is -- one feature repeated -- which is almost certainly
    what the original program said too.
    """
    step = 360.0 / n
    return (f"\n_one = result\n"
            f"_one_steps = list(_STEPS)\n"
            f"for _k in range(1, {n}):\n"
            f"    result = result.union(_one.rotate({p0!r}, {p1!r}, "
            f"_k * {step:.6f}))\n"
            f"    _STEPS = _STEPS + [(_op, _s.rotate({p0!r}, {p1!r}, "
            f"_k * {step:.6f})) for _op, _s in _one_steps]")


def _clamp_to_bbox(rings, mesh, plane):
    """Pull any vertex outside the mesh's bounding box back onto it.

    Returns how many vertices moved. Only ever called on EXTRAPOLATED profiles:
    a measured profile is inside the part by construction, so if this fires on
    one, something upstream is wrong and the count says so.
    """
    spec = PLANES[plane]
    lo = [float(mesh.bounds[0][spec["u"]]), float(mesh.bounds[0][spec["v"]])]
    hi = [float(mesh.bounds[1][spec["u"]]), float(mesh.bounds[1][spec["v"]])]
    moved = 0
    for r in rings:
        for s in r["segments"]:
            for key in ("from_xy", "to_xy"):
                p = list(s[key])
                q = [min(max(p[i], lo[i]), hi[i]) for i in (0, 1)]
                if q != p:
                    s[key] = tuple(q)
                    moved += 1
    return moved


def _finish_sketch(args, code, ax, level, cut_at, snapped, detail, op,
                   profiles=None, pre=None):
    """`pre`: a whole feature (header, code and glue) to put between the base and
    this one -- a concave blend's cut-back (blends.trims)."""
    # Built in a frame: turn it back first, so a --pattern about a world axis
    # and the glue onto the base both see it in the target's frame.
    frame = getattr(args, "_frame_info", None)
    frame_line = ""
    if frame:
        from mesh2cad.frames import AXES, back_code
        f_name, fr, what = frame
        code = code + back_code(fr, f_name)
        op = f"{op} in frame {f_name}"
        frame_line = (f"\n# frame: built with {what} on "
                      f"{'+' if fr['sign'] > 0 else '-'}{AXES[fr['k']]}, coordinates "
                      f"above in that frame; turned back {fr['angle']:.4f} deg")
    # Pattern the new feature BEFORE unioning onto the base, so the repeat
    # applies to what was just built and not to everything built so far.
    if getattr(args, "pattern", None):
        if ".." in args.pattern:
            n, step, label = _linear_step(args, args._target_mesh, args.pattern,
                                          PLANES[args.plane]["axis"])
            code = code + linear_code(n, step)
            op = f"{op} {label}"
        else:
            n, p0, p1 = parse_pattern(args.pattern)
            code = code + pattern_code(n, p0, p1)
            op = f"{op} x{n} patterned"
    if getattr(args, "mirror", None):
        k, coord, label = _mirror_plane(args, args._target_mesh, args.mirror,
                                        PLANES[args.plane]["axis"])
        code = code + mirror_code(k, coord)
        op = f"{op}, mirrored about {label}"

    header = (f"# sketch on {args.plane} at {ax} = {level:.4f}"
              + (f"  ({snapped})" if snapped else "")
              + f"\n# profile measured at {ax} = {cut_at:.4f}"
              + f"\n# {op}: {detail}" + frame_line)
    combine = getattr(args, "combine", "add") or "add"
    into = getattr(args, "with_feature", None)
    if into is not None:
        if combine != "intersect":
            raise SystemExit("--with N goes with --combine intersect: it names the "
                             "one feature the new one clips")
        if not args.base:
            raise SystemExit("--with N needs --base PROG: the program whose feature "
                             "#N the new one clips")
        if getattr(args, "replace", None) is not None:
            raise SystemExit("--with N and --replace N both rewrite one feature; "
                             "give one of them")
        try:
            feats, frame_src = split_program(Path(args.base).read_text())
            if not 0 <= into < len(feats):
                raise ProgramError(f"there is no feature #{into}: the program has "
                                   f"{len(feats)} (#0..#{len(feats) - 1})")
        except ProgramError as exc:
            raise SystemExit(f"--with: {exc}") from None
        tool = header + f"\n# combine: intersect with feature #{into} only\n" + code
        feats[into] = intersect_into(feats[into], tool)
        full = join_program(feats, frame_src or frame_block(getattr(args, "_frame", None)).strip())
        op = f"{op}, intersected with feature #{into} only"
        print(f"SKETCH PLANE: {ax} = {level:.4f}")
        print(f"operation:    {op}  {detail}   [feature #{into} becomes #{into} "
              f"intersected with this; the rest is unchanged]")
        if profiles:
            print(f"profiles:     {profiles}")
        _show_program(args, full)
        _save_and_check(args, full, f"sketch {op}", feature=into)
        return
    if combine != "add":
        if not args.base:
            raise SystemExit(f"--combine {combine} needs --base PROG: the "
                             f"program the new feature is "
                             + {"cut": "cut from", "intersect": "intersected with",
                                "separate": "put beside"}[combine])
        header += {"cut": "\n# combine: cut (subtracted from everything before it)",
                   "intersect": "\n# combine: intersect (keeps only what lies "
                                "inside it)",
                   "separate": "\n# combine: separate (a body of its own, not "
                               "fused into the rest)"}[combine]
    replace = getattr(args, "replace", None)
    if replace is not None and not args.base:
        raise SystemExit("--replace N needs --base PROG: the program whose "
                         "feature #N the new one takes the place of")
    if replace is not None:
        # The new feature, standalone, takes the place of #N; every other
        # feature and the glue between them stay exactly as they were written.
        try:
            full = replace_feature(Path(args.base).read_text(), replace,
                                   header + "\n" + code)
        except ProgramError as exc:
            raise SystemExit(f"--replace: {exc}") from None
    else:
        prev = ""
        if args.base:
            src = Path(args.base).read_text()
            src = src.replace("import cadquery as cq\n", "")
            src = src.replace("show_object(result)", "").rstrip()
            # _FRAME is a property of the target, not of a feature, so it is
            # written once at the end. Inlining a base that already carries one
            # would otherwise leave two, and the second silently wins.
            src = without_slivers(FRAME_BLOCK.sub("", src)).rstrip()
            prev = src + "\n_prev = result\n_PREV_STEPS = list(_STEPS)\n\n" + (pre or "")
            code = code + "\n" + GLUES[combine]
        full = ("import cadquery as cq\n\n" + prev + header + "\n" + code
                + frame_block(getattr(args, "_frame", None))
                + "\n\nshow_object(result)\n")

    print(f"SKETCH PLANE: {ax} = {level:.4f}"
          + (f"   (snapped to the {snapped})" if snapped else "")
          + (f"   in the frame of {frame[0]}, turned back when written"
             if frame else ""))
    if abs(cut_at - level) > 1e-9:
        # Where the profile was taken, said where the model reads it, not only
        # in an emitted comment: a profile measured slightly off the plane named
        # (in a bend, say) builds too much.
        print(f"MEASURED AT:  {ax} = {cut_at:.4f}   ({abs(cut_at - level):.4f} "
              f"{'above' if cut_at > level else 'below'} the sketch plane, inside "
              f"the material: the profile is the section there)")
    print(f"operation:    {'' if combine == 'add' else combine + ': '}{op}  {detail}"
          + (f"   [replaces feature #{replace}]" if replace is not None else ""))
    if profiles:
        print(f"profiles:     {profiles}")
    _show_program(args, full)
    note = (f"sketch {'' if combine == 'add' else combine + ' '}{op}") + (f" (replaces #{replace})" if replace is not None else "")
    _save_and_check(args, full, note, feature=-1 if replace is None else replace)


# A --both far end this close to a named level is that level: the gap is the
# emit grid's rounding of a refined level, not a design.
BOTH_SNAP = 2.5 * 10.0 ** -EMIT_DECIMALS


def _both_far_end(level, target, marks):
    """The named level a `--both` feature's far end lands on, or None.

    --both builds the same distance either side of --at, so its far end is
    2*at - to. With --at mid(A,B) of refined levels, that end misses the
    neighbour's level by the grid's rounding, and a gap of 1e-6 leaves the part
    in pieces. Within BOTH_SNAP of a named level, the far end is that level.
    """
    far = 2.0 * float(level) - float(target)
    near = min(marks or [], key=lambda m: abs(m["value"] - far), default=None)
    if near is None or abs(near["value"] - far) > BOTH_SNAP:
        return None
    return near, far


def _both_span(args, ax, level, target):
    """(start, stop, note) for a --both extrude whose far end is a named level,
    so it is built between the two levels exactly; else None (symmetric)."""
    got = _both_far_end(level, target, getattr(args, "_marks", None))
    if got is None:
        return None
    near, far = got
    note = (f"note: --both: the far end, {ax}={far:.6f}, is level {near['name']} "
            f"({ax}={near['value']:.4f}) to within the emit grid; built from "
            f"{near['name']} to --to so both ends meet their neighbours exactly")
    return float(near["value"]), float(target), note


LAW_HELP = ("fit a designed family -- circle, slot or rounded rectangle -- to the "
            "profile at levels between this and the other one named, and say how "
            "each of its numbers (centre, half-sizes, corner R) changes: a straight "
            "line, or not. A drafted pocket or tray whose corners keep or grow "
            "their radius is a rounded rectangle on such lines")


def _rings_like(ring, polys):
    """The rings of `polys` that can be `ring` at another level: outlines for an
    outline, openings for an opening (also one named as a cutter). Matching by
    overlap alone can take a drafted outline a millimetre deeper for its
    opening: two nested walls of one tray."""
    hole = ring.get("was_role", ring.get("role")) == "hole"
    return [r for p in polys for r in (p.interiors if hole else [p.exterior])]


def _strip_here(args):
    """Whether --along W# is the strip of ring R# in the section this sketch
    cuts: printed last by `section` on this plane, or printed by nothing (the
    old reading, where W# was only ever that)."""
    rec, _ = readback.printed_path(args.along, frame=_frame_key(args))
    return rec is None or (rec["source"] == "section" and rec["plane"] == args.plane
                           and not rec.get("pieces"))


def _sweep_face_plane(args, mesh):
    """`--face F# --op sweep --along W#` takes the face's own plane: the path
    keeps the plane it was named on, so a --plane naming that one is not the
    profile's."""
    if not (getattr(args, "face", None) and args.op == "sweep" and args.plane
            and (args.along or "").startswith("W")):
        return
    from mesh2cad.frames import PLANE_ACROSS
    from mesh2cad.segment import principal_of, segment_cached
    s = segment_cached(mesh, cache_dir=_cache_dir(args)).by_name(args.face.upper())
    if s is None or s.kind != "plane":
        return
    k = principal_of(s.params["normal"])
    if k is None:
        return
    across = PLANE_ACROSS["XYZ".index(k)]
    if across != args.plane:
        print(f"note: --face {s.name} lies across {k}, so the profile is on {across}; "
              f"--plane {args.plane}" + (f" and --at {args.at}" if args.at is not None else "")
              + f" are not used: --along {args.along} keeps the plane it was named on")
        args.plane, args.at = across, None


def _known_paths(args, mesh):
    """The W# printed for this part, and where each comes from, for a
    refusal. Only what was printed: segmenting the part to list its pipes
    held a refusal on a coil for minutes."""
    out = []
    for n, r in sorted(readback.printed_paths(frame=_frame_key(args)).items()):
        if r["source"] == "faces":
            chain = r.get("chain", "").split(" -> ")
            out.append(f"{n} as `faces` lists it (the pipe "
                       f"{' -> '.join(chain if len(chain) <= 5 else chain[:2] + ['...'] + chain[-2:])}"
                       f", end faces {r.get('ends', '?')})")
        elif r["source"] == "section":
            ax = "XYZ"[PLANES[r["plane"]]["axis"]]
            out.append(f"{n} as `section --plane {r['plane']}` at {ax}={r['level']:.4f} "
                       f"printed it ("
                       + (f"the tube between {' and '.join(r['walls'])}" if r.get("walls")
                          else f"the strip {r['ring']}") + ")")
        else:
            out.append(f"{n} as `faces` listed it (a wire R{r['radius']:.4f}, "
                       f"{r['length']:.2f} long)")
    return ("; ".join(out) + " -- a W# is the path printed under that name last") if out else (
        "none yet -- `faces` lists a tube's pipe paths, `section` a strip's centreline "
        "in its own plane")


def _path_of(args, mesh, name):
    """(world pieces, what the path is, width) of --along W#: the pipe `faces`
    chained, or a strip's centreline on the plane and at the level `section`
    printed it -- whichever printed the name last."""
    from mesh2cad import pipes
    from mesh2cad.segment import segment_cached
    from mesh2cad.strips import StripError, centreline
    rec, older = readback.printed_path(name, frame=_frame_key(args))
    for o in older:
        print(f"note: {name} is the path {'`faces` listed' if rec['source'] == 'faces' else '`section --plane ' + rec['plane'] + '` printed'} "
              f"last; {'`faces`' if o['source'] == 'faces' else '`section --plane ' + o['plane'] + '`'} "
              f"printed another {name} before it. Re-run that command to sweep along that one")
    if rec is not None and rec.get("pieces"):
        return ([(q[0],) + tuple(tuple(p) for p in q[1:]) for q in rec["pieces"]],
                rec["what"], float(rec["width"]))
    if rec is not None and rec["source"] == "wires":
        return ([("spline",) + tuple(tuple(p) for p in rec["points"])], rec["what"],
                float(rec["width"]))
    if rec is None or rec["source"] == "faces":
        found = {w["name"]: w for w in pipes.pipe_paths(
            mesh, segment_cached(mesh, cache_dir=_cache_dir(args)))}
        if name not in found:
            raise SystemExit(f"--along {name}: no such path. W# here: {_known_paths(args, mesh)}"
                             + _path_printed_elsewhere(args, name))
        w = found[name]
        chain = [f[0] for f in w["faces"]]
        what = (f"{name}, the pipe {' -> '.join(chain if len(chain) <= 5 else chain[:2] + ['...'] + chain[-2:])} "
                f"(tube R{w['tube']:.4f})")
        return w["pieces"], what, 2.0 * w["tube"]
    P, level, ring = rec["plane"], float(rec["level"]), rec["ring"]
    polys, rings, _, _ = _rings_at(mesh, P, level, args.tol, args.grid)
    r = next((x for x in rings if x.get("name") == ring), None)
    if r is None:
        raise SystemExit(f"--along {name}: ring {ring} is not in the section on {P} at "
                         f"{level:.4f} any more; re-run that section. W# here: "
                         f"{_known_paths(args, mesh)}")
    k = r.get("poly_index")
    try:
        c = centreline(r, polys[k] if k is not None and k < len(polys) else None)
    except StripError as exc:
        raise SystemExit(f"--along {name}: ring {ring} on {P} is {exc}") from None
    ax = "XYZ"[PLANES[P]["axis"]]
    what = f"{name}, the centreline of strip {ring} in the {P} section at {ax}={level:.4f}"
    pieces = pipes.to_world(c["pieces"], PLANES[P], level)
    kink = pipes.max_kink(pieces)
    if kink > pipes.KINK_DEG:
        # A U of three bends can come back as one arc meeting its runs off
        # their line; swept, the profile folds through itself.
        print(f"note: the lines and arcs fitted to {ring}'s middle meet at up to "
              f"{kink:.1f} deg, so {name} is swept along a smooth curve through the "
              f"middle line itself")
        pieces = pipes.to_world([pipes.smooth_spline(c["points"], c["width"])],
                                PLANES[P], level)
        what += " (a smooth curve through its middle)"
    return pieces, what, c["width"]


def _wire_path(args):
    """The record of --along W# when it is a wire off the mesh and no profile
    was named, else None."""
    if (args.along or "")[:1] != "W" or args.use or getattr(args, "face", None):
        return None
    rec, _ = readback.printed_path(args.along, frame=_frame_key(args))
    return rec if rec is not None and rec["source"] == "wires" else None


def _path_printed_elsewhere(args, name):
    """"; W2 was printed by `section --plane YZ --out run`: add --out run" when
    another --out directory under this one holds a record of `name` for this
    mesh, else "". Without it the refusal says only "none yet", and the model
    spends calls finding out where the name came from."""
    import hashlib
    import json
    key = hashlib.sha1(str(Path(args.mesh).resolve()).encode()).hexdigest()[:12]
    try:
        here = Path(out_dir(args)).resolve()
    except Exception:                                              # noqa: BLE001
        here = None
    seen = []
    for pat in (".m2c/read", "*/.m2c/read", "*/*/.m2c/read", "*/*/*/.m2c/read"):
        for d in sorted(Path(".").glob(pat)):
            for f in sorted(d.glob(f"{key}_all_*_paths.json")):
                out = f.parent.parent.parent
                if here is not None and out.resolve() == here:
                    continue
                try:
                    recs = (json.loads(f.read_text()).get("paths") or {}).get(name) or {}
                except (OSError, ValueError):
                    continue
                if recs:
                    seen.append((max(r.get("t", 0) for r in recs.values()), out,
                                 max(recs.values(), key=lambda r: r.get("t", 0))))
    if not seen:
        return ""
    _, out, rec = max(seen, key=lambda s: s[0])
    by = ("`faces`" if rec.get("source") == "faces" else
          f"`section --plane {rec.get('plane')}`" if rec.get("plane") else "an earlier command")
    return (f". This command read the names kept under {here}; {name} was printed by {by} "
            f"--out {out}: add `--out {out}`")


def _check_to_profile(args):
    """--to-profile goes with one op and one kind of path."""
    if not getattr(args, "to_profile", None):
        return
    if args.op != "sweep" or (args.along or "")[:1] != "W":
        raise SystemExit("--to-profile is the second profile of --op sweep --along W#: the "
                         "section at the path's other end")
    if getattr(args, "face", None):
        raise SystemExit("--to-profile takes rings of a section at both ends: give the first "
                         "with --use, not --face")


def _sketch_wire(args, rec):
    """`--op sweep --along W#` of a wire `faces` read off the mesh: a
    round wire of its radius along it. It has no plane or section to name."""
    from mesh2cad.emit import build_wire_sweep
    from mesh2cad.frames import PLANE_ACROSS
    pts = [tuple(p) for p in rec["points"]]
    k = int(np.argmax(np.abs(np.subtract(pts[1], pts[0]))))
    args.plane = PLANE_ACROSS[k]
    level = float(pts[0][k])
    code = build_wire_sweep(pts, float(rec["radius"]))
    detail = (f"a round wire R{rec['radius']:.4f} along {rec['name']}, "
              f"{rec['length']:.4f} long, read off the mesh by `faces`")
    _finish_sketch(args, code, "XYZ"[k], level, level, None, detail, "sweep",
                   profiles=f"path {rec['name']}")


# Seconds the kernel gets to carry one section into another.
PATH_LOFT_SECONDS = 60.0


PATH_LOFT_ADVICE = "Name a section of the same shape: `end`, or a level nearer this one"


def _try_path_loft(code, near_names, far_names, name, advice=PATH_LOFT_ADVICE):
    """Build the loft before it is written, in a child that can be stopped: None
    when it builds, else what the kernel said. A section with a blend's rings in
    it can fail only at --check ("PipeShell : uncompatible wires"), in a program
    already written, or never come back at all. A stop on time is a refusal,
    with `advice`."""
    import os
    import tempfile
    fd, said = tempfile.mkstemp(prefix="m2c_loft_")
    os.close(fd)

    def work():
        _tb.stage(f"carrying {near_names} into {far_names} along {name}")
        try:
            exec(compile("import cadquery as cq\n" + code, "<path loft>", "exec"), {})
        except Exception as exc:                                   # noqa: BLE001
            Path(said).write_text(str(exc) or type(exc).__name__)
            sys.exit(3)
    # The stop says $M2C_TIME_BUDGET allows longer, so it has to.
    limit = PATH_LOFT_SECONDS if os.environ.get(_tb.BUDGET_ENV) is None else _tb.budget()
    try:
        if limit is None:
            try:
                work()
                got = 0
            except SystemExit as exc:
                got = exc.code
        else:
            got = _tb.run(work, limit, "`--to-profile`", advice)
        text = Path(said).read_text().strip()
    finally:
        os.unlink(said)
    if not got:
        return None
    return text or f"the build stopped with exit code {got}"


def _far_section(args, mesh, rec, name, path):
    """(plane, level, the rings named) of --to-profile <level>:R#,..., on the
    plane square to `path` at its far end. `end` is the section `section`
    printed at that end of a tube's path."""
    from mesh2cad import pipes
    from mesh2cad.frames import PLANE_ACROSS
    expr = args.to_profile
    what, colon, names = expr.partition(":")
    names = [n.strip().upper() for n in names.split(",") if n.strip()]
    if not colon or not names:
        raise SystemExit(f"--to-profile {expr}: give the section's level and its rings, "
                         f"<level>:R#,R#, e.g. end:R0,R1 or B2:R0,R1")
    t = pipes.tangent_at(path[-1], 1.0)
    k = int(np.argmax(np.abs(t)))
    plane = PLANE_ACROSS[k]
    if what.strip().lower() == "end":
        ends = [e for e in ((rec or {}).get("ends") or []) if e["plane"] == plane]
        if not ends:
            raise SystemExit(f"--to-profile {expr}: no section was printed at the far end of "
                             f"{name}. Name a level across it, on {plane}: <level>:R#,R#")
        far = pipes.point_at(path[-1], 1.0)[k]
        level = min((float(e["level"]) for e in ends), key=lambda v: abs(v - far))
    else:
        level = resolve_level(what.strip(), _levels(args, mesh, plane)[0], "--to-profile",
                              mesh, plane)
    at = float(level)
    hair = 1e-3 * float(np.ptp(mesh.bounds[:, k])) if mesh is not None else 0.0
    if mesh is not None and min(abs(at - float(b)) for b in mesh.bounds[:, k]) < hair:
        # `end` on the part's end face is measured a hair inside, where
        # `section` named its rings; on the face itself the cut is degenerate
        at += hair * (1.0 if float(mesh.bounds.mean(axis=0)[k]) > at else -1.0)
    polys, rings, _, _ = _rings_at(mesh, plane, at, args.tol, args.grid)
    by_name = {r["name"]: r for r in rings if r.get("name")}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise SystemExit(f"--to-profile {expr}: {', '.join(missing)} not in the {plane} section "
                         f"at {'XYZ'[k]}={float(level):.4f}, which has "
                         f"{', '.join(by_name) or 'no rings'}")
    return plane, float(level), [by_name[n] for n in names]


def _to_level(path, k, level, name):
    """`path` ending on coordinate k = level: cut at its last crossing there,
    or carried straight on from its end."""
    from mesh2cad import pipes
    cross = pipes.crossings(path, k, level)
    if cross:
        i, f = cross[-1]
        return list(path[:i]) + [pipes._part(path[i], 0.0, f)]
    end = pipes.point_at(path[-1], 1.0)
    t = pipes.tangent_at(path[-1], 1.0)
    step = (level - end[k]) / t[k] if abs(t[k]) > 1e-9 else -1.0
    if step <= 0 or step > pipes.length(path):
        raise SystemExit(f"--to-profile: {name} doesn't reach {'XYZ'[k]}={level:.4f}: it ends "
                         f"at {'XYZ'[k]}={end[k]:.4f}. Name a level it crosses")
    return list(path) + [("line", tuple(end), tuple(end + t * step))]


def _sweep_path_loft(args, mesh, rec, name, outer, holes, path, spec, ax, sense, level, cut_at,
                     snapped, what, how):
    """--op sweep --along W# --to-profile: the section lofted into the one at
    the path's other end, holes as holes -- for a bend that goes from a round
    pipe to a rounded-rectangle duct, which one section swept along it cannot
    follow."""
    from shapely.geometry import Polygon

    from mesh2cad import pipes
    from mesh2cad.emit import build_path_loft
    if path is None:
        raise SystemExit(f"--to-profile: this profile is a section across {name}, swept both "
                         f"ways. A second profile needs it at an end of the path: name one")
    plane, far, rings = _far_section(args, mesh, rec, name, path)
    outers = [r for r in rings if r.get("role") != "hole"]
    inner = [r for r in rings if r.get("role") == "hole"]
    near_names = "+".join(r.get("name", "?") for r in [outer, *holes])
    far_names = "+".join(r.get("name", "?") for r in rings)
    if len(outers) != 1 or len(inner) != len(holes):
        raise SystemExit(f"--to-profile: the rings don't pair: {near_names} here is 1 outline "
                         f"and {len(holes)} hole(s), {far_names} there is {len(outers)} and "
                         f"{len(inner)}. Name the same ones at both ends")
    k = PLANES[plane]["axis"]
    path = _to_level(path, k, far, name)

    def by_area(rs):
        return sorted(rs, key=lambda r: -Polygon(readback.outline(r)).buffer(0).area)
    pairs = [(outer, outers[0]), *zip(by_area(holes), by_area(inner), strict=True)]
    near_at = (args.plane, sense, level)
    far_at = (plane, float(PLANES[plane]["normal"][k]), far)
    if (args.to_profile or "").partition(":")[0].strip().lower() == "end":
        advice = (f"`end` was this call's far section: name a level across {name} nearer this "
                  f"end, on {plane} (one `section --plane {plane}` prints), or sweep this "
                  f"section alone (no --to-profile)")
    else:
        advice = PATH_LOFT_ADVICE
    n_st = _station_count(args, pairs)
    if n_st > 2:
        _sweep_stations(args, mesh, name, path, pairs, (args.plane, level), (plane, far), n_st,
                        near_names, far_names, advice, ax, level, cut_at, snapped, what, how)
        return
    code = build_path_loft(near_at, far_at, pairs, path)
    said = _try_path_loft(code, near_names, far_names, name, advice)
    order = ""
    if said is not None:
        # The same two sections, the other way along the path: a sweep the
        # kernel refuses near -> far can build far -> near.
        back = build_path_loft(far_at, near_at, [(b, a) for a, b in pairs], pipes.reverse(path))
        again = _try_path_loft(back, far_names, near_names, name, advice)
        if again is not None:
            raise SystemExit(f"--to-profile: the kernel can't carry {near_names} into "
                             f"{far_names} along {name}, either way ({said}; from the far end: "
                             f"{again}). {advice}")
        code = back
        order = (f"; built from the far end back (the kernel refused {near_names} into "
                 f"{far_names}: {said})")
        print(f"note: {name}: {near_names} into {far_names} was refused by the kernel ({said}); "
              f"the same two sections built the other way, from {far_names} back along "
              f"{name} -- the same solid")
    detail = (f"{near_names} into {far_names} of the {plane} section at {'XYZ'[k]}={far:.4f}, "
              f"along {what}; {how}{order}")
    _finish_sketch(args, code, ax, level, cut_at, snapped, detail, "sweep",
                   profiles=f"path {name}")


# Points each ring is resampled to at every station, by angle about its
# centroid from the station's own first axis.
STATION_POINTS = 96
# Stations a --to-profile sweep takes when the two end sections differ.
STATIONS_DEFAULT = 5


def _station_count(args, pairs):
    """--stations, or STATIONS_DEFAULT when the end outlines differ in area by
    more than 5% (2 -- the ends alone -- when they don't)."""
    from shapely.geometry import Polygon
    if getattr(args, "stations", None) is not None:
        if args.stations < 2:
            raise SystemExit(f"--stations {args.stations}: the two ends are stations already; "
                             f"give 2 or more")
        return int(args.stations)
    a, b = (Polygon(readback.outline(r)).buffer(0).area for r in pairs[0])
    return STATIONS_DEFAULT if abs(a / max(b, 1e-12) - 1.0) > 0.05 else 2


def _path_frames(path, fs, e0):
    """[(point, tangent, first axis)] at the arc-length fractions `fs` of the
    path; the first axis starts as `e0` square to the path and is carried along
    it without turning about it."""
    from mesh2cad import pipes
    lens = [pipes.piece_length(q) for q in path]
    total = sum(lens)
    out, e1 = [], None
    for f in fs:
        s, i = f * total, 0
        while i < len(path) - 1 and s > lens[i]:
            s -= lens[i]
            i += 1
        g = min(max(s / lens[i], 0.0), 1.0) if lens[i] > 0 else 0.0
        p, t = pipes.point_at(path[i], g), pipes.tangent_at(path[i], g)
        e = np.asarray(e0 if e1 is None else e1, float)
        e = e - (e @ t) * t
        e1 = e / np.linalg.norm(e)
        out.append((np.asarray(p, float), np.asarray(t, float), e1))
    return out


def _sample_ring(pts, n=STATION_POINTS):
    """`n` points of a closed 2-D ring, one every 360/n deg about its centroid
    from the first axis, counter-clockwise: matched rings across stations."""
    from shapely.geometry import LineString, Point, Polygon
    poly = Polygon(pts).buffer(0)
    if poly.is_empty:
        return None
    if poly.geom_type != "Polygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    c = np.array(poly.centroid.coords[0])
    reach = 4.0 * float(np.hypot(*(np.array(poly.bounds[2:]) - np.array(poly.bounds[:2]))))
    out = []
    for th in np.linspace(0.0, 2 * np.pi, n, endpoint=False):
        d = np.array([np.cos(th), np.sin(th)])
        hit = LineString([c, c + reach * d]).intersection(poly.exterior)
        pts_h = [np.array(g.coords[0]) for g in getattr(hit, "geoms", [hit])
                 if not g.is_empty and g.geom_type == "Point"]
        if not pts_h:
            return None
        out.append(max(pts_h, key=lambda q: Point(*q).distance(Point(*c))))
    return np.array(out)


def _cut_square(mesh, p, t, e1):
    """The target's section on the plane through `p` square to `t`, in the
    station's (e1, t x e1) coordinates: (outline, [holes]) of the polygon
    round the path, or None."""
    from shapely.geometry import Point, Polygon
    e2 = np.cross(t, e1)
    sec = mesh.section(plane_origin=p, plane_normal=t)
    if sec is None:
        return None
    M = np.eye(4)
    M[0, :3], M[1, :3], M[2, :3] = e1, e2, t
    M[:3, 3] = -M[:3, :3] @ p
    planar, _ = sec.to_2D(to_2D=M, check=False)
    round_it = [g for g in planar.polygons_full
                if g is not None and Polygon(g.exterior).contains(Point(0.0, 0.0))]
    if not round_it:
        return None
    g = min(round_it, key=lambda g: Polygon(g.exterior).area)
    return np.asarray(g.exterior.coords)[:-1], [np.asarray(h.coords)[:-1] for h in g.interiors]


def _sweep_stations(args, mesh, name, path, pairs, near, far, n_st, near_names, far_names,
                    advice, ax, level, cut_at, snapped, what, how):
    """--to-profile through `n_st` stations along the path: the named rings at
    its ends, the target cut square to the path between them, every ring
    resampled to matched points and swept through together, holes as holes.
    Two sections cannot follow a bend whose height stays put for a stretch and
    then falls; stations between them can."""
    from shapely.geometry import Polygon

    from mesh2cad.emit import build_station_sweep
    spec0 = PLANES[near[0]]
    e0 = np.zeros(3)
    e0[spec0["u"]] = 1.0
    frames = _path_frames(path, np.linspace(0.0, 1.0, n_st), e0)

    def world(r, plane, lvl):
        sp = PLANES[plane]
        out = []
        for u, v in readback.outline(r):
            q = np.zeros(3)
            q[sp["u"]], q[sp["v"]], q[sp["axis"]] = u, v, lvl
            out.append(q)
        return np.array(out)
    sections, lines = [], []
    for j, (p, t, e1) in enumerate(frames):
        e2 = np.cross(t, e1)
        if j in (0, n_st - 1):
            (plane, lvl), side = (near, 0) if j == 0 else (far, 1)
            k = PLANES[plane]["axis"]
            rings2d = [np.c_[(w - p) @ e1, (w - p) @ e2] for w in
                       (world(pr[side], plane, lvl) for pr in pairs)]
        else:
            got = _cut_square(mesh, p, t, e1)
            if got is None:
                raise SystemExit(f"--to-profile --stations {n_st}: the target cut square to "
                                 f"{name} at {100 * j / (n_st - 1):.0f}% of its length has no "
                                 f"section round the path. Give fewer --stations")
            outline, inner = got
            if len(inner) != len(pairs) - 1:
                raise SystemExit(f"--to-profile --stations {n_st}: at {100 * j / (n_st - 1):.0f}% "
                                 f"of {name} the section has {len(inner)} hole(s), the ends "
                                 f"{len(pairs) - 1}: the rings don't pair there. Give fewer "
                                 f"--stations, or name sections nearer each other")
            inner = sorted(inner, key=lambda h: -Polygon(h).area)
            rings2d = [outline, *inner]
            plane = None
        sec, sizes = [], []
        for r2 in rings2d:
            s = _sample_ring(r2)
            if s is None:
                raise SystemExit(f"--to-profile --stations {n_st}: a ring at "
                                 f"{100 * j / (n_st - 1):.0f}% of {name} can't be resampled")
            q = p + np.outer(s[:, 0], e1) + np.outer(s[:, 1], e2)
            if plane is not None:
                # an end ring stays on its own plane, which needn't be square
                q = q + np.outer((lvl - q[:, k]) / t[k], t)
            sec.append(q)
            sizes.append(f"{np.ptp(s[:, 0]):.2f} x {np.ptp(s[:, 1]):.2f}")
        sections.append(sec)
        lines.append(f"  station {j + 1}/{n_st} at {100 * j / (n_st - 1):3.0f}% of {name}: "
                     f"outline {sizes[0]}"
                     + "".join(f", hole {s}" for s in sizes[1:]))
    code = build_station_sweep(path, sections)
    said = _try_path_loft(code, near_names, far_names, name, advice)
    if said is not None:
        raise SystemExit(f"--to-profile --stations {n_st}: the kernel can't sweep through the "
                         f"stations along {name} ({said}). Give --stations 2 (the two ends "
                         f"alone) or fewer stations. {advice}")
    print(f"note: {near_names} into {far_names} through {n_st} sections square to {name} "
          f"(the ends named, {n_st - 2} cut from the target between them), each ring as "
          f"{STATION_POINTS} points matched by angle; sizes along the path:")
    print("\n".join(lines))
    detail = (f"{near_names} into {far_names} through {n_st} sections square to {name}, "
              f"along {what}; {how}")
    _finish_sketch(args, code, ax, level, cut_at, snapped, detail, "sweep",
                   profiles=f"path {name}, {n_st} stations")


def _sweep_along(args, mesh, rings, outer, holes, spec, ax, sense, level, cut_at, snapped):
    """`--op sweep --along W#`. A strip of this very section swept
    as a round wire of its width, as before; any other profile -- an end face,
    a section across the path, a ring on another plane -- swept as it is,
    holes and all, along the path where it was named."""
    from shapely.geometry import Point, Polygon

    from mesh2cad import pipes
    from mesh2cad.emit import build_path_sweep, build_pipe_sweep
    from mesh2cad.strips import StripError, centreline
    name = args.along
    rec, _ = readback.printed_path(name, frame=_frame_key(args))
    ring_name = "R" + name[1:]
    here = (not getattr(args, "face", None) and outer.get("name") == ring_name
            and (rec is None or (rec["source"] == "section" and rec["plane"] == args.plane)))
    if here:
        ring = next((r for r in rings if r.get("name") == ring_name), None)
        k = ring.get("poly_index")
        cut = section_at(mesh, args.plane, cut_at)
        try:
            c = centreline(ring, cut[k] if k is not None and k < len(cut) else None)
        except StripError as exc:
            raise SystemExit(f"--along {name}: ring {ring_name} is {exc}. W# here: "
                             f"{_known_paths(args, mesh)}") from None
        if getattr(args, "_use_from_path", False) and _strip_is_sheet(
                mesh, args.plane, cut_at, ring, c["width"]):
            # A round wire nobody asked for, swept along a joggled plate's S-bend,
            # would change nothing and say nothing.
            recipe = _sheet_recipe(mesh, args.plane, level, ring, c,
                                   getattr(args, "_marks", []), args.mesh) or []
            raise SystemExit(
                f"--along {name} with no profile: {ring_name} is a sheet "
                f"{c['width']:.4f} thick cut across, and a round wire along its middle is "
                f"not it. Name the profile to sweep (--use, --face), or build the plate:\n"
                + "\n".join(recipe[1:]))
        code = build_path_sweep(args.plane, level, c["pieces"], c["width"] / 2,
                                capped=bool(c["caps"]))
        detail = (f"a round wire R{c['width'] / 2:.4f} along {name}, the "
                  f"centreline of ring {ring_name} ({len(c['pieces'])} pieces, "
                  f"{c['length']:.4f} long)"
                  + (", ends rounded" if c["caps"] else ", ends flat"))
        _finish_sketch(args, code, ax, level, cut_at, snapped, detail, "sweep",
                       profiles=f"path {name}")
        return
    pieces, what, width = _path_of(args, mesh, name)
    shape = Polygon(readback.outline(outer)).buffer(0)
    lo_u, lo_v, hi_u, hi_v = shape.bounds
    size = float(np.hypot(hi_u - lo_u, hi_v - lo_v))
    # Round the profile, not in it: a half tube cut at its axis (by --split)
    # has the path in its open bore.
    grown = shape.convex_hull.buffer(0.02 * size)
    k = spec["axis"]

    def inside(P):
        return bool(grown.contains(Point(P[spec["u"]], P[spec["v"]])))
    try:
        paths, how = pipes.from_plane(pieces, k, level, inside, reach=max(0.5 * size, width))
    except pipes.PipeError as exc:
        a, b = pipes.point_at(pieces[0], 0.0), pipes.point_at(pieces[-1], 1.0)
        span = sorted(pipes.point_at(q, f)[k] for q in pieces for f in (0.0, 0.5, 1.0))
        raise SystemExit(
            f"--along {name}: {exc} ({outer.get('name') or 'the profile'} at {ax}="
            f"{level:.4f}). {what} runs from ({a[0]:.4f}, {a[1]:.4f}, {a[2]:.4f}) to "
            f"({b[0]:.4f}, {b[1]:.4f}, {b[2]:.4f}), {ax} {span[0]:.4f}..{span[-1]:.4f}. "
            f"Give a profile where it starts or ends (an end face: --face F#) or a "
            f"section across it, around the path. W# here: {_known_paths(args, mesh)}") from None
    # A profile reaching past the tightest bend's radius folds through itself
    # there, into a mesh of broken faces.
    start = np.asarray(paths[0][0][1], float)
    reach = max(float(np.hypot(u - start[spec["u"]], v - start[spec["v"]]))
                for u, v in readback.outline(outer))
    bend, centre = pipes.tightest_bend(paths)
    if bend is not None and reach >= bend:
        raise SystemExit(
            f"--along {name}: {outer.get('name') or 'the profile'} reaches {reach:.4f} from "
            f"the path, and the path bends at R{bend:.4f} about ({centre[0]:.4f}, "
            f"{centre[1]:.4f}, {centre[2]:.4f}): swept there, the profile would run through "
            f"itself. A flange or boss on a tube is a feature of its own -- sweep the tube's "
            f"own section (a cut across a straight run, or the end face of the bare tube), "
            f"then build the rest on it with --base")
    if getattr(args, "to_profile", None):
        _sweep_path_loft(args, mesh, rec, name, outer, holes, paths[0] if len(paths) == 1 else None,
                         spec, ax, sense, level, cut_at, snapped, what, how)
        return
    code = build_pipe_sweep(args.plane, sense, level, outer, holes, paths)
    detail = (f"{outer.get('name') or 'the profile'}"
              + (f" with {len(holes)} hole(s)" if holes else "")
              + f" along {what}; {how} ({' + '.join(f'{pipes.length(p):.4f}' for p in paths)} "
              f"long, {sum(len(p) for p in paths)} pieces)")
    _finish_sketch(args, code, ax, level, cut_at, snapped, detail, "sweep",
                   profiles=f"path {name}")


STRIP_MAX_PRIMS = 400   # rings longer than this are not searched for a strip
# A strip is a sheet cut across, not a wire in its own plane, when the part runs
# this many of its widths along the cut's normal (a thin joggled plate that
# runs far across the cut that named its S-bend).
SHEET_RUN = 5.0


def _strip_is_sheet(mesh, plane, level, ring, width):
    """Whether a strip is a sheet cut across: the part runs SHEET_RUN widths
    through the plane, and a cut a tenth of the way across finds the same
    strip there -- a plate bent in this plane is a prism across it; a tube cut
    along its axis is not, its walls move apart."""
    k = PLANES[plane]["axis"]
    lo, hi = float(mesh.bounds[0][k]), float(mesh.bounds[1][k])
    if hi - lo < SHEET_RUN * width:
        return False
    step = 0.1 * (hi - lo)
    other = level + step if level + step < hi else level - step
    try:
        _, there, _, _ = _rings_at(mesh, plane, other, None, None)
    except Exception:                                              # noqa: BLE001
        return False
    return max((readback.overlap(ring, r) for r in there), default=0.0) >= 0.85


def _sheet_recipe(mesh, plane, level, ring, c, marks, mesh_name):
    """Lines saying a strip is a bent plate cut across, and how to build one:
    the strip extruded across, trimmed to the plate's outline seen through
    its thickness. None when the strip is a wire or a tube's wall."""
    spec = PLANES[plane]
    k = spec["axis"]
    lo, hi = float(mesh.bounds[0][k]), float(mesh.bounds[1][k])
    if not _strip_is_sheet(mesh, plane, level, ring, c["width"]):
        return None
    ring = ring["name"]
    ax = "XYZ"[k]
    a = np.asarray(c["pieces"][0][1], float)
    b = np.asarray(c["pieces"][-1][-1], float)
    run = int(np.argmax(np.abs(b - a)))              # 0: along u, 1: along v
    thick = spec["v"] if run == 0 else spec["u"]     # the world axis through it
    side = next(pl for pl, sp in PLANES.items() if sp["axis"] == thick)
    names = sorted((m for m in marks if m.get("name")), key=lambda m: m["value"])
    e0 = names[0]["name"] if names else "E0"
    e1 = names[-1]["name"] if names else "E1"
    mid = abs(level - 0.5 * (lo + hi)) <= 0.05 * (hi - lo)
    across = (f"--to {e1} --both" if mid else f"--to {e1}, and again --to {e0} --base it")
    return [f"  this strip is a sheet {c['width']:.4f} thick cut across, not a wire: the "
            f"part runs {ax} {lo:.4f}..{hi:.4f} through this plane. A bent plate (a joggle) "
            f"is the strip extruded across and trimmed to the plate's outline:",
            f"    sketch {mesh_name} --plane {plane} --at {level:.4f} --use {ring} --op extrude "
            f"{across}",
            f"    sketch {mesh_name} --plane {side} --at E0 --project E1 --use <the outline "
            f"`section --plane {side} --offset E0 --project E1` names> --op extrude --to E1 "
            f"--combine intersect --base <that program>"]



def _strip_lines(rings, polys=(), named=None, sheet=None, centres=None):
    """`path W#` lines: the centreline of each ring that is a strip -- a wire
    or pipe cut in its own plane -- named for `--op sweep --along W#`. The
    rings whose W# was printed are appended to `named`, and the centreline of
    each kept in `centres`. `sheet(ring, c)` gives the lines for a strip that
    is a sheet cut across, or None."""
    from mesh2cad.strips import StripError, centreline
    out = []
    for r in rings:
        if not r.get("name") or len(r.get("segments", [])) > STRIP_MAX_PRIMS:
            continue
        k = r.get("poly_index")
        try:
            c = centreline(r, polys[k] if k is not None and k < len(polys) else None)
        except StripError:
            continue
        except Exception as exc:                                   # noqa: BLE001
            # A GEOS failure in this optional hint must not be the traceback
            # that ends the whole section.
            out.append(f"\nnote: no strip centreline measured for ring {r['name']} "
                       f"({type(exc).__name__}: {str(exc)[:80]}); the section above "
                       f"is unaffected")
            continue
        if centres is not None:
            centres[r["name"]] = c
        w = "W" + r["name"][1:]
        kinds = [p[0] for p in c["pieces"]]
        out.append(f"\npath {w}: the centreline of ring {r['name']}, a strip {c['width']:.4f} "
                   f"wide (a round wire R{c['width'] / 2:.4f}): {len(kinds)} pieces "
                   f"({kinds.count('line')} lines, {kinds.count('arc')} arcs), "
                   f"{c['length']:.4f} long, "
                   + (f"{len(c['caps'])} rounded end(s)" if c["caps"] else "open ends"))
        recipe = sheet(r, c) if sheet is not None else None
        if recipe:
            out.extend(recipe)
        else:
            out.append(f"  build it: sketch ... --op sweep --along {w} (a round wire along it; "
                       f"or --face F# / a section across it, any plane, for that profile "
                       f"along it)")
        if named is not None:
            named.append(r["name"])
    return out


# Fractions of an end piece of a tube's path, in from its end, to look for the
# section across it.
TUBE_INSETS = (0.02, 0.05, 0.1)


def _across_tube(args, mesh, p, t):
    """(plane, level, rings) of the section square to a tube's path at `p`,
    where it runs along `t`: the outline round the path and the bores in it."""
    from shapely.geometry import Point

    from mesh2cad.frames import PLANE_ACROSS
    k = int(np.argmax(np.abs(t)))
    plane, level = PLANE_ACROSS[k], float(p[k])
    spec = PLANES[plane]
    here = Point(p[spec["u"]], p[spec["v"]])
    try:
        polys, rings, _, _ = _rings_at(mesh, plane, level, args.tol, args.grid)
    except CutError:
        return plane, level, []
    from shapely.geometry import Polygon
    named = [r for r in rings if r.get("name")
             and Polygon(readback.outline(r)).buffer(0).contains(here)]
    return plane, level, [r["name"] for r in named]


# A tube's end section, taken a little way in from its traced ends, is
# put on the part's end face when that face is this near beyond it (of the
# tube's width); otherwise every sweep would stop short of the end face.
TUBE_REACH = 0.1


def _ends_to_faces(args, mesh, world, ends, width):
    """Move each end section of `ends` [(plane, level, rings)] that lies within
    TUBE_REACH of the tube's width short of the part's extent, beyond it along
    the section's axis, onto that face (its rings named a hair inside), in
    place; the faces moved to, as "Z=-55.5482"."""
    from mesh2cad import pipes
    faces = []
    for i, (q, f) in enumerate(((world[0], 0.0), (world[-1], 1.0))):
        plane, lvl, rings = ends[i]
        k = PLANES[plane]["axis"]
        t = np.asarray(pipes.tangent_at(q, f), float) * (-1.0 if i == 0 else 1.0)
        s = 1.0 if t[k] > 0 else -1.0
        b = float(mesh.bounds[1 if s > 0 else 0][k])
        if not 0.0 < (b - lvl) * s <= TUBE_REACH * width:
            continue
        p = np.array(pipes.point_at(q, f), float)
        p[k] = b - s * 1e-3 * float(np.ptp(mesh.bounds[:, k]))
        _, _, named = _across_tube(args, mesh, p, t)
        if named:
            ends[i] = (plane, b, named)
            faces.append(f"{'XYZ'[k]}={b:.4f}")
    return faces


def _tube_paths(args, mesh, polys, centres, n_rings, level):
    """(printed lines, path records): a tube cut along its axis comes back as
    two wall strips, and their centrelines aren't its path. The middle of each
    pair of walls whose space between is empty is printed as a path, with the
    sections square to it at both ends."""
    import shapely

    from mesh2cad import pipes
    from mesh2cad.strips import fit_pieces
    spec, taken, lines, recs = PLANES[args.plane], set(), [], []
    solid = [p.buffer(0) for p in polys]
    items = list(centres.items())
    for i, (ra, ca) in enumerate(items):
        for rb, cb in items[i + 1:]:
            if ra in taken or rb in taken:
                continue
            wall = max(ca["width"], cb["width"])
            t = pipes.tube_midline(ca["points"], cb["points"], wall)
            if t is None or any(shapely.contains_xy(g, t["points"][:, 0], t["points"][:, 1]).any()
                                for g in solid):
                continue
            taken.update((ra, rb))
            pieces = fit_pieces(t["points"], 0.05 * wall)
            if pipes.max_kink(pipes.to_world(pieces, spec, level)) > pipes.KINK_DEG:
                pieces = [pipes.smooth_spline(t["points"], wall)]
            world = pipes.to_world(pieces, spec, level)
            name = f"W{n_rings + len(recs)}"
            # A little way in: the walls' ends are trace ends, and the midline
            # would run past the part.
            ends = []
            for q, fs in ((world[0], TUBE_INSETS), (world[-1], [1.0 - f for f in TUBE_INSETS])):
                for f in fs:
                    ends.append(_across_tube(args, mesh, pipes.point_at(q, f),
                                             pipes.tangent_at(q, f)))
                    if ends[-1][2]:
                        break
                    if f != fs[-1]:
                        ends.pop()
            faces = _ends_to_faces(args, mesh, world, ends, t["width"][1])
            kinds = [q[0] for q in pieces]
            head = (f"\npath {name}: the middle of the tube between walls {ra} and {rb}, "
                    f"{t['width'][0]:.4f} to {t['width'][1]:.4f} across: "
                    + (f"{len(kinds)} pieces ({kinds.count('line')} lines, "
                       f"{kinds.count('arc')} arcs)" if "spline" not in kinds else "a smooth curve")
                    + f", {t['length']:.4f} long")
            if len(pieces) == 1 and pieces[0][0] == "arc":
                head += f", R{_arc_radius(pieces[0]):.4f}"
            if faces:
                head += (", from face to face: its end sections are on the face"
                         + ("s" if len(faces) > 1 else "") + " its walls end on ("
                         + ", ".join(faces) + ")")
            lines.append(head)
            for (plane, lvl, rings), f in zip(ends, (0.0, 1.0), strict=True):
                ax = "XYZ"[PLANES[plane]["axis"]]
                lines.append(f"  end at {_at_path(world, f)}: the {plane} section at {ax}={lvl:.4f}"
                             + (f", rings {','.join(rings)}" if rings else ", no ring round it"))
            first = ends[0]
            lines.append(f"  build it: sketch {args.mesh} --plane {first[0]} --at {first[1]:.4f} "
                         f"--use {','.join(first[2]) or '<the rings there>'} --op sweep "
                         f"--along {name} (that section, bore and all, along it)"
                         + (f"; add --to-profile end:{','.join(ends[1][2])} if the section "
                            f"changes along it" if ends[1][2] else ""))
            recs.append({"name": name, "source": "section", "plane": args.plane,
                         "level": float(level), "walls": [ra, rb], "pieces": world,
                         "width": t["width"][0],
                         "ends": [{"plane": e[0], "level": e[1], "rings": e[2]} for e in ends],
                         "what": f"{name}, the middle of the tube between walls {ra} and {rb}"})
    return lines, recs


def _arc_radius(q):
    from mesh2cad.strips import _circle
    return _circle(*q[1:])[1]


def _at_path(world, f):
    from mesh2cad import pipes
    p = pipes.point_at(world[0] if f == 0.0 else world[-1], f)
    return "(" + ", ".join(f"{x:.4f}" for x in p) + ")"


def _law_of(mesh, plane, ring, a, b, ax):
    """(fits, levels, laws) of `ring`'s family between levels a and b, or a
    loud refusal."""
    from shapely.geometry import Polygon

    from mesh2cad import laws
    ref = Polygon(_ring_points(ring)).buffer(0)
    size = max(ref.bounds[2] - ref.bounds[0], ref.bounds[3] - ref.bounds[1])
    tol = max(3.0 * float(ring.get("tol") or 0.0), 0.002 * size)
    lo, hi = sorted((float(a), float(b)))
    inset = laws.LAW_INSET * (hi - lo)
    zs = list(np.linspace(lo + inset, hi - inset, laws.LAW_LEVELS))
    fits = []
    for z in zs:
        cands = [Polygon(r.coords).buffer(0) for r in _rings_like(ring, section_at(mesh, plane, z))]
        cands = [c for c in cands if c.area > 0]

        def iou(c):
            u = c.union(ref).area
            return c.intersection(ref).area / u if u > 0 else 0.0
        if not cands or iou(max(cands, key=iou)) <= 0.0:
            raise SystemExit(f"--law: nothing at {ax}={z:.4f} lies over ring "
                             f"{ring.get('name', '?')}")
        try:
            fits.append(laws.fit(max(cands, key=iou), tol))
        except laws.LawError as exc:
            raise SystemExit(f"--law: ring {ring.get('name', '?')} at {ax}={z:.4f} is "
                             f"{exc}") from None
    try:
        lw, dropped = laws.law(fits, zs, tol)
    except laws.LawError as exc:
        raise SystemExit(f"--law: ring {ring.get('name', '?')}: {exc}") from None
    for z in dropped:
        i = zs.index(z)
        print(f"note: --law: ring {ring.get('name', '?')} at {ax}={z:.4f} is off the "
              f"line its other levels follow (a transition: a bend, a step) and is "
              f"left out; the law is fitted on the rest")
        del zs[i], fits[i]
    return fits, zs, lw


def _law_lines(ring, fits, zs, lw, ax, a, b):
    from mesh2cad import laws
    out = [f"\nlaw of ring {ring.get('name', '?')}: {fits[0]['family']}, fitted at "
           f"{len(zs)} levels from {ax}={zs[0]:.4f} to {ax}={zs[-1]:.4f}",
           f"  {ax:>9}  {'centre u':>9} {'centre v':>9} {'half-w':>9} {'half-h':>9} "
           f"{'corner R':>9} {'off':>7}"]
    for z, f in zip(zs, fits, strict=True):
        out.append(f"  {z:9.4f}  {f['cx']:9.4f} {f['cy']:9.4f} {f['hw']:9.4f} "
                   f"{f['hh']:9.4f} {f['r']:9.4f} {f['off']:7.4f}")
    lines = []
    for p in laws.PARAMS:
        k, c, miss = lw[p]
        va, vb = k * a + c, k * b + c
        lines.append(f"{laws.NAMES[p]} {va:.4f} -> {vb:.4f}"
                     + ("" if abs(vb - va) > 1e-4 else " (constant)")
                     + f" (worst {miss:.4f} off the line)")
    out.append(f"  linear from {ax}={a:.4f} to {ax}={b:.4f}: " + "; ".join(lines))
    out.append("  build it: sketch ... --op loft --law --use "
               f"{ring.get('name', '?')} (a ruled loft between the family at the two levels)")
    return out


def _taper_measure(args, mesh, spec, ax, outer, holes, level, cut_at, target, marks):
    """`--taper`: the draft angle between the profile at --at and the one at
    --to, measured, never typed. The far profile must be the near one offset by
    a constant -- what a draft is -- or this refuses and says by how much it is
    not. Drafted pockets and sockets are `extrude(taper=...)`, not a loft or a
    stair-step."""
    import math

    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    if args.both or args.split or holes:
        raise SystemExit("--taper drafts one outline from --at to --to: not with "
                         "--both, --split or hole rings in --use (build those as "
                         "their own features)")
    near = unary_union([Polygon(_ring_points(r)) for r in outer]).buffer(0)
    side = "below" if target > level else "above"
    _, far_cut, _ = measure_offset(mesh, args.plane, target, marks, args.samples,
                                   side=side)
    # Like with like: the profile is one ring, filled (an opening named as a
    # cutter is filled too), so the far one is the ring of the far section --
    # an outline or a hole, filled -- that best overlaps it. Taking the far
    # polygons as they come would compare a filled outline with a wall's
    # annulus and "find" a large offset.
    rings_far = [Polygon(r.coords) for r in _rings_like(outer[0], section_at(mesh, args.plane, far_cut))]
    rings_far = [r.buffer(0) for r in rings_far if r.area > 0]

    def iou(r):
        u = r.union(near).area
        return r.intersection(near).area / u if u > 0 else 0.0
    far = max(rings_far, key=iou) if rings_far else Polygon()
    if far.is_empty or iou(far) <= 0.0:
        raise SystemExit(f"--taper: nothing at {ax}={far_cut:.4f} (measured "
                         f"inside --to) lies over the profile, so there is no draft "
                         f"to measure")
    h = abs(far_cut - cut_at)
    size = max(near.bounds[2] - near.bounds[0], near.bounds[3] - near.bounds[1])
    lo, hi = -size, size
    for _ in range(60):            # offset with the far profile's area
        mid = 0.5 * (lo + hi)
        a = near.buffer(-mid, join_style="mitre").area
        lo, hi = (mid, hi) if a > far.area else (lo, mid)
    d = 0.5 * (lo + hi)
    fit = near.buffer(-d, join_style="mitre")
    miss = fit.boundary.hausdorff_distance(far.boundary) if not fit.is_empty else size
    tol = max(3.0 * max((r.get("tol") or 0.0) for r in outer), 0.002 * size)
    if miss > tol + 0.05 * abs(d):
        raise SystemExit(f"--taper: the profile at {ax}={far_cut:.4f} is not the one "
                         f"at {ax}={cut_at:.4f} offset by a constant -- after the best "
                         f"offset ({d:+.4f} mm) they still differ by up to {miss:.4f} "
                         f"mm. That is not a draft: loft it (--op loft)")
    angle = round(math.degrees(math.atan2(d, h)), 4)
    if abs(angle) < 1e-3:
        print(f"note: --taper found no draft: the profiles at {ax}={cut_at:.4f} and "
              f"{ax}={far_cut:.4f} agree to {miss:.4f} mm; built straight")
        return None
    print(f"taper: {angle:g} deg {'inward' if angle > 0 else 'outward'} -- the profile "
          f"at {ax}={far_cut:.4f} is the one at {ax}={cut_at:.4f} offset {d:+.4f} mm "
          f"over {h:.4f} mm, to within {miss:.4f} mm")
    return angle


def _helix_profile(args, mesh, ring, level, spec, helix):
    """(profile in the helix's local frame, z0, theta, r_mean, r_max, centre,
    axis) for a ring measured on a section plane that contains the helix axis."""
    from mesh2cad.emit import HELIX_FRAMES, arc_midpoint
    axis = helix["axis"]
    M = np.asarray(HELIX_FRAMES[axis], float)
    k = "XYZ".index(axis)
    # The helix's own axis (a body's, not the part's middle), else the
    # part's middle.
    centre = (np.asarray(helix["centre"], float) if helix.get("centre")
              else mesh.bounds.mean(axis=0).copy())
    centre[k] = 0.0
    if spec["axis"] == k:
        raise SystemExit(f"--op sweep along H0 wants the profile from a section that "
                         f"CONTAINS the helix axis ({axis}); {args.plane} is across it")

    def local(uv):
        w = np.zeros(3)
        w[spec["u"]], w[spec["v"]], w[spec["axis"]] = uv[0], uv[1], level
        return M.T @ (w - centre)
    if len(ring["segments"]) < 2:
        raise SystemExit(f"--op sweep: ring {ring.get('name', '?')} is a sliver "
                         f"({len(ring['segments'])} segment), not a profile")
    pts = [local(s["from_xy"]) for s in ring["segments"]]
    theta = float(np.degrees(np.arctan2(np.mean([p[1] for p in pts]),
                                        np.mean([p[0] for p in pts]))))
    c, s_ = np.cos(np.radians(theta)), np.sin(np.radians(theta))

    def rz(uv):
        p = local(uv)
        return (float(p[0] * c + p[1] * s_), float(p[2]))
    prof = [("move", [rz(ring["segments"][0]["from_xy"])])]
    for s in ring["segments"]:
        if s["kind"] == "arc":
            prof.append(("arc", [rz(arc_midpoint(s)), rz(s["to_xy"])]))
        else:
            prof.append(("line", [rz(s["to_xy"])]))
    rs = [pt[0] for _, ps in prof for pt in ps]
    if min(rs) < -1e-6:
        raise SystemExit("--op sweep: the profile crosses the helix axis; cut it at "
                         "the axis (--split) first")
    z0 = float(np.mean([pt[1] for _, ps in prof for pt in ps]))
    return prof, z0, theta, float(np.mean(rs)), float(max(rs)), centre, axis


# How far a round wire's arcs may disagree, as a fraction of their median
# radius, and still be read as pieces of one circle. The section fits a clean
# wire to microns (0.1% here is already loose); a thread's meridian mixes
# 0.3 mm root fillets with 12 mm flanks.
ONE_CIRCLE_TOL = 0.05


def _one_circle(arcs, name):
    """Refuse a round-wire sweep whose profile is not one circle.

    The coil sweep reads the wire's radius and its distance from the axis as
    the mean over the ring's arcs. A meridian ring that is a whole thread
    profile -- dozens of arcs, fillets to flanks, centres on both sides of the
    axis -- would put the coil radius near 0 and nothing would say so. A ring
    whose arcs are not one circle is a profile, and the profile sweep takes its
    shape as it is.
    """
    radii = np.array([s_["radius"] for s_ in arcs], float)
    centres = np.array([s_["centre"] for s_ in arcs], float)
    r_med = float(np.median(radii))
    spread_r = float(radii.max() - radii.min())
    spread_c = float(np.max(np.linalg.norm(centres - np.median(centres, axis=0),
                                           axis=1)))
    if max(spread_r, spread_c) <= ONE_CIRCLE_TOL * r_med:
        return
    raise SystemExit(
        f"sweep refused: a round wire is one circle, and the {len(arcs)} arc(s) of "
        f"ring {name} are not -- radii {radii.min():.4f} to {radii.max():.4f}, "
        f"centres up to {spread_c:.4f} mm apart. Averaging them would give a wire "
        f"and a coil radius that are on no part of the mesh. This ring is a "
        f"profile, not a wire: to sweep its own shape along the helix, cut the "
        f"section through the axis at the core's radius (`--split`), name one "
        f"whole tooth (`--profile R#`) and the stretch of the axis it covers "
        f"(`--between K0,K1`).")


# Half-turns whose pitches agree this well are one region of the coil.
PITCH_SAME = 0.02


def _pitch_regions(args, mesh, ring, level, spec, helix, cut_at, r_mean):
    """[(pitch, anchor profile, z lo, z hi)] of the coil, region by region,
    from where the section crosses it. A plane containing the axis cuts a coil
    once a half-turn, alternately either side of the axis; the crossings of the
    same ring as the named one, in height order, are the coil's own pitch
    half-turn by half-turn -- closed end turns are tighter than the body, and
    one pitch for the whole coil leaves part of it unbuilt. Runs of half-turns
    of one pitch are one region, swept from its own measured crossing."""
    from shapely.geometry import Polygon
    _, rings, _, _ = _rings_at(mesh, args.plane, cut_at, args.tol, args.grid)
    area = Polygon(readback.outline(ring)).area
    # Every region sweeps the named ring itself, moved to its crossing: a
    # crossing on the far side of the axis is that ring mirrored in the plane
    # (a half-turn about the axis maps x to -x there), and wound the other way
    # it built its coil inside out.
    named, z_n, _, _, rx_n, _, _ = _helix_profile(args, mesh, ring, level, spec, helix)
    cross = []
    for r in rings:
        if r.get("role") != ring.get("role"):
            continue
        a = Polygon(readback.outline(r)).area
        # down to a third: a ground end turn is cut where it is half ground away
        if not (0.3 * area <= a <= 1.6 * area):
            continue
        try:
            _, z0, theta, rm, _, _, _ = _helix_profile(args, mesh, r, level, spec, helix)
        except SystemExit:
            continue
        if abs(rm - r_mean) <= 0.2 * r_mean:
            moved = [(op, [(q[0], q[1] - z_n + z0) for q in ps]) for op, ps in named]
            cross.append((z0, theta, moved, rx_n))
    cross.sort(key=lambda c: c[0])
    if len(cross) < 3:
        raise SystemExit(f"--along {args.along}: the section crosses the coil {len(cross)} "
                         f"time(s) beside {ring.get('name', '?')}; @all reads the pitch "
                         f"between crossings and needs three. Use H0@body or H0@ends")
    # each step: half a turn when it changes side, a whole one when it does not
    half = []
    for a, b in zip(cross, cross[1:], strict=False):
        turn = 0.5 if abs(((a[1] - b[1]) + 180.0) % 360.0 - 180.0) > 90.0 else 1.0
        half.append((b[0] - a[0]) / turn)
    regions, start = [], 0
    for i in range(1, len(half) + 1):
        if i == len(half) or abs(half[i] - half[start]) > PITCH_SAME * abs(half[start]):
            ps = half[start:i]
            regions.append((float(np.mean(ps)), cross[start], cross[start][0], cross[i][0]))
            start = i
    return regions


# Of a pitch: vertices this near one radius are on the thread's tip line, and
# a window this far off a whole pitch is no tooth.
TOOTH_SPAN = 0.15
# Of a pitch: left between neighbouring turns, which would otherwise touch.
TOOTH_GAP = 0.01
# Of the depth: a cut's groove reaches this far past the tips, into the bore.
TOOTH_OVER = 0.05
# Of a pitch: a cut's profile swept as given must leave at least this between
# its turns, or the thin web left is pierced once a turn and the cut opens new
# through-holes.
CUT_WEB = 0.05


def _tips(pts, r, tol):
    """The middles along the axis of each run of ring vertices on radius r: a
    thread's tip is a land, and its two corners are one tip."""
    on = np.abs(pts[:, 0] - r) <= tol
    out, run = [], []
    for z, hit in zip(pts[:, 1], on, strict=True):
        if hit:
            run.append(z)
        elif run:
            out.append(0.5 * (min(run) + max(run)))
            run = []
    if run:
        out.append(0.5 * (min(run) + max(run)))
    return sorted(out)


def _one_tooth(prof, pitch, cut):
    """([one tooth's profile], teeth, (lo, hi) of its window, root, crest, its
    word) of a ring along a thread, or None if it isn't one: tips along its
    inner or outer edge, a pitch apart -- e.g. an internal thread, the inner
    edge of a wall's section, which a sweep can take without the model
    computing a --split at the root and the crest.

    The thread's depth is what the ring's edge does between two tips, whatever
    else the ring holds. For a cut the window runs from tip to tip and the tool
    is the groove, the empty part of the window; for an add, from groove to
    groove, and the tool is the tooth."""
    import shapely
    from shapely.geometry import Polygon, box
    pts = np.array([pt for _, ps in prof for pt in ps], float)
    ring = Polygon(pts).buffer(0)
    if ring.is_empty or ring.geom_type != "Polygon":
        return None
    best = None
    for tip_r in (float(pts[:, 0].min()), float(pts[:, 0].max())):
        tips = _tips(pts, tip_r, TOOTH_SPAN * pitch)
        near = [i for i in range(len(tips) - 1)
                if abs(tips[i + 1] - tips[i] - pitch) <= TOOTH_SPAN * pitch]
        if near and (best is None or len(near) > len(best[2])):
            best = (tip_r, tips, near)
    if best is None:
        return None
    tip_r, tips, near = best
    middle = 0.5 * (tips[0] + tips[-1])
    i = min(near, key=lambda j: abs(0.5 * (tips[j] + tips[j + 1]) - middle))
    a, b = float(tips[i]), float(tips[i + 1])
    inside = pts[(pts[:, 1] > a) & (pts[:, 1] < b), 0]
    if not len(inside):
        return None
    crest = float(inside[np.argmax(np.abs(inside - tip_r))])
    depth = abs(crest - tip_r)
    if depth <= 0:
        return None
    gap = TOOTH_GAP * pitch
    if cut:
        over = TOOTH_OVER * depth * (-1.0 if crest > tip_r else 1.0)
        window = box(min(tip_r + over, crest), a + gap, max(tip_r + over, crest), b - gap)
        shape, what = window.difference(ring), "groove"
    else:
        mid = 0.5 * (a + b)
        a, b = mid - 0.5 * pitch, mid + 0.5 * pitch
        window = box(min(tip_r, crest), a + gap, max(tip_r, crest), b - gap)
        shape, what = ring.intersection(window), "tooth"
    if shape.geom_type != "Polygon":
        parts = [g for g in getattr(shape, "geoms", []) if g.geom_type == "Polygon"]
        if not parts:
            return None
        shape = max(parts, key=lambda g: g.area)
    out = list(shapely.simplify(shape, 1e-4 * depth).exterior.coords)[:-1]
    tooth = [("move", [out[0]])] + [("line", [p]) for p in out[1:]] + [("line", [out[0]])]
    return tooth, len(tips), (a, b), tip_r, crest, what


def _groove_beside(prof, pitch):
    """([the groove's profile], (lo, hi) of its window, tip, base) beside a
    lone tooth ring, or None if the ring doesn't narrow towards one tip line.
    Sweeping the tooth itself -- the target's own material -- with --combine
    cut would remove the teeth and leave the grooves, and nothing would say why.

    The window runs from the end of the tooth's land at its tip to the start
    of the land of the tooth a pitch below, and from past the tip (TOOTH_OVER
    of the depth, into the air, so no edge lies on the bore) to the base; the
    groove is the window less the two teeth."""
    import shapely
    from shapely import affinity
    from shapely.geometry import Polygon, box
    pts = np.array([pt for _, ps in prof for pt in ps], float)
    tooth = Polygon(pts).buffer(0)
    if tooth.is_empty or tooth.geom_type != "Polygon":
        return None
    r0, r1 = float(pts[:, 0].min()), float(pts[:, 0].max())
    depth = r1 - r0
    if depth <= 0:
        return None

    def land(r):
        on = pts[np.abs(pts[:, 0] - r) <= 1e-3 * depth, 1]
        return (float(on.min()), float(on.max())) if len(on) else None

    l0, l1 = land(r0), land(r1)
    if l0 is None or l1 is None or abs((l0[1] - l0[0]) - (l1[1] - l1[0])) <= 0.05 * pitch:
        return None
    (tip, (a, b)), base = ((r0, l0), r1) if l0[1] - l0[0] < l1[1] - l1[0] else ((r1, l1), r0)
    lo, hi = b - pitch, a
    if hi - lo <= 0:
        return None
    over = TOOTH_OVER * depth * (-1.0 if tip < base else 1.0)
    win = box(min(tip + over, base), lo, max(tip + over, base), hi)
    shape = win.difference(tooth).difference(affinity.translate(tooth, 0.0, -pitch))
    if shape.geom_type != "Polygon":
        parts = [g for g in getattr(shape, "geoms", []) if g.geom_type == "Polygon"]
        if not parts:
            return None
        shape = max(parts, key=lambda g: g.area)
    out = list(shapely.simplify(shape, 1e-4 * depth).exterior.coords)[:-1]
    groove = [("move", [out[0]])] + [("line", [q]) for q in out[1:]] + [("line", [out[0]])]
    return groove, (lo, hi), tip, base


def _target_material(args, ring):
    """True when `ring` is material of the target: an outer ring of a section of
    the target itself, not of a residual export (whose rings are its air)."""
    mesh = getattr(args, "mesh", None)
    return (ring.get("was_role", ring.get("role", "outer")) == "outer"
            and (mesh is None or export_source(mesh) is None))


def _sweep_profile(args, mesh, ring, level, spec, ax, cut_at, snapped, marks):
    """`--op sweep --along H0` of any profile through a band of the helix axis
    -- the tooth of a worm, a thread, a rib wound round a shaft."""
    from mesh2cad.emit import HELIX_FRAMES, build_profile_sweep
    hname = (args.along or "H0").split("@")[0]
    helix = _helix_fit(mesh, hname)
    if not helix.get("is_helix"):
        raise SystemExit(f"no helix {hname} was fitted on this mesh. "
                         + (f"({helix.get('reason')}) " if helix.get("reason") else "")
                         + f"The other paths are W#: {_known_paths(args, mesh)}")
    want = (args.along or "H0").split("@")[1:] or ["body"]
    if want[0] not in ("body", "ends", "all"):
        raise SystemExit(f"{hname} has @body, @ends and @all; {want[0]!r} is none of them")
    pitch = helix["pitch_whole_part"] if want[0] == "ends" else helix["pitch"]
    lefthand = helix.get("handedness", "right") != "right"
    prof, z0, theta, r_mean, r_max, centre, axis = _helix_profile(
        args, mesh, ring, level, spec, helix)
    zs = [pt[1] for _, ps in prof for pt in ps]
    tall, cut = max(zs) - min(zs), getattr(args, "combine", "add") == "cut"
    name = ring.get("name", "?")
    if cut and tall < abs(pitch) and _target_material(args, ring):
        lone = _groove_beside(prof, abs(pitch))
        if lone is None:
            print(f"note: {name} is material of the target, swept as given: a cut of it "
                  f"removes the target's own material. For a thread's groove cut the air "
                  f"beside the tooth (a ring of a residual export), or --split past the root "
                  f"so the ring holds whole teeth")
        else:
            prof, (lo, hi), tip, base = lone
            tall = hi - lo
            rs = [pt[0] for _, ps in prof for pt in ps]
            z0 = float(np.mean([pt[1] for _, ps in prof for pt in ps]))
            r_mean, r_max = float(np.mean(rs)), float(max(rs))
            print(f"note: {name} is one tooth of the target (its own material, narrowing to "
                  f"its tip at R{tip:.3f}): a cut of it would take the teeth away, so the "
                  f"groove beside it is swept instead, {helix['axis']} {lo:.2f}..{hi:.2f}, "
                  f"tip R{tip:.3f} to R{base:.3f}")
    elif cut and (1.0 - CUT_WEB) * abs(pitch) < tall < abs(pitch):
        raise SystemExit(f"--op sweep --combine cut: ring {name} is {tall:.4f} mm tall along "
                         f"{helix['axis']} of {hname}'s {abs(pitch):g} mm pitch, so the web "
                         f"between its turns would be {abs(pitch) - tall:.4f} mm (under "
                         f"{CUT_WEB:g} of a pitch): the cut pierces it, one hole a turn. Cut "
                         f"the target's own section along its helix (a ring split past the "
                         f"root holds whole teeth, and the tool takes one groove), or a ring "
                         f"that leaves room between its turns")
    if tall >= abs(pitch):
        # A profile several pitches tall -- a nut's wall, not a tooth -- winds
        # into turns that overlap, and no boolean can use the coil.
        got = _one_tooth(prof, abs(pitch), getattr(args, "combine", "add") == "cut")
        if got is None:
            raise SystemExit(f"--op sweep: ring {ring.get('name', '?')} spans "
                             f"{max(zs) - min(zs):.4f} mm along {helix['axis']}, "
                             f"{(max(zs) - min(zs)) / abs(pitch):.1f} pitches of {hname} "
                             f"({abs(pitch):g} mm), with no teeth a pitch apart along it: "
                             f"wound round the axis its turns would overlap. A thread's "
                             f"profile is one tooth, less than a pitch tall: cut it out with "
                             f"--split at the root and the crest, or name the tooth ring")
        prof, n, (lo, hi), root, crest, what = got
        rs = [pt[0] for _, ps in prof for pt in ps]
        z0 = float(np.mean([pt[1] for _, ps in prof for pt in ps]))
        r_mean, r_max = float(np.mean(rs)), float(max(rs))
        print(f"note: {ring.get('name', '?')}: {n} teeth; swept {what} {helix['axis']} "
              f"{lo:.2f}..{hi:.2f}, root R{root:.3f}, crest R{crest:.3f}: one pitch of "
              f"{abs(pitch):g} mm, the only way round the axis that doesn't overlap")
    k = "XYZ".index(axis)
    lo_b, hi_b = (helix["extent"] if helix.get("extent")
                  else (float(mesh.bounds[0][k]), float(mesh.bounds[1][k])))
    if getattr(args, "between", None):
        plane_a = {0: "YZ", 1: "XZ", 2: "XY"}[k]
        amarks, _ = _levels(args, mesh, plane=plane_a)
        names = [n.strip() for n in args.between.split(",") if n.strip()]
        if len(names) != 2:
            raise SystemExit("--between wants two levels on the helix axis, e.g. K0,K3")
        za, zb = sorted(resolve_level(n, amarks, "--between", mesh, plane_a) for n in names)
    else:
        za, zb = lo_b, hi_b
    # The levels are world coordinates along the axis; the local frame's z is
    # that coordinate less nothing (centre[k] = 0), whatever the frame's sign.
    M = np.asarray(HELIX_FRAMES[axis], float)
    sgn = float(M[k, 2])
    if want[0] == "all" and not getattr(args, "between", None):
        # the whole coil, end turns and all: the fit's extent is the body's
        za, zb = float(mesh.bounds[0][k]), float(mesh.bounds[1][k])
    za, zb = sorted((za * sgn, zb * sgn))
    if want[0] == "all":
        _sweep_regions(args, mesh, ring, level, spec, helix, cut_at, r_mean, r_max, lefthand,
                       (za, zb), sgn, axis, centre, ax, snapped)
        return
    code = build_profile_sweep(prof, pitch, lefthand, z0, (za, zb), axis, centre, theta,
                               r_mean, r_max)
    detail = (f"ring {ring.get('name', '?')} along {hname}@{want[0]} about {axis}: pitch "
              f"{abs(pitch):g} mm, {'left' if lefthand else 'right'}-handed, from "
              f"{axis}={min(za, zb) * sgn:.4f} to {axis}={max(za, zb) * sgn:.4f}, swept "
              f"over whole turns and trimmed")
    _finish_sketch(args, code, ax, level, cut_at, snapped, detail, "sweep",
                   profiles=f"profile {ring.get('name', '?')}")


def _sweep_regions(args, mesh, ring, level, spec, helix, cut_at, r_mean, r_max, lefthand,
                   band, sgn, axis, centre, ax, snapped):
    """--along H0@all: one coil per region of its own pitch, each trimmed to
    its stretch of the axis, from its own crossing, and unioned."""
    from mesh2cad.emit import build_profile_sweep
    regions = _pitch_regions(args, mesh, ring, level, spec, helix, cut_at, r_mean)
    za, zb = band
    # A closed end turn is wound tighter than the profile is tall: no coil of
    # it can be swept (its turns overlap). Its stretch goes to its neighbour.
    tall = max(max(pt[1] for _, ps in r[1][2] for pt in ps) - min(pt[1] for _, ps in r[1][2]
                                                                  for pt in ps)
               for r in regions)
    closed = [r for r in regions if abs(r[0]) <= 1.05 * tall]
    regions = [r for r in regions if abs(r[0]) > 1.05 * tall]
    if not regions:
        raise SystemExit(f"--along {args.along}: every stretch of the coil is wound tighter "
                         f"than {ring.get('name', '?')} is tall ({tall:.4f}); use H0@body")
    # region i ends, and i + 1 begins, at the crossing between them
    cuts = [za] + [regions[i][3] for i in range(len(regions) - 1)] + [zb]
    parts, said = [], []
    for i, (pitch, (z0, theta, prof, _rx), _lo, _hi) in enumerate(regions):
        # neighbours overlap a little: meeting exactly on a plane, two coils of
        # slightly different lead left an open seam where they were unioned
        eps = 0.02 * abs(pitch)
        lo = cuts[i] - (eps if i else 0.0)
        hi = cuts[i + 1] + (eps if i < len(regions) - 1 else 0.0)
        if hi - lo <= 1e-6:
            continue
        code = build_profile_sweep(prof, pitch, lefthand, z0, (lo, hi), axis, centre, theta,
                                   r_mean, r_max)
        body = code.split("result = _solid")[0]
        parts.append(body + f"_part{i} = _solid\n")
        said.append(f"{abs(pitch):.4f} mm over {axis}={lo * sgn:.4f}..{hi * sgn:.4f}")
    if not parts:
        raise SystemExit(f"--along {args.along}: no region of the coil lies in the band")
    names = [ln.split(" = ")[0] for p_ in parts for ln in p_.splitlines()
             if ln.startswith("_part")]
    code = ("".join(parts) + "_solid = " + names[0]
            + "".join(f".union({n})" for n in names[1:]) + "\n"
            + "result = _solid\n" + '_STEPS = [("add", _solid)]')
    detail = (f"ring {ring.get('name', '?')} along {args.along} about {axis}: "
              f"{len(parts)} region(s) of the coil's own pitch, read off this section's "
              f"crossings of it -- " + "; ".join(said)
              + f", {'left' if lefthand else 'right'}-handed"
              + (f"; {len(closed)} closed turn stretch(es) (pitch "
                 + ", ".join(f"{abs(r[0]):.4f}" for r in closed)
                 + f" under the profile's {tall:.4f}) swept at their neighbour's pitch"
                 if closed else ""))
    _finish_sketch(args, code, ax, level, cut_at, snapped, detail, "sweep",
                   profiles=f"profile {ring.get('name', '?')}")


def _replaced_program(args):
    """The warning for an --emit that throws away the program already there.

    `sketch --emit P` without `--base P` writes the new feature alone, so a
    feature meant to join the first can replace it instead, and the model only
    finds out from the residual. The write still happens -- restarting a feature
    is sometimes the intent -- but it is said, with the way back. `edit` always
    starts from --built, so it is never warned.
    """
    if getattr(args, "base", None) or getattr(args, "built", None):
        return ""
    old = Path(args.emit)
    if not old.exists():
        return ""
    try:
        feats, _ = split_program(old.read_text())
    except ProgramError:
        return ""
    if not feats:
        return ""
    return (f"WARNING: {args.emit} held {len(feats)} feature(s) "
            f"({'; '.join(f'#{i} {describe(f)}' for i, f in enumerate(feats))}); "
            f"without --base this --emit replaced them with the new feature. If "
            f"you meant to add to them: `edit <mesh> --built {args.emit} --undo "
            f"--emit {args.emit}` brings them back, then `sketch ... --base "
            f"{args.emit}` adds to them.")


def _program_frame(args, info, base=None):
    """Set `args._frame`, the `_FRAME` the program written from this mesh
    declares, and -- for a residual export -- `args._check_mesh`, the target
    `--check` measures against.

    The frame is the one ingest put this mesh in, after the one a residual
    export's coordinates were already in; else the one `--base` declares. A
    residual export is millimetre-sized, so ingest leaves it alone, and a
    sketch on one would write no `_FRAME` at all: the part would be delivered
    at 10x or 100x its size.
    """
    own = ingest_frame(info)
    frame, notes = own, []
    up = export_source(args.mesh)
    args._check_mesh = args._check_name = None
    if up is not None:
        if up["frame"] is not None:
            frame = up["frame"] if own is None else own @ up["frame"]
        if own is None and up["source"] is not None:
            args._check_mesh, _ = ingest(str(up["source"]))
            args._check_name = up["named"]
        elif up["source"] is None:
            notes.append(f"note: {args.mesh} is a residual export of {up['named']}, "
                         f"which is not there any more, so --check measures "
                         f"against the export itself")
    if base and Path(base).is_file():
        try:
            declared = declared_frame(Path(base).read_text())
        except ValueError as exc:
            raise SystemExit(f"{base}: {exc}") from None
        if declared is not None and frame is None:
            frame = declared
            notes.append(f"note: {args.mesh} needed no frame at ingest; the program "
                         f"keeps the _FRAME of {base} "
                         f"({describe_program_frame(declared)}), the frame its "
                         f"features were measured in")
        elif declared is not None and not same_frame(declared, frame):
            why = (f"{args.mesh} is a residual export of {up['named']}, and "
                   f"ingest {describe_program_frame(own)} it again on its own"
                   if up is not None and own is not None else
                   f"{args.mesh} is {describe_program_frame(frame)} at ingest")
            raise SystemExit(
                f"refused: {base} is written in the frame "
                f"{describe_program_frame(declared)}, but {why}, so a feature "
                f"measured on it would not sit where the features before it do. "
                + (f"Sketch on {up['named']} instead: the leftover is where it was "
                   f"in the target, and the names section and ortho print there "
                   f"are the target's." if up is not None else
                   "Sketch on the mesh that program was built against, or start a "
                   "new program on this one (no --base)."))
        elif declared is None and frame is not None:
            notes.append(f"note: {base} declares no _FRAME; this write adds the "
                         f"target's ({describe_program_frame(frame)}), so export and "
                         f"scoring undo it")
    args._frame = frame
    for ln in notes:
        print(ln)


def _carry_frame(args, full):
    """`full` declaring the program's frame: no write drops it."""
    want = getattr(args, "_frame", None)
    have = declared_frame(full)
    if same_frame(have, want):
        return full
    if have is not None:
        raise SystemExit(f"refused: this program declares the frame "
                         f"{describe_program_frame(have)} and {args.mesh} is "
                         f"{describe_program_frame(want) if want is not None else 'in none'}"
                         f"; writing it would score it in the wrong one")
    body, show, tail = full.rpartition("show_object(result)")
    if not show:
        return full.rstrip() + frame_block(want) + "\n"
    return body.rstrip() + frame_block(want) + "\n\n" + show + tail


def _check_target(args):
    """The mesh `--check` measures against: the target a residual export came
    from, else the mesh the sketch was given."""
    got = getattr(args, "_check_mesh", None)
    return got if got is not None else getattr(args, "_target_mesh", None)


def _sliver_floor(target, vols):
    """The volume under which a solid of the build is a fragment, not a body:
    the floor `--check` already judges leftovers by (leftover.FLOOR of the
    target), and below half the target's smallest body when that one is smaller
    still, so a real small part is never one. None when an open target in
    pieces can't say how big its smallest is."""
    from mesh2cad import leftover
    if target is None:
        return leftover.FLOOR * sum(vols)
    if not target.is_watertight:
        return None if _target_bodies(target)[0] > 1 else leftover.FLOOR * sum(vols)
    bodies = [float(b.volume) for b in target.split() if b.volume > 0]
    return min(leftover.FLOOR * float(target.volume), 0.5 * min(bodies, default=np.inf))


def _sizes(vols):
    """"of 0.0102 mm^3 each", or the range when they differ."""
    lo, hi = min(vols), max(vols)
    if len(vols) > 1 and hi - lo <= 0.01 * hi:
        return f"of {hi:.3g} mm^3 each"
    return f"of {hi:.3g} mm^3" if lo == hi else f"of {lo:.3g}-{hi:.3g} mm^3"


def _drop_slivers(args, full):
    """(the program to write, its namespace, or None if it did not build). A
    written program leaves no solid below the sliver floor: a pattern's cut can
    leave tiny solids at pocket walls that the mesh and `--check` never see as
    bodies, but the delivered STEP counts. Worked out afresh at every write, so
    a later feature that joins a fragment gets it back."""
    src = without_slivers(full)
    ns = {"show_object": lambda o, *a, **k: None}
    try:
        exec(compile(src, "<sketch>", "exec"), ns)
        solids = ns["result"].solids().vals()
    except Exception:                                              # noqa: BLE001
        return full, None                  # `--check`, or the next read, says what broke
    vols = [float(s.Volume()) for s in solids]
    if len(vols) < 2:
        return src, ns
    target = _check_target(args)
    floor = _sliver_floor(target, vols)
    small = [] if floor is None else [s for s, v in zip(solids, vols, strict=True) if v < floor]
    if not small or max(vols) < floor:     # nothing to be a fragment of
        return src, ns
    block = sliver_block(len(small), floor)
    body, show, tail = src.rpartition("show_object(result)")
    src = body.rstrip() + "\n\n" + block + "\n" + show + tail
    exec(compile(block, "<slivers>", "exec"), ns)
    boxes = [s.BoundingBox() for s in small]
    where = ", ".join(f"{a} {min(lo):.4g}..{max(hi):.4g}" for a, lo, hi in (
        ("X", [b.xmin for b in boxes], [b.xmax for b in boxes]),
        ("Y", [b.ymin for b in boxes], [b.ymax for b in boxes]),
        ("Z", [b.zmin for b in boxes], [b.zmax for b in boxes])))
    print(f"note: dropped {len(small)} solid{'s' if len(small) > 1 else ''} "
          f"{_sizes([float(s.Volume()) for s in small])} at {where}: under the "
          f"{floor:.3g} mm^3 sliver floor, what a cut left beside the part, not a "
          f"part of it. A solid above the floor stays, and `--check` reports it")
    return src, ns


# Directions a cavity's skin is looked for along, with their names.
_SIDES = (("X+", (1, 0, 0)), ("X-", (-1, 0, 0)), ("Y+", (0, 1, 0)),
          ("Y-", (0, -1, 0)), ("Z+", (0, 0, 1)), ("Z-", (0, 0, -1)))
# Extra pieces, through-holes and cavities placed in one `--check`.
TOPOLOGY_SHOWN = 6


def _feature_states(full):
    """The build after each feature, for saying which one changed a count."""
    from mesh2cad.program import GLUE
    ns = {"show_object": lambda o, *a, **k: None, "_AFTER": []}
    exec(compile(full.replace(GLUE, "_AFTER.append(result)\n" + GLUE), "<states>", "exec"), ns)
    return ns["_AFTER"] + [ns["result"]]


def _skins(wp, void, target):
    """[(thickness, side, the point it ends at, its direction)] of the walls
    round a cavity that the target doesn't have: thinnest first."""
    from mesh2cad.topology import hits
    out = []
    for side, d in _SIDES:
        d = np.array(d, dtype=float)
        ts = hits(wp, void.centre, d)
        if len(ts) < 2:
            continue
        probes = np.array(void.centre) + np.outer(np.linspace(ts[0], ts[1], 5)[1:-1], d)
        if not target.contains(probes).any():
            out.append((ts[1] - ts[0], side, np.array(void.centre) + ts[1] * d, d))
    return sorted(out, key=lambda x: x[0])


def _mouth_face(args, target, point, along, reach):
    """The flat face of the target a hole comes out of at `point`, the one that
    `--to` can stop on: facing along the hole, with the most area within
    `reach`. None if there isn't one."""
    from mesh2cad.segment import segment_cached
    try:
        seg = segment_cached(target, cache_dir=_cache_dir(args))
    except Exception:                                              # noqa: BLE001
        return None
    near = (np.linalg.norm(target.triangles_center - point, axis=1) <= reach) & (
        target.face_normals @ along >= 0.7)
    best, most = None, 0.0
    for surf in seg.surfaces:
        if surf.kind != "plane":
            continue
        area = float(target.area_faces[np.asarray(surf.faces)][near[np.asarray(surf.faces)]].sum())
        if area > most:
            best, most = surf.name, area
    return best


def _first_feature(states, found):
    """The first feature after which `found(topology)` holds, staying so to the
    end, or None."""
    k = None
    for i, t in enumerate(states):
        k = (i if k is None else k) if found(t) else None
    return k


def _where(pts):
    """"X a..b, Y c..d, Z e..f" of a set of (low, high) pairs."""
    lo, hi = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    return ", ".join(f"{a} {x:.4g}..{y:.4g}" for a, x, y in zip("XYZ", lo.min(axis=0),
                                                                hi.max(axis=0), strict=True))


def _at(p):
    return "(" + ", ".join(f"{round(float(x), 3) + 0.0:.5g}" for x in p) + ")"


def _extra_solids(now, want, tops):
    """Lines for the solids past the target's count, grouped by the feature
    that left them."""
    extra = sorted(now.solids, key=lambda s: -s.volume)[want:]
    groups = {}
    for s in extra:
        k = _first_feature(tops, lambda t, s=s: any(
            abs(o.volume - s.volume) <= 0.01 * s.volume and np.allclose(o.low, s.low, atol=1e-3)
            for o in t.solids))
        groups.setdefault(k, []).append(s)
    return [(f"  {len(g)} extra solid{'s' if len(g) > 1 else ''} "
             f"{_sizes([s.volume for s in g])} at {_where([(s.low, s.high) for s in g])}"
             + (f": feature #{k} left {'them' if len(g) > 1 else 'it'}" if k is not None else "")
             + ". `--emit` drops solids under the sliver floor; a bigger one has to be fused "
               "to the part or removed with `edit --drop`") for k, g in groups.items()]


def _features_at(feats, pts):
    """Which of the features' own solids the points `pts` lie on."""
    import cadquery as cq
    near = []
    probe = [cq.Vertex.makeVertex(*map(float, p)) for p in pts[:: max(1, len(pts) // 4)]]
    for j, f in enumerate(feats):
        ns = {"show_object": lambda o, *a, **k: None}
        try:
            exec(compile("import cadquery as cq\n" + f, f"<feature {j}>", "exec"), ns)
            own = ns["result"].val()
            if any(own.distance(v) <= STL_TOL for v in probe):
                near.append(j)
        except Exception:                                          # noqa: BLE001
            continue
    return near


def _extra_holes(now, want, states, tops, feats):
    """Lines for the through-holes features added past the target's count: the
    rims new after each feature that took the count above it."""
    from mesh2cad.topology import outer_rims
    seen = []
    for k, t in enumerate(tops):
        if t.holes <= want or (k and t.holes <= tops[k - 1].holes):
            continue
        old = outer_rims(states[k - 1]) if k else []
        for _length, mid, half, pts in outer_rims(states[k]):
            span = float(np.linalg.norm(half)) + 1e-6
            if any(np.linalg.norm(np.subtract(mid, o[1])) <= 0.05 * span + 1e-3 for o in old):
                continue
            seen.append((k, mid, half, [j for j in _features_at(feats, pts) if j != k]))
    lines = [f"  a through-hole at {_at(mid)}, {2 * max(half):.3g} mm across its rim: "
             f"feature #{k} opened it"
             + (f", where it meets #{', #'.join(map(str, meet))}" if meet else "")
             for k, mid, half, meet in seen[:TOPOLOGY_SHOWN]]
    if len(seen) > TOPOLOGY_SHOWN:
        lines.append(f"  and {len(seen) - TOPOLOGY_SHOWN} more")
    missing = now.holes - want - len(seen)
    if missing > 0:
        lines.append(f"  {missing} more not placed: no closed run of edges found around "
                     f"{'them' if missing > 1 else 'it'}")
    return lines


def _extra_voids(args, now, wp, tops, target, feats):
    """Lines for the cavities the target doesn't have, each with the feature
    that closed it and the skin to open. `--op hole` takes its depth from the
    stack and no --to (013: `--to F6` changed nothing), so the advice for one
    is a cut that can take it."""
    lines = []
    for v in now.voids[:TOPOLOGY_SHOWN]:
        k = _first_feature(tops, lambda t, v=v: any(
            all(a <= c <= b for a, b, c in zip(o.low, o.high, v.centre, strict=True))
            for o in t.voids))
        by = f"feature #{k}" if k is not None else "a feature"
        skins = _skins(wp, v, target)
        if not skins:
            lines.append(f"  a cavity of {v.volume:.1f} mm^3 at {_at(v.centre)}, from {by}: "
                         f"the target has material all round it, so it isn't a hole stopped "
                         f"short. Fill it or cut through")
            continue
        reach = float(np.linalg.norm(np.subtract(v.high, v.low)))
        faces = [_mouth_face(args, target, s[2], s[3], reach) for s in skins]
        sides = " and ".join(f"{s[1]} ({f})" if f else s[1]
                             for s, f in zip(skins, faces, strict=True))
        named = [f for f in faces if f]
        hole = k is not None and k < len(feats) and "hole" in describe(feats[k])
        reach_it = f"`--to {named[0]}`" if named else "through that skin"
        lines.append(f"  a cavity of {v.volume:.1f} mm^3 at {_at(v.centre)}: {by} stops short, "
                     f"under a {skins[0][0]:.2g} mm skin on {sides}, where the target's hole "
                     f"runs through. "
                     + (f"`--op hole` stops at its stack's ends and takes no --to: `--replace "
                        f"{k}` with a cut of its ring, {reach_it}" if hole else
                        (f"{reach_it} opens it" if named else f"Extend the cut {reach_it}")))
    return lines


def _open_shell(now, tops):
    """A line for edges with one face, or wires that don't close: a topology
    scorer can't read such a B-rep and reads the mesh instead."""
    k = _first_feature(tops, lambda t: t.unreadable)
    return [f"  {now.unreadable}"
            + (f" at {_where([(p, p) for p in now.open_edges])}" if now.open_edges else "")
            + (f": feature #{k} left it" if k is not None else "")
            + ". OCC calls it valid, but the benchmark can't read this B-rep and falls back "
              "to the mesh. `--replace` that feature"]


TOPOLOGY_WORDS = ("bodies", "through-holes", "cavities")


def _topology_of(src, ref):
    """(pieces, through-holes, cavities) of program `src`: as recorded with its
    version in `ref`'s history when it is there, else read off its build. None
    when it doesn't build."""
    from mesh2cad.program import history_dir
    from mesh2cad.topology import read
    idx = history_dir(ref) / "index.jsonl" if ref else None
    if idx is not None and idx.is_file():
        import json
        for row in reversed([json.loads(ln) for ln in idx.read_text().splitlines()
                             if ln.strip()]):
            f = idx.parent / f"{row.get('version', 0):03d}.py"
            if row.get("topology") and f.is_file() and f.read_text() == src:
                return tuple(row["topology"])
    try:
        ns = {"show_object": lambda o, *a, **k: None}
        exec(compile(src, "<before>", "exec"), ns)
        t = read(ns["result"], quick=True)
        return (t.pieces, t.holes, t.cavities)
    except Exception:                                              # noqa: BLE001
        return None


def _topology_moved(now, target, before, saved):
    """The line for a write that took the counts further from the target's than
    the version before it had them, or None -- extra through-holes are not
    "boolean artifacts", and must not be delivered as if they were."""
    if before is None:
        return None

    def off(c):
        return sum(abs(a - b) for a, b in zip(c, target, strict=True))
    if off(now) <= off(before):
        return None
    moved = [f"{w} {b} -> {n}, target {t}"
             for w, n, b, t in zip(TOPOLOGY_WORDS, now, before, target, strict=True)
             if abs(n - t) > abs(b - t)]
    if not moved:
        return None
    back = ", ".join(f"{w} {b}" for w, n, b, t in zip(TOPOLOGY_WORDS, now, before, target,
                                                      strict=True) if abs(n - t) > abs(b - t))
    how = ("`edit --undo` restores " if saved else "the program it was built on had ") + back
    return (f"topology   this write took {'; '.join(moved)}: further from the target than "
            f"the version before. {how}")


def _topology_lines(args, full, ns, target, before=None, out=None):
    """The build's pieces, through-holes and cavities against the target's,
    with each excess placed and tied to a feature. Fewer of anything is no
    fault here: every step of a build has fewer. A failure to read says so and
    costs the check nothing. `before` (the program this write replaces, and
    where its history is) adds a line when the counts moved away from the
    target's; `out` gets the counts, for the history."""
    from mesh2cad.topology import mesh_counts, read
    try:
        wp, now = ns["result"], read(ns["result"])
        counts = mesh_counts(target) if target is not None else None
        pieces = _target_bodies(target)[0] if counts is None else counts[0]
        if out is not None:
            out["topology"] = [now.pieces, now.holes, now.cavities]
        head = [f"bodies {now.pieces}, target {pieces}"]
        if counts is None:
            head += [f"through-holes {now.holes}", f"cavities {now.cavities}"]
        else:
            head += [f"through-holes {now.holes}, target {counts[1]}",
                     f"cavities {now.cavities}, target {counts[2]}"]
        lines = ["topology   " + "; ".join(head)
                 + (" (the target isn't closed, so it has no holes to count)"
                    if counts is None else "")]
        if counts is not None and before is not None and before[0] is not None and \
                (now.pieces, now.holes, now.cavities) != tuple(counts):
            moved = _topology_moved((now.pieces, now.holes, now.cavities), tuple(counts),
                                    _topology_of(*before[:2]), before[2])
            if moved:
                lines.append(moved)
        extra = now.pieces > pieces
        holes = counts is not None and now.holes > counts[1]
        voids = counts is not None and now.cavities > counts[2]
        if not (extra or holes or voids or now.unreadable or now.inner):
            return lines
        states = _feature_states(full)
        tops = [read(s, quick=True) for s in states]
        if extra:
            lines += _extra_solids(now, pieces, tops)
        if holes:
            feats, _ = split_program(full)
            lines += _extra_holes(now, counts[1], states, tops, feats)
        if voids:
            lines += _extra_voids(args, now, wp, tops, target, split_program(full)[0])
        if now.unreadable:
            lines += _open_shell(now, tops)
        for v in now.inner:
            k = _first_feature(tops, lambda t, v=v: any(
                all(a <= c <= b for a, b, c in zip(o.low, o.high, v.centre, strict=True))
                for o in t.voids))
            lines.append(f"  an inner shell of {v.volume:.1f} mm^3 at {_where([(v.low, v.high)])}"
                         + (f", from feature #{k}" if k is not None else "")
                         + ": it touches the outer shell, so the benchmark counts no cavity. "
                           "`--replace` that feature if the space should be open or solid")
        return lines
    except Exception as exc:                                       # noqa: BLE001
        return [f"topology   NOT READ: {type(exc).__name__}: {exc}"]


def _show_program(args, full):
    """The whole program, only when asked for.

    Printed on every call, it would be most of every sketch output and a large
    share of the tool text in a long prompt. The model works with names and
    never reads it. The file holds it.
    """
    if getattr(args, "show_program", False):
        print("\n" + full)
    elif not args.emit:
        print("program not printed and not saved: --emit keeps it, "
              "--show-program prints it")


def _feature_line(full, index, grew):
    """`feature: #4 cut extrude XY Z 0.00->5.00, +312.000 mm^3`, or None.

    `index` is the feature this command wrote, -1 for the last one. `grew` is the
    change in volume against the version before it, None when that is unknown.
    """
    try:
        feats, _ = split_program(full)
    except ProgramError:
        return None
    i = len(feats) - 1 if index < 0 else index
    return (f"feature: #{i} {describe(feats[i])}"
            + ("" if grew is None else f", {grew:+.3f} mm^3"))


def _save_and_check(args, full, note, feature=None):
    """Write `full` (with a history snapshot) if --emit, then measure it if
    --check. Shared by `sketch` and `edit`, so both report the same lines and
    agent_run's stopping rules read them the same way. `feature` is the index
    of the feature this command wrote, for the one-line summary."""
    full = _carry_frame(args, full)
    ns = None
    if args.emit:
        full, ns = _drop_slivers(args, full)
    saved_before = volume_before = grew = None
    ref = args.emit or getattr(args, "base", None) or getattr(args, "built", None)
    if ref:
        saved_before = last_residual(ref)
        volume_before = last_volume(ref)
    # The program this write replaces, to say when its counts were nearer
    # the target's
    was = next((Path(p) for p in (getattr(args, "base", None), args.emit)
                if p and Path(p).is_file()), None)
    before = (was.read_text(), args.emit or str(was), bool(args.emit)) if was else None
    if args.emit:
        replaced = _replaced_program(args)
        if replaced:
            # so the `--undo` the warning names has a version to go back to
            keep_current(args.emit, "before an --emit without --base")
        Path(args.emit).parent.mkdir(parents=True, exist_ok=True)
        Path(args.emit).write_text(full)
        n = snapshot(args.emit, full, note)
        print(f"wrote {args.emit}  (saved as version {n}; `edit --undo` restores "
              f"the version before it)")
        if replaced:
            print(replaced)
    elif args.check:
        # R2, and an expensive omission if missed. `--check` without `--emit`
        # measures a program that is never written, so a clean residual here says
        # nothing about the file on disk, which may still hold something much worse
        # from the command before.
        print("NOT SAVED: this checked the program without writing it. Any "
              "residual below describes that program, not a file. Re-run with "
              "--emit to keep it.")

    if args.check:
        target = _check_target(args)
        try:
            if ns is None:
                ns = {"show_object": lambda o, *a, **k: None}
                exec(compile(full, "<sketch>", "exec"), ns)
            solid = ns["result"].val()
            bb = solid.BoundingBox()
            if volume_before is not None:
                grew = solid.Volume() - volume_before
            print(f"\ncheck: builds OK   volume {solid.Volume():.3f} mm^3   "
                  f"bbox ({bb.xmin:.3f}, {bb.ymin:.3f}, {bb.zmin:.3f}) - "
                  f"({bb.xmax:.3f}, {bb.ymax:.3f}, {bb.zmax:.3f})")
            if getattr(args, "_check_name", None):
                print(f"check: measured against {args._check_name}, the target "
                      f"{args.mesh} was exported from by `residual --export`, so "
                      f"the residual is the whole part's")
            escaped = _outside_target(target, bb)
            if escaped:
                print("check: " + escaped)
            health = _mesh_health(ns["result"], ns.get("_STEPS"),
                                  target=target,
                                  levels=_named_levels(args))
            print("check: " + health)
            got = {}
            for ln in _topology_lines(args, full, ns, target, before, got):
                print("check: " + ln)
            same = _changed_nothing(ns)
            if same:
                print("check: " + same)
            line = _residual_line(target,
                                  ns.get("_STEPS"), result=ns.get("result"),
                                  cache_dir=_cache_dir(args))
            # The health line goes into the history with the residual: which
            # version a run delivers is chosen from these rows, and a version
            # the kernel rejects must never be chosen.
            if args.emit:
                record_residual(args.emit, line, health=health,
                                volume=round(solid.Volume(), 3), **got)
            if line:
                print("check: " + line)
                delta = delta_line(saved_before, line)
                if delta:
                    print("check: " + delta)
                surface = _surface_line(target, ns, line,
                                        cache_dir=_cache_dir(args))
                if surface:
                    print("check: " + surface)
                effect = _blend_effect(args, solid.Volume())
                if effect:
                    print("check: blends   " + effect)
                for ln in _blend_check(args):
                    print("check: blends   " + ln)
        except Exception as exc:                                   # noqa: BLE001
            print(f"\ncheck: FAILED  {type(exc).__name__}: {exc}")
            if args.emit:
                record_residual(args.emit, None,
                                check_failed=f"{type(exc).__name__}: {exc}"[:300])
    if feature is not None:
        line = _feature_line(full, feature, grew)
        if line:
            print(line)
    summary = summary_line(full)
    if summary:
        print(summary + "   (`--replace N` / `edit --drop N` address these)")



def _edit_repeat(args, mesh, feats, frame):
    """`edit --feature N --mirror PLANE | --pattern EXPR`: feature #N, whatever
    it is -- a cut too -- mirrored or repeated in place. The rest of the
    program is untouched."""
    if args.feature is None or not (args.mirror or args.pattern):
        raise SystemExit("edit: --feature N goes with --mirror PLANE or --pattern "
                         "EXPR, and each of those with --feature N")
    if args.drop is not None or args.undo:
        raise SystemExit("edit: --feature N with --mirror/--pattern is its own edit; "
                         "--drop and --undo are others")
    n = args.feature
    if not 0 <= n < len(feats):
        raise SystemExit(f"edit: there is no feature #{n}: the program has "
                         f"{len(feats)} (#0..#{len(feats) - 1})")
    f, what = feats[n], []
    if args.pattern:
        if ".." in args.pattern:
            k, step, label = _linear_step(args, mesh, args.pattern)
            f += linear_code(k, step)
        else:
            k, p0, p1 = parse_pattern(args.pattern)
            f += pattern_code(k, p0, p1)
            label = f"x{k} about {args.pattern.split('@', 1)[1]}"
        what.append(f"repeated {label}")
    if args.mirror:
        k, coord, label = _mirror_plane(args, mesh, args.mirror)
        f += mirror_code(k, coord)
        what.append(f"mirrored about {label}")
    lines = f.split("\n")
    head = max(i for i, ln in enumerate(lines[:12]) if ln.startswith("# ")) + 1
    lines.insert(head, f"# {', '.join(what)}")
    feats[n] = "\n".join(lines)
    try:
        full = join_program(feats, frame)
    except ProgramError as exc:
        raise SystemExit(f"edit: {exc}") from None
    print(f"edit: feature #{n} {describe(feats[n])} {', '.join(what)}; the other "
          f"features are unchanged" + ("" if args.emit else "  [preview: nothing saved]"))
    _show_program(args, full)
    _save_and_check(args, full, f"edit #{n} {', '.join(what)}", feature=n)


def cmd_edit(args):
    """List, drop or undo features of a program `sketch` wrote."""
    mesh, _info = ingest(args.mesh)
    args._target_mesh = mesh
    _program_frame(args, _info, args.built)
    src = Path(args.built).read_text()
    try:
        feats, _frame = split_program(src)
    except ProgramError as exc:
        raise SystemExit(f"edit: {exc}") from None

    if args.mirror or args.pattern or args.feature is not None:
        return _edit_repeat(args, mesh, feats, _frame)
    if args.drop is None and not args.undo:
        print(f"{args.built}: {len(feats)} feature(s), in build order "
              f"(each feature's own solid, before it is unioned):")
        for i, f in enumerate(feats):
            ns = {"show_object": lambda o, *a, **k: None}
            try:
                exec(compile("import cadquery as cq\n" + f, f"<feature {i}>",
                             "exec"), ns)
                sol = ns["result"].val()
                bb = sol.BoundingBox()
                own = (f"volume {sol.Volume():.4g} mm^3  bbox ({bb.xmin:.4g}, "
                       f"{bb.ymin:.4g}, {bb.zmin:.4g}) - ({bb.xmax:.4g}, "
                       f"{bb.ymax:.4g}, {bb.zmax:.4g})")
            except Exception as exc:                               # noqa: BLE001
                own = f"does not build on its own: {type(exc).__name__}"
            print(f"  #{i}  {describe(f)}\n       {own}")
        print("replace one with `sketch ... --base PROG --replace N --emit PROG`, "
              "remove one with `edit --drop N --emit PROG`")
        if args.check or args.emit:
            _save_and_check(args, src, "edit (list)")
        return

    if args.undo:
        target = args.emit or args.built
        try:
            full, ver = undo(target, commit=bool(args.emit))
        except ProgramError as exc:
            raise SystemExit(f"edit --undo: {exc}") from None
        print(f"undo: restoring version {ver} of {target}"
              + ("" if args.emit else "  [preview: history unchanged]"))
        note = f"edit --undo (restored version {ver})"
    else:
        try:
            full = drop_feature(src, args.drop)
        except ProgramError as exc:
            raise SystemExit(f"edit --drop: {exc}") from None
        print(f"drop: removed feature #{args.drop} ({describe(feats[args.drop])}); "
              f"the features after it are renumbered")
        note = f"edit --drop {args.drop}"
    _save_and_check(args, full, note)

def hi_v(marks, name):
    for m in marks:
        if m.get("name") == name:
            return float(m["value"])
    return 0.0


def _bands(spec_out, marks):
    """What the section looks like between each pair of adjacent levels.

    Read straight out of the sweep that produced the levels, so it costs
    nothing beyond a lookup per band.
    """
    prof = spec_out.get("profile") or []
    if not prof or len(marks) < 2:
        return []
    named = sorted((m for m in marks if m.get("name")),
                   key=lambda m: float(m["value"]))
    out = []
    for a, b in zip(named[:-1], named[1:], strict=True):
        lo, hi = float(a["value"]), float(b["value"])
        inside = [q for q in prof if lo < q["offset"] < hi]
        if not inside:
            continue
        mid = inside[len(inside) // 2]
        areas = [q["area"] for q in inside]
        out.append((a["name"], b["name"], mid["n"], mid["holes"], mid["area"],
                    min(areas), max(areas)))
    return out




def _split_levels(spec):
    """`--to` as a list of levels, or None when it names just one.

    Splits on commas at depth zero only. `mid(A,B)` is itself a level
    expression and contains a comma, so a plain split turned `mid(E0,E1),E1`
    into `mid(E0` and `E1)` and refused both -- the first test written against
    this feature caught it.
    """
    if spec is None:
        return None
    text = str(spec)
    parts, depth, cur = [], 0, []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur).strip())
    parts = [p for p in parts if p]
    return parts if len(parts) > 1 else None


def _centroid(r):
    return np.mean([sg["from_xy"] for sg in r["segments"]], axis=0)

def _region(r, section):
    """What a ring bounds itself in its section: its outline less every ring
    nested inside it -- an outer ring's own material, a hole's own opening
    (a hole used as a cutter's outline too). Rings nested deeper are inside
    those and go with them."""
    from shapely.ops import unary_union
    g = readback.shape(r)
    if g is None or not section:
        return g
    kids = []
    for c in section:
        if c is r or c.get("name") == r.get("name"):
            continue
        h = readback.shape(c)
        if h is not None and h.area < g.area and g.contains(h.representative_point()):
            kids.append(h)
    return g.difference(unary_union(kids)) if kids else g


def _match_profiles(named, available, what, ax, target, at=None, near=None, quiet=False):
    """The far counterpart of each profile the model actually named.

    Demanding that the far section have exactly as many loops as were named
    would make loft unusable whenever the feature is one loop among several --
    "a region where the loop count changes is more than one feature" is a true
    sentence about the section and a false one about the feature.

    Each named profile is paired with the unused far ring of its own role
    (a hole used as a cutter's outline pairs with a far hole) that overlaps
    it most, and the pairing is printed. Pairing by nearest centroid would take
    a big circle round the same axis for a small ring inside it. Rings that
    overlap nothing are paired by centroid, and an overlap not clearly better
    than the runner-up is said. What overlaps is what each ring bounds itself
    (`_region`; `near` is the named rings' own section, `available` the far
    one), so a hollow ring's filled outline does not take another ring's
    pairing.
    """
    if not named:
        return []

    def role(r):
        return r.get("was_role", r.get("role", what))
    for want in sorted({role(r) for r in named}):
        n = sum(1 for r in named if role(r) == want)
        have = [f for f in available if f["role"] == want]
        if len(have) < n:
            raise SystemExit(
                f"loft: {n} {want} profile(s) named"
                + (f" at {ax}={at:.4f}" if at is not None else "")
                + f" but only {len(have)} found at "
                f"{ax}={target:.4f}; there is nothing to loft them to."
                + (" The section changes topology between the two levels -- a "
                   "hole that closes, or one ring that splits -- and no loft can "
                   "join a ring to nothing. Loft the outer ring(s) alone, then "
                   "make the hole as its own feature with `--combine cut` "
                   "(any --op builds the cutter)" if what == "hole" else
                   " The section changes topology between the two levels -- a "
                   "ring that ends or splits -- so loft only to a level where it "
                   "is still there, and build the rest as its own feature"))
    used, out = set(), []
    for r in named:
        cand = [(j, f) for j, f in enumerate(available)
                if j not in used and f["role"] == role(r)]
        mine = _region(r, near)

        def iou(f, mine=mine):
            g = _region(f, available)
            if mine is None or g is None or mine.is_empty or g.is_empty:
                return 0.0
            u = mine.union(g).area
            return float(mine.intersection(g).area / u) if u > 0 else 0.0
        scored = sorted(((iou(f), j, f) for j, f in cand), key=lambda x: -x[0])
        if scored and scored[0][0] > 0:
            iou, j0, f0 = scored[0]
            nxt = scored[1] if len(scored) > 1 and scored[1][0] > 0 else None
            if not quiet:
                print(f"loft pairs {r.get('name', '?')} with {f0.get('name', '?')} at "
                      f"{ax}={target:.4f} (overlap {iou:.2f}"
                      + (f"; next {nxt[2].get('name', '?')} {nxt[0]:.2f}" if nxt else "")
                      + ")")
            if nxt and nxt[0] > 0.67 * iou and not quiet:
                print(f"WARNING: {what} profile {r.get('name', '?')} overlaps two rings "
                      f"at {ax}={target:.4f} nearly as much ({f0.get('name', '?')} "
                      f"{iou:.2f}, {nxt[2].get('name', '?')} {nxt[0]:.2f}); taking "
                      f"the first -- name the level differently if that is the "
                      f"wrong pairing")
        else:
            c = _centroid(r)
            order = sorted(((float(np.linalg.norm(c - _centroid(f))), j, f)
                            for j, f in cand), key=lambda x: x[0])
            d0, j0, f0 = order[0]
            if not quiet:
                print(f"loft pairs {r.get('name', '?')} with {f0.get('name', '?')} at "
                      f"{ax}={target:.4f}: no ring there overlaps it, so the nearest "
                      f"centre ({d0:.3f} mm off)")
            if len(order) > 1 and order[1][0] < d0 * 1.5 and not quiet:
                print(f"WARNING: {what} profile {r.get('name', '?')} is nearly "
                      f"equidistant from two profiles at {ax}={target:.4f} "
                      f"({d0:.3f} mm and {order[1][0]:.3f} mm away); taking "
                      f"the nearer one, but name the level differently if "
                      f"that is the wrong pairing")
        used.add(j0)
        if f0["role"] != r.get("role", what):
            f0 = dict(f0, was_role=f0["role"], role=r.get("role", what))
        out.append(f0)
    return out


def _even_turn(mesh, args, named, turns, rings, ax, level, target, cut_at, far_cut):
    """Say when a turned profile does not turn evenly: a ruled loft between
    two levels turns at one rate, and blades that turn faster towards one end
    need more levels."""
    from mesh2cad.fit import twist_between
    if not any(abs(t) >= 5.0 for t in turns):
        return
    mid = 0.5 * (cut_at + far_cut)
    _, mid_rings, _, _ = _rings_at(mesh, args.plane, mid, args.tol, args.grid)
    for a, t in zip(named, turns, strict=True):
        if abs(t) < 5.0:
            continue
        try:
            m = _match_profiles([a], mid_rings, "outer", ax, mid, level, rings, quiet=True)[0]
        except SystemExit:
            continue
        tm = twist_between(a, m)[0]
        if abs(tm - 0.5 * t) > max(2.0, 0.2 * abs(t)):
            print(f"note: half-way, at {ax}={mid:.4f}, {a.get('name', '?')} has turned "
                  f"{tm:+.1f} deg of the {t:+.1f}: the turn is not even, and a ruled loft "
                  f"between two levels turns at one rate. A loft through levels between "
                  f"them follows it: --to mid(<this level>,<that level>),<that level> "
                  f"or more, each one a name")


def _align_starts(near, far, ax, target):
    """Each far ring started where its near ring starts. Rings that pair
    primitive for primitive are left to that; otherwise a ring turned from its
    near one (blades turning along the loft) is rebuilt with it with matched
    vertices, paired turned, and any other has its start moved to the vertex
    facing the near start -- the loft pairs the two from there. Returns (near,
    far, the turn of each far ring in degrees)."""
    from mesh2cad.fit import (
        TWIST_MIN_DEG,
        align_rings,
        conform_rings,
        start_rotated,
        twist_between,
    )
    a_out, b_out, turns = [], [], []
    for a, b in zip(near, far, strict=True):
        if align_rings(a, b) is not None:
            a_out.append(a)
            b_out.append(b)
            turns.append(0.0)
            continue
        turn, base, best = twist_between(a, b)
        turns.append(turn)
        if abs(turn) >= TWIST_MIN_DEG:
            a2, b2 = conform_rings([a, b], twists=[0.0, turn])
            print(f"start aligned: {b.get('name', '?')} at {ax}={target:.4f} is turned "
                  f"{turn:+.1f} deg from {a.get('name', '?')} (overlap {base:.2f} -> "
                  f"{best:.2f} turned back), so both are rebuilt with "
                  f"{a2['n_primitives']} matched vertices paired along that turn")
            a_out.append(a2)
            b_out.append(b2)
            continue
        b2, moved = start_rotated(a, b)
        if moved > 5.0:
            print(f"start aligned: {b.get('name', '?')} at {ax}={target:.4f} now starts "
                  f"at the vertex facing {a.get('name', '?')}'s start (moved "
                  f"{moved:.1f} deg round its centre)")
        a_out.append(a)
        b_out.append(b2)
    return a_out, b_out, turns


# A loft profile whose mean width is under this fraction of its own extent is a
# sliver: a crescent between two nearly equal arcs, measured where the section is
# mid-change. One such profile (width 0.05 mm across 34 mm) was lofted from a
# real 172 mm^2 section and the part never built.
_LOFT_SLIVER = 0.005


# A ring that keeps its shape while it turns between two levels is a
# twist. `twist_between` searches only +-45 deg, so a bar turning 90 deg
# between its plates would be read wrong and its corners paired wrongly by a
# two-level loft. The turn is measured as the sum of the long axis's turns at
# stations this far apart, signed, with no limit.
TWIST_STATION = 2.5
# Below this the turn is a loft's own pairing, not a twist.
TWIST_MIN = 5.0
# The near ring turned by the measured angle must cover the far one this well.
TWIST_SAME = 0.9
# Mid-way, a ring this near the ends' size (linear) turns rigidly; smaller, it
# narrows as a loft between turned copies does (1/sqrt2 at 90 deg).
TWIST_RIGID = 0.9
# Sections a rigid twist is lofted through.
TWIST_RIGID_SECTIONS = 9


def _long_axis(poly):
    """The angle (deg, in (-90, 90]) of a polygon's long axis in (u, v), from
    its second moments of area, or None when it has none (round, square)."""
    from shapely.geometry.polygon import orient
    x, y = np.asarray(orient(poly, 1.0).exterior.coords, float).T
    a = x[:-1] * y[1:] - x[1:] * y[:-1]
    area = a.sum() / 2.0
    if area <= 0:
        return None
    cx = ((x[:-1] + x[1:]) * a).sum() / (6 * area)
    cy = ((y[:-1] + y[1:]) * a).sum() / (6 * area)
    suu = ((x[:-1] ** 2 + x[:-1] * x[1:] + x[1:] ** 2) * a).sum() / 12 - area * cx * cx
    svv = ((y[:-1] ** 2 + y[:-1] * y[1:] + y[1:] ** 2) * a).sum() / 12 - area * cy * cy
    suv = ((x[:-1] * y[1:] + 2 * x[:-1] * y[:-1] + 2 * x[1:] * y[1:] + x[1:] * y[:-1])
           * a).sum() / 24 - area * cx * cy
    if np.hypot(suu - svv, 2 * suv) < 0.05 * (suu + svv):
        return None
    return float(np.degrees(0.5 * np.arctan2(2 * suv, suu - svv)))


def _ring_at(mesh, plane, at, centre):
    """The section polygon at `at` that holds `centre` (else the nearest)."""
    from shapely.geometry import Point
    cut = [p for p in section_at(mesh, plane, at) if p.area > 0]
    if not cut:
        return None
    c = Point(*centre)
    return min(cut, key=lambda p: (not p.contains(c), p.distance(c), -p.area))


def _turn_between(mesh, plane, a, b, centre):
    """(turn in deg, [(level, long-axis angle)]) of the section about `centre`
    from `a` to `b`: the long axis's turns at stations TWIST_STATION apart,
    each step wrapped to +-90 and summed. None where a station has no ring or
    no long axis."""
    n = max(2, int(np.ceil(abs(b - a) / TWIST_STATION)))
    rows = []
    for t in np.linspace(a, b, n + 1):
        p = _ring_at(mesh, plane, float(t), centre)
        ang = _long_axis(p) if p is not None else None
        if ang is None:
            return None
        rows.append((float(t), ang))
    turn = sum(((b2 - a2) + 90.0) % 180.0 - 90.0 for (_, a2), (_, b2) in zip(rows, rows[1:], strict=False))
    return float(turn), rows


def _turned_ring(ring, deg, centre):
    """`ring` turned by `deg` about `centre` in (u, v): every point and arc
    centre, nothing else changed -- the same primitives in the same order."""
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))

    def rot(p):
        u, v = p[0] - centre[0], p[1] - centre[1]
        return [round(float(centre[0] + c * u - s * v), 6), round(float(centre[1] + s * u + c * v), 6)]
    out = {**ring, "segments": [], "nodes": [{**n, "uv": rot(n["uv"])} for n in ring.get("nodes", [])]}
    out.pop("_prims", None)
    for sg in ring["segments"]:
        new = {**sg, "from_xy": rot(sg["from_xy"]), "to_xy": rot(sg["to_xy"])}
        if sg.get("centre") is not None:
            new["centre"] = rot(sg["centre"])
        out["segments"].append(new)
    return out


def _twist_loft(args, mesh, outer, holes, far, ax, sense, level, target, cut_at, far_cut,
                snapped):
    """`--op loft --at L --to L2` of a ring whose far section is the same ring
    turned: built as a twist, the far ring the near one turned (exact vertex
    pairing). True when it built one; False leaves the loft to the other
    paths."""
    from shapely import affinity
    from shapely.geometry import Polygon

    from mesh2cad.emit import build_twist
    if len(outer) != 1 or len(far) != 1:
        return False
    p0 = Polygon(readback.outline(outer[0])).buffer(0)
    p1 = Polygon(readback.outline(far[0])).buffer(0)
    if p0.is_empty or p1.is_empty or p0.geom_type != "Polygon":
        return False
    c0 = np.array([p0.centroid.x, p0.centroid.y])
    got = _turn_between(mesh, args.plane, cut_at, far_cut, tuple(c0))
    if got is None:
        return False
    turn, rows = got
    if abs(turn) < TWIST_MIN:
        return False
    # the axis: the point the turn carries the near ring's centroid onto the
    # far one's about (rings can sit slightly off it)
    t = np.radians(turn)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    c1 = np.array([p1.centroid.x, p1.centroid.y])
    c = tuple(float(x) for x in np.linalg.solve(np.eye(2) - rot, c1 - rot @ c0))
    turned = affinity.rotate(p0, turn, origin=c)
    same = turned.intersection(p1).area / max(turned.union(p1).area, 1e-12)
    if same < TWIST_SAME:
        print(f"note: {outer[0]['name']} turns {turn:+.1f} deg between {ax}={level:.4f} and "
              f"{ax}={target:.4f}, but turned it covers the far ring only {same:.2f}: not a "
              f"twist of one shape, lofted as measured")
        return False
    mid = _ring_at(mesh, args.plane, 0.5 * (cut_at + far_cut), c)
    size = float(np.sqrt(mid.area / p0.area)) if mid is not None else 1.0
    name = outer[0]["name"]
    if size >= TWIST_RIGID:
        law, how = "rigid", (f"the same size ({size:.2f} of the ends'): it turns rigidly, "
                             f"lofted through {TWIST_RIGID_SECTIONS} turned copies")
        ts = np.linspace(0.0, 1.0, TWIST_RIGID_SECTIONS)
    else:
        law, how = "loft", (f"{size:.2f} of the ends' size, as a loft between turned copies "
                            f"narrows (1/sqrt2 at 90 deg): lofted chordally from the ring to "
                            f"itself turned, tangent at both ends")
        ts = np.array([0.0, 1.0])
    sections = [(float(level + t * (target - level)), _turned_ring(outer[0], t * turn, c),
                 [_turned_ring(h, t * turn, c) for h in holes]) for t in ts]
    code = build_twist(args.plane, sense, sections, law)
    print(f"note: {name} turns {turn:+.1f} deg about {ax} between {ax}={level:.4f} and "
          f"{ax}={target:.4f} (its long axis at {len(rows)} stations; + turns "
          f"{PLANES[args.plane]['u_name']} towards {PLANES[args.plane]['v_name']}); turned, "
          f"it covers the far ring {same:.2f}. Mid-way the target's ring is {how}")
    detail = (f"twist ({law}) of {name}{' and ' + ', '.join(h['name'] for h in holes) if holes else ''}"
              f" {turn:+.1f} deg, from {ax}={level:.4f} to {ax}={target:.4f}")
    _finish_sketch(args, code, ax, level, cut_at, snapped, detail, "loft",
                   profiles=f"{sum(r['n_primitives'] for r in [outer[0], *holes])} primitives, "
                            f"turned")
    return True


def _loft_problem(code):
    """Why an emitted loft would not build as a scoreable solid, or None.

    Checked here rather than left to `--check`, because a loft that OpenCascade
    refuses fails with nothing but `BRep_API: command not done` -- no hint that
    the profiles, not the levels or the operation, are what is wrong. Meshing is
    part of "builds": a smooth loft through many short edges can come back a
    valid BRep whose STL is open, and an open feature cannot be scored.
    """
    import cadquery as cq

    from mesh2cad.build import _tessellate

    ns = {"cq": cq}
    try:
        exec(compile(code, "<loft>", "exec"), ns)
    except Exception as exc:                                       # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    for name in sorted(ns):
        if name.startswith("_loft") and hasattr(ns[name], "val"):
            solid = ns[name].val()
            if not solid.isValid():
                return f"{name} builds but is not a valid solid"
            if solid.Volume() <= 0:
                return f"{name} builds inside out (volume {solid.Volume():.4g})"
            if not _tessellate(ns[name], STL_TOL).is_watertight:
                return f"{name} builds but does not mesh closed at tol={STL_TOL}"
    return None


# A loft through more levels than this is built as a chain of lofts this long:
# one OCC loft through many levels, or through dense rings, fails with
# StdFail_NotDone or RecursionError, or comes out inside out.
LOFT_CHAIN = 5


def _chained_loft(args, sense, levels, rings_at, holes_at, ax, cuts):
    """(code, rings_at, holes_at): one loft through every level when it has at
    most LOFT_CHAIN of them and builds; else a chain of lofts of at most
    LOFT_CHAIN levels, each sharing its last level with the next, unioned into
    one feature -- and said, with where they join."""
    def one(lo, hi):
        return _buildable_loft(
            lambda R, H: build_loft_through(args.plane, sense, levels[lo:hi], R, H,
                                            ruled=not args.smooth),
            rings_at[lo:hi], holes_at[lo:hi], ax, cuts[lo:hi], smooth=args.smooth)

    why = None
    if len(levels) <= LOFT_CHAIN:
        try:
            (code, _), r_at, h_at = one(0, len(levels))
            return code, r_at, h_at
        except SystemExit as exc:
            if "cannot be lofted" not in str(exc):
                raise
            if len(levels) > 2:
                return _paired_loft(one, 0, len(levels), levels, ax, exc)
            raise
    starts = list(range(0, len(levels) - 1, LOFT_CHAIN - 1))
    pieces, used_r, used_h = [], [], []
    for i, lo in enumerate(starts):
        hi = min(lo + LOFT_CHAIN, len(levels))
        if hi - lo < 2:
            continue
        try:
            (code, _), r_at, h_at = one(lo, hi)
        except SystemExit as exc:
            if hi - lo <= 2:
                raise SystemExit(f"loft (the piece {ax}={levels[lo]:.4f}..{levels[hi - 1]:.4f} "
                                 f"of a chain): {exc}") from None
            # Dense or unlike rings loft two at a time where they will not
            # loft in fives.
            code, r_at, h_at = _paired_loft(one, lo, hi, levels, ax, exc)
        pieces.append(code + f"\n_chain{i} = result")
        used_r += r_at if not used_r else r_at[1:]
        used_h += h_at if not used_h else h_at[1:]
    joins = ", ".join(f"{ax}={levels[lo]:.4f}" for lo in starts[1:])
    print(f"note: the loft through {len(levels)} levels is built as {len(pieces)} lofts "
          f"of at most {LOFT_CHAIN} levels, joined at {joins} and unioned"
          + (f" ({why})" if why else ""))
    names = [f"_chain{i}" for i in range(len(pieces))]
    code = ("\n".join(pieces) + f"\nresult = {names[0]}"
            + "".join(f".union({n})" for n in names[1:])
            + "\n_STEPS = [" + ", ".join(f'("add", {n})' for n in names) + "]")
    return code, used_r, used_h


def _paired_loft(one, lo, hi, levels, ax, why):
    """(code, rings_at, holes_at): levels lo..hi as two-level lofts unioned,
    for a stretch that will not loft as one; said. A pair that will not loft
    either is refused by name."""
    parts, r_all, h_all = [], [], []
    for j in range(lo, hi - 1):
        try:
            (code, _), r_at, h_at = one(j, j + 2)
        except SystemExit as exc:
            raise SystemExit(f"loft (the pair {ax}={levels[j]:.4f}..{levels[j + 1]:.4f}): "
                             f"{exc}") from None
        parts.append(code + f"\n_pair{j} = result")
        r_all += r_at if not r_all else r_at[1:]
        h_all += h_at if not h_all else h_at[1:]
    print(f"note: {ax}={levels[lo]:.4f}..{levels[hi - 1]:.4f} would not loft in one piece "
          f"({str(why).split('. ')[0]}); built as {hi - lo - 1} two-level lofts")
    names = [f"_pair{j}" for j in range(lo, hi - 1)]
    code = ("\n".join(parts) + f"\nresult = {names[0]}" + "".join(f".union({n})" for n in names[1:])
            + "\n_STEPS = [" + ", ".join(f'("add", {n})' for n in names) + "]")
    return code, r_all, h_all


def _buildable_loft(build, rings_at, holes_at, ax, cuts, smooth=False):
    """`build(rings_at, holes_at)` with profiles that OpenCascade can loft.

    As measured when they build. Otherwise every profile is rebuilt with the
    same number of matched vertices (fit.conform_rings), which is what the loft
    needs and cannot reliably do for itself. When even that does not build, or a
    profile is a sliver that no operation should be lofting to, this refuses --
    a program that cannot run is worse than no program, because it costs a round
    trip to find out.

    Returns `(build result, rings_at, holes_at)` with the profiles actually used.
    """
    for group, cut in zip(zip(rings_at, holes_at, strict=True), cuts, strict=True):
        for ring in group[0] + group[1]:
            if not ring.get("segments") or len(ring_polyline(ring)) < 3:
                # A ring fitted away to nothing -- a coarse --tol, a residual
                # export's sliver -- must not be a traceback in ring_thinness
                # ("zero-size array").
                raise SystemExit(
                    f"loft: the profile {ring.get('name', '?')} measured at "
                    f"{ax}={cut:.4f} is empty: it has no primitives left once fitted "
                    f"(it only grazes the section there, or --tol is coarser than "
                    f"the ring). Name a level inside the feature, a finer --tol, or "
                    f"leave {ring.get('name', '?')} out of --use.")
            thin = ring_thinness(ring)
            if thin < _LOFT_SLIVER:
                raise SystemExit(
                    f"loft: the profile {ring.get('name', '?')} measured at "
                    f"{ax}={cut:.4f} is a sliver (mean width {thin:.2%} of its "
                    f"extent) -- the section is mid-change there, not a profile "
                    f"of this feature. Pick a level with more room, or build "
                    f"this region with another operation.")

    built = build(rings_at, holes_at)
    problem = _loft_problem(built[0])
    if problem is None:
        return built, rings_at, holes_at

    counts = " + ".join(str(sum(r["n_primitives"] for r in g + h))
                        for g, h in zip(rings_at, holes_at, strict=True))

    def _turns(p):
        from mesh2cad.fit import TWIST_MIN_DEG, twist_between
        tw = [0.0]
        for a, b in zip(p, p[1:], strict=False):
            tw.append(tw[-1] + twist_between(a, b)[0])
        if max(abs(t) for t in tw) >= TWIST_MIN_DEG:
            print(f"note: {p[0].get('name', '?')} turns "
                  + " -> ".join(f"{t:+.1f}" for t in tw)
                  + " deg through the levels; its vertices are paired along the turn")
        return tw

    def _conform(per_level):
        by_profile = []
        for p in zip(*per_level, strict=True):
            tw = _turns(p)
            by_profile.append(conform_rings(list(p), twists=tw) if any(tw)
                              else conform_rings(list(p)))
        return [list(level) for level in zip(*by_profile, strict=True)]             if by_profile else [[] for _ in per_level]

    rings_c, holes_c = _conform(rings_at), _conform(holes_at)
    built = build(rings_c, holes_c)
    retry = _loft_problem(built[0])
    if retry is not None:
        raise SystemExit(
            f"loft: these profiles ({counts} primitives) cannot be lofted. As "
            f"measured: {problem}. Rebuilt with matched vertices: {retry}. Pick "
            f"levels whose sections are the same shape"
            + (", drop --smooth (a ruled loft through the same profiles may "
               "build)" if smooth else "")
            + ", or build this region with another operation.")
    print(f"NOTE: the profiles as measured ({counts} primitives) do not loft "
          f"({problem}), so every profile was rebuilt as "
          f"{sum(r['n_primitives'] for r in rings_c[0] + holes_c[0])} straight "
          f"edges with matched vertices; every measured corner is kept and arcs "
          f"become chords.")
    return built, rings_c, holes_c


def _vol(x, ref):
    """Format a volume with enough digits to be readable at the part's scale.

    `%.1f` is right for a 140,000 mm^3 bracket and useless for a mesh normalised
    into a unit box, where every residual would print as "EXTRA 0.0 mm^3 (target
    volume 0.0)" -- a number the model cannot act on and cannot even tell is
    small. Digits are chosen from the reference volume so the line carries the
    same information at any scale.
    """
    if ref <= 0:
        return f"{x:.4g}"
    import math
    digits = max(1, min(9, int(math.ceil(-math.log10(ref))) + 4))
    return f"{x:.{digits}f}"


def _outside_target(target, bb, tol=None):
    """How far the built solid reaches beyond the target's own bounding box.

    The cheapest correctness check there is, and it needs no ground truth: the
    mesh is the INPUT, and a solid that sticks out past the input's extent is
    wrong however good its residual looks -- a feature reaching twice the part's
    height would otherwise pass unremarked.

    Evidence, not a verdict: it reports which face was crossed and by how
    much, and does not refuse to build or guess what the model meant.
    """
    if target is None:
        return None
    # `_target_mesh` is a loaded trimesh, not a path. Assuming the path form
    # made this raise on every call, and the bare `except` underneath turned
    # that into "nothing to report" -- a check that never fired and never said
    # it had not fired, which is worse than not having written it.
    # Relative to the part, and set above fitting noise rather than at zero.
    # An absolute 1e-3 mm is far below the noise on a 100 mm bracket and 0.3% of
    # a mesh normalised into a unit box. It is also not enough to be relative:
    # an arc fitted through sampled points bulges outside their hull by its own
    # sagitta, so a correct profile overshoots slightly by construction. At
    # 0.5% of the longest extent such a fitting bulge is quiet and a real
    # overshoot is not.
    bounds = getattr(target, "bounds", None)
    if bounds is None:
        import trimesh
        bounds = trimesh.load(str(target), force="mesh").bounds
    lo, hi = bounds
    if tol is None:
        tol = 0.005 * float(max(hi[i] - lo[i] for i in range(3)))
    got_lo = (bb.xmin, bb.ymin, bb.zmin)
    got_hi = (bb.xmax, bb.ymax, bb.zmax)
    over = []
    for i, ax in enumerate("XYZ"):
        if got_lo[i] < lo[i] - tol:
            over.append(f"{ax}- by {lo[i] - got_lo[i]:.3f} mm")
        if got_hi[i] > hi[i] + tol:
            over.append(f"{ax}+ by {got_hi[i] - hi[i]:.3f} mm")
    if not over:
        return None
    return ("OUTSIDE THE TARGET: what you built reaches past the part's own "
            "bounding box on " + ", ".join(over) +
            f".  The target spans ({lo[0]:.3f}, {lo[1]:.3f}, {lo[2]:.3f}) - "
            f"({hi[0]:.3f}, {hi[1]:.3f}, {hi[2]:.3f}). A distance or a level is "
            "wrong -- no amount of extra material belongs there.")


def _close_if_possible(mesh):
    """A copy with small holes filled, when that is enough to close it.

    Cheap and often decisive: some meshes are a handful of missing triangles
    from watertight. Returns the original when filling does not close it, so
    the caller can see it failed rather than get a differently-broken mesh.
    """
    if mesh.is_watertight:
        return mesh
    try:
        import trimesh
        fixed = mesh.copy()
        trimesh.repair.fill_holes(fixed)
        fixed.remove_unreferenced_vertices()
        return fixed if fixed.is_watertight else mesh
    except Exception:                                              # noqa: BLE001
        return mesh


def _voxel_residual(target, built, max_points=100_000):
    """(missing, extra, target volume, pitch) by occupancy, for meshes that will
    not close: the mean over three lattices fixed to the target
    (mesh2cad.leftover.voxel_residual, which also gives their spread)."""
    from mesh2cad import leftover
    got = leftover.voxel_residual(target, built, max_points=max_points)
    if got is None:
        return None
    return got.missing, got.extra, got.target, got.pitch


def _kinds(items):
    """` (1 lump inside a film, 2 faces 0.23-0.41 mm off)`: the bodies that are
    not plain blocks, which a volume alone would not explain."""
    lumps = [q for q in items if q.kind == "lump"]
    offs = sorted(q.thickness for q in items if q.kind == "offset")
    out = []
    if lumps:
        out.append(f"{len(lumps)} lump{'s' if len(lumps) > 1 else ''} inside a film")
    if offs:
        span = f"{offs[0]:.2g}" if offs[0] == offs[-1] else f"{offs[0]:.2g}-{offs[-1]:.2g}"
        out.append(f"{len(offs)} face{'s' if len(offs) > 1 else ''} {span} mm off")
    return f" ({', '.join(out)})" if out else ""


# The leftover bodies the last residual line counted, as (side, lo, hi,
# thickness, face names), for `_blend_check` to match against the target's
# blend faces.
_LEFTOVER = []
_BUILT = []          # the build's closed mesh, for blends.short_of


def _on_names(q):
    """The target faces a leftover body lies on, or None when unnamed."""
    on = getattr(q, "on", None)
    return None if not on else [n for g, _a in on for n in g.names]


def _blend_check(args):
    """`check: blends` lines: a leftover body that lies along a fillet or
    chamfer of the target, and the command that makes it from the sharp edge.

    Said where the model reads what is left: listed only in `faces`, blends go
    unused -- the model reads `faces` while planning the body, not the blends.
    Only bodies that lie on the blend face (mesh2cad.blends.along)."""
    if not _LEFTOVER:
        return []
    try:
        from mesh2cad.blends import along, along_text, candidates, short_of, short_text
        from mesh2cad.section import LEVELS_SEGMENT_BUDGET
        from mesh2cad.segment import segment_cached
        mesh = _check_target(args)
        if mesh is None:
            return []
        seg = segment_cached(mesh, cache_dir=_cache_dir(args), budget=LEVELS_SEGMENT_BUDGET)
        found = candidates(mesh, seg, boxes=True)
        short = short_of(mesh, seg, _BUILT[0], found) if _BUILT else []
        return along_text(along(_LEFTOVER, found)) + short_text(short)
    except Exception:                                              # noqa: BLE001
        return []


def _residual_line(target, steps, result=None, cache_dir=None):
    """One line saying what is still missing, right after a feature is built.

    Building without saying what remained would leave the model to remember to
    ask -- and it can stop after one feature and ship a part missing a whole
    region, with the information a `residual` call away.

    This is evidence, not a verdict. It says what is left over; it does not
    say what to do about it, whether the part is finished, or which feature to
    build next. A residual that is empty is equally informative and equally
    silent about what to do.

    Each side names the target faces its bodies lie on (`on cone F12 + plane
    F3`), read off the segmentation the level table already made: the names
    when a feature will not mesh on its own, so the line still measures
    something rather than saying NOT MEASURED check after check while the model
    builds the rest of the part blind.
    """
    _LEFTOVER.clear()
    _BUILT.clear()
    if target is None or not steps:
        return None
    notes = []
    try:
        import trimesh

        from mesh2cad import leftover
        from mesh2cad.build import _mesh_steps, _tessellate
        try:
            built = _mesh_steps(steps, STL_TOL)
        except Exception as exc:                                   # noqa: BLE001
            if result is None:
                raise
            # The fused solid instead, as `export_stl.py` and `residual` do,
            # and said: its mesh may be open, and then the lattice below
            # measures it.
            built = _tessellate(result, STL_TOL)
            notes.append(f"measured on the fused solid: {exc}")
        seg = _target_faces(target, cache_dir, notes)
        # The mesh the segmentation labels: closing it below may renumber
        # its triangles.
        segmented = target
        # Try to close them before giving up on the boolean. Some meshes are a
        # handful of missing triangles away.
        tgt = _close_if_possible(target)
        blt = _close_if_possible(built)
        if not (blt.is_watertight and tgt.is_watertight):
            # Occupancy instead of a boolean, rather than nothing at all: on open
            # meshes the model would otherwise get no residual line -- nothing to
            # build against, and no way for a run ever to be complete.
            got = leftover.voxel_residual(target, built, seg=seg,
                                          floor=leftover.FLOOR * float(target.volume))
            if got is None:
                which = ("the target" if not tgt.is_watertight
                         else "what you built")
                return (f"residual   NOT MEASURED: {which} is not a closed "
                        f"solid, and it could not be voxelised either. Nothing "
                        f"here says the part is finished or unfinished.")
            parts = []
            for word, v, (a, b), side in (("MISSING", got.missing, got.missing_range, "missing"),
                                          ("EXTRA", got.extra, got.extra_range, "extra")):
                if v > 0:
                    spread = (f" ({_vol(a, got.target)}-{_vol(b, got.target)} over "
                              f"{len(leftover.LATTICES)} lattices)" if b > a else "")
                    parts.append(f"{word} about {_vol(v, got.target)} mm^3{spread}"
                                 + _on(got.bodies.get(side, [])))
            body = ", ".join(parts) if parts else "no difference found"
            unclosed = ("the target" if not tgt.is_watertight
                        else "what you built")
            tail = f"  [{'; '.join(notes)}]" if notes else ""
            return (f"residual   {body}  (target volume about "
                    f"{_vol(got.target, got.target)}) -- measured on a {got.pitch:.4g} mm "
                    f"voxel grid, NOT a boolean, because {unclosed} is not a "
                    f"closed solid. The number is quantised to the cell, so "
                    f"treat small values as 'near enough' rather than exact." + tail)
        target, built = tgt, blt
        _BUILT.append(built)
        # Judged by thickness as well as volume (mesh2cad/leftover.py): a floor
        # on volume alone would show a thin film as the one body left, and hide
        # a real plug fused to it.
        diag = float(np.linalg.norm(target.extents))
        miss = leftover.classify(trimesh.boolean.difference([target, built]),
                                 float(target.volume), diag, tol=STL_TOL)
        over = leftover.classify(trimesh.boolean.difference([built, target]),
                                 float(target.volume), diag, tol=STL_TOL)
        fins = [] if miss.items or over.items else leftover.spikes(built, target, diag, tol=STL_TOL)
        if seg is not None:
            for left in (miss, over):
                named = leftover.faces_on([q.mesh for q in left.items], segmented, seg, tol=STL_TOL)
                for q, on in zip(left.items, named, strict=True):
                    q.on = on
        floor = miss.floor
        for side, left in (("missing", miss), ("extra", over)):
            _LEFTOVER.extend((side, q.centre - q.extents / 2, q.centre + q.extents / 2,
                              q.thickness, _on_names(q)) for q in left.items)
        # What the floor threw away, counted rather than forgotten. The floor is
        # PER BODY, so a difference shattered into many small pieces passes it
        # entirely: "nothing above the sliver floor either way" while a real volume
        # sits below the floor in many fragments, and the stopping rule -- which
        # reads this line -- would call the run complete. `significant()` already
        # reports what it drops; this line, the one that decides when a run stops,
        # must too.
        hidden = miss.slivers + over.slivers
        hidden_vol = float(sum(hidden))
        # An aggregate budget as well as a per-body one. Tessellation noise on a
        # correct part totals far less than this; a fragmented real difference
        # does not.
        aggregate = 0.002 * float(target.volume)
    except Exception as exc:                                       # noqa: BLE001
        # Not silence. This used to `return None` for every kind of failure, so
        # "I could not measure this" and "there is nothing to measure" reached
        # the model as the same absence of a line.
        return (f"residual   NOT MEASURED: {type(exc).__name__}: {exc}. "
                f"Nothing here says the part is finished or unfinished.")
    films = leftover.Leftover(films=miss.films + over.films)
    if films.films:
        notes.append(leftover.film_note(films, lambda x: _vol(x, target.volume)))
    if hidden:
        notes.append(f"{len(hidden)} sliver(s) below the {_vol(floor, target.volume)} mm^3 "
                     f"per-body floor, {_vol(hidden_vol, target.volume)} mm^3 in total")
    tail = f"  [{'; '.join(notes)}]" if notes else ""
    if not miss.items and not over.items:
        if fins:
            return "residual   " + leftover.spike_text(fins, lambda x: _vol(x, target.volume)) + tail
        if hidden_vol <= aggregate:
            return ("residual   nothing above the sliver floor either way -- what "
                    "is built matches the target to within tessellation" + tail)
        # Not clean, and deliberately not phrased as clean: the stopping rule
        # keys off the wording above, and this difference is real, merely
        # shattered.
        return (f"residual   nothing SINGLE body is above the floor, but "
                f"{_vol(hidden_vol, target.volume)} mm^3 is unaccounted for across {len(hidden)} "
                f"fragments ({100 * hidden_vol / target.volume:.2f}% of the "
                f"part) -- run `residual` to see where. A difference broken into "
                f"small pieces is still a difference.")
    parts = []
    for word, left in (("MISSING", miss), ("EXTRA", over)):
        if left.items:
            n = len(left.items)
            parts.append(f"{word} {_vol(sum(q.volume for q in left.items), target.volume)} mm^3 in "
                         f"{n} bod{'y' if n == 1 else 'ies'}{_kinds(left.items)}{_on(left.items)}")
    return ("residual   " + ", ".join(parts)
            + f"  (target volume {_vol(target.volume, target.volume)}) -- run `residual` for "
              f"where they are" + tail)


def _target_faces(target, cache_dir, notes):
    """The target's segmentation, as the level table made it (same budget, same
    cache), or None with a note saying why the faces go unnamed."""
    try:
        from mesh2cad.section import LEVELS_SEGMENT_BUDGET
        from mesh2cad.segment import segment_cached
        return segment_cached(target, cache_dir=cache_dir, budget=LEVELS_SEGMENT_BUDGET)
    except Exception as exc:                                       # noqa: BLE001
        notes.append(f"faces not named: the segmentation failed ({type(exc).__name__})")
        return None


def _on(items):
    """` on cone F12 + plane F3`: the target faces under one side's bodies,
    weighted by area over all of them, at most three."""
    from mesh2cad import leftover
    area = {}
    for q in items:
        for s, a in getattr(q, "on", None) or []:
            area[s.names] = (s, area.get(s.names, (s, 0.0))[1] + a)
    if not area:
        return ""
    named = sorted(area.values(), key=lambda sa: -sa[1])[:leftover.ON_LISTED]
    return " on " + leftover.on_text(named)


# Within this share of the part's volume the build is in its last stretch, and
# --check weighs the target's surface too: a run can concede with several
# percent left, all of it skins on bends, which a 1% threshold (the DONE
# rule's) would never look at. Earlier the residual already says everything.
SURFACE_NEAR = 0.10


def _surface_line(target, ns, residual, cache_dir=None):
    """The target-surface line under --check, once the residual is clean
    or within SURFACE_NEAR of the part. None when it does not apply; said
    when it cannot be measured."""
    if target is None or residual is None:
        return None
    totals = residual_totals(residual)
    m = re.search(r"\(target volume (?:about )?([0-9.eE+-]+)\)", residual)
    near = (totals is not None and m is not None
            and sum(totals) <= SURFACE_NEAR * float(m.group(1)))
    if not (near or "nothing above the sliver floor" in residual):
        return None
    try:
        from mesh2cad.build import _tessellate
        from mesh2cad.surface_gate import surface_check
        seg = _target_faces(target, cache_dir, []) if cache_dir is not None else None
        line, _shaped = surface_check(target, _tessellate(ns["result"], STL_TOL), seg=seg)
        return line
    except Exception as exc:                                       # noqa: BLE001
        return (f"surface   not measured ({type(exc).__name__}: {exc}); the "
                f"volume residual is all there is")


def _named_levels(args):
    """The named levels of this sketch's plane, as [{name, value, axis}]."""
    marks, plane = getattr(args, "_marks", None), getattr(args, "plane", None)
    if not marks or plane not in PLANES:
        return None
    k = PLANES[plane]["axis"]
    return [{"name": m["name"], "value": float(m["value"]), "axis": k}
            for m in marks if m.get("name")]


def _gaps_between(parts, levels=None, limit=3):
    """Where separate bodies nearly meet: lines naming the axis, the value one
    stops at and the other starts at, and the named level between, if any.

    "the part is in pieces" alone says that, not where, and a model would go on
    typing the raw offsets that opened the gap. Bodies are
    numbered as their volumes are listed, largest first. Nothing is moved.
    """
    parts = sorted(parts, key=lambda p: -p.volume)
    lo = np.min([p.bounds[0] for p in parts], axis=0)
    hi = np.max([p.bounds[1] for p in parts], axis=0)
    hair = 1e-3 * float(np.max(hi - lo))
    out = []
    for i in range(len(parts)):
        for j in range(len(parts)):
            if i == j:
                continue
            (ai, bi), (aj, bj) = parts[i].bounds, parts[j].bounds
            for k in range(3):
                gap = aj[k] - bi[k]                 # i stops, j starts above it
                others = [m for m in range(3) if m != k]
                overlap = all(min(bi[m], bj[m]) > max(ai[m], aj[m]) for m in others)
                if not (0.0 <= gap <= hair and overlap):
                    continue
                ax = "XYZ"[k]
                near = [lv for lv in (levels or [])
                        if lv.get("axis", k) == k
                        and min(bi[k], aj[k]) - hair <= lv["value"] <= max(bi[k], aj[k]) + hair]
                name = (f"; level {near[0]['name']} is {ax}={near[0]['value']:.4f}"
                        if near else "")
                out.append(f"body {i + 1} stops at {ax}={bi[k]:.6f} and body {j + 1} "
                           f"starts at {ax}={aj[k]:.6f}: a {gap:.1e} gap{name}")
    return out[:limit]


def _target_bodies(target):
    """(how many bodies the target has, a note on how they were counted).

    Closed shells for a closed target. An open one has none -- `--check` would
    say "where the target has 0" on a valid single solid -- so it is counted by
    the pieces its vertices connect into, and said so."""
    if target is None:
        return 1, ""
    if target.is_watertight:
        return len(target.split()), ""
    import trimesh
    comps = trimesh.graph.connected_components(target.edges, min_len=3)
    return max(1, len(comps)), " (counted by connected vertices: the target is not closed)"


# Points sampled inside each target body to see which built body holds it.
FUSED_SAMPLES = 300
# A built body holds a target body when this share of the target body's points
# lie inside it.
FUSED_SHARE = 0.5


def _fused(parts, target):
    """"body 1 fills target bodies 1 and 2" when one built body holds two or
    more of the target's, else None. Fewer bodies than the target is excused
    while a build is incomplete -- and that excuse must not hide a worm fused
    to its wheel, or an assembly fused into one."""
    if not target.is_watertight:
        return None
    tbodies = sorted((b for b in target.split() if b.volume > 0), key=lambda b: -b.volume)
    if len(tbodies) < 2:
        return None
    rng = np.random.default_rng(0)
    inside = []
    for tb in tbodies:
        pts = rng.uniform(tb.bounds[0], tb.bounds[1], (FUSED_SAMPLES, 3))
        inside.append(pts[tb.contains(pts)])
    for i, part in enumerate(sorted(parts, key=lambda x: -x.volume)):
        held = [k + 1 for k, pts in enumerate(inside)
                if len(pts) and float(np.mean(part.contains(pts))) >= FUSED_SHARE]
        if len(held) >= 2:
            names = ", ".join(str(k) for k in held[:-1]) + f" and {held[-1]}"
            return f"body {i + 1} fills target bodies {names}"
    return None


def _changed_nothing(ns):
    """A line when the new feature left the program's volume exactly as it was
    (a feature built from the wrong ring adds 0 mm^3, and `--check` would say
    nothing). None otherwise."""
    prev, res = ns.get("_prev"), ns.get("result")
    if prev is None or res is None:
        return None
    try:
        a = float(sum(s.Volume() for s in prev.solids().vals()))
        b = float(sum(s.Volume() for s in res.solids().vals()))
    except Exception:                                              # noqa: BLE001
        return None
    if a > 0 and abs(b - a) <= 1e-6 * a:
        return (f"NOTHING CHANGED   the program's volume is {b:.3f} mm^3 before and "
                f"after this feature: it lies wholly inside what is built (an add) or "
                f"wholly outside it (a cut). Check the rings in --use and the levels")
    return None


def _mesh_health(wp, steps=None, target=None, levels=None):
    """Tessellate what was just built and say whether it is a valid solid.

    "Builds OK" is not the same as "is a solid". A union of two features whose
    shared face was fitted twice, at two different sections, leaves a sliver:
    CadQuery reports success, the volume looks right, and all ten views come out
    yellow -- while the tessellation is non-manifold and every volumetric metric
    is undefined. That combination cost a whole run before this line existed,
    because nothing in the loop could see it. So look, and say.
    """
    import os
    import tempfile

    import cadquery as cq
    import trimesh

    from mesh2cad.build import _mesh_steps
    shape = wp.toCompound() if hasattr(wp, "toCompound") else wp.val()

    # Mesh exactly the way the scorer will, or "looks fine here" and "scores
    # here" can disagree for a reason nobody can see.
    if steps:
        try:
            m = _mesh_steps(steps, STL_TOL)
            # A closed shell of negative volume is a void inside a body -- a clip
            # by a revolved envelope can leave a spherical one -- not a piece of
            # the part; counted apart, never as a body.
            split = m.split()
            parts = [p for p in split if p.volume > 0]
            voids = len(split) - len(parts)
            bodies = len(parts)
            hollow = (f", with {voids} internal cavit{'y' if voids == 1 else 'ies'}"
                      if voids else "")
            # How many bodies the TARGET has, not how many look tidy. Some parts are
            # genuinely two shells -- a ball and a stem tangent at one circle, apart
            # everywhere else -- and calling those broken would be the tool inventing
            # a defect. What matters is whether the reconstruction agrees.
            want, counted = _target_bodies(target)
            # FEWER bodies than the target is not a fault to shout about: every
            # intermediate step of an incremental build has fewer, and a check
            # that fires on every step of every build is a check nobody reads.
            # MORE is always wrong -- the target says how many pieces this part
            # comes in, and a reconstruction may not invent extra ones.
            # Fewer pieces than the target is no fault -- unless two of them
            # nearly meet. A hairline between bodies is never a design, and
            # "expected while the build is incomplete" must not hide it.
            gaps = _gaps_between(parts, levels) if bodies > 1 else []
            if m.is_watertight and bodies <= want and gaps:
                return (f"BODY COUNT   watertight, {bodies} bodies ({want} in the "
                        f"target), but two of them nearly meet -- a gap that thin "
                        f"is the two ends of one level written differently, not a "
                        f"design. Name the level at both ends."
                        "\n         where they nearly meet: " + "\n         ".join(gaps))
            fused = (_fused(parts, target) if m.is_watertight and bodies < want
                     and target is not None else None)
            if fused:
                return (f"BODY COUNT   watertight, {bodies} "
                        f"{'body' if bodies == 1 else 'bodies'} where the target has "
                        f"{want}{counted}, and {fused} -- separate parts fused into "
                        f"one. Build each part as its own body: `--combine separate`"
                        + hollow)
            if m.is_watertight and bodies <= want:
                note = ("" if bodies == want == 1 else
                        f", matching the target{counted}" if bodies == want else
                        f" ({want} in the target{counted} -- expected while the build "
                        f"is incomplete)")
                return (f"solid OK   watertight via per-feature meshing, "
                        f"{bodies} {'body' if bodies == 1 else 'bodies'}{note}{hollow}")
            if m.is_watertight:
                # A part in pieces is wrong, and it is wrong in a way nothing else
                # here can see: volumetric IoU scores a part in several fragments, or
                # with a feature floating free, nearly perfect.
                #
                # The cause is always the same. manifold3d unions two solids
                # meeting on an exact face into one body, and tolerates a gap
                # down to about 1e-9 -- but leaves two at 1e-6. So a hairline
                # between features, usually from the two ends of a shared level
                # resolving slightly differently, is enough to break the part.
                vols = ", ".join(f"{p.volume:.1f}" for p in
                                 sorted(parts, key=lambda x: -x.volume)[:4])
                where = ("\n         where they nearly meet: " + "\n         ".join(gaps)
                         if gaps else "")
                return (f"BODY COUNT   watertight, but {bodies} separate "
                        f"bodies (volumes {vols}{', ...' if bodies > 4 else ''}) "
                        f"where the target has {want}{counted}"
                        f" -- the part is in pieces. Two features that meet on a "
                        f"level must use the SAME value for it; a gap of 1e-6 is "
                        f"enough. Check that both ends name the level rather "
                        f"than one naming it and the other a number near it."
                        + where)
        except Exception:                                          # noqa: BLE001
            pass

    f = tempfile.mktemp(suffix=".stl")
    try:
        cq.exporters.export(shape, f, tolerance=STL_TOL, angularTolerance=0.1)
        m = trimesh.load(f, force="mesh")
        m.merge_vertices()
    except Exception as exc:                                       # noqa: BLE001
        return f"tessellation FAILED  {type(exc).__name__}: {exc}"
    finally:
        if os.path.exists(f):
            os.unlink(f)

    bodies = len(m.split())
    if m.is_watertight:
        return (f"solid OK   watertight, {bodies} "
                f"{'body' if bodies == 1 else 'bodies'}")

    # Two very different failures look identical from the mesh alone, and
    # conflating them sends you hunting the wrong thing. Ask the kernel which
    # one it is. The first time this fired I reported "not a valid solid" and
    # went looking for a geometry error; the BRep was valid all along and the
    # defect was in the export.
    broken = len(trimesh.repair.broken_faces(m))
    try:
        from OCP.BRepCheck import BRepCheck_Analyzer
        valid = bool(BRepCheck_Analyzer(shape.wrapped).IsValid())
    except Exception:                                              # noqa: BLE001
        valid = None

    if valid is False:
        return (f"NOT A VALID SOLID   the kernel rejects the shape itself: "
                f"{broken} broken faces, {bodies} closed bodies. Usual cause: "
                f"two features share a face fitted separately at two different "
                f"sections, so their union leaves a sliver -- measure both from "
                f"the same section, or build the shared face once.")
    # The advice here used to be "export finer". That was wrong, and wrong in
    # the expensive direction: on a capsule the count stayed at exactly 4 from
    # tol=0.01 down to 0.0005, so following it burned four exports to learn
    # nothing. Degenerate triangles at revolve poles are now dropped in
    # mesh2cad.build, so a count that survives to here is something else.
    return (f"VALID SOLID, BAD TESSELLATION   the BRep is valid ({bodies} closed "
            f"bodies) but STL export at tol={STL_TOL} leaves {broken} broken "
            f"faces, and that is what scoring meshes. IoU and GMS will come back "
            f"undefined. Pole degeneracies are already repaired, so this is a "
            f"real sliver: look for two faces that nearly coincide -- usually a "
            f"feature joined to another exactly on a face rather than "
            f"overlapping it. Exporting finer will NOT fix it; a broken-face "
            f"count that does not move with tolerance never does.")


def cmd_progress(args):
    """Show the reconstruction so far against the target, in the iso views."""
    import os
    import tempfile

    import cadquery as cq
    import trimesh

    mesh, _ = ingest(args.mesh)

    built = None
    if args.built:
        ns = {"show_object": lambda o, *a, **k: None}
        exec(compile(Path(args.built).read_text(), args.built, "exec"), ns)
        obj = ns.get("result")
        shape = obj.toCompound() if hasattr(obj, "toCompound") else obj.val()
        f = tempfile.mktemp(suffix=".stl")
        cq.exporters.export(shape, f, tolerance=0.01, angularTolerance=0.1)
        built = trimesh.load(f, force="mesh")
        os.unlink(f)
        built.merge_vertices()

    d = out_dir(args)
    png = d / ("progress_ortho.png" if args.views == "ortho" else "progress.png")

    # Boolean difference rather than two translucent solids on top of each
    # other: over most of their area the two share a surface, and coincident
    # transparent faces render as noise. The differences have no shared surface
    # and answer the question directly.
    def significant(diff, ref_volume):
        """Keep only the chunks of a difference that mean something.

        A boolean difference between two tessellations is dominated by slivers:
        the target's "circle" is a 48-gon, a fitted true circle sits just
        outside it everywhere, and the difference is a micron-thick film over
        the whole side wall. Rendered, that film hides the one chunk that
        matters. Split into connected pieces and drop the films and the pieces
        too small to be a feature -- reporting what was dropped, so nothing
        vanishes silently.
        """
        if diff is None or not len(diff.vertices):
            return None, 0.0, 0
        # Thickness, not volume, is the criterion (mesh2cad/leftover.py), the
        # same rule as `residual` and the line under --check: a thickness or
        # volume-share floor of its own would hide a real plug fused to a film
        # while it sat in plain sight on the comparison sheet.
        left = leftover.classify(diff, ref_volume, float(np.linalg.norm(mesh.extents)), tol=0.01)
        keep = [q.mesh for q in left.items if q.mesh is not None]
        dropped_vol = left.film_volume + left.sliver_volume
        dropped_n = len(left.films) + len(left.slivers)
        floors[0] = left.floor
        floors[1] = leftover.film_limit(float(np.linalg.norm(mesh.extents)), 0.01)
        if not keep:
            return None, dropped_vol, dropped_n
        import trimesh as _tm
        return _tm.util.concatenate(keep), dropped_vol, dropped_n

    from mesh2cad import leftover
    floors = [0.0, 0.0]
    missing = extra = None
    miss_drop = extra_drop = (0.0, 0)
    if built is not None:
        try:
            missing, dv, dn = significant(mesh.difference(built), mesh.volume)
            miss_drop = (dv, dn)
            extra, dv, dn = significant(built.difference(mesh), mesh.volume)
            extra_drop = (dv, dn)
        except Exception as exc:                                   # noqa: BLE001
            print(f"note: boolean difference failed ({type(exc).__name__}), "
                  f"showing the target outline instead")
            missing = mesh

    layers = [(built, "#e0a458", 0.35 if built is not None else 1.0),
              (missing, "#4a6b8a", 1.0),
              (extra, "#c0392b", 1.0)]
    layers = [l for l in layers if l[0] is not None]
    if args.views == "ortho" or args.mode == "channel":
        # Two sheets, not one: each mesh painted by its own distance to the
        # other. The difference only paints on the mesh that HAS triangles
        # there, so material the reconstruction omitted can only appear on the
        # target sheet and material it invented only on the prediction sheet.
        if built is None:
            raise SystemExit("these views compare two solids -- pass --built")
        if args.views == "ortho":
            if not args.plane:
                raise SystemExit("--views ortho needs --plane")
            # The section normal on screen-up, as `ortho` does it, so the
            # orientation matches the plane the model is sketching on.
            kind, up = "ortho", "XYZ"[PLANES[args.plane]["axis"]]
        else:
            kind, up = "iso", "Y"
        sheets, bands = comparison_sheets(mesh, built, d, kind,
                                          tile=args.tile, ortho_up=up)
        for path, _which in sheets:
            print(f"IMAGE: {path}")
        # What "slight" and "large" are, in mm. A colour band whose distance is
        # never stated is a colour, not a measurement.
        print(f"deviation bands: slight = further than {bands[0]:.4g} mm from "
              f"the other solid, large = further than {bands[1]:.4g} mm "
              f"(0.1% and 1% of the largest axis span of the two)")
    else:
        overlay_views(layers, png, cfg=RenderConfig(tile=args.tile))
        print(f"IMAGE: {png}")
    print(f"target      volume {mesh.volume:10.3f} mm^3")
    if built is not None:
        print(f"built       volume {built.volume:10.3f} mm^3")
        print(f"difference  {mesh.volume - built.volume:10.3f} mm^3 "
              f"({(mesh.volume - built.volume) / max(mesh.volume, 1e-9) * 100:.2f}% of target "
              f"still unaccounted for)")
        lo_t, hi_t = mesh.bounds
        lo_b, hi_b = built.bounds
        if missing is not None and len(missing.vertices):
            print(f"missing     volume {missing.volume:10.3f} mm^3  "
                  f"(bbox along the normal axes: "
                  f"{np.round(missing.bounds[0], 3).tolist()} - "
                  f"{np.round(missing.bounds[1], 3).tolist()})")
        if extra is not None and len(extra.vertices):
            print(f"EXTRA       volume {extra.volume:10.3f} mm^3  "
                  f"-- built material that is not in the target")
        for what, (dv, dn) in (("missing", miss_drop), ("extra", extra_drop)):
            if dn:
                print(f"  ({dn} {what} film(s) and sliver(s) totalling {dv:.3f} mm^3 hidden: "
                      f"thin on average (under {floors[1]:.3g} mm) or "
                      f"below {floors[0]:.4g} mm^3 -- tessellation difference, not a feature)")
        print(f"target bbox ({lo_t[0]:.3f}, {lo_t[1]:.3f}, {lo_t[2]:.3f}) - "
              f"({hi_t[0]:.3f}, {hi_t[1]:.3f}, {hi_t[2]:.3f})")
        print(f"built  bbox ({lo_b[0]:.3f}, {lo_b[1]:.3f}, {lo_b[2]:.3f}) - "
              f"({hi_b[0]:.3f}, {hi_b[1]:.3f}, {hi_b[2]:.3f})")


def _contrib_runner(entry):
    """Adapt a registered instrument to this CLI's fn(args) convention.

    Contributed instruments return a Result rather than printing, so the
    contract can be checked on the way out. A violation is reported and the
    output still goes through: blocking a run mid-reconstruction over a style
    rule would be worse than the rule is worth. CI is where it is fatal.
    """
    def run(args):
        result = entry.run(args)
        if result is None:
            return
        problems = contract.check(
            result, line_budget=entry.line_budget or contract.DEFAULT_LINE_BUDGET)
        for p in problems:
            print(f"contract violation in {entry.name}: {p}", file=sys.stderr)
        sys.stdout.write(result.render())
    return run


def build_parser():
    """The whole CLI surface: built-in stages plus everything in the registry.

    Kept separate from main() so the docs generator can introspect the real
    parser instead of a second, drifting description of it.
    """
    ap = argparse.ArgumentParser(prog="m2c")
    ap.add_argument("--out", default=DEFAULT_OUT)

    # `--out` is also accepted AFTER the subcommand. argparse puts a global
    # option before the subcommand and nowhere else, which reads as an
    # arbitrary rule to anyone writing the command rather than the parser:
    # `m2c.py overview part.stl --out here` is the obvious thing to type, and
    # it failed with "unrecognized arguments". I made that mistake myself twice
    # in one session, and it was the first thing a smaller model driving this
    # CLI got wrong too -- which is the more useful evidence, because it cost a
    # call out of a budget of ten.
    class _Sub:
        def __init__(self, real):
            self._real = real

        def add_parser(self, *a, **k):
            p = self._real.add_parser(*a, **k)
            # SUPPRESS, not None: a plain default here would overwrite the
            # global --out in the namespace whenever the subcommand omitted it.
            p.add_argument("--out", default=argparse.SUPPRESS,
                           help="where to write renders (may go before or "
                                "after the subcommand)")
            return p

    sub = _Sub(ap.add_subparsers(dest="cmd", required=True))

    p = sub.add_parser("overview", help="stage 0+1: normalise, classify, render "
                                        "4 isometric + 6 orthographic projections")
    p.add_argument("mesh")
    p.add_argument("--built", default=None,
                   help="CadQuery program built so far. Reported, but NOT drawn: "
                        "these two renders show one solid each. Use `progress "
                        "--built` for the comparison")
    p.add_argument("--tile", type=int, default=440,
                   help="size in pixels of each view in the two renders "
                        "(default 440). Raise it to make small features "
                        "legible; a bigger picture costs more to read")
    p.set_defaults(fn=cmd_overview)

    p = sub.add_parser("survey", help="stage 2: score XY/YZ/XZ and recommend a section plane")
    p.add_argument("mesh"); p.set_defaults(fn=cmd_survey)

    p = sub.add_parser("spectrum", help="stage 3: candidate offsets along a plane normal")
    p.add_argument("mesh"); p.add_argument("--plane", required=True, choices=list(PLANES),
                                         help=PLANE_HELP)
    p.add_argument("--samples", type=int, default=900,
                   help="how many evenly spaced cuts to take along the normal "
                        "(default 900); a feature thinner than about two "
                        "steps produces no level")
    p.set_defaults(fn=cmd_spectrum)

    p = sub.add_parser("ortho", help="stage 3b: 4 views with the feature levels marked")
    p.add_argument("mesh"); p.add_argument("--plane", default=None, choices=list(PLANES),
                                         help=PLANE_HELP + "; required except with "
                                              "--frame, which sets it")
    p.add_argument("--offset", default=None,
                   help="optional; omit while still choosing the offset. "
                        "a level NAME from `ortho` (E0, B3), or mid(B2,B5), or a flat face across the axis (F5), or a number. Naming carries whatever precision the level has; a raw number carries whatever you typed")
    p.add_argument("--tile", type=int, default=ORTHO_TILE,
                   help=f"size in pixels of each elevation pane (default "
                        f"{ORTHO_TILE}). Larger spreads closely spaced levels "
                        f"further apart on screen")
    # 900, not 240. The sweep is about 3 seconds against 30 or more of model
    # generation, and at 240 the taper detector's blind radius on a mid-sized
    # part hides a chamfer a fraction of a millimetre from a step, so the level
    # simply is not there. A note telling the reader to raise --samples is not
    # a command.
    p.add_argument("--samples", type=int, default=900,
                   help="sweep resolution used to locate the feature levels")
    p.add_argument("--built", default=None,
                   help="CadQuery program built so far. Reported, but NOT drawn: "
                        "these panes render one solid. Use `progress --views "
                        "ortho` for the comparison")
    p.add_argument("--between", default=None,
                   help="offset midway between two levels, e.g. --between -11.063,11.063")
    p.add_argument("--no-boundaries", action="store_true",
                   help="draw only the part ends, hiding the prism-boundary rules; "
                        "use when they crowd the view enough to obscure the geometry")
    p.add_argument("--levels", choices=list(LEVEL_SOURCES), default=None,
                   help=LEVELS_HELP)
    p.add_argument("--frame", default=None, help=FRAME_HELP)
    p.set_defaults(fn=cmd_ortho)

    p = sub.add_parser("section", help="stage 4: the cut at one offset, as named lines and arcs")
    p.add_argument("--detail", action="append", default=None, metavar="R#",
                   help="list every primitive of this ring. A ring of "
                        f"{GROUP_MIN_PRIMITIVES} or more primitives otherwise prints "
                        "its repeats as groups, and caps what fits no repeat at "
                        "its longest primitives. Repeat the flag for more rings")
    p.add_argument("mesh"); p.add_argument("--plane", default=None, choices=list(PLANES),
                                         help=PLANE_HELP + "; required except "
                                              "with --meridian, or an inclined "
                                              "--face or --frame, which sets it")
    p.add_argument("--offset", default=None, help="a level NAME from `ortho` (E0, B3), or mid(B2,B5), or a flat face across the axis (F5), or a number. Naming carries whatever precision the level has; a raw number carries whatever you typed")
    p.add_argument("--tol", type=float, default=None,
                   help="fit tolerance in mm; default is derived from "
                        "the mesh's own sampling density")
    p.add_argument("--grid", type=float, default=None,
                   help="snap coordinates to this grid in mm (e.g. 0.05)")
    # 900, not 240. The sweep is about 3 seconds against 30 or more of model
    # generation, and at 240 the taper detector's blind radius on a mid-sized
    # part hides a chamfer a fraction of a millimetre from a step, so the level
    # simply is not there. A note telling the reader to raise --samples is not
    # a command.
    p.add_argument("--samples", type=int, default=900,
                   help="sweep resolution used to locate the feature levels")
    p.add_argument("--tile", type=int, default=560,
                   help="size in pixels of each pane of the companion "
                        "elevation render (default 560)")
    p.add_argument("--between", default=None,
                   help="offset midway between two levels, e.g. --between -11.063,11.063")
    p.add_argument("--split", default=None, action="append",
                   help='cut the profile at a line: "X=M0" (a level along X), '
                        '"X=N2" (a node\'s X), S# (a mirror plane), F# (a flat '
                        'face across the plane, or a round face\'s axis in it), '
                        'H# (a helix axis in it), or "X=0". Give it twice -- '
                        'each with its own --keep -- to cut at the axis and at '
                        'a level (a revolve of one stretch of a shaft), or '
                        'write both with their sides: "X>=M0,Z<=B2"')
    p.add_argument("--keep", default=None, action="append", choices=["+", "-"],
                   help="which side of --split to keep (+ by default); one per "
                        "--split, in the same order")
    p.add_argument("--use", default=None,
                   help="restrict the split to these rings: ring names (R1), "
                        "primitive names (L1_0) or an array (P0), as --use "
                        "takes them in sketch")
    p.add_argument("--side", choices=["above", "below"], default=None,
                   help="which prism to measure when the offset lands on a boundary; "
                        "omit to see both")
    p.add_argument("--built", default=None,
                   help="the program built so far: its section at the same cut "
                        "is drawn over the target's (dashed red), and every "
                        "region where they differ is listed -- MISSING (target "
                        "only) and EXTRA (built only), with its area, centre "
                        "and the target ring it lies on")
    p.add_argument("--fill", action="store_true",
                   help="with --built: paint the difference -- green missing, "
                        "red extra, grey both")
    p.add_argument("--project", default=None, metavar="LEVEL",
                   help=PROJECT_HELP)
    p.add_argument("--law", default=None, metavar="LEVEL", help=LAW_HELP)
    p.add_argument("--face", default=None,
                   help="take the profile from a flat face `faces` named (F12) "
                        "instead of cutting: the face's own outline, at its own "
                        "level, with holes that go into the material and "
                        "footprints of things standing on it filled. An inclined "
                        "face is shown in its own frame (see --frame)")
    p.add_argument("--sharp", action="store_true", help=SHARP_HELP)
    p.add_argument("--levels", choices=list(LEVEL_SOURCES), default=None,
                   help=LEVELS_HELP)
    p.add_argument("--meridian", default=None, help=MERIDIAN_HELP)
    p.add_argument("--frame", default=None, help=FRAME_HELP)
    p.set_defaults(fn=cmd_section)

    p = sub.add_parser("sketch", help="compose a sketch from named primitives and "
                                      "emit CadQuery for one feature")
    p.add_argument("mesh")
    p.add_argument("--plane", default=None, choices=list(PLANES),
                   help=PLANE_HELP + "; required except with --op hole, whose "
                        "stack sets its own axis, with --meridian, and "
                        "with an inclined --face or --frame, which set it")
    p.add_argument("--at", default=None,
                   help="the sketch plane. a level NAME from `ortho` (E0, B3), or mid(B2,B5), or a flat face across the axis (F5), or a number. Naming carries whatever precision the level has; a raw number carries whatever you typed. Not needed with --face")
    p.add_argument("--use", default=None,
                   help="comma-separated primitive names, e.g. 'A0_0,A1_0' or "
                        "'A0_0@upper', ring names (R1) or an array (P0). Not "
                        "needed with --face, or with --split, which takes the "
                        "whole section unless --use narrows it: the names are "
                        "the section's (`section --split` prints the kept rings "
                        "under them), they pick from what the cut keeps, and a "
                        "hole stays a hole; a hole named alone is a cutter "
                        "(--combine cut)")
    p.add_argument("--op", choices=["extrude", "revolve", "loft", "sweep", "hole",
                                    "fillet", "chamfer"],
                   default="extrude",
                   help="extrude (default): constant section between --at and "
                        "--to. revolve: rings or a half profile about --axis. loft: a "
                        "section that changes between the levels. sweep: a "
                        "profile carried along --along. hole: a drilled, "
                        "counterbored or countersunk hole built from the "
                        "coaxial stacks `faces` names (--use C3 or C3,C4,...): "
                        "each bore step, countersink, chamfer and drill point "
                        "as measured, along any axis (an inclined stack is cut "
                        "in its own frame); cuts, so it takes --base and no "
                        "--plane/--at/--to. fillet / chamfer: round or bevel a "
                        "sharp edge of the build into the target's blend face "
                        "that `faces` names (--use F12 or F12,F13): a fillet is "
                        "a cylinder (straight edge) or torus (circular edge), a "
                        "chamfer a narrow plane or cone; cut on a convex edge, "
                        "added on a concave one (a cove); takes --base and no "
                        "--plane/--at/--to")
    p.add_argument("--project", default=None, metavar="LEVEL",
                   help=PROJECT_HELP + "; the profile is that outline, sketched "
                        "at --at")
    p.add_argument("--law", action="store_true",
                   help="with --op loft and one ring in --use: loft between that "
                        "ring's fitted family (circle, slot, rounded rectangle) at "
                        "--at and at --to, from the straight lines its numbers "
                        "follow in between -- not the traced rings. See section --law")
    p.add_argument("--taper", action="store_true",
                   help="with --op extrude: a draft. The angle is measured, not "
                        "typed: the profile at --to must be the one at --at offset "
                        "by a constant, and the offset over the height is the "
                        "draft (a drafted pocket, a socket entry, a cast wall). "
                        "Refused, with by how much, when it is not")
    p.add_argument("--between", default=None, metavar="A,B",
                   help="with --op sweep --along H0: the two levels on the helix "
                        "axis the swept profile runs between (default: the whole "
                        "part). Any ring is swept -- a thread or worm tooth, not "
                        "only a round wire -- one turn at a time, patterned by "
                        "the pitch and trimmed to the levels")
    p.add_argument("--profile", default=None, metavar="R#",
                   help="with --split and --op sweep: which kept piece is the "
                        "profile, as `section --split` names them (a tooth cut "
                        "off the core at its radius)")
    p.add_argument("--to-profile", dest="to_profile", default=None,
                   help="the section at the other end of --along W#, as <level>:R#,R#: rings "
                        "of the section square to the path there (`end` for the level "
                        "`section` printed at that end of a tube's path). The --use profile "
                        "is lofted into it along the path, holes as holes; for a pipe that "
                        "changes shape as it bends. Refused when the two ring sets don't pair")
    p.add_argument("--stations", type=int, default=None, metavar="K",
                   help="with --to-profile: sweep through K sections square to the path, "
                        "the two named ends and K-2 cut from the target between them, "
                        "each ring resampled to matched points (default 5 when the two "
                        "ends differ in area, else 2: the ends alone)")
    p.add_argument("--along", default=None,
                   help="path for --op sweep: H0, the helix `overview` fitted "
                        "(H0@body or H0@ends when the end turns are wound "
                        "tighter than the body; H1, H2 ... on other bodies of "
                        "the mesh), or W#: a tube's path that `faces` "
                        "chains from its round faces, or the centreline of the "
                        "strip ring R# that `section` names (a wire or pipe cut "
                        "in its own plane). W# is the path last printed under "
                        "that name, on the plane it was printed on. The profile "
                        "is --face F# (an end face, holes and all) or --use "
                        "rings of a section across the path, on any plane, "
                        "swept from where it meets the path; --use of the strip "
                        "itself sweeps a round wire of the strip's width")
    p.add_argument("--smooth", action="store_true",
                   help="for --op loft: blend the two profiles with curvature "
                        "instead of straight lines. Use it when the section "
                        "turns between the levels rather than tapering; a "
                        "helical gear's tooth costs 2.5 IoU built ruled")
    p.add_argument("--split", default=None, action="append",
                   help='cut the profile at a line: "X=M0" (a level along X), '
                        '"X=N2" (a node\'s X), S# (a mirror plane), F# (a flat '
                        'face across the plane, or a round face\'s axis in it), '
                        'H# (a helix axis in it), or "X=0". Give it twice -- '
                        'each with its own --keep -- to cut at the axis and at '
                        'a level (a revolve of one stretch of a shaft), or '
                        'write both with their sides: "X>=M0,Z<=B2"')
    p.add_argument("--keep", default=None, action="append", choices=["+", "-"],
                   help="which side of --split to keep (+ by default); one per "
                        "--split, in the same order")
    p.add_argument("--axis", default=None,
                   help='revolve axis: a round face\'s axis (F#), a helix '
                        '`overview` fitted (H#), a rotation axis `symmetry` names '
                        '(S#), or two world constraints '
                        'with levels or numbers, e.g. --axis "Y=mid(B2,E1),Z=E0" '
                        'or "X=0,Y=0"')
    p.add_argument("--to", default=None,
                   help="where the feature stops: a level NAME (E1, B3, K0, "
                        "M0, T0), mid(A,B), a flat face across the axis (F5: "
                        "the level it lies at), or a number. There is no "
                        "'boundary' or 'end' keyword: every level has a name, "
                        "so name the one you mean. "
                        "For --op loft, a comma-separated list lofts through "
                        "every level in turn (--to B1,B2,B3), which is what a "
                        "section that TURNS as it travels needs; two levels can "
                        "only say it goes from this shape to that one. "
                        "For --op revolve of rings, a face F#: the revolve "
                        "turns until the profile is square to it (the flat end "
                        "face, or the cylinder that carries on from a bend), "
                        "whichever way round stays inside the part")
    p.add_argument("--angle", type=float, default=360.0,
                   help="revolve angle in degrees (signed: the sense is the "
                        "right hand about --axis). Rings named with --use and "
                        "an --axis that crosses none of them -- an annulus "
                        "about a pipe bend's axis, a half-profile beside the "
                        "part's axis -- revolve with their holes kept as "
                        "holes; --to F# ends one without a sign to choose")
    p.add_argument("--tol", type=float, default=None,
                   help="fit tolerance in mm; default is derived from "
                        "the mesh's own sampling density")
    p.add_argument("--grid", type=float, default=None,
                   help="snap coordinates to this grid in mm (e.g. 0.05)")
    # The same 900 as spectrum, ortho and section, and for a reason those three
    # do not have to carry: sketch is the instrument that CONSUMES level names.
    # At 240 it minted a different grid from the one the names were published
    # on, so `--to B0` referred to a level sketch could not see and `--at K1`
    # matched nothing -- the model was told a name either resolves or fails
    # loudly, and instead it resolved to somewhere else. Four of the fourteen
    # complex parts lost calls to that, and two fell back to typing raw offsets,
    # which is the one thing A3 exists to prevent.
    p.add_argument("--samples", type=int, default=900,
                   help="sweep resolution used to resolve level names; keep "
                        "it equal to the value the names came from")
    p.add_argument("--side", choices=["above", "below"], default=None,
                   help="force a side at a boundary; normally inferred from --use")
    p.add_argument("--base", default=None,
                   help="a program built so far; the new feature is unioned onto it "
                        "(or cut from it, see --combine)")
    p.add_argument("--combine", choices=["add", "cut", "intersect", "separate"], default="add",
                   help="with --base: what the new feature does to the program "
                        "so far. add (default) unions it; cut subtracts it "
                        "(holes, pockets, slots, a cylinder or sphere seat -- "
                        "any --op builds the cutter); intersect keeps only what "
                        "lies inside it (a clipping solid), or with --with N "
                        "clips feature #N alone; separate keeps it a body of "
                        "its own beside the rest (a screw in a plate, a bush in "
                        "a housing, a second part of an assembly), which later "
                        "features fuse with only where they touch it. The tool is the finished "
                        "feature, inner rings included; a hole ring named alone "
                        "in --use is filled to make it")
    p.add_argument("--with", dest="with_feature", type=int, default=None, metavar="N",
                   help="with --combine intersect: clip feature #N only (as the "
                        "`program:` line numbers them) and leave the rest alone "
                        "-- #N becomes #N intersected with the new feature. A "
                        "joggled plate: the strip from a cut across the bend, "
                        "then its outline extruded with --combine intersect "
                        "--with <the strip>")
    p.add_argument("--replace", type=int, default=None, metavar="N",
                   help="with --base: the new feature takes the place of feature "
                        "#N of the base instead of being added after the last "
                        "one. Every other feature stays as it was. The numbers "
                        "are the ones each --check prints on its `program:` line")
    p.add_argument("--emit", default=None, help="write the program to this path")
    p.add_argument("--check", action="store_true", help="execute it and report bbox/volume")
    p.add_argument("--show-program", action="store_true",
                   help="print the whole CadQuery program. It is not printed "
                        "otherwise; the `feature:` and `program:` lines say what "
                        "it holds, by the names you work with")
    p.add_argument("--grow", default=None,
                   help='push named profile edges outward before building, e.g. '
                        '--grow "L0_3=1.0", so the feature OVERLAPS what it '
                        'joins instead of sharing a face with it. Per-edge and '
                        'by name on purpose: growing the whole profile would '
                        'move dimensions that were measured exactly')
    p.add_argument("--both", action="store_true",
                   help="extrude the same distance EITHER SIDE of the sketch "
                        "plane, for a feature measured at its own mid-thickness")
    p.add_argument("--pattern", default=None,
                   help='repeat the feature N times evenly about a world axis, '
                        'e.g. --pattern "5@Y=0,Z=0" (use the angles `residual` '
                        'measured rather than counting them off a picture), or '
                        'in a row: --pattern "4@B1..B4", the last copy moved '
                        'by the distance from level B1 to B4 along the sketch '
                        'axis ("4@X:B1..B4" along another axis). A cut is '
                        'patterned the same way')
    p.add_argument("--mirror", default=None, metavar="PLANE",
                   help="add the feature's mirror image too: about a mirror "
                        "plane `symmetry` names (S0), or a level -- M0, any "
                        "level, mid(A,B) -- on the sketch axis (\"Y:M0\" on "
                        "another axis). A cut is mirrored the same way. Build "
                        "one side of a symmetric part and mirror it")
    p.add_argument("--face", default=None,
                   help="sketch on a flat face `faces` named (F12): the face is "
                        "the sketch plane and its outline the profile, and the "
                        "feature goes into the side the material is on. An "
                        "inclined face is sketched in its own frame (see "
                        "--frame): --to then names the levels across it, such "
                        "as a parallel face")
    p.add_argument("--sharp", action="store_true", help=SHARP_HELP)
    p.add_argument("--levels", choices=list(LEVEL_SOURCES), default=None,
                   help=LEVELS_HELP)
    p.add_argument("--meridian", default=None, help=MERIDIAN_HELP)
    p.add_argument("--frame", default=None, help=FRAME_HELP)
    p.set_defaults(fn=cmd_sketch)

    p = sub.add_parser("progress", help="the 4 iso views with target and "
                                        "reconstruction superimposed")
    p.add_argument("mesh")
    p.add_argument("--built", default=None, help="CadQuery program built so far")
    p.add_argument("--tile", type=int, default=680,
                   help="size in pixels of each view (default 680)")
    p.add_argument("--views", choices=["iso", "ortho"], default="iso",
                   help="iso (default): the four isometric views; ortho: the "
                        "four elevations of --plane")
    p.add_argument("--mode", choices=["channel", "diff"], default="channel",
                   help="channel: two opaque renders combined by colour channel; "
                        "diff: translucent build plus boolean difference")
    p.add_argument("--plane", default=None, choices=list(PLANES),
                   help="required for --views ortho")
    p.set_defaults(fn=cmd_progress)

    p = sub.add_parser("edit", help="list, drop or undo the features of a "
                                    "program `sketch` wrote")
    p.add_argument("mesh")
    p.add_argument("--built", required=True,
                   help="the program to edit (normally the one you --emit to)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--drop", type=int, default=None, metavar="N",
                   help="remove feature #N; the features after it keep their "
                        "order and are renumbered. Without --drop or --undo, "
                        "edit lists the features with each one's own volume "
                        "and bbox")
    g.add_argument("--undo", action="store_true",
                   help="restore the program as it was before its last --emit "
                        "(every --emit keeps a numbered version; repeated "
                        "--undo walks further back)")
    p.add_argument("--feature", type=int, default=None, metavar="N",
                   help="with --mirror or --pattern: the feature #N (as the "
                        "list numbers them) to mirror or repeat, a cut too")
    p.add_argument("--mirror", default=None, metavar="PLANE",
                   help="with --feature N: add feature #N's mirror image, about "
                        "S# from `symmetry` or a level on a named axis, e.g. "
                        "\"Y:M0\"")
    p.add_argument("--pattern", default=None,
                   help='with --feature N: repeat feature #N, "5@Y=0,Z=0" '
                        'about an axis or "4@X:B1..B4" in a row')
    p.add_argument("--emit", default=None,
                   help="write the edited program to this path (normally the "
                        "same path as --built). Without it nothing is saved")
    p.add_argument("--show-program", action="store_true",
                   help="print the whole program after --mirror or --pattern")
    p.add_argument("--check", action="store_true",
                   help="build the edited program and report bbox, volume, "
                        "residual and the change against the last saved version")
    p.set_defaults(fn=cmd_edit)

    # Everything under mesh2cad/contrib/instruments/. Dropping a file there is
    # the entire installation step -- no central list to edit, so two people
    # adding two instruments never touch the same line.
    for entry in registry.entries("instrument"):
        p = sub.add_parser(entry.name, help=entry.summary)
        if entry.configure:
            entry.configure(p)
        p.set_defaults(fn=_contrib_runner(entry))

    p = sub.add_parser("recall", help="print a past command's output again, whole "
                                      "or in part")
    p.add_argument("command", type=int,
                   help="its number, from the [command N/...] line of its output")
    which = p.add_mutually_exclusive_group()
    which.add_argument("--grep", default=None, help="only the lines holding this text")
    which.add_argument("--ring", default=None, help="only this ring of a section, R2")
    which.add_argument("--lines", default=None, help="only these lines, a:b")
    p.set_defaults(fn=cmd_recall)

    return ap


# Options whose value can start with "-": a negative number, or a list or a
# plane that begins with one. argparse reads `--to -6.3,B0` as a flag after
# --to and refuses ("expected one argument").
NEGATIVE_OK = ("--to", "--between", "--at", "--offset", "--grow", "--mirror", "--split",
               "--axis", "--pattern", "--level", "--to-profile")


def _join_negative(argv):
    """argv with `--to -6.3,B0` written `--to=-6.3,B0`, which argparse takes."""
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        if (a in NEGATIVE_OK and i + 1 < len(argv)
                and re.match(r"-\.?\d", argv[i + 1])):
            out.append(f"{a}={argv[i + 1]}")
            i += 2
            continue
        out.append(a)
        i += 1
    return out


def _budgeted(args, argv):
    """(what, what to do instead) for a command that runs under the time
    budget, or None. These can run into the 1800 s command limit on dense or
    many-body meshes and print nothing."""
    cmd = next((a for a in argv if a in ("section", "ortho", "residual", "sketch")), None)
    if cmd in ("section", "ortho"):
        return (f"`{cmd}`", "Ask for less at once: a coarser --tol, fewer --samples, "
                            "another level, or a smaller --face")
    if cmd == "residual":
        return ("`residual`", "Compare less at once: --samples fewer, --views iso, or "
                              "`sketch ... --check`, which measures the feature it builds")
    if cmd == "sketch" and getattr(args, "op", None) in ("chamfer", "fillet"):
        return (f"`sketch --op {args.op}`", "Name fewer faces in --use, a few at a time")
    return None


def cmd_recall(args):
    root = Path(os.environ.get("M2C_OUTPUTS", "run/outputs"))
    path = root / f"{args.command:03d}.txt"
    if not path.exists():
        raise SystemExit(f"m2c recall: no output of command {args.command} in {root}")
    print(recall_output(path.read_text(), args.grep, args.ring, args.lines))


def attach_negative_values(argv):
    """`--to -6.3,B0` as `--to=-6.3,B0`.

    argparse only takes a plain number for a negative value, so anything else
    starting with a minus would be read as an unknown option -- the commonest
    parser error. No option here takes more than one value, so a minus and a
    digit right after an option can only be that option's value.
    """
    out = []
    for arg in argv:
        if (re.match(r"-\.?\d", arg) and out and out[-1].startswith("--")
                and "=" not in out[-1]):
            out[-1] = f"{out[-1]}={arg}"
        else:
            out.append(arg)
    return out


def _run(args):
    # Every call is journalled, by the CLI rather than by a wrapper, so cost
    # cannot be measured only when someone remembers to measure it.
    try:
        journal.run_and_record(args.fn, args, sys.argv[1:])
    except (CutError, EmitError) as e:
        # A cut through a broken stretch of the target (section._cut_polygons),
        # or a feature with nothing left to write (emit.drawable): say which
        # and what to do, not a traceback.
        sys.exit(str(e))


def _refuse_non_mesh(args):
    """A first argument that isn't a mesh, refused by name: a program put
    where the target goes (`sketch run/reconstruction.py ...`) would come back
    as a page of NotImplementedError from the loader."""
    import trimesh
    mesh = getattr(args, "mesh", None)
    if mesh and Path(mesh).suffix.lower().lstrip(".") not in trimesh.available_formats():
        sys.stderr.write(f"{mesh} is not a mesh: the first argument is the target mesh "
                         f"(part.stl); a program goes after --base / --built\n")
        sys.exit(2)


def main():
    args = build_parser().parse_args(_join_negative(attach_negative_values(sys.argv[1:])))
    _refuse_non_mesh(args)
    if getattr(args, "mesh", None) and getattr(args, "out", None):
        # Where `section` and `ortho` keep what they printed, so a later command
        # can resolve a name read at another level.
        readback.set_context(lambda: out_dir(args), args.mesh)
    limit, what = _tb.budget(), _budgeted(args, sys.argv[1:])
    if limit and what:
        sys.exit(_tb.run(lambda: _run(args), limit, *what))
    _run(args)


if __name__ == "__main__":
    main()
