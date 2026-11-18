"""residual -- what is still missing, and what was built that should not be.

The ten-view collage answers "does it agree?" and cannot answer "with what?".
A difference hidden behind other geometry reads as agreement, and a difference
in plain sight is still only a coloured patch: the model can see that something
is green but not how big it is, where its centre is, or whether the five green
things are the same thing five times.

So take the boolean difference both ways, split it into connected bodies, and
name each one. The remainder stops being a picture and becomes a work list:

    m2c.py residual part.stl --built r.py --axis "Y=0,Z=0"
    MISSING (in the target, not yet built): 6 bodies
      D0   vol  20154.3  centre (  9.9,  57.5,  -0.0)   r= 57.5  ang=   0.0deg
      D1   vol  20154.3  centre (  9.9,  17.8,  54.7)   r= 57.5  ang=  72.0deg
      ...

With `--axis` the centres are reported in cylindrical coordinates about it, so
a circular pattern reads straight off the angle column -- which is the question
"where do the spokes go?" answered as five numbers instead of a picture.
"""

from pathlib import Path

import numpy as np

from mesh2cad import leftover, registry
from mesh2cad.build import STL_TOL, load_program
from mesh2cad.contract import Result
from mesh2cad.frame import describe_frame, ingest_frame, write_export_source
from mesh2cad.normalize import ingest
from mesh2cad.render import (
    RESIDUAL_LEGEND,
    RESIDUAL_SOLO_LEGEND,
    SKIN_LEVEL,
    RenderConfig,
    Residual,
    ortho_views,
    projections,
)
from mesh2cad.section import PLANES, feature_levels

# Below this share of the target's volume a residual body is a boolean sliver,
# not a feature. Two surfaces that should coincide never quite do, and every
# such pair leaves a wafer behind. Above it, thickness decides between a
# feature and a film (mesh2cad/leftover.py), so the floor can sit 5x lower than
# the 0.05% it was when volume alone had to tell them apart.
MIN_VOLUME_FRAC = leftover.FLOOR
# Bodies listed per side; the rest are summed in one line. An unfinished build
# leaves dozens, and a list longer than the model reads is a list it skips.
MAX_LISTED = 10
KIND = {"block": "block", "lump": "lump in film", "offset": "face off"}


def configure(p):
    p.add_argument("mesh")
    p.add_argument("--built", required=True,
                   help="the CadQuery program built so far")
    p.add_argument("--axis", default=None,
                   help='report each body in cylindrical coordinates about this '
                        'axis, e.g. --axis "Y=0,Z=0"; a circular pattern then '
                        'reads off the angle column')
    p.add_argument("--min-volume", type=float, default=None,
                   help=f"ignore bodies below this volume in mm^3; default "
                        f"{MIN_VOLUME_FRAC:.2%} of the target")
    p.add_argument("--tile", type=int, default=440,
                   help="size in pixels of each view in the residual render (default 440)")
    p.add_argument("--solo", action="store_true",
                   help="render the residual ON ITS OWN, with neither the target "
                        "nor the reconstruction alongside it")
    p.add_argument("--views", choices=["all", "iso", "ortho"], default="all",
                   help="which projections to draw")
    p.add_argument("--plane", choices=list(PLANES), default=None,
                   help="draw the four elevations in which this plane is "
                        "edge-on, WITH a scale and the residual's own feature "
                        "levels marked -- so the extent of each leftover body "
                        "can be read off and passed to --at")
    p.add_argument("--samples", type=int, default=240,
                   help="sweep resolution used to locate the levels")
    p.add_argument("--frame", default=None, metavar="F#",
                   help="measure across an inclined face or axis, as section and "
                        "sketch do: the target and the program are turned into "
                        "that frame first, so --plane and every position is in it")
    p.add_argument("--export", default=None,
                   help="write the residual to this STL. It then behaves like "
                        "any other mesh, so `section`, `spectrum` and the rest "
                        "of the loop can be pointed straight at what is left "
                        "instead of at the whole part")


def _framed(args, target, built):
    """Target and program turned into --frame's frame, as section and sketch
    take it. The volumes are the same in any frame; positions and views are in
    this one."""
    from mesh2cad.frames import FrameError, direction_of, frame_for, inclined, turn_mesh
    from mesh2cad.frames import describe_frame as describe_turn
    from mesh2cad.segment import frame_source, segment_cached

    if getattr(args, "export", None):
        raise SystemExit("--export writes the leftover in the target's own frame; "
                         "drop --frame for it (a sketch on the export takes --frame "
                         "itself)")
    name = args.frame.strip().upper()
    seg = segment_cached(target)
    try:
        d, what = direction_of(seg, name)
    except FrameError as exc:
        raise SystemExit(f"--frame {name}: {exc}") from None
    fr = frame_for(d)
    if not inclined(d):
        return target, built, (f"note: --frame {name}: {what} already lies on "
                               f"{'XYZ'[fr['k']]}, so no frame is needed")
    turned = turn_mesh(target, fr["R"])
    frame_source(turned, target, fr["R"])
    return turned, turn_mesh(built, fr["R"]), describe_turn(fr, name, what) + (
        "; volumes are the same in any frame, positions and views here are in this one")


def run(args):
    import trimesh

    target, info = ingest(args.mesh)
    built = load_program(args.built)
    frame_note = None
    if getattr(args, "frame", None):
        target, built, frame_note = _framed(args, target, built)
        print(frame_note)
    floor = (args.min_volume if args.min_volume is not None
             else MIN_VOLUME_FRAC * float(target.volume))
    seg, seg_note = _segmentation(args, target)
    axis = _parse_axis(args.axis) if args.axis else None

    # A non-watertight input makes the boolean meaningless, and the caller has
    # to know that rather than read "none" as "finished". Occupancy needs no
    # closure, so the leftover is still measured, on a lattice, and the output
    # says so.
    why = None
    for label, m in (("target", target), ("built", built)):
        if not m.is_watertight:
            broken = len(trimesh.repair.broken_faces(m))
            why = (f"the {label} mesh is not watertight ({broken} broken faces), "
                   f"so a boolean difference against it is meaningless")
            break
    if why is None:
        try:
            from mesh2cad.budget import stage
            stage(f"the boolean difference of the target ({len(target.faces)} triangles) "
                  f"and the program")
            missing, mfilm = _bodies(trimesh, target, built, floor, target, seg)
            extra, efilm = _bodies(trimesh, built, target, floor, target, seg)
        except Exception as exc:                                   # noqa: BLE001
            why = f"the boolean difference failed ({type(exc).__name__}: {exc})"
    if why is not None:
        return _by_voxels(args, target, built, floor, seg, seg_note, axis, why)

    r = Result()
    r.line(f"residual of {args.mesh} against {args.built}")
    r.line(f"target volume {target.volume:.3f}   built {built.volume:.3f}   "
           f"difference {target.volume - built.volume:+.3f} mm^3")
    r.line(f"bodies below {floor:.3f} mm^3 are treated as boolean slivers "
           f"and dropped")
    r.line("fill = volume as a share of the bounding box: a low fill means a "
           "shell or a plate, not a block")
    r.line("kind: block = a solid piece; lump in film = thick material inside a "
           "thin skin; face off = a face built that far from the target's; "
           "thick = mean thickness")
    r.line(ON_HELP if seg is not None else seg_note)

    where = _where(args, target, missing[:MAX_LISTED] + extra[:MAX_LISTED])
    for b, w in zip(missing[:MAX_LISTED] + extra[:MAX_LISTED], where, strict=True):
        b["at"] = w
    if any(where):
        r.line(AT_HELP)
    _report(r, "MISSING (in the target, not yet built)", "D", missing, axis, mfilm)
    _report(r, "EXTRA (built, but not in the target)", "E", extra, axis, efilm)

    if missing or extra:
        from mesh2cad.surface_gate import faces_off
        try:
            off = faces_off(target, built, seg)
        except Exception as exc:                                   # noqa: BLE001
            off = [f"not measured ({type(exc).__name__}: {exc})"]
        if off:
            r.line()
            r.line("target faces off (each face the build misses, signed: short = "
                   "the build stops before the face, past = it runs beyond it) "
                   + off[0] + ":")
            for ln in off[1:]:
                r.line(f"  {ln}")

    blends = _blend_lines(seg, target, missing, extra, built)
    if blends:
        r.line()
        r.line("blends (a sharp edge where the target is rounded or bevelled):")
        for ln in blends:
            r.line(f"  {ln}")

    if not missing and not extra:
        r.line()
        fins = leftover.spikes(built, target, float(np.linalg.norm(target.extents)), tol=STL_TOL)
        if fins:
            r.line(leftover.spike_text(fins, lambda x: f"{x:.3f}"))
            return r
        r.line("nothing above the sliver floor either way: the reconstruction "
               "is complete to within tessellation.")
        return r

    if args.export and (missing or extra):
        import trimesh
        lump = trimesh.util.concatenate(
            [b["mesh"] for b in missing + extra])
        Path(args.export).parent.mkdir(parents=True, exist_ok=True)
        lump.export(args.export)
        T = ingest_frame(info)
        write_export_source(args.export, args.mesh, T)
        r.line()
        r.line(f"wrote the residual to {args.export} -- section it like any "
               f"other mesh. It is in {args.mesh}'s frame"
               + (f" ({describe_frame(T)} at ingest)" if T is not None else "")
               + f"; a sketch on it is written in that frame and --check "
               f"measures it against {args.mesh}")

    png, marks = _render(args, built, missing, extra)
    if png is not None:
        r.image(png)
    if marks:
        ax = "XYZ"[PLANES[args.plane]["axis"]]
        r.line()
        r.line(f"levels of the RESIDUAL along {ax} ({len(marks)}):")
        for m in marks:
            r.line(f"    {m['value']:9.3f}  {m['label']}")
    return r


def _bodies(trimesh, a, b, floor, target, seg=None):
    """(what is left of a minus b, largest first; its films), judged by
    thickness as well as volume against the target's size
    (mesh2cad/leftover.py), each named by the target faces it lies on.

    Raises on a failed boolean rather than returning nothing: otherwise "the
    boolean crashed" and "there is nothing left" would print the same word,
    which is the most confidently wrong answer this instrument could give.
    """
    diff = trimesh.boolean.difference([a, b])
    if diff is None or diff.is_empty:
        return [], leftover.Leftover()
    left = leftover.classify(diff, float(target.volume), float(np.linalg.norm(target.extents)),
                             floor=floor, tol=STL_TOL)
    shown = left.items[:MAX_LISTED]
    named = (leftover.faces_on([q.mesh for q in shown], target, seg, tol=STL_TOL)
             if seg is not None else [[] for _ in shown])
    return [{"mesh": q.mesh, "volume": q.volume, "centre": q.centre, "extents": q.extents,
             "kind": q.kind, "thickness": q.thickness,
             "on": named[i] if i < len(named) else []}
            for i, q in enumerate(left.items)], left


def _blend_lines(seg, target, missing, extra, built=None):
    """The leftover bodies that lie along a fillet or chamfer of the target, with
    the command that makes each from the sharp edge (mesh2cad.blends): only
    bodies that lie on the blend face; and the convex blends the build stops
    short of (`short_of`). Nothing without a segmentation."""
    if seg is None or (not missing and not extra):
        return []
    try:
        from mesh2cad.blends import along, along_text, candidates, short_of, short_text
        left = [(side, b["centre"] - b["extents"] / 2, b["centre"] + b["extents"] / 2,
                 b.get("thickness"),
                 [n for g, _a in b["on"] for n in g.names] if b.get("on") else None)
                for side, bodies in (("missing", missing), ("extra", extra)) for b in bodies]
        found = candidates(target, seg, boxes=True)
        short = short_of(target, seg, built, found) if built is not None else []
        return along_text(along(left, found)) + short_text(short)
    except Exception:                                              # noqa: BLE001
        return []


ON_HELP = ("on = the target's faces the body lies on, as `faces` names them: "
           "where the material should be, or what the extra material sits against")
AT_HELP = ("at = the band of named levels the body lies in, along the axis it is "
           "thinnest on, and the ring of `section` at the band's middle nearest it: "
           "the names --at, --to and --use take")
# A level counts as bounding a body when it lies within this share of the
# diagonal of the body's end.
AT_REACH = 0.002
# The sweep `section` and `sketch` name their levels on (their --samples
# default), so a band here is written in the names they print and take.
SECTION_SAMPLES = 900


def _where(args, target, bodies):
    """For each body, `Z B1..E1 R0`: the named levels around it along its
    thinnest axis and the nearest ring at the middle of that band. The faces
    alone can leave many leftovers all "on" one shared fillet face; the band
    says which stretch of the part each is in. "" for a body with no level on
    either side; [] on any failure, said by nothing because the table stands
    without it."""
    from shapely.geometry import Point

    from mesh2cad.section import section_at
    out = Path(getattr(args, "out", None) or "out/agent")
    if str(out) == "out/agent":
        out = out / Path(args.mesh).resolve().parent.name
    diag = float(np.linalg.norm(target.extents))
    marks, cuts, res = {}, {}, []
    try:
        for b in bodies:
            k = int(np.argmin(b["extents"]))
            plane = next(p for p, spec in PLANES.items() if spec["axis"] == k)
            if plane not in marks:
                got = feature_levels(target, plane, SECTION_SAMPLES,
                                     cache_dir=out / ".m2c" / "cache")
                marks[plane] = sorted((m for m in (got[0] if isinstance(got, tuple) else got)
                                       if m.get("name")), key=lambda m: m["value"])
            lo = float(b["centre"][k] - b["extents"][k] / 2)
            hi = float(b["centre"][k] + b["extents"][k] / 2)
            reach = AT_REACH * diag
            below = [m for m in marks[plane] if m["value"] <= lo + reach]
            above = [m for m in marks[plane] if m["value"] >= hi - reach]
            if not below or not above:
                res.append("")
                continue
            a, z = below[-1], above[0]
            text = f"{'XYZ'[k]} {a['name']}..{z['name']}"
            mid = round((a["value"] + z["value"]) / 2, 6)
            if (plane, mid) not in cuts:
                rings, ri = [], 0
                for poly in section_at(target, plane, mid):
                    for ring in [poly.exterior] + list(poly.interiors):
                        rings.append((f"R{ri}", ring))
                        ri += 1
                cuts[(plane, mid)] = rings
            spec = PLANES[plane]
            pt = Point(b["centre"][spec["u"]], b["centre"][spec["v"]])
            near = min(cuts[(plane, mid)], key=lambda nr: nr[1].distance(pt), default=None)
            if near is not None and near[1].distance(pt) <= float(np.max(b["extents"])):
                text += f" {near[0]}"
            res.append(text)
    except Exception:                                              # noqa: BLE001
        return [""] * len(bodies)
    return res


def _segmentation(args, target):
    """(segmentation, note): the target's faces as the level table and `faces`
    made them -- same cache, same 90 s budget -- or None and why not (R2)."""
    from mesh2cad.section import LEVELS_SEGMENT_BUDGET
    from mesh2cad.segment import segment_cached
    out = Path(getattr(args, "out", None) or "out/agent")
    if str(getattr(args, "out", "") or "out/agent") == "out/agent":
        out = out / Path(args.mesh).resolve().parent.name
    try:
        return segment_cached(target, cache_dir=out / ".m2c" / "cache",
                              budget=LEVELS_SEGMENT_BUDGET), ""
    except Exception as exc:                                       # noqa: BLE001
        return None, (f"on: faces not named -- the segmentation failed "
                      f"({type(exc).__name__}: {exc})")


def _by_voxels(args, target, built, floor, seg, seg_note, axis, why):
    """The residual by occupancy on a lattice fixed to the target, for a pair
    the boolean cannot take. Bodies are face-connected cells, named by the
    target faces within a cell and a half of them. Says which measure it is,
    and never in the words that mean "finished" (R1)."""
    got = leftover.voxel_residual(target, built, floor=floor, seg=seg)
    r = Result()
    r.line(f"residual of {args.mesh} against {args.built}")
    if got is None:
        r.line(f"NOT MEASURED: {why}, and the pair could not be put on a lattice "
               f"either. Nothing here says the part is finished or unfinished.")
        return r
    n = len(leftover.LATTICES)
    r.line(f"measured by OCCUPANCY, NOT a boolean: {why}. The lattice is "
           f"{got.pitch:.4g} mm, fixed to the target; each number is the mean over "
           f"{n} lattices shifted by a fraction of a cell, with their range, and is "
           f"quantised to the cell")
    r.line(f"target volume about {got.target:.3f}   missing about {got.missing:.3f} "
           f"({got.missing_range[0]:.3f}-{got.missing_range[1]:.3f})   extra about "
           f"{got.extra:.3f} ({got.extra_range[0]:.3f}-{got.extra_range[1]:.3f}) mm^3")
    r.line(f"bodies below {floor:.3f} mm^3 are dropped; size includes one cell")
    r.line(ON_HELP if seg is not None else seg_note)

    def rows(items):
        return [{"mesh": None, "volume": q.volume, "centre": q.centre, "extents": q.extents,
                 "kind": None, "thickness": None, "on": q.on} for q in items]
    _report(r, "MISSING (in the target, not yet built)", "D",
            rows(got.bodies.get("missing", [])), axis)
    _report(r, "EXTRA (built, but not in the target)", "E",
            rows(got.bodies.get("extra", [])), axis)
    if args.export or args.plane or args.solo:
        r.line()
        r.line("no picture and no --export here: the bodies are lattice cells, "
               "not a mesh")
    return r


def _parse_axis(expr):
    """'Y=0,Z=0' -> (free axis index, the two fixed coordinates)."""
    names = "XYZ"
    fixed = {}
    for part in expr.split(","):
        k, _, v = part.partition("=")
        k = k.strip().upper()
        if k not in names:
            raise SystemExit(f"--axis: {k!r} is not one of X, Y, Z")
        fixed[names.index(k)] = float(v)
    if len(fixed) != 2:
        raise SystemExit('--axis needs exactly two constraints, e.g. "Y=0,Z=0"')
    free = next(i for i in range(3) if i not in fixed)
    return free, fixed


def _report(r, title, prefix, bodies, axis, left=None):
    r.line()
    films = leftover.film_note(left) if left is not None else ""
    if not bodies:
        r.line(f"{title}: none" + (f"  ({films})" if films else ""))
        return
    r.line(f"{title}: {len(bodies)} "
           f"{'body' if len(bodies) == 1 else 'bodies'}")
    head = (f"  {'name':<5} {'volume':>11}  {'centre':<24} {'size':<22} {'fill':>5}"
            f"  {'kind':<12} {'thick':>7}")
    if axis:
        head += "  radius   angle"
    if any(b.get("at") for b in bodies[:MAX_LISTED]):
        head += "  at"
    if any(b.get("on") for b in bodies[:MAX_LISTED]):
        head += "  on"
    r.line(head)
    for i, b in enumerate(bodies[:MAX_LISTED]):
        name = f"{prefix}{i}"
        c, e = b["centre"], b["extents"]
        # Fill fraction, because a bounding box is not a shape: a body that fills
        # a few percent of its box is a thin panel, and reading its box as a block
        # sends the model down the wrong path.
        box = float(np.prod(np.maximum(e, 1e-9)))
        fill = 100.0 * b["volume"] / box if box > 0 else 0.0
        thick = b.get("thickness")
        line = (f"  {name:<5} {b['volume']:11.3f}  "
                f"({c[0]:7.2f},{c[1]:7.2f},{c[2]:7.2f})  "
                f"{e[0]:6.2f} x{e[1]:6.2f} x{e[2]:6.2f} {fill:4.0f}%"
                f"  {KIND.get(b.get('kind'), ''):<12} "
                f"{'' if thick is None else format(thick, '7.3g'):>7}")
        if axis:
            free, fixed = axis
            u, v = sorted(fixed)
            du, dv = c[u] - fixed[u], c[v] - fixed[v]
            line += (f"  {np.hypot(du, dv):7.2f}  "
                     f"{np.degrees(np.arctan2(dv, du)) % 360.0:6.1f}deg")
        if b.get("at"):
            line += f"  at {b['at']}"
        if b.get("on"):
            line += f"  {leftover.on_text(b['on'])}"
        r.line(line)
        r.label(name, f"{'missing' if prefix == 'D' else 'extra'} body, "
                      f"volume {b['volume']:.3f} mm^3")
    rest = bodies[MAX_LISTED:]
    if rest:
        r.line(f"  ... and {len(rest)} smaller, {sum(b['volume'] for b in rest):.3f} mm^3 "
               f"in all (the largest is {rest[0]['volume']:.3f})")
    if films:
        r.line(f"  also {films}")


def _render(args, built, missing, extra):
    """The table as a picture: the D bodies green, the E bodies red, and the
    build dim grey behind them for where they sit (render.residual_compose).

    Drawing every body green against the build in red would not work: an extra
    body lies inside the build, so it comes out yellow -- agreement -- and the
    build red wherever nothing was left in front of it, contradicting the
    table. A skin (`face off`) is drawn dim, because its area is not its volume.
    """
    import trimesh
    bodies = missing + extra
    if not bodies:
        return None, None
    out = Path(args.out) if getattr(args, "out", None) else Path("out/agent")
    out.mkdir(parents=True, exist_ok=True)
    png = out / "residual.png"
    lump = trimesh.util.concatenate([b["mesh"] for b in bodies])

    def parts(bs):
        return [(b["mesh"], SKIN_LEVEL if b.get("kind") == "offset" else 1.0) for b in bs]
    res = Residual(parts(missing), parts(extra), None if args.solo else built)

    if args.plane:
        # Levels of the RESIDUAL, not of the target. The whole point is to read
        # where the leftover starts and stops: a picture of it without a scale
        # is the thing this project exists not to hand back.
        marks, _ = feature_levels(lump, args.plane, args.samples)
        png = out / f"residual_{args.plane}.png"
        ortho_views(lump, args.plane, None, png,
                    cfg=RenderConfig(tile=args.tile), marks=marks,
                    pair=(lump, None if args.solo else built), residual=res)
        return png, marks

    if args.solo:
        # Nothing to compare against, on purpose. With the whole part in the frame
        # the eye keeps reading the leftover bodies as part of it; alone, they are
        # just bodies and their shape is the only thing on screen.
        caption = RESIDUAL_SOLO_LEGEND
        other = None
    else:
        caption = RESIDUAL_LEGEND
        other = built
    projections(lump, other, png, cfg=RenderConfig(tile=args.tile),
                caption=caption, views=args.views, residual=res)
    return png, None


registry.register(
    name="residual",
    kind="instrument",
    summary="boolean difference against the target, split into named bodies "
            "with volume, centre and angle -- the remainder as a work list",
    author="Dmitry Zhemchuzhnikov",
    module=__name__,
    configure=configure,
    run=run,
    # The contract smoke test needs something to diff against; this program is
    # a deliberately partial build of the test fixture, so the run finds exactly
    # one missing body and nothing extra.
    meta={"smoke_args": [
        "--built",
        str(Path(__file__).resolve().parents[3] / "tests" / "data"
            / "smoke_program.py")]},
)
