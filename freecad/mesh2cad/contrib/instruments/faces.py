"""faces -- the analytic surfaces the mesh was tessellated from, named.

Every other instrument reads the part through cuts. This one reads its faces:
which planes, cylinders, cones, spheres and tori it is made of, where each one
is, which way it faces and how each pair meets. A fillet is a cylinder or a
torus tangent to the faces it rounds; a hole is a concave cylinder; a boss at
an angle is a cylinder whose axis is not X, Y or Z -- and says so. A spline
turned round an axis is a revolve: its axis and its profile, ring by ring.

    m2c.py faces part.stl
    F0    cylinder  19.9%  concave  R=1.5700 axis |Z at X=-27.9480 Y=-3.1800 ...
    F3    cylinder   6.7%  convex   R=1.6000 axis |X at Y=-3.1800 Z=15.9412 ...
    repeated: F3, F4  2x cylinder R=1.6000 axis |X convex

The numbers are measurements, fitted to every vertex of the face to the
tolerance printed at the top; the names are what later stages will take. The
segmentation itself is `mesh2cad.segment`, and this file only reports it.
"""

import json
import os
from pathlib import Path

import numpy as np

from mesh2cad import registry
from mesh2cad.contract import Result
from mesh2cad.frames import PLANE_ACROSS, frame_for
from mesh2cad.holes import stack_names
from mesh2cad.normalize import ingest
from mesh2cad.segment import (
    coaxial_groups,
    directions,
    equal_groups,
    principal_of,
    segment_cached,
)

DEFAULT_TOP = 30
AXES = "XYZ"
# Seconds `faces` may spend finding round surfaces. It had no limit, and on a
# 670k-face worm-gear pair the search ran past an hour, so the agent's 30-minute
# command limit stopped it with nothing to show, in every run. Half that limit
# leaves room for the rest of the command; a search cut short says so and lists
# what it found. $M2C_FACES_BUDGET overrides it; 0 means no limit.
FACES_SEGMENT_BUDGET = 900.0
# Wires listed, each with its copies; all of them are named.
WIRES_SHOWN = 12


def _budget():
    raw = os.environ.get("M2C_FACES_BUDGET")
    if raw is None:
        return FACES_SEGMENT_BUDGET
    try:
        val = float(raw)
    except ValueError:
        return FACES_SEGMENT_BUDGET
    return None if val <= 0 else val


def configure(p):
    p.add_argument("mesh")
    p.add_argument("--tol", type=float, default=None,
                   help="how far a vertex may sit off a surface and still be on "
                        "it, in mm; default is read off the mesh (coplanar "
                        "neighbours agree to float precision, the rest do not)")
    p.add_argument("--top", type=int, default=DEFAULT_TOP,
                   help="how many surfaces to list, largest first; the rest are "
                        "counted by kind and all of them are in faces.json")
    p.add_argument("--tile", type=int, default=440,
                   help="size in pixels of each view in the surface render (default 440)")
    p.add_argument("--no-render", action="store_true",
                   help="skip the picture; the text and faces.json are unchanged")
    p.add_argument("--cover", default=None, metavar="F#",
                   help="does this surface really run where you think it does? "
                        "A cylinder F# (axis along X, Y or Z) is continued over "
                        "--between, or the whole part, and the target's surface "
                        "on it is measured: the share of that stretch it covers, "
                        "where, and how far round. A plane F# gives the area on "
                        "it and its extent. Before building one feature across a "
                        "stretch, check that its surface spans it")
    p.add_argument("--between", default=None, metavar="A,B",
                   help="with --cover: two named levels along the cylinder's axis")


def _out_dir(args):
    """Where the picture and the JSON go: the same place `overview` writes."""
    out = Path(getattr(args, "out", None) or "out/agent")
    if str(getattr(args, "out", "")) == "out/agent":
        out = out / Path(args.mesh).resolve().parent.name
    out.mkdir(parents=True, exist_ok=True)
    return out


def run(args):
    mesh, info = ingest(args.mesh)
    out = _out_dir(args)
    # Before the segmentation, which can run out of time on a dense wire cage and
    # see none of it: kept at once, so the paths outlive a command that is stopped.
    wires = [] if getattr(args, "cover", None) else _wires_of(mesh, args)
    seg = segment_cached(mesh, cache_dir=out / ".m2c" / "cache", tol=args.tol,
                         budget=_budget())
    S = seg.surfaces
    if getattr(args, "cover", None):
        return _cover(args, mesh, seg, out)
    if getattr(args, "between", None):
        raise SystemExit("--between goes with --cover F#: the stretch of the "
                         "surface's axis to check")

    r = Result()
    r.line(f"faces of {args.mesh}: {len(S)} surfaces from {seg.n_faces} triangles")
    if seg.partial:
        r.line(f"PARTIAL: the search for round surfaces was stopped after "
               f"{seg.budget:.0f} s on these {seg.n_faces} triangles; the surfaces "
               f"below are what it found, and the facets it did not reach are "
               f"counted as planes or freeform (see the note at the end)")
    r.line(f"tolerance {seg.tol_how}; override with --tol")
    if info["alignment"].get("applied"):
        r.line(f"frame: `overview` rotated this part "
               f"{info['alignment']['angle_deg']:.3f} deg onto the axes; "
               f"everything below is in that frame")
    if info.get("scale", 1.0) != 1.0:
        r.line(f"scale: `overview` scaled this part by {info['scale']:g} to "
               f"millimetre size; every number below is in that scale")
    by_kind = {}
    for s in S:
        n, a = by_kind.get(s.kind, (0, 0.0))
        by_kind[s.kind] = (n + 1, a + s.area)
    r.line("area:  " + "   ".join(
        f"{k} {by_kind[k][0]} ({by_kind[k][1] / seg.area:.1%})"
        for k in ("plane", "cylinder", "cone", "sphere", "torus", "revolve", "freeform")
        if k in by_kind))
    for note in seg.notes:
        r.line(f"NOTE: {note}")

    top = max(1, int(args.top))
    shown = S[:top]
    r.line()
    r.line("  name  kind       area  sense    where")
    for s in shown:
        r.line(f"  {s.name:<5} {s.kind:<9} {s.area / seg.area:5.1%}  "
               f"{s.sense:<8} {_where(s)}")
        for note in s.notes:
            r.line(f"        {note}")
    rest = S[top:]
    if rest:
        kinds = {}
        for s in rest:
            kinds[s.kind] = kinds.get(s.kind, 0) + 1
        r.line(f"  ... {len(rest)} more, each under {rest[0].area / seg.area:.2%} "
               f"of the area: " + ", ".join(f"{n} {k}" for k, n in sorted(
                   kinds.items(), key=lambda kv: -kv[1]))
               + f" -- `--top {len(S)}` lists them all; faces.json has every one")

    _joints_block(r, seg)
    _blends_block(r, mesh, seg)
    _groups_block(r, seg)
    _directions_block(r, seg, mesh)
    _wires_block(r, wires, _pipes_block(r, mesh, seg, args), args)

    path = out / "faces.json"
    data = seg.to_dict()
    data["holes"] = [{"name": name, "faces": [S[i].name for i in g]}
                     for name, g in stack_names(seg)]
    path.write_text(json.dumps(data, indent=1))
    r.line()
    r.line(f"every surface, joint and parameter: {path}")

    if not getattr(args, "no_render", False):
        png = _render(args, out, mesh, seg, shown)
        if png is not None:
            r.image(png)

    for s in S:
        if _mentioned(s.name, r.text):
            r.label(s.name, f"{s.kind} ({s.area / seg.area:.1%} of the area)")
    return r


def _mentioned(name, text):
    import re
    return re.search(rf"(?<![A-Za-z0-9_]){name}(?![0-9])", text) is not None


def _where(s):
    p = s.params
    if s.kind == "plane":
        n = p["normal"]
        ax = principal_of(n)
        if ax:
            k = AXES.index(ax)
            sign = "+" if n[k] > 0 else "-"
            txt = f"{ax}={p['offset'] * np.sign(n[k]):+.4f} facing {sign}{ax}"
        else:
            txt = f"normal {_vec(n)} offset {p['offset']:+.4f} ({_off_axis(n)})"
        return txt + (f", {s.loops} loops" if s.loops > 1 else "")
    if s.kind == "freeform":
        return (f"{len(s.faces)} facets that fit no plane, cylinder, cone, sphere "
                f"or torus: a spline, a loft or a variable blend")
    if s.kind == "sphere":
        return f"R={p['radius']:.4f} centre {_pt(p['centre'])}"
    if s.kind == "revolve":
        return (f"revolve of a {len(p['profile'])}-ring profile, R "
                f"{s.extent['radius'][0]:.4f}..{s.extent['radius'][1]:.4f} "
                f"{_axis_text(s)}, {s.extent.get('sweep', 360.0):.1f}deg; "
                f"profile {_profile_text(s)}")
    size = {"cylinder": lambda: f"R={p['radius']:.4f}",
            "cone": lambda: (f"half-angle {p.get('half_angle', 0.0):.3f}deg "
                             f"R {s.extent['radius'][0]:.4f}..{s.extent['radius'][1]:.4f}"),
            "torus": lambda: (f"R={p['major']:.4f} r={p['minor']:.4f}"
                              + (" (spindle: the tube crosses the axis)"
                                 if p["major"] < p["minor"] else ""))}[s.kind]()
    return f"{size} {_axis_text(s)}, {s.extent.get('sweep', 360.0):.1f}deg"


def _principal(s):
    """X, Y or Z when the surface's axis is one of them to within what its fit
    can resolve: a revolve is round only to a micron or so, and an axis known
    to that over a few millimetres is not known to a millionth of a radian."""
    lo, hi = s.extent.get("bbox", (None, None))
    L = float(np.linalg.norm(np.asarray(hi) - np.asarray(lo))) if lo is not None else 0.0
    tol = max(1e-6, 3.0 * float(s.residual) / L) if L > 0 else 1e-6
    return principal_of(s.params["axis"], tol_rad=tol)


def _profile_text(s, most=12):
    """The rings of a revolve as (place along the axis, radius): the place in
    world terms when the axis is X, Y or Z, as every other instrument names
    places; otherwise the distance along the axis from its point."""
    a, c = np.asarray(s.params["axis"], float), np.asarray(s.params["point"], float)
    ax = _principal(s)
    prof = s.params["profile"]
    if ax:
        k = AXES.index(ax)
        pts = [(float(c[k] + t * a[k]), r) for t, r in prof]
        head = f"({ax}, R)"
    else:
        pts = [(t, r) for t, r in prof]
        head = "(t along the axis, R)"
    txt = ", ".join(f"({x:.4f}, {r:.4f})" for x, r in pts)
    if len(pts) > most:
        half = most // 2
        txt = (", ".join(f"({x:.4f}, {r:.4f})" for x, r in pts[:half])
               + f", ... {len(pts) - most} more ..., "
               + ", ".join(f"({x:.4f}, {r:.4f})" for x, r in pts[-half:]))
    return f"{head}: {txt}"


def _axis_text(s):
    """The axis line in the terms every other instrument uses: which axis it
    is parallel to, where it crosses the other two, and how far along it the
    face runs."""
    a, c = s.params["axis"], s.params["point"]
    lo, hi = s.extent.get("along", (0.0, 0.0))
    ax = _principal(s)
    if ax:
        k = AXES.index(ax)
        others = " ".join(f"{AXES[i]}={c[i]:.4f}" for i in range(3) if i != k)
        base = float(c[k])
        return f"axis |{ax} at {others}, {ax} {base + lo:.4f}..{base + hi:.4f}"
    return (f"axis {_vec(a)} through {_pt(c)} ({_off_axis(a)}), "
            f"length {hi - lo:.4f}")


def _vec(v):
    return "(" + ", ".join(f"{x:+.4f}" for x in v) + ")"


def _pt(v):
    return "(" + ", ".join(f"{x:.4f}" for x in v) + ")"


def _off_axis(d):
    d = np.asarray(d, float)
    angs = np.degrees(np.arccos(np.clip(np.abs(d) / np.linalg.norm(d), 0, 1)))
    k = int(np.argmin(angs))
    return f"{angs[k]:.3f}deg off {AXES[k]}"


def _joints_block(r, seg):
    if not seg.joints:
        return
    kinds = {}
    for j in seg.joints:
        kinds[j.kind] = kinds.get(j.kind, 0) + 1
    r.line()
    r.line("joints (how neighbouring surfaces meet): " + ", ".join(
        f"{kinds.get(k, 0)} {k}" for k in ("convex", "concave", "tangent", "mixed")))
    tan = [j for j in seg.joints if j.kind == "tangent"]
    if tan:
        names = [f"{seg.surfaces[j.a].name}~{seg.surfaces[j.b].name}" for j in tan]
        more = len(names) - 24
        r.line("  tangent -- a blend meeting the face it rounds off: "
               + ", ".join(names[:24]) + (f" and {more} more" if more > 0 else ""))


def _blends_block(r, mesh, seg, most=16):
    """The faces `sketch --op fillet|chamfer` can make from a sharp edge, said
    where the model reads the faces: a round named here is one command, rather
    than a profile redrawn with arcs or left square."""
    from mesh2cad.blends import candidates

    found = candidates(mesh, seg)
    if not found:
        return
    r.line()
    r.line("blends -- a sharp edge of the build becomes the face named here with "
           "`sketch --op fillet|chamfer --use F# --base PROG` (convex: cut, concave: added):")
    for op in ("fillet", "chamfer"):
        items = [f"{n} {t}" for n, o, t in found if o == op]
        if items:
            more = len(items) - most
            r.line(f"  {op}: " + ", ".join(items[:most])
                   + (f" and {more} more" if more > 0 else ""))


def _wires_of(mesh, args, start=0):
    """The wires of `mesh`, their paths kept; [] when it has none."""
    from mesh2cad.wires import wire_paths
    try:
        found = wire_paths(mesh, start)
    except Exception as exc:                                       # noqa: BLE001
        print(f"note: no wire paths measured ({type(exc).__name__}: {str(exc)[:80]})")
        return []
    if found:
        _keep_wires(found, args)
    return found


def _keep_wires(found, args):
    """The wires as path records, under this mesh and, for a residual export,
    under the target it came from, the mesh `sketch` is given."""
    from mesh2cad import readback
    from mesh2cad.frame import export_source
    recs = [{"name": w["name"], "source": "wires", "width": 2.0 * w["radius"],
             "radius": w["radius"], "length": w["length"], "points": w["points"],
             "what": f"{w['name']}, a wire off the mesh ({w['length']:.4f} long, "
                     f"R{w['radius']:.4f})"} for w in found]
    readback.save_paths(None, args.mesh, recs)
    up = export_source(args.mesh)
    if up is not None and up["source"] is not None:
        readback.save_paths(None, str(up["source"]), recs)


def _wires_block(r, wires, taken, args):
    """The wires, listed once with their copies, after the pipe paths:
    numbered on from them."""
    from mesh2cad.frame import export_source
    if not wires:
        return
    if taken:
        for i, w in enumerate(wires):
            w["name"] = f"W{taken + i}"
        names = {f"W{i}": f"W{taken + i}" for i in range(len(wires))}
        for w in wires:
            w["copies"] = [names[c] for c in w["copies"]]
        _keep_wires(wires, args)
    up = export_source(args.mesh)
    mesh = up["named"] if up is not None else args.mesh
    total = sum(w["length"] for w in wires)
    r.line()
    r.line(f"wires -- {len(wires)} thin round bodies, {total:.2f} long in all, their "
           f"centrelines read off the mesh (no faces needed):")
    seen, shown = set(), 0
    for w in wires:
        if w["name"] in seen:
            continue
        seen.update(w["copies"])
        if shown == WIRES_SHOWN:
            r.line(f"  ... and more, all named: `--along W#` takes any of "
                   f"W{wires[0]['name'][1:]}..{wires[-1]['name']}")
            break
        shown += 1
        a, b = w["points"][0], w["points"][-1]
        r.line(f"  {w['name']}: R{w['radius']:.4f}, {w['length']:.2f} long, from {_pt(a)} to "
               f"{_pt(b)}" + (f"; the same size: {', '.join(w['copies'])}" if w["copies"] else ""))
    r.line(f"     build one: sketch {mesh} --op sweep --along {wires[0]['name']} --base PROG "
           f"(a round wire of its radius along it)")


def _pipes_block(r, mesh, seg, args):
    """The pipe paths W#: round faces on one centreline chained into a tube's
    whole path, for `sketch --op sweep --along W#`. Without it a bent tube is a
    list of named faces and no path, and the model revolves its bends one by
    one. How many it named."""
    from mesh2cad import pipes, readback

    try:
        found = pipes.pipe_paths(mesh, seg)
    except Exception as exc:                                       # noqa: BLE001
        r.line()
        r.line(f"note: no pipe paths measured ({type(exc).__name__}: {str(exc)[:80]}); "
               f"the faces above are unaffected")
        return 0
    if not found:
        return 0
    r.line()
    r.line("pipe paths -- a tube's centreline, chained from the round faces on it "
           "(walls and bores of its straight runs and bends):")
    for w in found:
        head, ends = pipes.describe(w)
        r.line(f"  {head}")
        r.line(f"  {ends}")
        # An end face that clears the bends, on an axis first (an inclined one
        # is sketched in its frame). A flange wider than its bend radius, swept,
        # would fold through itself.
        bend = pipes.tightest_bend([w["pieces"]])[0]
        faces, wide = [], []
        for p, f in w["ends"]:
            if f:
                (faces if bend is None or pipes.reach_of_face(mesh, seg, f, p) < bend
                 else wide).append(f)
        faces.sort(key=lambda f: principal_of(seg.by_name(f).params["normal"]) is None)
        cut = pipes.across_run(w)
        if faces:
            r.line(f"     build it: sketch {args.mesh} --face {faces[0]} --op sweep --along "
                   f"{w['name']} (the end face, with its bore, swept along the path)")
        else:
            where = (f"--plane {PLANE_ACROSS[cut[0]]} --at {cut[1]:.4f}" if cut
                     else "--plane P --at <a level across a straight run>")
            r.line(f"     build it: sketch {args.mesh} {where} --use <the tube's rings there> "
                   f"--op sweep --along {w['name']} (a section across the tube, swept both "
                   f"ways from it)")
        if wide:
            r.line(f"     end face(s) {', '.join(wide)} reach past the R{bend:.4f} bend from "
                   f"the path -- a flange, built on its own, not swept")
    readback.save_paths(None, args.mesh, [{"name": w["name"], "source": "faces",
                                           "chain": " -> ".join(f[0] for f in w["faces"]),
                                           "ends": "/".join(f or "open" for _, f in w["ends"])}
                                          for w in found])
    return len(found)


def _groups_block(r, seg):
    S = seg.surfaces
    eq = equal_groups(seg)
    if eq:
        r.line()
        r.line("repeated (same kind and size, parallel axes -- a pattern, a "
               "mirror pair, or one feature cut twice):")
        for g in eq[:8]:
            s = S[g[0]]
            r.line(f"  {', '.join(S[i].name for i in g)}: {len(g)}x {s.kind} "
                   f"{_size(s)} {_axis_dir(s)} {s.sense}")
        if len(eq) > 8:
            r.line(f"  ... {len(eq) - 8} more groups in faces.json")
    co = coaxial_groups(seg)
    if co:
        r.line()
        r.line("coaxial (several surfaces round one axis -- the faces of one "
               "revolve profile, or a stack of them; `section --meridian F#` gives "
               "that profile at every angle, whatever holes or blades cross a cut "
               "through the axis):")
        for g in co[:8]:
            s = S[g[0]]
            r.line(f"  {_axis_text(s).split(',')[0]}: "
                   + ", ".join(f"{S[i].name} {S[i].kind} {_size(S[i])}" for i in g))
        if len(co) > 8:
            r.line(f"  ... {len(co) - 8} more groups in faces.json")
    holes = stack_names(seg)
    if holes:
        r.line()
        r.line("holes (concave cylinders and cones on one axis -- a bore with its "
               "steps, countersink, chamfers and drill point; cut one with "
               "`sketch --op hole --use C#`):")
        for name, g in holes[:24]:
            s = S[g[0]]
            # A convex surface on the hole's axis is not the hole's: `--op hole`
            # leaves it. Listed in the stack, a turned end cone would read as part of
            # the bore, and the model would expect the hole to cut it.
            own = [i for i in g if S[i].sense == "concave"]
            other = [i for i in g if S[i].sense != "concave"]
            r.line(f"  {name:<4} {_axis_text(s).split(',')[0]}: "
                   + ", ".join(f"{S[i].name} {S[i].kind} {_size(S[i])}" for i in own)
                   + (f"; on its axis but not the hole's ({', '.join(sorted({S[i].sense for i in other}))}, "
                      f"left by --op hole): {' '.join(S[i].name for i in other)}" if other else ""))
        if len(holes) > 24:
            r.line(f"  ... C24..C{len(holes) - 1} in faces.json")


def _size(s):
    p = s.params
    if s.kind in ("cylinder", "sphere"):
        return f"R={p['radius']:.4f}"
    if s.kind == "torus":
        return f"R={p['major']:.4f} r={p['minor']:.4f}"
    if s.kind == "cone":
        return f"half-angle {p.get('half_angle', 0.0):.3f}deg"
    if s.kind == "revolve":
        return f"R={s.extent['radius'][0]:.4f}..{s.extent['radius'][1]:.4f}"
    return ""


def _axis_dir(s):
    if "axis" not in s.params:
        return ""
    ax = _principal(s)
    return f"axis |{ax}" if ax else f"axis {_vec(s.params['axis'])}"


def _directions_block(r, seg, mesh):
    groups = directions(seg, mesh)
    if not groups:
        return
    r.line()
    r.line("directions carried by plane normals and round-surface axes, by "
           "area -- an operation made along a direction leaves its caps "
           "facing it and its round walls around it:")
    main = [g for g in groups if g["principal"]]
    r.line("  " + "   ".join(f"{g['principal']} {g['frac']:.1%}" for g in main))
    odd = [g for g in groups if not g["principal"]]
    for g in odd[:8]:
        names = ", ".join(seg.surfaces[k].name for k in g["members"][:8])
        fr = frame_for(g["dir"])
        r.line(f"  dir {_vec(g['dir'])} {g['frac']:.1%} -- "
               f"{_off_axis(g['dir'])}: {names}; `--frame "
               f"{seg.surfaces[g['members'][0]].name}` lays it on "
               f"{AXES[fr['k']]} (plane {PLANE_ACROSS[fr['k']]})")
    if odd:
        r.line("  off-axis: ortho / section / sketch work across one with --frame "
               "F# (any face listed on it); `sketch --face F#` on a flat face of "
               "it and `--op hole --use C#` on a stack along it set the frame "
               "themselves")
    if len(odd) > 8:
        rest = odd[8:]
        r.line(f"  ... {len(rest)} more off-axis directions, together "
               f"{sum(g['frac'] for g in rest):.2%} of the area -- mostly single "
               f"facets no surface took; all are in faces.json")


# How far a target vertex may sit off the continued surface and count as on
# it, in the segmentation's tolerance: what round surfaces were allowed.
COVER_TOL = 3.0
# Stretches of the axis shorter than this share of it are not listed apart.
COVER_GAP = 0.01


def _cover(args, mesh, seg, out):
    """`faces --cover F# [--between A,B]`: whether a surface the model means to
    build across a stretch is there across it -- e.g. that a bore runs the whole
    length of a block, rather than only along a groove. Named surfaces and
    levels only."""
    from mesh2cad.section import PLANES, feature_levels
    s = seg.by_name(args.cover)
    if s is None:
        raise SystemExit(f"--cover {args.cover}: no such surface; `faces` names "
                         f"F0..F{len(seg.surfaces) - 1}")
    tol = COVER_TOL * seg.tol
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces)
    r = Result()
    if s.kind == "plane":
        if getattr(args, "between", None):
            raise SystemExit("--between is a stretch of an axis; a plane has none. "
                             "--cover F# alone gives the plane's area and extent")
        n = np.asarray(s.params["normal"], float)
        on_v = np.abs(V @ n - float(s.params["offset"])) <= tol
        tris = np.flatnonzero(on_v[F].all(axis=1) & (np.abs(mesh.face_normals @ n) > 0.99))
        area = float(mesh.area_faces[tris].sum())
        r.line(f"cover: plane {s.name} {_where(s)}: the target has {area:.3f} mm^2 "
               f"of surface on it ({area / seg.area:.1%} of the part's)"
               + ("" if not len(tris) else ", spanning "
               + ", ".join(f"{AXES[k]} {V[F[tris]][..., k].min():.3f}..{V[F[tris]][..., k].max():.3f}"
                           for k in range(3) if abs(n[k]) < 0.99)))
        return r
    if s.kind != "cylinder":
        raise SystemExit(f"--cover takes a cylinder or a plane; {s.name} is a {s.kind}")
    a = np.asarray(s.params["axis"], float)
    a = a / np.linalg.norm(a)
    k = int(np.argmax(np.abs(a)))
    if abs(a[k]) < 0.999:
        raise SystemExit(f"--cover {s.name}: its axis is not X, Y or Z, and levels "
                         f"are named only along those")
    c = np.asarray(s.params["point"], float)
    R = float(s.params["radius"])
    q = V - c
    t = q @ a
    rad = np.linalg.norm(q - np.outer(t, a), axis=1)
    on_v = np.abs(rad - R) <= tol
    tris = np.flatnonzero(on_v[F].all(axis=1))
    lo, hi = (float(x) for x in mesh.bounds[:, k])
    span_txt = "the whole part"
    if getattr(args, "between", None):
        names = [x.strip() for x in args.between.split(",") if x.strip()]
        if len(names) != 2:
            raise SystemExit("--between wants two level names, e.g. --between B0,B3")
        plane = next(p for p, sp in PLANES.items() if sp["axis"] == k)
        got = feature_levels(mesh, plane, 900, cache_dir=out / ".m2c" / "cache")
        marks = {m["name"]: float(m["value"])
                 for m in (got[0] if isinstance(got, tuple) else got) if m.get("name")}
        bad = [x for x in names if x not in marks]
        if bad:
            raise SystemExit(f"--between: {', '.join(bad)} is not a level along "
                             f"{AXES[k]}; the levels are "
                             + ", ".join(f"{n}={v:.4f}" for n, v in marks.items()))
        lo, hi = sorted(marks[x] for x in names)
        span_txt = f"{names[0]}..{names[1]}"
    # Axial intervals the on-surface triangles cover, in world coordinates.
    iv = []
    for f in F[tris]:
        z = V[f, k]
        iv.append((float(z.min()), float(z.max())))
    iv.sort()
    merged = []
    eps = COVER_GAP * (hi - lo)
    for x0, x1 in iv:
        x0, x1 = max(x0, lo), min(x1, hi)
        if x1 <= x0:
            continue
        if merged and x0 <= merged[-1][1] + eps:
            merged[-1][1] = max(merged[-1][1], x1)
        else:
            merged.append([x0, x1])
    covered = sum(b - a_ for a_, b in merged)
    share = covered / (hi - lo) if hi > lo else 0.0
    # How far round: the 5-degree sectors its vertices fall in.
    verts = np.unique(F[tris])
    u, w = [i for i in range(3) if i != k]
    radial = q[verts] - np.outer(t[verts], a)
    ang = np.degrees(np.arctan2(radial[:, w], radial[:, u])) % 360.0
    round_ = 5.0 * len(np.unique(np.floor(ang / 5.0)))
    head = (f"cover: cylinder {s.name} R={R:.4f} {_axis_text(s)}, continued over "
            f"{AXES[k]}={lo:.3f}..{hi:.3f} ({span_txt}): ")
    if not merged:
        r.line(head + "the target has no surface on it there")
        return r
    where = ", ".join(f"{AXES[k]}={x0:.3f}..{x1:.3f}" for x0, x1 in merged[:6])
    more = f" and {len(merged) - 6} more" if len(merged) > 6 else ""
    r.line(head + f"the target has surface on it over {share:.1%} of that length -- "
           f"{where}{more}; about {min(round_, 360.0):.0f} deg round")
    if share < 0.99:
        r.line(f"  it does not span the stretch: a feature built across all of "
               f"{AXES[k]}={lo:.3f}..{hi:.3f} on this surface puts it where the "
               f"target has none")
    return r


def _render(args, out, mesh, seg, shown):
    try:
        from mesh2cad.render import RenderConfig, surface_views
    except Exception:                                              # noqa: BLE001
        return None
    png = out / "faces.png"
    try:
        surface_views(mesh, seg, png, cfg=RenderConfig(tile=args.tile),
                      named=[s.name for s in shown])
    except Exception as exc:                                       # noqa: BLE001
        # A picture that failed is said to have failed; the text stands alone.
        print(f"note: the faces render failed ({type(exc).__name__}: {exc}); "
              f"the table above is complete without it")
        return None
    return png


registry.register(
    name="faces",
    kind="instrument",
    summary="the planes, cylinders, cones, spheres, tori and revolved splines "
            "the mesh was tessellated from, named, with how each pair meets and "
            "which directions they carry",
    author="mesh2cad",
    module=__name__,
    configure=configure,
    run=run,
)
