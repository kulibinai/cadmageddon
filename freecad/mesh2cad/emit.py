"""Turn named section primitives into FreeCAD source.

The model should not be writing coordinates by hand. It has just been shown a
section whose every edge carries a name, a radius and a centre; the natural next
instruction is "sketch A0_0 and A1_0 at Y=0 and extrude to the boundary", and
this module is what makes that instruction executable.

Emitting code rather than building geometry directly keeps the result auditable:
the program is the deliverable, and a model that can read what was emitted can
correct it.

What is emitted is the BODY of one feature block of a FreeCAD program: sketches
and PartDesign features written through the prelude's helpers
(mesh2cad/fc/prelude.py) on the block `_b`. The block decides whether `_pad`
becomes a Pad or a Pocket, so each function here writes its feature once,
whatever `--combine` asks for. A feature that removes from its own solid (a loft
with holes, a coil trimmed to its band) says so with `_b.mixed = True`.
"""

import math

import numpy as np

# Every number in an emitted program is written to this many decimals, so this
# is the grid the built solid actually lands on.
EMIT_DECIMALS = 4


class EmitError(ValueError):
    """A feature that cannot be written as asked; the message says why."""


def quantise(x):
    """The value that will actually be emitted for `x`.

    Levels are put through this BEFORE any distance is computed from them, and
    that ordering is the whole point. A feature emitted as a start plane and a
    travel distance rounded INDEPENDENTLY would reach round(level) +
    round(target - level) -- which can miss round(target) by a full 1e-4.

    That matters because of a threshold measured rather than assumed:
    manifold3d unions two solids meeting on a face into ONE body at a gap of
    1e-7, and leaves TWO at 1e-6. Independent rounding can therefore open a gap
    a hundred times wider than the one that breaks the part -- and the failure
    is invisible, because volumetric IoU cannot see a part in pieces.

    Quantising first makes any two features that name the same level agree on
    it exactly, by construction, whatever arithmetic happens in between. The
    same measurement says profile mismatch does NOT matter: two prisms meeting
    face to face union into one body whether the second is narrower, wider or
    offset. Only the gap along the axis ever mattered.
    """
    return round(float(x), EMIT_DECIMALS)


def _fmt(x):
    # "-0" and "0" are one point; ring_code compares points as written.
    s = f"{float(x):.4f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def _pt(p):
    return f"({_fmt(p[0])}, {_fmt(p[1])})"


def arc_midpoint(seg):
    """A point on the arc, halfway along it -- what threePointArc needs."""
    c = np.asarray(seg["centre"], float)
    r = float(seg["radius"])
    a0 = np.arctan2(seg["from_xy"][1] - c[1], seg["from_xy"][0] - c[0])
    sweep = np.radians(seg["sweep_deg"]) * (1.0 if seg["ccw"] else -1.0)
    a = a0 + sweep / 2.0
    return c + r * np.array([np.cos(a), np.sin(a)])


def ring_code(ring, sk):
    """The prelude call that draws one closed ring into sketch `sk`.

    A ring that is a single full circle is `_circle` rather than a degenerate
    arc chain -- shorter, and it survives a reader's eye. Arcs are written
    through three points on the grid (`("A3", start, mid, end)`), so the
    circle is fixed by points the program states exactly.
    """
    segs = ring["segments"]
    s = _circle_of(segs)
    if s is not None:
        return f"_circle({sk}, {_pt(s['centre'])}, {_fmt(s['radius'])})"

    # Every point as it will be WRITTEN. A segment shorter than the 4-decimal
    # grid ends on the point the pen is already at, and the kernel refuses the
    # zero-length edge, so the program cannot be scored at all. Such a
    # segment is dropped; an arc whose middle lands on one of its ends is
    # written as the line it is.
    start = _pt(segs[0]["from_xy"])
    out, at = [], start
    path = [(None, segs[0]["from_xy"])]
    segs = _closing(segs, max(HEAL_TOL, float(ring.get("tol") or 0.0)))
    for k, s in enumerate(segs):
        to = _pt(s["to_xy"])
        if to == at:
            continue
        if s.get("_close"):
            # The last primitive ends within the fit tolerance of the
            # start, so it ends there: a closing edge a few microns long
            # is refused by the kernel.
            if s["kind"] == "line":
                continue
            to = start
        # A point that lands within HEAL_TOL of where the pen is, or -- for
        # the last one -- of where the ring started, is the same corner
        # fitted twice. Closing on it draws a bow-tie a few microns across, and
        # the kernel rejects the whole solid. The corner is written once.
        last = k == len(segs) - 1
        if s["kind"] == "line" and (_near(s["to_xy"], _back(at)) or
                                    (last and to != start
                                     and _near(s["to_xy"], segs[0]["from_xy"]))):
            continue
        mid = _pt(arc_midpoint(s)) if s["kind"] == "arc" else None
        if mid is None or mid in (at, to):
            out.append(f'("L", {at}, {to})')
            path.append((None, s["to_xy"]))
        else:
            out.append(f'("A3", {at}, {mid}, {to})')
            path.append((s, s["to_xy"]))
        at = to
    if at != start:
        out.append(f'("L", {at}, {start})')          # the closing edge
    code = f"_ring({sk}, [{_wrap(out, 3)}])"
    crossing = _crossing(path)
    if crossing is not None:
        # Said, not refused: the kernel tolerates some crossings a polygon
        # test does not, and a program that builds is worth more than a
        # refusal. When it does not, this names the ring that broke it.
        print(f"note: ring {ring.get('name', '?')} crosses itself near "
              f"({crossing[0]:.4f}, {crossing[1]:.4f}) as written; if --check says "
              f"NOT A VALID SOLID, this ring is why -- measure it at another level "
              f"(--at mid(A,B) inside the feature) or leave it out of --use")
    return code


def _wrap(items, per_line):
    """Items joined with ", ", `per_line` to a line, so a 500-segment ring is a
    readable block rather than one line of 40 000 characters."""
    rows = [", ".join(items[i:i + per_line]) for i in range(0, len(items), per_line)]
    return ",\n           ".join(rows)


# Points nearer than this are one corner fitted twice (see ring_code). Ten
# grid steps: far below any real edge, far above float noise.
HEAL_TOL = 10.0 ** (1 - EMIT_DECIMALS)


def _closing(segs, tol):
    """`segs` with the last one marked `_close` when it ends within `tol` (the
    ring's fit tolerance) of the ring's start without being on it: lines under
    `tol` at the end are dropped first, so the primitive before them closes the
    ring. A ring of fewer than three primitives is left as it is."""
    if len(segs) < 3:
        return segs
    start = segs[0]["from_xy"]

    def gap(p):
        return float(np.hypot(p[0] - start[0], p[1] - start[1]))
    out = list(segs)
    while (len(out) > 3 and out[-1]["kind"] == "line"
           and float(np.hypot(*(np.subtract(out[-1]["to_xy"], out[-1]["from_xy"])))) < tol
           and gap(out[-1]["to_xy"]) < tol):
        out.pop()
    last = out[-1]
    if _pt(last["to_xy"]) != _pt(start) and gap(last["to_xy"]) < tol:
        out[-1] = {**last, "_close": True}
    return out


def _back(p_written):
    """A written point "(x, y)" back as numbers."""
    x, y = p_written.strip("()").split(",")
    return (float(x), float(y))


def _near(a, b):
    return float(np.hypot(quantise(a[0]) - b[0], quantise(a[1]) - b[1])) < HEAL_TOL


def _crossing(path):
    """Where a written ring crosses itself, or None. Arcs are sampled densely,
    so a valid arc beside a line is not mistaken for a crossing."""
    from shapely.geometry import Polygon
    from shapely.validation import explain_validity
    pts = []
    for seg, to in path:
        if seg is not None:
            c = np.asarray(seg["centre"], float)
            r = float(seg["radius"])
            a0 = np.arctan2(seg["from_xy"][1] - c[1], seg["from_xy"][0] - c[0])
            sw = np.radians(seg["sweep_deg"]) * (1.0 if seg["ccw"] else -1.0)
            for t in np.linspace(0, 1, 17)[1:-1]:
                pts.append(tuple(c + r * np.array([np.cos(a0 + sw * t), np.sin(a0 + sw * t)])))
        pts.append((quantise(to[0]), quantise(to[1])))
    if len(pts) < 4:
        return None
    pg = Polygon(pts)
    if pg.is_valid:
        return None
    why = explain_validity(pg)
    import re
    m = re.search(r"\[(-?[\d.e+-]+) (-?[\d.e+-]+)\]", why)
    return (float(m.group(1)), float(m.group(2))) if m else tuple(pts[0])


def _circle_of(segs):
    """The lone arc of a ring that is one full circle, else None.

    A lone arc that ends where it started IS a circle, whatever sweep got
    recorded. Going through --split can lose the 360: a closed circle the cut
    never touches comes back out with sweep=0, and an arc through three
    coincident points is refused by OCC (`GC_MakeArcOfCircle::Value() - no
    result`).
    """
    if len(segs) != 1 or segs[0]["kind"] != "arc":
        return None
    s = segs[0]
    span = float(np.linalg.norm(np.asarray(s["from_xy"], float)
                                - np.asarray(s["to_xy"], float)))
    if abs(abs(s["sweep_deg"]) - 360) < 1 or span <= 1e-6 * max(1.0, s["radius"]):
        return s
    return None


def written_area(ring):
    """The area the ring encloses as WRITTEN, on the 4-decimal grid (arcs by
    their three points)."""
    segs = ring["segments"]
    c = _circle_of(segs)
    if c is not None:
        return math.pi * quantise(c["radius"]) ** 2
    pts = [segs[0]["from_xy"]]
    for s in segs:
        if s["kind"] == "arc":
            pts.append(arc_midpoint(s))
        pts.append(s["to_xy"])
    xy = np.array([[quantise(p[0]), quantise(p[1])] for p in pts])
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def nothing_left(rings, op):
    """The refusal when every ring asked for encloses no area.

    Said so it can be acted on: a ring sketched at the level where a feature
    starts only grazes the material, and a vaguer refusal is easily misread
    as the ring name not being parsed.
    """
    names = ", ".join(str(r.get("name", "?")) for r in rings) or "the rings"
    return (f"{names} enclose{'s' if len(rings) == 1 else ''} no area once written "
            f"on the 1e-4 grid: the section at this level only grazes the material "
            f"(a level where a feature starts or ends cuts it edge-on). Sketch at a "
            f"level inside the feature, e.g. --at mid(B0,B1), and extend with --to "
            f"or --both; nothing to {op}")


def drawable(rings):
    """(kept, note): the rings that still enclose area once written, and what
    was left out, said in the program.

    A sliver thinner than the grid comes out of `ring_code` as
    `moveTo(P).close()` once its zero-length segments are dropped, and OCC
    stops the whole program on it ("No start point specified - cannot close").
    """
    kept = [r for r in rings if written_area(r) > 10.0 ** (-2 * EMIT_DECIMALS)]
    gone = len(rings) - len(kept)
    note = (f"# note: {gone} ring(s) thinner than the 1e-4 grid the program is "
            f"written on left out -- they enclose no area and OCC cannot close them\n"
            if gone else "")
    return kept, note


def _half_mid(h):
    c = np.asarray(h["centre"], float)
    p0 = np.asarray(h["from"], float)
    p1 = np.asarray(h["to"], float)
    # The far side of the arc from the chord: rotate the chord midpoint out to
    # the radius. Which of the two possible bulges is correct is decided by the
    # half we were asked to keep, which is encoded in the endpoint order.
    chord_mid = (p0 + p1) / 2.0
    d = chord_mid - c
    n = np.linalg.norm(d)
    if n < 1e-9:
        # Diameter through the centre: pick the normal to the chord instead.
        t = p1 - p0
        d = np.array([-t[1], t[0]])
        d /= np.linalg.norm(d)
        # Endpoint order sets the sweep direction, so the bulge follows it.
        t2 = p1 - p0                      # 2-D cross product (np.cross wants 3-D in numpy 2)
        if t2[0] * d[1] - t2[1] * d[0] < 0:
            d = -d
        return c + d * float(h["radius"])
    return c + d / n * float(h["radius"])


def plane_expr(plane, level):
    """The prelude placement of a sketch on `plane` exactly at world `level`.

    The level is written on the grid every feature uses, so two features
    naming one level agree on it to the last digit (see `quantise`). The
    placement is explicit -- (u, v, n) of section.PLANES, XZ's normal -Y -- so
    nothing depends on how FreeCAD orients its own named planes.
    """
    return f'_plane_at("{plane}", {_fmt(quantise(level))})'


def workplane_expr(plane, sense, level):
    """`plane_expr` under the name callers use. `sense` is folded into the
    explicit placement and not needed."""
    return plane_expr(plane, level)


def extrude_distance(level, target, sense):
    """Signed distance for .extrude() to travel from `level` to `target`.

    From the QUANTISED endpoints, so the solid stops exactly on the value a
    neighbouring feature will emit for the same level. See `quantise`.
    """
    return (quantise(target) - quantise(level)) * sense


def _pad_args(dist, both=False, taper=None):
    """The `_pad` arguments for a signed distance along the sketch normal."""
    if both:
        # `both` means d EACH way; FreeCAD's Symmetric length is the total.
        return f'{_fmt(2 * abs(dist))}, side="sym"'
    out = _fmt(abs(dist)) + (", reversed=True" if dist < 0 else "")
    if taper:
        # The harness measures a draft positive INWARD; FreeCAD's positive
        # taper widens the section.
        out += f", taper={_fmt(-float(taper))}"
    return out


def _poly_of(ring):
    """The ring as written, as a shapely polygon (arcs by their three points)."""
    from shapely.geometry import Polygon
    segs = ring["segments"]
    c = _circle_of(segs)
    if c is not None:
        from shapely.geometry import Point
        return Point(quantise(c["centre"][0]), quantise(c["centre"][1])).buffer(
            quantise(c["radius"]), 64)
    pts = [segs[0]["from_xy"]]
    for s in segs:
        if s["kind"] == "arc":
            pts.append(arc_midpoint(s))
        pts.append(s["to_xy"])
    return Polygon([(quantise(p[0]), quantise(p[1])) for p in pts]).buffer(0)


def _holes_by_outer(outer_rings, hole_rings):
    """([(outer, [its holes])], mixed, notes): each hole goes into the sketch of
    the outer ring it lies in, which makes it a hole of THAT face. A hole that
    lies in two outer rings at once is cut from their union -- one face cannot
    say that, so the feature is mixed (tool body, holes as their own minus
    features). A hole in no outer ring cuts nothing and is left out, saying so."""
    polys = [_poly_of(r) for r in outer_rings]
    groups = [(r, []) for r in outer_rings]
    mixed, notes = False, []
    for h in hole_rings:
        q = _poly_of(h).representative_point()
        inside = [i for i, pg in enumerate(polys) if pg.contains(q)]
        if len(inside) == 1:
            groups[inside[0]][1].append(h)
        elif inside:
            mixed = True
        else:
            notes.append(f"# note: hole ring {h.get('name', '?')} lies in no outer "
                         f"ring; it cuts nothing and is left out")
    return groups, mixed, notes


def build_extrude(plane, sense, level, target, outer_rings, hole_rings,
                  both=False, taper=None):
    """Pad each outer ring with the holes that lie in it, from `level` to
    `target` (or `both` ways: the same distance either side of the plane).

    `both` exists because a feature centred on the plane it was measured on --
    a spoke cut at its own mid-thickness -- otherwise has to be built as two
    halves and unioned, which puts a coincident face straight down the middle
    of it for no reason.

    A hole is a hole of its outer ring's face, so one Pad (or Pocket) carries
    both: the outline minus its holes. A drafted outline keeps straight holes,
    which one face cannot say: then the holes are their own features (mixed).
    """
    pl = plane_expr(plane, level)
    dist = extrude_distance(level, target, sense)
    asked = outer_rings
    outer_rings, note = drawable(outer_rings)
    hole_rings, hole_note = drawable(hole_rings)
    if not outer_rings:
        raise EmitError(nothing_left(asked, "extrude"))
    groups, mixed, notes = _holes_by_outer(outer_rings, hole_rings)
    mixed = mixed or bool(taper and hole_rings)
    lines = (note + hole_note).splitlines() + notes
    if mixed:
        lines.append("_b.mixed = True")
    for i, (ring, holes) in enumerate(groups):
        v = f"_s{i}"
        lines += [f"{v} = _sketch(_b, {pl})", ring_code(ring, v)]
        if not mixed:
            lines += [ring_code(h, v) for h in holes]
        lines.append(f"_pad(_b, {v}, {_pad_args(dist, both, taper)})")
    if mixed:
        for j, h in enumerate(hole_rings):
            v = f"_h{j}"
            lines += [f"{v} = _sketch(_b, {pl})", ring_code(h, v),
                      f"_pad(_b, {v}, {_pad_args(dist, both)}, minus=True)"]
    return "\n".join(lines), dist


def revolve_axis(angle, ax0, ax1):
    """(p0, p1, angle) a signed revolve is written with: the sense the right
    hand gives about ax0 -> ax1, as a positive angle. A negative angle turns
    |angle| about the reversed axis -- the same line, the other way. A FreeCAD
    Revolution takes no negative angle at all and refuses the feature ("Angle
    of revolution too small")."""
    if angle < 0:
        return ax1, ax0, -angle
    return ax0, ax1, angle


def build_revolve_rings(plane, sense, level, rings, axis, angle=360.0):
    """Revolve a closed profile that was cut from the section at `axis`.

    This is the general case the half-arc path could not express. A meridian is
    usually a mix of lines and arcs -- every frustum, cup, cone and stepped
    shaft -- and revolving it needs the whole ring, not one primitive. Cutting
    the section at the axis and closing it there turns any section into a valid
    revolve profile, so the operation stops being sphere-only.

    One Revolution per ring, about one datum axis: rings revolved together in
    one sketch would make a ring inside another a hole instead of adding it.
    """
    pl = plane_expr(plane, level)
    a0, a1, turn = revolve_axis(angle, *axis)
    asked = rings
    rings, note = drawable(rings)
    if not rings:
        raise EmitError(nothing_left(asked, "revolve"))
    lines = note.splitlines() + [f"_ax = _axis_in(_b, {pl}, {_pt(a0)}, {_pt(a1)})"]
    for i, ring in enumerate(rings):
        v = f"_s{i}"
        lines += [f"{v} = _sketch(_b, {pl})", ring_code(ring, v),
                  f"_revolve(_b, {v}, _ax, {_fmt(turn)})"]
    return "\n".join(lines), angle


def build_revolve_profile(plane, sense, level, outer_rings, hole_rings, axis, angle=360.0):
    """Revolve closed rings -- outlines and the holes in them -- about an axis in
    the sketch plane that crosses none of them.

    Each outline is revolved and unioned, each hole revolved and cut, so an
    annulus about an axis off the profile is a hollow pipe bend. `--split`
    would turn the hole into a second solid and fill the bore.

    In the tree a hole is a hole of its outline's sketch, so one Revolution is
    the hollow bend; a hole lying in two outlines is cut from their union --
    one sketch cannot say that, so
    then the holes are Revolutions of their own that remove from the block's
    own solid (mixed), as in build_extrude.
    """
    pl = plane_expr(plane, level)
    a0, a1, turn = revolve_axis(angle, *axis)
    asked = outer_rings
    outer_rings, note = drawable(outer_rings)
    hole_rings, hole_note = drawable(hole_rings)
    if not outer_rings:
        raise EmitError(nothing_left(asked, "revolve"))
    groups, mixed, notes = _holes_by_outer(outer_rings, hole_rings)
    lines = (note + hole_note).splitlines() + notes
    if mixed:
        lines.append("_b.mixed = True")
    lines.append(f"_ax = _axis_in(_b, {pl}, {_pt(a0)}, {_pt(a1)})")
    for i, (ring, holes) in enumerate(groups):
        v = f"_s{i}"
        lines += [f"{v} = _sketch(_b, {pl})", ring_code(ring, v)]
        if not mixed:
            lines += [ring_code(h, v) for h in holes]
        lines.append(f"_revolve(_b, {v}, _ax, {_fmt(turn)})")
    if mixed:
        for j, h in enumerate(hole_rings):
            v = f"_h{j}"
            lines += [f"{v} = _sketch(_b, {pl})", ring_code(h, v),
                      f"_revolve(_b, {v}, _ax, {_fmt(turn)}, minus=True)"]
    return "\n".join(lines), angle


def build_revolve(plane, sense, level, half, angle=360.0):
    """A half circle closed by its own diameter, revolved about that diameter."""
    pl = plane_expr(plane, level)
    a0, a1, turn = revolve_axis(angle, *half["revolve_axis"])
    mid = _pt(_half_mid(half))
    code = (f"_s0 = _sketch(_b, {pl})\n"
            f"_ring(_s0, [(\"A3\", {_pt(half['from'])}, {mid}, {_pt(half['to'])}), "
            f"(\"L\", {_pt(half['to'])}, {_pt(half['from'])})])\n"
            f"_ax = _axis_in(_b, {pl}, {_pt(a0)}, {_pt(a1)})\n"
            f"_revolve(_b, _s0, _ax, {_fmt(turn)})")
    return code, angle
def build_helical_sweep(pitch, height, base, coil_r, wire_r, handed="right",
                        phase=0.0):
    """Sweep a measured circular profile along a fitted helix.

    Every number here is measured, and each by whichever instrument measures it
    best. `wire_r` and `coil_r` come from the section -- it fits the wire's own
    arc to a few microns, where the helix detector's slice centroids can put
    the coil radius well out. `pitch` and `handedness` come from the helix fit, which
    is the only thing that sees them. `height` and `base` come from the mesh's
    extent along the axis, because turns-times-pitch does not: the turn count is
    fitted on single-loop slices, which deliberately skip the ground end turns,
    so multiplying the two overshoots.

    The path is a Part::Helix; the circle stands square to it at its start
    (where the helix's own tangent says, so a left-handed coil is square too)
    and is swept along it in Frenet mode -- measured exact against pi r^2 L.

    `phase` is where the coil starts. A helix turned about its own axis is a
    different solid: with the height and both radii correct but the phase left
    at zero, the coil sits in the right envelope and almost nowhere near
    itself. It is
    measured from the ring the model named, which sits at a known height and a
    known angle and so anchors the turn it belongs to.
    """
    left = handed != "right"
    code = (f"_hx = _helix_path(_b, {_fmt(quantise(abs(pitch)))}, {_fmt(quantise(height))}, "
            f"{_fmt(quantise(coil_r))}, base={_fmt(quantise(base))}, left={left}, "
            f"phase={_fmt(quantise(phase))})\n"
            f"_s0 = _sketch(_b, _helix_start(_hx), world=True)\n"
            f"_circle(_s0, (0, 0), {_fmt(quantise(wire_r))})\n"
            f"_pipe(_b, _s0, _hx, mode=\"Frenet\")")
    return code, height


# Local frame of a helix: its axis is local +Z. Rows: world = M @ local.
HELIX_FRAMES = {"Z": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                "Y": ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
                "X": ((0, 0, 1), (1, 0, 0), (0, 1, 0))}


def _chords(a, m, b, max_deg=12.0):
    """Points along the arc through a, m, b (exclusive of a, inclusive of b)."""
    (ax, ay), (bx, by), (cx_, cy_) = a, m, b
    d = 2 * (ax * (by - cy_) + bx * (cy_ - ay) + cx_ * (ay - by))
    if abs(d) < 1e-12:
        return [b]
    ux = ((ax * ax + ay * ay) * (by - cy_) + (bx * bx + by * by) * (cy_ - ay)
          + (cx_ * cx_ + cy_ * cy_) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx_ - bx) + (bx * bx + by * by) * (ax - cx_)
          + (cx_ * cx_ + cy_ * cy_) * (bx - ax)) / d
    t0, t1, t2 = (math.atan2(y - uy, x - ux) for x, y in (a, m, b))

    def near(t, ref):
        while t - ref > math.pi:
            t -= 2 * math.pi
        while t - ref < -math.pi:
            t += 2 * math.pi
        return t
    t1 = near(t1, t0)
    t2 = near(t2, t1)
    r = math.hypot(ax - ux, ay - uy)
    n = max(2, int(math.ceil(abs(math.degrees(t2 - t0)) / max_deg)))
    return [(ux + r * math.cos(t0 + (t2 - t0) * k / n), uy + r * math.sin(t0 + (t2 - t0) * k / n))
            for k in range(1, n + 1)]


def build_profile_sweep(profile, pitch, lefthand, z0, band, axis, centre, theta, r_mean,
                        r_max, var=None):
    """Any closed profile carried along a helix through the band [zA, zB]
    as a ruled solid trimmed to the band.

    `profile` is [(op, points)] in the helix's local frame -- the profile's own
    meridian plane laid on local XZ, r along X, the axis along Z -- with op
    "move"/"line" (one point) or "arc" (mid point, end point; written as chords
    of at most 12 deg). Every profile vertex is wound round the axis as its own
    helix; neighbouring helices bound one ruled face, the profile caps both ends,
    and the shell is the solid. Whole turns are built from a whole number of
    turns below the profile, where the same profile is, so the profile anchors
    the phase; `theta` turns it back to the angle it was measured at, and `axis`,
    `centre` put the local frame where the helix is.

    Not OCC's sweep: a sweep of a tooth may not mesh watertight and comes out
    too big, and one-turn sweeps patterned along the axis do not fuse -- N
    turns are an invalid compound of exactly N x one turn. The ruled solid is valid, meshes closed, and
    unions with the core. It is computed by the prelude's `_ruled_coil` and
    stored as its block's base solid (it has no PartDesign equivalent).
    """
    za, zb = band
    # The pitch as it is written: the helices are a whole number of turns of
    # THIS pitch, or the end cap, turned by whole turns, misses their ends and
    # the shell does not close, and SolidFromShell fails.
    p = float(_fmt(abs(float(pitch))))
    pts = []
    for op, ps in profile:
        if op == "arc":
            pts.extend(_chords(pts[-1], ps[0], ps[1]))
        else:
            pts.append(ps[0])
    # The last point is the ring's closing one, and a ring fitted from a cut
    # closes onto its start within the fit, not to float, and that point,
    # kept, makes a sliver face that builds the coil inside out.
    size = max(math.dist(pts[0], b) for b in pts) or 1.0
    if math.dist(pts[0], pts[-1]) < max(1e-6, 0.02 * size):
        pts.pop()
    lo = min(z for _, z in pts) - z0
    hi = max(z for _, z in pts) - z0
    k0 = int(math.floor((za - (z0 + hi)) / p)) - 1
    k1 = int(math.ceil((zb - (z0 + lo)) / p))
    dz, h = k0 * p, (k1 - k0) * p
    rows = ",\n         ".join(", ".join(f"({_fmt(r)}, {_fmt(z + dz)})" for r, z in pts[i:i + 4])
                              for i in range(0, len(pts), 4))
    turn_end = (-1.0 if lefthand else 1.0) * 360.0 * (k1 - k0)
    # The local frame turned onto the world's: a third of a turn about (1,1,1)
    # cycles the axes (HELIX_FRAMES), then the axis is moved where the helix is.
    turn = {"Z": 0, "Y": -120, "X": 120}[axis]
    coil = (f"_ruled_coil({var or ''}_pts, {_fmt(p)}, {_fmt(h)}, {bool(lefthand)}, "
            f"{_fmt(turn_end)}, ({_fmt(za)}, {_fmt(zb)}), {_fmt(r_max)}, "
            f"theta={_fmt(theta)}, turn={turn}, "
            f"centre=({_fmt(centre[0])}, {_fmt(centre[1])}, {_fmt(centre[2])}))")
    if var:
        # One region of a coil of several: the caller stores them fused.
        return f"{var}_pts = [{rows}]\n{var} = {coil}"
    return (f"_b.mixed = True      # the coil is trimmed to its band before it joins\n"
            f"_pts = [{rows}]\n"
            f"_stored(_b, {coil})")
def build_path_sweep(plane, level, pieces, radius, capped=False):
    """A round wire swept along a planar path given as centreline pieces of a
    strip: ("line", p0, p1) and ("arc", p0, mid, p1) in the section's
    (u, v) at `level`. The circle starts square to the path; joints are swept
    round. `capped` adds a ball at each end, as a wire with rounded ends -- the
    half-circles that close a strip's ends in the section."""
    from mesh2cad.section import PLANES
    spec = PLANES[plane]

    def w(p):
        v = [0.0, 0.0, 0.0]
        v[spec["u"]], v[spec["v"]], v[spec["axis"]] = p[0], p[1], quantise(level)
        return f"({_fmt(v[0])}, {_fmt(v[1])}, {_fmt(v[2])})"
    first = pieces[0]
    p0 = np.asarray(first[1], float)
    if first[0] == "line":
        t = np.asarray(first[2], float) - p0
    else:
        # The tangent at the start of the arc through p0, mid, p1.
        a, m, b = (np.asarray(x, float) for x in first[1:])
        d = 2 * (a[0] * (m[1] - b[1]) + m[0] * (b[1] - a[1]) + b[0] * (a[1] - m[1]))
        c = np.array([((a @ a) * (m[1] - b[1]) + (m @ m) * (b[1] - a[1]) + (b @ b) * (a[1] - m[1])) / d,
                      ((a @ a) * (b[0] - m[0]) + (m @ m) * (a[0] - b[0]) + (b @ b) * (m[0] - a[0])) / d])
        rv = a - c
        t = np.array([-rv[1], rv[0]])
        if np.dot(t, m - a) < 0:
            t = -t
    t = t / (np.linalg.norm(t) or 1.0)       # a short first piece must not round to 0
    t3 = [0.0, 0.0, 0.0]
    t3[spec["u"]], t3[spec["v"]] = float(t[0]), float(t[1])
    segs = []
    for p in pieces:
        if p[0] == "line":
            segs.append(f'("L", {_pt(p[1])}, {_pt(p[2])})')
        else:
            segs.append(f'("A3", {_pt(p[1])}, {_pt(p[2])}, {_pt(p[3])})')
    code = (f"_s0 = _sketch(_b, {plane_expr(plane, level)})\n"
            f"_ring(_s0, [{_wrap(segs, 3)}], closed=False)\n"
            f"_s1 = _sketch(_b, _towards({w(p0)}, ({_fmt(t3[0])}, {_fmt(t3[1])}, "
            f"{_fmt(t3[2])})))\n"
            f"_circle(_s1, (0, 0), {_fmt(radius)})\n"
            # 'Transformed': the snapped pieces meet within a degree of their
            # tangents, and there a round corner can build the wire as an
            # invalid solid.
            f"_pipe(_b, _s1, _s0, transition=\"Transformed\")")
    if capped:
        last = pieces[-1][-1]
        for q in (pieces[0][1], last):
            code += f"\n_sphere(_b, {w(q)}, {_fmt(radius)})"
    return code


def _w3(p):
    return f"({_fmt(p[0])}, {_fmt(p[1])}, {_fmt(p[2])})"


def _d3(v):
    """A unit direction, written finely enough that a sketch placed on it puts
    its points back within the 4-decimal grid."""
    return f"({float(v[0]):.9f}, {float(v[1]):.9f}, {float(v[2]):.9f})"


def _path_code(path, var="_path0"):
    """The `_path` of 3-D path pieces ("line", p0, p1), ("arc", p0, mid, p1)
    or ("spline", p0, p1, ...), in world coordinates: the spine a pipe is
    swept along, beside the Body (prelude `_path`)."""
    kinds = {"line": "L", "arc": "A3", "spline": "S"}
    pieces = ",\n    ".join(f'("{kinds[q[0]]}", ' + ", ".join(_w3(p) for p in q[1:]) + ")"
                            for q in path)
    return f"{var} = _path(_b, [\n    {pieces}])"


def _ring_plane(pts):
    """(origin, u, v, n, uv) of a closed ring of 3-D points that lie in one
    plane: the plane's frame, n chosen so the ring runs counter-clockwise in
    it (every station's ring the same way round), and the ring's (u, v)."""
    P = np.asarray(pts, float)
    o = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - o)
    u, n = vt[0], vt[2]
    v = np.cross(n, u)
    uv = np.c_[(P - o) @ u, (P - o) @ v]
    x, y = uv[:, 0], uv[:, 1]
    if float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) < 0:
        n, v = -n, -v
        uv[:, 1] = -uv[:, 1]
    return o, u, v, n, uv


def build_station_sweep(path, sections):
    """A section swept along a 3-D path through K sections: `sections`
    is one list per station, [outline points, hole points, ...], each ring as
    3-D points matched across the stations (same count, start and direction).
    The outline's multi-section sweep, each hole's cut from it.

    Each station ring is a sketch on its own plane (the ring's points are
    planar: a station is square to the path, an end ring on the plane it was
    named on) and the pipe carries the first through the others
    (Transformation "Multisection")."""
    lines = [_path_code(path)]
    if len(sections[0]) > 1:
        lines.append("_b.mixed = True")
    for i in range(len(sections[0])):
        names = []
        for j, sec in enumerate(sections):
            o, u, v, n, uv = _ring_plane(sec[i])
            sk = f"_w{i}_{j}"
            pts = [f"({_fmt(a)}, {_fmt(b)})" for a, b in uv]
            lines.append(f"{sk} = _sketch(_b, _frame({_w3(o)}, {_d3(u)}, {_d3(v)}, {_d3(n)}))")
            lines.append(f"_polygon({sk}, [{_wrap(pts, 6)}])")
            names.append(sk)
        lines.append(f"_pipe(_b, {names[0]}, _path0, sections=[{', '.join(names[1:])}]"
                     + (", minus=True)" if i else ")"))
    return "\n".join(lines)


def build_path_loft(near, far, pairs, path):
    """A section lofted along a 3-D path: `near` and `far` are (plane,
    sense, level) of the sections at its two ends, and `pairs` the rings
    [(near ring, far ring)], the outline first, then each hole. A hole's own
    multi-section sweep is cut from the outline's, so it stays a hole: in the
    tree, a pipe that removes from the block's own solid (mixed)."""
    wp_a, wp_b = workplane_expr(*near), workplane_expr(*far)
    lines = [_path_code(path)]
    if len(pairs) > 1:
        lines.append("_b.mixed = True")
    for i, (a, b) in enumerate(pairs):
        ra, note_a = drawable([a])
        rb, note_b = drawable([b])
        if not ra or not rb:
            raise EmitError(nothing_left([a if not ra else b], "sweep"))
        lines += (note_a + note_b).splitlines()
        lines += [f"_near{i} = _sketch(_b, {wp_a})", ring_code(ra[0], f"_near{i}"),
                  f"_far{i} = _sketch(_b, {wp_b})", ring_code(rb[0], f"_far{i}"),
                  f"_pipe(_b, _near{i}, _path0, sections=[_far{i}]"
                  + (", minus=True)" if i else ")")]
    return "\n".join(lines)


# Tangency copies of a twist's end faces, this far past each end: with them a
# chordal loft leaves the plates square (1e-4 .. 1e-2 all give the same
# solid).
TWIST_EPS = 1e-3


def _ring_points(ring):
    """The ring's vertices as written (each segment's start), as complex u + iv."""
    return np.array([complex(quantise(s["from_xy"][0]), quantise(s["from_xy"][1]))
                     for s in ring["segments"]])


def _similar(ring, A, B):
    """`ring` mapped by z -> A z + B (complex: turned by arg A, scaled by |A|):
    lines stay lines and arcs arcs, of radius x |A|, in the same sense."""
    def f(p):
        z = A * complex(p[0], p[1]) + B
        return (z.real, z.imag)
    segs = []
    for s in ring["segments"]:
        t = dict(s, from_xy=f(s["from_xy"]), to_xy=f(s["to_xy"]))
        if s["kind"] == "arc":
            t["centre"] = f(s["centre"])
            t["radius"] = float(s["radius"]) * abs(A)
        segs.append(t)
    return dict(ring, segments=segs)


def _twist_stations(near, far, z0, z1):
    """[(level, ring)] for the chordal loft of `near` into `far` (the same
    ring turned, vertex for vertex), with a tangency copy TWIST_EPS past each
    end.

    The intended loft goes through the four sections [copy, near, far, copy]
    with each vertex kept to its own turned copy. FreeCAD's loft always runs
    OCC's compatibility check, which pairs every vertex with the NEAREST one
    of the next section -- on a rectangle turned 60 deg that is the next
    corner, and the solid comes out wrong.
    The blend itself is known exactly: OCC interpolates each vertex through its
    four positions with one cubic, at chord-length parameters, so the section
    at parameter t is (c0 + c1) z + (c2 + c3) (A z + B) -- the ring turned and
    scaled (a similarity: arcs stay arcs), at level sum c_k z_k. Those sections,
    a few degrees of turn apart, are what is lofted here; consecutive ones are
    near enough that the check pairs each vertex with itself, and the solid is
    the intended one."""
    p, q = _ring_points(near), _ring_points(far)
    # the move near -> far: q = A p + B, |A| = 1 (least squares)
    pm, qm = p.mean(), q.mean()
    A = complex(np.vdot(p - pm, q - qm) / max(np.vdot(p - pm, p - pm).real, 1e-30))
    A /= abs(A) or 1.0
    B = qm - A * pm
    z0, z1 = quantise(z0), quantise(z1)         # the ends on the grid every feature uses
    s = 1.0 if z1 > z0 else -1.0
    zs = [z0 - s * TWIST_EPS, z0, z1, z1 + s * TWIST_EPS]
    secs = [p, p, A * p + B, A * p + B]

    def poles(z, c):
        # The concatenated degree-1 curve OCC lofts: the vertices, the first
        # one again at the end.
        pts = np.r_[c, c[:1]]
        return np.c_[pts.real, pts.imag, np.full(len(pts), z)]
    d = [0.0]
    for k in range(1, 4):
        d.append(d[-1] + float(np.sqrt(((poles(zs[k], secs[k]) - poles(zs[k - 1], secs[k - 1])) ** 2)
                                       .sum())))
    t = np.array(d) / d[-1]
    T = np.array([[math.comb(3, j) * (1 - x) ** (3 - j) * x ** j for j in range(4)] for x in t])
    # Bezier weights of the four sections at parameter x: c(x) = bern(x) @ inv(T)
    Tinv = np.linalg.inv(T)
    turn = abs(math.degrees(math.atan2(A.imag, A.real)))
    k = 1 + 16 * max(1, math.ceil(turn / 90.0))
    out = []
    for x in [0.0, *np.linspace(t[1], t[2], k), 1.0]:
        bern = np.array([math.comb(3, j) * (1 - x) ** (3 - j) * x ** j for j in range(4)])
        c = bern @ Tinv
        level = float(c @ np.asarray(zs))
        a_, b_ = c[0] + c[1], c[2] + c[3]
        out.append((level, _similar(near, a_ + b_ * A, b_ * B)))
    return out


def build_twist(plane, sense, sections, law):
    """A ring turning between two levels. `sections` is [(level, outer
    ring, [hole rings])], each ring the first one turned (exact vertex
    pairing). law "loft": the two end faces lofted chordally, with a tangency
    copy past each end and each vertex kept to its own turned copy -- the
    section narrows mid-way, as a loft between turned copies does; written as
    the sections of that blend (_twist_stations). law "rigid": a smooth loft through every section
    given. Holes are lofted the same way and cut (removing from the block's
    own solid: mixed)."""
    secs = [(lvl, [ring, *holes]) for lvl, ring, holes in sections]
    lines = ["_b.mixed = True"] if len(secs[0][1]) > 1 else []
    for i in range(len(secs[0][1])):
        for _lvl, rings in secs:
            r, note = drawable([rings[i]])
            if not r:
                raise EmitError(nothing_left([rings[i]], "loft"))
            lines.extend(note.splitlines())
        if law == "loft":
            stations = _twist_stations(secs[0][1][i], secs[-1][1][i], secs[0][0], secs[-1][0])
        else:
            stations = [(lvl, rings[i]) for lvl, rings in secs]
        names = []
        for j, (lvl, ring) in enumerate(stations):
            v = f"_t{i}_{j}"
            where = _level_expr(plane, lvl) if law == "loft" else plane_expr(plane, lvl)
            lines += [f"{v} = _sketch(_b, {where})", ring_code(ring, v)]
            names.append(v)
        lines.append(f"_loft(_b, [{', '.join(names)}], ruled=False"
                     + (", minus=True)" if i else ")"))
    return "\n".join(lines)


def _level_expr(plane, level):
    """`_plane_at` on `plane` at a level NOT on the 4-decimal grid: a twist's
    tangency copies are 1e-3 past its ends and its sections between them at
    levels of the blend, which the grid would move."""
    return f'_plane_at("{plane}", {float(level):.9f})'


def build_pipe_sweep(plane, sense, level, outer, holes, paths):
    """The profile `outer` with its `holes` -- rings on the sketch plane at
    `level` -- swept along each 3-D path in `paths`: pieces ("line", p0,
    p1) and ("arc", p0, mid, p1) in world coordinates, each path starting on
    the profile's plane. One path when the profile closes an end; two when it
    is a section across the middle, swept both ways from it. The holes travel
    inside the outline as one face, so a tube is hollow in one sweep: one
    sketch with the outline and its holes, one pipe per path (transition
    "Right corner")."""
    pl = plane_expr(plane, level)
    rings, note = drawable([outer])
    if not rings:
        raise EmitError(nothing_left([outer], "sweep"))
    hole_rings, hole_note = drawable(list(holes))
    lines = (note + hole_note).splitlines()
    for i, path in enumerate(paths):
        v = f"_s{i}"
        lines += [f"{v} = _sketch(_b, {pl})", ring_code(rings[0], v)]
        lines += [ring_code(h, v) for h in hole_rings]
        lines.append(_path_code(path, f"_path{i}"))
        lines.append(f"_pipe(_b, {v}, _path{i}, transition=\"Right corner\")")
    return "\n".join(lines)


def build_wire_sweep(points, radius):
    """A circle of `radius` swept along the 3-D curve through `points`,
    square to it at the first point (to the chord to the second)."""
    from mesh2cad.pipes import _straight
    p0, p1 = (np.asarray(p, float) for p in points[:2])
    t = (p1 - p0) / np.linalg.norm(p1 - p0)
    # A spline along a straight line has no normal for the sweep to keep: the
    # profile comes out too big. As in build_pipe_sweep's paths.
    q = _straight(("spline",) + tuple(tuple(p) for p in points))
    piece = ("line", q[1], q[2]) if q[0] == "line" else ("spline",) + tuple(tuple(p) for p in points)
    return "\n".join([
        _path_code([piece]),
        f"_s0 = _sketch(_b, _towards({_w3(p0)}, ({float(t[0]):.6f}, {float(t[1]):.6f}, "
        f"{float(t[2]):.6f})))",
        f"_circle(_s0, (0, 0), {_fmt(radius)})",
        "_pipe(_b, _s0, _path0, transition=\"Right corner\")"])
def build_loft(plane, sense, level_a, level_b, rings_a, rings_b,
               holes_a=(), holes_b=(), ruled=True):
    """Loft between the profile at one level and the profile at another.

    A tapered region is not a prism, so extruding its mid-section overfills,
    and no combination of extrude and revolve
    expresses it. Two measured sections and a loft do.

    `ruled=True` gives straight-line blending between corresponding edges, which
    is what a draft or a chamfered cap is, and it stays the default because a
    smooth loft invents curvature that was never measured.

    It is not always right, though: straight-line blending across a turning
    tooth, such as a helical gear's, costs IoU that no other operation can
    recover.
    The model asks for the smooth one by naming the curvature it measured, with
    --smooth, and the emitted program says which it used.

    Holes loft too, between their own pair of profiles, and are cut afterwards.
    Without that the operation would silently fill every bore it lofts across.
    A loft that cannot express a
    tapered hole is not an alternative to extrude, it is a trap next to it.
    """
    code, steps = build_loft_through(
        plane, sense, [level_a, level_b], [rings_a, rings_b],
        [list(holes_a), list(holes_b)], ruled=ruled)
    return code, steps[0]


def build_loft_through(plane, sense, levels, rings_at, holes_at=None,
                       ruled=True):
    """Loft through the profiles measured at THREE or more levels, in one call.

    Two levels can only say "it goes from this to that". A section that turns as
    it travels needs the intermediate shapes, and chaining two-level lofts is
    not the same solid: each pair gets its own blend and the joins are creases.

    Every level is quantised before any distance is taken from it, for the same
    reason extrude_distance is: a loft ending 1e-4 from the level the next
    feature starts on leaves the part in two pieces.
    """
    if len(levels) < 2:
        raise ValueError("a loft needs at least two levels")
    if len(rings_at) != len(levels):
        raise ValueError(f"{len(levels)} levels but {len(rings_at)} "
                         f"profile sets -- every level needs its own section")
    holes_at = holes_at or [[] for _ in levels]
    q = [quantise(x) for x in levels]
    steps = [(q[i + 1] - q[i]) * sense for i in range(len(q) - 1)]

    def _chain(per_level, stem, minus):
        # strict: a mismatched count would silently loft the shorter list and
        # drop profiles without saying so.
        out = []
        for i, group in enumerate(zip(*per_level, strict=True)):
            names = []
            for j, ring in enumerate(group):
                v = f"{stem}{i}_{j}"
                out += [f"{v} = _sketch(_b, {plane_expr(plane, levels[j])})",
                        ring_code(ring, v)]
                names.append(v)
            out.append(f"_loft(_b, [{', '.join(names)}], ruled={bool(ruled)}"
                       + (", minus=True)" if minus else ")"))
        return out

    lines = []
    if any(holes_at):
        # The holes loft too and are cut from the lofted outline -- from its
        # own solid, which a cut block cannot say with Subtractive lofts alone.
        lines.append("_b.mixed = True")
    lines += _chain(rings_at, "_s", False)
    if any(holes_at):
        lines += _chain(holes_at, "_h", True)
    return "\n".join(lines), steps
