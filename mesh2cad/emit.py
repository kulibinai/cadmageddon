"""Turn named section primitives into CadQuery source.

The model should not be writing coordinates by hand. It has just been shown a
section whose every edge carries a name, a radius and a centre; the natural next
instruction is "sketch A0_0 and A1_0 at Y=0 and extrude to the boundary", and
this module is what makes that instruction executable.

Emitting code rather than building geometry directly keeps the result auditable:
the program is the deliverable, and a model that can read what was emitted can
correct it.
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
    that ordering is the whole point. A feature used to be emitted as a start
    plane and a travel distance rounded INDEPENDENTLY, so the endpoint it
    reached was round(level) + round(target - level) -- which can miss
    round(target) by a full 1e-4.

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


def ring_code(ring, var):
    """Chained CadQuery calls that draw one closed ring.

    A ring that is a single full circle is emitted as `.circle(r)` rather than
    as a degenerate arc chain -- shorter, and it survives a reader's eye.
    """
    segs = ring["segments"]
    s = _circle_of(segs)
    if s is not None:
        return f"{var}.moveTo{_pt(s['centre'])}.circle({_fmt(s['radius'])})"

    # Every point as it will be WRITTEN. A segment shorter than the 4-decimal
    # grid ends on the point the pen is already at, and OCC refuses the edge
    # ("BRep_API: command not done" in makeLine): a `moveTo(P).lineTo(P)` leaves
    # a program that cannot be scored at all. Such a segment is dropped; an arc
    # whose middle lands on one of its ends is written as the line it is.
    start = _pt(segs[0]["from_xy"])
    parts, at = [f"{var}.moveTo{start}"], start
    path = [(None, segs[0]["from_xy"])]
    segs = _closing(segs, max(HEAL_TOL, float(ring.get("tol") or 0.0)))
    for k, s in enumerate(segs):
        to = _pt(s["to_xy"])
        if to == at:
            continue
        if s.get("_close"):
            # The last primitive ends within the fit tolerance of the start, so it
            # ends there: a last arc a few microns short would make `.close()` add a
            # micron-long edge that the kernel refuses in every `--to-profile`.
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
            parts.append(f".lineTo{to}")
            path.append((None, s["to_xy"]))
        else:
            parts.append(f".threePointArc({mid}, {to})")
            path.append((s, s["to_xy"]))
        at = to
    parts.append(".close()")
    crossing = _crossing(path)
    if crossing is not None:
        # Said, not refused: the kernel tolerates some crossings a polygon
        # test does not, and a program that builds is worth more than a refusal.
        # When it does not, this names the ring that broke it.
        print(f"note: ring {ring.get('name', '?')} crosses itself near "
              f"({crossing[0]:.4f}, {crossing[1]:.4f}) as written; if --check says "
              f"NOT A VALID SOLID, this ring is why -- measure it at another level "
              f"(--at mid(A,B) inside the feature) or leave it out of --use")
    return "".join(parts)


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
    recorded. Going through --split loses the 360: the equatorial section of a
    torus is one closed circle that a cut through the axis never touches, and
    it comes back out with sweep=0 -- a `threePointArc` with all three points on
    top of each other, which OCC answers `GC_MakeArcOfCircle::Value() - no
    result`.
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

    Said so it can be acted on: a ring sketched at the level where a window
    starts, where the section only grazes the material, is thinner than the
    grid, and the model must not read that as its ring name not being parsed.
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
    stops the whole program on it ("No start point specified - cannot close") --
    as when a --split cuts a revolve profile into pieces, one of them two points
    a grid step apart.
    """
    kept = [r for r in rings if written_area(r) > 10.0 ** (-2 * EMIT_DECIMALS)]
    gone = len(rings) - len(kept)
    note = (f"# note: {gone} ring(s) thinner than the 1e-4 grid the program is "
            f"written on left out -- they enclose no area and OCC cannot close them\n"
            if gone else "")
    return kept, note


def half_arc_code(h, var):
    """A half circle closed by its own diameter -- a revolve profile."""
    mid = _pt(_half_mid(h))
    return (f"{var}.moveTo{_pt(h['from'])}"
            f".threePointArc({mid}, {_pt(h['to'])})"
            f".close()")


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
        if np.cross(p1 - p0, d) < 0:
            d = -d
        return c + d * float(h["radius"])
    return c + d / n * float(h["radius"])


def workplane_expr(plane, sense, level):
    """A CadQuery workplane sitting exactly on `level` in world coordinates.

    CadQuery's named planes start at the origin, and `workplane(offset=)` moves
    along the plane normal -- which for XZ points at -Y. Folding that sign in
    here is the difference between a sketch on the face and a sketch mirrored to
    the far side of the part.
    """
    off = quantise(level) * sense
    if abs(off) < 1e-9:
        return f'cq.Workplane("{plane}")'
    return f'cq.Workplane("{plane}").workplane(offset={_fmt(off)})'


def extrude_distance(level, target, sense):
    """Signed distance for .extrude() to travel from `level` to `target`.

    From the QUANTISED endpoints, so the solid stops exactly on the value a
    neighbouring feature will emit for the same level. See `quantise`.
    """
    return (quantise(target) - quantise(level)) * sense


def build_extrude(plane, sense, level, target, outer_rings, hole_rings,
                  both=False, taper=None):
    """`both` extrudes the same distance either side of the sketch plane.

    A feature centred on the plane it was measured on -- a spoke cut at its own
    mid-thickness -- otherwise has to be built as two halves and unioned, which
    puts a coincident face straight down the middle of it for no reason.
    """
    wp = workplane_expr(plane, sense, level)
    dist = extrude_distance(level, target, sense)
    asked = outer_rings
    outer_rings, note = drawable(outer_rings)
    hole_rings, hole_note = drawable(hole_rings)
    if not outer_rings:
        raise EmitError(nothing_left(asked, "extrude"))
    lines, names = (note + hole_note).splitlines(), []
    draft = f", taper={_fmt(taper)}" if taper else ""
    for i, ring in enumerate(outer_rings):
        v = f"_solid{i}" if len(outer_rings) > 1 else "_solid"
        lines.append(f"{v} = ({ring_code(ring, wp)}\n         "
                     f".extrude({_fmt(dist)}{', both=True' if both else ''}{draft}))")
        names.append(v)
    body = names[0] if len(names) == 1 else ".union(".join(names) + ")" * (len(names) - 1)
    if len(names) > 1:
        body = names[0] + "".join(f".union({n})" for n in names[1:])

    cuts = []
    for j, ring in enumerate(hole_rings):
        v = f"_hole{j}"
        lines.append(f"{v} = ({ring_code(ring, wp)}\n         "
                     f".extrude({_fmt(dist)}{', both=True' if both else ''}))")
        body += f".cut({v})"
        cuts.append(v)
    lines.append(f"result = {body}")
    # The feature operands, in order, kept alongside the fused result. OCC can
    # union these into a solid it then cannot mesh; meshing each one separately
    # and combining the MESHES always closes. Recording them costs one line and
    # is the only way a later step can still get at the pieces.
    lines.append(_steps_line([("add", n) for n in names]
                             + [("cut", c) for c in cuts]))
    return "\n".join(lines), dist


def revolve_call(angle, ax0, ax1):
    """`.revolve(...)` for a signed angle, the sense the right hand about
    ax0 -> ax1. OCC builds `.revolve(-90, a, b)` as 270 deg -- three quadrants
    where one was asked -- so a negative angle turns |angle| about the reversed
    axis: the same line, the other way."""
    if angle < 0:
        return f".revolve({_fmt(-angle)}, {_pt(ax1)}, {_pt(ax0)})"
    return f".revolve({_fmt(angle)}, {_pt(ax0)}, {_pt(ax1)})"


def build_revolve_rings(plane, sense, level, rings, axis, angle=360.0):
    """Revolve a closed profile that was cut from the section at `axis`.

    This is the general case the half-arc path could not express. A meridian is
    usually a mix of lines and arcs -- every frustum, cup, cone and stepped
    shaft -- and revolving it needs the whole ring, not one primitive. Cutting
    the section at the axis and closing it there turns any section into a valid
    revolve profile, so the operation stops being sphere-only.
    """
    wp = workplane_expr(plane, sense, level)
    (ax0, ax1) = axis
    asked = rings
    rings, note = drawable(rings)
    if not rings:
        raise EmitError(nothing_left(asked, "revolve"))
    lines, names = note.splitlines(), []
    for i, ring in enumerate(rings):
        v = f"_prof{i}" if len(rings) > 1 else "_prof"
        lines.append(f"{v} = ({ring_code(ring, wp)}\n"
                     f"         {revolve_call(angle, ax0, ax1)})")
        names.append(v)
    body = names[0] + "".join(f".union({n})" for n in names[1:])
    lines.append(f"result = {body}")
    lines.append(_steps_line([("add", n) for n in names]))
    return "\n".join(lines), angle


def build_revolve_profile(plane, sense, level, outer_rings, hole_rings, axis, angle=360.0):
    """Revolve closed rings -- outlines and the holes in them -- about an axis in
    the sketch plane that crosses none of them.

    Each outline is revolved and unioned, each hole revolved and cut, so an
    annulus about an axis off the profile is a hollow pipe bend. Revolving rings
    through `--split` would turn the hole into a second solid and build pipe
    bends with the bore filled.
    """
    wp = workplane_expr(plane, sense, level)
    (ax0, ax1) = axis
    asked = outer_rings
    outer_rings, note = drawable(outer_rings)
    hole_rings, hole_note = drawable(hole_rings)
    if not outer_rings:
        raise EmitError(nothing_left(asked, "revolve"))
    lines, names, cuts = (note + hole_note).splitlines(), [], []
    turn = revolve_call(angle, ax0, ax1)
    for i, ring in enumerate(outer_rings):
        v = f"_prof{i}" if len(outer_rings) > 1 else "_prof"
        lines.append(f"{v} = ({ring_code(ring, wp)}\n         {turn})")
        names.append(v)
    body = names[0] + "".join(f".union({n})" for n in names[1:])
    for j, ring in enumerate(hole_rings):
        v = f"_hole{j}"
        lines.append(f"{v} = ({ring_code(ring, wp)}\n         {turn})")
        body += f".cut({v})"
        cuts.append(v)
    lines.append(f"result = {body}")
    lines.append(_steps_line([("add", n) for n in names] + [("cut", c) for c in cuts]))
    return "\n".join(lines), angle


def build_revolve(plane, sense, level, half, angle=360.0):
    wp = workplane_expr(plane, sense, level)
    (ax0, ax1) = half["revolve_axis"]
    prof = half_arc_code(half, wp)
    code = (f"_prof = ({prof}\n"
            f"         {revolve_call(angle, ax0, ax1)})\n"
            f"result = _prof\n"
            + _steps_line([("add", "_prof")]))
    return code, angle


def build_helical_sweep(pitch, height, base, coil_r, wire_r, handed="right",
                        phase=0.0):
    """Sweep a measured circular profile along a fitted helix.

    The regime classifier recognises a helix -- axis, pitch, turns, handedness
    -- and the answer needs somewhere to go: `--op` offered extrude, revolve and
    loft, and every program reaches disk through `--op`, so without this a
    spring would be detectable and unbuildable in the same breath.

    Every number here is measured, and each by whichever instrument measures it
    best. `wire_r` and `coil_r` come from the section -- it fits the wire's own
    arc to a few microns, where the helix detector's slice centroids put the
    coil radius well out. `pitch` and `handedness` come from the helix fit, which
    is the only thing that sees them. `height` and `base` come from the mesh's
    extent along the axis, because turns-times-pitch does not: the turn count is
    fitted on single-loop slices, which deliberately skip the ground end turns,
    so multiplying the two overshoots the spring's length.

    The profile is placed at the path's start and tilted by the lead angle,
    atan(pitch / 2*pi*r) -- a swept section stands perpendicular to its path,
    and drawing it flat leaves the coil elliptical.

    `phase` is where the coil starts. makeHelix always begins at angle zero, and
    a helix turned about its own axis is a different solid: with the height and
    both radii correct but the phase left at zero, the coil sits in the right
    envelope and almost nowhere near itself. It is measured from the ring the
    model named, which sits at a known height and a known angle and so anchors
    the turn it belongs to.
    """
    lead = math.degrees(math.atan2(pitch, 2 * math.pi * coil_r))
    # makeHelix has no handedness flag; a left-handed coil is the right-handed
    # one built on a negative pitch.
    signed = quantise(pitch if handed == "right" else -pitch)
    code = (f"_path = cq.Wire.makeHelix({_fmt(signed)}, {_fmt(quantise(height))}, "
            f"{_fmt(quantise(coil_r))}, cq.Vector(0, 0, {_fmt(quantise(base))}))\n"
            f"_solid = (cq.Workplane('XY')\n"
            f"          .transformed(offset=cq.Vector({_fmt(quantise(coil_r))}, 0, "
            f"{_fmt(quantise(base))}), rotate=cq.Vector({_fmt(quantise(lead - 90))}, 0, 0))\n"
            f"          .circle({_fmt(quantise(wire_r))})\n"
            f"          .sweep(cq.Workplane(obj=_path), isFrenet=True))\n")
    if abs(quantise(phase)) > 1e-4:
        code += (f"_solid = _solid.rotate((0, 0, 0), (0, 0, 1), "
                 f"{_fmt(quantise(phase))})\n")
    code += ("result = _solid\n" + _steps_line([("add", "_solid")]))
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
                        r_max):
    """Any closed profile carried along a helix through the band [zA, zB],
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

    Not OCC's sweep: a sweep of a thread tooth does not mesh watertight and
    comes out too big, and one-turn sweeps patterned along the axis do not fuse
    -- the result is an invalid compound of N x one turn. The ruled solid is
    valid, meshes closed, and unions with the core.
    """
    za, zb = band
    # The pitch as it is written: the helices are a whole number of turns of
    # THIS pitch, or the end cap, turned by whole turns, misses their ends and
    # the shell does not close -- rounding the pitch and the height
    # independently gives 3.00001 turns, and SolidFromShell fails.
    p = float(_fmt(abs(float(pitch))))
    pts = []
    for op, ps in profile:
        if op == "arc":
            pts.extend(_chords(pts[-1], ps[0], ps[1]))
        else:
            pts.append(ps[0])
    # The last point is the ring's closing one, and a ring fitted from a cut
    # closes onto its start within the fit, not to float: that point, kept,
    # makes a sliver face that builds the coil inside out.
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
    code = (f"_pts = [{rows}]\n"
            f"_hel = [cq.Wire.makeHelix({_fmt(p)}, {_fmt(h)}, _r, cq.Vector(0, 0, _z), "
            f"lefthand={bool(lefthand)}) for _r, _z in _pts]\n"
            f"_faces = [cq.Face.makeRuledSurface(_hel[_i].Edges()[0], "
            f"_hel[(_i + 1) % len(_pts)].Edges()[0]) for _i in range(len(_pts))]\n"
            f"_cap = cq.Face.makeFromWires(cq.Wire.makePolygon("
            f"[cq.Vector(_r, 0, _z) for _r, _z in _pts], close=True))\n"
            f"_cap1 = _cap.rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), {_fmt(turn_end)})"
            f".translate(cq.Vector(0, 0, {_fmt(h)}))\n"
            f"_coil = cq.Workplane(obj=cq.Solid.makeSolid(cq.Shell.makeShell(_faces + [_cap, _cap1])))\n"
            f"_band = cq.Workplane('XY').workplane(offset={_fmt(za)}).circle({_fmt(r_max + 1.0)})"
            f".extrude({_fmt(zb - za)})\n"
            # The fuzzy common of a valid coil and its band can come back
            # empty where the exact one is the coil; the fuzzy one is the
            # fallback, and a coil empty either way is said by name.
            f"_trim = _coil.intersect(_band)\n"
            f"if not _trim.solids().vals():\n"
            f"    _trim = _coil.intersect(_band, tol=0.01)\n"
            f"if not _trim.solids().vals():\n"
            f"    raise ValueError('the swept coil is void inside its band z={_fmt(za)}..{_fmt(zb)} "
            f"(local): the turns do not meet the band, or they overlap each other')\n"
            f"_coil = _trim\n")
    if abs(theta) > 1e-6:
        code += f"_coil = _coil.rotate((0, 0, 0), (0, 0, 1), {_fmt(theta)})\n"
    # The local frame turned onto the world's: a third of a turn about (1,1,1)
    # cycles the axes (HELIX_FRAMES), then the axis is moved where the helix is.
    turn = {"Z": 0, "Y": -120, "X": 120}[axis]
    if turn:
        code += f"_coil = _coil.rotate((0, 0, 0), (1, 1, 1), {turn})\n"
    code += (f"_solid = _coil.translate(({_fmt(centre[0])}, {_fmt(centre[1])}, "
             f"{_fmt(centre[2])}))\n"
             "result = _solid\n" + _steps_line([("add", "_solid")]))
    return code


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
        v[spec["u"]], v[spec["v"]], v[spec["axis"]] = p[0], p[1], level
        return f"V({_fmt(v[0])}, {_fmt(v[1])}, {_fmt(v[2])})"
    edges = []
    for p in pieces:
        if p[0] == "line":
            edges.append(f"cq.Edge.makeLine({w(p[1])}, {w(p[2])})")
        else:
            edges.append(f"cq.Edge.makeThreePointArc({w(p[1])}, {w(p[2])}, {w(p[3])})")
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
    t3 = [0.0, 0.0, 0.0]
    t3[spec["u"]], t3[spec["v"]] = float(t[0]), float(t[1])
    joined = ",\n          ".join(edges)
    code = ("V = cq.Vector\n"
            f"_path = cq.Wire.assembleEdges([{joined}])\n"
            f"_circ = cq.Wire.makeCircle({_fmt(radius)}, {w(p0)}, "
            f"V({_fmt(t3[0])}, {_fmt(t3[1])}, {_fmt(t3[2])}))\n"
            # 'transformed': the snapped pieces meet within a degree of their
            # tangents, and there 'round' can build the wire as an invalid solid.
            "_solid = cq.Workplane(obj=_circ).toPending().sweep(cq.Workplane(obj=_path), "
            "transition='transformed')\n")
    if capped:
        last = pieces[-1][-1]
        for q in (pieces[0][1], last):
            code += (f"_solid = _solid.union(cq.Workplane(obj=cq.Solid.makeSphere("
                     f"{_fmt(radius)}, {w(q)}, angleDegrees1=-90, angleDegrees2=90)))\n")
    code += "result = _solid\n" + _steps_line([("add", "_solid")])
    return code


def _path_edges(path):
    """CadQuery edges of 3-D path pieces, ("line", p0, p1), ("arc", p0, mid,
    p1) or ("spline", ...), in world coordinates."""
    def w(p):
        return f"V({_fmt(p[0])}, {_fmt(p[1])}, {_fmt(p[2])})"
    return ",\n    ".join(
        f"cq.Edge.makeLine({w(q[1])}, {w(q[2])})" if q[0] == "line" else
        f"cq.Edge.makeSpline([{', '.join(w(p) for p in q[1:])}])" if q[0] == "spline" else
        f"cq.Edge.makeThreePointArc({w(q[1])}, {w(q[2])}, {w(q[3])})" for q in path)


def build_station_sweep(path, sections):
    """A section swept along a 3-D path through K sections: `sections`
    is one list per station, [outline points, hole points, ...], each ring as
    3-D points matched across the stations (same count, start and direction).
    The outline's multi-section sweep, each hole's cut from it."""
    def w(p):
        return f"V({_fmt(p[0])}, {_fmt(p[1])}, {_fmt(p[2])})"
    lines, names = ["V = cq.Vector",
                    f"_path0 = cq.Wire.assembleEdges([\n    {_path_edges(path)}])"], []
    for i in range(len(sections[0])):
        ws = []
        for j, sec in enumerate(sections):
            lines.append(f"_w{i}_{j} = cq.Wire.makePolygon([{', '.join(w(p) for p in sec[i])}], "
                         f"close=True)")
            ws.append(f"_w{i}_{j}")
        v = "_solid" if i == 0 else f"_void{i - 1}"
        lines.append(f"{v} = cq.Workplane(obj=cq.Solid.sweep_multi([{', '.join(ws)}], _path0))")
        names.append(v)
    lines.append("result = " + names[0] + "".join(f".cut({n})" for n in names[1:]))
    lines.append(_steps_line([("add", names[0])] + [("cut", n) for n in names[1:]]))
    return "\n".join(lines)


def build_path_loft(near, far, pairs, path):
    """A section lofted along a 3-D path: `near` and `far` are (plane,
    sense, level) of the sections at its two ends, and `pairs` the rings
    [(near ring, far ring)], the outline first, then each hole. A hole's own
    multi-section sweep is cut from the outline's, so it stays a hole."""
    wp_a, wp_b = workplane_expr(*near), workplane_expr(*far)
    lines, names = ["V = cq.Vector",
                    f"_path0 = cq.Wire.assembleEdges([\n    {_path_edges(path)}])"], []
    for i, (a, b) in enumerate(pairs):
        ra, note_a = drawable([a])
        rb, note_b = drawable([b])
        if not ra or not rb:
            raise EmitError(nothing_left([a if not ra else b], "sweep"))
        lines += (note_a + note_b).splitlines()
        lines.append(f"_near{i} = ({ring_code(ra[0], wp_a)}).val()")
        lines.append(f"_far{i} = ({ring_code(rb[0], wp_b)}).val()")
        names.append(f"cq.Workplane(obj=cq.Solid.sweep_multi([_near{i}, _far{i}], _path0))")
    lines.append(f"_solid = {names[0]}" + "".join(f".cut({n})" for n in names[1:]))
    lines.append("result = _solid")
    lines.append(_steps_line([("add", "_solid")]))
    return "\n".join(lines)


# Tangency copies of a twist's end faces, this far past each end: with
# them a chordal loft leaves the plates square (the result is insensitive
# to the exact value over two orders of magnitude).
TWIST_EPS = 1e-3


def build_twist(plane, sense, sections, law):
    """A ring turning between two levels. `sections` is [(level, outer
    ring, [hole rings])], each ring the first one turned (exact vertex
    pairing). law "loft": the two end faces lofted chordally, with a tangency
    copy past each end and `compat=False` -- the section narrows mid-way, as a
    loft between turned copies does, which a plain two-level loft misses.
    law "rigid": a smooth loft through every section given. Holes are lofted
    the same way and cut."""
    from mesh2cad.section import PLANES
    k = PLANES[plane]["axis"]
    d = [0.0, 0.0, 0.0]
    d[k] = TWIST_EPS * (1.0 if sections[-1][0] > sections[0][0] else -1.0)
    lines = ["from cadquery import func as _F", f"_d = cq.Vector({d[0]}, {d[1]}, {d[2]})"]

    secs = [(lvl, [ring, *holes]) for lvl, ring, holes in sections]

    def wires(i):
        out = []
        for lvl, rings in secs:
            r, note = drawable([rings[i]])
            if not r:
                raise EmitError(nothing_left([rings[i]], "loft"))
            lines.extend(note.splitlines())
            out.append(f"({ring_code(r[0], workplane_expr(plane, sense, lvl))}).val()")
        return out
    names = []
    for i in range(len(secs[0][1])):
        w = wires(i)
        v = "_solid" if i == 0 else f"_void{i - 1}"
        if law == "loft":
            lines.append(f"{v}a = cq.Face.makeFromWires({w[0]})")
            lines.append(f"{v}b = cq.Face.makeFromWires({w[-1]})")
            lines.append(f"{v} = cq.Workplane(obj=_F.loft({v}a.translate(_d * -1), {v}a, {v}b, "
                         f"{v}b.translate(_d), parametrization='chordal', compat=False))")
        else:
            lines.append(f"{v} = cq.Workplane(obj=cq.Solid.makeLoft([\n    "
                         + ",\n    ".join(w) + "], ruled=False))")
        names.append(v)
    lines.append("result = " + names[0] + "".join(f".cut({n})" for n in names[1:]))
    lines.append(_steps_line([("add", names[0])] + [("cut", n) for n in names[1:]]))
    return "\n".join(lines)


def build_pipe_sweep(plane, sense, level, outer, holes, paths):
    """The profile `outer` with its `holes` -- rings on the sketch plane at
    `level` -- swept along each 3-D path in `paths`: pieces ("line", p0,
    p1) and ("arc", p0, mid, p1) in world coordinates, each path starting on
    the profile's plane. One path when the profile closes an end; two when it
    is a section across the middle, swept both ways from it. The holes travel
    inside the outline as one face, so a tube is hollow in one sweep."""
    wp = workplane_expr(plane, sense, level)
    rings, note = drawable([outer])
    if not rings:
        raise EmitError(nothing_left([outer], "sweep"))
    hole_rings, hole_note = drawable(list(holes))
    lines = (note + hole_note).splitlines() + ["V = cq.Vector"]
    prof = ring_code(rings[0], wp)
    for h in hole_rings:
        prof = ring_code(h, prof)
    names = []
    for i, path in enumerate(paths):
        lines.append(f"_path{i} = cq.Wire.assembleEdges([\n    {_path_edges(path)}])")
        lines.append(f"_sweep{i} = ({prof}\n         .sweep(cq.Workplane(obj=_path{i}), "
                     f"transition='right'))")
        names.append(f"_sweep{i}")
    lines.append("_solid = " + names[0] + "".join(f".union({n})" for n in names[1:]))
    lines.append("result = _solid")
    lines.append(_steps_line([("add", "_solid")]))
    return "\n".join(lines)


def build_wire_sweep(points, radius):
    """A circle of `radius` swept along the 3-D curve through `points`,
    square to it at the first point."""
    from mesh2cad.pipes import _straight
    p0, p1 = (np.asarray(p, float) for p in points[:2])
    t = (p1 - p0) / np.linalg.norm(p1 - p0)

    def w(p):
        return f"V({_fmt(p[0])}, {_fmt(p[1])}, {_fmt(p[2])})"
    # A spline along a straight line has no normal for the sweep to keep: the
    # profile came out 30% too big. As in build_pipe_sweep's paths.
    q = _straight(("spline",) + tuple(tuple(p) for p in points))
    edge = (f"cq.Edge.makeLine({w(q[1])}, {w(q[2])})" if q[0] == "line" else
            f"cq.Edge.makeSpline([{', '.join(w(p) for p in points)}])")
    return "\n".join([
        "V = cq.Vector",
        f"_path0 = cq.Wire.assembleEdges([{edge}])",
        f"_solid = (cq.Workplane(cq.Plane(origin=({_fmt(p0[0])}, {_fmt(p0[1])}, {_fmt(p0[2])}), "
        f"normal=({float(t[0]):.6f}, {float(t[1]):.6f}, {float(t[2]):.6f})))"
        f".circle({_fmt(radius)})\n         .sweep(cq.Workplane(obj=_path0), transition='right'))",
        "result = _solid",
        _steps_line([("add", "_solid")])])


def _steps_line(steps):
    """`_STEPS = [("add", _solid), ("cut", _hole0)]` -- the operands, in order."""
    body = ", ".join(f'("{op}", {var})' for op, var in steps)
    return f"_STEPS = [{body}]"


def build_loft(plane, sense, level_a, level_b, rings_a, rings_b,
               holes_a=(), holes_b=(), ruled=True):
    """Loft between the profile at one level and the profile at another.

    The operation the harness kept needing and did not have. A tapered region is
    not a prism, so extruding its mid-section overfills it, and no combination of
    extrude and revolve expresses it. Two measured sections and a loft do.

    `ruled=True` gives straight-line blending between corresponding edges, which
    is what a draft or a chamfered cap is, and it stays the default because a
    smooth loft invents curvature that was never measured.

    It is not always right, though. A helical gear's tooth is a smooth loft:
    straight-line blending across a turning tooth costs IoU that no other
    operation can recover. The model asks for the smooth one by naming the
    curvature it measured, with --smooth, and the emitted program says which it
    used.

    Holes loft too, between their own pair of profiles, and are cut afterwards.
    Without that the operation would silently fill every bore it lofted across,
    taking the outer ring and dropping the inner one on the floor. A loft that
    cannot express a tapered hole is not an alternative to extrude, it is a trap
    next to it.
    """
    code, steps = build_loft_through(
        plane, sense, [level_a, level_b], [rings_a, rings_b],
        [list(holes_a), list(holes_b)], ruled=ruled)
    return code, steps[0]


# A loft of dense rings is one long chain of CadQuery calls, and each call walks
# back up the chain: past ~1000 primitives Python's default depth is exceeded
# (RecursionError on a knurled ring, say). The program raises its own limit,
# so it builds wherever it is run.
LOFT_DEPTH = "import sys\nsys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))\n"


def build_loft_through(plane, sense, levels, rings_at, holes_at=None,
                       ruled=True):
    """Loft through the profiles measured at THREE or more levels, in one call.

    Two levels can only say "it goes from this to that". A section that turns as
    it travels needs the intermediate shapes, and chaining two-level lofts is
    not the same solid: each pair gets its own blend and the joins are creases.

    Measured on the helical gear, whose reference lofts three profiles in one
    call: dropping the middle one costs 9.6 IoU, four times what the blend mode
    costs. That was the larger half of a gap `--smooth` only half closed.

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
    wp = workplane_expr(plane, sense, levels[0])
    q = [quantise(x) for x in levels]
    steps = [(q[i + 1] - q[i]) * sense for i in range(len(q) - 1)]

    def _chain(per_level, stem):
        # strict: a mismatched count would silently loft the shorter list and
        # drop profiles without saying so.
        out, got = [], []
        first = per_level[0]
        for i, group in enumerate(zip(*per_level, strict=True)):
            v = f"{stem}{i}" if len(first) > 1 else stem
            body = [f"{v} = ({ring_code(group[0], wp)}"]
            for ring, step in zip(group[1:], steps, strict=True):
                body.append(f"         .workplane(offset={_fmt(step)})")
                body.append(f"         {ring_code(ring, '')}")
            body.append(f"         .loft(ruled={bool(ruled)}))")
            # A loft through many levels can come out inside out; turned
            # outward here, where it is made, rather than refused.
            body.append(f"if {v}.val().Volume() < 0:\n"
                        f"    {v} = cq.Workplane(obj=cq.Shape.cast({v}.val().wrapped.Reversed()))")
            out.append("\n".join(body))
            got.append(v)
        return out, got

    lines, names = _chain(rings_at, "_loft")
    if any(holes_at):
        cut_lines, cuts = _chain(holes_at, "_lofthole")
        lines.extend(cut_lines)
    else:
        cuts = []

    body = names[0] + "".join(f".union({n})" for n in names[1:])
    body += "".join(f".cut({c})" for c in cuts)
    lines.append(f"result = {body}")
    lines.append(_steps_line([("add", n) for n in names]
                             + [("cut", c) for c in cuts]))
    return LOFT_DEPTH + "\n".join(lines), steps
