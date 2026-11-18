"""Profile families and their laws across levels.

A section is traced into lines and arcs, and a designed profile comes back as
whatever the tessellation made of it: a drafted tray's rounded corners are
traced as several arcs and lines per corner, different at every level.
The designer drew a rounded rectangle and moved it: half-size shrinking with
the draft, corner radius growing with depth. Fitted as that family
at several levels, its four numbers per level line up on straight lines, and a
ruled loft between the family at the two named levels is the wall.

Families fitted: circle, slot and axis-aligned rounded rectangle; a corner
radius is solved from the section's area, so it needs no arc to have been
traced cleanly. Tangent-arc chains are not fitted.
"""

import math

import numpy as np

LAW_LEVELS = 7          # levels sampled between the two named ones
LAW_INSET = 0.06        # share of the span kept clear at each end (the bends)
PARAMS = ("cx", "cy", "hw", "hh", "r")
NAMES = {"cx": "centre u", "cy": "centre v", "hw": "half-width", "hh": "half-height",
         "r": "corner R"}


class LawError(ValueError):
    """A profile that is no family, or a law that does not hold; says why."""


def shape(f):
    """The fitted family as a shapely polygon."""
    from shapely.geometry import Point, box
    cx, cy, hw, hh, r = (f[k] for k in PARAMS)
    if f["family"] == "circle":
        return Point(cx, cy).buffer(r, quad_segs=64)
    if r <= 1e-9:
        return box(cx - hw, cy - hh, cx + hw, cy + hh)
    core = box(cx - hw + r, cy - hh + r, cx + hw - r, cy + hh - r)
    return core.buffer(r, quad_segs=64)


def fit(poly, tol):
    """The family `poly` is, with its parameters and how far it is from it,
    or a LawError saying it is none of them."""
    minx, miny, maxx, maxy = poly.bounds
    cx, cy, hw, hh = (minx + maxx) / 2, (miny + maxy) / 2, (maxx - minx) / 2, (maxy - miny) / 2
    # The corner radius from the area: A = 4 hw hh - (4 - pi) r^2 -- immune to
    # a corner traced in pieces.
    r = float(np.sqrt(max(0.0, 4 * hw * hh - poly.area) / (4 - np.pi)))
    r = min(r, hw, hh)
    if abs(hw - hh) <= tol and abs(r - hw) <= tol:
        family, r = "circle", (hw + hh) / 2
        hw = hh = r
    elif abs(r - min(hw, hh)) <= tol:
        family, r = "slot", min(hw, hh)
    elif r <= tol:
        family, r = "rect", 0.0
    else:
        family = "rounded rect"
    f = {"family": family, "cx": cx, "cy": cy, "hw": hw, "hh": hh, "r": r}
    f["off"] = float(shape(f).boundary.hausdorff_distance(poly.boundary))
    if f["off"] > tol:
        raise LawError(f"not a circle, slot or rounded rectangle: the nearest "
                       f"({family}, r {r:.4f}) is {f['off']:.4f} mm off it")
    return f


LAW_MIN = 4             # levels a law must hold on
LAW_DROP = 2            # end levels it may leave out, from each end


def _lines(fits, zs):
    z = np.asarray(zs, float)
    out = {}
    for p in PARAMS:
        v = np.asarray([f[p] for f in fits], float)
        k, c = np.polyfit(z, v, 1)
        out[p] = (float(k), float(c), float(np.max(np.abs(v - (k * z + c)))))
    return out


def law(fits, zs, tol=None):
    """({param: (slope, intercept, worst miss)}, [left-out levels]): a straight
    line through each parameter over the levels.

    With `tol`, every line must hold to it. An end level off the line is a
    transition -- a level sampled just under a flange lies in the bend -- and
    is left out, at most
    LAW_DROP from each end, while LAW_MIN levels remain; the caller says so.
    A law that holds on no such run is a LawError naming the worst number."""
    fams = {f["family"] if f["family"] != "rect" else "rounded rect" for f in fits}
    if len(fams) > 1:
        raise LawError(f"the profile changes family between the levels ({', '.join(sorted(fams))})")
    out = _lines(fits, zs)
    if tol is None or max(m for *_, m in out.values()) <= tol:
        return out, []
    n = len(fits)
    best = None
    for a in range(LAW_DROP + 1):
        for b in range(LAW_DROP + 1):
            if n - a - b < LAW_MIN or (a, b) == (0, 0):
                continue
            got = _lines(fits[a:n - b], zs[a:n - b])
            if max(m for *_, m in got.values()) <= tol and (best is None or a + b < best[0]):
                best = (a + b, got, list(zs[:a]) + list(zs[n - b:]))
    if best is None:
        p, (_, _, miss) = max(out.items(), key=lambda kv: kv[1][2])
        raise LawError(f"its {NAMES[p]} is not a straight line between the levels "
                       f"(worst {miss:.4f} off, tolerance {tol:.4f}), even with "
                       f"{LAW_DROP} end levels left out")
    return best[1], best[2]


def at(fits, laws, level):
    """The family evaluated at `level` from its laws."""
    f = {"family": fits[0]["family"]}
    for p in PARAMS:
        k, c, _ = laws[p]
        f[p] = k * level + c
    if f["family"] == "slot":
        f["r"] = min(f["hw"], f["hh"])
    return f





def outline(f):
    """The family's closed outline as prelude ring segments, CCW, starting at
    the same place for every member of a family -- so a ruled loft pairs the
    same edges at every level. ("L", p0, p1) and ("A", p0, p1, centre, True);
    None for a circle.

    slot: two straight sides and two half circles, overall length
    2*max(hw, hh) along the longer side. rect: four sides, rounded by r at
    every corner when r > 0 (what fillet2D on every vertex made)."""
    cx, cy, hw, hh, r = (float(f[k]) for k in PARAMS)
    if f["family"] == "circle":
        return None
    if f["family"] == "slot":
        r = min(hw, hh)
    elif r <= 1e-9 or f["family"] == "rect":
        r = 0.0
    x0, x1, y0, y1 = cx - hw, cx + hw, cy - hh, cy + hh
    if r <= 1e-9:
        pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        return [("L", a, b) for a, b in zip(pts, pts[1:] + pts[:1], strict=True)]
    # (centre, start, end) of each corner arc, CCW from the bottom right.
    corners = [((x1 - r, y0 + r), (x1 - r, y0), (x1, y0 + r)),
               ((x1 - r, y1 - r), (x1, y1 - r), (x1 - r, y1)),
               ((x0 + r, y1 - r), (x0 + r, y1), (x0, y1 - r)),
               ((x0 + r, y0 + r), (x0, y0 + r), (x0 + r, y0))]
    segs = []
    for i, (c, a, b) in enumerate(corners):
        prev_end = corners[i - 1][2]
        if math.dist(prev_end, a) > 1e-9:            # a side of zero length is not drawn
            segs.append(("L", prev_end, a))
        segs.append(("A", a, b, c, True))
    return segs


def wire_code(f, pl, var):
    """The prelude calls that sketch the family's closed outline in `var`, a
    new sketch at placement expression `pl` (already at the level)."""
    from mesh2cad.emit import _fmt
    cx, cy, r = f["cx"], f["cy"], f["r"]
    out = f"{var} = _sketch(_b, {pl})\n"
    segs = outline(f)
    if segs is None:
        return out + f"_circle({var}, ({_fmt(cx)}, {_fmt(cy)}), {_fmt(r)})"

    def pt(p):
        return f"({_fmt(p[0])}, {_fmt(p[1])})"
    body = ", ".join(f'("L", {pt(s[1])}, {pt(s[2])})' if s[0] == "L"
                     else f'("A", {pt(s[1])}, {pt(s[2])}, {pt(s[3])}, True)' for s in segs)
    return out + f"_ring({var}, [{body}])"


def loft_code(plane, sense, levels_fits):
    """A ruled loft through the family at each (level, fit)."""
    from mesh2cad.emit import plane_expr
    lines, names = [], []
    for i, (level, f) in enumerate(levels_fits):
        v = f"_lw{i}"
        lines.append(wire_code(f, plane_expr(plane, level), v))
        names.append(v)
    lines.append(f"_loft(_b, [{', '.join(names)}], ruled=True)")
    return "\n".join(lines)
