"""Profile families and their laws across levels.

A section is traced into lines and arcs, and a designed profile comes back as
whatever the tessellation made of it: a tray's rounded corners trace as three
to five arcs and lines per corner, different at every level. The designer
drew a rounded rectangle and moved it: half-size shrinking with the draft,
corner radius growing with depth. Fitted as that family at several levels,
its four numbers per level line up on straight lines, and a ruled loft
between the family at the two named levels is the wall.

Fitted here: circle, slot and axis-aligned rounded rectangle; a corner radius
is solved from the section's area, so it needs no arc to have been traced
cleanly. Tangent-arc chains are not fitted.
"""

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
    # The corner radius from the area: A = 4 hw hh - (4 - pi) r^2 -- immune to a
    # corner traced in pieces.
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
    transition (a level sampled just inside a bend, say) and is left out, at most
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


def wire_code(f, wp, var):
    """CadQuery that leaves the family's closed wire in `var`, on workplane
    expression `wp` (already at the level)."""
    from mesh2cad.emit import _fmt
    cx, cy, hw, hh, r = (f[k] for k in PARAMS)
    at_ = f"{wp}.center({_fmt(cx)}, {_fmt(cy)})"
    if f["family"] == "circle":
        return f"{var} = {at_}.circle({_fmt(r)}).val()"
    if f["family"] == "slot":
        length, diameter = 2 * max(hw, hh), 2 * min(hw, hh)
        angle = 0 if hw >= hh else 90
        return f"{var} = {at_}.slot2D({_fmt(length)}, {_fmt(diameter)}, {angle}).val()"
    base = f"{at_}.rect({_fmt(2 * hw)}, {_fmt(2 * hh)}).val()"
    if r <= 1e-9 or f["family"] == "rect":
        return f"{var} = {base}"
    return f"{var} = {base}\n{var} = {var}.fillet2D({_fmt(r)}, {var}.Vertices())"


def loft_code(plane, sense, levels_fits):
    """A ruled loft through the family at each (level, fit)."""
    from mesh2cad.emit import _steps_line, workplane_expr
    lines, names = [], []
    for i, (level, f) in enumerate(levels_fits):
        v = f"_lw{i}"
        lines.append(wire_code(f, workplane_expr(plane, sense, level), v))
        names.append(v)
    lines.append(f"_solid = cq.Workplane(obj=cq.Solid.makeLoft([{', '.join(names)}], True))")
    lines.append("result = _solid")
    lines.append(_steps_line([("add", "_solid")]))
    return "\n".join(lines)
