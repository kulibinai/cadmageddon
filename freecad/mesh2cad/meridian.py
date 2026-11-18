"""The meridian: a part's profile about an axis, whatever crosses an axial cut.

A section through a revolve's axis is its profile only when nothing else
crosses that plane. Holes on a bolt circle, blades, teeth and fins do, at most
angles: a cut through a drum's axis that goes through its bores revolves into
separate bodies with the web between them missing. A meridian takes the cut at
every angle instead -- 72 half-planes through the axis, one every 5 deg -- and
gives two profiles of them:
- the ENVELOPE, their union: material at ANY angle. Holes are filled, blades
  and teeth become full rings. Revolve it, then cut what is not round.
- the CORE, their intersection: material at EVERY angle. Blades and teeth are
  gone and holes leave a gap. Revolve it, then add what stands on it.

The profiles are polygons, named and fitted like any section, so
`sketch --meridian` can revolve one by name.
"""

import re

import numpy as np

from mesh2cad.emit import EMIT_DECIMALS

ANGLES = 72
AXES = "XYZ"
# The world plane the profile is drawn in, per axis: it contains the axis, and
# its first in-plane coordinate (u) or second (v) runs along it.
PLANE_FOR = {0: "XZ", 1: "XY", 2: "XZ"}


class MeridianError(ValueError):
    """An axis no meridian can be taken about; the message says why."""


# An axis this close to a world axis is that axis: drafted cones fit an axis
# a fraction of a degree off, and a face `faces` lists as "axis |Z" must not be
# refused for a tiny fitting error.
SNAP_DEG = 1.0


def _others(k):
    return [i for i in range(3) if i != k]


def axis_from(name, seg_of):
    """(k, centre, what): a world axis k and where it crosses the plane across
    it, from a round face F#, a hole stack C#, or "Y=0,Z=-152" (two coordinates
    fixed, the free one is the axis). `seg_of()` gives the segmentation, asked
    for only when a name needs it."""
    from mesh2cad.holes import stack_names
    from mesh2cad.segment import principal_of

    text = name.strip()
    m = re.fullmatch(r"([XYZxyz])\s*=\s*(-?[\d.eE+-]+)\s*,\s*([XYZxyz])\s*=\s*(-?[\d.eE+-]+)",
                     text)
    if m:
        fixed = {AXES.index(m.group(1).upper()): float(m.group(2)),
                 AXES.index(m.group(3).upper()): float(m.group(4))}
        if len(fixed) != 2:
            raise MeridianError(f"{text}: name two different coordinates")
        k = next(i for i in range(3) if i not in fixed)
        i, j = _others(k)
        return k, (fixed[i], fixed[j]), f"the line {text}"
    up = text.upper()
    seg = seg_of()
    if up.startswith("C") and up[1:].isdigit():
        stacks = dict(stack_names(seg))
        if up not in stacks:
            raise MeridianError(f"no hole stack {up}; `faces` names "
                                + (", ".join(stacks) or "none"))
        s = seg.surfaces[stacks[up][0]]
        what = f"hole stack {up}"
    else:
        s = seg.by_name(up)
        if s is None:
            raise MeridianError(f"no surface {up}; name a round face F# or a hole "
                                f"stack C# from `faces`, or a line like Y=0,Z=0")
        if "axis" not in s.params:
            raise MeridianError(f"{up} is a {s.kind}: it has no axis. Name a "
                                f"cylinder, cone or torus, or a hole stack")
        what = f"{s.kind} {up}"
    a = np.asarray(s.params["axis"], float)
    a = a / (np.linalg.norm(a) or 1.0)
    k = int(np.argmax(np.abs(a)))
    off = float(np.degrees(np.arccos(min(1.0, abs(a[k])))))
    if principal_of(a, tol_rad=1e-4) is None and off > SNAP_DEG:
        raise MeridianError(f"the axis of {what} runs off X, Y and Z ({off:.2f} deg off "
                            f"{AXES[k]}); a meridian is taken about a world axis -- turn "
                            f"the part onto this one first with --frame {up} (section "
                            f"and sketch take it)")
    p = np.asarray(s.params["point"], float)
    i, j = _others(k)
    note = (f" ({off:.2f} deg off {AXES[k]}: taken as {AXES[k]}, through the face's own "
            f"point)" if principal_of(a, tol_rad=1e-4) is None else "")
    return k, (float(p[i]), float(p[j])), f"the axis of {what}{note}"


def _half_cut(mesh, k, centre, theta):
    """The cut by the half-plane at angle theta about the axis, as shapely
    polygons in (t, r): t the coordinate along the axis, r >= 0 the distance
    from it."""
    from shapely.geometry import box
    from shapely.ops import unary_union

    i, j = _others(k)
    e = np.zeros(3)
    e[k] = 1.0
    d = np.zeros(3)
    d[i], d[j] = np.cos(theta), np.sin(theta)
    n = np.cross(e, d)
    o = np.zeros(3)
    o[i], o[j] = centre
    sec = mesh.section(plane_origin=o, plane_normal=n)
    if sec is None:
        return None
    T = np.eye(4)
    T[0, :3], T[1, :3], T[2, :3] = e, d, n
    T[:3, 3] = -T[:3, :3] @ o
    planar, _ = sec.to_2D(to_2D=T, check=False)
    polys = [p for p in planar.polygons_full if p is not None and p.area > 0]
    if not polys:
        return None
    region = unary_union(polys)
    if not region.is_valid:
        region = region.buffer(0)
    lo, hi = region.bounds[0] - 1.0, region.bounds[2] + 1.0
    return region.intersection(box(lo, 0.0, hi, max(region.bounds[3], 0.0) + 1.0))


# Besides the ANGLES regular cuts, one through each corner the part has about
# the axis: a section's farthest point is one of its outermost vertices, and
# between two cuts 5 deg apart the envelope comes up short of it, and a body
# clipped by it loses true material.
CORNER_SLABS = 32       # slabs along the axis, each with its own hull
CORNER_MAX = 96         # corner cuts at most
CORNER_SAME = np.radians(0.25)


def corner_angles(mesh, k, centre, regular):
    """Angles about the axis of the outermost vertices, slab by slab along it,
    that no regular cut already passes within CORNER_SAME of."""
    from scipy.spatial import ConvexHull

    i, j = _others(k)
    v = np.asarray(mesh.vertices, float)
    t = v[:, k]
    uv = v[:, [i, j]] - np.asarray(centre, float)
    edges = np.linspace(t.min(), t.max(), CORNER_SLABS + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        pts = uv[(t >= a) & (t <= b)]
        if len(pts) < 3:
            continue
        try:
            hull = pts[ConvexHull(pts).vertices]
        except Exception:                                          # noqa: BLE001
            continue
        out.extend(np.arctan2(hull[:, 1], hull[:, 0]) % (2 * np.pi))
    out = np.unique(np.round(np.asarray(out, float), 6))
    reg = np.asarray(regular, float)
    far = [a for a in out
           if np.min(np.abs((reg - a + np.pi) % (2 * np.pi) - np.pi)) > CORNER_SAME]
    kept = []
    for a in far:
        if all(abs((a - b + np.pi) % (2 * np.pi) - np.pi) > CORNER_SAME for b in kept):
            kept.append(a)
    if len(kept) > CORNER_MAX:     # a round part: its hull is a polygon of facets
        kept = [kept[int(q)] for q in np.linspace(0, len(kept) - 1, CORNER_MAX)]
    return kept


def profiles(mesh, k, centre, n=ANGLES):
    """(envelope, core, notes): shapely geometries in (t, r)."""
    from shapely.ops import unary_union

    cuts, failed = [], 0
    regular = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    for theta in list(regular) + corner_angles(mesh, k, centre, regular):
        try:
            c = _half_cut(mesh, k, centre, theta)
        except Exception:                                          # noqa: BLE001
            c = None
        if c is None or c.is_empty:
            failed += 1
            continue
        cuts.append(c)
    notes = []
    if failed:
        notes.append(f"{failed} of {n} half-plane cuts were empty or did not close "
                     f"(an open mesh, or nothing at that angle) and are left out")
    if not cuts:
        raise MeridianError("no half-plane cut about this axis met the part")
    env = unary_union(cuts)
    core = cuts[0]
    for c in cuts[1:]:
        core = core.intersection(c)
        if core.is_empty:
            break
    return env.buffer(0), (core.buffer(0) if not core.is_empty else core), notes


def shortfall(mesh, k, centre, side):
    """How much of the envelope the two half-cuts of the axial plane through
    world axis `side` miss: [share for +side, share for -side].

    A cut through the axis is the revolve profile only when nothing else
    crosses it. When holes, blades or teeth do, one or both halves come out
    short of the material that is there at other angles.
    """
    from shapely.ops import unary_union

    i, j = _others(k)
    env = unary_union([c for c in (_safe_cut(mesh, k, centre, t)
                                   for t in np.linspace(0.0, 2 * np.pi, ANGLES,
                                                        endpoint=False))
                       if c is not None and not c.is_empty])
    if env.is_empty or env.area <= 0:
        return []
    base = 0.0 if side == i else np.pi / 2
    out = []
    for theta in (base, base + np.pi):
        c = _safe_cut(mesh, k, centre, theta)
        have = 0.0 if c is None or c.is_empty else c.area
        out.append(max(0.0, (env.area - have) / env.area))
    return out


def _safe_cut(mesh, k, centre, theta):
    try:
        return _half_cut(mesh, k, centre, theta)
    except Exception:                                              # noqa: BLE001
        return None


def to_plane(geom, k, centre):
    """Polygons of a (t, r) geometry in the world plane PLANE_FOR[k], with that
    plane's level and the axis as two (u, v) points."""
    from shapely.affinity import affine_transform

    plane = PLANE_FOR[k]
    if k == 0:          # XZ at Y = centre[0]: u = X = t, v = Z = cZ + r
        level, m = centre[0], [1, 0, 0, 1, 0, centre[1]]
        axis = ((0.0, centre[1]), (1.0, centre[1]))
    elif k == 1:        # XY at Z = centre[1]: u = X = cX + r, v = Y = t
        level, m = centre[1], [0, 1, 1, 0, centre[0], 0]
        axis = ((centre[0], 0.0), (centre[0], 1.0))
    else:               # XZ at Y = centre[1]: u = X = cX + r, v = Z = t
        level, m = centre[1], [0, 1, 1, 0, centre[0], 0]
        axis = ((centre[0], 0.0), (centre[0], 1.0))
    if geom is None or geom.is_empty:
        return plane, level, [], axis
    g = affine_transform(geom, m)
    # Snapped to the grid the program is written on (emit.quantise, 4
    # decimals). The union of 72 cuts keeps vertex pairs closer than that, and
    # written out they become zero-length lines OCC refuses ("BRep_API:
    # command not done").
    from shapely import set_precision
    g = set_precision(g, 10.0 ** -EMIT_DECIMALS)
    polys = [p for p in getattr(g, "geoms", [g]) if p.geom_type == "Polygon" and p.area > 0]
    polys.sort(key=lambda p: -p.area)
    return plane, level, polys, axis


def revolve_code(plane, level, outer, holes, axis, angle=360.0):
    """The named outer rings revolved, the named hole rings cut from them --
    one feature (a block of the program) -- by `angle` (signed, as
    emit.revolve_axis takes it).

    A hole goes into the sketch of the outer ring it lies in, so one
    Revolution of that face carries it -- the outline revolved minus the
    hole revolved. A hole in no single outer ring
    cannot be said by one face: the feature is then built in its own tool body
    (`_b.mixed`), its holes as Grooves there, so they cut only this feature
    and never the part already built."""
    from mesh2cad.emit import (
        EmitError,
        _fmt,
        _holes_by_outer,
        _pt,
        drawable,
        nothing_left,
        plane_expr,
        revolve_axis,
        ring_code,
    )

    pl = plane_expr(plane, level)
    a0, a1, turn = revolve_axis(angle, *axis)
    asked = outer
    outer, note = drawable(outer)
    holes, hole_note = drawable(holes)
    if not outer:
        raise EmitError(nothing_left(asked, "revolve"))
    groups, mixed, notes = _holes_by_outer(outer, holes)
    lines = (note + hole_note).splitlines() + notes
    if mixed:
        lines.append("_b.mixed = True")
    lines.append(f"_ax = _axis_in(_b, {pl}, {_pt(a0)}, {_pt(a1)})")
    for i, (ring, inner) in enumerate(groups):
        v = f"_s{i}"
        lines += [f"{v} = _sketch(_b, {pl})", ring_code(ring, v)]
        if not mixed:
            lines += [ring_code(h, v) for h in inner]
        lines.append(f"_revolve(_b, {v}, _ax, {_fmt(turn)})")
    if mixed:
        for j, h in enumerate(holes):
            v = f"_h{j}"
            lines += [f"{v} = _sketch(_b, {pl})", ring_code(h, v),
                      f"_revolve(_b, {v}, _ax, {_fmt(turn)}, minus=True)"]
    return "\n".join(lines)
