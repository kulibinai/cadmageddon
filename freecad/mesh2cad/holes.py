"""Hole stacks: a drilled, counterbored or countersunk hole as one cutter.

`faces` lists, as `C0`, `C1`, ..., every set of round surfaces on one axis
that has a concave cylinder or cone in it. A hole is such a set's CONCAVE faces:
a cylinder per bore step, a cone per countersink, chamfer or drill point.
`sketch --op hole --use C3` turns them into one solid -- a cylinder over each
cylinder's measured axial range, a frustum over each cone's -- for
`--combine cut` to subtract.

Every dimension is a measured surface: radius and range of each cylinder, the
two end radii of each cone. Nothing is typed, nothing is assumed about thread
or drill standards: they come off the faces. A counterbore's step is a second ring
at another level that a single extrude cannot reach.
"""

import numpy as np

from mesh2cad.emit import _fmt, quantise
from mesh2cad.segment import coaxial_groups

AXIS_NAMES = "XYZ"
# the sketch plane whose normal is each world axis
PLANE_OF_AXIS = {0: "YZ", 1: "XZ", 2: "XY"}


class HoleError(ValueError):
    """A stack that cannot be made into a hole; the message says why."""


def stack_names(seg):
    """[(name, [surface index, ...])]: every set of round surfaces on one axis
    that has a concave cylinder or cone in it -- a hole, from a single drilled
    bore up to a counterbore with a chamfer and a drill point -- in the order
    `faces` prints them."""
    S = seg.surfaces
    groups = [g for g in coaxial_groups(seg, min_size=1)
              if any(S[i].sense == "concave" and S[i].kind in ("cylinder", "cone")
                     for i in g)]
    return [(f"C{n}", g) for n, g in enumerate(groups)]


def _world_axis(s):
    a = np.asarray(s.params["axis"], float)
    k = int(np.argmax(np.abs(a)))
    if abs(a[k]) < 1 - 1e-6:
        return None, a
    return k, a


def pieces(seg, names):
    """The cutter pieces of the named stacks.

    Returns (k, pieces, notes): k is the world axis the holes run along, and
    each piece is (kind, centre_uv, lo, hi, r_lo, r_hi) along it -- r_lo / r_hi
    the radius at the lo / hi end (equal for a cylinder).
    """
    known = dict(stack_names(seg))
    S = seg.surfaces
    k_all, out, notes = None, [], []
    for name in names:
        if name not in known:
            have = ", ".join(sorted(known, key=lambda n: int(n[1:]))) or "none"
            raise HoleError(f"no stack {name!r}; `faces` names them: {have}")
        group = [S[i] for i in known[name]]
        concave = [s for s in group if s.sense == "concave"
                   and s.kind in ("cylinder", "cone")]
        if not concave:
            raise HoleError(f"{name} has no concave cylinder or cone -- it is a "
                            f"boss or a shaft, not a hole")
        skipped = [s.name for s in group if s not in concave]
        if skipped:
            notes.append(f"{name}: left out {', '.join(skipped)} (convex, or not a "
                         f"cylinder or cone)")
        for s in concave:
            k, a = _world_axis(s)
            if k is None:
                raise HoleError(f"{name}'s axis runs off the axes of this frame "
                                f"({_fmt(a[0])}, {_fmt(a[1])}, {_fmt(a[2])}); a stack "
                                f"is cut in its own axis's frame, so leave out "
                                f"--frame or give --frame {name}")
            if k_all is None:
                k_all = k
            elif k != k_all:
                raise HoleError("the stacks named run along different axes; give "
                                "the ones along one axis per command")
            p = np.asarray(s.params["point"], float)
            t0, t1 = (float(x) for x in s.extent["along"])
            w0, w1 = p[k] + t0 * a[k], p[k] + t1 * a[k]
            if s.kind == "cylinder":
                r0 = r1 = float(s.params["radius"])
            else:
                off, slope = float(s.params["offset"]), float(s.params["slope"])
                r0, r1 = off + slope * t0, off + slope * t1
            if w0 > w1:
                w0, w1, r0, r1 = w1, w0, r1, r0
            i, j = [x for x in range(3) if x != k]
            out.append((s.kind, (float(p[i]), float(p[j])), w0, w1,
                        max(r0, 0.0), max(r1, 0.0)))
    return k_all, out, notes


def profiles(parts):
    """[(centre, [(t, r), ...])]: every hole's profile -- its radius along its
    axis, as a closed polyline that leaves the axis, runs along the hole and
    comes back -- one per run of pieces that touch or overlap along one axis.

    Where pieces overlap (a countersink over its bore) the profile takes the
    wider; a step is a radial edge. Points are on the 4-decimal grid the
    program is written on, with repeats and straight-through points dropped.
    """
    by_centre = {}
    for p in parts:
        by_centre.setdefault((round(p[1][0], 4), round(p[1][1], 4)), []).append(p)
    out = []
    for ps in by_centre.values():
        centre = ps[0][1]
        segs = [(lo, hi, r0, r1) for _k, _c, lo, hi, r0, r1 in ps if hi - lo > 1e-9]

        def r_at(s, t):
            return s[2] + (s[3] - s[2]) * (t - s[0]) / (s[1] - s[0])

        ts = {x for s in segs for x in s[:2]}
        for a in range(len(segs)):
            for b in range(a + 1, len(segs)):
                lo, hi = max(segs[a][0], segs[b][0]), min(segs[a][1], segs[b][1])
                if hi <= lo:
                    continue
                da = r_at(segs[a], lo) - r_at(segs[b], lo)
                db = r_at(segs[a], hi) - r_at(segs[b], hi)
                if da * db < 0:
                    ts.add(lo + (hi - lo) * da / (da - db))
        ts = sorted(ts)
        runs, run = [], []
        for a, b in zip(ts[:-1], ts[1:], strict=True):
            m = 0.5 * (a + b)
            cover = [s for s in segs if s[0] <= m <= s[1]]
            if not cover:
                if run:
                    runs.append(run)
                run = []
                continue
            s = max(cover, key=lambda s: r_at(s, m))
            run.append((a, b, r_at(s, a), r_at(s, b)))
        if run:
            runs.append(run)
        for run in runs:
            pts = [(run[0][0], 0.0)]
            for a, b, ra, rb in run:
                pts += [(a, ra), (b, rb)]
            pts.append((run[-1][1], 0.0))
            pts = _tidy(pts)
            if len(pts) >= 3:
                out.append((centre, pts))
    return out


def _tidy(pts):
    """Points on the emit grid, without repeats or points a straight edge
    runs through."""
    q = []
    for t, r in pts:
        p = (quantise(t), quantise(max(r, 0.0)))
        if not q or p != q[-1]:
            q.append(p)
    out = []
    for p in q:
        while len(out) >= 2:
            (x0, y0), (x1, y1) = out[-2], out[-1]
            cross = (x1 - x0) * (p[1] - y1) - (y1 - y0) * (p[0] - x1)
            dot = (x1 - x0) * (p[0] - x1) + (y1 - y0) * (p[1] - y1)
            if abs(cross) < 1e-9 and dot >= 0:
                out.pop()
            else:
                break
        out.append(p)
    return out


def cutter_code(k, parts):
    """The named holes as a feature block body: each hole ONE solid of
    revolution, its profile (`profiles`) sketched on a plane containing its
    axis and revolved about a datum on that axis -- a Groove in a cut block.

    Not a solid per face, cylinders and frustums set end to end and fused:
    every joint between two of them is a pair of coincident discs, on which
    OCC's fuse loses volume (a cut of many B-spline pieces can return no
    solid at all), and the mesh route keeps each one as a sheet of no
    thickness, so the STL falls into pieces when read back from disk. A
    revolved profile has no joints inside it.
    """
    from mesh2cad.emit import _pt, plane_expr
    from mesh2cad.section import PLANES

    plane = "XY" if k in (0, 1) else "XZ"          # a plane containing the axis
    spec = PLANES[plane]
    n_ax, u_ax, v_ax = spec["axis"], spec["u"], spec["v"]
    q = u_ax if v_ax == k else v_ax                # in the plane, across the axis
    i, j = [x for x in range(3) if x != k]
    lines, names = [], []
    for n, (c, pts) in enumerate(profiles(parts)):  # noqa: B007
        cw = {i: float(c[0]), j: float(c[1])}

        def local(t, r, cw=cw):
            w = [0.0, 0.0, 0.0]
            w[k], w[q], w[n_ax] = t, cw[q] + r, cw[n_ax]
            return (w[u_ax], w[v_ax])

        ring = ", ".join(_pt(local(t, r)) for t, r in pts)
        a0, a1 = local(0.0, 0.0), local(1.0, 0.0)
        pl = plane_expr(plane, cw[n_ax])
        lines += [f"_s{n} = _sketch(_b, {pl})",
                  f"_polygon(_s{n}, [{ring}])",
                  f"_ax{n} = _axis_in(_b, {pl}, {_pt(a0)}, {_pt(a1)})",
                  f"_revolve(_b, _s{n}, _ax{n})"]
        names.append(n)
    if not names:
        raise HoleError("every face of the named stacks has zero length along "
                        "its axis")
    return "\n".join(lines)


def describe_parts(parts, k):
    ax = AXIS_NAMES[k]
    cyl = sorted({round(p[4], 4) for p in parts if p[0] == "cylinder"})
    cones = sum(1 for p in parts if p[0] == "cone")
    lo, hi = min(p[2] for p in parts), max(p[3] for p in parts)
    return (f"{len(parts)} faces: cylinder R {', '.join(f'{r:g}' for r in cyl) or '-'}"
            f"{f', {cones} cone(s)' if cones else ''}; {ax}={lo:.4f} to "
            f"{ax}={hi:.4f}")
