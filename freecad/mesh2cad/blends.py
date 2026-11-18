"""Fillets and chamfers on named faces, built as solids.

    m2c.py sketch part.stl --op fillet --use F12,F13 --base r.py --emit r.py
    m2c.py sketch part.stl --op chamfer --use F7 --base r.py --emit r.py

F12 is a blend face of the TARGET, as `faces` lists it: a cylinder (a round on
a straight edge) or a torus (a round on a circular edge) tangent to the faces
it rounds; a chamfer is a narrow plane (straight edge) or cone (circular edge)
between two faces. Each named face becomes one solid, the region between that
face and the corner its two neighbours would meet at:

  fillet   the arc's end tangents meet at the corner; the region between the
           arc and that corner, swept along the edge or round the axis;
  chamfer  the triangle between the strip and the corner of the two faces it
           joins (found among its neighbours, or a right angle if they are not).

It is cut where the edge is convex -- the build has material there the target
rounds off -- and added where it is concave, a cove the build lacks. Either way
the region is what the target has there (air for a cut, material for an add),
so the solid is right whatever the build holds at that corner.

The named face is only where the solid starts. The segmentation can split one
blend into pieces of other kinds (one chamfer cone as several cones, freeform
pieces and a cylinder), so the faces
joined to it that lie on its surface are taken with it (`same_surface`). And
the solid is kept only where the target agrees with it -- material for an add,
air for a cut (`keep_right_side`): a cove can end against other rounds
before its face does, and a full-length solid would stick out there.

A flat end cannot follow a blend that runs out at a slant, though: where a
wall cuts a blend obliquely, a solid kept by its share of right samples runs
on past the part, even past the bounding box. So the target just past the corner -- the plate or wall the blend
runs on to -- is cut, and the solid clipped to it (`clip`): to the material
there for an add, to the air for a cut.

Its ends are placed, not rounded to a station (`_end`), and it runs on past the
end of its face while the target agrees (`_overrun`): where two rounds of a
chain meet, the corner between them belongs to neither face.

Never an OCC fillet. OCC fillets can fail (on a cove, on a cone/square
junction); and an OCC fillet changes the accumulated solid, which the per-feature
mesh route (mesh2cad.build) cannot follow. A solid is an operand like any other.
"""
import numpy as np

AXES = "XYZ"
PLANE_OF_AXIS = {0: "YZ", 1: "XZ", 2: "XY"}
MAX_SWEEP = 179.0       # degrees: an arc this wide is a hole or a boss, not a blend
SIDE_TOL = 0.05         # x strip width: how near a neighbour plane must pass its edge
CHAMFER_ANGLES = (15.0, 75.0)   # degrees between a chamfer and each face it joins
CHAMFER_WIDTH = 0.05    # x diagonal: wider strips are faces, not chamfers (for the hint)
FILLET_RADIUS = 0.05    # x diagonal: larger rounds are profile arcs, not blends (for the hint)
BLEND_FLOOR = 2.0       # mm: a blend this small is one on any part, whatever its diagonal (for the hint)
RUN_ON = 0.02           # x chord: the solid's sides run this far past the faces it meets
BLEND_DEPTH = 1.0       # x blend size: a leftover thicker than this on average is more than a blend
ON_SURFACE = 2e-4       # x diagonal: a face lies on a blend's surface when 90% of its vertices are this near it
STATIONS = 24           # places along a solid (or round it) where the target is checked against it
RIGHT_SIDE = 0.5        # share of a station's samples the target must agree with for the solid to run through it
END_SHARE = 0.15        # the same at a solid's end, where a clip takes off what the target lacks
END_STEPS = 6           # halvings of a station that place a solid's end
OVERRUN = 1.0           # x blend size: how far past its face a solid may run ...
OVERRUN_SHARE = 0.9     # ... where this share of its profile still agrees with the target
TURNED_WIDTH = 0.15     # x diagonal: a convex cone on a hole's axis (a turned end) is a chamfer up to this wide
CLIP_PAST = 0.05        # x blend size: how far past the corner the target is cut for `clip`
CLIP_NOISE = 0.05       # mm: an overhang narrower than this is two tessellations, not an overhang
CLIP_KEEP = 0.5         # share of the solid's footprint a clip must leave, else it cut the wrong thing
CLIP_POINTS = 200       # outline points a clip may have; finer is not a wall
SHORT_TOL = 0.1         # mm: a build this far inside a convex blend's face stops short of it
SHORT_SHARE = 0.5       # share of the blend's face the build must stop short of, for `short_of` to say so
SHORT_SAMPLES = 200     # points on a blend's faces for `short_of`
# A blend shapes both sides of its face. A build whose wall stands a little off
# the target's does not end where the target's faces do, and a solid bounded by
# those faces leaves the difference standing: a fin beside a cut round, or a
# wall standing into a cove's open side, which an added cove cannot take away.
OUT_BAND = 0.25         # x blend size: how far a convex blend's cut runs on past the faces it meets
IN_BAND = 0.3           # x blend size: how deep a concave blend trims the build on its open side
BAND_SHARE = 0.9        # share of a band's samples that must be air in the target, at each station
# A turned blend's open-side trim is checked at this many places round
# the axis, and kept where this share of it is air: at STATIONS and BAND_SHARE
# a trim can find air at every place and be cut a full turn, deep into flats
# between them (a hex); at 72 and 0.97 it is separate runs between them.
TRIM_STATIONS = 72
TRIM_SHARE = 0.97
BAND_ARC = 24           # chords on each arc of a trim's outline


class BlendError(ValueError):
    """A face that cannot be made into a fillet or chamfer; the message says why."""


def _basis(a):
    a = np.asarray(a, float)
    a = a / np.linalg.norm(a)
    h = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(a, h)
    u /= np.linalg.norm(u)
    return u, np.cross(a, u)


def _span(deg, step):
    """(start, sweep) in degrees of the arc a set of angles covers: everything
    but the largest gap, or (0, 360) when no gap is wider than the facets."""
    a = np.sort(np.mod(np.asarray(deg, float), 360.0))
    gaps = np.diff(np.r_[a, a[0] + 360.0])
    k = int(np.argmax(gaps))
    if gaps[k] <= 1.5 * max(float(step), 1e-6) + 1e-6:
        return 0.0, 360.0
    return float(a[(k + 1) % len(a)]), float(360.0 - gaps[k])


def _points(mesh, *surfaces):
    F = np.asarray(mesh.faces)
    return np.asarray(mesh.vertices)[np.unique(np.concatenate([F[o.faces].ravel() for o in surfaces]))]


def same_surface(mesh, seg, s):
    """The blend face and the faces joined to it, one after another, that lie
    on its own cylinder, torus or cone: 90% of their vertices within
    ON_SURFACE, and of their triangles' centres within 4 x that (a side plane
    can have its vertices on the cone's edge, its triangles nowhere near it).
    A solid from the named face alone can cover a fraction of the turn and stop
    short of the stock's corners, leaving tabs."""
    if s.kind not in ("cylinder", "torus", "cone"):
        return [s]
    a = np.asarray(s.params["axis"], float)
    a /= np.linalg.norm(a)
    p = np.asarray(s.params["point"], float)

    def polar(V):
        q = V - p
        t = q @ a
        return t, np.linalg.norm(q - np.outer(t, a), axis=1)

    if s.kind == "cylinder":
        R = float(s.params["radius"])

        def off(V):
            return np.abs(polar(V)[1] - R)
    elif s.kind == "torus":
        Rm, r = float(s.params["major"]), float(s.params["minor"])

        def off(V):
            t, rho = polar(V)
            return np.abs(np.hypot(rho - Rm, t) - r)
    else:
        # The cone's line in (along, radius), fitted to the named face itself.
        k, r0 = np.polyfit(*polar(_points(mesh, s)), 1)

        def off(V):
            t, rho = polar(V)
            return np.abs(rho - (r0 + k * t)) / np.hypot(1.0, k)
    S = seg.surfaces
    nb = {}
    for j in seg.joints:
        nb.setdefault(j.a, []).append(j.b)
        nb.setdefault(j.b, []).append(j.a)
    tol = ON_SURFACE * float(seg.diag)
    centres = np.asarray(mesh.triangles_center)
    first = S.index(s)
    keep, todo, seen = [first], [first], {first}
    while todo:
        for j in nb.get(todo.pop(), ()):
            if j in seen:
                continue
            seen.add(j)
            if (np.percentile(off(_points(mesh, S[j])), 90) <= tol
                    and np.percentile(off(centres[S[j].faces]), 90) <= 4.0 * tol):
                keep.append(j)
                todo.append(j)
    return [S[i] for i in keep]


def _outward(mesh, s):
    """Area-weighted mean of the face's triangle normals (they point out of the part)."""
    w = np.asarray(mesh.area_faces)[s.faces]
    n = (np.asarray(mesh.face_normals)[s.faces] * w[:, None]).sum(axis=0)
    return n / (np.linalg.norm(n) or 1.0)


def _joint_sense(seg, idx):
    """'convex', 'concave' or None: how the face meets its non-tangent neighbours."""
    conv = conc = 0.0
    for j in seg.joints:
        if idx in (j.a, j.b):
            if j.kind == "convex":
                conv += j.length
            elif j.kind == "concave":
                conc += j.length
    if conv > 2 * conc:
        return "convex"
    if conc > 2 * conv:
        return "concave"
    return None


def _arc(centre, r, a0, sweep):
    """End points, mid point and corner of an arc in 2-D: the corner is where the
    tangents at its ends meet."""
    phi = np.radians(sweep / 2.0)
    t0, t1, tm = np.radians(a0), np.radians(a0 + sweep), np.radians(a0 + sweep / 2.0)
    c = np.asarray(centre, float)
    e0 = c + r * np.array([np.cos(t0), np.sin(t0)])
    e1 = c + r * np.array([np.cos(t1), np.sin(t1)])
    m = c + r * np.array([np.cos(tm), np.sin(tm)])
    p = c + (r / np.cos(phi)) * np.array([np.cos(tm), np.sin(tm)])
    return e0, e1, m, p


def fillet(mesh, seg, s):
    """The solid for a round face: a cylinder (straight edge) or a torus
    (circular edge)."""
    if s.kind not in ("cylinder", "torus"):
        raise BlendError(f"{s.name} is a {s.kind}; --op fillet takes a round blend face "
                         f"(a cylinder or a torus tangent to its neighbours)"
                         + ("; a flat or conical strip is --op chamfer" if s.kind in ("plane", "cone")
                            else ""))
    mode = {"convex": "cut", "concave": "add"}.get(s.sense)
    if mode is None:
        raise BlendError(f"{s.name} is {s.sense}: neither convex (a rounded edge, cut) "
                         f"nor concave (a cove, added), so there is no one corner to round")
    grp = same_surface(mesh, seg, s)
    V = _points(mesh, *grp)
    step = max(o.facet_angle for o in grp)
    named = {"faces": [o.name for o in grp], "kind": s.kind, "note": None}
    a = np.asarray(s.params["axis"], float)
    a /= np.linalg.norm(a)
    p = np.asarray(s.params["point"], float)
    u, v = _basis(a)
    q = V - p
    t = q @ a
    alpha = np.degrees(np.arctan2(q @ v, q @ u))
    if s.kind == "cylinder":
        R = float(s.params["radius"])
        a0, sweep = _span(alpha, step)
        if sweep > MAX_SWEEP:
            raise BlendError(f"{s.name} goes {sweep:.0f} deg round its axis: a hole or a "
                             f"boss, not a blend (--op hole cuts holes)")
        e0, e1, m, c = _arc((0.0, 0.0), R, a0, sweep)
        lo, hi = float(t.min()), float(t.max())
        return {"face": s.name, "op": "fillet", "mode": mode, "how": "prism",
                "origin": p + lo * a, "x": u, "normal": a, "pts": [e0, c, e1], "mid": m,
                "length": hi - lo, "axis": a, "along": (p, lo, hi),
                "size": f"R={R:.4g} {sweep:.0f}deg", "w": R, "arc": ((0.0, 0.0), R), **named}
    Rm, r = float(s.params["major"]), float(s.params["minor"])
    if Rm <= r:
        raise BlendError(f"{s.name} is a spindle torus (its tube crosses the axis): no "
                         f"edge to round there")
    rho = np.linalg.norm(q - np.outer(t, a), axis=1)
    psi = np.degrees(np.arctan2(t, rho - Rm))
    p0, tube = _span(psi, step)
    if tube > MAX_SWEEP:
        raise BlendError(f"{s.name} goes {tube:.0f} deg round its tube: a bead or a "
                         f"groove, not a blend of two faces")
    e0, e1, m, c = _arc((Rm, 0.0), r, p0, tube)
    if c[0] < 0:
        raise BlendError(f"the corner {s.name} rounds would lie across its axis")
    a0, sweep = _span(alpha, step)
    x = np.cos(np.radians(a0)) * u + np.sin(np.radians(a0)) * v
    return {"face": s.name, "op": "fillet", "mode": mode, "how": "revolve",
            "origin": p, "x": x, "normal": np.cross(x, a), "pts": [e0, c, e1], "mid": m,
            "angle": sweep, "axis": a, "along": (p, float(t.min()), float(t.max())),
            "size": f"r={r:.4g} on R={Rm:.4g}", "w": r, "arc": ((Rm, 0.0), r), **named}


def chamfer(mesh, seg, s):
    """The solid for a chamfer face: a plane strip (straight edge) or a cone
    (circular edge)."""
    if s.kind not in ("plane", "cone"):
        raise BlendError(f"{s.name} is a {s.kind}; --op chamfer takes a flat strip or a "
                         f"cone between two faces"
                         + ("; a round blend is --op fillet" if s.kind in ("cylinder", "torus")
                            else ""))
    idx = seg.surfaces.index(s)
    sense = _joint_sense(seg, idx)
    if sense is None:
        raise BlendError(f"{s.name} meets its neighbours neither all convex (cut) nor all "
                         f"concave (added), so there is no one corner to bevel")
    mode = "cut" if sense == "convex" else "add"
    out = _outward(mesh, s)
    V = _points(mesh, s)
    if s.kind == "plane":
        return _chamfer_strip(mesh, seg, s, idx, V, out, mode)
    grp = same_surface(mesh, seg, s)
    V = _points(mesh, *grp)
    a = np.asarray(s.params["axis"], float)
    a /= np.linalg.norm(a)
    p = np.asarray(s.params["point"], float)
    u, v = _basis(a)
    q = V - p
    t = q @ a
    rho = np.linalg.norm(q - np.outer(t, a), axis=1)
    near = max(2.0 * float(seg.tol), 1e-9)
    lo, hi = float(t.min()), float(t.max())
    ra = float(np.median(rho[t <= lo + near]))
    rb = float(np.median(rho[t >= hi - near]))
    # Outward direction in the (radius, along) half-plane, from the triangles.
    c3 = np.asarray(mesh.triangles_center)[s.faces] - p
    tc = c3 @ a
    rh = c3 - np.outer(tc, a)
    rh /= np.linalg.norm(rh, axis=1)[:, None] + 1e-300
    fn = np.asarray(mesh.face_normals)[s.faces]
    w = np.asarray(mesh.area_faces)[s.faces]
    n2 = np.array([(np.einsum("ij,ij->i", fn, rh) * w).sum(), ((fn @ a) * w).sum()])
    ea, eb = np.array([ra, lo]), np.array([rb, hi])
    mid = (ea + eb) / 2.0
    # The faces a cone chamfer joins are, almost always, a plane across the axis
    # and a cylinder round it: the corner is at one end's radius and the other's
    # level. Of the two such corners, the cut takes the one outside the strip.
    want = 1.0 if mode == "cut" else -1.0
    c = next(x for x in (np.array([ra, hi]), np.array([rb, lo]))
             if want * float((x - mid) @ n2) > 0)
    alpha = np.degrees(np.arctan2(q @ v, q @ u))
    a0, sweep = _span(alpha, max(o.facet_angle for o in grp))
    x = np.cos(np.radians(a0)) * u + np.sin(np.radians(a0)) * v
    return {"face": s.name, "op": "chamfer", "mode": mode, "how": "revolve",
            "origin": p, "x": x, "normal": np.cross(x, a), "pts": [ea, c, eb], "mid": None,
            "angle": sweep, "axis": a, "along": (p, lo, hi),
            "size": f"{np.hypot(*(eb - ea)):.4g} wide, {s.params.get('half_angle', 0.0):.1f}deg",
            "note": None, "w": float(np.hypot(*(eb - ea))), "arc": None,
            "faces": [o.name for o in grp], "kind": s.kind}


def _chamfer_strip(mesh, seg, s, idx, V, n, mode):
    org = V.mean(axis=0)
    Q = V - org
    Q -= np.outer(Q @ n, n)
    # The strip runs along its longest in-plane direction.
    _, _, vt = np.linalg.svd(Q, full_matrices=False)
    e = vt[0] - (vt[0] @ n) * n
    e /= np.linalg.norm(e)
    wdir = np.cross(n, e)
    sv, wv = Q @ e, Q @ wdir
    s0, s1, w0, w1 = float(sv.min()), float(sv.max()), float(wv.min()), float(wv.max())
    W = w1 - w0
    # The faces it joins: planes along its two long edges.
    lines = {}
    for j in seg.joints:
        if idx not in (j.a, j.b):
            continue
        o = seg.surfaces[j.b if j.a == idx else j.a]
        if o.kind != "plane":
            continue
        m = _outward(mesh, o)
        if abs(float(m @ e)) > 0.2:
            continue
        A, B = float(m @ wdir), float(m @ n)
        C = float(m @ (_points(mesh, o).mean(axis=0) - org))
        norm = np.hypot(A, B) or 1.0
        for side, wx in ((0, w0), (1, w1)):
            if abs(A * wx - C) / norm <= SIDE_TOL * W + 2 * float(seg.tol):
                if side not in lines:
                    lines[side] = (A, B, C)
    note = None
    if 0 in lines and 1 in lines:
        M = np.array([lines[0][:2], lines[1][:2]])
        if abs(np.linalg.det(M)) < 1e-9:
            raise BlendError(f"the faces on either side of {s.name} are parallel: no corner "
                             f"for a chamfer")
        c = np.linalg.solve(M, [lines[0][2], lines[1][2]])
    else:
        c = np.array([(w0 + w1) / 2.0, (W / 2.0) * (1 if mode == "cut" else -1)])
        note = (f"note: {s.name}: the faces it joins are not both planes in the "
                f"segmentation, so their corner is taken as a right angle")
    if (c[1] > 0) != (mode == "cut"):
        raise BlendError(f"the corner beside {s.name} lies on the wrong side of it for a "
                         f"{'convex' if mode == 'cut' else 'concave'} edge")
    return {"face": s.name, "op": "chamfer", "mode": mode, "how": "prism",
            "origin": org + s0 * e, "x": wdir, "normal": e,
            "pts": [np.array([w0, 0.0]), c, np.array([w1, 0.0])], "mid": None,
            "length": s1 - s0, "axis": e, "along": (org, s0, s1),
            "size": f"{W:.4g} wide", "note": note, "w": float(W), "arc": None,
            "faces": [s.name], "kind": s.kind}


def parts(mesh, seg, names, op):
    """The solid specs for the named faces, all cut or all added: one per
    surface (faces on one surface make one solid), each kept where the target
    agrees with it."""
    if not names:
        raise BlendError(f"--op {op} needs --use with face names from `faces` (F12, ...)")
    specs = []
    for name in names:
        s = seg.by_name(name)
        if s is None:
            raise BlendError(f"{name} is not a face of this mesh (`faces` names them F0, F1, ...)")
        specs.append(fillet(mesh, seg, s) if op == "fillet" else chamfer(mesh, seg, s))
    modes = {x["mode"] for x in specs}
    if len(modes) > 1:
        cut = ", ".join(x["face"] for x in specs if x["mode"] == "cut")
        add = ", ".join(x["face"] for x in specs if x["mode"] == "add")
        raise BlendError(f"{cut} round convex edges (cut) and {add} concave ones (added): "
                         f"one command does one or the other -- name them in two")
    one = {}
    for x in specs:
        key = frozenset(x["faces"])
        if key in one:
            one[key]["named"].append(x["face"])
        else:
            one[key] = dict(x, named=[x["face"]])
    out = []
    for x in one.values():
        # On to where little of the profile is left, if a clip can take that
        # off every piece; otherwise a solid can stop short of the target's blend.
        far = [clip(mesh, _band_out(mesh, y)) for y in keep_right_side(mesh, x, END_SHARE)]
        out.extend(far if all("clips" in y for y in far) else
                   [clip(mesh, _band_out(mesh, _overrun(mesh, y))) for y in keep_right_side(mesh, x)])
    return out


def _air_by_station(mesh, x, uv, n=STATIONS):
    """Per station (of `n`), the share of the profile points `uv` that are air
    in the target."""
    import trimesh

    P = _stations(x, uv, (np.arange(n) + 0.5) / n)
    sd = trimesh.proximity.signed_distance(mesh, P.reshape(-1, 3)).reshape(n, -1)
    return (sd < 0).mean(axis=1)


def _band_out(mesh, x):
    """A convex blend's cut run on past the two faces it meets, by OUT_BAND of
    its size instead of RUN_ON: beyond those faces the target is air, so the
    build's wall standing a little wide of them goes too, as far as the blend
    reaches. Kept at RUN_ON where the target is not air out there."""
    if x["mode"] != "cut":
        return x
    e0, c, e1 = (np.asarray(q, float) for q in x["pts"])
    d = OUT_BAND * float(x["w"])
    if d <= RUN_ON * float(np.linalg.norm(e1 - e0)):
        return x
    full = _outline(e0, c, e1, d)
    # Points half way out, along both sides and round the corner.
    t = np.linspace(0.1, 0.9, 5)[:, None]
    mid = [(full[0] + full[1]) / 2, (full[2] + c) / 2, (full[3] + full[4]) / 2]
    uv = np.vstack([mid[0] + t * (mid[1] - mid[0]), mid[1] + t * (mid[2] - mid[1])])
    if float(_air_by_station(mesh, x, uv).min()) < BAND_SHARE:
        return x
    return dict(x, run_on=d)


def trims(mesh, specs):
    """For concave blends (added), the solids that cut the build back to the
    blend on its open side: a band IN_BAND of the blend's size deep, under the
    arc for a fillet or beside the strip for a chamfer, kept at the stations
    where it is air in the target. An added cove cannot take away a wall that
    stands into its opening; this does. [] for convex blends, whose cut already
    reaches out (`_band_out`)."""
    out = []
    for x in specs:
        if x["mode"] != "add":
            continue
        e0, c, e1 = (np.asarray(q, float) for q in x["pts"])
        depth = IN_BAND * float(x["w"])
        if x.get("arc") is not None:
            centre, r = x["arc"]
            centre = np.asarray(centre, float)
            if depth >= 0.95 * r:
                depth = 0.5 * r
            a0 = np.arctan2(*(e0 - centre)[::-1])
            a1 = np.arctan2(*(e1 - centre)[::-1])
            am = np.arctan2(*(np.asarray(x["mid"], float) - centre)[::-1])
            # The way round from e0 to e1 that passes the arc's mid point.
            span = (a1 - a0) % (2 * np.pi)
            if (am - a0) % (2 * np.pi) > span:
                span -= 2 * np.pi
            th = a0 + np.linspace(0.0, span, BAND_ARC + 1)
            ring = np.c_[np.cos(th), np.sin(th)]
            poly = list(centre + r * ring) + list(centre + (r - depth) * ring[::-1])
        else:
            t = e1 - e0
            n = np.array([-t[1], t[0]]) / (np.linalg.norm(t) or 1.0)
            if float((c - e0) @ n) > 0:
                n = -n
            poly = [e0, e1, e1 + depth * n, e0 + depth * n]
        y = {k: v for k, v in x.items() if k not in ("clips", "note", "named")}
        y.update(mode="cut", poly=[np.asarray(p, float) for p in poly], mid=None, arc=None, trim=True)
        prism = y["how"] == "prism"
        n = STATIONS if prism else TRIM_STATIONS
        air = _air_by_station(mesh, y, _region(y), n)
        ok = air >= (BAND_SHARE if prism else TRIM_SHARE)
        if not ok.any():
            continue
        full = float(y["length"] if prism else y["angle"])
        circular = not prism and full >= 360.0 - 1e-6
        # On past the blend's ends while the opening goes on: a cove can start on
        # a slanted face, and a band stopped at its end leaves a ledge.
        for k0, k1 in _runs(ok, circular):
            out.append(_overrun(mesh, _slice(y, k0 / n, k1 / n) if (k0, k1) != (0, n) else y))
    return out


def describe_trims(found):
    """`F39 open side 3 mm deep, Z=8.0019 to Z=12.2549 | ...`."""
    out = []
    for x in found:
        a = np.asarray(x["axis"], float)
        k = int(np.argmax(np.abs(a)))
        if x["how"] == "prism":
            n = np.asarray(x["normal"], float)
            o = np.asarray(x["origin"], float)
            k = int(np.argmax(np.abs(n)))
            where = f"{AXES[k]}={o[k]:.4f} to {AXES[k]}={o[k] + float(x['length']) * n[k]:.4f}"
        else:
            where = f"{float(x['angle']):.0f} deg round {AXES[k]}"
        out.append(f"{x['face']} open side {IN_BAND * float(x['w']):.3g} mm deep, {where}")
    return " | ".join(out)


def _region(x, n=12):
    """Profile points inside the solid, a little in from its edges: the
    triangle between its ends and its corner, less the arc's disc for a fillet;
    for a trim (`trims`), its own outline."""
    if x.get("poly") is not None:
        from shapely.geometry import Point, Polygon

        ring = Polygon([tuple(map(float, p)) for p in x["poly"]]).buffer(0)
        lo, hi = np.asarray(ring.bounds[:2]), np.asarray(ring.bounds[2:])
        g = np.stack(np.meshgrid(np.linspace(lo[0], hi[0], 2 * n), np.linspace(lo[1], hi[1], 2 * n)), -1).reshape(-1, 2)
        inner = ring.buffer(-0.03 * float(np.min(hi - lo)))
        inside = [p for p in g if inner.contains(Point(p))]
        return np.asarray(inside) if inside else np.asarray(ring.representative_point().coords)
    e0, c, e1 = (np.asarray(q, float) for q in x["pts"])
    lo, hi = np.minimum.reduce([e0, c, e1]), np.maximum.reduce([e0, c, e1])
    g = np.stack(np.meshgrid(np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], n)), -1).reshape(-1, 2)
    lam = np.linalg.solve(np.c_[e0 - c, e1 - c], (g - c).T).T
    inside = (lam[:, 0] > 0.03) & (lam[:, 1] > 0.03) & (lam.sum(axis=1) < 0.97)
    if x.get("arc") is not None:
        centre, r = x["arc"]
        inside &= np.linalg.norm(g - np.asarray(centre, float), axis=1) > 1.03 * r
    if inside.any():
        return g[inside]
    mid = x.get("mid")
    return (((np.asarray(mid, float) + c) / 2.0) if mid is not None else (e0 + c + e1) / 3.0)[None]


def _stations(x, uv, f):
    """The profile points `uv` placed at fractions `f` of the way along the
    solid (prism) or round its axis (revolve): (len(f), len(uv), 3)."""
    o, X = np.asarray(x["origin"], float), np.asarray(x["x"], float)
    f = np.asarray(f, float)
    if x["how"] == "prism":
        n = np.asarray(x["normal"], float)
        flat = np.outer(uv[:, 0], X) + np.outer(uv[:, 1], np.cross(n, X))
        return o + flat[None] + (f * x["length"])[:, None, None] * n
    a = np.asarray(x["axis"], float)
    th = np.radians(f * x["angle"])
    d = np.cos(th)[:, None] * X + np.sin(th)[:, None] * np.cross(a, X)
    return o + uv[None, :, 1, None] * a + uv[None, :, 0, None] * d[:, None, :]


def _runs(ok, circular):
    """(first, past-last) station index of each run of True; a run may wrap
    past the end when the solid goes all the way round."""
    k = len(ok)
    if circular and ok[0] and ok[-1]:
        start = int(np.argmin(ok))          # the first station that disagrees
        idx = [(start + i) for i in range(k)]
    else:
        idx = list(range(k))
    runs, cur = [], None
    for i in idx:
        if ok[i % k]:
            cur = [i, i + 1] if cur is None else [cur[0], i + 1]
        elif cur is not None:
            runs.append(tuple(cur))
            cur = None
    if cur is not None:
        runs.append(tuple(cur))
    return runs


def _end(mesh, x, uv, inside, limit, share, boxed=False):
    """The fraction where the solid stops agreeing with the target, going from
    `inside`, where it agrees, towards `limit`: out a station at a time, then
    by halves; `boxed`, not out of the part's box. An end on the grid of
    stations left a step at every joint of a chain of rounds."""
    import trimesh

    add = x["mode"] == "add"
    lo, hi = np.asarray(mesh.bounds)

    def agrees(f):
        P = _stations(x, uv, [f]).reshape(-1, 3)
        if boxed and (np.any(P < lo) or np.any(P > hi)):
            return False
        sd = trimesh.proximity.signed_distance(mesh, P)
        return float(((sd > 0) if add else (sd < 0)).mean()) >= share

    step = np.sign(limit - inside) / STATIONS
    a = inside
    while abs(limit - a) > 1e-9:
        b = a + step if abs(limit - a) > abs(step) else limit
        if not agrees(b):
            break
        a = b
    else:
        return limit
    for _ in range(END_STEPS):
        m = (a + b) / 2.0
        if agrees(m):
            a = m
        else:
            b = m
    return a


def _slice(x, f0, f1):
    """The spec for fractions f0 to f1 of its length or sweep."""
    y = dict(x)
    if x["how"] == "prism":
        n = np.asarray(x["normal"], float)
        p, lo, _hi = x["along"]
        full = float(x["length"])
        y.update(origin=np.asarray(x["origin"], float) + f0 * full * n, length=(f1 - f0) * full,
                 along=(p, lo + f0 * full, lo + f1 * full))
    else:
        a, X = np.asarray(x["axis"], float), np.asarray(x["x"], float)
        full = float(x["angle"])
        th = np.radians(f0 * full)
        nx = np.cos(th) * X + np.sin(th) * np.cross(a, X)
        y.update(x=nx, normal=np.cross(nx, a), angle=(f1 - f0) * full)
    return y


def _overrun(mesh, x):
    """The spec run on past the ends of its face while the target goes on
    agreeing with it. Where two rounds of a chain meet, the corner between them
    belongs to neither face, and with each solid stopped at its own the corner
    stays as a tab."""
    if x["how"] == "prism":
        full = float(x["length"])
    else:
        full = np.radians(float(x["angle"])) * max(float(q[0]) for q in x["pts"])
        if float(x["angle"]) >= 360.0 - 1e-6:
            return x
    more = OVERRUN * float(x["w"]) / (full or 1.0)
    uv = _region(x)
    f0 = _end(mesh, x, uv, 0.0, -more, OVERRUN_SHARE, boxed=True)
    f1 = _end(mesh, x, uv, 1.0, 1.0 + more, OVERRUN_SHARE, boxed=True)
    if f0 == 0.0 and f1 == 1.0:
        return x
    if x["how"] == "revolve" and (f1 - f0) * float(x["angle"]) >= 360.0:
        return x
    return _slice(x, f0, f1)


def keep_right_side(mesh, x, share=RIGHT_SIDE):
    """The spec cut down to where the target agrees with it -- material where
    it adds, air where it cuts -- at STATIONS places along it (prism) or round
    it (revolve): [spec, ...], one per run of stations that agree, each ending
    where `share` of the profile stops agreeing.

    A face's extent is not the solid's: part of a cove's corner can be air
    (other rounds and a flat end there), and a full-length solid would stick
    out of the part."""
    import trimesh

    uv = _region(x)
    P = _stations(x, uv, (np.arange(STATIONS) + 0.5) / STATIONS)
    sd = trimesh.proximity.signed_distance(mesh, P.reshape(-1, 3)).reshape(STATIONS, -1)
    add = x["mode"] == "add"
    right = (sd > 0) if add else (sd < 0)
    ok = right.mean(axis=1) >= RIGHT_SIDE
    what = "material" if add else "air"
    wrong = 1.0 - float(right[ok].mean()) if ok.any() else 1.0
    if wrong > 0.5 * RIGHT_SIDE:
        raise BlendError(f"{x['face']}: the target is not {what} in {100 * wrong:.0f}% of where this "
                         f"{x['op']} would {'add' if add else 'cut'} -- the faces round it are not the "
                         f"corner it rounds off; check `residual` for what is really there")
    if ok.all():
        return [x]
    prism = x["how"] == "prism"
    full = float(x["length"] if prism else x["angle"])
    circular = not prism and full >= 360.0 - 1e-6
    runs = _runs(ok, circular)
    ends = []
    for i, (k0, k1) in enumerate(runs):
        before = runs[i - 1][1] / STATIONS if i else (runs[-1][1] / STATIONS - 1.0 if circular else 0.0)
        after = runs[i + 1][0] / STATIONS if i + 1 < len(runs) else (runs[0][0] / STATIONS + 1.0 if circular else 1.0)
        ends.append((_end(mesh, x, uv, (k0 + 0.5) / STATIONS, before, share),
                     _end(mesh, x, uv, (k1 - 0.5) / STATIONS, after, share)))
    out = []
    for f0, f1 in ends:
        out.append(dict(_slice(x, f0, f1), note=None))
    kept = sum(f1 - f0 for f0, f1 in ends) * full
    unit = "mm" if prism else "deg"
    note = (f"note: {x['face']}: the target is not {what} all along this {x['op']} (it ends "
            f"against other faces); the solid keeps {kept:.4g} of {full:.4g} {unit}"
            + (f", in {len(out)} pieces" if len(out) > 1 else ""))
    out[0]["note"] = "\n".join(t for t in (x.get("note"), note) if t)
    return out


def _lift(x):
    """(profile (u, v) -> world), at the solid's start."""
    o, X = np.asarray(x["origin"], float), np.asarray(x["x"], float)
    Y = np.asarray(x["axis"], float) if x["how"] == "revolve" else np.cross(np.asarray(x["normal"], float), X)
    return lambda q: o + float(q[0]) * X + float(q[1]) * Y


def _sides(x):
    """[(m, e)]: the faces the solid ends on, each by its normal m, pointing
    past the corner, and a direction e in it. Both sides of a prism; of a
    revolve the plate across its axis -- a wall round the axis is what it sweeps
    along, so it has no slanted end to meet."""
    e0, c, e1 = (np.asarray(q, float) for q in x["pts"])
    if x["how"] == "revolve":
        flat = [(e, o) for e, o in ((e0, e1), (e1, e0)) if abs(e[1] - c[1]) <= 0.1 * abs(e[0] - c[0])]
        if not flat:
            return []
        other = flat[0][1]
        return [(np.asarray(x["axis"], float) * np.sign(c[1] - other[1]), np.asarray(x["x"], float))]
    n, X = np.asarray(x["normal"], float), np.asarray(x["x"], float)
    Y = np.cross(n, X)
    out = []
    for e, other in ((e0, e1), (e1, e0)):
        t = (e - c) / (np.linalg.norm(e - c) or 1.0)
        m = np.array([-t[1], t[0]])
        s = float((c - other) @ m)
        if abs(s) > 1e-9:
            m *= np.sign(s)
            out.append((m[0] * X + m[1] * Y, n))
    return out


def _footprint(x, outline, o, e, f):
    """The solid's shadow on the plane at `o` with axes e, f, as a shapely area."""
    from shapely.geometry import MultiPoint, Point, Polygon

    lift = _lift(x)
    if x["how"] == "prism":
        n = np.asarray(x["normal"], float)
        ends = [lift(q) + d * n - o for q in outline for d in (0.0, float(x["length"]))]
        return MultiPoint([(p @ e, p @ f) for p in ends]).convex_hull
    a, X = np.asarray(x["axis"], float), np.asarray(x["x"], float)
    c = np.asarray(x["origin"], float) - o
    c = (float(c @ e), float(c @ f))
    r0, r1 = min(float(q[0]) for q in outline), max(float(q[0]) for q in outline)
    if float(x["angle"]) >= 360.0 - 1e-6:
        return Point(c).buffer(r1, 256).difference(Point(c).buffer(r0, 256))
    th = np.radians(np.linspace(0.0, float(x["angle"]), max(int(x["angle"] / 2.0), 8)))
    d = np.outer(np.cos(th), X) + np.outer(np.sin(th), np.cross(a, X))
    return Polygon([(c[0] + r * float(q @ e), c[1] + r * float(q @ f))
                    for r, seq in ((r1, d), (r0, d[::-1])) for q in seq]).buffer(0)


def _material(mesh, o, e, f, n):
    """The target's section on the plane at `o`, in (e, f), or None."""
    from shapely.ops import unary_union

    try:
        sec = mesh.section(plane_origin=o, plane_normal=n)
        if sec is None:
            return None
        T = np.eye(4)
        T[0, :3], T[1, :3], T[2, :3] = e, f, n
        T[:3, 3] = -T[:3, :3] @ o
        planar, _ = sec.to_2D(to_2D=T, check=False)
        polys = [p for p in planar.polygons_full if p is not None and p.area > 0]
    except (ValueError, IndexError):
        return None
    if not polys:
        return None
    region = unary_union(polys)
    return region if region.is_valid else region.buffer(0)


def _loops(area, tol):
    """[[(u, v), ...], ...]: each ring of a shapely area, thinned to `tol`."""
    polys = [area] if area.geom_type == "Polygon" else [g for g in area.geoms if g.geom_type == "Polygon"]
    out = []
    for p in polys:
        q = p.simplify(tol)
        if q.geom_type == "Polygon" and not q.is_empty:
            out.extend([tuple(map(float, xy)) for xy in list(r.coords)[:-1]] for r in (q.exterior, *q.interiors))
    return out


def clip(mesh, x):
    """The spec with `clips`, the prisms its solid is cut down by where it runs
    past the plate or wall it ends on; unchanged when it does not. See the
    module's docstring."""
    outline = _outline(*x["pts"], x.get("run_on"))
    past = CLIP_PAST * float(x["w"])
    run_on = RUN_ON * float(np.linalg.norm(np.asarray(x["pts"][2], float) - np.asarray(x["pts"][0], float)))
    lift = _lift(x)
    corner = lift(x["pts"][1])
    clips = []
    for m, e in _sides(x):
        n = -m / np.linalg.norm(m)
        e = e - (e @ n) * n
        e = e / np.linalg.norm(e)
        f = np.cross(n, e)
        o = corner + past * m
        foot = _footprint(x, outline, o, e, f)
        solid = _material(mesh, o, e, f, n)
        if solid is None or foot.is_empty:
            continue
        keep = foot.intersection(solid) if x["mode"] == "add" else foot.difference(solid)
        lost = foot.difference(keep).buffer(-CLIP_NOISE).buffer(CLIP_NOISE)
        if lost.is_empty or keep.area < CLIP_KEEP * foot.area:
            continue
        # Cut wider than the footprint: its arcs are chords, and a clip drawn
        # along them leaves a thin skin of the solid standing at the full length
        # (past the box, and not watertight).
        wide = foot.buffer(2.0 * run_on, join_style=2)
        wide = wide.intersection(solid) if x["mode"] == "add" else wide.difference(solid)
        loops = _loops(wide.buffer(-CLIP_NOISE).buffer(CLIP_NOISE), 0.02)
        if not loops or sum(map(len, loops)) > CLIP_POINTS:
            continue
        clips.append({"origin": o - 2.0 * run_on * n, "x": e, "normal": n, "loops": loops,
                      "depth": max(float((lift(q) - o) @ n) for q in outline) + 3.0 * run_on,
                      "lost": float(lost.area), "area": float(foot.area)})
    if not clips:
        return x
    share = sum(c["lost"] for c in clips) / sum(c["area"] for c in clips)
    note = (f"note: {x['face']}: the part ends at a slant across this {x['op']}, so the solid is "
            f"clipped to its outline there ({100 * share:.0f}% of its footprint)")
    return dict(x, clips=clips, note="\n".join(t for t in (x.get("note"), note) if t))


def _num(x, places=4):
    # Coordinates on the emit grid (1e-4), as every other feature writes them:
    # a fitted axis 1e-6 off a face is 0 at that grid, and OCC turns 1e-6 into
    # sliver faces.
    s = f"{float(x):.{places}f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def _vec(v, places=4):
    return "(" + ", ".join(_num(x, places) for x in v) + ")"


def _dir(v):
    """A direction, exactly X, Y or Z when it is one to 1e-6."""
    v = np.asarray(v, float)
    v = v / np.linalg.norm(v)
    k = int(np.argmax(np.abs(v)))
    if abs(abs(v[k]) - 1.0) < 1e-6:
        sign = 1.0 if v[k] >= 0 else -1.0
        v = np.zeros(3)
        v[k] = sign
    return _vec(v, 6)


def _pt(p):
    return f"({_num(p[0])}, {_num(p[1])})"


def _outline(e0, c, e1, d=None):
    """E0, corner, E1 with the two sides through the corner moved out by RUN_ON
    (or `d`, see `_band_out`): away from the solid, so they end past the faces
    they lie on instead of on them -- into air for a cut, into material for an
    add, harmless both ways."""
    e0, c, e1 = (np.asarray(x, float) for x in (e0, c, e1))
    if d is None:
        d = RUN_ON * float(np.linalg.norm(e1 - e0))

    def away(p, q, other):
        t = q - p
        nrm = np.array([-t[1], t[0]]) / (np.linalg.norm(t) or 1.0)
        return nrm if float((other - p) @ nrm) < 0 else -nrm

    o0, o1 = away(e0, c, e1), away(c, e1, e0)
    return [e0, e0 + d * o0, c + d * (o0 + o1), e1 + d * o1, e1]


def _ccw(pts, mid):
    """The outline counter-clockwise (and its arc's mid point, unchanged): drawn
    clockwise, the kernel can make an inside-out solid that cuts nothing, and
    the part keeps its sharp edge: the outline's direction should not be the
    kernel's to guess."""
    ring = list(pts) + ([mid] if mid is not None else [])
    x = np.array([q[0] for q in ring])
    y = np.array([q[1] for q in ring])
    area = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return (list(pts) if area >= 0 else list(pts)[::-1]), mid


def _segs(x):
    """The spec's outline as ("L" / "A3") segment tuples, counter-clockwise."""
    if x.get("poly") is not None:
        pts, mid = _ccw(x["poly"], None)
    else:
        pts, mid = _ccw(_outline(*x["pts"], x.get("run_on")), x["mid"])
    segs = [f'("L", {_pt(a)}, {_pt(b)})' for a, b in zip(pts, pts[1:], strict=False)]
    segs.append(f'("A3", {_pt(pts[-1])}, {_pt(mid)}, {_pt(pts[0])})' if mid is not None
                else f'("L", {_pt(pts[-1])}, {_pt(pts[0])})')
    return segs


def code(specs):
    """The blend solids as a feature block body: each one's outline sketched in
    its own plane (origin, x along the edge's cross-section, normal along the
    edge), then padded along the edge (a straight edge) or revolved about the
    plane's y axis (a circular edge). A Pocket / Groove in a cut block.

    A solid with `clips` (see `clip`) is cut down to the part's outline where
    the part ends at a slant across it. One feature of a Body cannot be clipped
    alone, so those solids are made by the prelude's `_blend_clipped` and
    stored, fused, as the base of the block's own tool body (mixed); the
    others follow as sketches in it."""
    lines, clipped = [], []
    if any(x.get("clips") for x in specs):
        lines.append("_b.mixed = True      # clipped blend solids are stored, then joined")
        for i, x in enumerate(specs):
            if not x.get("clips"):
                continue
            size = _num(x["length"]) if x["how"] == "prism" else _num(x["angle"])
            cl = ", ".join(
                f"(_frame_xn({_vec(c['origin'])}, {_dir(c['x'])}, {_dir(c['normal'])}), "
                f"[{', '.join('[' + ', '.join(_pt(q) for q in r) + ']' for r in c['loops'])}], "
                f"{_num(c['depth'])})" for c in x["clips"])
            lines.append(f"_c{i} = _blend_clipped({x['how']!r}, _frame_xn({_vec(x['origin'])}, "
                         f"{_dir(x['x'])}, {_dir(x['normal'])}), [{', '.join(_segs(x))}], "
                         f"{size}, [{cl}])")
            clipped.append(f"_c{i}")
        lines.append(f"_stored(_b, {clipped[0]}"
                     + (f".fuse([{', '.join(clipped[1:])}])" if len(clipped) > 1 else "") + ")")
    for i, x in enumerate(specs):
        if x.get("clips"):
            continue
        lines.append(f"_pl = _frame_xn({_vec(x['origin'])}, {_dir(x['x'])}, "
                     f"{_dir(x['normal'])})")
        lines.append(f"_s{i} = _sketch(_b, _pl)")
        lines.append(f"_ring(_s{i}, [{', '.join(_segs(x))}])")
        lines.append(f"_pad(_b, _s{i}, {_num(x['length'])})" if x["how"] == "prism"
                     else f"_revolve(_b, _s{i}, _axis_in(_b, _pl, (0, 0), (0, 1)), "
                          f"{_num(x['angle'])})")
    return "\n".join(lines)
def describe(specs):
    """`F12 fillet R=4.15 90deg convex, X=0.0000 to X=50.0000 | ...`."""
    out = []
    for x in specs:
        a = x["axis"]
        k = int(np.argmax(np.abs(a)))
        p, lo, hi = x["along"]
        where = (f"{AXES[k]}={p[k] + lo * a[k]:.4f} to {AXES[k]}={p[k] + hi * a[k]:.4f}"
                 if abs(abs(a[k]) - 1.0) < 1e-6 else f"length {hi - lo:.4f}")
        sense = "convex" if x["mode"] == "cut" else "concave"
        also = [n for n in x.get("faces", ()) if n != x["face"]]
        with_ = (f", with {' '.join(also[:8])}{' ...' if len(also) > 8 else ''} on its {x['kind']}"
                 if also else "")
        out.append(f"{x['face']} {x['op']} {x['size']} {sense}, {where}{with_}")
    return " | ".join(out)


def candidates(mesh, seg, boxes=False):
    """[(name, op, text)] for faces that look like blends, for `faces` to point
    at; with `boxes`, [(name, op, text, mode, (lo, hi), size, faces)] -- the
    extent of the face with the faces on its surface (`same_surface`), its
    radius or width in mm, and those faces' names, for matching a leftover
    body to it.

    Small ones only, and none a hole already owns. Listing every arc tangent to
    its neighbours names large profile arcs and coil pieces as "fillets", and
    hole chamfers, which `--op hole` cuts; a list like that goes unused. Small
    is 5% of the diagonal, or BLEND_FLOOR on a small part, whose chamfers can
    be over its 5%."""
    from mesh2cad.holes import stack_names

    # A hole's own walls and mouth chamfers (concave, in a stack) are cut by
    # `--op hole`; an outer chamfer coaxial with a bore is not the hole's.
    # A cove (a concave quarter cylinder) lands in a stack too: only what goes
    # more than half way round its axis is a hole's.
    owned = {i for _n, g in stack_names(seg) for i in g
             if seg.surfaces[i].sense == "concave" and seg.surfaces[i].kind in ("cylinder", "cone")
             and seg.surfaces[i].extent.get("sweep", 360.0) > MAX_SWEEP}
    out = []
    # Tangent, or within 1.5 facets of it: bends can meet their flanges at about
    # one facet's angle, which the segmentation's own 0.75-facet test calls
    # convex.
    tangent = {}
    S = seg.surfaces
    for j in seg.joints:
        step = max(S[j.a].facet_angle, S[j.b].facet_angle)
        if j.kind == "tangent" or (step > 0 and j.angle <= 1.5 * step + 1e-6):
            tangent[j.a] = tangent.get(j.a, 0) + 1
            tangent[j.b] = tangent.get(j.b, 0) + 1
    turned = _turned(seg)
    for i, s in enumerate(seg.surfaces):
        if i in owned:
            continue
        try:
            if s.kind in ("cylinder", "torus") and tangent.get(i, 0) >= 1:
                r = s.params["radius"] if s.kind == "cylinder" else s.params["minor"]
                if r > max(FILLET_RADIUS * float(seg.diag), BLEND_FLOOR):
                    continue
                x = fillet(mesh, seg, s)
            elif s.kind in ("plane", "cone") and _chamfer_like(mesh, seg, i, s, wide=i in turned):
                x = chamfer(mesh, seg, s)
            else:
                continue
        except BlendError:
            continue
        text = f"{x['size']} {'convex' if x['mode'] == 'cut' else 'concave'}"
        if boxes:
            V = _points(mesh, *(seg.by_name(n) for n in x["faces"]))
            out.append((s.name, x["op"], text, x["mode"], (V.min(axis=0), V.max(axis=0)),
                        float(x.get("w") or 0.0), list(x["faces"])))
        else:
            out.append((s.name, x["op"], text))
    return out


def along(leftovers, found, margin=0.05):
    """[(side, centre, op, [names], {name: faces})]: the blends each leftover
    body lies along, one name per surface, with the faces on that surface the
    name's solid takes with it: each face's solid takes the whole surface, so
    naming several faces of one surface makes all but the first do nothing.

    `leftovers` is [(side, lo, hi)], optionally with the body's mean thickness
    and the names of the target faces it lies on (mesh2cad.leftover.faces_on):
    (side, lo, hi, thickness, names). `found` is `candidates(..., boxes=True)`.
    A convex blend leaves EXTRA on a sharp build (the corner it rounds off), a
    concave one MISSING (the cove); a blend belongs to a body when its extent
    lies inside the body's box, grown by `margin` of the box's size, and --
    when the body's faces are known -- when the blend face is one of them; else
    when the body is no thicker than BLEND_DEPTH times the blend.

    A body that is most of the part holds every blend in its box: shown that
    early, the hint names blends the model cannot act on yet. Thickness alone
    does not catch it on a sheet part (thin, with larger fillets inside); the
    faces do -- a body lies on a blend face only when the blend is what it is.
    Any face on the blend's surface counts: a leftover can lie on freeform
    pieces of the cone a chamfer is on."""
    out = []
    for left in leftovers:
        side, lo, hi = left[:3]
        thick = left[3] if len(left) > 3 else None
        on = set(left[4]) if len(left) > 4 and left[4] is not None else None
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        grow = margin * float(np.linalg.norm(hi - lo)) + 1e-9
        want = "cut" if side == "extra" else "add"
        hits, taken = {}, set()
        for name, op, _t, mode, (flo, fhi), *rest in found:
            if mode != want:
                continue
            w = rest[0] if rest else 0.0
            group = rest[1] if len(rest) > 1 else [name]
            if on is not None:
                # On a face of the blend's surface, and touching its extent:
                # tabs at the corners of a ring-wide cone lie inside no tab's box.
                if not on & set(group):
                    continue
                near = np.all(flo <= hi + grow) and np.all(fhi >= lo - grow)
            else:
                if thick is not None and w > 0 and thick > BLEND_DEPTH * w:
                    continue
                near = np.all(flo >= lo - grow) and np.all(fhi <= hi + grow)
            if near and frozenset(group) not in taken:
                taken.add(frozenset(group))
                hits.setdefault(op, {})[name] = len(group) - 1
        for op, names in hits.items():
            out.append((side, (lo + hi) / 2.0, op, list(names), names))
    return out


def along_text(matches, most=4):
    """Lines for `residual` and `--check`: which leftover is which blend, and
    the command that makes it -- one line per command, however many bodies it
    is for (several corner tabs can be one cone)."""
    lines, seen = [], {}
    for side, c, op, names, *more in matches:
        key = (side, op, tuple(names))
        if key in seen:
            seen[key][1] += 1
            continue
        seen[key] = [(c, more[0] if more else {}), 0]
    for (side, op, names), ((c, more), extra) in list(seen.items())[:most]:
        what = ", ".join(f"{n} (+{more[n]} faces on its surface)" if more.get(n) else n for n in names[:8])
        bodies = f" and {extra} more bod{'y' if extra == 1 else 'ies'}" if extra else ""
        lines.append(f"{side.upper()} at ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}){bodies} "
                     f"lie{'s' if not extra else ''} along {op} {what}{' ...' if len(names) > 8 else ''}: "
                     f"`sketch --op {op} --use {','.join(names[:8])} --base PROG` makes "
                     f"{'it' if not extra else 'them'} from the sharp edge")
    if len(seen) > most:
        lines.append(f"... and {len(seen) - most} more commands for leftover bodies along blends")
    return lines


def _turned(seg):
    """Convex cones on the axis of a hole: a turned end, chamfered on the lathe
    about the bore. One can be wider than the 5% a chamfer may be, so the hint
    would never name it."""
    from mesh2cad.holes import stack_names

    S = seg.surfaces
    axes = []
    for _n, g in stack_names(seg):
        a = np.asarray(S[g[0]].params["axis"], float)
        axes.append((a / np.linalg.norm(a), np.asarray(S[g[0]].params["point"], float)))
    near = 10 * ON_SURFACE * float(seg.diag)
    out = set()
    for i, s in enumerate(S):
        if s.kind != "cone" or s.sense != "convex":
            continue
        b = np.asarray(s.params["axis"], float)
        b /= np.linalg.norm(b)
        q = np.asarray(s.params["point"], float)
        for a, p in axes:
            d = q - p
            if abs(a @ b) > np.cos(np.radians(0.5)) and np.linalg.norm(d - (d @ a) * a) <= near:
                out.add(i)
                break
    return out


def short_of(target, seg, built, found, tol=SHORT_TOL):
    """[(op, name, faces on its surface, share, depth)]: convex blends of the
    target the build stops short of -- most of the blend's face lies outside
    the build, by no more than the blend is wide. The body under it was built
    from an outline the blend had already cut back: a sharp build leaves
    EXTRA along the blend instead (`along`), and one not built yet is further
    away than the blend is wide.

    A block extruded from a face its turned ends had trimmed is short by a
    lens at each end, which `residual` names by the planes round it; read as
    "missing", it gets lofted, and the cone is never cut -- there is no edge
    left to cut it from."""
    import trimesh

    labels = np.asarray(seg.labels)
    index = {s.name: i for i, s in enumerate(seg.surfaces)}
    rng = np.random.default_rng(0)
    out, seen = [], set()
    for name, op, _t, mode, _box, w, group in found:
        if mode != "cut" or frozenset(group) in seen:
            continue
        seen.add(frozenset(group))
        tris = np.flatnonzero(np.isin(labels, [index[f] for f in group if f in index]))
        if not len(tris):
            continue
        area = target.area_faces[tris]
        pick = rng.choice(tris, size=SHORT_SAMPLES, p=area / area.sum())
        bary = rng.dirichlet(np.ones(3), size=SHORT_SAMPLES)
        pts = np.einsum("ij,ijk->ik", bary, target.triangles[pick])
        sd = trimesh.proximity.signed_distance(built, pts)        # > 0 inside the build
        short = sd < -tol
        if short.mean() < SHORT_SHARE:
            continue
        depth = float(-sd.min())
        if depth > max(w, tol) * 1.2:
            continue
        if not _built_beside(target, seg, built, group, pts, max(w, tol), tol):
            continue
        out.append((op, name, len(group) - 1, float(short.mean()), depth))
    return out


def _built_beside(target, seg, built, group, pts, reach, tol, covered=0.8):
    """Whether one flat face the blend meets is built where it meets it: so the
    body is there and was cut back, rather than not built that far yet (a
    plate short of its turned end, not yet extruded to it, was not cut back)."""
    import trimesh
    from scipy.spatial import cKDTree

    S = seg.surfaces
    index = {s.name: i for i, s in enumerate(S)}
    mine = {index[f] for f in group if f in index}
    planes = {j for jt in seg.joints for a, j in ((jt.a, jt.b), (jt.b, jt.a))
              if a in mine and j not in mine and S[j].kind == "plane"}
    labels = np.asarray(seg.labels)
    near = cKDTree(pts)
    rng = np.random.default_rng(1)
    for j in sorted(planes):
        tris = np.flatnonzero(labels == j)
        area = target.area_faces[tris]
        if not len(tris) or area.sum() <= 0:
            continue
        pick = rng.choice(tris, size=4 * SHORT_SAMPLES, p=area / area.sum())
        bary = rng.dirichlet(np.ones(3), size=len(pick))
        on = np.einsum("ij,ijk->ik", bary, target.triangles[pick])
        d, _ = near.query(on)
        close = on[d <= reach]
        if len(close) < 10:
            continue
        sd = trimesh.proximity.signed_distance(built, close)
        if float(np.mean(np.abs(sd) <= tol)) >= covered:
            return True
    return False


def short_text(hits, most=4):
    """Lines for `--check` and `residual`: each convex blend the build stops short of."""
    lines = []
    for op, name, more, share, depth in hits[:most]:
        on = f" (+{more} faces on its surface)" if more else ""
        lines.append(f"MISSING under {op} {name}{on}: the build stops short of {share:.0%} of it, by up "
                     f"to {depth:.2f} mm -- that body was built from an outline the {op} had already "
                     f"cut back, so `--op {op}` has no edge to cut and nothing adds the rest back. "
                     f"Rebuild the body sharp (`sketch --face F# --sharp`, or a section clear of the "
                     f"{op}), then `sketch --op {op} --use {name} --base PROG`")
    if len(hits) > most:
        lines.append(f"... and {len(hits) - most} more blends the build stops short of")
    return lines


def _chamfer_like(mesh, seg, i, s, wide=False):
    """Narrow, and meeting two faces at a bevel's angle, both convex or both
    concave. `wide`: a turned end (`_turned`), allowed TURNED_WIDTH."""
    js = [j for j in seg.joints if i in (j.a, j.b) and j.kind in ("convex", "concave")]
    bevel = [j for j in js if CHAMFER_ANGLES[0] <= j.angle <= CHAMFER_ANGLES[1]]
    if len(bevel) < 2 or len({j.kind for j in bevel}) > 1:
        return False
    V = _points(mesh, s)
    if s.kind == "cone":
        a = np.asarray(s.params["axis"], float)
        a /= np.linalg.norm(a)
        q = V - np.asarray(s.params["point"], float)
        t = q @ a
        rho = np.linalg.norm(q - np.outer(t, a), axis=1)
        width = float(np.hypot(np.ptp(t), np.ptp(rho)))
    else:
        n = _outward(mesh, s)
        Q = V - V.mean(axis=0)
        Q -= np.outer(Q @ n, n)
        _, _, vt = np.linalg.svd(Q, full_matrices=False)
        width = float(np.ptp(Q @ vt[1])) if len(vt) > 1 else 0.0
    return width <= max((TURNED_WIDTH if wide else CHAMFER_WIDTH) * float(seg.diag), BLEND_FLOOR)
