"""Levels from faces: where along an axis something starts or stops, read off
the surfaces that make it start or stop.

The sweep finds a level where the AREA of a cut changes. That is a symptom, and
it has other causes than an operation ending: a cylinder lying across the axis
changes the area at every height it spans, and on one bracket that produced
fourteen of eighteen levels along Z -- none of them a place anything stops.
Four were real, and they are exactly the heights of the four flat faces
perpendicular to Z.

So a level here is a FACE, not a symptom:

    cap         a flat face perpendicular to the axis. Its height is exact --
                every vertex of it is on one plane -- and which way it faces
                says which side the material is on.
    wall end    where a cylinder running along the axis stops, flat.
    taper end   where a cone, a sloped plane, a torus, a sphere band or a
                revolve stops: a chamfer, a draft, a fillet, a ball's join, the
                end of a spline turned round the axis. The section is changing
                between the two ends of such a face and constant outside them,
                which is what the sweep's `K` levels guessed at.
    equator     the widest ring of a torus, a sphere or a revolve, when the
                face spans it; for a revolve whose profile dips inside it, the
                narrowest ring (a waist). A revolve's profile is known only at
                its rings of vertices, so its extreme is a ring, uncertain by
                half the spacing to the next.

and a surface that runs ACROSS the axis -- a cylinder lying on its side, a
spline face -- gives it no level at all. The heights it spans are where the
sweep saw area changing for no operation's sake; they are returned as ramps so
the caller can say which of the sweep's levels they explain.

Evidence, not verdicts (A2): every level carries the faces it came from, and
nothing here says which to build to.
"""

import numpy as np

AXES = "XYZ"
# Strongest first. A flat face is where an extrusion ends. Where a cone, a
# torus, a sphere band or a sloped face begins, the section starts changing --
# and that says more than the cylinder wall that happens to stop at the same
# height: a post's wall ends exactly where its chamfer starts, and the level is
# the chamfer's. A wall's end alone is still a place something stops.
ROLES = ("cap", "cone end", "blend end", "sphere end", "revolve end", "slope end",
         "wall end", "equator", "waist")
KIND_OF = {"cap": "boundary", "wall end": "boundary", "cone end": "taper",
           "blend end": "taper", "sphere end": "taper", "revolve end": "taper",
           "slope end": "taper", "equator": "turning", "waist": "turning"}


def _boundary_vertices(mesh, faces):
    e = np.sort(mesh.faces[faces][:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    uniq, cnt = np.unique(e, axis=0, return_counts=True)
    return np.unique(uniq[cnt == 1])


def candidates(seg, mesh, k):
    """Every place along world axis `k` that a face says something stops.

    Returns (cands, ramps, across): candidate levels, the (lo, hi, surface)
    intervals over which some face bends through the cut, and the surfaces that
    run across the axis and so give it no level.
    """
    V = np.asarray(mesh.vertices, float)
    tol = seg.tol
    cands, ramps, across = [], [], []
    for i, s in enumerate(seg.surfaces):
        vid = np.unique(mesh.faces[s.faces])
        z = V[vid, k]
        lo, hi = float(z.min()), float(z.max())
        bb = s.extent.get("bbox")
        L = (float(np.linalg.norm(np.asarray(bb[1]) - np.asarray(bb[0])))
             if bb is not None else seg.diag) or seg.diag
        # How well this face's direction is known: its vertices sit within
        # `residual` of it over an extent L. Parallel means parallel to that.
        ang = max(3.0 * max(s.residual, tol) / L, 1e-9)
        zb = V[_boundary_vertices(mesh, s.faces), k]

        def flat(v, need, zb=zb):
            """Does the face END flat at height v -- several boundary vertices
            on that one height -- or only touch it at a point, as a boss's
            saddle-shaped foot on a cross cylinder does?"""
            return int(np.sum(np.abs(zb - v) <= 4.0 * tol)) >= need

        def add(value, role, unc=0.0, i=i, **info):
            cands.append({"value": float(value), "role": role, "surface": i,
                          "unc": float(unc), **info})

        if s.kind == "plane":
            n = np.asarray(s.params["normal"], float)
            c = abs(float(n[k]))
            if c >= np.cos(ang):
                add(float(z.mean()), "cap", unc=(hi - lo) / 2.0,
                    facing="+" if n[k] > 0 else "-")
            elif c <= np.sin(ang):
                continue      # a wall: along the axis, its ends are the caps
            else:
                tilt = float(np.degrees(np.arccos(min(1.0, c))))
                for v in (lo, hi):
                    if flat(v, 2):
                        add(v, "slope end", tilt=tilt)
                ramps.append((lo, hi, i))
        elif s.kind == "sphere":
            for v in (lo, hi):
                if flat(v, 3):
                    add(v, "sphere end")
            ck = float(s.params["centre"][k])
            if lo + 4 * tol < ck < hi - 4 * tol:
                add(ck, "equator")
            ramps.append((lo, hi, i))
        elif s.kind in ("cylinder", "cone", "torus"):
            a = np.asarray(s.params["axis"], float)
            if abs(float(a[k])) < np.cos(ang):
                across.append(i)
                ramps.append((lo, hi, i))
                continue
            role = {"cylinder": "wall end", "cone": "cone end",
                    "torus": "blend end"}[s.kind]
            for v in (lo, hi):
                if flat(v, 3):
                    add(v, role)
            if s.kind == "torus":
                ck = float(s.params["point"][k])
                if lo + 4 * tol < ck < hi - 4 * tol:
                    add(ck, "equator")
            if s.kind != "cylinder":
                ramps.append((lo, hi, i))
        elif s.kind == "revolve":
            a = np.asarray(s.params["axis"], float)
            if abs(float(a[k])) < np.cos(ang):
                across.append(i)
                ramps.append((lo, hi, i))
                continue
            for v in (lo, hi):
                if flat(v, 3):
                    add(v, "revolve end")
            for h, unc, role in _profile_extremes(s.params, k):
                if lo + 4 * tol < h < hi - 4 * tol:
                    add(h, role, unc=unc)
            ramps.append((lo, hi, i))
        else:
            across.append(i)
            ramps.append((lo, hi, i))
    return cands, ramps, across


def _profile_extremes(params, k):
    """A revolve's widest and narrowest rings that lie INSIDE the profile --
    not at an end, where the face simply stops. Returns (height along world
    axis k, uncertainty, role) for each. The profile is known only at its rings,
    so the true extreme is within half a ring spacing of the ring found."""
    prof = params.get("profile") or []
    if len(prof) < 3:
        return []
    t = np.array([p[0] for p in prof], float)
    r = np.array([p[1] for p in prof], float)
    a = np.asarray(params["axis"], float)
    c = np.asarray(params["point"], float)
    out = []
    for j, role in ((int(np.argmax(r)), "equator"), (int(np.argmin(r)), "waist")):
        if 0 < j < len(prof) - 1:
            unc = max(t[j + 1] - t[j], t[j] - t[j - 1]) / 2.0 * abs(float(a[k]))
            out.append((float((c + t[j] * a)[k]), float(unc), role))
    return out


def anchored_slopes(cands, seg):
    """Keep a sloped face's ends only if the slope starts or stops at a level
    something else puts there.

    A chamfer starts at the cap it bevels; a draft starts where its wall meets
    a floor. But a gear cut along Z has thirty-two flat tooth flanks, and seen
    along X each is a sloped face whose two ends are nothing but corners of the
    tooth profile -- the flanks meet only each other. They turned 24 sweep
    levels along X into 34. A slope anchored at one end to a cap, a wall's end
    or a round face's end is a feature of this axis; one that touches only
    other slopes belongs to some other axis's profile.

    The part's own ends do not anchor: a star's tip touches the end of the part
    at a single corner. A flat end there is a cap, already a candidate.
    Returns (kept candidates, number of sloped faces set aside).
    """
    tol = 8.0 * seg.tol
    anchors = np.asarray([c["value"] for c in cands if c["role"] != "slope end"],
                         float)
    ends = {}
    for c in cands:
        if c["role"] == "slope end":
            ends.setdefault(c["surface"], []).append(c["value"])
    loose = {i for i, vs in ends.items()
             if not any(np.any(np.abs(anchors - v) <= tol) for v in vs)}
    kept = [c for c in cands
            if not (c["role"] == "slope end" and c["surface"] in loose)]
    return kept, len(loose)


def merge(cands, seg, k):
    """Candidates at one height are one level. Returns level dicts ready to be
    named: value, kind, label, uncertainty, and the faces behind them."""
    if not cands:
        return []
    tol = 8.0 * seg.tol
    cands = sorted(cands, key=lambda c: c["value"])
    groups, cur = [], [cands[0]]
    for c in cands[1:]:
        if c["value"] - cur[-1]["value"] <= tol + c["unc"] + cur[-1]["unc"]:
            cur.append(c)
        else:
            groups.append(cur)
            cur = [c]
    groups.append(cur)
    out = []
    for g in groups:
        caps = [c for c in g if c["role"] == "cap"]
        src = caps or g
        value = float(np.mean([c["value"] for c in src]))
        unc = max(max(c["unc"] for c in src),
                  float(np.ptp([c["value"] for c in src])) / 2.0)
        best = min(g, key=lambda c: ROLES.index(c["role"]))
        names = sorted({seg.surfaces[c["surface"]].name for c in g},
                       key=lambda n: int(n[1:]))
        out.append({"value": value, "kind": KIND_OF[best["role"]],
                    "label": _label(g, seg, k), "uncertainty": unc,
                    "faces": names, "roles": sorted({c["role"] for c in g},
                                                    key=ROLES.index),
                    "surfaces": sorted({c["surface"] for c in g}),
                    "source": "faces", "refined": True})
    return out


CURVED = ("cone", "torus", "sphere", "revolve")
# What a smooth profile is made of: round faces, and the spline pieces between
# them that no analytic surface fitted.
CHAIN = CURVED + ("freeform",)


def seams(levels, seg, mesh, k):
    """Split off the levels that are seams inside one smooth curved profile.

    A spline profile of revolution comes back from the segmentation as a chain
    of cones, tori and spheres on one axis, with here and there a piece that
    fits nothing, each running tangentially into the next -- the right pieces,
    since each fits its stretch to every vertex. But every join between them is
    then an "end", and one such part had 28 of them along its axis where the
    sweep alone had 9. A join where one curved face runs smoothly into the next
    is not a place anything starts or stops.

    Only curved-to-curved, and only strictly inside the chain: where a torus
    meets a cylinder or a plane a fillet begins, and where the chain itself
    ends the profile does. Those levels stay.
    Returns (kept, hidden, names of the faces at the hidden seams).
    """
    from mesh2cad.segment import coaxial_groups

    S = seg.surfaces
    parent = list(range(len(S)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    same_axis = {}
    for g in coaxial_groups(seg):
        for i in g:
            same_axis[i] = g[0]
    for j in seg.joints:
        a, b = S[j.a].kind, S[j.b].kind
        if j.kind != "tangent" or a not in CHAIN or b not in CHAIN:
            continue
        if a in CURVED and b in CURVED and (
                j.a not in same_axis or same_axis.get(j.b) != same_axis[j.a]):
            continue
        parent[find(j.a)] = find(j.b)

    V = np.asarray(mesh.vertices, float)
    span = {}
    for i, s in enumerate(S):
        if s.kind in CHAIN:
            z = V[np.unique(mesh.faces[s.faces]), k]
            r = find(i)
            lo, hi = span.get(r, (np.inf, -np.inf))
            span[r] = (min(lo, float(z.min())), max(hi, float(z.max())))

    kept, hidden, names = [], 0, set()
    tol = 8.0 * seg.tol
    for m in levels:
        surf = m.get("surfaces") or []
        roots = {find(i) for i in surf}
        seam = (bool(surf) and all(S[i].kind in CURVED for i in surf)
                and len(roots) == 1 and m["kind"] != "turning")
        if seam:
            lo, hi = span[next(iter(roots))]
            seam = lo + tol < m["value"] < hi - tol
        if seam:
            hidden += 1
            names.update(S[i].name for i in surf)
        else:
            kept.append(m)
    return kept, hidden, sorted(names, key=lambda n: int(n[1:]))


def _label(group, seg, k):
    """One line saying what the faces at this height are, strongest first."""
    ax = AXES[k]
    by_role = {}
    for c in group:
        by_role.setdefault(c["role"], []).append(c)
    parts = []
    for role in ROLES:
        cs = by_role.get(role)
        if not cs:
            continue
        names = ", ".join(sorted({seg.surfaces[c["surface"]].name for c in cs},
                                 key=lambda n: int(n[1:])))
        if role == "cap":
            facing = {c["facing"] for c in cs}
            # Which side the solid is on, in the words the axis allows: along Z
            # that is above and below, along X or Y it is a signed side.
            below = "below it" if ax == "Z" else f"on its -{ax} side"
            above = "above it" if ax == "Z" else f"on its +{ax} side"
            if facing == {"+"}:
                side = f"facing +{ax}, material {below}"
            elif facing == {"-"}:
                side = f"facing -{ax}, material {above}"
            else:
                side = f"facing both ways along {ax}"
            parts.append(f"flat face {names}, {side}")
        elif role == "wall end":
            parts.append(f"end of cylinder {names}")
        elif role == "cone end":
            parts.append(f"taper end: cone {names}")
        elif role == "blend end":
            parts.append(f"blend end: torus {names}")
        elif role == "sphere end":
            parts.append(f"end of sphere {names}")
        elif role == "revolve end":
            parts.append(f"end of revolve {names}")
        elif role == "slope end":
            tilt = cs[0].get("tilt", 0.0)
            parts.append(f"end of sloped face {names} ({tilt:.3f}deg off {ax})")
        elif role == "waist":
            parts.append(f"narrowest ring of {names}")
        else:
            parts.append(f"widest ring of {names}")
    more = len(parts) - 2
    return "; ".join(parts[:2]) + (f" (+{more} more)" if more > 0 else "")


def explained_by(value, ramps, reach):
    """The surfaces that bend through the cut at this height, if any."""
    return [i for lo, hi, i in ramps if lo - reach <= value <= hi + reach]
