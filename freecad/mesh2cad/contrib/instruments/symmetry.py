"""symmetry -- find the mirror planes and rotation axes, and name them.

The reference contribution. It is here to be copied: it is small enough to read
in one sitting and it does the thing the whole harness is about, which is to do
the measuring in code and hand the model a name plus the exact flag to pass.

It answers a question the agent currently has to guess at. `--split` needs a
line to cut on and `--op revolve` needs an axis, and both are typed by hand
today from a picture. Here they are measured: mirror the sampled surface through
a candidate plane and ask the mesh how many of the moved points landed back on
it within tolerance.

    m2c.py symmetry part.stl
    S0   mirror plane   X = 0.0000     agreement 1.000   --split "X=0" --keep +
    S2   rotation axis  X=0, Y=0       agreement 0.999   --axis "X=0,Y=0"

Agreement is the FRACTION OF POINTS that land within tolerance, not the mean
distance scaled by the part size: on a compact solid the mean distance is small
compared with the diagonal no matter what you do to it, so it scores rotation
axes the part does not have. A fraction-within-tolerance saturates only when
the symmetry is real.
"""

import numpy as np

from mesh2cad import registry
from mesh2cad.contract import Result
from mesh2cad.normalize import ingest

AXES = ("X", "Y", "Z")
ORDERS = (2, 3, 4, 5, 6, 8, 12)
# A candidate has to put this fraction of the surface back onto the surface
# before it is worth naming.
MIN_AGREEMENT = 0.98
# Reported as PARTIAL: most of the part obeys it, some named feature does not.
PARTIAL_AGREEMENT = 0.80
# Default tolerance as a fraction of the bounding diagonal.
TOL_FRAC = 0.002
# Sampling seed: the same part must give the same reading twice.
SEED = 20260812


def configure(p):
    p.add_argument("mesh")
    p.add_argument("--samples", type=int, default=3000,
                   help="surface points used to test each candidate")
    p.add_argument("--tol", type=float, default=None,
                   help=f"distance in mm counted as 'landed back on the "
                        f"surface'; default {TOL_FRAC:.3%} of the bbox diagonal")
    p.add_argument("--top", type=int, default=8, help="how many candidates to name")


def candidates(mesh, samples=3000, tol=None, top=8):
    """(named, near, tol, n_samples): the candidates `symmetry` lists as S0..,
    best first, and the partial ones (Q0..). `sketch --mirror S#` reads its
    plane from here, so a name means the same plane in both commands."""
    pts = _sample(mesh, samples)
    diag = float(np.linalg.norm(mesh.extents))
    tol = tol if tol is not None else TOL_FRAC * diag
    centre = _centre(mesh)
    cands = []
    for i in range(3):
        cands.append(_mirror(mesh, pts, tol, i, centre[i]))
    for i in range(3):
        best = _rotation(mesh, pts, tol, i, centre)
        if best is not None:
            cands.append(best)
    near = [c for c in cands
            if PARTIAL_AGREEMENT <= c["agreement"] < MIN_AGREEMENT]
    cands = [c for c in cands if c["agreement"] >= MIN_AGREEMENT]
    cands.sort(key=lambda c: -c["agreement"])
    near.sort(key=lambda c: -c["agreement"])
    return cands[:top], near, tol, len(pts)


def run(args):
    mesh, info = ingest(args.mesh)
    diag = float(np.linalg.norm(mesh.extents))
    cands, near, tol, n_pts = candidates(mesh, args.samples, args.tol, args.top)

    r = Result()
    r.line(f"symmetry of {args.mesh}")
    r.line(f"bbox diagonal {diag:.4f} mm; {n_pts} surface samples; "
           f"tolerance {tol:.4f} mm")
    r.line("agreement = fraction of sampled points that land back on the "
           "surface within tolerance after the transform")
    r.line()

    if not cands and not near:
        r.line("no mirror plane or rotation axis above the reporting threshold "
               f"of {MIN_AGREEMENT:.2f}. Treat the part as asymmetric: choose "
               "the section plane from `survey` instead.")
        return r

    r.line("  name  kind            about                agreement  pass this")
    for k, c in enumerate(cands):
        name = f"S{k}"
        r.line(f"  {name:<5} {c['kind']:<15} {c['about']:<20} "
               f"{c['agreement']:9.4f}  {c['flag']}")
        if c.get("detail"):
            r.line(f"        {c['detail']}")
        r.label(name, f"{c['kind']} {c['about']} (agreement {c['agreement']:.4f})")

    if near:
        # A single feature can break a symmetry the rest of the part obeys -- a
        # five-spoke wheel with one handle is 5-fold everywhere except the handle.
        # Reporting only pass/fail throws away both halves of that: that the body
        # IS n-fold, and that exactly one thing is not. Build the symmetric body
        # from the pattern, then add the exception.
        r.line()
        r.line("PARTIAL -- most of the surface obeys these, some feature does not:")
        for k, c in enumerate(near):
            name = f"Q{k}"
            miss = 100.0 * (1.0 - c["agreement"])
            r.line(f"  {name:<5} {c['kind']:<15} {c['about']:<20} "
                   f"{c['agreement']:9.4f}  {c['flag']}")
            r.line(f"        {c.get('detail', '')}"
                   f"   {miss:.1f}% of the surface disagrees -- that is the "
                   f"feature breaking it")
            r.label(name, f"partial {c['kind']} {c['about']} "
                          f"(agreement {c['agreement']:.4f})")

    if not cands:
        return r

    r.line()
    # Only a genuinely high-order axis settles the route. A bare order-2 axis is
    # a free consequence of two perpendicular mirror planes -- every rectangular
    # part has one -- and recommending a revolve on the strength of it would send
    # a block of flat fins down the revolve path.
    def tier(c):
        if c.get("full"):
            return 3                      # a real body of revolution
        if c["kind"] == "rotation axis" and c.get("n_orders", 0) > 1:
            return 2                      # an n-fold pattern axis
        if c["kind"] == "mirror plane":
            return 1
        return 0                          # a bare 2-fold axis: says nothing

    best = max(cands, key=lambda c: (c["agreement"], tier(c),
                                     c.get("n_orders", 0)))
    name = _name_of(cands, best)
    if best.get("full"):
        r.line(f"{name} is the one to act on: section on a plane that CONTAINS "
               f"that axis and build with --op revolve. Sectioning across it "
               f"gives a cross-section that changes at every step, which is why "
               f"`survey` will report near-zero prismaticity.")
    elif best["kind"] == "rotation axis":
        fold = max(best["orders"])
        article = "an" if str(fold)[0] in "8" else "a"
        r.line(f"{name} is {article} {fold}-fold axis, not a body of "
               f"revolution: expect a circular pattern of features about it "
               f"rather than one revolved profile. Build the prism first and "
               f"let `section` group the pattern.")
    else:
        r.line(f"{name} is the one to act on: it is the natural line to pass to "
               f"--split, which halves the profile and closes it along the cut.")
        if any(c["kind"] == "rotation axis" and c.get("orders") == (2,)
               for c in cands):
            r.line("The order-2 axis above follows from the two mirror planes "
                   "and carries no extra information; do not read it as a "
                   "revolve.")
    if info["regime"]["kind"] == "revolve" and not any(
            c["kind"] == "rotation axis" for c in cands):
        r.line("NOTE: `overview` called this a revolve but no rotation axis "
               "cleared the threshold here. Trust this one -- it is measured "
               "against the surface, not inferred from face normals.")
    return r


def _name_of(cands, c):
    return f"S{cands.index(c)}"


def _sample(mesh, n, seed=SEED):
    """Surface samples, seeded.

    Unseeded this passed the determinism test anyway, because on these parts
    every agreement came out exactly 1.0000 and the printed text did not move.
    That is luck, not determinism: the first part with a fractional score would
    have made the suite flaky and the agent's readings unrepeatable.
    """
    import trimesh
    try:
        pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    except TypeError:                       # older trimesh: no seed argument
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            pts, _ = trimesh.sample.sample_surface(mesh, n)
        finally:
            np.random.set_state(state)
    return np.asarray(pts, float)


def _centre(mesh):
    lo, hi = mesh.bounds
    return (np.asarray(lo, float) + np.asarray(hi, float)) / 2.0


def _agreement(mesh, moved, tol):
    """Fraction of moved points that land back on the surface within tol.

    Distance to the surface rather than to the original samples: the samples are
    a random draw, so a nearest-sample metric would report the sampling density
    as asymmetry.
    """
    _, dist, _ = mesh.nearest.on_surface(moved)
    return float(np.mean(np.asarray(dist) <= tol))


def _mirror(mesh, pts, tol, axis, coord):
    moved = pts.copy()
    moved[:, axis] = 2.0 * coord - moved[:, axis]
    return {
        "kind": "mirror plane",
        "axis": axis,
        "coord": float(coord),
        "about": f"{AXES[axis]} = {coord:.4f}",
        "agreement": _agreement(mesh, moved, tol),
        "flag": f'--split "{AXES[axis]}={coord:.4f}" --keep +',
    }


def _standout_order(scores):
    """The order that is a local PEAK against the small-angle trend.

    Agreement rises monotonically as the rotation angle shrinks -- a 30-degree
    turn displaces points less than a 72-degree one, so more land back inside
    the tolerance whether or not the symmetry exists. Both "highest score" and
    "largest margin over the median" therefore pick order 12 on almost any part,
    and refining the axis against a symmetry the part does not have drags the
    axis off the one it does.

    A real symmetry is not merely high, it is high *for its angle*: it stands
    above its immediate neighbours in angle. On a five-spoke wheel the scores
    might read 0.41 at 90deg, 0.61 at 72deg, 0.52 at 60deg -- five is a peak,
    twelve is just the end of the ramp.
    """
    order = sorted(scores, key=lambda k: -2 * np.pi / k)      # by angle, descending
    best, best_resid = order[0], -1.0
    for i, k in enumerate(order):
        neigh = [scores[order[j]] for j in (i - 1, i + 1) if 0 <= j < len(order)]
        resid = scores[k] - float(np.mean(neigh))
        if resid > best_resid:
            best, best_resid = k, resid
    return best


def _refine_axis(mesh, pts, tol, axis, centre, order):
    """Search for where the rotation axis actually is.

    The bbox centre is only the axis when the part is already symmetric, which
    is the very thing being measured. One asymmetric feature moves it, by enough
    at a tight tolerance to hide a real symmetry completely. A few hundred cheap
    evaluations find it.
    """
    u, v = [i for i in range(3) if i != axis]
    ext = mesh.extents
    best = np.array(centre, float)
    best_score = _agreement(mesh, _rotate(pts, axis, best, 2 * np.pi / order), tol)
    for reach in (0.02, 0.004):
        step_u, step_v = reach * ext[u], reach * ext[v]
        improved = True
        while improved:
            improved = False
            for du in (-step_u, 0.0, step_u):
                for dv in (-step_v, 0.0, step_v):
                    if du == 0.0 and dv == 0.0:
                        continue
                    cand = best.copy()
                    cand[u] += du
                    cand[v] += dv
                    sc = _agreement(
                        mesh, _rotate(pts, axis, cand, 2 * np.pi / order), tol)
                    if sc > best_score + 1e-6:
                        best, best_score, improved = cand, sc, True
    return best


def _rotation(mesh, pts, tol, axis, centre):
    """Every rotation order that maps the surface back onto itself.

    A part where all of them pass is a body of revolution, for which the whole
    prism workflow is the wrong question. Where only some pass, the set is the
    answer: an 8-tooth wheel reports 2,4,8 and only the 8 is about its teeth.
    """
    # Locate the axis before scoring it, on a cheap subsample.
    #
    # Which order to refine against is not obvious, and the obvious choice is
    # wrong: agreement is biased towards HIGH orders, because a 30-degree
    # rotation displaces points less than a 72-degree one and more of them land
    # back inside the tolerance whether or not the symmetry is real. Picking the
    # top-scoring order therefore picks 12 almost every time, and refining
    # against a symmetry the part does not have drags the axis away from the one
    # it does. Pick by MARGIN over the other orders instead -- a real 5-fold
    # stands out from its neighbours even when measured about the wrong axis.
    probe = pts[::max(1, len(pts) // 600)]
    trial = {k: _agreement(mesh, _rotate(probe, axis, centre, 2 * np.pi / k), tol)
             for k in ORDERS}
    centre = _refine_axis(mesh, probe, tol, axis, centre, _standout_order(trial))

    passing = []
    scores = {}
    for k in ORDERS:
        moved = _rotate(pts, axis, centre, 2.0 * np.pi / k)
        a = _agreement(mesh, moved, tol)
        scores[k] = a
        if a >= MIN_AGREEMENT:
            passing.append(k)
    if not passing:
        # Nothing cleared the bar -- but "the body is 5-fold apart from one
        # feature" is a useful sentence, and returning None here would throw it
        # away. Hand back the best order so the caller can decide whether it is a
        # near miss worth reporting.
        best_k = _standout_order(scores)
        other = [i for i in range(3) if i != axis]
        about = ", ".join(f"{AXES[i]}={centre[i]:.4f}" for i in other)
        return {
            "kind": "rotation axis",
            "axis": axis,
            "centre": [float(x) for x in centre],
            "about": about,
            "agreement": scores[best_k],
            "flag": f'--axis "{about.replace(", ", ",")}"',
            "detail": f"best order {best_k}",
            "n_orders": 0,
            "orders": (best_k,),
            "full": False,
        }

    other = [i for i in range(3) if i != axis]
    about = ", ".join(f"{AXES[i]}={centre[i]:.4f}" for i in other)
    full = len(passing) == len(ORDERS)
    if full:
        note = "every order tested -- body of revolution"
    elif passing == [2]:
        # Two perpendicular mirror planes always meet in a 2-fold axis, so on
        # its own this says nothing new and certainly does not mean "revolve".
        note = "order 2 only -- implied by the mirror planes, not a revolve axis"
    else:
        # All of them, not just the smallest: on a toothed wheel the 2 is
        # incidental and the 8 is the tooth count, which is the number worth knowing.
        note = "orders " + ",".join(str(k) for k in passing)
    return {
        "kind": "rotation axis",
        "axis": axis,
        "centre": [float(x) for x in centre],
        "about": about,
        "agreement": min(scores[k] for k in passing),
        "flag": f'--axis "{about.replace(", ", ",")}"',
        "detail": note,
        "n_orders": len(passing),
        "orders": tuple(passing),
        "full": full,
    }


def _rotate(pts, axis, centre, angle):
    u, v = [i for i in range(3) if i != axis]
    p = pts - centre
    c, s = np.cos(angle), np.sin(angle)
    out = p.copy()
    out[:, u] = c * p[:, u] - s * p[:, v]
    out[:, v] = s * p[:, u] + c * p[:, v]
    return out + centre


registry.register(
    name="symmetry",
    kind="instrument",
    summary="name the mirror planes and rotation axes, with the --split / --axis "
            "flag to pass for each",
    author="Dmitry Zhemchuzhnikov",
    module=__name__,
    configure=configure,
    run=run,
)
