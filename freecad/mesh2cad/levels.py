"""Levels the sweep could not name: local mirror planes, and turning points.

The sweep detects **value jumps** and nothing else. Everything else it reported
was a `plateau edge` -- not a measurement but a greedy scan's opinion about
where a run stopped drifting. Three kinds of level need something better:

- the centre of a torus, which is a local **mirror plane** of the sweep and
  equally the **turning point** of its width;
- the start and stop of a taper (a chamfered foot, tapered fin tips).

This module does the first two. A taper's ends are a slope discontinuity and
are still open.

Why mirrors are the primary tool here and turning points the secondary one:
locating the centre of symmetry of a profile is a *correlation* problem and is
well conditioned -- the mismatch grows linearly as you step off the true
centre. Locating an extremum is quadratic: near a torus's equator the width
barely changes over the first tenth of a millimetre, so reading the peak
position to 0.01 mm means resolving the width to a part in 10^6. Both are
implemented, both are refined against real sections rather than against the
sweep grid, and each reports what it actually achieved.
"""

import numpy as np

from mesh2cad.section import section_at

# A mirror claimed on fewer than this many samples either side is noise, not a
# symmetry. Six is enough that a spurious match has to hold twelve independent
# comparisons.
MIRROR_MIN_HALF = 6

# Allowed mismatch between the two sides, as a fraction of the profile's own
# full range. The signature is normalised to [0, 1] over the whole sweep first,
# so this is scale-free -- a 300 mm flange and a 3 mm grub screw use the same
# number.
MIRROR_TOL = 0.01

# The window has to contain real geometry. Inside a plain cylinder EVERY offset
# is trivially a mirror centre, and without this the detector reports the whole
# sweep. Expressed in the same normalised units.
MIRROR_MIN_VARIATION = 0.05

# How far a turning point must stand out from its neighbourhood to count, again
# as a fraction of the signal's full range.
EXTREMUM_MIN_PROMINENCE = 0.01
EXTREMUM_MIN_HALF = 4

REFINE_PRECISION = 1e-3
REFINE_MAX_ITERS = 40

# --- taper ends -------------------------------------------------------------
#
# How much of the slope change has to survive when the fitting window is
# HALVED. This is the whole discriminator. A real kink is a jump in the
# derivative, so it reads the same at any scale; ordinary curvature spreads its
# bend across the window, so halving the window halves the apparent change. A
# genuine kink scores near 1.0, a smooth curve near 0.5.
TAPER_PERSISTENCE = 0.75
# And the change has to be large next to the steepest slope in its own segment,
# so that numerical noise on a flat run cannot qualify. Real kinks score close
# to 1.00 here; the tessellation wobble a smooth dome or capsule produces scores
# around 0.2. There is a wide gap between those and this sits in it.
TAPER_MIN_SLOPE_CHANGE = 0.35
# Window sizes to try, largest first. A kink can sit within a few samples of a
# value jump, and with only the largest window there is no room to fit a line
# on that side, so the taper it exists to find would be missed entirely.
TAPER_WINDOWS = ((8, 2), (5, 1), (4, 1))


def _signature(slices):
    """Per-slice signature: area, perimeter, and both extents.

    Four channels rather than one because they fail independently. A torus's
    equator is nearly invisible in area and obvious
    in width; a part whose section keeps its size while changing shape is the
    other way round. Each channel is scaled to its own range so none dominates,
    and a channel that never varies contributes nothing instead of dividing by
    zero.
    """
    cols = []
    for get in (lambda s: s.area, lambda s: s.perimeter,
                lambda s: s.extent_u, lambda s: s.extent_v):
        v = np.array([get(s) for s in slices], float)
        rng = float(v.max() - v.min())
        cols.append((v - v.min()) / rng if rng > 1e-12 else np.zeros_like(v))
    return np.column_stack(cols)


def mirror_levels(slices, tol=MIRROR_TOL, min_half=MIRROR_MIN_HALF,
                  min_variation=MIRROR_MIN_VARIATION):
    """Offsets about which the sweep is locally mirror-symmetric.

    For every candidate centre the window is grown outward while the two sides
    keep agreeing, and the centre is kept only if it reached `min_half` samples
    AND the window it reached contains real variation. Both halves of that test
    are load-bearing:

    - without the growth, a mirror is only ever found at the scale you guessed;
    - without the variation floor, a cylinder reports every offset it has.

    The growth is what makes these LOCAL. A ring on a stem is symmetric about its
    equator but the part is not -- the stem ruins it below the ring -- so the
    window grows to the ring's size and stops, which is the honest extent of the
    claim and is reported as `half_width`.
    """
    n = len(slices)
    if n < 2 * min_half + 3:
        return []
    z = np.array([s.offset for s in slices], float)
    sig = _signature(slices)
    topo = np.array([(s.n_loops, s.n_holes) for s in slices], int)

    # Centres are tried on the HALF grid: at each sample, and midway between
    # each pair. A mirror plane has no reason to land on a sample, and when it
    # does not, comparing about the nearest sample is comparing about a centre
    # half a step off -- more than enough to break the match on a part with
    # sharp features, and restricting centres to samples misses real mirror
    # planes that fall between two samples.
    #
    # `at` indexes the half grid: at = 2*i is the sample i, at = 2*i+1 is the
    # midpoint of samples i and i+1. A mirror about a sample compares i-j with
    # i+j; about a midpoint it compares i-j with i+1+j.
    step = float(z[1] - z[0])
    n_half = 2 * n - 1

    def _pair(at, j):
        if at % 2 == 0:
            i = at // 2
            return i - j, i + j
        i = at // 2
        return i - j, i + 1 + j

    def _centre(at):
        return float(z[at // 2]) if at % 2 == 0 else float(z[at // 2] + step / 2.0)

    reach = np.zeros(n_half, int)
    for at in range(n_half):
        k = 0
        while True:
            a, b = _pair(at, k + 1)
            if a < 0 or b >= n:
                break
            if not np.array_equal(topo[a], topo[b]):
                break
            if np.abs(sig[a] - sig[b]).max() > tol:
                break
            k += 1
        reach[at] = k

    out = []
    for at in range(n_half):
        k = reach[at]
        if k < min_half:
            continue
        # Only the best centre of each run: neighbours of a true mirror all
        # score well, and reporting eleven names for one plane is worse than
        # reporting none.
        lo = max(0, at - 2 * min_half)
        hi = min(n_half, at + 2 * min_half + 1)
        if reach[at] < reach[lo:hi].max():
            continue
        if at != int(lo + np.argmax(reach[lo:hi])):
            continue
        a, b = _pair(at, k)
        win = sig[a:b + 1]
        variation = float((win.max(axis=0) - win.min(axis=0)).max())
        if variation < min_variation:
            continue
        c = _centre(at)
        out.append({
            "offset": c,
            "kind": "mirror",
            "half_width": float(z[b] - c),
            "variation": variation,
            "bracket": [c - step, c + step],
            "located": "sampled",
            "uncertainty": step,
        })
    return out


def _sig_at(mesh, plane, offset):
    """(n_loops, n_holes, [area, perimeter, width, depth]) at one offset."""
    polys = section_at(mesh, plane, float(offset))
    if not polys:
        return None
    bb = np.array([p.bounds for p in polys], float)
    return (len(polys), int(sum(len(p.interiors) for p in polys)),
            np.array([sum(p.area for p in polys),
                      sum(p.length for p in polys),
                      bb[:, 2].max() - bb[:, 0].min(),
                      bb[:, 3].max() - bb[:, 1].min()], float))


def _mismatch(mesh, plane, centre, probes, scale):
    """How unlike each other the two sides of `centre` are, scale-free.

    All four channels, not area alone. Area is the one channel a torus's
    equator is nearly invisible in, so an area-only mismatch has almost nothing
    to minimise there and settles wherever the noise is lowest -- a plane
    visibly off, reported with a tiny uncertainty, which is the exact sin R1
    exists to prevent.

    A probe whose two sides have different topology is SKIPPED rather than
    poisoning the whole measurement: a local mirror is local, and its outer
    probes reach into geometry that only exists on one side.
    """
    used, total = 0, 0.0
    for d in probes:
        a = _sig_at(mesh, plane, centre - d)
        b = _sig_at(mesh, plane, centre + d)
        if a is None or b is None or (a[0], a[1]) != (b[0], b[1]):
            continue
        total += float(np.sum(((a[2] - b[2]) / scale) ** 2))
        used += 1
    if used < 2:
        return np.inf, 0
    return total / used, used


def refine_mirror(mesh, plane, centre, half_width, scale, step,
                  n_probe=7, precision=REFINE_PRECISION,
                  max_iters=REFINE_MAX_ITERS, accept=1e-3):
    """Locate a mirror plane by minimising the two sides' mismatch.

    Golden-section over a bracket of a few sweep steps -- detection already
    placed the centre to within one step, so searching half the feature is both
    wasteful and dangerous: it walks the probes out of the part, every
    comparison comes back topologically mismatched, and the search converges on
    an edge. That is what made this refuse on four parts whose centres it had
    already found exactly.

    Returns (offset, uncertainty, residual) or None. `residual` is the mismatch
    that remains at the answer, in the same normalised units as detection, and
    the caller is expected to look at it: a "mirror" that still disagrees with
    itself is an approximation and must not be presented as a measurement.
    """
    scale = np.asarray(scale, float)
    scale = np.where(scale > 1e-12, scale, 1.0)
    span = max(3.0 * step, precision * 10)
    # Probes spread over the symmetric region, kept clear of both the centre
    # (no signal) and the far edge (where the search span could walk them out).
    usable = max(half_width - span, half_width * 0.5)
    probes = [usable * f for f in np.linspace(0.15, 0.95, n_probe)]

    lo, hi = centre - span, centre + span
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    c1, c2 = hi - gr * (hi - lo), lo + gr * (hi - lo)
    f1 = _mismatch(mesh, plane, c1, probes, scale)[0]
    f2 = _mismatch(mesh, plane, c2, probes, scale)[0]
    if not np.isfinite(f1) and not np.isfinite(f2):
        return None
    for _ in range(max_iters):
        if hi - lo < precision:
            break
        if f1 < f2:
            hi, c2, f2 = c2, c1, f1
            c1 = hi - gr * (hi - lo)
            f1 = _mismatch(mesh, plane, c1, probes, scale)[0]
        else:
            lo, c1, f1 = c1, c2, f2
            c2 = lo + gr * (hi - lo)
            f2 = _mismatch(mesh, plane, c2, probes, scale)[0]
    got = 0.5 * (lo + hi)
    residual, used = _mismatch(mesh, plane, got, probes, scale)
    if not np.isfinite(residual) or used < 2:
        return None
    # Pinned to the edge of the bracket means the minimum is outside it, so the
    # detection was wrong about where the plane is.
    if abs(got - centre) > span * 0.98:
        return None

    # The minimum can be genuinely FLAT, and then there is no single answer to
    # find. Where the walls are constant-section, sliding the mirror plane a
    # little changes nothing any section can see: the residual is at float noise
    # and golden-section stops wherever it happens to be. Rather than dress that
    # up as a located plane, measure how far the flatness extends and report the
    # middle of it with an uncertainty that says so.
    flat = max(2.0 * float(np.sqrt(residual)), 1e-4)
    edges = []
    for direction in (-1.0, 1.0):
        a, b = 0.0, span
        for _ in range(24):
            if b - a < precision:
                break
            m = 0.5 * (a + b)
            v = _mismatch(mesh, plane, got + direction * m, probes, scale)[0]
            if np.isfinite(v) and np.sqrt(v) <= flat:
                a = m
            else:
                b = m
        edges.append(0.5 * (a + b))
    band_lo, band_hi = got - edges[0], got + edges[1]
    centre_of_band = 0.5 * (band_lo + band_hi)
    unc = max((band_hi - band_lo) / 2.0, (hi - lo) / 2.0, precision)
    return float(centre_of_band), float(unc), float(np.sqrt(residual))


def extremum_levels(slices, min_prominence=EXTREMUM_MIN_PROMINENCE,
                    min_half=EXTREMUM_MIN_HALF):
    """Offsets where the section is locally widest or narrowest.

    Reported per channel, because "widest" is a different fact from "largest
    area" and a part can have them in different places. A turning point has to
    stand clear of its neighbourhood by `min_prominence` of the channel's own
    range, which is what keeps a flat prism -- where every sample ties for the
    maximum -- from reporting all of them.
    """
    n = len(slices)
    if n < 2 * min_half + 3:
        return []
    z = np.array([s.offset for s in slices], float)
    sig = _signature(slices)
    topo = np.array([(s.n_loops, s.n_holes) for s in slices], int)
    names = ("area", "perimeter", "width", "depth")

    out = []
    for ch, label in enumerate(names):
        v = sig[:, ch]
        if float(v.max() - v.min()) <= 0:
            continue
        for sense in ("max", "min"):
            better = (lambda a, b: a >= b) if sense == "max" else (lambda a, b: a <= b)
            for i in range(1, n - 1):
                # Grow the window while this sample stays the extreme one in
                # it. Prominence measured against the WORSE of the two ends --
                # against the better one, a monotone step qualifies, because
                # only one side has to fall, and a shoulder would be reported
                # as a turning point while a real equator is missed.
                k = 0
                while i - k - 1 >= 0 and i + k + 1 < n:
                    j = k + 1
                    if not (np.array_equal(topo[i - j], topo[i]) and
                            np.array_equal(topo[i + j], topo[i])):
                        break     # a jump, not a turning point
                    if not (better(v[i], v[i - j]) and better(v[i], v[i + j])):
                        break
                    k = j
                if k < min_half:
                    continue
                ends = (v[i - k], v[i + k])
                prom = (float(v[i] - max(ends)) if sense == "max"
                        else float(min(ends) - v[i]))
                if prom < min_prominence:
                    continue
                # A PLATEAU is not a turning point. A capsule's width is
                # constant along its whole cylinder, so every sample there ties
                # for the maximum and every one grows a window out to the
                # hemispheres with full prominence -- 312 "turning points" for
                # one part, none of which names a place. If many samples share
                # the extreme value there is no located level here, and the
                # midpoint of the run is already reachable as mid() of the two
                # boundaries that bound it.
                tied = int(np.count_nonzero(
                    np.abs(v[i - k:i + k + 1] - v[i]) <= 1e-6))
                if tied > min_half:
                    continue
                out.append({
                    "offset": float(z[i]),
                    "kind": f"{sense} {label}",
                    "prominence": prom,
                    "channel": label,
                    "sense": sense,
                    "half_width": float(z[i + k] - z[i]),
                    "bracket": [float(z[i - k]), float(z[i + k])],
                    "located": "sampled",
                    "uncertainty": float(z[i + 1] - z[i - 1]) / 2.0,
                })
    return _dedupe(out)


def _dedupe(items, key="offset"):
    """One entry per place, per kind.

    Grouped by kind FIRST. Sorting everything by offset interleaves the eight
    channel/sense combinations, so consecutive entries were almost never the
    same kind and nothing ever merged -- which is how one plateau came back as
    312 separate findings that the deduplicator waved through.
    """
    if not items:
        return []
    out = []
    kinds = {}
    for t in items:
        kinds.setdefault(t["kind"], []).append(t)
    for group in kinds.values():
        group.sort(key=lambda t: t[key])
        kept = [group[0]]
        for t in group[1:]:
            width = max(abs(kept[-1]["bracket"][1] - kept[-1]["bracket"][0]), 1e-9)
            if abs(t[key] - kept[-1][key]) <= width * 0.5:
                if t.get("prominence", 0.0) > kept[-1].get("prominence", 0.0):
                    kept[-1] = t
                continue
            kept.append(t)
        out.extend(kept)
    return sorted(out, key=lambda t: t[key])


def _channel_value(mesh, plane, offset, channel):
    polys = section_at(mesh, plane, offset)
    if not polys:
        return 0.0
    if channel == "area":
        return float(sum(p.area for p in polys))
    if channel == "perimeter":
        return float(sum(p.length for p in polys))
    bb = np.array([p.bounds for p in polys], float)
    if channel == "width":
        return float(bb[:, 2].max() - bb[:, 0].min())
    return float(bb[:, 3].max() - bb[:, 1].min())


def refine_extremum(mesh, plane, offset, bracket, channel, sense,
                    precision=REFINE_PRECISION, max_iters=REFINE_MAX_ITERS):
    """Locate a turning point by golden-section on the channel itself.

    This is the badly conditioned one and the returned uncertainty says so. A
    smooth extremum is quadratic: the value falls off as the SQUARE of the
    distance, so a channel resolved to eps only fixes the position to about
    sqrt(eps). The bracket is narrowed to `precision` all the same -- there is
    no harm in a tight bracket -- but the uncertainty reported is the honest
    sqrt-scale one, derived from how much the channel actually moved.
    """
    lo, hi = float(bracket[0]), float(bracket[1])
    sign = 1.0 if sense == "max" else -1.0
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    c1, c2 = hi - gr * (hi - lo), lo + gr * (hi - lo)
    f1 = sign * _channel_value(mesh, plane, c1, channel)
    f2 = sign * _channel_value(mesh, plane, c2, channel)
    ends = [sign * _channel_value(mesh, plane, x, channel) for x in (lo, hi)]
    for _ in range(max_iters):
        if hi - lo < precision:
            break
        if f1 > f2:
            hi, c2, f2 = c2, c1, f1
            c1 = hi - gr * (hi - lo)
            f1 = sign * _channel_value(mesh, plane, c1, channel)
        else:
            lo, c1, f1 = c1, c2, f2
            c2 = lo + gr * (hi - lo)
            f2 = sign * _channel_value(mesh, plane, c2, channel)
    got = 0.5 * (lo + hi)
    peak = max(f1, f2)
    drop = peak - min(ends)
    half = float(bracket[1] - bracket[0]) / 2.0
    if drop <= 0:
        return None
    # A parabola through (centre, peak) and (centre +/- half, peak - drop) is
    # displaced by `half * sqrt(eps / drop)` when the value is only known to
    # eps. That, not the bracket width, is what this position is worth.
    eps = max(abs(peak) * 1e-9, 1e-12)
    unc = float(half * np.sqrt(eps / drop))
    return float(got), max(unc, precision)


# ------------------------------------------------------------- taper ends ---

def taper_blind_radius(slices):
    """How close to a value jump a taper end can hide, in mm, or None.

    A silence has to mean "there is no taper end here", not "I could not look".
    Finding a kink needs a clear run of samples either side to fit a line to,
    so within a few samples of a jump -- or of the part's own end -- there is
    nowhere to fit one and the detector cannot answer.

    A chamfer can stop a fraction of a millimetre after a cavity opens: at a
    coarse sampling the kink is inside the blind radius and nothing is found;
    sampled finer, it comes back at the same level at every finer sampling --
    which is exactly where the width reaches its final value and stops.

    None when the sweep has no jumps, in which case only the two ends are
    blind and nothing is hiding there.
    """
    if len(slices) < 3:
        return None
    topo = [(s.n_loops, s.n_holes) for s in slices]
    area = np.array([s.area for s in slices], float)
    has_jump = any(topo[i] != topo[i + 1] or
                   abs(area[i + 1] - area[i]) > 0.05 * max(area[i], 1e-9)
                   for i in range(len(slices) - 1))
    if not has_jump:
        return None
    step = float(slices[1].offset - slices[0].offset)
    return float(min(w + g for w, g in TAPER_WINDOWS) * step)


def _segments(slices):
    """Index ranges between value jumps, so no fit ever spans a discontinuity.

    This is the first thing to get right. A cavity opening drops the area by an
    order of magnitude in one step; fitting a line across that models the step,
    not the taper beside it, and the answer is meaningless in a way no threshold
    can rescue.
    """
    topo = [(s.n_loops, s.n_holes) for s in slices]
    area = np.array([s.area for s in slices], float)
    cuts = [0]
    for i in range(len(slices) - 1):
        jump = (topo[i] != topo[i + 1] or
                abs(area[i + 1] - area[i]) > 0.05 * max(area[i], 1e-9))
        if jump:
            cuts.extend([i + 1, i + 1])
    cuts.append(len(slices))
    smallest = min(w + g for w, g in TAPER_WINDOWS)
    return [(cuts[k], cuts[k + 1]) for k in range(0, len(cuts) - 1, 2)
            if cuts[k + 1] - cuts[k] >= 2 * smallest + 1]


def _line_fit(t, v):
    """(slope, intercept) by least squares, in whatever units are passed in."""
    a = np.polyfit(t, v, 1)
    return float(a[0]), float(a[1])


def _slope_change(t, v, i, window, gap):
    """Slopes of the runs either side of sample `i`, and where they cross.

    The `gap` samples next to `i` are excluded from both fits: those are the
    ones straddling the kink, and they belong to neither side.
    """
    lo0, lo1 = i - gap - window, i - gap
    hi0, hi1 = i + gap + 1, i + gap + 1 + window
    if lo0 < 0 or hi1 > len(t):
        return None
    ml, bl = _line_fit(t[lo0:lo1], v[lo0:lo1])
    mr, br = _line_fit(t[hi0:hi1], v[hi0:hi1])
    if abs(mr - ml) < 1e-15:
        return None
    cross = (bl - br) / (mr - ml)
    return ml, mr, cross


def taper_levels(slices, windows=TAPER_WINDOWS,
                 persistence=TAPER_PERSISTENCE,
                 min_change=TAPER_MIN_SLOPE_CHANGE):
    """Offsets where the section stops changing at one rate and starts at another.

    The ends of a taper: where a chamfer stops, where a fin's tip begins to
    narrow. These are the levels a loft actually wants, and the sweep has never
    been able to produce one -- it looks for jumps, and a taper end is a jump in
    the DERIVATIVE, which is invisible to it. What it offered instead was a
    `plateau edge`, a greedy scan's opinion about where a run stopped drifting,
    and a loft aimed at it can build the part outside its own bounding box.

    Two tests:

    1. Nothing is fitted across a value jump -- see `_segments`.
    2. A candidate must SURVIVE HALVING THE WINDOW. A kink is a discontinuity
       in the slope, so it measures the same at any scale. Curvature is spread
       out, so at half the window it shows half the change. Without this test
       every sample on a sphere qualifies: imaginary tapers on a smooth capsule.

    The offset returned is where the two fitted lines cross, not the sample the
    detector stopped at, so it is already better than the grid before any
    refinement.
    """
    n = len(slices)
    if n < 2 * max(w + g for w, g in windows) + 3:
        return []
    z = np.array([s.offset for s in slices], float)
    span = float(z[-1] - z[0]) or 1.0
    t = (z - z[0]) / span                 # dimensionless, so slopes compare
    sig = _signature(slices)
    names = ("area", "perimeter", "width", "depth")

    out = []
    for lo, hi in _segments(slices):
        for ch, label in enumerate(names):
            v = sig[:, ch]
            if float(v[lo:hi].max() - v[lo:hi].min()) <= 1e-9:
                continue
            local = np.abs(np.diff(v[lo:hi]) / np.diff(t[lo:hi]))
            scale = float(local.max())
            if scale <= 0:
                continue
            smallest = min(w + g for w, g in windows)
            for i in range(lo + smallest, hi - smallest):
                # Largest window that fits inside this segment on both sides.
                # A taper can stop a few samples after a jump, so with only the
                # largest window there is nowhere to fit a line on that side and
                # the taper is invisible.
                for window, gap in windows:
                    if i - gap - window < lo or i + gap + window >= hi:
                        continue
                    full = _slope_change(t, v, i, window, gap)
                    half = _slope_change(t, v, i, max(2, window // 2), gap)
                    if full is None or half is None:
                        continue
                    d_full = abs(full[1] - full[0])
                    d_half = abs(half[1] - half[0])
                    if d_full < min_change * scale:
                        continue
                    if d_half < persistence * d_full:
                        break     # the bend spreads out: curvature, not a kink
                    cross = full[2]
                    # The crossing has to be where the kink is. A pair of lines
                    # fitted either side of a smooth run can cross anywhere.
                    if not (t[i - gap] - 1e-12 <= cross <= t[i + gap] + 1e-12):
                        continue
                    out.append({
                        "offset": float(z[0] + cross * span),
                        "kind": f"taper end, {label}",
                        "channel": label,
                        "strength": float(d_full / scale),
                        "persistence": float(d_half / d_full),
                        "half_width": float(z[min(i + gap + window, n - 1)] - z[i]),
                        "bracket": [float(z[i - gap - window]),
                                    float(z[i + gap + window])],
                        "located": "sampled",
                        "uncertainty": float(z[i + 1] - z[i - 1]) / 2.0,
                    })
                    break
    return _dedupe(out)


def refine_taper(mesh, plane, offset, half_width, channel,
                 n_probe=9, gap_frac=0.12, iters=3,
                 precision=REFINE_PRECISION):
    """Re-measure a taper end against real sections and re-cross the two lines.

    The sweep's estimate already comes from a line crossing rather than from a
    sample, so this is not about escaping the grid so much as about placing the
    samples well: they go where the two runs actually are, evenly, with a gap
    around the kink that shrinks as the estimate improves.

    The uncertainty is propagated from the fits rather than asserted. Scatter
    about each line turns into scatter in the crossing, and on a run that is not
    really straight that comes out large -- which is the honest answer, because
    then the "kink" is a bend and its position is not a well-defined thing.
    """
    got = float(offset)
    unc = float(half_width)
    for _ in range(iters):
        gap = max(half_width * gap_frac, precision)
        left = np.linspace(got - half_width, got - gap, n_probe)
        right = np.linspace(got + gap, got + half_width, n_probe)
        vl = np.array([_channel_value(mesh, plane, x, channel) for x in left])
        vr = np.array([_channel_value(mesh, plane, x, channel) for x in right])
        if not (np.all(np.isfinite(vl)) and np.all(np.isfinite(vr))):
            return None
        ml, bl = _line_fit(left, vl)
        mr, br = _line_fit(right, vr)
        if abs(mr - ml) < 1e-12:
            return None
        cross = (bl - br) / (mr - ml)
        if not (got - half_width <= cross <= got + half_width):
            return None
        # Residual scatter about each line, carried through to the crossing.
        res = float(np.sqrt(np.mean(np.concatenate(
            [vl - (ml * left + bl), vr - (mr * right + br)]) ** 2)))
        unc = max(res / abs(mr - ml), precision)
        if abs(cross - got) < precision:
            got = cross
            break
        got = cross
    return float(got), float(unc)
