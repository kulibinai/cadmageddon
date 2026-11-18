"""The target's surface against the reconstruction's -- what a volume residual
cannot see.

The DONE rule reads the volume residual, and a shaped error that holds little
volume passes it: bends built as sharp corners leave little volume while a
large share of the target's triangles sit well off the build. Here the
comparison sheet's own yardstick (`comparison.py`: unsigned distance from each
target triangle to the reconstruction, the 0.1%-of-span band) is read as
numbers: how much of the target's surface is off, and how much of that lies in
connected patches big enough to be a shape rather than scatter.
"""

import sys
from pathlib import Path

import numpy as np

# A connected patch of off-band target triangles smaller than this share of the
# target's area is scatter -- a chord here and there -- and is not reported.
PATCH_MIN = 0.002
# Shaped patches covering at least this share of the target's surface: a shape
# is still missing, and the part is not finished, whatever the volume says.
GATE_SHARE = 0.01
LISTED = 3
NOT_FINISHED = "not finished"


# Target faces listed in `residual`'s table of faces off.
FACES_LISTED = 6


def _band_and_offsets(target, built):
    """(the 0.1%-of-span band, the signed distance from each target triangle's
    centroid to the build's surface). Positive: the point is outside the build,
    which stops SHORT of that face; negative: the build runs PAST it. The
    magnitude is comparison.py's own distance."""
    import pyvista as pv

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from comparison import FRACTIONS, max_axis_span, thresholds_for

    tgt, got = pv.wrap(target), pv.wrap(built)
    band = thresholds_for(max_axis_span([tgt, got]), FRACTIONS)[0]
    d = tgt.cell_centers().compute_implicit_distance(got)["implicit_distance"]
    return band, np.asarray(d, float)


def _labelled(target, seg):
    """The segmentation's triangle labels when they are this mesh's, else None."""
    labels = None if seg is None else np.asarray(seg.labels)
    return labels if labels is not None and len(labels) == len(target.faces) else None


def _named(tris, areas, labels, seg, listed=2):
    """`cylinder F8 R16 + plane F3`: the target faces under these triangles,
    one surface named once, largest share first."""
    from mesh2cad.leftover import Named, _family, on_text
    fam = {}
    for k in np.unique(labels[tris]):
        if k < 0:
            continue
        a = float(areas[tris][labels[tris] == k].sum())
        f = fam.setdefault(_family(seg.surfaces[int(k)]), [[], 0.0])
        f[0].append(seg.surfaces[int(k)])
        f[1] += a
    top = sorted(fam.values(), key=lambda sa: -sa[1])[:listed]
    return on_text([(Named.of(ss), a) for ss, a in top])


def _side(d):
    """The signed offset in words: how far the build stops short of or runs past."""
    med = float(np.median(d))
    return (f"built short of it by {abs(med):.3g} mm" if med > 0
            else f"built past it by {abs(med):.3g} mm")


def faces_off(target, built, seg, listed=FACES_LISTED):
    """The target's faces the build misses, ranked by the area off, each with
    its signed offset: which face is wrong and which way, where the volume
    residual gives only bodies. [] when nothing is off or the faces have no
    names."""
    labels = _labelled(target, seg)
    if labels is None:
        return []
    band, d = _band_and_offsets(target, built)
    areas = np.asarray(target.area_faces, float)
    off = np.abs(d) > band
    if not off.any():
        return []
    from mesh2cad.leftover import _family
    groups = {}
    for k in np.unique(labels):
        if k < 0:
            continue
        groups.setdefault(_family(seg.surfaces[int(k)]), []).append(int(k))
    rows = []
    for ks in groups.values():
        tris = np.flatnonzero(np.isin(labels, ks))
        total = float(areas[tris].sum())
        bad = tris[off[tris]]
        if total <= 0 or not len(bad):
            continue
        short, past = bad[d[bad] > 0], bad[d[bad] < 0]
        parts = []
        for side, word in ((short, "short of it"), (past, "past it")):
            if len(side):
                parts.append(f"built {word} by {float(np.median(np.abs(d[side]))):.3g} mm "
                             f"over {float(areas[side].sum()) / total:.0%}")
        rows.append((float(areas[bad].sum()),
                     f"{_named(tris, areas, labels, seg, listed=1)}: "
                     f"{float(areas[bad].sum()) / total:.0%} of it off -- {', '.join(parts)}"))
    rows.sort(key=lambda r: -r[0])
    out = [r[1] for r in rows[:listed]]
    if len(rows) > listed:
        out.append(f"... and {len(rows) - listed} more face(s) off")
    return [f"(band {band:.4g} mm, 0.1% of the span)"] + out


def surface_check(target, built, seg=None):
    """One line on how much of the target's surface the reconstruction misses,
    and in what patches. Each patch is named by the target faces under it and
    says which way the build is off there, when the segmentation is at hand.
    Raises if it cannot be measured (the caller says so)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    band, signed = _band_and_offsets(target, built)
    d = np.abs(signed)
    labels = _labelled(target, seg)
    areas = np.asarray(target.area_faces, float)
    total = float(areas.sum()) or 1.0
    off = d > band
    share = float(areas[off].sum()) / total
    idx = np.flatnonzero(off)
    patches = []
    if len(idx):
        pos = -np.ones(len(d), int)
        pos[idx] = np.arange(len(idx))
        adj = np.asarray(target.face_adjacency)
        both = adj[off[adj[:, 0]] & off[adj[:, 1]]]
        g = coo_matrix((np.ones(len(both)), (pos[both[:, 0]], pos[both[:, 1]])),
                       shape=(len(idx), len(idx)))
        _, label = connected_components(g, directed=False)
        centres = np.asarray(target.triangles_center)
        for c in np.unique(label):
            f = idx[label == c]
            a = float(areas[f].sum())
            if a / total >= PATCH_MIN:
                what = "" if labels is None else f" on {_named(f, areas, labels, seg)}"
                patches.append((a / total, (centres[f] * areas[f, None]).sum(0) / a,
                                f"{what}, {_side(signed[f])}"))
    patches.sort(key=lambda p: -p[0])
    shaped = sum(p[0] for p in patches)
    head = (f"surface   {share:.2%} of the target's surface is more than {band:.4g} "
            f"mm (0.1% of its span) off")
    if not patches:
        return (head + f"; none of it in a patch of {PATCH_MIN:.1%} of the surface or "
                f"more -- scatter, not a shape"), shaped
    where = "; ".join(f"{s:.2%} around ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}){w}"
                      for s, c, w in patches[:LISTED])
    more = f", and {len(patches) - LISTED} more" if len(patches) > LISTED else ""
    verdict = (f" -- a shape the volume residual cannot see (a bend, a draft, a "
               f"fillet, a lip): {NOT_FINISHED}" if shaped >= GATE_SHARE else "")
    return (head + f"; {shaped:.2%} of it in {len(patches)} shaped patch(es): "
            f"{where}{more}{verdict}"), shaped
