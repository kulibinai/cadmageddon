"""What the model last read on a plane -- the rings `section` printed and the
level table `ortho` printed -- kept beside the renders.

A name is only good at the level it was printed for: R4 of the section at
Z=40 is not R4 at Z=22.86, and L0_0 of the hexagon at X=1.5 is nothing at E0.
The model reads a section once and names its rings in a sketch at another
level, where the name may match nothing -- or another ring, building a pin
instead of the body without a warning. With what it read kept, a name that
is not here resolves to the ring here that overlaps the one it read, and says
so; a name that is here but was another ring when it was read is warned
about. Evidence for a name, never a choice made for the model: the ring used
is the one the name was read as.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

# Two rings are the same ring at two levels when they overlap by more than
# half of their union: a drafted wall or a chamfered edge moves a ring by a
# fraction of its width, a neighbouring ring is mostly elsewhere.
SAME_RING = 0.5
_CTX = {}


def set_context(out, mesh):
    """Where this command's records are (`out`: a path, or a callable giving
    one) and for which mesh. Set once per command by the CLI."""
    _CTX.clear()
    _CTX.update(out=out, mesh=mesh)


def _out(out=None):
    out = out if out is not None else _CTX.get("out")
    return Path(out() if callable(out) else out) if out is not None else None


def _path(out, mesh, plane, kind, frame=""):
    key = hashlib.sha1(str(Path(mesh).resolve()).encode()).hexdigest()[:12]
    return _out(out) / ".m2c" / "read" / f"{key}_{plane}_{frame or 'world'}_{kind}.json"


def _write(path, rec):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec) + "\n")
    except OSError:
        pass                     # a record that cannot be kept is not fatal


def _read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


# --- rings ---------------------------------------------------------------------

def outline(ring, step_deg=7.5):
    """The ring's outline as points: every primitive's start, and a point every
    `step_deg` along each arc, so a circle is a circle (0.3% short in area at
    7.5 deg) and not one degenerate chord."""
    pts = []
    segs = ring.get("segments", [])
    for s in segs:
        pts.append(tuple(map(float, s["from_xy"])))
        if s.get("kind") == "arc" and s.get("centre") is not None:
            c = np.asarray(s["centre"], float)
            r = float(s["radius"])
            a0 = np.arctan2(s["from_xy"][1] - c[1], s["from_xy"][0] - c[0])
            # A ring of one arc is a whole circle, whatever sweep it records: a
            # cut section's decomposition writes it as 0 deg.
            deg = 360.0 if len(segs) == 1 else float(s.get("sweep_deg", 0.0))
            sweep = np.radians(deg) * (1.0 if s.get("ccw", True) else -1.0)
            n = max(4, int(np.ceil(abs(deg) / step_deg)))
            for t in np.linspace(0.0, 1.0, n + 1)[1:-1]:
                a = a0 + sweep * t
                pts.append((float(c[0] + r * np.cos(a)), float(c[1] + r * np.sin(a))))
    return pts


def _polygon(pts):
    from shapely.geometry import Polygon
    if len(pts) < 3:
        return None
    g = Polygon(pts)
    if not g.is_valid:
        g = g.buffer(0)
    return g if not g.is_empty and g.area > 0 else None


def shape(ring):
    """The ring as a shapely polygon in (u, v), or None."""
    return _polygon(ring.get("outline") or outline(ring))


def overlap(a, b):
    """Intersection over union of two rings (dicts with `outline`, or rings
    as `section` returns them)."""
    pa = _polygon(a.get("outline") or outline(a))
    pb = _polygon(b.get("outline") or outline(b))
    if pa is None or pb is None:
        return 0.0
    union = pa.union(pb).area
    return float(pa.intersection(pb).area / union) if union > 0 else 0.0


def describe(ring):
    """`R2 (hole, area 13.57 mm^2, 4.157 x 4.157)`."""
    g = _polygon(ring.get("outline") or outline(ring))
    if g is None:
        return f"{ring.get('name', '?')} ({ring.get('role', '?')})"
    x0, y0, x1, y1 = g.bounds
    return (f"{ring.get('name', '?')} ({ring.get('role', '?')}, area {g.area:.4g} mm^2, "
            f"{x1 - x0:.4g} x {y1 - y0:.4g})")


def save_section(out, mesh, plane, level, rings, frame=""):
    """Keep the rings `section` just printed at `level` on `plane`."""
    rec = {"mesh": str(Path(mesh).resolve()), "plane": plane, "level": float(level),
           "rings": [{"name": r.get("name"), "role": r.get("role"),
                      "segments": [s["name"] for s in r.get("segments", [])],
                      "outline": outline(r)} for r in rings if r.get("name")]}
    _write(_path(out, mesh, plane, "section", frame), rec)


def last_section(out, mesh, plane, frame=""):
    """The rings the model last read on `plane` with `section`, or None."""
    return _read(_path(out, mesh, plane, "section", frame))


def ring_of(rec, name):
    """The recorded ring a name stood for: the ring itself, or the ring a
    primitive is part of (naming a primitive selects its ring)."""
    if rec is None or "@" in name:
        return None
    for r in rec["rings"]:
        if r["name"] == name or name in r["segments"]:
            return r
    return None


def best_match(was, rings):
    """(ring here, overlap) for the ring here of the same role that overlaps
    `was` most, or (None, 0.0)."""
    best, score = None, 0.0
    for r in rings:
        if r.get("name") and r.get("role") == was.get("role"):
            s = overlap(was, r)
            if s > score:
                best, score = r, s
    return best, score


def listing(rings, at_most=12):
    """The rings of a section, named with their sizes, for a refusal."""
    named = [r for r in rings if r.get("name")]
    text = "; ".join(describe(r) for r in named[:at_most])
    if len(named) > at_most:
        text += f"; and {len(named) - at_most} more"
    return text or "none"


# --- levels ----------------------------------------------------------------------

def save_levels(out, mesh, plane, source, marks, frame=""):
    """Keep the level table `ortho` just printed for `plane`."""
    rec = {"mesh": str(Path(mesh).resolve()), "plane": plane, "source": source or "default",
           "levels": [{"name": m["name"], "value": float(m["value"]),
                       "uncertainty": float(m.get("uncertainty", 0.0)),
                       "label": str(m.get("label", ""))}
                      for m in marks if m.get("name")]}
    _write(_path(out, mesh, plane, "levels", frame), rec)


def printed_level(name, plane, out=None, mesh=None, frame=""):
    """(level dict, record) for a level `name` the last level table printed
    on `plane`, or (None, None). Uses the command's context by default."""
    mesh = mesh or _CTX.get("mesh")
    if mesh is None or _out(out) is None or plane is None:
        return None, None
    rec = _read(_path(out, mesh, plane, "levels", frame))
    for m in (rec or {}).get("levels", []):
        if m["name"] == name:
            return m, rec
    return None, None


def planes_naming(name, out=None, mesh=None, frame=""):
    """The other planes whose last level table printed `name`: [(plane, level)]."""
    found = []
    for plane in ("XY", "YZ", "XZ"):
        m, _ = printed_level(name, plane, out, mesh, frame)
        if m is not None:
            found.append((plane, m))
    return found


# --- paths -----------------------------------------------------------------------

def save_paths(out, mesh, records, frame=""):
    """Keep the path names (W#) a command just printed and what each one is:
    {"name", "source": "faces", "chain"} for a pipe `faces` chained, or
    {"name", "source": "section", "plane", "level", "ring"} for a strip's
    centreline; a tube's midline between two strips has "walls" for "ring", and
    "pieces" and "width" for the path itself, as has {"source": "wires"} for a
    wire off the mesh. Two commands can print one name for two paths -- an
    in-plane section names its wall strips W0/W1, `faces` its tube W0 -- so each
    source is kept, with when it was printed."""
    if _out(out) is None or mesh is None:
        return
    path = _path(out, mesh, "all", "paths", frame)
    rec = _read(path) or {"mesh": str(Path(mesh).resolve()), "paths": {}}
    t = time.time()
    for r in records:
        key = f"section_{r['plane']}" if r["source"] == "section" else r["source"]
        rec["paths"].setdefault(r["name"], {})[key] = dict(r, t=t)
    _write(path, rec)


def printed_path(name, out=None, mesh=None, frame=""):
    """(what `name` stood for when it was last printed, [what else printed it
    earlier]) -- or (None, []) for a name no command printed."""
    mesh = mesh or _CTX.get("mesh")
    if mesh is None or _out(out) is None:
        return None, []
    rec = _read(_path(out, mesh, "all", "paths", frame)) or {}
    got = sorted((rec.get("paths") or {}).get(name, {}).values(), key=lambda r: -r["t"])
    return (got[0], got[1:]) if got else (None, [])


def printed_paths(out=None, mesh=None, frame=""):
    """Every path name printed for this mesh: {name: latest record}."""
    mesh = mesh or _CTX.get("mesh")
    if mesh is None or _out(out) is None:
        return {}
    rec = _read(_path(out, mesh, "all", "paths", frame)) or {}
    return {n: max(v.values(), key=lambda r: r["t"])
            for n, v in (rec.get("paths") or {}).items() if v}
