"""Building a feature across an inclined face or axis.

Every sketch plane is XY, YZ or XZ, every level is a coordinate along X, Y or
Z, and all of `section`, `ortho` and `sketch` is written in those terms. A face
or an axis lying across none of them -- an inclined joggle, a tilted boss with
its counterbore, bores tilted off a drum's axis -- would be out of reach:
`--op hole` would refuse the axis, `--face` the face.

A frame turns the TARGET instead, by the smallest rotation that lays the named
direction -- a flat face's normal, a round face's or hole stack's axis -- on the
world axis nearest it. In that frame the inclined face is an XY/YZ/XZ face like
any other, the faces parallel to it are levels, a cut across the axis is a
section, and every instrument works unchanged. The feature is built there and
turned back by the inverse rotation inside its own block, so the program stays
in the target's frame and `--base`, `--replace` and `edit` see an ordinary
feature.

The segmentation is not redone in the frame: the target's own is turned with
it (`turn_segmentation`), so F7 and C3 name the same surfaces in every frame and
the frame costs no second search for round faces.
"""

import copy
import re

import numpy as np

AXES = "XYZ"
# The sketch plane whose normal is each world axis (as mesh2cad.holes has it).
PLANE_ACROSS = {0: "YZ", 1: "XZ", 2: "XY"}
# Below this the direction already lies on its world axis: no frame is needed,
# and none is applied (the turn would only add rounding).
ON_AXIS_RAD = 1e-6


class FrameError(ValueError):
    """A name that gives no direction to build across; the message says why."""


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def direction_of(seg, name):
    """(unit direction, what it is) for a frame NAME from `faces`.

    A flat face F# gives its normal; a cylinder, cone, torus or revolve F#
    gives its axis; a hole stack C# gives the axis its faces share. A sphere
    or a freeform face has no direction and is refused, saying so.
    """
    from mesh2cad.holes import stack_names

    if name.upper().startswith("C") and name[1:].isdigit():
        stacks = dict(stack_names(seg))
        if name.upper() not in stacks:
            have = ", ".join(n for n, _ in stack_names(seg)) or "none"
            raise FrameError(f"no hole stack {name!r}; `faces` names them: {have}")
        s = seg.surfaces[stacks[name.upper()][0]]
        return _unit(s.params["axis"]), f"the axis of hole stack {name.upper()}"
    s = seg.by_name(name.upper())
    if s is None:
        raise FrameError(f"no surface {name!r}; `faces` lists F0..F"
                         f"{len(seg.surfaces) - 1} and the hole stacks C#")
    if s.kind == "plane":
        return _unit(s.params["normal"]), f"the normal of flat face {s.name}"
    if "axis" in s.params:
        return _unit(s.params["axis"]), f"the axis of {s.kind} {s.name}"
    raise FrameError(f"{s.name} is a {s.kind}: it has no direction to build "
                     f"across. Name a flat face (its normal) or a round face or "
                     f"hole stack (its axis)")


def inclined(d):
    """True when a direction lies on none of X, Y, Z."""
    d = _unit(d)
    k = int(np.argmax(np.abs(d)))
    return float(np.arccos(min(1.0, abs(d[k])))) > ON_AXIS_RAD


def frame_for(d):
    """The smallest rotation laying direction d on the world axis nearest it.

    Returns a dict: R (3x3, world -> frame), k (that axis), sign (+1 / -1: d
    lands on +axis or -axis), angle (degrees), about (unit rotation axis).
    """
    d = _unit(d)
    k = int(np.argmax(np.abs(d)))
    sign = 1.0 if d[k] > 0 else -1.0
    t = np.zeros(3)
    t[k] = sign
    c = float(np.clip(d @ t, -1.0, 1.0))
    w = np.cross(d, t)
    s = float(np.linalg.norm(w))
    if s < 1e-15:
        return {"R": np.eye(3), "k": k, "sign": sign, "angle": 0.0,
                "about": np.array([1.0, 0.0, 0.0])}
    w = w / s
    ang = float(np.arctan2(s, c))
    K = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
    R = np.eye(3) + np.sin(ang) * K + (1.0 - np.cos(ang)) * (K @ K)
    return {"R": R, "k": k, "sign": sign, "angle": float(np.degrees(ang)),
            "about": w}


def turn_mesh(mesh, R):
    """The mesh turned by R about the origin, triangles untouched."""
    import trimesh
    V = np.asarray(mesh.vertices, float) @ np.asarray(R, float).T
    return trimesh.Trimesh(vertices=V, faces=np.asarray(mesh.faces).copy(),
                           process=False)


_VECTORS = ("normal", "axis", "point", "centre")


def turn_segmentation(seg, R, turned_mesh):
    """`seg` of the target, re-expressed in the frame R turns it into.

    Labels, names, joints, tolerances and every distance are unchanged by a
    rotation about the origin; only the directions and points move, and the
    bounding boxes, which are recomputed from the turned vertices.
    """
    R = np.asarray(R, float)
    out = copy.deepcopy(seg)
    V = np.asarray(turned_mesh.vertices, float)
    F = np.asarray(turned_mesh.faces)
    for s in out.surfaces:
        for key in _VECTORS:
            if key in s.params:
                s.params[key] = R @ np.asarray(s.params[key], float)
        model = getattr(s, "_model", None)
        if model is not None:
            for key in _VECTORS:
                if key in model.p:
                    model.p[key] = R @ np.asarray(model.p[key], float)
            model._uv = None
        if "bbox" in s.extent and len(s.faces):
            P = V[np.unique(F[s.faces])]
            s.extent["bbox"] = [P.min(axis=0), P.max(axis=0)]
    return out


def back_code(fr, name):
    """Source that turns the feature just built back out of the frame.

    Appended to a feature block before any `--pattern` (whose axis is a world
    one) and before the glue, so `result` and `_STEPS` leave the block in the
    target's frame like every other feature's.
    """
    a = fr["about"]
    axis = "(" + ", ".join(f"{v:.12g}" for v in a) + ")"
    ang = f"{-fr['angle']:.12g}"
    return (f"\n# turned back out of the frame of {name}\n"
            f"result = result.rotate((0, 0, 0), {axis}, {ang})\n"
            f"_STEPS = [(_op, _s.rotate((0, 0, 0), {axis}, {ang})) "
            f"for _op, _s in _STEPS]")


HOLE = re.compile(r"(?m)^(_h\d+) = \(")


def bspline_code(code):
    """Source that makes a hole cutter's `result` its holes (the `_hN` of
    `code`, one solid of revolution each), every one written as B-splines --
    the same surfaces, exactly -- and held side by side, not fused.

    OCC gets cones and cylinders on a tilted axis wrong against a revolved
    body, and says nothing: bores turned back out of their frames can come off
    a revolved body as a fraction of what they overlap, every solid valid.
    Written as B-splines and given to one cut as separate tools (what
    `Workplane.cut` does with a workplane of several solids), the bores come off
    as the meshes say; build._tessellate meshes such a workplane as the union of
    its solids. `_STEPS` keeps its operands as they were.
    """
    names = HOLE.findall(code)
    return ("\n# each hole as B-splines and not fused: OCC cuts a tilted cone or "
            "cylinder out of a revolved body wrongly, and silently, in analytic form\n"
            "from OCP.BRepBuilderAPI import BRepBuilderAPI_NurbsConvert\n"
            "result = cq.Workplane().add([cq.Shape.cast(BRepBuilderAPI_NurbsConvert("
            "_s.val().wrapped, True).Shape()) for _s in (" + ", ".join(names) + ",)])")


def gather_code(blocks):
    """`result` as the pieces of the blocks `_c0`... side by side, not fused,
    for a cutter whose pieces are B-splines (see bspline_code)."""
    return ("\nresult = cq.Workplane().add("
            + " + ".join(f"_c{i}.vals()" for i in range(blocks)) + ")")


def describe_frame(fr, name, what):
    ax = ("+" if fr["sign"] > 0 else "-") + AXES[fr["k"]]
    a = fr["about"]
    return (f"frame of {name}: the target is turned {fr['angle']:.4f} deg about "
            f"({a[0]:+.4f}, {a[1]:+.4f}, {a[2]:+.4f}) so {what} lies on {ax}; "
            f"every coordinate, level and ring name below is in that frame "
            f"(the plane across it is {PLANE_ACROSS[fr['k']]}). A sketch with "
            f"the same --frame is built there and turned back when written")
