"""Resolve point / line specs from step JSON into folded-space geometry.

See docs/STEP_FORMAT.md for the spec.
"""

from __future__ import annotations

import numpy as np

from . import geometry as g
from .paper import PaperState

# Named points on the original sheet, as fractions of the paper size.
NAMED_POINTS = {
    "bl": (0.0, 0.0),
    "br": (1.0, 0.0),
    "tr": (1.0, 1.0),
    "tl": (0.0, 1.0),
    "bottom_mid": (0.5, 0.0),
    "right_mid": (1.0, 0.5),
    "top_mid": (0.5, 1.0),
    "left_mid": (0.0, 0.5),
    "center": (0.5, 0.5),
}


POINT_KEYS = ("orig", "mid", "lerp", "intersect")
LINE_KEYS = ("point_onto_line", "bisect", "perpendicular", "through", "onto", "horizontal", "vertical")


def resolve_point(state: PaperState, spec) -> np.ndarray:
    """Return a folded-space point for a point spec."""
    if isinstance(spec, str):
        if spec not in NAMED_POINTS:
            raise ValueError(f"unknown landmark {spec!r}; known: {sorted(NAMED_POINTS)}")
        fx, fy = NAMED_POINTS[spec]
        return state.to_folded((fx * state.size, fy * state.size))
    if isinstance(spec, (list, tuple)) and len(spec) == 2 and all(
        isinstance(v, (int, float)) for v in spec
    ):
        return np.array(spec, dtype=float)
    if isinstance(spec, dict):
        if "orig" in spec:
            return state.to_folded(spec["orig"])
        if "mid" in spec:
            p, q = (resolve_point(state, s) for s in spec["mid"])
            return (p + q) / 2
        if "lerp" in spec:
            p, q = (resolve_point(state, s) for s in spec["lerp"])
            return p + float(spec["t"]) * (q - p)
        if "intersect" in spec:
            l1, l2 = spec["intersect"]
            (a1, u1, _), (a2, u2, _) = resolve_line_ref(state, l1), resolve_line_ref(state, l2)
            return g.line_intersection(a1, a1 + u1, a2, a2 + u2)
    raise ValueError(f"bad point spec: {spec!r}")


def resolve_line_ref(state: PaperState, spec):
    """A line used as an *input* (edge, reference line).

    Either a pair of points [P, Q] (directed P -> Q) or any line spec dict.
    Returns (a, u, seg) where seg = (P, Q) for a point pair, else None.
    """
    if isinstance(spec, dict) and any(k in spec for k in LINE_KEYS):
        a, u, _ = resolve_line(state, spec)
        return a, u, None
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        p, q = (resolve_point(state, s) for s in spec)
        return p, g.unit(q - p), (p, q)
    raise ValueError(f"bad line reference (want [P, Q] or a line spec): {spec!r}")


def _perp(u: np.ndarray) -> np.ndarray:
    return np.array([-u[1], u[0]])


def _fold_onto(p: np.ndarray, q: np.ndarray):
    """Perpendicular bisector of pq: the fold that puts p onto q (p's side moves)."""
    d = g.unit(q - p)
    return (p + q) / 2, _perp(d), p


def resolve_line(state: PaperState, spec):
    """Return (a, u, default_move_point_or_None) for a line spec."""
    if not isinstance(spec, dict):
        raise ValueError(f"bad line spec: {spec!r}")
    if "point_onto_line" in spec:
        return _point_onto_line(state, spec)
    if "bisect" in spec:
        return _bisect(state, spec["bisect"])
    if "perpendicular" in spec:
        # Huzita O4: through point `at`, perpendicular to a line.
        _, u, _ = resolve_line_ref(state, spec["perpendicular"])
        return resolve_point(state, spec["at"]), _perp(u), None
    if "through" in spec:
        p, q = (resolve_point(state, s) for s in spec["through"])
        return p, g.unit(q - p), None
    if "onto" in spec:
        # Huzita O2: perpendicular bisector, brings point p onto point q.
        p, q = (resolve_point(state, s) for s in spec["onto"])
        return _fold_onto(p, q)
    if "horizontal" in spec:
        return resolve_point(state, spec["horizontal"]), np.array([1.0, 0.0]), None
    if "vertical" in spec:
        return resolve_point(state, spec["vertical"]), np.array([0.0, 1.0]), None
    raise ValueError(f"bad line spec: {spec!r}")


def _bisect(state: PaperState, refs):
    """Huzita O3: fold line 1 onto line 2.

    Of the two angle bisectors we take the one whose reflection maps direction
    P->Q of line 1 onto direction R->S of line 2. Parallel lines: the midline.
    Default move point: whichever of line 1's points P, Q is farther from the crease.
    """
    l1, l2 = refs
    a1, d1, seg1 = resolve_line_ref(state, l1)
    a2, d2, _ = resolve_line_ref(state, l2)
    if abs(g.cross(d1, d2)) < 1e-9:
        off = g.cross(d2, a1 - a2)  # signed distance of line 1 from line 2
        if abs(off) < g.EPS:
            raise ValueError("bisect: the two lines are the same line")
        a, u = a1 - _perp(d2) * off / 2, d2
    else:
        a = g.line_intersection(a1, a1 + d1, a2, a2 + d2)
        u = g.unit(d1 + d2)
    if seg1 is not None:
        move = max(seg1, key=lambda p: abs(g.side(a, u, p)))
    else:
        move = a1 if abs(g.side(a, u, a1)) > g.EPS else a1 + d1 * state.size
    if abs(g.side(a, u, move)) < g.EPS:
        move = None
    return a, u, move


def _point_onto_line(state: PaperState, spec):
    """Huzita O5 (`pivot`) / O7 (`perpendicular_to`): fold point P onto line L.

    O5: the crease passes through `pivot`. Up to two solutions; picked by `near`
    (landing spot closest to that point), else landing spots inside segment [P0, P1]
    of L are preferred, ties broken by closeness to P0 (L's first point).
    O7: the crease is perpendicular to line `perpendicular_to` (one solution).
    P's side moves by default.
    """
    p = resolve_point(state, spec["point_onto_line"])
    a, u, seg = resolve_line_ref(state, spec["line"])
    if ("pivot" in spec) == ("perpendicular_to" in spec):
        raise ValueError("point_onto_line needs exactly one of 'pivot' or 'perpendicular_to'")
    if "perpendicular_to" in spec:
        _, d, _ = resolve_line_ref(state, spec["perpendicular_to"])
        if abs(g.cross(d, u)) < 1e-9:
            raise ValueError("point_onto_line: P can't reach the line along that direction")
        target = g.line_intersection(p, p + d, a, a + u)
        if np.linalg.norm(target - p) < g.EPS:
            raise ValueError("point_onto_line: point is already on the line")
        return _fold_onto(p, target)

    r_pt = resolve_point(state, spec["pivot"])
    r = np.linalg.norm(p - r_pt)
    foot = a + u * float(np.dot(r_pt - a, u))
    h = np.linalg.norm(r_pt - foot)
    if h > r + 1e-9:
        raise ValueError("point_onto_line: no solution (pivot too far from the line)")
    t = np.sqrt(max(r * r - h * h, 0.0))
    cands = [foot + t * u, foot - t * u]
    cands = [c for c in cands if np.linalg.norm(c - p) > 1e-9]
    if not cands:
        raise ValueError("point_onto_line: point is already on the line at the pivot")
    if "near" in spec:
        hint = resolve_point(state, spec["near"])
        target = min(cands, key=lambda c: np.linalg.norm(c - hint))
    elif seg is not None:
        p0, p1 = seg
        length = np.linalg.norm(p1 - p0)

        def key(c):
            s = float(np.dot(c - p0, u))
            outside = s < -1e-9 or s > length + 1e-9
            return (outside, np.linalg.norm(c - p0))

        target = min(cands, key=key)
    else:
        target = min(cands, key=lambda c: np.linalg.norm(c - a))
    return _fold_onto(p, target)
