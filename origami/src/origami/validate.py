"""Physical validity checker: is each step something real paper can do?

The engine only checks that a step is well formed (and drags paper along so it never tears).
This module checks physics, after every step:

1. Layer order of the flat state (Flat-Folder / Akitaya et al. constraints, on face pairs that
   overlap in folded space):
   - every overlapping pair has an order;
   - transitivity: no 3 faces covering a common spot in a cycle (i over j over k over i);
   - taco-tortilla: no face (or flat-joined pair of faces) that reaches across a folded crease
     lies between the crease's two layers;
   - taco-taco: two folds along the same line on the same side don't interleave;
   - tortilla-tortilla: two flat-joined face pairs along the same line keep the same order on
     both sides of it;
   - no tear: faces adjacent on the sheet stay joined (same placement, or folded over their edge).
2. Local flat-foldability of the currently folded creases: Kawasaki (alternate angles sum to
   180 degrees) and Maekawa (|M - V| = 2) at each interior vertex. Plus (warning) tracked creases
   that end inside the paper without meeting another crease.
3. Motion of a plain fold (valley / mountain / unfold): the moving block turns about the fold line;
   in the thin-paper limit it hits a stationary layer iff that layer is on the swing side of a
   moving layer where they overlap, before the move (it is in the way) or after it (the flap had
   to pass it to land there). Tucks are allowed to violate the "after" part (warning: layers must
   be lifted). Macros (squash / petal / reverse) are not single rotations; optional `sampled`
   mode checks their animation (kinematics.py) for face-face intersections instead.

API: `validate_model(model, sampled=False) -> Report`; `check_step(before, after, step, k)`.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass, field

import numpy as np

from . import geometry as g
from .paper import OVERLAP_TOL, PaperState

TOL = 1e-6      # lengths (paper units)
TTOL = 1e-6     # transform comparison
ANG_TOL = 1e-4  # radians


# ---- report ---------------------------------------------------------------------------

@dataclass
class Issue:
    step: int                 # 1-based step number (0 = the flat sheet)
    kind: str                 # e.g. "collision", "taco_tortilla", "maekawa"
    severity: str             # "error" | "warning"
    message: str              # plain English, starts with "step k:"
    faces: list[int] = field(default_factory=list)  # face indices in the state after the step
    point: list[float] | None = None                # where (see message for which coordinates)
    sheet_point: list[float] | None = None          # same spot on the unfolded sheet (if known)
    sheet_line: list[list[float]] | None = None     # crease involved, on the unfolded sheet
    last_step: int | None = None                    # issue still present up to this step


@dataclass
class Report:
    model: str
    steps: int
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {"model": self.model, "steps": self.steps, "ok": self.ok,
                "issues": [asdict(i) for i in self.issues]}

    def to_text(self) -> str:
        head = (f"{self.model}: {self.steps} steps, {len(self.errors)} error(s), "
                f"{len(self.warnings)} warning(s)")
        lines = [head]
        for i in self.issues:
            lines.append(f"  [{i.severity}] {i.kind}: {i.message}")
        if not self.issues:
            lines.append("  all steps physically valid")
        return "\n".join(lines)


def _pt(p) -> list[float]:
    return [round(float(p[0]), 4), round(float(p[1]), 4)]


def _fmt(p) -> str:
    return f"({p[0]:.3f}, {p[1]:.3f})"


# ---- geometry helpers -----------------------------------------------------------------

def _bbox(poly):
    return tuple(poly.min(axis=0).tolist()), tuple(poly.max(axis=0).tolist())


def _bb_overlap(a, b, pad=1e-9) -> bool:
    """Boxes (lo, hi) in 2D or 3D overlap (plain floats: called millions of times)."""
    alo, ahi = a
    blo, bhi = b
    for k in range(len(alo)):
        if ahi[k] < blo[k] - pad or bhi[k] < alo[k] - pad:
            return False
    return True


def _overlap_poly(p, q):
    inter = g.clip_convex(p, q)
    if len(inter) < 3 or g.area(inter) <= OVERLAP_TOL:
        return None
    return inter


def _chord(poly, a, u):
    """Parameter interval (along u from a) where the line (a, u) crosses the interior of a convex
    polygon, or None if the polygon doesn't straddle the line."""
    d = u[0] * (poly[:, 1] - a[1]) - u[1] * (poly[:, 0] - a[0])  # = cross(u, p - a) per vertex
    if d.max() <= TOL or d.min() >= -TOL:
        return None
    ts = []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        dp, dq = d[i], d[(i + 1) % n]
        if abs(dp) <= TOL:
            ts.append(float(np.dot(p - a, u)))
        if (dp > TOL and dq < -TOL) or (dp < -TOL and dq > TOL):
            x = p + (q - p) * (dp / (dp - dq))
            ts.append(float(np.dot(x - a, u)))
    return (min(ts), max(ts)) if len(ts) >= 2 else None


def _iv_overlap(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    return (lo, hi) if hi - lo > TOL else None


# ---- state structure: edges between adjacent faces ------------------------------------

@dataclass
class _Edge:
    i: int
    j: int
    kind: str          # "taco" (folded over the edge), "flat" (same placement), "tear"
    p: np.ndarray      # orig endpoints
    q: np.ndarray
    a: np.ndarray      # folded line: point, unit direction, parameter interval [t0, t1]
    u: np.ndarray
    t: tuple
    side_i: int        # side of the folded line face i lies on (+1 / -1)
    side_j: int


def _face_edges(state: PaperState) -> list[_Edge]:
    faces = state.faces
    origs = [f.orig for f in faces]
    boxes = [_bbox(o) for o in origs]
    folded = [f.folded for f in faces]
    out = []
    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
            if not _bb_overlap(boxes[i], boxes[j], 1e-6):
                continue
            e = g.shared_edge(origs[i], origs[j])
            if e is None:
                continue
            p, q = np.asarray(e[0], float), np.asarray(e[1], float)
            if np.linalg.norm(q - p) < TOL:
                continue
            Ti, Tj = faces[i].T, faces[j].T
            if g.allclose(Ti, Tj, atol=TTOL):
                kind = "flat"
            elif g.allclose(Ti @ g.reflection(p, q - p), Tj, atol=TTOL):
                kind = "taco"
            else:
                kind = "tear"
            fp, fq = g.apply(Ti, p), g.apply(Ti, q)
            u = g.unit(fq - fp)
            a = fp
            si = 1 if g.side(a, u, folded[i].mean(axis=0)) > 0 else -1
            sj = 1 if g.side(a, u, folded[j].mean(axis=0)) > 0 else -1
            out.append(_Edge(i, j, kind, p, q, a, u, (0.0, float(np.linalg.norm(fq - fp))), si, sj))
    return out


def _sides(e: _Edge, ref: _Edge) -> tuple[int, int]:
    """Sides of e's faces i, j measured against ref's line direction (same line assumed)."""
    f = 1 if float(np.dot(e.u, ref.u)) > 0 else -1
    return e.side_i * f, e.side_j * f


def _collinear(e1: _Edge, e2: _Edge):
    """Overlap interval (in e1's line parameters) of two edges on the same folded line, or None."""
    if abs(g.cross(e1.u, e2.u)) > ANG_TOL or abs(g.side(e1.a, e1.u, e2.a)) > TOL:
        return None
    s0 = float(np.dot(e2.a - e1.a, e1.u))
    s1 = s0 + float(np.dot(e2.u, e1.u)) * (e2.t[1] - e2.t[0])
    return _iv_overlap(e1.t, (min(s0, s1), max(s0, s1)))


# ---- 1. layer order of the flat state -------------------------------------------------

class _Order:
    def __init__(self, state: PaperState):
        self.above = state.above

    def r(self, x, y):
        if (x, y) in self.above:
            return True
        if (y, x) in self.above:
            return False
        return None

    def between(self, c, a, b):
        """True / False if c lies strictly between a and b; None if some order is unknown."""
        ra, rb = self.r(a, c), self.r(c, b)
        if ra is None or rb is None:
            return None
        return ra == rb  # a > c > b  or  b > c > a


def check_layers(state: PaperState, k: int, edges: list[_Edge] | None = None) -> list[Issue]:
    issues: list[Issue] = []
    faces = state.faces
    folded = [f.folded for f in faces]
    boxes = [_bbox(p) for p in folded]
    order = _Order(state)
    edges = _face_edges(state) if edges is None else edges

    # tears
    for e in edges:
        if e.kind == "tear":
            issues.append(Issue(k, "tear", "error",
                                f"step {k}: faces {e.i} and {e.j} are joined on the sheet along "
                                f"{_fmt(e.p)}-{_fmt(e.q)} (unfolded-sheet coordinates) but are no longer "
                                "connected after this step: the paper would have to tear.",
                                [e.i, e.j], _pt((e.p + e.q) / 2)))

    # every overlapping pair ordered
    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
            if not _bb_overlap(boxes[i], boxes[j]):
                continue
            if order.r(i, j) is None:
                inter = _overlap_poly(folded[i], folded[j])
                if inter is not None:
                    issues.append(Issue(k, "unordered_overlap", "error",
                                        f"step {k}: faces {i} and {j} overlap near {_fmt(inter.mean(0))} "
                                        "but the simulator has no layer order for them (simulator bug), so "
                                        "this spot can't be checked.",
                                        [i, j], _pt(inter.mean(0))))

    # transitivity at points covered by three faces
    ups: dict[int, list[int]] = {}
    for i, j in state.above:
        ups.setdefault(j, []).append(i)  # i above j
    seen = set()
    for j0, i0 in ((j, i) for i, j in state.above):  # i0 above j0
        for k0 in ups.get(i0, ()):  # k0 above i0
            if (j0, k0) not in state.above:  # j0 above k0 would close the cycle
                continue
            key = frozenset((i0, j0, k0))
            if key in seen:
                continue
            seen.add(key)
            inter = _overlap_poly(folded[i0], folded[j0])
            inter = _overlap_poly(inter, folded[k0]) if inter is not None else None
            if inter is not None:
                c = inter.mean(0)
                issues.append(Issue(k, "cyclic_order", "error",
                                    f"step {k}: impossible layer order near {_fmt(c)}: face {k0} is above "
                                    f"face {i0}, which is above face {j0}, which is above face {k0} again.",
                                    sorted(key), _pt(c)))
    if not state.acyclic and not any(i.kind == "cyclic_order" for i in issues):
        issues.append(Issue(k, "no_global_stack", "warning",
                            f"step {k}: the layers have a cyclic order (no single bottom-to-top stack). "
                            "This can be physical (pinwheel-like overlaps) but drawings may be wrong.", []))

    tacos = [e for e in edges if e.kind == "taco"]
    flats = [e for e in edges if e.kind == "flat"]

    # taco-tortilla: a face reaching across a folded crease can't lie between its two layers
    for e in tacos:
        if e.side_i != e.side_j:
            continue  # not folded flat onto itself (shouldn't happen for a reflection)
        sgn = e.side_i
        seg_box = (np.minimum(e.a, e.a + e.u * e.t[1]) - TOL, np.maximum(e.a, e.a + e.u * e.t[1]) + TOL)
        for c in range(len(faces)):
            if c in (e.i, e.j) or not _bb_overlap(boxes[c], seg_box):
                continue
            ch = _chord(folded[c], e.a, e.u)
            if ch is None:
                continue
            iv = _iv_overlap(e.t, ch)
            if iv is None:
                continue
            if order.between(c, e.i, e.j):
                x = e.a + e.u * (iv[0] + iv[1]) / 2
                issues.append(Issue(k, "taco_tortilla", "error",
                                    f"step {k}: face {c} lies between the two layers of a fold (faces {e.i} "
                                    f"and {e.j}) but reaches across that fold's crease near {_fmt(x)}: "
                                    "the paper would pass through the crease.",
                                    [e.i, e.j, c], _pt(x), sheet_line=[_pt(e.p), _pt(e.q)]))
        for f in flats:
            iv = _collinear(e, f)
            if iv is None or {f.i, f.j} & {e.i, e.j}:
                continue
            c = f.i if _sides(f, e)[0] == sgn else f.j
            if order.between(c, e.i, e.j):
                x = e.a + e.u * (iv[0] + iv[1]) / 2
                issues.append(Issue(k, "taco_tortilla", "error",
                                    f"step {k}: face {c} lies between the two layers of a fold (faces {e.i} "
                                    f"and {e.j}) but continues flat across that fold's crease near {_fmt(x)} "
                                    f"(into face {f.i if c == f.j else f.j}): the paper would pass through the "
                                    "crease.",
                                    [e.i, e.j, f.i, f.j], _pt(x), sheet_line=[_pt(e.p), _pt(e.q)]))

    # taco-taco: folds along the same line on the same side must nest or stack, not interleave
    for e1, e2 in itertools.combinations(tacos, 2):
        if e1.side_i != _sides(e2, e1)[0] or {e1.i, e1.j} & {e2.i, e2.j}:
            continue
        iv = _collinear(e1, e2)
        if iv is None:
            continue
        b1, b2 = order.between(e2.i, e1.i, e1.j), order.between(e2.j, e1.i, e1.j)
        if b1 is None or b2 is None:
            continue
        if b1 != b2:
            x = e1.a + e1.u * (iv[0] + iv[1]) / 2
            issues.append(Issue(k, "taco_taco", "error",
                                f"step {k}: two folds share the crease line near {_fmt(x)} and their layers "
                                f"interleave (faces {e1.i},{e1.j} vs {e2.i},{e2.j}): one fold would cut "
                                "through the other.",
                                [e1.i, e1.j, e2.i, e2.j], _pt(x), sheet_line=[_pt(e1.p), _pt(e1.q)]))

    # tortilla-tortilla (i, j, k, k): a face reaching across a flat seam can't lie between the
    # seam's two sides (they are one continuous sheet there)
    for f in flats:
        seg_box = (np.minimum(f.a, f.a + f.u * f.t[1]) - TOL, np.maximum(f.a, f.a + f.u * f.t[1]) + TOL)
        for c in range(len(faces)):
            if c in (f.i, f.j) or not _bb_overlap(boxes[c], seg_box):
                continue
            ch = _chord(folded[c], f.a, f.u)
            if ch is None:
                continue
            iv = _iv_overlap(f.t, ch)
            if iv is None:
                continue
            if order.between(c, f.i, f.j):
                x = f.a + f.u * (iv[0] + iv[1]) / 2
                issues.append(Issue(k, "tortilla_tortilla", "error",
                                    f"step {k}: near {_fmt(x)} face {c} lies above one side and below the other "
                                    f"side of a single flat piece of paper (faces {f.i} and {f.j}, joined flat "
                                    "there) while reaching across the join: the paper would have to pass "
                                    "through itself.",
                                    [f.i, f.j, c], _pt(x), sheet_line=[_pt(f.p), _pt(f.q)]))

    # tortilla-tortilla: flat-joined pairs along one line keep their order on both sides
    for e1, e2 in itertools.combinations(flats, 2):
        if {e1.i, e1.j} & {e2.i, e2.j}:
            continue
        iv = _collinear(e1, e2)
        if iv is None:
            continue
        a2, b2 = (e2.i, e2.j) if _sides(e2, e1)[0] == e1.side_i else (e2.j, e2.i)
        r1, r2 = order.r(e1.i, a2), order.r(e1.j, b2)
        if r1 is None or r2 is None:
            continue
        if r1 != r2:
            x = e1.a + e1.u * (iv[0] + iv[1]) / 2
            issues.append(Issue(k, "tortilla_tortilla", "error",
                                f"step {k}: near {_fmt(x)} two flat sheets cross each other along a line: "
                                f"faces {e1.i}/{e1.j} and {a2}/{b2} swap order there, so the sheets would "
                                "have to pass through each other.",
                                [e1.i, e1.j, e2.i, e2.j], _pt(x), sheet_line=[_pt(e1.p), _pt(e1.q)]))
    for iss in issues:
        if iss.kind == "tear":
            iss.sheet_point = iss.point
        elif iss.point is not None and iss.faces:
            iss.sheet_point = _pt(g.apply(np.linalg.inv(faces[iss.faces[0]].T), np.array(iss.point)))
    return issues


# ---- 2. crease pattern: Kawasaki / Maekawa / dangling creases --------------------------

def folded_creases(state: PaperState, edges: list[_Edge] | None = None) -> list[tuple]:
    """Currently folded creases (p, q, "M"/"V") in unfolded-sheet coordinates (from the faces)."""
    edges = _face_edges(state) if edges is None else edges
    out = []
    for e in edges:
        if e.kind != "taco":
            continue
        fi = state.faces[e.i]
        r = _Order(state).r(e.j, e.i)
        if r is None:
            asg = "U"
        else:
            asg = "V" if r == fi.front_up else "M"
        out.append((tuple(e.p), tuple(e.q), asg))
    return out


def _on_boundary(v, size) -> bool:
    return min(v[0], v[1], size - v[0], size - v[1]) < 1e-6


def check_crease_pattern(creases, size: float = 1.0, k: int = 0) -> list[Issue]:
    """Kawasaki + Maekawa at every interior vertex of a crease pattern.

    creases: (p, q, assignment) segments on the unfolded sheet, assignment "M" / "V" ("U" = unknown,
    skipped for Maekawa). Segments may overlap / touch in the middle (T-junctions are handled).
    """
    segs = [(np.asarray(p, float), np.asarray(q, float), a) for p, q, a in creases]
    segs = [s for s in segs if np.linalg.norm(s[1] - s[0]) > TOL]
    verts: list[np.ndarray] = []
    for p, q, _ in segs:
        for v in (p, q):
            if not any(np.linalg.norm(v - w) < 1e-6 for w in verts):
                verts.append(v)
    issues = []
    for v in verts:
        if _on_boundary(v, size):
            continue
        rays: list[tuple[float, str]] = []
        for p, q, asg in segs:
            d = q - p
            L = float(np.linalg.norm(d))
            if abs(g.cross(d / L, v - p)) > 1e-6:
                continue
            t = float(np.dot(v - p, d)) / (L * L)
            if t < -1e-6 / L or t > 1 + 1e-6 / L:
                continue
            if t > 1e-6 / L:
                rays.append((math.atan2(-d[1], -d[0]), asg))
            if t < 1 - 1e-6 / L:
                rays.append((math.atan2(d[1], d[0]), asg))
        # merge duplicate rays (overlapping segments)
        rays.sort()
        merged: list[list] = []
        for ang, asg in rays:
            if merged and abs(ang - merged[-1][0]) < ANG_TOL:
                if merged[-1][1] != asg and "U" not in (asg, merged[-1][1]):
                    merged[-1][1] = "conflict"
                continue
            merged.append([ang, asg])
        if len(merged) > 1 and abs(merged[0][0] + 2 * math.pi - merged[-1][0]) < ANG_TOL:
            merged.pop()
        if not merged:
            continue
        n = len(merged)
        where = f"at {_fmt(v)} on the unfolded sheet"
        if n % 2 or n < 2:
            issues.append(Issue(k, "kawasaki", "error",
                                f"step {k}: {n} folded crease(s) meet {where}; a flat-folded vertex needs an "
                                "even number (at least 2). The paper can't lie flat there.", [], _pt(v)))
            continue
        angs = [m[0] for m in merged]
        sectors = [(angs[(i + 1) % n] - angs[i]) % (2 * math.pi) for i in range(n)]
        alt = sum(sectors[0::2]) - sum(sectors[1::2])
        if abs(alt) > 1e-4:
            issues.append(Issue(k, "kawasaki", "error",
                                f"step {k}: Kawasaki fails {where}: alternate angles between the creases sum to "
                                f"{math.degrees(sum(sectors[0::2])):.1f} and {math.degrees(sum(sectors[1::2])):.1f} "
                                "degrees (need 180 each). The paper can't lie flat there.", [], _pt(v)))
        asgs = [m[1] for m in merged]
        if "conflict" in asgs:
            issues.append(Issue(k, "maekawa", "error",
                                f"step {k}: a crease {where} is both mountain and valley.", [], _pt(v)))
            continue
        if "U" in asgs:
            continue
        M, V = asgs.count("M"), asgs.count("V")
        if abs(M - V) != 2:
            issues.append(Issue(k, "maekawa", "error",
                                f"step {k}: Maekawa fails {where}: {M} mountain and {V} valley creases meet "
                                "there (need the counts to differ by exactly 2). The layer order around this "
                                "point is impossible.", [], _pt(v)))
    for iss in issues:
        iss.sheet_point = iss.point
    return issues


def dangling_creases(creases, size: float = 1.0) -> list[np.ndarray]:
    """Crease end points inside the sheet that don't meet another crease (a collinear crease only
    counts if it carries the line on past the point)."""
    segs = [(np.asarray(c.p, float), np.asarray(c.q, float)) for c in creases]
    segs = [s for s in segs if np.linalg.norm(s[1] - s[0]) > TOL]
    out = []
    for i, (p, q) in enumerate(segs):
        for v, w in ((p, q), (q, p)):
            if _on_boundary(v, size):
                continue
            outward = g.unit(v - w)
            touched = False
            for j, (r, s) in enumerate(segs):
                if j == i:
                    continue
                d = s - r
                L = float(np.linalg.norm(d))
                if abs(g.cross(d / L, v - r)) > 1e-6:
                    continue
                t = float(np.dot(v - r, d)) / (L * L)
                if not (-1e-6 / L <= t <= 1 + 1e-6 / L):
                    continue
                if abs(g.cross(d / L, outward)) > ANG_TOL:
                    touched = True  # another crease meets / crosses here
                elif max(float(np.dot(r - v, outward)), float(np.dot(s - v, outward))) > 1e-6:
                    touched = True  # the same line continues past v
                if touched:
                    break
            if not touched and not any(np.linalg.norm(v - x) < 1e-6 for x in out):
                out.append(v)
    return out


# ---- 3. motion of a step --------------------------------------------------------------

def _pieces(before: PaperState, after: PaperState) -> list[dict]:
    """Common refinement of before/after faces (orig coords) with both placements."""
    out = []
    bb = [_bbox(f.orig) for f in before.faces]
    ab = [_bbox(f.orig) for f in after.faces]
    for ib, fb in enumerate(before.faces):
        for ia, fa in enumerate(after.faces):
            if not _bb_overlap(bb[ib], ab[ia]):
                continue
            inter = g.clip_convex(fb.orig, fa.orig)
            if len(inter) < 3 or g.area(inter) < 1e-9:
                continue
            M = fa.T @ np.linalg.inv(fb.T)
            out.append({"orig": inter, "bi": ib, "ai": ia, "Tb": fb.T, "Ta": fa.T, "M": M,
                        "moving": not g.allclose(M, np.eye(3), atol=TTOL)})
    return out


def _layer_from(state: PaperState, face: int, pt, top: bool) -> int:
    """1-based position of `face` in the stack at pt, counted from the top (or bottom)."""
    n = 0
    for k, f in enumerate(state.faces):
        if k != face and g.contains(f.folded, pt, tol=-1e-7):
            r = _Order(state).r(k, face)
            if r is (True if top else False):
                n += 1
    return n + 1


def _sweep_hits(state: PaperState, pieces, key: str, sign: int):
    """(moving face, static face, overlap polygon) where a static piece is on the swing side."""
    T, idx = ("Tb", "bi") if key == "b" else ("Ta", "ai")
    order = _Order(state)
    mov = [(p, g.apply(p[T], p["orig"])) for p in pieces if p["moving"]]
    sta = [(p, g.apply(p[T], p["orig"])) for p in pieces if not p["moving"]]
    hits = {}
    for pm, polym in mov:
        bm = _bbox(polym)
        for ps, polys in sta:
            if not _bb_overlap(bm, _bbox(polys)):
                continue
            r = order.r(ps[idx], pm[idx])  # static above moving?
            if r is None or r != (sign > 0):
                continue
            inter = _overlap_poly(polym, polys)
            if inter is None:
                continue
            kk = (pm[idx], ps[idx])
            if kk not in hits or g.area(inter) > g.area(hits[kk]):
                hits[kk] = inter
    return hits


def check_motion(before: PaperState, after: PaperState, step: dict, k: int) -> list[Issue]:
    """Combinatorial swing check for plain valley / mountain / unfold steps."""
    op = step.get("op")
    if op not in ("valley", "mountain", "unfold"):
        return []
    pieces = _pieces(before, after)
    moving = [p for p in pieces if p["moving"]]
    if not moving:
        return []
    if not all(g.allclose(p["M"], moving[0]["M"], atol=1e-5) for p in moving):
        return [Issue(k, "not_rigid", "warning",
                      f"step {k}: the moving paper doesn't move as one piece; motion not checked.", [])]
    signs = [1, -1] if op == "unfold" else [1 if op == "valley" else -1]
    results = []
    for sign in signs:
        pre = _sweep_hits(before, pieces, "b", sign)
        post = _sweep_hits(after, pieces, "a", sign)
        results.append((sign, pre, post))
        if not pre and not post:
            return []
    sign, pre, post = min(results, key=lambda r: len(r[1]) + len(r[2]))
    issues = []
    up = sign > 0
    word = "above" if up else "below"
    tuck = step.get("tuck") is not None
    verb = {"valley": "valley fold", "mountain": "mountain fold", "unfold": "unfold"}[op]
    if pre:
        (m, s), inter = max(pre.items(), key=lambda kv: g.area(kv[1]))
        c = inter.mean(0)
        statics = sorted({s for _, s in pre})
        lay = _layer_from(before, s, c, top=up)
        fix = ("include those layers in the fold (e.g. \"layers\": \"all\" or a larger \"top\" count), "
               "fold them out of the way first, or use a mountain fold") if up else \
              ("include those layers in the fold (e.g. \"layers\": \"all\" or a larger \"bottom\" count), "
               "fold them out of the way first, or use a valley fold")
        issues.append(Issue(k, "collision", "error",
                            f"step {k} ({verb}): the moving flap would pass through {len(statics)} stationary "
                            f"piece(s) of paper lying {word} it, e.g. layer {lay} from the {'top' if up else 'bottom'} "
                            f"near {_fmt(c)} (as the paper lies before the step). Fix: {fix}.",
                            sorted({m for m, _ in pre} | set(statics)), _pt(c)))
    if post:
        (m, s), inter = max(post.items(), key=lambda kv: g.area(kv[1]))
        c = inter.mean(0)
        statics = sorted({s for _, s in post})
        if tuck:
            issues.append(Issue(k, "tuck", "warning",
                                f"step {k} ({verb}, tuck): the flap ends {'under' if up else 'over'} "
                                f"{len(statics)} stationary piece(s) of paper near {_fmt(c)}; those layers must be lifted "
                                "open to slide it in (not checked further).",
                                sorted({m for m, _ in post} | set(statics)), _pt(c)))
        else:
            issues.append(Issue(k, "collision", "error",
                                f"step {k} ({verb}): the flap lands {'under' if up else 'over'} {len(statics)} "
                                f"stationary piece(s) of paper near {_fmt(c)} (as the paper lies after the step), so it "
                                "would have to pass through them on the way.",
                                sorted({m for m, _ in post} | set(statics)), _pt(c)))
    return issues


# ---- 3b. optional sampled check on the animation --------------------------------------

def _poly_hits(P, Q, margin):
    """A point where an edge of convex planar polygon P pierces the inset interior of Q, or None."""
    n = np.cross(Q[1] - Q[0], Q[2] - Q[0])
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return None
    n /= nn
    d = (P - Q[0]) @ n
    for i in range(len(P)):
        j = (i + 1) % len(P)
        if (d[i] > 1e-9 and d[j] < -1e-9) or (d[i] < -1e-9 and d[j] > 1e-9):
            x = P[i] + (P[j] - P[i]) * (d[i] / (d[i] - d[j]))
            inside = True
            for a in range(len(Q)):
                b = (a + 1) % len(Q)
                e = Q[b] - Q[a]
                inward = np.cross(n, e)
                inward /= np.linalg.norm(inward)
                if (x - Q[a]) @ inward < margin:
                    inside = False
                    break
            if inside:
                return x
    return None


def check_animation(model: dict, samples=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                    dz: float = 0.0, margin: float = 0.01) -> list[Issue]:
    """Slow mode: sample the connected animation (kinematics.py) and look for faces passing through
    each other. Pieces that touch on the sheet are skipped (they meet at hinges); hits within
    `margin` of a piece's edge are ignored (layer gap / hinge offsets). Warnings only: the animation
    is one plausible motion, not the only one."""
    from .kinematics import frames
    from .steps_export import build_timeline

    tl = build_timeline(model)
    size = float(tl["size"])
    issues = []
    for s in tl["steps"]:
        if s["op"] in ("flip", "rotate") or "kin_order" not in s:
            continue
        pcs = s["pieces"]
        origs = [g.ccw(np.asarray(p["orig"], float)) for p in pcs]
        touch = set()
        for i in range(len(origs)):
            for j in range(i + 1, len(origs)):
                if _bb_overlap(_bbox(origs[i]), _bbox(origs[j]), 1e-6) and any(
                        np.min(np.linalg.norm(origs[j] - v, axis=1)) < 1e-6 or g.contains(origs[j], v, 1e-7)
                        for v in origs[i]):
                    touch.add((i, j))
                elif _bb_overlap(_bbox(origs[i]), _bbox(origs[j]), 1e-6) and any(
                        g.contains(origs[i], v, 1e-7) for v in origs[j]):
                    touch.add((i, j))
        found = None
        for t in samples:
            G = frames(pcs, s["kin_order"], t, dz=dz * size)
            P3 = [(Gi @ np.c_[o, np.zeros(len(o)), np.ones(len(o))].T).T[:, :3] for Gi, o in zip(G, origs)]
            boxes = [(p.min(0), p.max(0)) for p in P3]
            for i in range(len(P3)):
                for j in range(i + 1, len(P3)):
                    if (i, j) in touch or not _bb_overlap(boxes[i], boxes[j]):
                        continue
                    x = _poly_hits(P3[i], P3[j], margin * size)
                    if x is None:
                        x = _poly_hits(P3[j], P3[i], margin * size)
                    if x is not None:
                        found = (t, i, j, x)
                        break
                if found:
                    break
            if found:
                break
        if found:
            t, i, j, x = found
            k = s["index"]
            issues.append(Issue(k, "sampled_collision", "warning",
                                f"step {k} ({s['op']}): in the animated motion, at {int(t * 100)}% of the move "
                                f"the paper passes through itself near ({x[0]:.3f}, {x[1]:.3f}) (as seen before "
                                f"the step; pieces from sheet points {_fmt(origs[i].mean(0))} and "
                                f"{_fmt(origs[j].mean(0))}).",
                                [int(pcs[i]["layer_before"]), int(pcs[j]["layer_before"])], _pt(x[:2])))
    return issues


# ---- 1b. the faces still make up the sheet ---------------------------------------------

def check_sheet(state: PaperState, k: int) -> list[Issue]:
    """Every face is placed by a rigid move (no stretch), the faces don't share sheet area (no
    duplicated paper) and together they cover the whole sheet (no lost paper)."""
    issues: list[Issue] = []
    faces = state.faces
    for i, f in enumerate(faces):
        A = f.T[:2, :2]
        if not g.allclose(A.T @ A, np.eye(2), atol=1e-9) or not g.allclose(f.T[2], [0, 0, 1], atol=1e-12):
            c = f.orig.mean(0)
            issues.append(Issue(k, "stretch", "error",
                                f"step {k}: face {i} (around {_fmt(c)} on the unfolded sheet) is stretched or "
                                "squashed: it is not just moved, turned or flipped. Paper can't stretch "
                                "(simulator bug).", [i], _pt(f.folded.mean(0)), _pt(c)))
    boxes = [_bbox(f.orig) for f in faces]
    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
            if not _bb_overlap(boxes[i], boxes[j], -1e-9):
                continue
            inter = _overlap_poly(faces[i].orig, faces[j].orig)
            if inter is not None:
                c = inter.mean(0)
                issues.append(Issue(k, "paper_overlap", "error",
                                    f"step {k}: faces {i} and {j} are both made from the same part of the "
                                    f"sheet near {_fmt(c)} (unfolded-sheet coordinates): that paper exists "
                                    "twice (simulator bug).", [i, j], None, _pt(c)))
    total, want = sum(f.area for f in faces), state.size ** 2
    if abs(total - want) > 1e-6 * max(want, 1.0):
        what = "missing" if total < want else "extra"
        issues.append(Issue(k, "paper_lost", "error",
                            f"step {k}: the pieces of paper add up to area {total:.4f} but the sheet has "
                            f"area {want:.4f}: some paper is {what} (simulator bug).", []))
    return issues


# ---- 3c. step-to-step invariants -----------------------------------------------------

def check_invariants(before: PaperState, after: PaperState, step: dict, k: int) -> list[Issue]:
    """Paper that moves as one block keeps its own layer order: two overlapping pieces that move
    by the same rigid motion keep their order (both still) or swap it (both turned over together).
    This also keeps every folded crease that the step doesn't bend folded the same way (its two
    sides are such a pair). Plain folds / unfold: all pieces; macros (several sub-moves): only
    pieces that stay still are compared."""
    if step.get("op") in ("flip", "rotate"):
        return []
    pieces = _pieces(before, after)
    plain = step.get("op") in ("valley", "mountain", "unfold")
    polys = [g.apply(p["Tb"], p["orig"]) for p in pieces]
    boxes = [_bbox(q) for q in polys]
    bad = []
    for x in range(len(pieces)):
        px = pieces[x]
        if px["moving"] and not plain:
            continue
        for y in range(x + 1, len(pieces)):
            py = pieces[y]
            if px["bi"] == py["bi"] or px["ai"] == py["ai"]:
                continue
            if not g.allclose(px["M"], py["M"], atol=1e-6):
                continue
            if not _bb_overlap(boxes[x], boxes[y]):
                continue
            rb = before.is_above(px["bi"], py["bi"])
            ra = after.is_above(px["ai"], py["ai"])
            if rb is None or ra is None:
                continue
            flipped = bool(np.linalg.det(px["M"][:2, :2]) < 0)
            if ra != (rb != flipped):
                inter = _overlap_poly(polys[x], polys[y])
                if inter is None:
                    continue
                bad.append((px, py, inter))
    if not bad:
        return []
    px, py, inter = max(bad, key=lambda b: g.area(b[2]))
    c = inter.mean(0)
    sheet = g.apply(np.linalg.inv(px["Tb"]), c)
    how = "turned over together" if px["moving"] else "stayed still"
    return [Issue(k, "order_changed", "error",
                  f"step {k} ({step.get('op')}): {len(bad)} pair(s) of layers that {how} swapped their "
                  f"order, e.g. near {_fmt(c)} (as the paper lies before the step): faces {px['ai']} and "
                  f"{py['ai']} (after the step). Layers that move as one block can't pass through each "
                  "other (simulator bug).",
                  [px["ai"], py["ai"]], _pt(g.apply(px["Ta"], sheet)), _pt(sheet))]


def check_selection(after: PaperState, step: dict, k: int, strict: bool = False) -> list[Issue]:
    """The engine had to drag along more layers than the step asked for (paper joined to the
    chosen layers by an edge off the fold line). Set by engine.fold as `selection_grew`."""
    note = getattr(after, "selection_grew", None)
    if not note:
        return []
    return [Issue(k, "selection_grew", "error" if strict else "warning",
                  f"step {k} ({step.get('op')}): the step asks to fold {note['asked']}, but those layers "
                  f"are joined to other paper near {_fmt(note['point'])} by an edge that is not on the fold "
                  f"line, so the selection grew from {note['n']} to {note['m']} layers at the thickest spot (the joined paper "
                  "must move too or it would tear). Fix: ask for all the layers that move, or fold along "
                  "a line through that joint.",
                  [], _pt(note["point"]), _pt(note["sheet_point"]))]


# ---- driver ---------------------------------------------------------------------------

def check_state(state: PaperState, k: int) -> list[Issue]:
    """Flat-state checks: layer order constraints + Kawasaki / Maekawa of the folded creases."""
    edges = _face_edges(state)
    issues = check_sheet(state, k)
    issues += check_layers(state, k, edges)
    issues += check_crease_pattern(folded_creases(state, edges), state.size, k)
    return issues


def check_step(before: PaperState, after: PaperState, step: dict, k: int,
               strict: bool = False) -> list[Issue]:
    """All per-step checks (motion + step invariants + resulting flat state). strict: a layer
    selection that had to grow (drag-along) is an error instead of a warning."""
    if step.get("op") in ("flip", "rotate"):
        return []  # view change only: the layer structure is unchanged
    return (check_selection(after, step, k, strict) + check_motion(before, after, step, k)
            + check_invariants(before, after, step, k) + check_state(after, k))


STATE_KINDS = ("stretch", "paper_overlap", "paper_lost", "tear", "unordered_overlap", "cyclic_order", "no_global_stack", "taco_tortilla",
               "taco_taco", "tortilla_tortilla", "kawasaki", "maekawa")


def _same_spot(a: Issue, b: Issue) -> bool:
    if a.kind != b.kind:
        return False
    if a.sheet_point is None or b.sheet_point is None:
        return a.sheet_point is None and b.sheet_point is None
    if float(np.linalg.norm(np.subtract(a.sheet_point, b.sheet_point))) < 0.02:
        return True
    if a.sheet_line and b.sheet_line:  # same crease line (faces get split by later folds)
        p, q = (np.asarray(x, float) for x in a.sheet_line)
        u = g.unit(q - p)
        if not all(abs(g.cross(u, np.asarray(x, float) - p)) < 2e-3 for x in b.sheet_line):
            return False
        ta = sorted((0.0, float(np.dot(q - p, u))))
        tb = sorted(float(np.dot(np.asarray(x, float) - p, u)) for x in b.sheet_line)
        return min(ta[1], tb[1]) - max(ta[0], tb[0]) > 1e-3  # overlapping stretch of one crease
    return False


def validate_states(states: list[PaperState], steps: list[dict], strict: bool = False) -> list[Issue]:
    """Check every step. A flat-state problem that carries over unchanged to later steps is
    reported once (at the step that caused it) with `last_step` = the last step it is still there."""
    issues: list[Issue] = []
    reported: list[np.ndarray] = []
    open_: list[Issue] = []  # flat-state issues of the previous checked state
    for k, step in enumerate(steps, 1):
        if k >= len(states):
            break
        if step.get("op") in ("flip", "rotate"):
            continue
        now = check_step(states[k - 1], states[k], step, k, strict)
        nxt, used = [], []
        for iss in now:
            if iss.kind in STATE_KINDS:
                old = next((o for o in open_ if _same_spot(o, iss) and o not in used), None)
                if old is not None:
                    used.append(old)
                    old.last_step = k
                    nxt.append(old)
                    continue
                nxt.append(iss)
            issues.append(iss)
        open_ = nxt
        for v in dangling_creases(states[k].creases, states[k].size):
            if any(np.linalg.norm(v - w) < 1e-6 for w in reported):
                continue
            reported.append(v)
            issues.append(Issue(k, "dangling_crease", "warning",
                                f"step {k}: a crease ends at {_fmt(v)} on the unfolded sheet without meeting "
                                "another crease or the paper edge (a crease can only stop in the middle of "
                                "the paper if it is just a pinch mark).", [], _pt(v), _pt(v)))
    for iss in issues:
        if iss.last_step is not None and iss.last_step > iss.step:
            iss.message += f" (Still there after step {iss.last_step}.)"
    return issues


def _op(step) -> str:
    return str(step.get("op")) if isinstance(step, dict) else "?"


def validate_model(model: dict, sampled: bool = False, strict: bool = False) -> Report:
    """Run the model step by step and check every step. Fold errors (also malformed step data:
    missing fields, wrong types) end the run and are reported as `fold_error`. strict: see
    check_step."""
    from .engine import FoldError, Simulator

    steps = model.get("steps", [])
    rep = Report(model.get("name", "model"), len(steps))
    sim = Simulator(float(model.get("paper", {}).get("size", 1.0)))
    states = [sim.state.copy()]
    failed = False
    for k, step in enumerate(steps, 1):
        try:
            sim.apply(step)
        except (FoldError, ValueError) as e:
            rep.issues.append(Issue(k, "fold_error", "error", f"step {k} ({_op(step)}): {e}", []))
            failed = True
            break
        except KeyError as e:
            rep.issues.append(Issue(k, "fold_error", "error",
                                    f"step {k} ({_op(step)}): the step is missing the field {e} it needs.", []))
            failed = True
            break
        except (TypeError, IndexError, AttributeError, np.linalg.LinAlgError) as e:
            rep.issues.append(Issue(k, "fold_error", "error",
                                    f"step {k} ({_op(step)}): bad step data ({type(e).__name__}: {e}).", []))
            failed = True
            break
        states.append(sim.state.copy())
    rep.issues += validate_states(states, steps[:len(states) - 1], strict)
    if sampled and not failed:
        rep.issues += check_animation(model)
    rep.issues.sort(key=lambda i: (i.step, i.severity != "error"))
    return rep
