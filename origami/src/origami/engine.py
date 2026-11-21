"""Fold engine: apply steps (valley, mountain, flip, rotate, unfold + macros) to a PaperState.

Layer order is local (pairwise, see paper.py). Every fold is built
from two primitives:
  _cut(state, line, only)          split faces by a line; pieces inherit pairwise order
  _move(state, chosen, line, kind) reflect chosen faces across a line and place them
                                   relative to the faces they now overlap (valley = above,
                                   mountain = below, or a "tuck" rule)
Macros (inside_reverse, squash, petal) are short sequences of _move calls on explicit
face sets (no drag-along), so intermediate states may be "torn"; only the final state of
a step is physical. Each _move is recorded as a sub-motion in `state.motions`.
"""

from __future__ import annotations

import itertools

import numpy as np

from . import geometry as g
from .landmarks import resolve_line, resolve_line_ref, resolve_point
from .paper import Crease, Face, PaperState, overlaps

FOLD_OPS = ("valley", "mountain")
VIEW_OPS = ("flip", "rotate")
MACRO_OPS = ("inside_reverse", "squash", "petal")
RESERVED_MACROS = ("outside_reverse", "rabbit_ear", "sink")

_uids = itertools.count()


class FoldError(ValueError):
    pass


class PhysicsError(FoldError):
    """A step is well formed but physically impossible (see validate.py). Raised only in strict
    mode (`cli.run_model(model, strict=True)`); `issues` holds the validate.Issue errors."""

    def __init__(self, message: str, issues=None):
        super().__init__(message)
        self.issues = list(issues or [])


# ---- working-state helpers --------------------------------------------------------

def _clone(f: Face, **attrs) -> Face:
    nf = Face(f.orig.copy(), f.T.copy())
    for k in ("src",):
        if hasattr(f, k):
            setattr(nf, k, getattr(f, k))
    nf.uid = next(_uids)
    for k, v in attrs.items():
        setattr(nf, k, v)
    return nf


def _begin(state: PaperState) -> PaperState:
    """Working copy: every face remembers its index in `state` (src) and gets a uid."""
    st = PaperState(faces=[_clone(f, src=i) for i, f in enumerate(state.faces)],
                    creases=list(state.creases), size=state.size, above=set(state.above))
    st.motions = []
    return st


def _idx(st: PaperState, uids) -> list[int]:
    uids = set(uids)
    return [k for k, f in enumerate(st.faces) if f.uid in uids]


def _side_of(f: Face, a, u) -> int:
    c = f.folded.mean(axis=0)
    s = g.side(a, u, c)
    return 1 if s > 0 else -1


def _on_line(a, u, pts, tol=1e-6) -> bool:
    return all(abs(g.side(a, u, p)) < tol for p in pts)


def _cut(st: PaperState, a, u, only=None) -> PaperState:
    """Split faces (all, or those with uid in `only`) by line (a, u). Order is inherited."""
    a, u = np.asarray(a, float), g.unit(np.asarray(u, float))
    faces: list[Face] = []
    parent: list[int] = []
    for i, f in enumerate(st.faces):
        if only is not None and f.uid not in only:
            faces.append(f)
            parent.append(i)
            continue
        left, right = g.split_convex(f.folded, a, u)
        if left is None or right is None:
            faces.append(f)
            parent.append(i)
            continue
        inv = np.linalg.inv(f.T)
        for poly in (left, right):
            if poly is not None and len(poly) >= 3 and g.area(poly) > g.EPS:
                faces.append(_clone(Face(g.apply(inv, poly), f.T), src=f.src, _p=f.uid))
                parent.append(i)
    kids: dict[int, list[int]] = {}
    for k, p in enumerate(parent):
        kids.setdefault(p, []).append(k)
    polys = [f.folded for f in faces]
    above = set()
    for i, j in st.above:
        for x in kids[i]:
            for y in kids[j]:
                if overlaps(polys[x], polys[y]):
                    above.add((x, y))
    out = PaperState(faces=faces, creases=st.creases, size=st.size, above=above)
    out.motions = st.motions
    return out


def _hinges(st: PaperState, chosen: set[int], a, u):
    """(moving, static, edge) for every chosen/non-chosen pair joined by an edge on the line."""
    out = []
    for m in chosen:
        for s in range(len(st.faces)):
            if s in chosen:
                continue
            e = g.shared_edge(st.faces[m].orig, st.faces[s].orig)
            if e is None:
                continue
            if _on_line(a, u, g.apply(st.faces[m].T, np.array(e))):
                out.append((m, s, e))
    return out


def _reach(st: PaperState, start, upward: bool) -> set[int]:
    """Faces transitively above (upward) / below the faces in `start` (excluding start)."""
    nxt: dict[int, list[int]] = {}
    for i, j in st.above:  # i above j
        a, b = (j, i) if upward else (i, j)
        nxt.setdefault(a, []).append(b)
    seen, todo = set(), list(start)
    while todo:
        x = todo.pop()
        for y in nxt.get(x, ()):
            if y not in seen:
                seen.add(y)
                todo.append(y)
    return seen - set(start)


def _same_layer(st: PaperState, H, exclude) -> set[int]:
    """H plus the static faces that continue H flat: joined by a paper edge with the same
    transform (an uncreased or unfolded crease), transitively. A flap tucked against H lies
    against this whole sheet, also where it reaches past H itself."""
    out, todo = set(H), list(H)
    Ts = [f.T.ravel().tolist() for f in st.faces]
    while todo:
        h = todo.pop()
        Th = Ts[h]
        for k, f in enumerate(st.faces):
            if k in out or k in exclude:
                continue
            if all(abs(x - y) <= 1e-9 + 1e-5 * abs(y) for x, y in zip(Ts[k], Th)) and \
                    g.shared_edge(f.orig, st.faces[h].orig) is not None:
                out.add(k)
                todo.append(k)
    return out


def _move(st: PaperState, chosen, a, u, kind: str, tuck=None, flap=None, phase: int = 0):
    """Reflect faces `chosen` (indices) across line (a, u); return (new state, hinges).

    Placement of a moved face m against a static face s it overlaps afterwards:
      tuck None       valley: m above s; mountain: m below s
      "inside"        m goes right next to the static faces it is hinged to (H): directly
                      above them (valley) / below them (mountain); static faces above
                      (below) H stay above (below) m. H is widened to the faces that
                      continue it flat (_same_layer), so this also holds where m reaches
                      past the hinge face itself.
    For each m the static faces it overlaps are sorted by the current draw order (a linear
    extension of the pairwise order) and m is inserted at ONE cut in that list, so the new
    order stays consistent (no cycles through m).
      {"under": k}    valley: the top k static layers over m stay above it
      {"over": k}     mountain: the bottom k static layers under m stay below it
    Moving faces keep their mutual order reversed (the moving block turns over).
    """
    chosen = set(chosen)
    if not chosen:
        raise FoldError("nothing to fold")
    a, u = np.asarray(a, float), g.unit(np.asarray(u, float))
    n = len(st.faces)
    up = kind == "valley"
    hinges = _hinges(st, chosen, a, u)
    H = {s for _, s, _ in hinges}
    R = g.reflection(a, u)

    moving = [i for i in range(n) if i in chosen]
    static = [i for i in range(n) if i not in chosen]
    order = static + moving[::-1] if up else moving[::-1] + static
    pos = {old: new for new, old in enumerate(order)}
    faces = []
    for old in order:
        f = st.faces[old]
        faces.append(_clone(f, uid=f.uid, T=R @ f.T) if old in chosen else f)
    polys = [f.folded for f in faces]
    lo_b = np.array([P.min(axis=0) for P in polys])
    hi_b = np.array([P.max(axis=0) for P in polys])
    static_pos = np.array([pos[t] for t in static], dtype=int)

    cache: dict[int, set[int]] = {}
    tuck_ctx: list = []  # (Hc, hi, lo, cut) for tuck 'inside', same for every moved face

    def above_m(m: int) -> set[int]:
        """Static faces that end above moved face m: one cut in the (linear) static order."""
        if m in cache:
            return cache[m]
        pm = polys[pos[m]]
        pmlo, pmhi = lo_b[pos[m]], hi_b[pos[m]]
        if len(static_pos):  # bbox reject (same test as overlaps' own) before the exact check
            near = ~((hi_b[static_pos] < pmlo + 1e-9).any(1) | (pmhi < lo_b[static_pos] + 1e-9).any(1))
            cands = [t for t, ok in zip(static, near.tolist()) if ok]
        else:
            cands = []
        S = sorted((t for t in cands if overlaps(polys[pos[t]], pm)), key=lambda t: pos[t])
        if tuck is None:
            k = len(S) if up else 0
        elif tuck == "inside":
            if not H:
                raise FoldError("tuck 'inside': the flap is not hinged to any other layer")
            if not tuck_ctx:
                Hc0 = _same_layer(st, H, chosen)
                tuck_ctx.append((Hc0, _reach(st, Hc0, True), _reach(st, Hc0, False),  # above / below
                                 max(pos[h] for h in H) if up else min(pos[h] for h in H)))
            Hc, hi, lo, cut = tuck_ctx[0]

            def is_above(t):
                if t in Hc:
                    return not up
                if t in hi and t not in lo:
                    return True
                if t in lo and t not in hi:
                    return False
                return pos[t] > cut if up else pos[t] >= cut

            cache[m] = {t for t in S if is_above(t)}
            return cache[m]
        elif isinstance(tuck, dict) and "under" in tuck:
            k = max(0, len(S) - int(tuck["under"]))
        elif isinstance(tuck, dict) and "over" in tuck:
            k = min(len(S), int(tuck["over"]))
        else:
            raise FoldError(f"bad tuck spec: {tuck!r}")
        cache[m] = set(S[k:])
        return cache[m]

    def place(m: int, s: int) -> bool:
        """True if moved face m ends above static face s."""
        return s not in above_m(m)

    above = set()
    # bbox-overlap matrix (the same cheap reject overlaps() starts with)
    bb = ~((hi_b[:, None, :] < lo_b[None, :, :] + 1e-9).any(2) | (hi_b[None, :, :] < lo_b[:, None, :] + 1e-9).any(2))
    bb = bb.tolist()
    for x in range(len(faces)):
        bbx = bb[x]
        for y in range(x + 1, len(faces)):
            if not bbx[y]:
                continue
            ox, oy = order[x], order[y]
            mx, my = ox in chosen, oy in chosen
            if mx == my:
                r = st.is_above(ox, oy)  # cheap; test overlap only for known pairs
                if r is None or not overlaps(polys[x], polys[y]):
                    continue
                if mx:
                    r = not r
            else:
                if not overlaps(polys[x], polys[y]):
                    continue
                m, s = (ox, oy) if mx else (oy, ox)
                m_up = place(m, s)
                r = m_up if mx else not m_up
            above.add((x, y) if r else (y, x))
    out = PaperState(faces=faces, creases=st.creases, size=st.size, above=above)
    out.normalize()
    sign = 1 if up else -1
    out.motions = st.motions + [{
        "type": "reflect", "a": [float(v) for v in a], "u": [float(v) for v in u],
        "sign": sign, "phase": phase, "tuck": tuck,
        "regions": [[[float(x), float(y)] for x, y in g.ccw(st.faces[i].orig)] for i in moving],
    }]
    uid_hinges = [(st.faces[m].uid, st.faces[s].uid, e) for m, s, e in hinges]
    return out, uid_hinges


def _hinge_creases(st: PaperState, uid_hinges, kind: str) -> list[Crease]:
    by_uid = {f.uid: k for k, f in enumerate(st.faces)}
    out = []
    for mu, su, e in uid_hinges:
        m, s = by_uid[mu], by_uid[su]
        r = st.is_above(m, s)
        if r is None:
            r = kind == "valley"
        valley_from_front = r == st.faces[s].front_up
        out.append(Crease(tuple(e[0]), tuple(e[1]), "V" if valley_from_front else "M"))
    return out


def _folded_edges(st: PaperState) -> list[tuple]:
    """(p, q, assignment) for every edge where two adjacent faces are folded onto each other."""
    out = []
    n = len(st.faces)
    for i in range(n):
        for j in range(i + 1, n):
            fi, fj = st.faces[i], st.faces[j]
            if g.allclose(fi.T, fj.T, atol=1e-7):
                continue
            e = g.shared_edge(fi.orig, fj.orig)
            if e is None:
                continue
            r = st.is_above(j, i)
            if r is None:
                continue
            out.append((tuple(e[0]), tuple(e[1]), "V" if r == fi.front_up else "M"))
    return out


def _covered(e, edges) -> bool:
    p, q, asg = np.array(e[0]), np.array(e[1]), e[2]
    d = q - p
    for r, s, a2 in edges:
        if a2 != asg:
            continue
        r, s = np.array(r), np.array(s)
        v = s - r
        L2 = float(v @ v)
        if L2 < 1e-12 or abs(g.cross(v, p - r)) > 1e-6 or abs(g.cross(v, q - r)) > 1e-6:
            continue
        t0, t1 = sorted((float((p - r) @ v / L2), float((q - r) @ v / L2)))
        if t0 > -1e-6 and t1 < 1 + 1e-6 and float(d @ d) > 0:
            return True
    return False


def _contradictory(st: PaperState, grp: list[int]) -> bool:
    """Some other face lies above one piece of the group and below another. Merged, the group
    would need both orders with that face at once, so it is kept in pieces (the checker then
    reports the face ordered between the two sides of a flat join)."""
    g_ = set(grp)
    over: dict[int, set[int]] = {}   # outside face -> pieces it lies above
    under: dict[int, set[int]] = {}  # outside face -> pieces it lies below
    for i, j in st.above:
        if j in g_ and i not in g_:
            over.setdefault(i, set()).add(j)
        elif i in g_ and j not in g_:
            under.setdefault(j, set()).add(i)
    return any(q in under for q in over)


def _finish(st: PaperState, before: PaperState) -> PaperState:
    """Merge pieces of one original face that ended with the same transform; tidy up.
    Pieces whose orders with some other face contradict each other are not merged."""
    groups: dict[int, list[int]] = {}
    for k, f in enumerate(st.faces):
        groups.setdefault(f.src, []).append(k)
    can_merge = {src: len(grp) > 1 and all(g.allclose(st.faces[x].T, st.faces[grp[0]].T, atol=1e-9)
                                           for x in grp) and not _contradictory(st, grp)
                 for src, grp in groups.items()}
    new_index: dict[int, int] = {}
    faces: list[Face] = []
    for k, f in enumerate(st.faces):
        grp = groups[f.src]
        mergeable = can_merge[f.src]
        if mergeable and k != grp[0]:
            new_index[k] = new_index[grp[0]]
            continue
        new_index[k] = len(faces)
        faces.append(Face(before.faces[f.src].orig.copy(), f.T.copy()) if mergeable
                     else Face(f.orig.copy(), f.T.copy()))
    above = {(new_index[i], new_index[j]) for i, j in st.above if new_index[i] != new_index[j]}
    above = {(i, j) for i, j in above if (j, i) not in above or i < j}
    out = PaperState(faces=faces, creases=list(st.creases), size=st.size, above=above)
    out.motions = st.motions
    out.normalize()
    return out


# ---- layer selection ----------------------------------------------------------------

def _select_layers(st: PaperState, cand, layers, move_pt=None, line=None):
    if layers in (None, "all"):
        return list(cand)
    if isinstance(layers, dict) and layers.get("local"):
        return _local_layers(st, cand, layers, move_pt, line)
    if isinstance(layers, dict) and layers.get("flap"):
        rest = {k: v for k, v in layers.items() if k != "flap"} or "all"
        return _only_flap(st, _select_layers(st, cand, rest), move_pt)
    if isinstance(layers, dict) and ("top" in layers or "bottom" in layers):
        top = "top" in layers
        n = int(layers["top" if top else "bottom"])
        out = []
        for k in cand:
            # depth = how many candidate faces lie over (or under) this one
            depth = sum(st.is_above(j, k) is top for j in cand if j != k)
            if depth < n:
                out.append(k)
        return out
    raise FoldError(f"bad layers spec: {layers!r}")


def _local_layers(st: PaperState, cand, layers, move_pt, line=None):
    """{"top"|"bottom": n, "local": true}: count layers only in the stack AT the move point.

    stack = moving-side faces that contain move_pt (they all overlap there, so their
    pairwise order is total); seeds = the top (bottom) n of them. Result = seeds plus every
    moving-side face attached to a seed by a paper edge that is not on the fold line, i.e.
    the whole flap(s) those layers belong to. Faces elsewhere on the moving side are never
    counted, so an unrelated flap lying over some other part of the same face (whose order
    vs. this flap may be arbitrary) can't push the wanted layers out.
    """
    if move_pt is None:
        raise FoldError("'local' layer option needs a move point")
    if not ("top" in layers or "bottom" in layers):
        raise FoldError("'local' needs 'top' or 'bottom'")
    top = "top" in layers
    n = int(layers["top" if top else "bottom"])
    stack = [k for k in cand if g.contains(st.faces[k].folded, move_pt)]
    if not stack:
        raise FoldError("'local': the move point is not on the paper's moving side")
    # depth inside the stack; faces of one layer that merely touch at move_pt don't overlap,
    # have no order, and so share a level
    seeds = [k for k in stack if sum(st.is_above(j, k) is top for j in stack if j != k) < n]
    if line is None:
        return seeds
    return sorted(_attached(st, cand, seeds, *line))


def _attached(st: PaperState, cand, chosen, a, u) -> set[int]:
    """`chosen` plus candidates joined to it by a paper edge NOT on line (a, u) (transitively)."""
    chosen = set(chosen)
    changed = True
    while changed:
        changed = False
        for k in cand:
            if k in chosen:
                continue
            for m in list(chosen):
                e = g.shared_edge(st.faces[k].orig, st.faces[m].orig)
                if e is None:
                    continue
                if not _on_line(a, u, g.apply(st.faces[m].T, np.array(e))):
                    chosen.add(k)
                    changed = True
                    break
    return chosen


def _only_flap(st: PaperState, chosen, move_pt):
    """Keep only chosen faces connected (through shared paper edges) to one under move_pt."""
    if move_pt is None:
        raise FoldError("'flap' layer option needs a move point")
    seeds = [k for k in chosen if g.contains(st.faces[k].folded, move_pt)]
    if not seeds:
        raise FoldError("'flap': the move point is not on any of the chosen layers")
    keep, todo = set(seeds), list(seeds)
    while todo:
        k = todo.pop()
        for j in chosen:
            if j not in keep and g.shared_edge(st.faces[k].orig, st.faces[j].orig) is not None:
                keep.add(j)
                todo.append(j)
    return [k for k in chosen if k in keep]


def _split_halves(st: PaperState, flap: list[int], split=None):
    """Front half / back half of a flap's layers (by local depth inside the flap)."""
    front, back = [], []
    for k in flap:
        d_top = sum(st.is_above(j, k) is True for j in flap if j != k)
        d_bot = sum(st.is_above(k, j) is True for j in flap if j != k)
        if split is not None:
            (front if d_top < int(split) else back).append(k)
        elif d_top < d_bot:
            front.append(k)
        elif d_top > d_bot:
            back.append(k)
        else:
            raise FoldError("can't split the flap into front and back halves "
                            "(odd number of layers?); give 'split': n")
    if not front or not back:
        raise FoldError("flap needs layers in front and behind to open it")
    return front, back


# ---- plain valley / mountain --------------------------------------------------------

def fold(state: PaperState, a, u, move_pt, kind: str, layers="all", tuck=None) -> PaperState:
    """Return a new state with a valley/mountain fold along line (a, u).

    The side of the line containing move_pt moves. `layers` picks which layers on that
    side move: "all", {"top": n} or {"bottom": n}; add "flap": True to keep only the
    connected flap under move_pt, or "local": True to count the n layers only in the stack
    at move_pt (see _local_layers). Paper connected to a chosen layer by an edge that is
    NOT on the fold line is dragged along (otherwise it would tear). `tuck` changes where
    the moved flap lands in the layer order (see _move).
    """
    if kind not in FOLD_OPS:
        raise FoldError(f"unknown fold kind {kind!r}")
    a, u = np.asarray(a, float), g.unit(np.asarray(u, float))
    s = g.side(a, u, move_pt)
    if abs(s) < g.EPS:
        raise FoldError("the 'move' point lies on the fold line")
    move_side = 1 if s > 0 else -1

    st = _cut(_begin(state), a, u)
    cand = [k for k, f in enumerate(st.faces) if _side_of(f, a, u) == move_side]
    if not cand:
        raise FoldError("nothing to fold on the moving side")
    picked = _select_layers(st, cand, layers, move_pt, line=(a, u))
    # Drag along candidates attached by a non-hinge edge (otherwise the paper would tear).
    chosen = _attached(st, cand, picked, a, u)
    note = _growth_note(st, layers, picked, chosen, a, u)

    out, hinges = _move(st, chosen, a, u, kind, tuck=tuck)
    out.creases = list(state.creases) + _hinge_creases(out, hinges, kind)
    res = _finish(out, state)
    res.selection_grew = note
    return res


def _layer_count(st: PaperState, faces) -> int:
    """Layers in a set of faces: the thickest stack of them, sampled at each face's centre."""
    faces = list(faces)
    polys = [st.faces[k].folded for k in faces]
    return max((sum(g.contains(q, p.mean(axis=0), tol=-1e-7) for q in polys) for p in polys), default=0)


def _asked(layers) -> str:
    if isinstance(layers, dict) and ("top" in layers or "bottom" in layers):
        side = "top" if "top" in layers else "bottom"
        n = int(layers[side])
        where = " at the move point" if layers.get("local") else ""
        return f"only the {side} {n} layer{'s' if n != 1 else ''}{where}"
    return f"the layers {layers!r}"


def _growth_note(st: PaperState, layers, picked, chosen, a, u):
    """Description of a selection widened by drag-along (more layers than asked), else None.
    Read by validate.check_selection."""
    extra = set(chosen) - set(picked)
    if layers in (None, "all") or not extra:
        return None
    n, m = _layer_count(st, picked), _layer_count(st, chosen)
    if m <= n:
        return None  # only more pieces of the same layers
    for d in sorted(extra):
        for p in sorted(picked):
            e = g.shared_edge(st.faces[d].orig, st.faces[p].orig)
            if e is None:
                continue
            fe = g.apply(st.faces[p].T, np.array(e))
            if _on_line(a, u, fe):
                continue
            return {"asked": _asked(layers), "n": n, "m": m, "point": fe.mean(axis=0).tolist(),
                    "sheet_point": np.mean(np.array(e, float), axis=0).tolist()}
    return None


# ---- macros -------------------------------------------------------------------------

def _macro_creases(st: PaperState, before: PaperState) -> list[Crease]:
    old = _folded_edges(before)
    new = [e for e in _folded_edges(st) if not _covered(e, old)]
    return list(before.creases) + [Crease(p, q, asg) for p, q, asg in new]


def _flap_faces(st, a, u, move, layers):
    side = 1 if g.side(a, u, move) > 0 else -1
    cand = [k for k, f in enumerate(st.faces) if _side_of(f, a, u) == side]
    chosen = _select_layers(st, cand, layers, move, line=(a, u))
    return _only_flap(st, chosen, move)


def inside_reverse(state: PaperState, step: dict) -> PaperState:
    """Inside reverse fold: the flap tip beyond `line` is pushed in between its own layers.

    Flat result: every layer of the tip is reflected across the line; the front half of
    the flap's layers turns behind the front half (mountain, tucked inside), the back half
    turns in front of the back half (valley, tucked inside). Tip order stays inside the flap.
    """
    a, u, move = fold_geometry(state, step)
    a, u = np.asarray(a, float), g.unit(np.asarray(u, float))
    st = _cut(_begin(state), a, u)
    flap = _flap_faces(st, a, u, move, step.get("layers", "all"))
    front, back = _split_halves(st, flap, step.get("split"))
    hinge_uids = {st.faces[s].uid for _, s, _ in _hinges(st, set(flap), a, u)}
    flap_uids = {st.faces[k].uid for k in flap} | hinge_uids
    back_uids = {st.faces[k].uid for k in back}
    st, _ = _move(st, front, a, u, "mountain", tuck="inside", flap=_idx(st, flap_uids), phase=0)
    st, _ = _move(st, _idx(st, back_uids), a, u, "valley", tuck="inside",
                  flap=_idx(st, flap_uids), phase=0)
    st.creases = _macro_creases(st, state)
    return _finish(st, state)


def _ray(state, ref, origin, hint):
    """Direction of a line reference pointing away from `origin` (toward its far end / hint)."""
    a, u, seg = resolve_line_ref(state, ref)
    if seg is not None:
        far = max(seg, key=lambda p: np.linalg.norm(p - origin))
        return g.unit(far - origin)
    return u if np.dot(hint - origin, u) >= 0 else -u


def squash(state: PaperState, step: dict) -> PaperState:
    """Squash fold: lift a flap on its `hinge` edge, open it and flatten it symmetrically.

    The flap's `spine` (folded edge joining its front and back halves) lands on the hinge
    line. Front half: turns over the hinge to the far side; back half stays. Each half's
    spine part is folded along c = bisector of spine and hinge at their meeting point O.
    """
    ah, uh, _ = resolve_line_ref(state, step["hinge"])
    asp, usp, _ = resolve_line_ref(state, step["spine"])
    move = resolve_point(state, step["move"])
    O = g.line_intersection(ah, ah + uh, asp, asp + usp)
    dh = _ray(state, step["hinge"], O, move)
    ds = _ray(state, step["spine"], O, move)
    uc = g.unit(dh + ds)
    st = _cut(_begin(state), ah, uh)
    flap = _flap_faces(st, ah, uh, move, step.get("layers", {"top": 2}))
    front, back = _split_halves(st, flap, step.get("split"))
    front_u = {st.faces[k].uid for k in front}
    back_u = {st.faces[k].uid for k in back}
    st = _cut(st, O, uc, only=front_u | back_u)
    spine_side = 1 if g.side(O, uc, O + ds) > 0 else -1

    def spine_part(uids):
        return {f.uid for f in st.faces if f.uid in uids and _side_of(f, O, uc) == spine_side}

    front_now = {f.uid for f in st.faces if _from(f, front_u)}
    back_now = {f.uid for f in st.faces if _from(f, back_u)}
    pb, qb = spine_part(front_now), spine_part(back_now)
    st, _ = _move(st, _idx(st, front_now), ah, uh, "valley", phase=0)
    st, _ = _move(st, _idx(st, qb), O, uc, "valley", phase=0)
    uc2 = g.apply(g.reflection(ah, uh), O + uc) - O
    st, _ = _move(st, _idx(st, pb), O, uc2, "valley", phase=1)
    st.creases = _macro_creases(st, state)
    return _finish(st, state)


def _from(f: Face, uids) -> bool:
    return f.uid in uids or getattr(f, "_p", None) in uids


def petal(state: PaperState, step: dict) -> PaperState:
    """Petal fold: lift the top layer's corner `move` up along `line`; the sides fold in.

    Kite creases run from the corner to where the line meets the flap's edges (K1, K2).
    Top layer below the line: middle part turns up (valley along line). Second layer outside
    the kites folds in along the kites on top of its own layer (valley); top-layer side parts
    fold over the lifted middle (valley across the reflected kites). Each side part ends
    flat-joined to its second-layer part (the crease from the kite point to the side corner
    unfolds); together they cover the front with raw edges meeting on the centre line.
    """
    a, u, B = fold_geometry(state, step)
    a, u = np.asarray(a, float), g.unit(np.asarray(u, float))
    st = _cut(_begin(state), a, u)
    side = 1 if g.side(a, u, B) > 0 else -1
    cand = [k for k, f in enumerate(st.faces) if _side_of(f, a, u) == side]
    two = _only_flap(st, _select_layers(st, cand, step.get("layers", {"top": 2}), B, line=(a, u)), B)
    one = [k for k in _select_layers(st, two, {"top": 1}) if k in two]
    one = _only_flap(st, one, B)
    ts = [float(np.dot(p - a, u)) for k in one for p in st.faces[k].folded if abs(g.side(a, u, p)) < 1e-6]
    if len(ts) < 2:
        raise FoldError("petal: the line does not cross the top layer")
    K1, K2 = a + min(ts) * u, a + max(ts) * u
    top_u = {st.faces[k].uid for k in one}
    sec_u = {st.faces[k].uid for k in two if k not in one}
    lines = [(B, g.unit(K1 - B)), (B, g.unit(K2 - B))]
    for la, lu in lines:
        st = _cut(st, la, lu, only=top_u | sec_u)
        top_u = {f.uid for f in st.faces if _from(f, top_u)}
        sec_u = {f.uid for f in st.faces if _from(f, sec_u)}
        for f in st.faces:
            f._p = None

    def region(f):  # 0 = between kites, 1 = outside kite 1, 2 = outside kite 2
        c = f.folded.mean(axis=0)
        if np.sign(g.side(B, lines[0][1], c)) != np.sign(g.side(B, lines[0][1], K2)):
            return 1
        if np.sign(g.side(B, lines[1][1], c)) != np.sign(g.side(B, lines[1][1], K1)):
            return 2
        return 0

    S = {r: {f.uid for f in st.faces if f.uid in top_u and region(f) == r} for r in (1, 2)}
    W = {r: {f.uid for f in st.faces if f.uid in sec_u and region(f) == r} for r in (1, 2)}
    st, _ = _move(st, _idx(st, top_u), a, u, "valley", phase=0)
    # Real paper: each second-layer part outside a kite folds in along the kite line ON TOP
    # of its own layer (valley), and the top-layer side part it is joined to (along the old
    # folded edge K-side corner, which opens flat) folds over the lifted petal (valley across
    # the reflected kite line). Both end with the same transform (one flat piece), frontmost:
    # their raw edges meet on the centre line (the slit on the front of a bird base) and the
    # visible side triangles keep the prelim's outside colour.
    for r, (la, lu) in zip((1, 2), lines):
        if W[r]:
            st, _ = _move(st, _idx(st, W[r]), la, lu, "valley", phase=0)
    Rh = g.reflection(a, u)
    for r, (la, lu) in zip((1, 2), lines):
        if S[r]:
            ka = g.apply(Rh, la)
            ku = g.apply(Rh, la + lu) - ka
            st, _ = _move(st, _idx(st, S[r]), ka, ku, "valley", phase=1)
    st.creases = _macro_creases(st, state)
    return _finish(st, state)


MACROS = {"inside_reverse": inside_reverse, "squash": squash, "petal": petal}


# ---- view ops / geometry ------------------------------------------------------------

def flip(state: PaperState, axis: str = "vertical") -> None:
    lo, hi = state.bbox()
    state.apply_view(g.mirror((lo + hi) / 2, axis), reverse_layers=True)


def rotate(state: PaperState, degrees: float) -> None:
    lo, hi = state.bbox()
    state.apply_view(g.rotation((lo + hi) / 2, degrees), reverse_layers=False)


def fold_geometry(state: PaperState, step: dict):
    """Resolve a fold step into (a, u, move_pt). Used by the engine and the renderer."""
    a, u, default_move = resolve_line(state, step["line"])
    if "move" in step:
        move = resolve_point(state, step["move"])
    elif default_move is not None:
        move = default_move
    else:
        raise FoldError("fold step needs a 'move' point")
    return a, u, move


class Simulator:
    """Applies steps one at a time; keeps history so `unfold` can undo the last fold."""

    def __init__(self, size: float = 1.0):
        self.state = PaperState.square(size)
        # stack of (state before a fold, view ops done since that fold)
        self._fold_history: list[tuple[PaperState, list[dict]]] = []

    def apply(self, step: dict) -> None:
        op = step.get("op")
        if op in FOLD_OPS:
            a, u, move = fold_geometry(self.state, step)
            before = self.state.copy()
            self.state = fold(self.state, a, u, move, op, step.get("layers", "all"),
                              tuck=step.get("tuck"))
            self._fold_history.append((before, []))
        elif op in MACRO_OPS:
            before = self.state.copy()
            self.state = MACROS[op](self.state, step)
            self._fold_history.append((before, []))
        elif op in VIEW_OPS:
            self._apply_view(self.state, step)
            self.state.motions = []
            for _, ops in self._fold_history:
                ops.append(step)
        elif op == "unfold":
            if not self._fold_history:
                raise FoldError("nothing to unfold")
            before, ops = self._fold_history.pop()
            creases = self.state.creases
            self.state = before.copy()
            self.state.creases = creases  # creases stay after unfolding
            self.state.motions = []
            self.state.selection_grew = None
            for v in ops:
                self._apply_view(self.state, v)
        elif op in RESERVED_MACROS:
            raise FoldError(f"'{op}' is reserved for a future macro and not implemented yet")
        else:
            raise FoldError(f"unknown op {op!r}")

    @staticmethod
    def _apply_view(state: PaperState, step: dict) -> None:
        if step["op"] == "flip":
            flip(state, step.get("axis", "vertical"))
        else:
            rotate(state, float(step["degrees"]))
