"""MENU action mode: list sensible, legal next folds so an agent can just pick a number.

    opts = candidates(env)            # FoldEnv / Simulator / PaperState
    opts[0].id, opts[0].description, opts[0].step   # step = compiled step JSON for env.apply

Candidates are built from Huzita-Justin-style alignments of the CURRENT paper's features
(regenerated every turn, nothing per model):
- key points  K  = folded face vertices (corners, crease ends, crease intersections) + the 9 landmarks;
- outline vertices V (corners of the silhouette) are the "things you move";
- segments       = folded face edges (outline edges + visible / hidden creases and raw edges);
- derived points = where the angle bisector at a vertex (between two segments meeting there)
                   first hits the outline (the classic "pinch" points: cup pivot, petal kite points).
Lines: point onto point (O2, V -> K), through two points (O1, V-K), horizontal / vertical / 45 deg
through a point, edge onto edge (O3, bisectors of outline edges and of segments meeting at a point),
"make this edge level / upright" (crease through a point on an edge, bisecting edge and axis).
Per line and side: valley with all layers / top n / one flap / tucked inside, mountain-into-pocket;
plus squash, petal, inside-reverse candidates and unfold.
Every candidate is run through the per-step physics checker (env.try_step); illegal ones and no-ops
are dropped; results are deduped by canonical state_hash (keeping the simplest description);
ranked by simplicity (cost) and listed in sections with quotas (see FAMILIES; `more` pages a section).
"""

from __future__ import annotations

import copy
import heapq
import io
import math
from dataclasses import dataclass, field

import numpy as np

from . import geometry as g
from .engine import Simulator, _attached, _layer_count, _begin, _cut, _on_line, _select_layers, _side_of
from .match import state_hash
from .observe import outline, point_name

# Menu = a few of the simplest options of each kind (diversity beats one global ranking):
FAMILIES = {"onto": 8, "through": 4, "edges": 2, "align": 6, "flap": 8, "tuck": 4}
DEFAULT_CAP = sum(FAMILIES.values())  # plain folds
# Sections whose page mixes the line constructions (round robin: cheapest of each kind first). Single-flap
# folds by an 'onto' / 'through' line are many and cheap; without the mix they buried the 'align' ones
# (kabuto horns: fold one flap so its edge lies level / parallel to an edge) on page 3.
MIXED = ("flap",)
MIX_POOL = 4.0  # results evaluated per page in a mixed section (x quota), cheapest first
MACRO_CAP = 8  # squash / petal options, listed after the plain folds
REPEAT_CAP = 2  # "repeat the last fold on the other side" options, listed first
_R = 6  # rounding digits for dedupe keys


@dataclass
class Option:
    id: int
    description: str
    step: dict
    cost: float
    hash: str
    kind: str = "fold"  # fold / squash / petal / inside_reverse / unfold
    fam: str = ""  # menu section (onto / through / edges / align / flap / tuck / special / unfold)
    fam_rank: int = 0  # rank inside its section
    state: object = field(default=None, repr=False)  # resulting PaperState
    after: str = ""  # cached after_text
    origin: str = ""  # flap section: construction of the crease line (onto / through / edges / align)

    def line(self) -> str:
        return f"{self.id}. {self.description}"


# ---- geometry helpers ----------------------------------------------------------------------

def _key(p) -> tuple:
    return (round(float(p[0]), _R), round(float(p[1]), _R))


def _ang(u) -> float:
    """Undirected line angle in degrees, [0, 180)."""
    a = math.degrees(math.atan2(u[1], u[0])) % 180.0
    return 0.0 if a > 180 - 1e-6 else a


def _dir(deg: float) -> np.ndarray:
    r = math.radians(deg)
    return np.array([math.cos(r), math.sin(r)])


def _line_key(a, u) -> tuple:
    u = g.unit(np.asarray(u, float))
    th = _ang(u)
    d = _dir(th)
    n = np.array([-d[1], d[0]])
    off = float(np.dot(np.asarray(a, float), n))
    return (round(th, 6) % 180.0, round(off, 7))


class _FrozenLabels:
    """Read-only copy of a Labeler's labels (names are all registered before the menu is built)."""

    def __init__(self, to_label: dict):
        self.to_label = to_label

    def label(self, name: str) -> str:
        return self.to_label[name]


class _Feat:
    """Features of the current state in folded space."""

    def __init__(self, state, lab=None):
        self.state = state
        self.size = state.size
        self.lab = lab  # agents.command_layer.Labeler: neutral sticker labels instead of sheet names
        polys = [f.folded for f in state.faces]
        self.polys = polys
        pts: dict[tuple, np.ndarray] = {}
        for P in polys:
            for p in P:
                pts.setdefault(_key(p), np.array(p, float))
        self.fv = set(pts)  # keys of real face vertices (corners, crease ends / crossings)
        # sheet names of folded points (for descriptions)
        self.names: dict[tuple, list[str]] = {}
        for f in state.faces:
            for p in f.orig:
                q = g.apply(f.T, np.array([p]))[0]
                nm = point_name(p, self.size)
                lst = self.names.setdefault(_key(q), [])
                if nm not in lst:
                    lst.append(nm)
        from .landmarks import NAMED_POINTS

        for nm, (fx, fy) in NAMED_POINTS.items():
            try:
                q = state.to_folded((fx * self.size, fy * self.size))
            except ValueError:
                continue
            pts.setdefault(_key(q), q)
            lst = self.names.setdefault(_key(q), [])
            if nm not in lst:
                lst.append(nm)
        self.K = pts
        loops = outline(state)
        self.loops = loops
        self.V: dict[tuple, np.ndarray] = {}
        self.out_edges: list[tuple[np.ndarray, np.ndarray]] = []
        exact = list(pts.values())

        def snap(p):  # outline() works at reduced precision: snap to the exact face vertex
            p = np.array(p, float)
            best = min(exact, key=lambda q: float(np.linalg.norm(q - p)))
            return np.array(best, float) if np.linalg.norm(best - p) < 1e-5 else p

        for L in loops:
            for i in range(len(L)):
                p, q = snap(L[i]), snap(L[(i + 1) % len(L)])
                self.V.setdefault(_key(p), p)
                self.K.setdefault(_key(p), p)
                self.out_edges.append((p, q))
        segs: dict[tuple, tuple] = {}
        for P in polys:
            for i in range(len(P)):
                p, q = P[i], P[(i + 1) % len(P)]
                if np.linalg.norm(q - p) < 1e-9:
                    continue
                k = tuple(sorted((_key(p), _key(q))))
                segs.setdefault(k, (np.array(p, float), np.array(q, float)))
        self.segs = list(segs.values())
        lo, hi = state.bbox()
        self.lo, self.hi = lo, hi
        self.c = (lo + hi) / 2
        self.span = float(max(hi - lo))
        # directions of segments at each key point
        self.at: dict[tuple, list[np.ndarray]] = {}
        self.exact: dict[tuple, np.ndarray] = {}
        for p, q in self.segs + self.out_edges:
            for s, t in ((p, q), (q, p)):
                self.exact.setdefault(_key(s), np.array(s, float))
                lst = self.at.setdefault(_key(s), [])
                d = g.unit(t - s)
                if all(np.dot(d, e) < 1 - 1e-7 for e in lst):
                    lst.append(d)
        # derived points: angle bisector at a point hits the outline
        self.D: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}  # key -> (point, vertex)
        self.hits: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}  # (point, vertex) incl. existing points
        for kv, dirs in self.at.items():
            v = self.exact[kv]
            for i in range(len(dirs)):
                for j in range(i + 1, len(dirs)):
                    b = dirs[i] + dirs[j]
                    if np.linalg.norm(b) < 1e-9:
                        continue
                    b = g.unit(b)
                    if not self._inside(v + b * 1e-4 * self.span):
                        continue
                    x = self._ray_hit(v, b)
                    if x is None:
                        continue
                    self.hits.setdefault((_key(x), kv), (x, v))
                    if _key(x) not in self.K:
                        self.D.setdefault(_key(x), (x, v))

    def __getstate__(self):  # for the process pool: the labeler holds a generator
        d = dict(self.__dict__)
        if d.get("lab") is not None:
            d["lab"] = _FrozenLabels(dict(d["lab"].to_label))
        return d

    def layers_at(self, p) -> int:
        return sum(1 for P in self.polys if g.contains(P, p, -1e-9))

    def _inside(self, p) -> bool:
        return any(g.contains(P, p, 1e-9) for P in self.polys)

    def _ray_hit(self, o, d):
        best = None
        for p, q in self.out_edges:
            e = q - p
            den = g.cross(d, e)
            if abs(den) < 1e-12:
                continue
            t = g.cross(p - o, e) / den
            s = g.cross(p - o, d) / den
            if t > 1e-6 and -1e-9 <= s <= 1 + 1e-9 and (best is None or t < best[0]):
                best = (t, o + t * d)
        return None if best is None else best[1]

    # -- words --------------------------------------------------------------------------
    def where(self, p) -> str:
        rel = (np.asarray(p) - self.c) / max(self.span, 1e-9)
        v = "top" if rel[1] > 0.12 else "bottom" if rel[1] < -0.12 else ""
        h = "right" if rel[0] > 0.12 else "left" if rel[0] < -0.12 else ""
        if v and h:
            return f"{v}-{h}"
        return v or h or "middle"

    def _labs(self, nm) -> str:
        return "/".join(self.lab.label(n) for n in nm[:2])

    def pname(self, p, corner_word=True) -> str:
        k = _key(p)
        nm = self.names.get(k)
        if self.lab is not None and nm:
            kind = "corner" if (corner_word and k in self.V) else "point"
            return f"{kind} {self._labs(nm)} (now at {self.where(p)})"
        tag = f" ({'/'.join(nm[:2])})" if nm else ""
        if corner_word and k in self.V:
            return f"the {self.where(p)} corner{tag}"
        if k in self.D:
            return f"the {self.where(p)} edge point where the angle at {self.pname(self.D[k][1])} is halved"
        return f"the {self.where(p)} point{tag}" if nm else f"the point ({p[0]:.2f}, {p[1]:.2f})"

    def _short(self, p) -> str:
        nm = self.names.get(_key(p))
        if self.lab is not None and nm:
            return self._labs(nm)
        return self.pname(p, False).removeprefix("the ")

    def ename(self, p, q) -> str:
        m = (p + q) / 2
        return f"the {self.where(m)} edge ({self._short(p)} - {self._short(q)})"


def _dir_word(v) -> str:
    v = g.unit(np.asarray(v, float))
    a = math.degrees(math.atan2(v[1], v[0])) % 360
    words = ["right", "up-right", "up", "up-left", "left", "down-left", "down", "down-right"]
    return words[int(((a + 22.5) % 360) // 45)]


# ---- candidate lines ------------------------------------------------------------------------

@dataclass
class _Line:
    a: np.ndarray
    u: np.ndarray
    cost: float
    desc: str
    sides: tuple = (1, -1)  # which sides may move (+1 = left of u)
    fam: str = "through"
    on_crease: bool = False  # runs along an existing edge / crease of the folded paper
    sdesc: dict = field(default_factory=dict)  # side -> description that only fits that side
    gdesc: str | None = None  # description that fits both sides


def _lines(F: _Feat) -> list[_Line]:
    out: dict[tuple, _Line] = {}

    def add(a, u, cost, desc, sides=(1, -1), mover=None, fam="through"):
        if np.linalg.norm(np.asarray(u, float)) < 1e-9:
            return
        u = _dir(_ang(np.asarray(u, float)))  # canonical direction, so 'sides' mean the same
        a = np.asarray(a, float)
        if mover is not None:
            sides = (1 if g.side(a, u, mover) > 0 else -1,)
        k = _line_key(a, u)
        if not _cuts(F, a, u):
            return
        old = out.get(k)
        if old is None or cost < old.cost:
            sd = sides if old is None else tuple(sorted(set(sides) | set(old.sides)))
            new = _Line(a, u, cost, desc, sd, fam)
            new.sdesc = dict(old.sdesc) if old is not None else {}
            out[k] = old = new
        else:
            old.sides = tuple(sorted(set(sides) | set(old.sides)))
        if mover is not None and sides[0] not in old.sdesc:  # e.g. "fold A onto B" only fits A's side
            old.sdesc[sides[0]] = desc
        if mover is None and old.gdesc is None:
            old.gdesc = desc

    Vs = list(F.V.values())
    Ks = list(F.K.values())

    def w(p):  # how 'visible / natural' a reference point is (0 = outline corner)
        k = _key(p)
        return 0.0 if k in F.V else 0.3 if k in F.fv else 1.0

    # O2: outline corner onto a key point (corner's side moves)
    for p in Vs:
        for q in Ks:
            if np.linalg.norm(q - p) < 1e-6:
                continue
            u = np.array([-(q - p)[1], (q - p)[0]])
            a = (p + q) / 2
            add(a, u, 1.0 + w(q), f"fold {F.pname(p)} onto {F.pname(q)}", fam="onto",
                mover=p)
    # O1: through a corner and a key point
    for i, p in enumerate(Ks):
        for q in Ks[i + 1:]:
            if np.linalg.norm(q - p) < 1e-6:
                continue
            if _key(p) not in F.V and _key(q) not in F.V:
                continue
            add(p, q - p, 1.2 + w(p) + w(q), f"fold along the line through {F.pname(p)} and {F.pname(q)}")
    # axis lines through key points and derived points
    for p in Ks + [x for x, _ in F.D.values()]:
        base = (1.4 if _key(p) in F.K else 2.4) + w(p)
        add(p, (1, 0), base, f"fold along the horizontal line through {F.pname(p)}")
        add(p, (0, 1), base, f"fold along the vertical line through {F.pname(p)}")
        add(p, (1, 1), base + 0.8, f"fold along the 45-degree line (up-right) through {F.pname(p)}")
        add(p, (1, -1), base + 0.8, f"fold along the 45-degree line (down-right) through {F.pname(p)}")
    # O3: outline edge onto outline edge
    E = F.out_edges
    for i in range(len(E)):
        for j in range(i + 1, len(E)):
            (p1, q1), (p2, q2) = E[i], E[j]
            d1, d2 = g.unit(q1 - p1), g.unit(q2 - p2)
            if abs(g.cross(d1, d2)) < 1e-9:
                off = g.cross(d2, p1 - p2)
                if abs(off) < 1e-9:
                    continue
                add(p1 - np.array([-d2[1], d2[0]]) * off / 2, d2, 1.8,
                    f"fold {F.ename(p1, q1)} onto {F.ename(p2, q2)}", fam="edges")
                continue
            x = g.line_intersection(p1, q1, p2, q2)
            for b in (d1 + d2, d1 - d2):
                if np.linalg.norm(b) > 1e-9:
                    add(x, b, 1.8, f"fold {F.ename(p1, q1)} onto {F.ename(p2, q2)}", fam="edges")
    # angle bisectors at a point (segment onto segment meeting there)
    for kv, dirs in F.at.items():
        v = F.exact[kv]
        for i in range(len(dirs)):
            for j in range(i + 1, len(dirs)):
                b = dirs[i] + dirs[j]
                if np.linalg.norm(b) > 1e-9:
                    add(v, b, 2.0 + w(v), f"fold so the two edges/creases meeting at {F.pname(v)} lie on each other", fam="edges")
    # "make this edge level / upright": crease through a point X on segment e, bisecting e and an axis
    #    ... or parallel to an outline edge (the moved edge ends parallel to that edge)
    pts = Ks + [x for x, _ in F.D.values()]
    targets = [(0.0, "level (horizontal)", 0.0), (90.0, "upright (vertical)", 0.0)]
    seen_t = {0.0, 90.0}
    for p2, q2 in F.out_edges:
        t2 = round(_ang(q2 - p2), 6)
        if t2 not in seen_t:
            seen_t.add(t2)
            targets.append((t2, f"parallel to {F.ename(p2, q2)}", 0.3))
    for p, q in F.out_edges + F.segs:
        d = g.unit(q - p)
        th = _ang(d)
        L = float(np.linalg.norm(q - p))
        for x in pts:
            s = float(np.dot(x - p, d))
            if abs(g.cross(d, x - p)) > 1e-7 or s < -1e-7 or s > L + 1e-7:
                continue
            base = 2.2 + w(x)
            for ax, word, extra in targets:
                if min(abs(th - ax), 180 - abs(th - ax)) < 1e-4:
                    continue
                for c in ((th + ax) / 2, (th + ax) / 2 + 90):
                    add(x, _dir(c), base + extra, f"fold through {F.pname(x)} so that {F.ename(p, q)} becomes {word}",
                        fam="align" if w(x) < 1 else "align2")  # align2: pivot on a landmark / derived point
    for L in out.values():
        for p, q in F.segs:
            if abs(g.side(L.a, L.u, p)) < 1e-7 and abs(g.side(L.a, L.u, q)) < 1e-7:
                L.on_crease = True
                L.cost -= 0.2
                tag = " (along an existing crease)"
                L.desc += tag
                L.gdesc = L.gdesc and L.gdesc + tag
                L.sdesc = {k: v + tag for k, v in L.sdesc.items()}
                break
    return sorted(out.values(), key=lambda L: L.cost)


def _cuts(F: _Feat, a, u) -> bool:
    """Line crosses the interior of some face."""
    for P in F.polys:
        s = [g.side(a, u, p) for p in P]
        if max(s) > 1e-7 and min(s) < -1e-7:
            return True
    return False


# ---- per-line variants ----------------------------------------------------------------------

class _CutCtx:
    """Per-cut-state caches for the menu's per-line work (same answers as the engine helpers).

    The paper cut along one candidate line is shared by both sides and every layer variant, and they
    all ask the same questions: which face centres lie inside which faces (g.contains), which faces
    share a paper edge (g.shared_edge), is that edge on the fold line. Containment is computed as a
    matrix with the same float operations as g.contains; shared edges are only looked up for faces
    whose sheet bounding boxes touch (a shared edge needs them within its 1e-6 tolerance)."""

    _BOX_PAD = 1e-5

    def __init__(self, st, a, u):
        self.st, self.a, self.u = st, a, u
        faces = st.faces
        self.n = len(faces)
        self.cent = [f.folded.mean(axis=0) for f in faces]
        self._C = np.array(self.cent, float).reshape(-1, 2)
        self._poly = []
        self.ok = True  # False: a degenerate face (zero-length edge); callers use the plain helpers
        for f in faces:
            P = g.ccw(f.folded).tolist()
            m = len(P)
            edges = []
            for i in range(m):
                ax, ay = P[i]
                bx, by = P[(i + 1) % m]
                ux, uy = bx - ax, by - ay
                L = math.sqrt(ux * ux + uy * uy)
                if L < g.EPS:
                    self.ok = False
                edges.append((ax, ay, ux / L if L else 0.0, uy / L if L else 0.0))
            self._poly.append(edges)
        self._mats: dict = {}
        self._nb = None
        self._pred: dict = {}

    def inside(self, tol: float) -> np.ndarray:
        """M[i, j] = g.contains(face j, centre of face i, tol)."""
        M = self._mats.get(tol)
        if M is None:
            xs, ys = self._C[:, 0], self._C[:, 1]
            M = np.ones((self.n, self.n), bool)
            for j, edges in enumerate(self._poly):
                col = np.ones(self.n, bool)
                for ax, ay, ux, uy in edges:
                    col &= ~(ux * (ys - ay) - uy * (xs - ax) < -tol)
                M[:, j] = col
            self._mats[tol] = M
        return M

    def nb(self, k: int) -> list[int]:
        """Faces whose sheet bbox touches face k's (superset of the faces sharing an edge with it)."""
        if self._nb is None:
            lo = np.array([f.orig.min(axis=0) for f in self.st.faces]).reshape(-1, 2)
            hi = np.array([f.orig.max(axis=0) for f in self.st.faces]).reshape(-1, 2)
            pad = self._BOX_PAD
            near = ((lo[:, None, :] <= hi[None, :, :] + pad) & (lo[None, :, :] <= hi[:, None, :] + pad)).all(axis=2)
            np.fill_diagonal(near, False)
            self._nb = [np.nonzero(r)[0].tolist() for r in near]
        return self._nb[k]

    def joined(self, k: int, m: int) -> bool:
        """_attached's test: faces k and m share a paper edge that is NOT on the fold line."""
        key = (k, m)
        r = self._pred.get(key)
        if r is None:
            fs = self.st.faces
            e = g.shared_edge(fs[k].orig, fs[m].orig)
            r = e is not None and not _on_line(self.a, self.u, g.apply(fs[m].T, np.array(e)))
            self._pred[key] = r
        return r

    def attached(self, cand, chosen) -> set[int]:
        """Same set as engine._attached(st, cand, chosen, a, u) (its least fixed point)."""
        if not self.ok:
            return _attached(self.st, cand, chosen, self.a, self.u)
        cs = set(cand)
        out = set(chosen)
        todo = list(out)
        while todo:
            m = todo.pop()
            for k in self.nb(m):
                if k in cs and k not in out and self.joined(k, m):
                    out.add(k)
                    todo.append(k)
        return out

    def layer_count(self, faces) -> int:
        """Same as engine._layer_count(st, faces)."""
        if not self.ok:
            return _layer_count(self.st, faces)
        idx = list(faces)
        if not idx:
            return 0
        return int(self.inside(-1e-7)[np.ix_(idx, idx)].sum(axis=1).max())


def _side_info(state, a, u, side, st=None, ctx=None):
    """Moving-side faces after cutting: depth (max layers) and flap components (top-face centroid).
    st: the state already cut along the line (shared by both sides); ctx: its _CutCtx."""
    if st is None:
        st = _cut(_begin(state), a, u)
    u = g.unit(np.asarray(u, float))
    cand = [k for k, f in enumerate(st.faces) if _side_of(f, a, u) == side]
    if not cand:
        return None
    fast = ctx is not None and ctx.ok
    if fast:
        cents = [ctx.cent[k] for k in cand]
        depth = max(1, int(ctx.inside(1e-9)[np.ix_(cand, cand)].sum(axis=1).max()))
    else:
        polys = [st.faces[k].folded for k in cand]
        cents = [P.mean(axis=0) for P in polys]
        depth = 1
        for c in cents:
            depth = max(depth, sum(1 for P in polys if g.contains(P, c, 1e-9)))
    # flap components (shared paper edges)
    comp = {}
    cid = 0
    for k in cand:
        if k in comp:
            continue
        stack = [k]
        comp[k] = cid
        while stack:
            m = stack.pop()
            if fast:  # only faces whose bbox touches m's can share an edge; same visiting order
                nbm = set(ctx.nb(m))
                near = [j for j in cand if j in nbm]
            else:
                near = cand
            for j in near:
                if j not in comp and g.shared_edge(st.faces[m].orig, st.faces[j].orig) is not None:
                    comp[j] = cid
                    stack.append(j)
        cid += 1
    flaps = []
    for c in range(cid):
        ks = [k for k in cand if comp[k] == c]
        top = max(ks)  # faces are in draw order (last = top)
        flaps.append(st.faces[top].folded.mean(axis=0))
    # a representative move point: centroid of the area farthest from the line
    far = max(cents, key=lambda c: abs(g.side(a, u, c)))
    return depth, flaps, far, (st, cand)


def _layer_word(layers, tuck=None, op="valley", thick: int = 1) -> tuple[str, float]:
    words, cost = [], 0.0
    if op == "mountain":
        if tuck != "inside":  # tuck: the slide-in wording says "from behind"
            words.append("fold it behind")
        cost += 0.6
    if layers == "all":
        words.append(f"all layers, {thick} layers move together" if thick > 1 else "all layers, just one layer here")
    else:
        n = layers.get("top")
        words.append("front layer only" if n == 1 else f"front {n} layers only")
        cost += 0.2 * n
        if layers.get("flap"):
            words[-1] += ", only this one flap"
            cost += 0.4
    if tuck == "inside":
        if layers == "all":  # the slide-in wording says it all; keep only a layer count
            words = [f"{thick} layers move together"] if thick > 1 else []
        words.append("between the layers")
        cost += 1.2
    return ", ".join(words), cost


def _line_cands(state, F: _Feat, L: _Line):
    """Yield (cost, kind, step, desc) for one crease line."""
    if True:
        a, u = L.a, L.u
        p1, p2 = a - u, a + u
        line = {"through": [[float(p1[0]), float(p1[1])], [float(p2[0]), float(p2[1])]]}
        cut = []  # the paper cut along the line, built once for both sides
        for side in L.sides:
            if not (L.sdesc.get(side) or L.gdesc):
                continue  # only constructions for the other side (e.g. "fold A onto B": A's side moves)
            try:
                if not cut:
                    cut.append(_cut(_begin(state), a, u))
                    cut.append(_CutCtx(cut[0], a, g.unit(u)))
                info = _side_info(state, a, u, side, cut[0], cut[1])
            except ValueError:
                continue
            if info is None:
                continue
            depth, flaps, far, (st, cand) = info
            seen_sel = set()
            mv = [float(far[0]), float(far[1])]
            n_dir = np.array([-u[1], u[0]]) * (-side)  # moving part travels across the line
            motion = f"the {F.where(far)} part goes {_dir_word(n_dir)}"
            variants = [("valley", "all", None, mv)]
            for n in range(1, min(depth, 4)):
                variants.append(("valley", {"top": n}, None, mv))
            if L.on_crease:  # tucks: into a pocket along an existing fold
                variants.append(("valley", "all", "inside", mv))
                for n in range(1, min(depth, 3)):
                    variants.append(("mountain", {"top": n}, "inside", mv))
            if len(flaps) > 1:
                for fp in flaps:
                    fm = [float(fp[0]), float(fp[1])]
                    for n in range(1, min(depth, 3) + 1):
                        variants.append(("valley", {"top": n, "flap": True}, None, fm))
            for op, layers, tuck, m in variants:
                try:
                    picked = _select_layers(st, cand, layers, np.array(m), line=(a, g.unit(u)))
                    chosen = frozenset(cut[1].attached(cand, picked))
                except Exception:  # noqa: BLE001
                    continue
                sk = (op, tuck, chosen)
                if not chosen or sk in seen_sel:
                    continue
                seen_sel.add(sk)
                step = {"op": op, "line": line, "move": m}
                if layers != "all":
                    step["layers"] = layers
                if tuck:
                    step["tuck"] = tuck
                thick = cut[1].layer_count(chosen)  # thick stacks are rarely folded all together
                lw, lc = _layer_word(layers, tuck, op, thick)
                mot = motion
                if isinstance(layers, dict) and layers.get("flap"):
                    mot = f"the flap at the {F.where(m)} goes {_dir_word(n_dir)}"
                if tuck == "inside":  # reads like what the hands do: open the pocket, slide the flap in
                    back = " from behind" if op == "mountain" else ""
                    mot = f"open the pocket{back} and slide the {F.where(far)} part {_dir_word(n_dir)} inside"
                ld = L.sdesc.get(side) or L.gdesc
                desc = f"{ld[0].upper()}{ld[1:]}: {mot} ({lw})"
                fam = ("tuck" if (tuck or op == "mountain") else
                       f"flap/{L.fam}" if isinstance(layers, dict) and layers.get("flap") else
                       L.fam.rstrip("2"))
                yield L.cost + lc + 0.15 * (thick - 1), "fold", step, desc, fam
            # inside reverse: tip of a multi-layer flap
            if depth >= 2:
                for fp in (flaps if len(flaps) > 1 else [far]):
                    step = {"op": "inside_reverse", "line": line, "move": [float(fp[0]), float(fp[1])]}
                    yield (L.cost + 2.5, "inside_reverse", step,
                           f"Inside-reverse fold the {F.where(fp)} flap tip along this line: {L.desc}", "tuck")


def _macro_cands(F: _Feat):
    # squash: hinge + spine = two segments meeting at a point O
    for kv, dirs in F.at.items():
        O = F.exact[kv]
        if len(dirs) < 2:
            continue
        for dh in dirs:
            for ds in dirs:
                if dh is ds:
                    continue
                ang = math.degrees(math.acos(max(-1, min(1, float(np.dot(dh, ds))))))
                if ang < 20 or ang > 135:
                    continue
                r = 0.15 * F.span
                mv = O + r * g.unit(dh + ds) * 0.8
                if not F._inside(mv) or F.layers_at(mv) < 2:  # a squash opens a multi-layer flap
                    continue
                h2, s2 = O + dh * r, O + ds * r
                step = {"op": "squash", "move": [float(mv[0]), float(mv[1])],
                        "hinge": [[float(O[0]), float(O[1])], [float(h2[0]), float(h2[1])]],
                        "spine": [[float(O[0]), float(O[1])], [float(s2[0]), float(s2[1])]]}
                yield (3.0, "squash", step,
                       f"Squash fold: lift the flap at {F.pname(O)} on its {_dir_word(dh)} edge, open it and "
                       f"flatten it so its {_dir_word(ds)} folded edge lands on that hinge")
    # petal: corner V; line perpendicular to a segment at V through a derived point made from V
    for kd, (x, v) in F.hits.items():
        if _key(v) not in F.V or F.layers_at(v + 0.3 * (x - v)) < 2:  # petal needs 2+ layers
            continue
        for d in F.at.get(_key(v), []):
            u = np.array([-d[1], d[0]])
            p1, p2 = x - u, x + u
            step = {"op": "petal", "move": [float(v[0]), float(v[1])],
                    "line": {"through": [[float(p1[0]), float(p1[1])], [float(p2[0]), float(p2[1])]]}}
            yield (3.5, "petal", step,
                   f"Petal fold: lift {F.pname(v)} (top layer) up along the crease through "
                   f"{F.pname(x)}; the sides fold in")


def _mirror_axes(state) -> list[tuple[np.ndarray, np.ndarray]]:
    """Mirror lines (a, u) of the folded silhouette (the outline maps onto itself)."""
    loops = outline(state)
    pts = np.array([p for L in loops for p in L], float)
    if len(pts) < 3:
        return []
    lo, hi = pts.min(0), pts.max(0)
    c = (lo + hi) / 2
    tol = 1e-5 * max(float(max(hi - lo)), 1e-9)
    # the fold structure must be symmetric too, not just the silhouette: every face corner and face-edge
    # midpoint must map onto one (a diamond/square has 4 silhouette mirror lines, but after a flap fold
    # only one of them is a mirror line of the paper -> 'repeat' used to mirror the horn across the wrong one)
    inner = []
    for f in state.faces:
        P = np.asarray(f.folded, float)
        inner.append(P)
        inner.append((P + np.roll(P, -1, axis=0)) / 2)
    inner = np.vstack(inner) if inner else pts
    out = []
    for k in range(12):  # every 15 degrees
        u = _dir(15.0 * k)
        R = g.reflection(c, u)
        if not all(float(np.min(np.linalg.norm(pts - x, axis=1))) < tol for x in g.apply(R, pts)):
            continue
        if all(float(np.min(np.linalg.norm(inner - x, axis=1))) < tol for x in g.apply(R, inner)):
            out.append((c, u))
    return out


def _repeat_cands(src, sim, F: _Feat):
    """'Repeat the last fold on the other side': the last fold mirrored across a mirror line of the
    paper as it was before that fold (kabuto: second horn, crane: other wing ...)."""
    traj = getattr(src, "trajectory", None)
    hist = getattr(sim, "_fold_history", None)
    if not traj or not hist or hist[-1][1]:  # no fold yet, or the view changed since the last fold
        return
    last = traj[-1]
    if last.get("op") not in ("valley", "mountain", "inside_reverse"):
        return
    if str(last.get("description", "")).startswith("Repeat the last fold"):
        return  # no chains of mirror images (Sonnet test: it kept picking 'repeat' through other mirror lines)
    from .engine import fold_geometry

    prev = hist[-1][0]
    try:
        a, u, mv = fold_geometry(prev, last)
    except Exception:  # noqa: BLE001
        return
    a, u, mv = np.asarray(a, float), g.unit(np.asarray(u, float)), np.asarray(mv, float)
    for c, ax in _mirror_axes(prev):
        R = g.reflection(c, ax)
        p1, p2, m = g.apply(R, np.array([a - u, a + u, mv]))
        if np.linalg.norm(p1 - (a - u)) < 1e-9 and np.linalg.norm(p2 - (a + u)) < 1e-9 \
                and np.linalg.norm(m - mv) < 1e-9:
            continue  # the fold is its own mirror image
        step = {"op": last["op"], "line": {"through": [[float(p1[0]), float(p1[1])], [float(p2[0]), float(p2[1])]]},
                "move": [float(m[0]), float(m[1])]}
        for key in ("layers", "tuck"):
            if key in last:
                step[key] = copy.deepcopy(last[key])
        what = (last.get("description") or "").split(":")[0].rstrip(".")
        what = f" of: {what[0].lower()}{what[1:]}" if what else ""
        yield (0.5, "fold", step, f"Repeat the last fold on the other side (mirror image{what}): "
                                  f"the {F.where(m)} part moves")


def _shape_penalty(F: _Feat, new_state) -> float:
    """Results whose silhouette gets more corners look less 'sensible' (rarely what a diagram does)."""
    n_old = sum(len(L) for L in F.loops)
    n_new = sum(len(L) for L in outline(new_state))
    return 0.15 * max(0, n_new - n_old)


def _light_try(sim: Simulator, step: dict, k: int, open_issues: list):
    """env.try_step without deep-copying the fold history (the new sim is only used to read its
    state). -> (new state or None, new open issues)."""
    new = Simulator.__new__(Simulator)
    new.state = sim.state
    new._fold_history = list(sim._fold_history)
    try:
        new.apply(step)
    except Exception:  # noqa: BLE001
        return None, open_issues
    return new.state, open_issues


def _legal(sim, new_state, step, k, open_issues, strict=False) -> bool:
    """Same verdict as check_step (no error except state issues already open before), but the
    checks run cheapest-first and stop at the first real error (most candidates are illegal)."""
    from . import validate as V

    if step.get("op") in ("flip", "rotate"):
        return True
    before = sim.state

    def bad(issues) -> bool:
        for iss in issues:
            if iss.severity != "error":
                continue
            if iss.kind in V.STATE_KINDS and any(V._same_spot(o, iss) for o in open_issues):
                continue
            return True
        return False

    edges: list = []

    def layers():
        edges[:] = V._face_edges(new_state)
        return V.check_layers(new_state, k, edges)

    stages = (lambda: V.check_selection(new_state, step, k, strict),
              lambda: V.check_motion(before, new_state, step, k),
              lambda: V.check_sheet(new_state, k),
              layers,
              lambda: V.check_crease_pattern(V.folded_creases(new_state, edges), new_state.size, k),
              lambda: V.check_invariants(before, new_state, step, k))
    return not any(bad(stage()) for stage in stages)


# ---- parallel evaluation of candidate rows (big states only) ------------------------------
# Each row = engine + physics checker + canonical hash, ~50-150 ms on a 30-60 face state. Rows are
# independent, so a process pool evaluates a batch speculatively; the caller then walks the results
# in the same order with the same stop rules, so the menu is identical to the sequential one.

PAR_MIN_FACES = 12  # below this a menu builds in ~1 s: not worth the inter-process traffic
_POOL = None
_POOL_N = 0
_WORKER_SIM: dict = {}


def _workers() -> int:
    import os
    import sys

    if sys.platform == "emscripten":  # Pyodide (browser): no processes
        return 0
    v = os.environ.get("ORIGAMI_MENU_WORKERS")
    if v is not None:
        try:
            return max(0, int(v))
        except ValueError:
            return 0
    return max(0, min(8, (os.cpu_count() or 1) - 1))


def _get_pool():
    global _POOL, _POOL_N
    n = _workers()
    if n < 2:
        return None
    if _POOL is None or _POOL_N != n:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        try:
            _POOL = ProcessPoolExecutor(max_workers=n, mp_context=mp.get_context("spawn"))
            _POOL_N = n
        except Exception:  # noqa: BLE001
            _POOL = None
    return _POOL


def _drop_pool():
    global _POOL
    if _POOL is not None:
        try:
            _POOL.shutdown(wait=False, cancel_futures=True)
        except Exception:  # noqa: BLE001
            pass
    _POOL = None


def _eval_row(sim, step, k, open_issues):
    """(new state or None, legal, hash or None) for one candidate step."""
    new_state, _ = _light_try(sim, step, k, open_issues)
    if new_state is None:
        return None, False, None
    if not _legal(sim, new_state, step, k, open_issues):
        return None, False, None
    return new_state, True, state_hash(new_state)


def _worker_eval(token, blob, step, k):
    """Pool worker: unpickle the (sim, open issues) once per menu build, then evaluate one row."""
    import pickle

    if _WORKER_SIM.get("token") != token:
        from . import engine

        sim, open_issues = pickle.loads(blob)
        top = max((getattr(f, "uid", -1) for st in [sim.state] + [b for b, _ in sim._fold_history]
                   for f in st.faces), default=-1)
        engine._uids = __import__("itertools").count(top + 1 + 10**9)  # fresh face ids stay unique
        _WORKER_SIM.clear()
        _WORKER_SIM.update(token=token, sim=sim, oi=open_issues)
    return _eval_row(_WORKER_SIM["sim"], step, k, _WORKER_SIM["oi"])


def _worker_lines(token, blob, lines):
    """Pool worker: candidate rows of some crease lines (same as _line_cands in the parent)."""
    import pickle

    if _WORKER_SIM.get("ftoken") != token:
        state, F = pickle.loads(blob)
        _WORKER_SIM.update(ftoken=token, fstate=state, F=F)
    return [list(_line_cands(_WORKER_SIM["fstate"], _WORKER_SIM["F"], L)) for L in lines]


class _Evaluator:
    """Evaluate rows in order; with a pool, in speculative parallel batches."""

    def __init__(self, sim, k, open_issues):
        self.sim, self.k, self.oi = sim, k, open_issues
        self.pool = _get_pool() if len(sim.state.faces) >= PAR_MIN_FACES else None
        self.blob = self.token = None
        if self.pool is not None:
            import pickle
            import uuid

            try:
                self.blob = pickle.dumps((sim, open_issues))
                self.token = uuid.uuid4().hex
            except Exception:  # noqa: BLE001
                self.pool = None
        self.n = 0  # rows evaluated (incl. speculative ones)

    def run(self, rows):
        """Yield (row, (new_state, legal, hash)) in row order. Stop pulling when the consumer stops."""
        if self.pool is None:
            for row in rows:
                self.n += 1
                yield row, _eval_row(self.sim, row[2], self.k, self.oi)
            return
        it = iter(rows)
        batch = 2 * _POOL_N
        while True:
            chunk = []
            for row in it:
                chunk.append(row)
                if len(chunk) >= batch:
                    break
            if not chunk:
                return
            self.n += len(chunk)
            try:
                futs = [self.pool.submit(_worker_eval, self.token, self.blob, r[2], self.k) for r in chunk]
                res = [f.result() for f in futs]
            except Exception:  # noqa: BLE001  (broken pool, e.g. a script without a __main__ guard)
                _drop_pool()
                self.pool = None
                res = [_eval_row(self.sim, r[2], self.k, self.oi) for r in chunk]
            for row, r in zip(chunk, res):
                yield row, r


class _LazyRows:
    """Candidate rows per section, cheapest first, building a line's candidates only when needed.

    Every row of a line costs >= the line's cost (layer / thickness terms are >= 0) and lines come
    sorted by cost, so a row may be handed out once no unbuilt line is cheaper than it. Same order
    as building everything and sorting each section by cost (ties: build order), but the costly
    per-line work (cutting the paper, choosing layers) is skipped for lines nobody reaches."""

    def __init__(self, state, F, lines):
        self.state, self.F, self.lines, self.i, self.seq = state, F, lines, 0, 0
        self.heaps: dict[str, list] = {f: [] for f in FAMILIES}

    def prebuild(self, pool) -> bool:
        """Build every line's rows in the process pool (big states reach nearly all lines anyway).
        Same rows, same order as building them one by one."""
        import pickle
        import uuid

        try:
            blob = pickle.dumps((self.state, self.F))
            token = uuid.uuid4().hex
            n = max(1, len(self.lines) // (4 * max(1, _POOL_N)) or 1)
            chunks = [self.lines[i:i + n] for i in range(self.i, len(self.lines), n)]
            futs = [pool.submit(_worker_lines, token, blob, ch) for ch in chunks]
            res = [r for f in futs for r in f.result()]
        except Exception:  # noqa: BLE001
            return False
        for rows in res:
            for row in rows:
                self._push(*row)
        self.i = len(self.lines)
        return True

    def _build_next(self):
        L = self.lines[self.i]
        self.i += 1
        for row in _line_cands(self.state, self.F, L):
            self._push(*row)

    def _push(self, c, kind, st, d, fam):
        sec, _, origin = fam.partition("/")  # "flap/onto": flap section, line made by an 'onto' construction
        heapq.heappush(self.heaps.setdefault(sec, []), (c, self.seq, kind, st, d, origin))
        self.seq += 1

    def rows(self, fam):
        h = self.heaps.setdefault(fam, [])
        from_lines = fam in FAMILIES  # "repeat" / "special" rows are all pushed up front
        while True:
            while from_lines and self.i < len(self.lines) and (not h or h[0][0] > self.lines[self.i].cost):
                self._build_next()
            if not h:
                return
            c, _, kind, st, d, origin = heapq.heappop(h)
            yield c, kind, st, d, origin


class Menu(list):
    """List of Options + which sections were cut (more options exist: ask with `more`)."""
    truncated: dict
    skipped: list  # sections not evaluated at all (deadline_s / sections)


def candidates(src, cap: bool | None = True, open_issues=None, k: int | None = None,
               with_unfold: bool = True, more: dict | None = None, pool: float = 2.0,
               scale: float = 1.0, labeler=None, deadline_s: float | None = None, on_section=None,
               sections=None, exclude_hashes=()) -> Menu:
    """Legal, deduped, ranked next folds for `src` (FoldEnv, Simulator or PaperState).

    The menu has sections (FAMILIES + "special" + unfold). Each section lists its `quota` simplest
    distinct results (cheapest first; candidates evaluated lazily, at most pool * quota results per
    section). `more` = {section: pages} shows `pages` times the quota for those sections (the
    agent's `more` tool). cap=None: no quotas (everything; slow).

    Time-limited / partial menus (browser):
    - deadline_s: seconds of wall time. Sections run in order; once the time is up (checked between
      sections) the remaining ones are left empty with truncated[section] = True and
      `menu.skipped` lists them. The sections that did run are exactly those of the full menu.
    - on_section(section, options): called after each evaluated section (options get their final
      ids only in the returned menu).
    - sections: evaluate only these sections (others are skipped as above).
    - exclude_hashes: result hashes already listed elsewhere (treated like earlier sections'
      results), so a section evaluated alone dedupes like it would in the full menu.
    The unfold option is added unless `sections` leaves it out (it is one cheap check)."""
    import time

    t_end = None if deadline_s is None else time.monotonic() + float(deadline_s)
    if hasattr(src, "sim"):  # FoldEnv
        sim, open_issues, k = src.sim, src.open_issues, len(src.trajectory) + 1
    elif isinstance(src, Simulator):
        sim = src
    else:
        sim = Simulator(src.size)
        sim.state = src.copy()
    open_issues = open_issues or []
    k = k or 1
    more = more or {}
    F = _Feat(sim.state, labeler)
    if labeler is not None:  # deterministic label order for every sheet point seen in this state
        labeler.register([nm for names in F.names.values() for nm in names])
    here = state_hash(sim.state)
    rows_src = _LazyRows(sim.state, F, _lines(F))
    rows_src.heaps["special"] = [(c, i, kind, st, d, "") for i, (c, kind, st, d) in enumerate(_macro_cands(F))]
    heapq.heapify(rows_src.heaps["special"])
    rows_src.heaps["repeat"] = [(c, i, kind, st, d, "") for i, (c, kind, st, d) in enumerate(_repeat_cands(src, sim, F))]
    heapq.heapify(rows_src.heaps["repeat"])
    quotas = dict(repeat=REPEAT_CAP, **FAMILIES, special=MACRO_CAP)
    shown = {here} | set(exclude_hashes)  # results already listed (and the no-op)
    opts = Menu()
    opts.truncated = {}
    opts.skipped = []
    evals = 0
    ev = _Evaluator(sim, k, open_issues)
    if ev.pool is not None:
        rows_src.prebuild(ev.pool)
    for fam, quota in quotas.items():
        if (sections is not None and fam not in sections) or (t_end is not None and time.monotonic() >= t_end):
            opts.truncated[fam] = True
            opts.skipped.append(fam)
            continue
        quota = max(1, int(round(quota * scale * max(1, int(more.get(fam, 1))))))
        got: list[Option] = []
        seen: set = set()
        stopped = False
        best: list[float] = []  # sorted final costs of the results so far
        mix = fam in MIXED  # page = round robin over line constructions, not the cheapest `quota`
        fpool = MIX_POOL if mix else pool

        def full(cost) -> bool:
            if cap is None:
                return False
            if len(got) >= fpool * quota:
                return True
            # early stop: final cost = row cost + a penalty >= 0 and rows come cheapest first, so
            # once `quota` results are cheaper than this row, no later row can make the page
            return not mix and len(best) >= quota and cost > best[quota - 1] + 1e-12

        def gate(src):  # stop rules that need no evaluation: checked before a row is evaluated
            for cost, kind, step, desc, origin in src:
                if full(cost):
                    return True
                yield cost, kind, dict(step, description=desc), desc, origin
            return False

        def stream():
            nonlocal stopped
            stopped = yield from gate(rows_src.rows(fam))

        for (cost, kind, step, desc, origin), (new_state, ok, h) in ev.run(stream()):
            # (re-check the stop rules: in a parallel batch, rows past the stop point were evaluated)
            if full(cost):
                stopped = True
                break
            evals += 1
            # legality before the hash (most candidates are illegal; the canonical hash is costly).
            # Same outcome as hash-then-check: an illegal result is never listed or marked seen.
            if not ok:
                continue  # another route to the same result may still be legal
            if h in shown or h in seen:
                continue
            seen.add(h)
            got.append(Option(0, desc, step, cost + _shape_penalty(F, new_state), h, kind, state=new_state,
                              origin=origin))
            best.append(got[-1].cost)
            best.sort()
        got.sort(key=lambda o: o.cost)
        if mix:  # cheapest of each construction first, then the 2nd cheapest of each, ...
            nth: dict = {}
            for o in got:
                nth[o.origin] = nth.get(o.origin, 0) + 1
                o._nth = nth[o.origin]
            got.sort(key=lambda o: (o._nth, o.origin == "align2", o.cost))  # pivots on landmarks last
        for r, o in enumerate(got, 1):
            o.fam, o.fam_rank = fam, r
        keep = got[:quota] if cap is not None else got
        opts.truncated[fam] = len(got) > len(keep) or stopped
        shown |= {o.hash for o in keep}
        opts += keep
        if on_section is not None:
            on_section(fam, keep)
    if with_unfold and (sections is None or "unfold" in sections) and getattr(sim, "_fold_history", None):
        step = {"op": "unfold", "description": "Unfold the last fold (its crease stays)"}
        new_state, _ = _light_try(sim, step, k, open_issues)
        if new_state is not None:
            o = Option(0, step["description"], step, 9.0, state_hash(new_state), "unfold", state=new_state)
            o.fam, o.fam_rank = "unfold", 1
            opts.append(o)
    for i, o in enumerate(opts, 1):
        o.id = i
    candidates.last_evals = evals
    return opts


def after_text(o: Option, colored_side: str = "up") -> str:
    """Short factual summary of the state after the option (no target info): outline, colours, layers, flaps."""
    if getattr(o, "after", None):
        return o.after
    from .observe import _Geo, _angles, side_colours

    st = o.state
    if st is None:
        return ""
    size = st.size
    loops = outline(st)
    geo = _Geo(st, colored_side)
    lo, hi = st.bbox()
    w, h = (hi[0] - lo[0]) / size, (hi[1] - lo[1]) / size
    parts = []
    tips = []
    if loops:
        lp = loops[0]
        ang = _angles(lp)
        n = len(lp)
        right = sum(abs(a - 90) < 8 for a in ang)
        shape = ("triangle" if n == 3 else "square" if n == 4 and right == 4 and abs(w - h) < 0.06 * max(w, h)
                 else "rectangle" if n == 4 and right == 4 else "quadrilateral" if n == 4
                 else "polygon")
        parts.append(f"{shape} with {n} corners, {w:.2f}x{h:.2f}")
        tips = [v for v, a in zip(lp, ang) if a < 75]
    cols = side_colours(geo, 24)
    parts.append(f"front {round(100 * cols['front'])}%/back {round(100 * cols['back'])}% coloured")
    depth = 1
    nf = len(st.faces)
    if nf:
        pts = np.array([f.folded.mean(axis=0) for f in st.faces])
        cov = geo.cover(pts)
        depth = max(1, int(cov.sum(axis=0).max())) if cov.size else 1
    parts.append(f"max {depth} layer{'s' if depth > 1 else ''}")
    if tips and nf > 1:
        c = np.mean(loops[0], axis=0)
        fr = bh = 0
        for v in tips:
            P = v + 0.3 * (c - v)
            idx = np.nonzero(geo.cover(P)[:, 0])[0]
            if not len(idx):
                continue
            r = float(idx.mean()) / max(nf - 1, 1)
            fr += r > 0.66
            bh += r < 0.34
        bits = [f"{k} {w_}" for k, w_ in ((fr, "in front"), (bh, "behind")) if k]
        if bits:
            parts.append(f"{len(tips)} pointed flaps: " + ", ".join(bits))
    o.after = "; ".join(parts)
    return o.after


SECTION_TITLES = {"repeat": "Repeat the last fold on the other side", "onto": "Fold a corner onto a point", "through": "Fold along a line through points",
                  "edges": "Fold an edge onto an edge", "align": "Fold so an edge becomes level / upright / parallel",
                  "flap": "Fold a single flap", "tuck": "Tuck into a pocket / fold behind / inside reverse",
                  "special": "Special techniques (squash, petal)", "unfold": "Unfold"}


def menu_text(opts: list[Option], with_after: bool = False, colored_side: str = "up") -> str:
    out, cur = [], None
    trunc = getattr(opts, "truncated", {})
    for o in opts:
        if o.fam != cur:
            if cur is not None and trunc.get(cur):
                out.append(f"   (more in this section: more(section=\"{cur}\"))")
            cur = o.fam
            if cur:
                out.append(f"[{cur}] {SECTION_TITLES.get(cur, cur)}:")
        out.append(f"{o.id}. {o.description}")
        if with_after:
            out.append(f"     after: {after_text(o, colored_side)}")
    if cur is not None and trunc.get(cur):
        out.append(f"   (more in this section: more(section=\"{cur}\"))")
    return "\n".join(out)


THUMB_MAX = 10  # options that get a picture (the rest are text only)


def thumbnails_png(opts: list[Option], colored_side: str = "up", cols: int = 4, cell: int = 300,
                   max_n: int = THUMB_MAX) -> bytes:
    """One image: big numbered front-view drawings of the first `max_n` options' results."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    opts = list(opts)[:max_n]
    n = max(1, len(opts))
    cols = min(cols, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * cell / 100, rows * cell / 100), dpi=100, squeeze=False)
    up = colored_side != "down"
    for ax in axes.ravel():
        ax.axis("off")
    for ax, o in zip(axes.ravel(), opts):
        st = o.state
        for f in st.faces:
            col = "#f28c28" if f.front_up == up else "#fff4dc"
            ax.add_patch(Polygon(f.folded, closed=True, fc=col, ec="#333", lw=1.0))
        lo, hi = st.bbox()
        pad = 0.08 * float(max(hi - lo))
        ax.set_xlim(lo[0] - pad, hi[0] + pad)
        ax.set_ylim(lo[1] - pad, hi[1] + pad)
        ax.set_aspect("equal")
        ax.set_title(f"Option {o.id}", fontsize=18, fontweight="bold")
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#999")
    fig.tight_layout(pad=0.6)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


__all__ = ["Option", "Menu", "candidates", "menu_text", "thumbnails_png", "after_text", "THUMB_MAX", "DEFAULT_CAP", "FAMILIES", "MACRO_CAP"]
