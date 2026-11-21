"""Paper state: faces (pieces of the original sheet), their local layer order, and creases.

Layer order: the source of truth is a *pairwise* relation
`above` = set of (i, j) meaning "face i lies above face j (closer to the viewer)", stored
only for faces that overlap (positive area) in folded space. `faces` is additionally kept
in a draw order (index 0 = bottom) that is a linear extension of `above` when one exists,
so old code that reads `faces` bottom-to-top keeps working.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from . import geometry as g

OVERLAP_TOL = 1e-6


@dataclass
class Face:
    """One flat piece of paper.

    orig: polygon in crease-pattern (unfolded sheet) coordinates.
    T:    3x3 affine isometry mapping orig -> folded-space coordinates.
    """

    orig: np.ndarray
    T: np.ndarray = field(default_factory=lambda: np.eye(3))

    @property
    def folded(self) -> np.ndarray:
        return g.apply(self.T, self.orig)

    @property
    def front_up(self) -> bool:
        """True if the paper's front (colored) side faces the viewer."""
        return bool(np.linalg.det(self.T[:2, :2]) > 0)

    @property
    def area(self) -> float:
        return g.area(self.orig)


@dataclass
class Crease:
    p: tuple[float, float]  # endpoints in original-sheet coords
    q: tuple[float, float]
    assignment: str  # "M" or "V", as seen from the front side of the sheet


_OVL: dict = {}


def overlaps(p: np.ndarray, q: np.ndarray) -> bool:
    """Positive-area overlap of two convex polygons (with a cheap bbox reject).
    Memoised on the exact polygon bytes: candidate folds (menu) re-test the same unchanged pairs."""
    key = (p.tobytes(), q.tobytes(), p.shape[0])
    r = _OVL.get(key)
    if r is None:
        if (p.max(0) < q.min(0) + 1e-9).any() or (q.max(0) < p.min(0) + 1e-9).any():
            r = False
        else:
            r = g.overlap_area(p, q) > OVERLAP_TOL
        if len(_OVL) > 500_000:
            _OVL.clear()
        _OVL[key] = r
    return r


def overlap_pairs(polys: list[np.ndarray]) -> list[tuple[int, int]]:
    out = []
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if overlaps(polys[i], polys[j]):
                out.append((i, j))
    return out


def topo_order(n: int, above: set[tuple[int, int]]) -> tuple[list[int], bool]:
    """Bottom-to-top linear extension of `above`, as close to index order as possible.

    Returns (order, acyclic). If the relation has a cycle, the lowest-index remaining
    face is forced out and acyclic=False (drawing is then only approximately right).
    """
    below_count = [0] * n  # number of faces that must be drawn before (below) i
    ups: list[list[int]] = [[] for _ in range(n)]
    for i, j in above:  # i above j  ->  j before i
        below_count[i] += 1
        ups[j].append(i)
    import heapq

    ready = [i for i in range(n) if below_count[i] == 0]
    heapq.heapify(ready)
    done = [False] * n
    order: list[int] = []
    acyclic = True
    while len(order) < n:
        if not ready:
            acyclic = False
            k = min(i for i in range(n) if not done[i])
            below_count[k] = 0
            ready = [k]
        i = heapq.heappop(ready)
        if done[i]:
            continue
        done[i] = True
        order.append(i)
        for k in ups[i]:
            below_count[k] -= 1
            if below_count[k] == 0 and not done[k]:
                heapq.heappush(ready, k)
    return order, acyclic


@dataclass
class PaperState:
    faces: list[Face]  # draw order: index 0 = bottom, last = top (consistent with `above`)
    creases: list[Crease]
    size: float = 1.0
    # (i, j) in above  <=>  face i lies above face j; only for overlapping faces.
    # None -> derived from the list order (old "global stack" semantics).
    above: set[tuple[int, int]] | None = None
    # Sub-motions of the step that produced this state (for animation); see engine.py.
    motions: list[dict] = field(default_factory=list)
    acyclic: bool = True

    def __post_init__(self):
        if self.above is None:
            self.above = self.order_from_list()

    @classmethod
    def square(cls, size: float = 1.0) -> "PaperState":
        sq = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=float)
        return cls(faces=[Face(sq)], creases=[], size=size)

    def copy(self) -> "PaperState":
        return copy.deepcopy(self)

    # ---- layer order ------------------------------------------------------------
    def order_from_list(self) -> set[tuple[int, int]]:
        """Pairwise order implied by the list order (j > i  ->  j above i) for overlapping faces."""
        return {(j, i) for i, j in overlap_pairs([f.folded for f in self.faces])}

    def is_above(self, i: int, j: int) -> bool | None:
        """True / False if faces i and j overlap and have an order, None if unrelated."""
        if (i, j) in self.above:
            return True
        if (j, i) in self.above:
            return False
        return None

    def normalize(self) -> None:
        """Re-sort `faces` into a draw order consistent with `above` (stable)."""
        order, self.acyclic = topo_order(len(self.faces), self.above)
        if order == list(range(len(self.faces))):
            return
        pos = {old: new for new, old in enumerate(order)}
        self.faces = [self.faces[k] for k in order]
        self.above = {(pos[i], pos[j]) for i, j in self.above}

    def draw_order(self) -> list[int]:
        """Face indices bottom-to-top (compat: faces are kept normalized, so 0..n-1)."""
        return list(range(len(self.faces)))

    def face_orders(self) -> list[tuple[int, int, int]]:
        """FOLD-style faceOrders [f, g, s] for "i above j" (toward the viewer).

        FOLD spec: s=+1 means f lies on the side g's normal points to. g's normal points at the
        viewer iff g is front-up, so s = +1 if faces[j].front_up else -1."""
        return [(i, j, 1 if self.faces[j].front_up else -1) for i, j in sorted(self.above)]

    # ---- geometry ---------------------------------------------------------------
    def total_area(self) -> float:
        return sum(g.area(f.folded) for f in self.faces)

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        pts = np.vstack([f.folded for f in self.faces])
        return pts.min(axis=0), pts.max(axis=0)

    def to_folded(self, orig_pt) -> np.ndarray:
        """Folded-space position of a point given in original-sheet coords."""
        pt = np.asarray(orig_pt, dtype=float)
        for f in reversed(self.faces):
            if g.contains(f.orig, pt):
                return g.apply(f.T, pt)
        raise ValueError(f"point {tuple(pt)} is not on the paper")

    def apply_view(self, T: np.ndarray, reverse_layers: bool) -> None:
        for f in self.faces:
            f.T = T @ f.T
        if reverse_layers:
            n = len(self.faces)
            self.faces.reverse()
            self.above = {(n - 1 - j, n - 1 - i) for i, j in self.above}
