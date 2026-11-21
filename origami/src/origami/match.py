"""Compare two folded states (goal matching) + canonical state hash (dead-end memory).


Fingerprint of a folded state (all in sheet units; scale is NOT normalised away, the paper is the
same sheet, so a 2x bigger shape is a different fold):
  - silhouette raster (union of faces),
  - visible-colour raster seen from the FRONT (top face: paper front side or back side),
  - visible-colour raster seen from the BACK (bottom face), stored in the same x/y frame,
  - translation normalised by putting the silhouette centroid at the origin.

Alignment group G (applied to state B to bring it onto A):
  rotation (any angle) x mirror image (fold every step mirrored) x turn-over (flip the model).
  Turn-over maps (front, back) -> (mirror(back), mirror(front)); mirror keeps front/back.
Search: candidate angles from dominant edge directions (exact for rotated copies) + a 15 deg grid,
scored on a coarse raster; best few refined by hill-climbing down to 1/16 deg on a fine raster.

Match levels (each a score in [0,1] + pass flag; each level requires the previous one):
  M1 outline : silhouette IoU >= 0.95 and Hausdorff distance <= 0.02*size + 1.5 px.
  M2 look    : M1 + visible-colour agreement >= 0.95 on front AND back (headline level).
               colour agreement = |pixels where both show paper of the same side| / |union|.
  M3 structure: M2 + current crease pattern (creases folded right now, M/V-agnostic) matches up to the
               sheet's 8 symmetries (D4): length-weighted F1 >= 0.9 at tolerance 0.01*size.
  M4 exact   : M3 + identical folded state: same geometry and same layer stack everywhere
               (canonical stack samples equal; implies same M/V).
Scores: M1 = IoU; M2 = min(IoU, colour_front, colour_back); M3 = min(M2, crease F1);
        M4 = min(M3, fraction of stack samples that agree).
shape_score = mean(IoU, mean colour agreement) (continuous partial credit).

Canonical hash: identical for the same folded state reached by different move orders, invariant to
translation / rotation / turn-over of the model and to the D4 symmetry of the sheet (and, by default,
to mirroring the whole model). Built from layer stacks sampled on a fixed lattice in a canonical frame.
Unfolded (history) creases are NOT part of the state hash; only the current folded geometry + layers.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from . import geometry as g
from .paper import PaperState
from .validate import folded_creases

# ---- thresholds (documented above; calibrated on the 6 models) ----
M1_IOU = 0.95
M1_HAUSDORFF = 0.02  # sheet units (x size), plus 1.5 pixels of raster slack
M2_COLOUR = 0.95
M3_CREASE_F1 = 0.90
CREASE_TOL = 0.01  # sheet units (x size)
M4_AGREE = 1.0

COARSE_N = 64
FINE_N = 160
HASH_STEP = 1 / 64  # lattice spacing for canonical stack samples (x size)
_P0 = np.array([0.1234567, 0.0456789])  # generic anchor point on the sheet (x size)
_OFF = np.array([0.3183099, 0.2718281])  # generic lattice offset (x HASH_STEP)
_PIX_OFF = 0.5 + 0.1234567  # generic sub-pixel offset: axis-aligned edges never sit on pixel centres


def _det(T: np.ndarray) -> float:
    return float(T[0, 0] * T[1, 1] - T[0, 1] * T[1, 0])


def _inv(T: np.ndarray) -> np.ndarray:
    """Inverse of a 3x3 affine map (closed form; np.linalg on macOS Accelerate raises spurious
    'overflow in det' warnings on these tiny matrices)."""
    L = T[:2, :2]
    d = _det(T)
    Li = np.array([[L[1, 1], -L[0, 1]], [-L[1, 0], L[0, 0]]]) / d
    out = np.eye(3)
    out[:2, :2] = Li
    out[:2, 2] = -Li @ T[:2, 2]
    return out


# =============================================================================================
# Fingerprint + rasters
# =============================================================================================

@dataclass
class Fingerprint:
    polys: list[np.ndarray]  # folded faces, CCW, translated so the silhouette centroid is at 0
    front_up: np.ndarray  # bool per face; faces in draw order (0 = bottom)
    centroid: np.ndarray  # silhouette centroid in the state's folded coords
    radius: float  # max distance of any vertex from the centroid
    size: float
    dirs: list[float]  # dominant edge directions, degrees mod 180

    def raster(self, n: int = FINE_N, half: float | None = None):
        """(silhouette, front_colour, back_colour) rasters, n x n over [-half, half]^2.
        Colour labels: 0 = no paper, 1 = paper front side visible, 2 = paper back side visible."""
        half = self.radius * 1.02 if half is None else half
        return _raster(self.polys, self.front_up, n, half)


def _hull(pts: np.ndarray) -> np.ndarray:
    """Convex hull (CCW), Andrew's monotone chain."""
    pts = np.unique(np.round(pts, 12), axis=0)
    if len(pts) < 3:
        return pts

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and g.cross(out[-1] - out[-2], p - out[-2]) <= 1e-15:
                out.pop()
            out.append(p)
        return out

    lower, upper = half(pts), half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def _centroid(polys: list[np.ndarray]) -> np.ndarray:
    """Area centroid of the convex hull of the silhouette. Exact and moves rigidly with the model
    (a raster centroid jitters by ~0.1 px, which flips whole pixel rows along long straight edges)."""
    H = _hull(np.vstack(polys))
    if len(H) < 3:
        return H.mean(0)
    x, y = H[:, 0], H[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y2 - x2 * y
    A = cr.sum() / 2
    return np.array([((x + x2) * cr).sum(), ((y + y2) * cr).sum()]) / (6 * A)


def _dominant_dirs(polys: list[np.ndarray], k: int = 4) -> list[float]:
    bins: dict[int, list] = {}
    for p in polys:
        d = np.roll(p, -1, axis=0) - p
        L = np.hypot(d[:, 0], d[:, 1])
        ang = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 180.0
        for a, l in zip(ang, L):
            if l < 1e-9:
                continue
            b = int(round(a * 2)) % 360
            e = bins.setdefault(b, [0.0, 0.0])
            e[0] += l
            e[1] += l * a
    top = sorted(bins.items(), key=lambda kv: -kv[1][0])[:k]
    return [v[1] / v[0] for _, v in top]


def fingerprint(state: PaperState) -> Fingerprint:
    polys = [g.ccw(f.folded) for f in state.faces]
    c = _centroid(polys)
    polys = [p - c for p in polys]
    r = float(max(np.hypot(p[:, 0], p[:, 1]).max() for p in polys))
    return Fingerprint(polys, np.array([_det(f.T) > 0 for f in state.faces]), c, r, float(state.size),
                       _dominant_dirs(polys))


def _stack_raster(polys, n, half):
    """Per pixel: index of the top face and of the bottom face covering it (-1 = none)."""
    px = 2 * half / n
    xs = (np.arange(n) + _PIX_OFF) * px - half
    top = np.full((n, n), -1, dtype=int)
    bot = np.full((n, n), -1, dtype=int)
    for k, P in enumerate(polys):
        lo, hi = P.min(0), P.max(0)
        i0 = max(0, int(np.floor((lo[0] + half) / px)))
        i1 = min(n, int(np.ceil((hi[0] + half) / px)) + 1)
        j0 = max(0, int(np.floor((lo[1] + half) / px)))
        j1 = min(n, int(np.ceil((hi[1] + half) / px)) + 1)
        if i0 >= i1 or j0 >= j1:
            continue
        X = xs[i0:i1][None, :]
        Y = xs[j0:j1][:, None]
        inside = np.ones((j1 - j0, i1 - i0), dtype=bool)
        m = len(P)
        for e in range(m):
            a, b = P[e], P[(e + 1) % m]
            inside &= (b[0] - a[0]) * (Y - a[1]) - (b[1] - a[1]) * (X - a[0]) >= -1e-12
        st = top[j0:j1, i0:i1]
        st[inside] = k
        sb = bot[j0:j1, i0:i1]
        sb[inside & (sb < 0)] = k
    return top, bot


def _raster(polys, front_up, n, half):
    top, bot = _stack_raster(polys, n, half)
    fu = np.asarray(front_up, dtype=bool)
    sil = top >= 0
    front = np.where(sil, np.where(fu[np.maximum(top, 0)], 1, 2), 0)
    back = np.where(sil, np.where(fu[np.maximum(bot, 0)], 2, 1), 0)
    return sil, front, back


def _iou(a, b) -> float:
    u = np.count_nonzero(a | b)
    return 1.0 if u == 0 else np.count_nonzero(a & b) / u


def _agree(la, lb) -> float:
    u = np.count_nonzero((la > 0) | (lb > 0))
    return 1.0 if u == 0 else np.count_nonzero((la == lb) & (la > 0)) / u


def _boundary(m):
    p = np.pad(m, 1)
    inner = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return m & ~inner


def _directed_hausdorff(a, b, px) -> float:
    out = np.argwhere(a & ~b)
    if len(out) == 0:
        return 0.0
    bb = np.argwhere(_boundary(b))
    if len(bb) == 0:
        return float("inf")
    best = 0.0
    for s in range(0, len(out), 2048):
        d = np.sqrt(((out[s:s + 2048, None, :] - bb[None, :, :]) ** 2).sum(-1)).min(1)
        best = max(best, float(d.max()))
    return best * px


def _hausdorff(a, b, px) -> float:
    return max(_directed_hausdorff(a, b, px), _directed_hausdorff(b, a, px))


# =============================================================================================
# Alignment over G
# =============================================================================================

_MX = np.diag([-1.0, 1.0])


def _rot(deg: float) -> np.ndarray:
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def _apply_g(fp: Fingerprint, deg: float, mirror: bool, flip: bool):
    """Polys + front_up of fp after mirror (model mirror image), flip (turn-over), then rotation."""
    L = _rot(deg) @ (_MX if (mirror ^ flip) else np.eye(2))
    polys = [p @ L.T for p in fp.polys] if not (mirror ^ flip) else [(p @ L.T)[::-1] for p in fp.polys]
    fu = fp.front_up.copy()
    if flip:
        polys = polys[::-1]
        fu = ~fu[::-1]
    return polys, fu


def _score(ra, rb) -> tuple[float, float, float, float]:
    iou = _iou(ra[0], rb[0])
    cf, cb = _agree(ra[1], rb[1]), _agree(ra[2], rb[2])
    return (iou + cf + cb) / 3, iou, cf, cb


@dataclass
class Alignment:
    rotation_deg: float  # applied to B (about its centroid) after mirror / flip
    mirror: bool
    flip: bool
    matrix: np.ndarray  # 3x3 map from B's folded coords to A's folded coords (geometry only)


def align(fa: Fingerprint, fb: Fingerprint, allow_mirror: bool = True):
    """Best G-transform of B onto A. Returns (Alignment, half-extent used)."""
    half = max(fa.radius, fb.radius) * 1.02 + 1e-9
    ra_c = _raster(fa.polys, fa.front_up, COARSE_N, half)
    cands = []
    for mirror in ((False, True) if allow_mirror else (False,)):
        for flip in (False, True):
            sgn = -1.0 if (mirror ^ flip) else 1.0
            bdirs = [(sgn * d) % 180 for d in fb.dirs]
            angles = {round((a - b) % 360, 6) for a in fa.dirs for b in bdirs}
            angles |= {round((x + 180) % 360, 6) for x in angles}
            angles |= {float(x) for x in range(0, 360, 15)}
            for deg in angles:
                pb, fub = _apply_g(fb, deg, mirror, flip)
                s = _score(ra_c, _raster(pb, fub, COARSE_N, half))[0]
                cands.append((s, deg, mirror, flip))
    cands.sort(key=lambda c: -c[0])
    ra_f = _raster(fa.polys, fa.front_up, FINE_N, half)

    def ev(deg, mirror, flip):
        pb, fub = _apply_g(fb, deg, mirror, flip)
        return _score(ra_f, _raster(pb, fub, FINE_N, half))[0]

    best = None
    seen = set()
    for _, deg, mirror, flip in cands:
        key = (round(deg, 3), mirror, flip)
        if key in seen:
            continue
        seen.add(key)
        s = ev(deg, mirror, flip)
        for step in (2.0, 1.0, 0.5, 0.25, 0.125, 0.0625):
            improved = True
            while improved:
                improved = False
                for d2 in (deg - step, deg + step):
                    s2 = ev(d2, mirror, flip)
                    if s2 > s + 1e-12:
                        s, deg, improved = s2, d2, True
        if best is None or s > best[0]:
            best = (s, deg % 360, mirror, flip)
        if len(seen) >= 3:
            break
    _, deg, mirror, flip = best
    L = _rot(deg) @ (_MX if (mirror ^ flip) else np.eye(2))
    M = np.eye(3)
    M[:2, :2] = L
    M[:2, 2] = fa.centroid - L @ fb.centroid
    return Alignment(float(deg), bool(mirror), bool(flip), M), half


# =============================================================================================
# Crease patterns (M3)
# =============================================================================================

def _d4(size: float) -> list[tuple[str, np.ndarray]]:
    """The 8 symmetries of the square sheet, as 3x3 affine maps on sheet coords."""
    c = np.array([size / 2, size / 2])
    out = []
    for name, L in [("id", np.eye(2)), ("r90", _rot(90)), ("r180", _rot(180)), ("r270", _rot(270)),
                    ("mx", _MX), ("mx_r90", _rot(90) @ _MX), ("mx_r180", _rot(180) @ _MX),
                    ("mx_r270", _rot(270) @ _MX)]:
        L = np.round(L)
        S = np.eye(3)
        S[:2, :2] = L
        S[:2, 2] = c - L @ c
        out.append((name, S))
    return out


def current_creases(state: PaperState) -> list[tuple]:
    """Creases folded right now (p, q, 'M'/'V'/'U') on the unfolded sheet."""
    return [c for c in folded_creases(state) if np.hypot(*np.subtract(c[1], c[0])) > 1e-9]


def _samples(creases, step):
    pts, w, dirs, asg = [], [], [], []
    for p, q, a in creases:
        p, q = np.asarray(p, float), np.asarray(q, float)
        L = float(np.linalg.norm(q - p))
        k = max(1, int(np.ceil(L / step)))
        t = (np.arange(k) + 0.5) / k
        pts.append(p + t[:, None] * (q - p))
        w += [L / k] * k
        dirs += [(q - p) / L] * k
        asg += [a] * k
    if not pts:
        return np.zeros((0, 2)), np.zeros(0), np.zeros((0, 2)), np.array([], dtype=object)
    return np.vstack(pts), np.array(w), np.array(dirs), np.array(asg, dtype=object)


def _covered_weight(pts, w, dirs, asg, segs, tol) -> tuple[float, float]:
    """Length of the sampled creases covered by `segs` (M/V-agnostic, M/V-aware)."""
    if len(pts) == 0 or not segs:
        return 0.0, 0.0
    P = np.array([s[0] for s in segs], float)
    Q = np.array([s[1] for s in segs], float)
    A = np.array([s[2] for s in segs], dtype=object)
    D = Q - P
    L2 = np.maximum((D ** 2).sum(1), 1e-18)
    t = np.clip(((pts[:, None, :] - P[None]) * D[None]).sum(-1) / L2[None], 0, 1)
    closest = P[None] + t[..., None] * D[None]
    dist = np.hypot(*(pts[:, None, :] - closest).transpose(2, 0, 1))
    U = D / np.sqrt(L2)[:, None]
    par = np.abs(dirs[:, 0:1] * U[None, :, 1] - dirs[:, 1:2] * U[None, :, 0]) < 0.05
    ok = (dist <= tol) & par
    ok_mv = ok & (asg[:, None] == A[None, :])
    return float(w[ok.any(1)].sum()), float(w[ok_mv.any(1)].sum())


def _f1(r, p):
    return 0.0 if r + p == 0 else 2 * r * p / (r + p)


def crease_f1_both(ca, cb, size: float = 1.0) -> tuple[float, float]:
    """(M/V-agnostic, M/V-aware) length-weighted F1 of two crease lists in the same sheet frame,
    tolerance CREASE_TOL*size; a sample counts only if a parallel crease is that close."""
    if not ca and not cb:
        return 1.0, 1.0
    if not ca or not cb:
        return 0.0, 0.0
    tol, step = CREASE_TOL * size, 0.005 * size
    sa, sb = _samples(ca, step), _samples(cb, step)
    ra, ra_mv = (x / sa[1].sum() for x in _covered_weight(*sa, cb, tol))
    pb, pb_mv = (x / sb[1].sum() for x in _covered_weight(*sb, ca, tol))
    return _f1(ra, pb), _f1(ra_mv, pb_mv)


def crease_f1(ca, cb, size: float = 1.0, mv: bool = False) -> float:
    return crease_f1_both(ca, cb, size)[1 if mv else 0]


def _map_creases(creases, S):
    return [(tuple(g.apply(S, np.asarray(p, float))), tuple(g.apply(S, np.asarray(q, float))), a)
            for p, q, a in creases]


# =============================================================================================
# Canonical stacks + hash (M4, dead-end memory)
# =============================================================================================

def _canonical_rows(state: PaperState, S: np.ndarray) -> np.ndarray:
    """Layer stacks of `state` in the canonical frame for sheet symmetry S.

    Canonical sheet coords p' <-> original sheet point S p'. The face holding S*P0 is mapped to
    identity (X = (T_f S)^-1); stacks are read toward that face's front side. Samples on a fixed
    lattice in the canonical folded frame. Rows: (kx, ky, rank, qx, qy) with q = canonical sheet
    coords of the layer, quantised to 1e-6*size."""
    size = float(state.size)
    p0 = g.apply(S, _P0 * size)
    f = next((k for k, F in enumerate(state.faces) if g.contains(F.orig, p0, tol=1e-9)), None)
    if f is None:
        f = next(k for k, F in enumerate(state.faces) if g.contains(F.orig, p0))
    X = _inv(state.faces[f].T @ S)
    Sinv = _inv(S)
    reverse = not _det(state.faces[f].T) > 0
    h = HASH_STEP * size
    off = _OFF * h
    rows = []
    for k, F in enumerate(state.faces):
        P = g.ccw(g.apply(X, F.folded))
        lo, hi = P.min(0), P.max(0)
        kx = np.arange(int(np.ceil((lo[0] - off[0]) / h)), int(np.floor((hi[0] - off[0]) / h)) + 1)
        ky = np.arange(int(np.ceil((lo[1] - off[1]) / h)), int(np.floor((hi[1] - off[1]) / h)) + 1)
        if len(kx) == 0 or len(ky) == 0:
            continue
        KX, KY = np.meshgrid(kx, ky)
        KX, KY = KX.ravel(), KY.ravel()
        pts = np.stack([KX * h + off[0], KY * h + off[1]], 1)
        inside = np.ones(len(pts), dtype=bool)
        m = len(P)
        for e in range(m):
            a, b = P[e], P[(e + 1) % m]
            inside &= (b[0] - a[0]) * (pts[:, 1] - a[1]) - (b[1] - a[1]) * (pts[:, 0] - a[0]) > 1e-12
        if not inside.any():
            continue
        # canonical sheet coords of these points on face k: S^-1 T_k^-1 X^-1 q
        M = Sinv @ _inv(F.T) @ _inv(X)
        q = g.apply(M, pts[inside])
        qi = np.round(q / size * 1e6).astype(np.int64)
        layer = np.full(len(qi), k)
        rows.append(np.column_stack([KX[inside], KY[inside], layer, qi]))
    if not rows:
        return np.zeros((0, 5), dtype=np.int64)
    R = np.vstack(rows).astype(np.int64)
    key_layer = R[:, 2] if reverse else -R[:, 2]  # top of the canonical stack first
    R = R[np.lexsort((key_layer, R[:, 1], R[:, 0]))]
    # replace face index by rank within each sample point's stack
    newpt = np.ones(len(R), dtype=bool)
    newpt[1:] = (R[1:, 0] != R[:-1, 0]) | (R[1:, 1] != R[:-1, 1])
    start = np.maximum.accumulate(np.where(newpt, np.arange(len(R)), 0))
    R[:, 2] = np.arange(len(R)) - start
    return R


def _stack_dict(rows: np.ndarray) -> dict:
    out: dict = {}
    for r in rows:
        out.setdefault((int(r[0]), int(r[1])), []).append((int(r[3]), int(r[4])))
    return {k: tuple(v) for k, v in out.items()}


def _sym_set(size, allow_mirror):
    d4 = _d4(size)
    return d4 if allow_mirror else d4[:4]


def state_hash(state: PaperState, allow_mirror: bool = True) -> str:
    """Canonical hash of the folded state (geometry + layer order), invariant to translation,
    rotation and turn-over of the model, to the sheet's symmetries (D4; C4 if allow_mirror=False)
    and to the order in which independent moves were made. 16 hex chars."""
    hs = []
    for _, S in _sym_set(state.size, allow_mirror):
        R = _canonical_rows(state, S)
        hs.append(hashlib.sha1(R.tobytes()).hexdigest()[:16])
    return min(hs)


@dataclass
class VisitedEntry:
    step: int  # step index where this state was first reached
    notes: list[str] = field(default_factory=list)
    count: int = 1  # how many times it was reached


class VisitedTable:
    """hash -> first step index + notes. Dead-end / tabu memory for the agent."""

    def __init__(self, allow_mirror: bool = True):
        self.allow_mirror = allow_mirror
        self.table: dict[str, VisitedEntry] = {}

    def key(self, state: PaperState) -> str:
        return state_hash(state, self.allow_mirror)

    def add(self, state: PaperState, step: int, note: str | None = None) -> tuple[str, bool]:
        """Record a visit. Returns (hash, is_new)."""
        h = self.key(state)
        e = self.table.get(h)
        if e is None:
            self.table[h] = VisitedEntry(step, [note] if note else [])
            return h, True
        e.count += 1
        if note:
            e.notes.append(note)
        return h, False

    def get(self, state: PaperState) -> VisitedEntry | None:
        return self.table.get(self.key(state))

    def note(self, state: PaperState, text: str) -> None:
        h = self.key(state)
        if h not in self.table:
            raise KeyError("state not visited yet")
        self.table[h].notes.append(text)

    def __contains__(self, state: PaperState) -> bool:
        return self.key(state) in self.table

    def __len__(self) -> int:
        return len(self.table)


# =============================================================================================
# match()
# =============================================================================================

@dataclass
class MatchResult:
    alignment: Alignment
    iou: float
    hausdorff: float  # sheet units
    colour_front: float
    colour_back: float
    crease_f1: float  # M/V-agnostic, best over sheet symmetries
    crease_f1_mv: float  # M/V-aware, same symmetry
    crease_sym: str  # D4 element mapping A's sheet onto B's sheet
    stack_agree: float  # fraction of canonical stack samples that agree (best symmetry)
    m1: bool
    m2: bool
    m3: bool
    m4: bool
    scores: dict[str, float]

    @property
    def headline(self) -> bool:
        return self.m2

    @property
    def level(self) -> int:
        """Highest level passed (0..4)."""
        return sum((self.m1, self.m2, self.m3, self.m4))

    @property
    def shape_score(self) -> float:
        return (self.iou + (self.colour_front + self.colour_back) / 2) / 2


def match(state_a: PaperState, state_b: PaperState, allow_mirror: bool = True) -> MatchResult:
    """Compare folded state B (attempt) against A (target) at levels M1-M4."""
    size = float(state_a.size)
    fa, fb = fingerprint(state_a), fingerprint(state_b)
    al, half = align(fa, fb, allow_mirror)
    pb, fub = _apply_g(fb, al.rotation_deg, al.mirror, al.flip)
    ra = _raster(fa.polys, fa.front_up, FINE_N, half)
    rb = _raster(pb, fub, FINE_N, half)
    _, iou, cf, cb = _score(ra, rb)
    px = 2 * half / FINE_N
    hd = _hausdorff(ra[0], rb[0], px)
    m1 = iou >= M1_IOU and hd <= M1_HAUSDORFF * size + 1.5 * px
    m2 = m1 and cf >= M2_COLOUR and cb >= M2_COLOUR

    ca, cbr = current_creases(state_a), current_creases(state_b)
    syms = _sym_set(size, allow_mirror)
    f1, f1_mv, _, sym_name = max((*crease_f1_both(_map_creases(ca, S), cbr, size), -k, name)
                                 for k, (name, S) in enumerate(syms))  # ties -> simplest symmetry
    m3 = m2 and f1 >= M3_CREASE_F1

    da = _stack_dict(_canonical_rows(state_a, np.eye(3)))
    agree = 0.0
    for _, S2 in syms:
        db = _stack_dict(_canonical_rows(state_b, S2))
        keys = set(da) | set(db)
        a = 1.0 if not keys else sum(da.get(k) == db.get(k) for k in keys) / len(keys)
        agree = max(agree, a)
        if agree >= 1.0:
            break
    m4 = m3 and agree >= M4_AGREE

    s1 = iou
    s2 = min(s1, cf, cb)
    s3 = min(s2, f1)
    s4 = min(s3, agree)
    return MatchResult(al, iou, hd, cf, cb, f1, f1_mv, sym_name, agree, m1, m2, m3, m4,
                       {"M1": s1, "M2": s2, "M3": s3, "M4": s4})


# =============================================================================================
# Helpers (tests, variant generation)
# =============================================================================================

def transformed(state: PaperState, rotate: float = 0.0, mirror: bool = False, flip: bool = False,
                shift=(0.0, 0.0)) -> PaperState:
    """Copy of `state` as seen after: mirror (mirror-image model: every step mirrored, sheet x -> size-x),
    flip (turn the model over), rotate (degrees about the bbox centre), then shift."""
    st = state.copy()
    size = float(st.size)
    if mirror:
        Ms = g.mirror(np.array([size / 2, size / 2]), "vertical")
        Mf = g.mirror(np.zeros(2), "vertical")
        for f in st.faces:
            f.orig = g.apply(Ms, f.orig)
            f.T = Mf @ f.T @ Ms
        for c in st.creases:
            c.p = tuple(g.apply(Ms, np.array(c.p)))
            c.q = tuple(g.apply(Ms, np.array(c.q)))
    lo, hi = st.bbox()
    c = (lo + hi) / 2
    if flip:
        st.apply_view(g.mirror(c, "vertical"), reverse_layers=True)
    if rotate:
        st.apply_view(g.rotation(c, rotate), reverse_layers=False)
    if any(shift):
        T = np.eye(3)
        T[:2, 2] = shift
        st.apply_view(T, reverse_layers=False)
    return st
