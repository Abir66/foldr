"""Inside reverse fold animation: "fake spread" (scripted, not rigid-panel kinematics).

With rigid panels an inside reverse fold can't be animated without the tip passing through the
flap's own layers, unless the whole flap opens flat. Real
paper bends instead: the two halves of the flap open a little, the tip slips in between them and
the halves close again. This module fakes that:

  * the still layers of the flap on the tip's side of the spine (the folded edge joining the
    flap's front and back halves) turn about the spine: front half up, back half down, by
    alpha(u) = A sin(pi u) (fading out away from the tip, so the far body barely moves);
  * each tip half (the pieces of one sub-motion) turns rigidly about the vertex V where the
    reverse line meets the spine: its spine edge sweeps in the middle plane between the halves
    from the old spine direction to its mirror image across the reverse line (gamma: 0 -> 2 beta),
    while the half itself turns over about that edge (omega: 0 -> pi), staying on its own side of
    the middle plane, so the front and back tip halves never cross each other. omega is chosen
    per key by a small search (dynamic programming over a grid) that keeps the tip on the inner
    side of the spread halves and off the still paper beside the spine, while staying close to
    the reverse crease;
  * every piece's layer offset is interpolated from its before to its after height.

The tip's joint along the reverse crease (and the still layers' joints where the spread fades) open
a little mid-move: that is the allowed "fake". Ends are exact: at u = 0 / 1 every frame is the
before / after state.

Output per piece: `fkeys`, frames sampled uniformly in the path parameter u (S + 1 keys); each key is
15 numbers [R (3x3 row-major), t (3), w (3)] meaning frame = [[R, t + dz * w], [0, 1]] (dz = layer gap).
"""

from __future__ import annotations

import math

import numpy as np

from . import geometry as g

PI = math.pi
_DEBUG = bool(__import__("os").environ.get("SPREAD_DEBUG"))
FKEYS = 48            # key frames per step
NOMEGA = 73           # omega grid (0..pi)
SPREAD_DEG = 50.0     # peak spread of the flap halves (each half, degrees) for a ~160 deg reverse
RAMP = 0.3            # alpha opens over the first 30% of the move, closes over the last 30%
LAG = 0.15            # the tip starts turning after 15% (the flap opens first) and ends 15% early


def _rot(axis_p, axis_d, th: float) -> np.ndarray:
    """4x4 rotation by th about the 3D line through axis_p with direction axis_d."""
    k = np.asarray(axis_d, float)
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)
    G = np.eye(4)
    G[:3, :3] = R
    P = np.asarray(axis_p, float)
    G[:3, 3] = P - R @ P
    return G


def _E3(T) -> np.ndarray:
    T = np.asarray(T, float)
    G = np.eye(4)
    G[:2, :2] = T[:2, :2]
    G[:2, 3] = T[:2, 2]
    G[2, 2] = 1.0 if np.linalg.det(T[:2, :2]) > 0 else -1.0
    return G


def _det(T) -> float:
    return 1.0 if np.linalg.det(np.asarray(T, float)[:2, :2]) > 0 else -1.0


def _key(P: np.ndarray, B: np.ndarray, zl: float) -> list[float]:
    """Key for frame P . B . H(dz * zl): rotation, translation, dz-direction."""
    M = P @ B
    w = M[:3, 2] * zl
    return [round(float(v), 7) for v in [*M[:3, :3].ravel(), *M[:3, 3], *w]]


def key_frame(k, dz: float = 0.0) -> np.ndarray:
    G = np.eye(4)
    G[:3, :3] = np.asarray(k[:9]).reshape(3, 3)
    G[:3, 3] = np.asarray(k[9:12]) + dz * np.asarray(k[12:15])
    return G


def frame_at(fkeys, u: float, dz: float = 0.0) -> np.ndarray:
    """Frame at path parameter u (linear blend of the two neighbouring keys)."""
    S = len(fkeys) - 1
    x = min(S, max(0.0, u * S))
    i = min(S - 1, int(math.floor(x)))
    f = x - i
    k = (1 - f) * np.asarray(fkeys[i], float) + f * np.asarray(fkeys[i + 1], float)
    return key_frame(k, dz)


def _dist_to_poly(p: np.ndarray, poly: np.ndarray) -> float:
    """Distance from point p to the polygon's boundary (0 if inside)."""
    if g.contains(g.ccw(poly), p):
        return 0.0
    best = math.inf
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        e = b - a
        s = 0.0 if e @ e < 1e-18 else min(1.0, max(0.0, float((p - a) @ e / (e @ e))))
        best = min(best, float(np.linalg.norm(a + s * e - p)))
    return best


def _smooth(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def _samples(poly: np.ndarray) -> np.ndarray:
    c = poly.mean(axis=0)
    pts = [c]
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        for s in (0.0, 0.5):
            p = a + s * (b - a)
            pts += [p, 0.5 * (p + c), 0.85 * p + 0.15 * c]
    return np.array(pts)


def spread_keys(pieces: list[dict], motions: list[dict], *, S: int = FKEYS,
                spread_deg: float = SPREAD_DEG) -> dict | None:
    """Scripted inside-reverse motion. pieces: orig, Tb, Ta, bi, ai, chain. Returns
    {"fkeys": [per piece], "hinge_gap": worst reverse-crease gap, "hits": sampled tip-vs-still hits}
    or None if the step doesn't look like an inside reverse fold (then the caller keeps the old path)."""
    if len(motions) != 2:
        return None
    n = len(pieces)
    origs = [g.ccw(np.asarray(p["orig"], float)) for p in pieces]
    Tb = [np.asarray(p["Tb"], float) for p in pieces]
    Ta = [np.asarray(p["Ta"], float) for p in pieces]
    mset = [{c["m"] for c in (p.get("chain") or [])} for p in pieces]
    grp = [next(iter(m)) if len(m) == 1 else (-1 if not m else -2) for m in mset]
    if any(x == -2 for x in grp):
        return None
    tips = [[i for i in range(n) if grp[i] == m] for m in (0, 1)]
    if not tips[0] or not tips[1]:
        return None
    up = 0 if np.mean([pieces[i]["bi"] for i in tips[0]]) > np.mean([pieces[i]["bi"] for i in tips[1]]) else 1
    thr = 0.5 * (min(pieces[i]["bi"] for i in tips[up]) + max(pieces[i]["bi"] for i in tips[1 - up]))

    # spine: joints between the two tip halves (folded space)
    segs = []
    for i in tips[0]:
        for j in tips[1]:
            e = g.shared_edge(origs[i], origs[j])
            if e is not None:
                segs += [g.apply(Tb[i], np.asarray(e, float))]
    if not segs:
        return None
    P = np.concatenate(segs)
    c0 = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c0)
    e1 = vt[0]
    a = np.asarray(motions[0]["a"], float)
    uL = g.unit(np.asarray(motions[0]["u"], float))
    if abs(g.cross(e1, uL)) < 1e-6:
        return None
    V = g.line_intersection(c0, c0 + e1, a, a + uL)
    tipc = np.mean([g.apply(Tb[i], origs[i]).mean(axis=0) for i in tips[0] + tips[1]], axis=0)
    if np.dot(tipc - V, e1) < 0:
        e1 = -e1
    e2 = np.array([-e1[1], e1[0]])
    if np.dot(tipc - V, e2) < 0:
        e2 = -e2
    l = uL if np.dot(uL, e2) > 0 else -uL
    beta = math.acos(max(-1.0, min(1.0, float(np.dot(e1, l)))))
    hand = 1.0 if g.cross(e1, e2) > 0 else -1.0     # e1 x e2 = hand * z
    E1, E2, Z = np.r_[e1, 0.0], np.r_[e2, 0.0], np.array([0.0, 0.0, 1.0])
    V3 = np.r_[V, 0.0]

    # still pieces on the tip's side of the spine spread (fading with distance from V)
    reach = max(float(np.max(np.linalg.norm(g.apply(Tb[i], origs[i]) - V, axis=1))) for i in tips[0] + tips[1])
    D1, D2 = reach, 1.6 * reach
    wgt = [0.0] * n
    sgn = [0.0] * n
    for i in range(n):
        if grp[i] >= 0:
            continue
        F = g.apply(Tb[i], origs[i])
        c = F.mean(axis=0)
        if np.dot(c - V, e2) <= 1e-6:
            continue
        d = _dist_to_poly(V, F)
        wgt[i] = 1.0 if d <= D1 else max(0.0, (D2 - d) / (D2 - D1))
        sgn[i] = 1.0 if pieces[i]["bi"] > thr else -1.0

    us = np.linspace(0.0, 1.0, S + 1)
    # open first, then the tip turns through, then close: alpha plateaus mid-move, gamma lags behind.
    # A shallow reverse (small beta) needs less room, so it opens less.
    A = math.radians(spread_deg) * min(1.0, beta / math.radians(80.0))
    alpha = A * np.array([_smooth(min(u, 1 - u) / RAMP) for u in us])
    gamma = 2 * beta * np.array([_smooth((u - LAG) / (1 - 2 * LAG)) for u in us])

    def spread(i, al):
        if wgt[i] == 0.0:
            return np.eye(4)
        return _rot(V3, E1, hand * sgn[i] * al * wgt[i])

    def tip_pose(gm, om, upper: bool):
        s = math.cos(gm) * E1 + math.sin(gm) * E2
        nn = -math.sin(gm) * E1 + math.cos(gm) * E2
        r2 = math.cos(om) * nn + math.sin(om) * Z
        r3 = hand * np.cross(s, r2)
        R = np.column_stack([s, r2, r3]) @ np.linalg.inv(np.column_stack([E1, E2, hand * np.cross(E1, E2)]))
        G = np.eye(4)
        G[:3, :3] = R
        G[:3, 3] = V3 - R @ V3
        if not upper:
            M = np.diag([1.0, 1.0, -1.0, 1.0])
            G = M @ G @ M
        return G

    B = [_E3(Tb[i]) for i in range(n)]
    up_tip = tips[up]
    samp = {}
    for i in up_tip:
        X = _samples(origs[i])
        samp[i] = np.c_[X, np.zeros(len(X)), np.ones(len(X))]
    still = [i for i in range(n) if grp[i] < 0 and not (wgt[i] > 0 and sgn[i] < 0)]
    hinge_pts = []  # reverse-crease joints of the upper tip: (tip piece, still piece, segment)
    for i in up_tip:
        for j in range(n):
            if grp[j] < 0:
                e = g.shared_edge(origs[i], origs[j])
                if e is not None:
                    hinge_pts.append((i, j, np.asarray(e, float)))

    omegas = np.linspace(0.0, PI, NOMEGA)

    # still pieces as padded edge arrays for a vectorised point-in-polygon test
    J = len(still)
    Emax = max(len(origs[j]) for j in still) if still else 0
    ea = np.zeros((J, Emax, 2))
    en = np.zeros((J, Emax, 2))
    for r, j in enumerate(still):
        P = origs[j]
        for q in range(len(P)):
            e = P[(q + 1) % len(P)] - P[q]
            L = float(np.linalg.norm(e))
            if L > 1e-12:
                ea[r, q] = P[q]
                en[r, q] = np.array([-e[1], e[0]]) / L     # inward normal (ccw polygon)
            else:
                ea[r, q] = P[q] - 1.0                     # degenerate: always passes
                en[r, q] = 0.0
    spread_j = np.array([wgt[j] > 0 for j in still])
    detj = np.array([B[j][2, 2] for j in still])
    hp_i = np.array([i for i, _, e in hinge_pts for _ in e], int)
    hp_j = [j for _, j, e in hinge_pts for _ in e]
    hp_x = np.array([[v[0], v[1], 0.0, 1.0] for _, _, e in hinge_pts for v in e])

    def cost_at(k: int):
        """Per omega: (sampled tip points on the wrong side of still paper, worst reverse-crease gap) at key k."""
        al = alpha[k]
        inv = np.array([np.linalg.inv(spread(j, al) @ B[j]) for j in still]) if J else np.zeros((0, 4, 4))
        hq = np.array([(spread(j, al) @ B[j]) @ x for j, x in zip(hp_j, hp_x)])[:, :3] if len(hp_j) else None
        viol = np.zeros(len(omegas))
        gap = np.zeros(len(omegas))
        for oi, om in enumerate(omegas):
            Pt = tip_pose(gamma[k], om, True)
            X = np.concatenate([(samp[i] @ (Pt @ B[i]).T) for i in up_tip])
            if J:
                L = np.einsum("jab,mb->jma", inv, X)                       # (J, m, 4) local coords
                cr = np.einsum("jemc,jec->jem", L[:, None, :, :2] - ea[:, :, None, :], en)
                ins = np.all(cr > 1e-3, axis=1)                               # (J, m)
                z = L[:, :, 2]
                bad = np.where(spread_j[:, None], z * detj[:, None] > 1e-4, np.abs(z) < 0.02)
                viol[oi] = int(np.sum(ins & bad))
            if hq is not None:
                tq = np.array([(Pt @ B[i] @ x)[:3] for i, x in zip(hp_i, hp_x)])
                gap[oi] = float(np.max(np.linalg.norm(tq - hq, axis=1)))
        return viol, gap

    C = [cost_at(k) for k in range(S + 1)]
    # dynamic programming over omega: few hits first, then a small crease gap, then smoothness
    big, lam = 1.0, 2.0
    dmax = max(2, int(math.ceil(8 * (NOMEGA - 1) / S)))
    INF = 1e18
    cost = np.full(NOMEGA, INF)
    cost[0] = 0.0
    back = np.zeros((S + 1, NOMEGA), int)
    for k in range(1, S + 1):
        node = big * C[k][0] + C[k][1]
        new = np.full(NOMEGA, INF)
        for oi in range(NOMEGA):
            lo, hi = max(0, oi - dmax), min(NOMEGA, oi + dmax + 1)
            prevc = cost[lo:hi] + lam * ((omegas[lo:hi] - omegas[oi]) ** 2)
            j = int(np.argmin(prevc))
            new[oi] = prevc[j] + node[oi]
            back[k, oi] = lo + j
        cost = new
    path = [NOMEGA - 1]
    for k in range(S, 0, -1):
        path.append(back[k, path[-1]])
    path = path[::-1]
    om = omegas[path]
    hits = int(sum(C[k][0][path[k]] for k in range(S + 1)))
    hgap = float(max(C[k][1][path[k]] for k in range(S + 1)))

    fkeys = []
    for i in range(n):
        hb = _det(Tb[i]) * pieces[i]["bi"]
        ha = _det(Ta[i]) * pieces[i]["ai"]
        ks = []
        for k, u in enumerate(us):
            zl = hb + (ha - hb) * u
            if grp[i] >= 0:
                Pt = tip_pose(gamma[k], om[k], grp[i] == up)
            else:
                Pt = spread(i, alpha[k])
            ks.append(_key(Pt, B[i], zl))
        fkeys.append(ks)
    # exact ends (the scripted pose already lands on them; this removes rounding)
    for i in range(n):
        if _DEBUG:
            end = np.asarray(fkeys[i][-1]) - np.asarray(_key(np.eye(4), _E3(Ta[i]), _det(Ta[i]) * pieces[i]["ai"]))
            assert np.abs(end).max() < 1e-5, (i, np.abs(end).max())
        fkeys[i][0] = _key(np.eye(4), B[i], _det(Tb[i]) * pieces[i]["bi"])
        fkeys[i][-1] = _key(np.eye(4), _E3(Ta[i]), _det(Ta[i]) * pieces[i]["ai"])
    if _DEBUG:
        for k in range(S + 1):
            print(f"  k={k:2d} a={math.degrees(alpha[k]):5.1f} g={math.degrees(gamma[k]):6.1f} om={math.degrees(om[k]):6.1f}"
                  f" hits={int(C[k][0][path[k]])} min_hits={int(C[k][0].min())} gap={C[k][1][path[k]]:.3f}")
    return {"fkeys": fkeys, "hinge_gap": hgap, "hits": hits, "omega": [float(x) for x in om]}
