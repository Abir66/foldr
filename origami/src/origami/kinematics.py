"""Connected (kinematic) fold animation: the paper stays joined along every crease.

A step is described on pieces (the before/after common refinement, see steps_export). Pieces that
share an edge in the ORIGINAL sheet are joined by a hinge. In a flat state a hinge has a fold angle
  0 (flat open)  or  +pi / -pi (folded; + = the child lies above the parent in the parent's frame).
A spanning tree is grown from an anchor piece (the largest one that does not move). Each piece's 3D
frame is its parent's frame turned about the shared edge by the interpolated angle, so tree-joined
pieces always meet at their hinge. At t = 0 / 1 the tree reproduces the before / after state exactly.

Frames are 4x4 maps from orig coords (x, y, 0) to ENGINE 3D (x, y, z), where z is measured in layer
units times DZ (the viewer's layer gap). A flat 2D isometry T lifts to E3(T) (z -> det(T) z); a
piece at draw-order layer l sits at E3(T) . H(det(T) l), H = translation along local z.
For a tree edge parent -> child:  G_child = G_parent . R(hinge, theta) . H(delta),
where delta keeps the child at its layer height at both ends (theta = 0: h_c - h_p, theta = +-pi:
h_c + h_p, with h = det(T) * layer) and is interpolated in between.

Non-tree edges (loops) close at t = 0 and 1. With plain angle interpolation they stay closed for
simple folds (all moving hinges collinear) but open mid-motion where several creases meet at a vertex
(squash, petal, reverse fold). The tree is picked among a few greedy candidates to minimise that gap
and the drift of pieces that should stay put; if a gap remains, the whole angle path is solved so the
loops stay closed (`_solve_loops`, stored as per-piece `keys`).
"""

from __future__ import annotations

import heapq
import math

import numpy as np

from . import geometry as g

PI = math.pi
_DEBUG = False
TTOL = 1e-5


def ease(t: float) -> float:
    """Cubic in-out; identical to the viewer's `ease`."""
    return 4 * t * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def E3(T: np.ndarray) -> np.ndarray:
    G = np.eye(4)
    G[:2, :2] = T[:2, :2]
    G[:2, 3] = T[:2, 2]
    G[2, 2] = 1.0 if np.linalg.det(T[:2, :2]) > 0 else -1.0
    return G


def H(z: float) -> np.ndarray:
    G = np.eye(4)
    G[2, 3] = z
    return G


def rot_about(p, d, th: float) -> np.ndarray:
    """Rotation by th about the line through (p, 0) with direction (d, 0) (right-hand rule)."""
    k = np.array([d[0], d[1], 0.0])
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)
    G = np.eye(4)
    G[:3, :3] = R
    P = np.array([p[0], p[1], 0.0])
    G[:3, 3] = P - R @ P
    return G


def _det(T) -> float:
    return 1.0 if np.linalg.det(np.asarray(T)[:2, :2]) > 0 else -1.0


def _relation(TA, TB, p, q) -> str | None:
    """'id' (flat across the edge), 'refl' (folded over it) or None (not joined)."""
    if np.allclose(TA, TB, atol=TTOL):
        return "id"
    S = g.reflection(p, q - p)
    if np.allclose(TA @ S, TB, atol=TTOL):
        return "refl"
    return None


def _adjacent(pa: np.ndarray, pb: np.ndarray):
    lo_a, hi_a = pa.min(axis=0), pa.max(axis=0)
    lo_b, hi_b = pb.min(axis=0), pb.max(axis=0)
    if np.any(lo_a > hi_b + 1e-6) or np.any(lo_b > hi_a + 1e-6):
        return None
    return g.shared_edge(pa, pb)


def _window(t: float, w) -> float:
    w0, w1 = w
    if w1 - w0 < 1e-12:
        return 1.0 if t >= w1 else 0.0
    return ease(min(1.0, max(0.0, (t - w0) / (w1 - w0))))


def build(pieces: list[dict], *, nphases: int = 1, motions: list[dict] | None = None,
          step_sign: int = 1, drive: int | None = None, solve: bool = True) -> dict:
    """Kinematic description of one step.

    pieces: dicts with orig (n,2), Tb, Ta (3x3), bi, ai (layers) and optional chain
    ([{m, dir}] for macro steps; `motions[m]["phase"]` is already 0..nphases-1).
    drive: index of the sub-motion that drives a scripted motion (petal: the lift). Its hinges turn
    evenly and the rest follow by loop closure (`_solve_driven`) instead of the whole-path solve.
    solve: False skips the loop solve (the caller supplies scripted frames, e.g. reverse_anim).
    Returns {"pieces": [per-piece kin fields], "order": [...], "bends": [...]}.
    """
    n = len(pieces)
    origs = [g.ccw(np.asarray(p["orig"], float)) for p in pieces]
    cents = [o.mean(axis=0) for o in origs]
    areas = [g.area(o) for o in origs]
    lay = {"b": [p["bi"] for p in pieces], "a": [p["ai"] for p in pieces]}
    T = {"b": [np.asarray(p["Tb"], float) for p in pieces], "a": [np.asarray(p["Ta"], float) for p in pieces]}
    hval = {k: [_det(T[k][i]) * lay[k][i] for i in range(n)] for k in ("b", "a")}
    still = [bool(np.allclose(T["b"][i], T["a"][i], atol=TTOL)) for i in range(n)]
    msets = [{c["m"] for c in (p.get("chain") or [])} for p in pieces]

    # ---- hinge graph ----
    edges = []
    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            seg = _adjacent(origs[i], origs[j])
            if seg is None:
                continue
            p, q = (np.asarray(x, float) for x in seg)
            rb = _relation(T["b"][i], T["b"][j], p, q)
            ra = _relation(T["a"][i], T["a"][j], p, q)
            if rb is None or ra is None:
                continue
            win = [0.0, 1.0]
            if motions and nphases > 1:
                ph = [int(motions[m]["phase"]) for m in msets[i] ^ msets[j]]
                if ph:
                    win = [min(ph) / nphases, (max(ph) + 1) / nphases]
            e = {"i": i, "j": j, "p": p, "q": q, "len": float(np.linalg.norm(q - p)), "rel": {"b": rb, "a": ra}, "win": win}
            e["changing"] = rb != ra or (rb == "refl" and _theta(e, i, j, "b", T, lay) != _theta(e, i, j, "a", T, lay))
            adj[i].append(len(edges))
            adj[j].append(len(edges))
            edges.append(e)

    def directed(ei: int, par: int) -> dict:
        e = edges[ei]
        ch = e["j"] if par == e["i"] else e["i"]
        p, q = e["p"], e["q"]
        u = g.unit(q - p)
        nrm = np.array([-u[1], u[0]])
        if np.dot(cents[ch] - p, nrm) < 0:
            nrm = -nrm                                   # in-plane normal pointing into the child
        d = np.array([nrm[1], -nrm[0]])                 # +theta lifts the child toward the parent's +z
        th = [_theta(e, par, ch, k, T, lay) for k in ("b", "a")]
        dl = [hval[k][ch] - hval[k][par] if e["rel"][k] == "id" else hval[k][ch] + hval[k][par]
              for k in ("b", "a")]
        return {"parent": par, "child": ch, "p": p, "q": q, "d": d, "n": nrm, "theta": th, "delta": dl,
                "win": e["win"] if e["changing"] else [0.0, 1.0]}

    # ---- components and anchors ----
    comp = [-1] * n
    comps = []
    for s in range(n):
        if comp[s] >= 0:
            continue
        stack, members = [s], []
        comp[s] = len(comps)
        while stack:
            x = stack.pop()
            members.append(x)
            for ei in adj[x]:
                e = edges[ei]
                y = e["j"] if e["i"] == x else e["i"]
                if comp[y] < 0:
                    comp[y] = len(comps)
                    stack.append(y)
        comps.append(members)

    def prim(root: int, key) -> dict[int, int]:
        """child -> edge index, grown greedily from root by `key(edge)`."""
        seen, par_edge, heap, cnt = {root}, {}, [], 0
        for ei in adj[root]:
            heapq.heappush(heap, (key(edges[ei]), cnt, ei, root)); cnt += 1
        while heap:
            _, _, ei, x = heapq.heappop(heap)
            e = edges[ei]
            y = e["j"] if e["i"] == x else e["i"]
            if y in seen:
                continue
            seen.add(y)
            par_edge[y] = ei
            for ej in adj[y]:
                heapq.heappush(heap, (key(edges[ej]), cnt, ej, y)); cnt += 1
        return par_edge

    strategies = [
        lambda e: (1 if e["changing"] else 0, -e["len"]),   # rigid (unchanged) joints first
        lambda e: (0 if e["changing"] else 1, -e["len"]),   # crease being folded first
        lambda e: (-e["len"],),                              # longest joints first
    ]

    out_p: list[dict] = [{} for _ in range(n)]
    order: list[int] = []
    gap = 0.0
    for members in comps:
        cands = [i for i in members if still[i]] or members
        root = max(cands, key=lambda i: areas[i])
        best = None
        for si, key in enumerate(strategies):
            pe = prim(root, key)
            kin = _assemble(root, pe, edges, directed, pieces, T, hval, cents, step_sign, still[root])
            score = _score(kin, edges, pe, members, still, T, origs)
            if best is None or score < best[0] - 1e-9:
                best = (score, kin, pe)
            if _DEBUG:
                print(f"    strategy {si}: score {score:.4f}")
        kin = best[1]
        if solve and best[0] > 1e-6:  # loops don't close mid-motion with plain interpolation: solve for the angles
            tree = set(best[2].values())
            gd = None
            if drive is not None:
                drivers = {c for c, kc in kin["pieces"].items()
                           if kc["parent"] >= 0 and msets[c] ^ msets[kc["parent"]] == {drive}}
                gd = _solve_driven(kin, edges, tree, still, drivers)
            gap = max(gap, gd if gd is not None else _solve_loops(kin, edges, tree, nphases, still))
        for i, kp in kin["pieces"].items():
            if "Tb" in kp:
                kp["Tb"] = [[float(v) for v in row] for row in np.asarray(kp["Tb"])[:2]]
            out_p[i] = kp
        order += kin["order"]

    # bends: every joint folded (+-pi) at either end; the viewer draws it in f's frame, bulging toward n
    # (into o's side of the sheet), whenever o currently lies folded flat onto f
    bends = []
    for ei, e in enumerate(edges):
        if "refl" not in (e["rel"]["b"], e["rel"]["a"]):
            continue
        # frame piece: the tree parent if this is a tree edge, else the lower index
        f, o = e["i"], e["j"]
        if out_p[e["i"]].get("edge") == ei:
            f, o = e["j"], e["i"]
        dd = directed(ei, f)
        bends.append({"f": f, "o": o, "seg": [_pt(e["p"]), _pt(e["q"])], "n": _pt(dd["n"]),
                      "theta": dd["theta"]})
    for kp in out_p:
        kp.pop("edge", None)
    # timing: solved paths are not even in time (a reverse fold snaps through the middle), so keyed
    # steps get a retime table that makes the largest vertex move at a steady speed (before easing)
    keyed = any("keys" in kp for kp in out_p)
    if not solve:   # scripted frames replace this path; the caller computes timing
        return {"pieces": out_p, "order": order, "bends": bends, "gap": gap, "retime": None, "path_len": 0.0}
    prof = path_profile(out_p, order, origs, RETIME_SAMPLES if keyed else 16)
    retime = retime_table(prof) if keyed else None
    return {"pieces": out_p, "order": order, "bends": bends, "gap": gap, "retime": retime,
            "path_len": float(prof.sum())}


def _theta(e: dict, par: int, ch: int, k: str, T, lay) -> float:
    if e["rel"][k] == "id":
        return 0.0
    above = _det(T[k][par]) * (lay[k][ch] - lay[k][par]) > 0
    return PI if above else -PI


def _pt(v) -> list[float]:
    return [float(v[0]), float(v[1])]


def _root_motion(pc: dict, Tb, Ta, cent, step_sign: int) -> dict | None:
    M = Ta @ np.linalg.inv(Tb)
    if np.allclose(M, np.eye(3), atol=TTOL):
        return None
    L, t = M[:2, :2], M[:2, 2]
    if np.linalg.det(L) < 0:
        S = L + np.eye(2)
        u = g.unit(S[:, int(np.argmax(np.linalg.norm(S, axis=0)))])
        a = t / 2
        c = g.apply(Tb, cent)
        nrm = np.array([-u[1], u[0]])
        if np.dot(c - a, nrm) < 0:
            nrm = -nrm
        return {"type": "hinge", "p": _pt(a), "d": _pt([nrm[1], -nrm[0]]), "angle": float(step_sign * PI)}
    ang = math.atan2(L[1, 0], L[0, 0])
    if abs(ang) < 1e-9:
        return {"type": "translate", "t": _pt(t)}
    c = np.linalg.solve(np.eye(2) - L, t)
    return {"type": "rotate", "center": _pt(c), "degrees": math.degrees(ang)}


def _assemble(root, par_edge, edges, directed, pieces, T, hval, cents, step_sign, root_still) -> dict:
    """Per-piece kin fields; `par_edge` (child -> edge) is in prim order, so parents come first."""
    out = {root: {"parent": -1, "h": [hval["b"][root], hval["a"][root]],
                  "root_motion": None if root_still else _root_motion(pieces[root], T["b"][root], T["a"][root],
                                                                       cents[root], step_sign),
                  "Tb": T["b"][root]}}
    order = [root]
    for c, ei in par_edge.items():
        e = edges[ei]
        dd = directed(ei, e["j"] if e["i"] == c else e["i"])
        out[c] = {"parent": int(dd["parent"]), "edge": ei, "hinge": [_pt(dd["p"]), _pt(dd["d"])],
                  "theta": [float(x) for x in dd["theta"]], "delta": [float(x) for x in dd["delta"]],
                  "win": [float(x) for x in dd["win"]], "h": [hval["b"][c], hval["a"][c]]}
        order.append(c)
    return {"pieces": out, "order": order}


# ---- forward kinematics (mirror of the viewer's JS; used for tree choice and tests) ----

def _root_at(kp: dict, e: float) -> np.ndarray:
    """Root frame (without layer offset) at eased progress e."""
    G = E3(np.asarray(kp["Tb"], float))
    m = kp.get("root_motion")
    if m:
        if m["type"] == "hinge":
            G = rot_about(m["p"], m["d"], m["angle"] * e) @ G
        elif m["type"] == "rotate":
            G = _rot_z(m["center"], m["degrees"] * e) @ G
        else:
            M = np.eye(4)
            M[:2, 3] = np.asarray(m["t"]) * e
            G = M @ G
    return G


def _rot_z(c, deg: float) -> np.ndarray:
    a = math.radians(deg)
    G = np.eye(4)
    G[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    G[:2, 3] = np.asarray(c) - G[:2, :2] @ np.asarray(c)
    return G


def retime_at(retime, e: float) -> float:
    """Path parameter u for eased progress e (piecewise linear table over uniform e; identity if None)."""
    if not retime:
        return e
    R = len(retime) - 1
    x = min(R, max(0.0, e * R))
    i = min(R - 1, int(math.floor(x)))
    return retime[i] + (retime[i + 1] - retime[i]) * (x - i)


def hinge_angle(kp: dict, t: float, u: float | None = None) -> float:
    """Current fold angle of a piece's hinge (same rule as the viewer's `hingeAngle`).
    u: path parameter (default ease(t)); solved `keys` are sampled uniformly in u."""
    if u is None:
        u = ease(t)
    keys = kp.get("keys")
    if keys:
        S = len(keys) - 1
        x = min(S, max(0.0, u * S))
        i = min(S - 1, int(math.floor(x)))
        return keys[i] + (keys[i + 1] - keys[i]) * (x - i)
    w = kp["win"]
    f = u if (w[0] <= 0.0 and w[1] >= 1.0) else _window(t, w)
    return kp["theta"][0] + (kp["theta"][1] - kp["theta"][0]) * f


def frames(kpieces: list[dict], order: list[int], t: float, dz: float = 0.0, retime=None) -> list[np.ndarray]:
    """4x4 frame per piece at linear step progress t: local (x, y, 0) -> engine 3D (z in world units).
    retime: the step's `retime` table (path parameter u = retime(ease(t))), if any."""
    return _frames_u(kpieces, order, t, retime_at(retime, ease(t)), dz)


def _frames_u(kpieces: list[dict], order: list[int], t: float, u: float, dz: float = 0.0) -> list[np.ndarray]:
    if kpieces and kpieces[0].get("fkeys"):   # scripted frames (inside reverse "fake spread", reverse_anim.py)
        from .reverse_anim import frame_at
        return [frame_at(kp["fkeys"], u, dz) for kp in kpieces]
    G: list[np.ndarray | None] = [None] * len(kpieces)
    for i in order:
        kp = kpieces[i]
        if kp["parent"] < 0:
            hb, ha = kp["h"]
            G[i] = _root_at(kp, u) @ H(dz * (hb + (ha - hb) * u))
        else:
            th = hinge_angle(kp, t, u)
            span = kp["theta"][1] - kp["theta"][0]
            f = min(1.0, max(0.0, (th - kp["theta"][0]) / span)) if abs(span) > 1e-9 and not kp.get("keys") else u
            dl = kp["delta"][0] + (kp["delta"][1] - kp["delta"][0]) * f
            (p, d) = kp["hinge"]
            G[i] = G[kp["parent"]] @ rot_about(p, d, th) @ H(dz * dl)
    return G


# ---- timing: even speed along the path ----

RETIME_SAMPLES = 64


def path_profile(kpieces: list[dict], order: list[int], polys: list, n: int = 64) -> np.ndarray:
    """Largest vertex displacement between consecutive samples u_s = s/n of the path parameter
    (pure geometry, no layer offsets). Length n."""
    V = [np.array([[x, y, 0.0, 1.0] for x, y in poly]) for poly in polys]
    prev, out = None, []
    for s in range(n + 1):
        u = s / n
        G = _frames_u(kpieces, order, u, u)
        X = np.concatenate([(V[i] @ G[i].T)[:, :3] for i in range(len(V))])
        if prev is not None:
            out.append(float(np.max(np.linalg.norm(X - prev, axis=1))))
        prev = X
    return np.array(out)


def retime_table(d: np.ndarray, samples: int = RETIME_SAMPLES) -> list[float]:
    """Inverse cumulative-motion map: entry i is the path parameter u at which a fraction i/samples of
    the total motion is done. Monotone, ends exactly 0 and 1. d: per-interval motion over uniform u."""
    d = np.asarray(d, float)
    n = len(d)
    d = d + 1e-3 * max(float(d.mean()), 1e-12)          # strictly increasing even where nothing moves
    c = np.concatenate([[0.0], np.cumsum(d)])
    c /= c[-1]
    us = np.linspace(0.0, 1.0, n + 1)
    tab = np.interp(np.linspace(0.0, 1.0, samples + 1), c, us)
    tab = np.maximum.accumulate(tab)
    tab[0], tab[-1] = 0.0, 1.0
    return [round(float(v), 6) for v in tab]


def _score(kin, edges, par_edge, members, still, T, origs) -> float:
    """Worst loop gap + worst drift of pieces that should not move, sampled mid-motion."""
    kp = [kin["pieces"].get(i, {"parent": -1, "h": [0, 0], "Tb": np.eye(3)}) for i in range(len(origs))]
    tree = set(par_edge.values())
    worst = 0.0
    for t in (0.25, 0.5, 0.75):
        G = frames(kp, kin["order"], t)
        for ei, e in enumerate(edges):
            if ei in tree or e["i"] not in kin["pieces"]:
                continue
            for v in (e["p"], e["q"]):
                x = np.array([v[0], v[1], 0.0, 1.0])
                worst = max(worst, float(np.linalg.norm((G[e["i"]] @ x - G[e["j"]] @ x)[:3])))
        for i in members:
            if still[i]:
                R = E3(T["b"][i])
                for v in origs[i]:
                    x = np.array([v[0], v[1], 0.0, 1.0])
                    worst = max(worst, 2 * float(np.linalg.norm((G[i] @ x - R @ x)[:3])))
    return worst


# ---- loop closure: rigid-origami style angle path solve ----

KEYS_PER_PHASE = 24     # angle samples per phase (at least 32 per step)
FREEZE_MAX_GAP = 2e-3  # freezing still joints may not open loops wider than this (else solve without)


def _solve_loops(kin: dict, edges: list[dict], tree: set[int], nphases: int, still: list[bool] | None = None) -> float:
    """Solve the angle path with joints between still pieces frozen; if that opens the loops too
    much (e.g. reverse folds, where the body layers must part to let the flap in), solve again with
    those joints only softly held and keep the better path."""
    if not still:
        return _solve_path(kin, edges, tree, nphases, None)
    gap = _solve_path(kin, edges, tree, nphases, still)
    if gap <= FREEZE_MAX_GAP:
        return gap
    kp = kin["pieces"]
    saved = {c: kp[c]["keys"] for c in kp if "keys" in kp[c]}
    gap2 = _solve_path(kin, edges, tree, nphases, None)
    if gap2 < gap:
        return gap2
    for c, k in saved.items():
        kp[c]["keys"] = k
    return gap


def _solve_path(kin: dict, edges: list[dict], tree: set[int], nphases: int, still: list[bool] | None) -> float:
    """Hinge-angle path that keeps every non-tree joint (loop) closed, as far as possible.

    Plain interpolation of fold angles does not keep loops closed (e.g. squash, petal, reverse fold:
    several creases meet at a vertex and must move in a coupled way). The whole path is solved at
    once: angles x_s at S+1 samples of eased progress u = s/S, ends fixed to the before / after
    angles, minimising
        lam * sum |loop gap(x_s)|^2  +  mu * sum |x_{s+1} - x_s|^2  +  alpha * sum |W (x_s - lerp_s)|^2
    (Levenberg-Marquardt, lam raised in stages). The smoothness term keeps the path continuous and
    short, the small prior keeps it near the plain interpolation; unchanged joints get a higher
    weight W so they only move if they must; a joint between two pieces that both stay put is frozen
    (otherwise still layers flex mid-motion and pass through each other). Flat states are singular for per-sample solves (several
    branches leave them), which is why the path is solved as one piece.
    Angles are stored per piece as `keys` (viewer: theta(t) = interp(keys, ease(t))).
    Returns the worst remaining gap (paper units).
    """
    S = max(32, KEYS_PER_PHASE * max(1, nphases))
    us = np.linspace(0.0, 1.0, S + 1)
    sysm = _loop_system(kin, edges, tree, us)
    if sysm is None:
        return 0.0
    kp, idx, m, th0, th1, residual = (sysm[k] for k in ("kp", "idx", "m", "th0", "th1", "residual"))
    kids = list(idx)
    Wt = np.where(th0 != th1, 1.0, 4.0) ** 2
    frozen = np.array([bool(still and still[c] and still[kp[c]["parent"]]) for c in kids])
    Wt[frozen] = 1e8

    x0, x1 = th0.copy(), th1.copy()
    prior = th0[None] + (th1 - th0)[None] * us[:, None]       # plain interpolation (in eased progress)
    X = prior.copy()
    n_in = S - 1
    MU, ALPHA = 1.0, 0.02

    def cost(X, lam):
        cl = sum(float(np.sum(residual(X[s], s, jac=False)[0] ** 2)) for s in range(1, S))
        sm = float(np.sum(np.diff(X, axis=0) ** 2))
        pr = float(np.sum(Wt * (X[1:-1] - prior[1:-1]) ** 2))
        return lam * cl + MU * S * sm + ALPHA * pr

    for lam in (1e2, 1e3, 1e4, 1e5):
        nu = 1e-3
        cur = cost(X, lam)
        for _ in range(25):
            # normal equations: block tridiagonal, diagonal blocks D_s, off-diagonal -cS * I
            D, gv = [], []
            for s in range(1, S):
                c, J = residual(X[s], s)
                D.append(lam * J.T @ J + 2 * MU * S * np.eye(m) + ALPHA * np.diag(Wt))
                gv.append(lam * J.T @ c + MU * S * (2 * X[s] - X[s - 1] - X[s + 1]) + ALPHA * Wt * (X[s] - prior[s]))
            improved = False
            for _ in range(8):
                dx = _block_tridiag_solve([Ds + nu * np.diag(np.diag(Ds)) for Ds in D], MU * S, [-g_ for g_ in gv])
                Xn = X.copy()
                Xn[1:-1] += dx
                new = cost(Xn, lam)
                if new < cur:
                    X, cur, nu, improved = Xn, new, max(nu / 3, 1e-7), True
                    break
                nu *= 4
            if not improved or np.max(np.abs(dx)) < 1e-7:
                break
    X[0], X[-1] = x0, x1
    X[:, frozen] = th0[frozen]
    worst = max(float(np.max(np.abs(residual(X[s], s, jac=False)[0]))) for s in range(S + 1))
    turn = float(np.max(np.abs(np.diff(X, axis=0))))
    if _DEBUG:
        print(f"    path: {len(sysm['loops'])} loop joints, {m} hinges, {S} samples, worst gap {worst:.2e}, "
              f"largest turn per sample {turn:.3f} rad")
    for c, k in idx.items():
        col = [round(float(v), 6) for v in X[:, k]]
        col[0], col[-1] = float(x0[k]), float(x1[k])          # ends exact
        kp[c]["keys"] = col
    return worst


def _loop_system(kin: dict, edges: list[dict], tree: set[int], us) -> dict | None:
    """Forward kinematics of the hinge tree + loop residuals at path samples `us` (shared by the solvers).
    residual(x, s): gaps of every non-tree joint (both end points, 3D) for hinge angles x at sample s,
    and its Jacobian d gap / d x."""
    kp, order = kin["pieces"], kin["order"]
    kids = [c for c in order if kp[c]["parent"] >= 0]
    loops = [e for ei, e in enumerate(edges) if ei not in tree and e["i"] in kp and e["j"] in kp]
    if not kids or not loops:
        return None
    m = len(kids)
    idx = {c: k for k, c in enumerate(kids)}
    pieces = list(kp)
    pidx = {c: k for k, c in enumerate(pieces)}
    anc = np.zeros((len(pieces), m), bool)
    for c in order:
        par = kp[c]["parent"]
        if par >= 0:
            anc[pidx[c]] = anc[pidx[par]]
            anc[pidx[c], idx[c]] = True
    A = np.array([pidx[e["i"]] for e in loops for _ in (0, 1)])
    B = np.array([pidx[e["j"]] for e in loops for _ in (0, 1)])
    V = np.array([[*v, 0.0, 1.0] for e in loops for v in (e["p"], e["q"])])
    mA, mB = anc[A].astype(float)[:, :, None], anc[B].astype(float)[:, :, None]
    hp = np.array([[*kp[c]["hinge"][0], 0.0] for c in kids])
    hd = np.array([[*kp[c]["hinge"][1], 0.0] for c in kids])
    hd /= np.linalg.norm(hd, axis=1, keepdims=True)
    K = np.zeros((m, 3, 3))                                   # cross-product matrices of the axes
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -hd[:, 2], hd[:, 1], -hd[:, 0]
    K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = hd[:, 2], -hd[:, 1], hd[:, 0]
    KK = K @ K
    par_ix = [pidx[kp[c]["parent"]] for c in kids]
    kid_ix = [pidx[c] for c in kids]
    roots = [c for c in order if kp[c]["parent"] < 0]
    th0 = np.array([kp[c]["theta"][0] for c in kids])
    th1 = np.array([kp[c]["theta"][1] for c in kids])

    root_G = [{pidx[r]: _root_at(kp[r], float(u)) for r in roots} for u in us]

    def fk(x, s):
        sn, cs = np.sin(x)[:, None, None], np.cos(x)[:, None, None]
        R = np.eye(3) + sn * K + (1 - cs) * KK                 # (m, 3, 3)
        Ms = np.zeros((m, 4, 4))
        Ms[:, :3, :3] = R
        Ms[:, :3, 3] = hp - np.einsum("mij,mj->mi", R, hp)
        Ms[:, 3, 3] = 1
        G = np.empty((len(pieces), 4, 4))
        for r, Gr in root_G[s].items():
            G[r] = Gr
        for k in range(m):
            G[kid_ix[k]] = G[par_ix[k]] @ Ms[k]
        return G

    def residual(x, s, jac=True):
        G = fk(x, s)
        XA = np.einsum("rij,rj->ri", G[A], V)[:, :3]
        XB = np.einsum("rij,rj->ri", G[B], V)[:, :3]
        c = (XA - XB).ravel()
        if not jac:
            return c, None
        Gp = G[par_ix]
        P = np.einsum("mij,mj->mi", Gp[:, :3, :3], hp) + Gp[:, :3, 3]
        w = np.einsum("mij,mj->mi", Gp[:, :3, :3], hd)
        J = (np.cross(w[None], XA[:, None] - P[None]) * mA - np.cross(w[None], XB[:, None] - P[None]) * mB)
        return c, J.transpose(0, 2, 1).reshape(-1, m)

    return {"kp": kp, "idx": idx, "m": m, "th0": th0, "th1": th1, "residual": residual, "loops": loops}


# ---- scripted (driven) motion: petal fold ----

DRIVE_SAMPLES = 96      # keys per driven step
DRIVE_END_TOL = 0.05    # rad: the followed branch must land on the after angles


def _solve_driven(kin: dict, edges: list[dict], tree: set[int], still: list[bool], drivers: set[int]) -> float | None:
    """Animate a step the way a person does it: one driving fold (the petal's lift) turns evenly from
    its before to its after angle; every other free hinge follows so the loops stay closed.

    Per sample the driver angles are set, then the free angles are corrected from a predictor
    (lerp step first, then secant from the previous two samples) by Gauss-Newton with minimum-norm
    steps, i.e. the joints move as little as possible from the previous frame. Creases that are flat at
    both ends may open mid-motion if closure needs it. Joints between two still pieces are frozen.
    Returns the worst loop gap, or None if no driver is a tree hinge or the branch followed does not end
    at the after state (caller falls back to the general path solve)."""
    S = DRIVE_SAMPLES
    us = np.linspace(0.0, 1.0, S + 1)
    sysm = _loop_system(kin, edges, tree, us)
    if sysm is None:
        return None
    kp, idx, th0, th1, residual = (sysm[k] for k in ("kp", "idx", "th0", "th1", "residual"))
    kids = list(idx)
    drv = np.array([c in drivers for c in kids])
    frozen = np.array([bool(still[c] and still[kp[c]["parent"]]) for c in kids]) & ~drv
    free = ~drv & ~frozen
    if not drv.any():
        return None
    X = np.empty((S + 1, len(kids)))
    X[0] = th0
    for s in range(1, S + 1):
        x = X[s - 1].copy() if s == 1 else 2 * X[s - 1] - X[s - 2]
        if s == 1:
            x[free] += (th1 - th0)[free] / S
        x[drv] = th0[drv] + (th1 - th0)[drv] * us[s]
        x[frozen] = th0[frozen]
        for _ in range(40):
            r, J = residual(x, s)
            if np.max(np.abs(r)) < 1e-11:
                break
            dx = -np.linalg.lstsq(J[:, free], r, rcond=1e-8)[0]
            big = np.max(np.abs(dx)) if dx.size else 0.0
            if big > 0.3:
                dx *= 0.3 / big
            x[free] += dx
        X[s] = x
    # The followed branch may land on th1 + 2k*pi: the same pose (a fold flat by +pi or -pi is one
    # rotation), reached by swinging the way loop closure allows. Keep the continuous angle as the
    # end key so the joint does not spin a full turn in the last frame.
    off = np.angle(np.exp(1j * (X[-1] - th1)))
    if np.max(np.abs(off)) > DRIVE_END_TOL:
        if _DEBUG:
            print(f"    driven: wrong branch, end off by {np.max(np.abs(off)):.3f} rad")
        return None
    X[-1] = th1 + np.round((X[-1] - th1) / (2 * PI)) * 2 * PI
    worst = max(float(np.max(np.abs(residual(X[s], s, jac=False)[0]))) for s in range(S + 1))
    for c, k in idx.items():
        col = [round(float(v), 6) for v in X[:, k]]
        col[0], col[-1] = float(th0[k]), float(X[-1, k])
        kp[c]["keys"] = col
    return worst


def _block_tridiag_solve(D: list[np.ndarray], c: float, r: list[np.ndarray]) -> np.ndarray:
    """Solve D_s x_s - c x_{s-1} - c x_{s+1} = r_s (block Thomas algorithm)."""
    n = len(D)
    Dinv, rp = [None] * n, [None] * n
    for s in range(n):
        Ds, rs = D[s], r[s]
        if s:
            Ds = Ds - c * c * Dinv[s - 1]
            rs = rs + c * (Dinv[s - 1] @ rp[s - 1])
        Dinv[s], rp[s] = np.linalg.inv(Ds), rs
    x = np.zeros((n, len(r[0])))
    x[-1] = Dinv[-1] @ rp[-1]
    for s in range(n - 2, -1, -1):
        x[s] = Dinv[s] @ (rp[s] + c * x[s + 1])
    return x
