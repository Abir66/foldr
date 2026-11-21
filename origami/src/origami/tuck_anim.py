"""Tuck-inside animation: open the pocket, slide the flap in (scripted, not rigid-panel kinematics).

A valley (mountain) fold with `"tuck": "inside"` ends with the flap UNDER (over) some still layers:
the "cover". A plain rotation would swing the flap out in front of the cover and then jump behind
it when the layer order switches. Real paper: you lift the cover a little and slide the flap in.
This module fakes that with key frames (same format as reverse_anim.py, so the viewer needs nothing
new):

  * cover = still layers that end above (valley) / below (mountain) the flap, plus the layers
    fixed to them; they turn about the joint where the cover meets the rest of the still paper
    (the closed side of the pocket) by alpha(u) = A * ramp: the pocket opens, then closes again;
  * the flap turns about its hinge (0 -> pi), but its height off the paper is squashed so it
    always stays under the opened cover (a little margin below the cover's inner face): it looks
    like the flap bends and slides along inside the pocket instead of standing up in front of it;
  * every piece's layer offset is interpolated from its before to its after height.

Ends are exact: at u = 0 / 1 every frame is the before / after state.
"""

from __future__ import annotations

import math

import numpy as np

from . import geometry as g
from .reverse_anim import _E3, _det, _key, _rot, _smooth

PI = math.pi
FKEYS = 48            # key frames per step
OPEN_DEG = 30.0       # peak pocket opening (degrees)
RAMP = 0.3            # the pocket opens over the first 30% of the move and closes over the last 30%
LAG = 0.12            # the flap starts turning after 12% (the pocket opens first) and ends 12% early
MARGIN = 0.75         # the flap stays below this fraction of the opened cover's height


def _cover(pieces: list[dict], faces: list[tuple[int, int]], after, sign: int) -> set[int]:
    """Still pieces whose after-face ends above (valley) / below (mountain) a moving after-face."""
    mov_faces = {faces[i][1] for i, p in enumerate(pieces) if p["moving"]}
    cov_faces = set()
    for i, p in enumerate(pieces):
        if p["moving"]:
            continue
        fa = faces[i][1]
        if any(after.is_above(fa, m) is (sign > 0) for m in mov_faces):
            cov_faces.add(fa)
    return {i for i, p in enumerate(pieces) if not p["moving"] and faces[i][1] in cov_faces}


def tuck_keys(pieces: list[dict], faces: list[tuple[int, int]], after, a, u, sign: int, *,
              S: int = FKEYS, open_deg: float = OPEN_DEG) -> dict | None:
    """pieces: orig, Tb, Ta, bi, ai (compact heights), moving. faces: (before, after) face index per piece.
    a, u: fold line (folded space of the before state); sign: +1 valley, -1 mountain.
    Returns {"fkeys": per piece, "cover": [...], "open_deg": ...} or None (no cover: keep the plain path)."""
    n = len(pieces)
    mov = [i for i in range(n) if pieces[i]["moving"]]
    if not mov:
        return None
    cov = _cover(pieces, faces, after, sign)
    if not cov:
        return None
    origs = [g.ccw(np.asarray(p["orig"], float)) for p in pieces]
    Tb = [np.asarray(p["Tb"], float) for p in pieces]
    Ta = [np.asarray(p["Ta"], float) for p in pieces]
    Fb = [g.apply(Tb[i], origs[i]) for i in range(n)]
    a = np.asarray(a, float)
    u = g.unit(np.asarray(u, float))

    # closed side of the pocket: joints between cover and the other still pieces
    rest = [j for j in range(n) if j not in cov and not pieces[j]["moving"]]
    pts = []
    for i in cov:
        for j in rest:
            e = g.shared_edge(origs[i], origs[j])
            if e is not None:
                pts += list(g.apply(Tb[i], np.asarray(e, float)))
    cc = np.concatenate([Fb[i] for i in cov]).mean(axis=0)
    if pts:
        P = np.asarray(pts)
        c0 = P.mean(axis=0)
        _, sv, vt = np.linalg.svd(P - c0)
        d = vt[0]
        if len(P) > 2 and sv[1] > 1e-3 * max(1.0, sv[0]):
            # joints don't lie on one line: use the one farthest from the fold line, parallel to it
            far = max(pts, key=lambda p: abs(g.cross(u, p - a)))
            c0, d = np.asarray(far), u
    else:
        # cover only joined through the flap: open it about its far edge, parallel to the fold line
        nrm = np.array([-u[1], u[0]])
        if np.dot(cc - a, nrm) < 0:
            nrm = -nrm
        far = max(np.concatenate([Fb[i] for i in cov]), key=lambda p: float(np.dot(p - a, nrm)))
        c0, d = np.asarray(far), u
    d = g.unit(d)
    nc = np.array([-d[1], d[0]])           # toward the cover
    if np.dot(cc - c0, nc) < 0:
        nc = -nc
    C3, D3 = np.r_[c0, 0.0], np.r_[d, 0.0]

    def side(i):
        return float(np.dot(Fb[i].mean(axis=0) - c0, nc))

    cov = {i for i in cov if side(i) > 1e-9}   # pieces on the far side of the joint can't open with it
    if not cov:
        return None
    # open direction: the cover lifts away from the paper on the flap's side (sign)
    probe = _rot(C3, D3, 0.1) @ np.r_[cc, 0.0, 1.0]
    osgn = 1.0 if probe[2] * sign > 0 else -1.0

    # flap: turns about the fold line, starting toward `sign`
    A3, U3 = np.r_[a, 0.0], np.r_[u, 0.0]
    mc = np.concatenate([Fb[i] for i in mov]).mean(axis=0)
    probe = _rot(A3, U3, 0.1) @ np.r_[mc, 0.0, 1.0]
    fsgn = 1.0 if probe[2] * sign > 0 else -1.0
    Vm = np.concatenate([np.c_[Fb[i], np.zeros(len(Fb[i])), np.ones(len(Fb[i]))] for i in mov])

    us = np.linspace(0.0, 1.0, S + 1)
    A = math.radians(open_deg)
    alpha = A * np.array([_smooth(min(x, 1 - x) / RAMP) for x in us])
    theta = PI * np.array([_smooth((x - LAG) / (1 - 2 * LAG)) for x in us])
    B = [_E3(Tb[i]) for i in range(n)]

    lams = []
    for k in range(S + 1):
        X = (_rot(A3, U3, fsgn * theta[k]) @ Vm.T).T
        z = X[:, 2] * sign
        dist = np.maximum((X[:, :2] - c0) @ nc, 0.0)
        cap = MARGIN * math.tan(alpha[k]) * dist
        lam = 1.0
        for zi, ci in zip(z, cap):
            if zi > 1e-9:
                lam = min(lam, ci / zi)
        lams.append(max(0.0, lam))

    fkeys = []
    for i in range(n):
        hb = _det(Tb[i]) * pieces[i]["bi"]
        ha = _det(Ta[i]) * pieces[i]["ai"]
        ks = []
        for k, x in enumerate(us):
            if pieces[i]["moving"]:
                Sq = np.diag([1.0, 1.0, lams[k], 1.0])
                M = Sq @ _rot(A3, U3, fsgn * theta[k]) @ B[i]
                hw = pieces[i]["bi"] + (pieces[i]["ai"] - pieces[i]["bi"]) * x   # world height (layers)
                ks.append([round(float(v), 7) for v in [*M[:3, :3].ravel(), *M[:3, 3], 0.0, 0.0, hw]])
            else:
                P = _rot(C3, D3, osgn * alpha[k]) if i in cov else np.eye(4)
                ks.append(_key(P, B[i], hb + (ha - hb) * x))
        fkeys.append(ks)
    for i in range(n):
        fkeys[i][0] = _key(np.eye(4), B[i], _det(Tb[i]) * pieces[i]["bi"])
        fkeys[i][-1] = _key(np.eye(4), _E3(Ta[i]), _det(Ta[i]) * pieces[i]["ai"])
    return {"fkeys": fkeys, "cover": sorted(cov), "open_deg": open_deg,
            "min_squash": round(float(min(lams[1:-1])), 4)}
