"""Fold-connected face pairs (shared by the 3D viewer export and the PNG renderer).
NOTE: duplicated from steps_export.py (importing it from render.py is circular); dedupe later."""

from __future__ import annotations

import numpy as np

from . import geometry as g
from .paper import PaperState


def _shared_edge(pa: np.ndarray, pb: np.ndarray):
    """Longest collinear overlap between an edge of polygon pa and an edge of pb (orig coords)."""
    best = None
    for i in range(len(pa)):
        a0, a1 = pa[i], pa[(i + 1) % len(pa)]
        la = np.linalg.norm(a1 - a0)
        if la < 1e-9:
            continue
        u = (a1 - a0) / la
        for j in range(len(pb)):
            b0, b1 = pb[j], pb[(j + 1) % len(pb)]
            nrm = np.array([-u[1], u[0]])
            if abs(np.dot(b0 - a0, nrm)) > 1e-6 or abs(np.dot(b1 - a0, nrm)) > 1e-6:
                continue
            t0, t1 = sorted([float(np.dot(b0 - a0, u)), float(np.dot(b1 - a0, u))])
            lo, hi = max(0.0, t0), min(la, t1)
            if hi - lo > 1e-6 and (best is None or hi - lo > best[1] - best[0]):
                best = (lo, hi, a0, u)
    if best is None:
        return None
    lo, hi, a0, u = best
    return a0 + lo * u, a0 + hi * u


def bends(state: PaperState) -> list[dict]:
    """Folded connections: faces adjacent in the ORIGINAL sheet across a crease whose images are
    now stacked on the same side of that edge (the paper wraps around it). Raw edges are not listed.
    Each: faces a < b (layer indices), seg = shared crease segment in orig coords."""
    out = []
    faces = state.faces
    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
            fi, fj = faces[i], faces[j]
            if np.allclose(fi.T, fj.T, atol=1e-6):
                continue  # still flat: no fold between them
            seg = _shared_edge(g.ccw(fi.orig), g.ccw(fj.orig))
            if seg is None:
                continue
            p, q = seg
            m = (p + q) / 2
            u = g.unit(q - p)
            nrm = np.array([-u[1], u[0]])
            ci, cj = (np.mean(f.orig, axis=0) for f in (fi, fj))
            # same side of the edge after folding <=> folded centroids on one side of the folded edge
            P, Q = g.apply(fi.T, p), g.apply(fi.T, q)
            P2, Q2 = g.apply(fj.T, p), g.apply(fj.T, q)
            if not (np.allclose(P, P2, atol=1e-6) and np.allclose(Q, Q2, atol=1e-6)):
                continue
            fu = g.unit(Q - P)
            fn = np.array([-fu[1], fu[0]])
            si = np.dot(g.apply(fi.T, ci) - P, fn)
            sj = np.dot(g.apply(fj.T, cj) - P, fn)
            if si * sj > 1e-9:
                out.append({"a": i, "b": j, "seg": [[float(v) for v in p], [float(v) for v in q]]})
    return out
