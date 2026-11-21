"""Export a PaperState to the FOLD format (https://github.com/edemaine/fold).

Key frame  = CURRENT crease pattern (unfolded sheet): B boundary, M/V creases folded right now,
             F flat face edges; faces_vertices = our faces. Crease history: `origami:creases_history`.
file_frames[0] = folded form: faces in folded coords + faceOrders.
"""

from __future__ import annotations

import json

import numpy as np

from . import geometry as g
from .paper import PaperState
from .validate import folded_creases

ROUND = 9


class _VertexIndex:
    def __init__(self):
        self.coords: list[list[float]] = []
        self._ids: dict[tuple, int] = {}

    def get(self, p) -> int:
        key = (round(float(p[0]), ROUND - 3), round(float(p[1]), ROUND - 3))
        if key not in self._ids:
            self._ids[key] = len(self.coords)
            self.coords.append([round(float(p[0]), ROUND), round(float(p[1]), ROUND)])
        return self._ids[key]


def _face_loops(state: PaperState, vi: _VertexIndex) -> list[list[int]]:
    """Face vertex loops (CCW in sheet coords), with T-junction vertices inserted on edges."""
    for f in state.faces:
        for p in f.orig:
            vi.get(p)
    all_orig = np.array(vi.coords)
    loops = []
    for f in state.faces:
        loop: list[int] = []
        n = len(f.orig)
        for i in range(n):
            a, b = f.orig[i], f.orig[(i + 1) % n]
            d = b - a
            on = [
                (float((x - a) @ d / (d @ d)), k)
                for k, x in enumerate(all_orig)
                if abs(g.cross(d, x - a)) < 1e-7 and 1e-7 < (x - a) @ d / (d @ d) < 1 - 1e-7
            ]
            loop.append(vi.get(a))
            loop += [k for _, k in sorted(on)]
        if not g.signed_area(f.orig) > 0:
            loop.reverse()
        loops.append(loop)
    return loops


def _current_edges(state: PaperState, vi: _VertexIndex, loops) -> tuple[list, list]:
    """Edges of the CURRENT crease pattern: B = sheet boundary, M/V = creases folded right now
    (validate.folded_creases), F = every other face edge (flat)."""
    fc = folded_creases(state)
    pts = np.array(vi.coords)

    def folded_asg(u: int, v: int) -> str | None:
        m = (pts[u] + pts[v]) / 2
        for p, q, asg in fc:
            a, b = np.asarray(p, float), np.asarray(q, float)
            d = b - a
            L = float(np.linalg.norm(d))
            if abs(g.cross(d / L, m - a)) < 1e-6 and -1e-9 <= (m - a) @ d / L**2 <= 1 + 1e-9:
                return asg
        return None

    count: dict[tuple[int, int], int] = {}
    for loop in loops:
        for i in range(len(loop)):
            e = tuple(sorted((loop[i], loop[(i + 1) % len(loop)])))
            count[e] = count.get(e, 0) + 1
    ev, ea = [], []
    for e, c in count.items():
        ev.append(list(e))
        if c == 1:
            ea.append("B")
        else:
            a = folded_asg(*e)
            ea.append(a if a in ("M", "V") else "F")
    return ev, ea


def _folded_form(state: PaperState, vi: _VertexIndex, loops) -> dict:
    folded_of: dict[int, list[float]] = {}
    for f, loop in zip(state.faces, loops):
        for k in loop:
            folded_of.setdefault(k, [round(float(x), ROUND) for x in g.apply(f.T, vi.coords[k])])
    return {
        "frame_classes": ["foldedForm"],
        "frame_inherit": False,
        "vertices_coords": [folded_of[k] for k in range(len(vi.coords))],
        "origami:vertices_orig_coords": vi.coords,
        "faces_vertices": loops,
        # faceOrders [f, g, s]: s=+1 means f lies on the side g's normal points to (FOLD spec),
        # i.e. g is front-up. Taken from the pairwise layer order (exact even without a global stack).
        "faceOrders": [list(o) for o in state.face_orders()],
        "origami:faces_layer_index": list(range(len(state.faces))),
    }


def to_fold(state: PaperState, title: str = "") -> dict:
    """Key frame = the CURRENT crease pattern (unfolded sheet; what is folded right now),
    file_frames[0] = folded form. The crease history is kept as `origami:creases_history`."""
    vi = _VertexIndex()
    loops = _face_loops(state, vi)
    ev, ea = _current_edges(state, vi, loops)
    return {
        "file_spec": 1.1,
        "file_creator": "origami (step simulator)",
        "file_title": title,
        "file_classes": ["singleModel"],
        "frame_classes": ["creasePattern"],
        "frame_attributes": ["2D"],
        "vertices_coords": vi.coords,
        "edges_vertices": ev,
        "edges_assignment": ea,
        "faces_vertices": loops,
        "origami:creases_history": [
            {"p": [float(x) for x in c.p], "q": [float(x) for x in c.q], "assignment": c.assignment}
            for c in state.creases
        ],
        "file_frames": [_folded_form(state, vi, loops)],
    }


def write_fold(path, state: PaperState, title: str = "") -> None:
    with open(path, "w") as fh:
        json.dump(to_fold(state, title), fh, indent=1)
