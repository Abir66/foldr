"""Export a model run as an animation "timeline" (JSON) for the 3D viewer.

Faces are not stable across steps (a fold splits them, an unfold merges them), so each
step is described on the *common refinement* of the before/after states: every piece is
(before face ∩ after face) in original-sheet coords, with its transform and layer in both
states. The motion of a step is one plane isometry M = T_after · T_before⁻¹ shared by all
moving pieces:
  - reflection (valley / mountain / unfold / flip): 3D rotation by 180° about the mirror line;
  - rotation (rotate op): in-plane rotation about its center.
Pieces with M = identity stay put. Nothing in the engine is changed; motion is derived by
diffing the two states.

The viewer animates every non-flip step with connected kinematics (kinematics.py): each piece also
carries `parent`, `hinge`, `theta` [before, after], `delta`, `h`, `win` (and solved `keys` where
creases meet at a vertex); the step carries `kin_order`, `kin_bends`, `kin_gap`, `path_len` and, for
solved steps, `retime` (even speed along the path). `motion` /
`motions` / `chain` stay for hints (fold lines) and tests.
"""

from __future__ import annotations

import json
import math

import numpy as np

from . import geometry as g
from .bends import bends as _bends
from .cli import run_model
from .engine import fold_geometry
from .kinematics import RETIME_SAMPLES as _RETIME_SAMPLES
from .kinematics import build as _kinematics
from .kinematics import path_profile as _path_profile
from .kinematics import retime_table as _retime_table
from .reverse_anim import spread_keys as _spread_keys
from .tuck_anim import tuck_keys as _tuck_keys
from .paper import PaperState

TOL = 1e-6
STACK_K = 6.0  # layer_heights: stack depth where the gap starts to shrink (gap ~ 1 / (1 + depth / K))


def layer_heights(state: PaperState) -> list[float]:
    """Compact stacking height per face (in layer units), bottom-up in draw order.

    The engine's layer index counts EVERY face below, even ones far away in the plane, so a deep
    model (crane: ~100 faces) would be drawn ~100 gaps thick and a narrow flap looked like a rod.
    Here a face only sits one gap above the highest face it actually overlaps: overlapping faces
    keep their order (no z-fighting / pass-through), while the stack is as thin as real paper.
    Deep stacks are squeezed further (log curve, STACK_K): the first few layers keep a full gap, so
    simple models look the same, and a 30+ layer crane body stays a thin slab instead of a block.
    Heights are symmetric under turning the stack over (flip): z = (f(below) + f(H) - f(above)) / 2
    with below / above = longest chain of overlapping faces under / over a face, H = the deepest
    chain, f = the log squeeze. Overlapping faces stay >= one (squeezed) gap apart."""
    polys = [g.ccw(g.apply(f.T, f.orig)) for f in state.faces]
    n = len(polys)
    eps = 1e-7 * state.size * state.size
    lo = [p.min(axis=0) for p in polys]
    hi = [p.max(axis=0) for p in polys]

    def overlap(i: int, j: int) -> bool:
        if np.any(lo[i] >= hi[j] - 1e-9) or np.any(lo[j] >= hi[i] - 1e-9):
            return False
        inter = g.clip_convex(polys[i], polys[j])
        return len(inter) >= 3 and g.area(inter) > eps

    over = [[j for j in range(i) if overlap(i, j)] for i in range(n)]  # overlapping faces below i
    below = [0] * n
    for i in range(n):
        below[i] = max((below[j] + 1 for j in over[i]), default=0)
    above = [0] * n
    for i in reversed(range(n)):
        for j in over[i]:
            above[j] = max(above[j], above[i] + 1)
    H = max(below, default=0)
    f = lambda x: STACK_K * math.log1p(x / STACK_K)  # noqa: E731
    return [(f(below[i]) + f(H) - f(above[i])) / 2 for i in range(n)]


def _T23(T: np.ndarray) -> list[list[float]]:
    return [[float(v) for v in T[0]], [float(v) for v in T[1]]]


def _poly(p: np.ndarray) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in g.ccw(p)]


def _face_lines(state: PaperState, orig: np.ndarray) -> list[list[list[float]]]:
    """Crease segments (orig coords) that lie inside this face."""
    out = []
    for c in state.creases:
        seg = g.clip_segment(np.array(c.p, float), np.array(c.q, float), orig)
        if seg is not None and np.linalg.norm(seg[1] - seg[0]) > 1e-6:
            out.append([[float(v) for v in seg[0]], [float(v) for v in seg[1]]])
    return out


def _world_for(states: list[PaperState], steps: list[dict]) -> list[dict]:
    """Per-state map from engine coordinates to a fixed *world* frame.

    A `flip` mirrors the engine's coordinates and reverses its layers, but the paper itself
    does not move: the viewer turns the camera instead. So each state gets
      world_xy = M @ (engine_xy, 1),  world_z = s * engine_z + zoff * DZ
    where a flip multiplies M by the engine's mirror and negates s (zoff keeps the layer
    stack continuous). Everything else (folds, rotate, unfold) keeps the world map. M and s
    together are a proper rotation, so front/back faces stay correct.
    Batch form; TimelineBuilder.append applies the same rule one step at a time (tests compare them).
    """
    M, s, zoff = np.eye(3), 1, 0.0
    out = []
    for k, state in enumerate(states):
        if k > 0 and steps[k - 1].get("op") == "flip":
            before = states[k - 1]
            lo, hi = before.bbox()
            M = M @ g.mirror((lo + hi) / 2, steps[k - 1].get("axis", "vertical"))
            zoff += s * max(layer_heights(before))
            s = -s
        out.append({"M": _T23(M), "s": s, "zoff": float(zoff)})
    return out


def state_json(state: PaperState) -> dict:
    hts = layer_heights(state)
    return {
        "bends": _bends(state),
        "faces": [
            {
                "id": i,
                "orig": _poly(f.orig),
                "T": _T23(f.T),
                "layer": hts[i],  # compact stacking height (layer units), see layer_heights
                "front_up": f.front_up,
                "lines": _face_lines(state, f.orig),
            }
            for i, f in enumerate(state.faces)
        ],
        "bbox": [[float(v) for v in x] for x in state.bbox()],
    }


def _pieces(before: PaperState, after: PaperState) -> list[dict]:
    """Common refinement of before/after faces (in orig coords)."""
    out = []
    for ib, fb in enumerate(before.faces):
        for ia, fa in enumerate(after.faces):
            inter = g.clip_convex(fb.orig, fa.orig)
            if len(inter) < 3 or g.area(inter) < 1e-9:
                continue
            M = fa.T @ np.linalg.inv(fb.T)
            out.append({"orig": inter, "bi": ib, "ai": ia, "Tb": fb.T, "Ta": fa.T, "M": M})
    return out


def _is_identity(M: np.ndarray) -> bool:
    return bool(np.allclose(M, np.eye(3), atol=TOL))


def _motion(M: np.ndarray) -> dict:
    """Describe a plane isometry as a reflection axis or a rotation center + angle."""
    L, t = M[:2, :2], M[:2, 2]
    if np.linalg.det(L) < 0:
        # L = 2uu^T - I  ->  L + I = 2uu^T; the midpoint of p and M p lies on the axis.
        S = L + np.eye(2)
        col = S[:, int(np.argmax(np.linalg.norm(S, axis=0)))]
        u = g.unit(col)
        a = t / 2
        return {"type": "reflect", "a": [float(a[0]), float(a[1])], "u": [float(u[0]), float(u[1])]}
    ang = math.atan2(L[1, 0], L[0, 0])
    if abs(ang) < 1e-9:
        return {"type": "translate", "t": [float(t[0]), float(t[1])]}
    c = np.linalg.solve(np.eye(2) - L, t)
    return {"type": "rotate", "center": [float(c[0]), float(c[1])], "degrees": math.degrees(ang)}


def _axis_segment(state: PaperState, a, u) -> list[list[float]]:
    """Fold line clipped to the paper's extent (folded space), slightly padded."""
    a, u = np.asarray(a, float), np.asarray(u, float)
    ts = []
    big = 10 * state.size
    for f in state.faces:
        seg = g.clip_segment(a - big * u, a + big * u, f.folded)
        if seg is not None:
            ts += [float(np.dot(p - a, u)) for p in seg]
    if not ts:
        lo, hi = state.bbox()
        ts = [float(np.dot(p - a, u)) for p in (lo, hi)]
    pad = 0.08 * state.size
    t0, t1 = min(ts) - pad, max(ts) + pad
    return [[float(v) for v in a + t0 * u], [float(v) for v in a + t1 * u]]


def step_json(k: int, step: dict, before: PaperState, after: PaperState) -> dict:
    op = step.get("op")
    raw = _pieces(before, after)
    moving = [p for p in raw if not _is_identity(p["M"])]

    motions = list(getattr(after, "motions", None) or [])
    shared = not moving or all(np.allclose(p["M"], moving[0]["M"], atol=1e-5) for p in moving)
    if len(motions) > 1 or (not shared and motions):
        return _multi_step_json(k, step, before, after, raw, motions)
    if not shared:
        raise ValueError(f"step {k + 1}: moving pieces don't share one motion")
    motion = _motion(moving[0]["M"]) if moving else None

    # Which way the flap swings in 3D: +1 = up toward the viewer, -1 = underneath.
    sign = 1
    if op == "mountain":
        sign = -1
    elif op == "unfold" and moving:
        mv = np.mean([p["bi"] for p in moving])
        still = [p["bi"] for p in raw if _is_identity(p["M"])]
        sign = 1 if not still or mv >= np.mean(still) else -1

    axis = None
    if motion and motion["type"] == "reflect":
        axis = _axis_segment(before, motion["a"], motion["u"])

    arrow = None
    if op in ("valley", "mountain"):
        a, u, move = fold_geometry(before, step)
        target = g.apply(g.reflection(a, u), move)
        arrow = {"from": [float(v) for v in move], "to": [float(v) for v in target]}

    out = {
        "index": k + 1,
        "op": op,
        "arrow": arrow,  # fold-diagram arrow: move point -> landing spot (folded space, BEFORE state)
        "flip_axis": step.get("axis", "vertical") if op == "flip" else None,
        "degrees": float(step["degrees"]) if op == "rotate" else None,
        "description": step.get("description", op),
        "motion": motion,
        "axis": axis,  # [[x,y],[x,y]] in folded space of the BEFORE state
        "sign": sign,
        "moving_before": sorted({p["bi"] for p in moving}),
        "moving_after": sorted({p["ai"] for p in moving}),
        "pieces": [
            {
                "orig": _poly(p["orig"]),
                "Tb": _T23(p["Tb"]),
                "Ta": _T23(p["Ta"]),
                "layer_before": p["bi"],
                "layer_after": p["ai"],
                "moving": not _is_identity(p["M"]),
                "lines": _face_lines(before, p["orig"]),  # creases already in the paper (drawn while moving)
            }
            for p in raw
        ],
    }
    if op != "flip":  # flip = camera move, no paper motion
        # tuck inside: scripted "open the pocket, slide the flap in" frames (tuck_anim.py)
        tuck = None
        if op in ("valley", "mountain") and motion and motion["type"] == "reflect" and motions \
                and motions[-1].get("tuck") == "inside":
            tuck = _tuck_keys([dict(r, moving=not _is_identity(r["M"])) for r in _compact_heights(raw, before, after)],
                [(p["bi"], p["ai"]) for p in raw], after, motion["a"], motion["u"], sign)
        _attach_kin(out, raw, before, after, nphases=1, motions=None, step_sign=sign, solve=tuck is None)
        if tuck is not None:
            _attach_tuck(out, tuck)
    return out


def _attach_tuck(out: dict, res: dict) -> None:
    """Tuck inside (tuck_anim.py): the cover opens a little, the flap slides in under it."""
    for pj, fk in zip(out["pieces"], res["fkeys"]):
        pj["fkeys"] = fk
        pj.pop("keys", None)
    out["tuck_anim"] = {"cover": res["cover"], "open_deg": res["open_deg"], "min_squash": res["min_squash"]}
    out["kin_gap"] = 0.0
    # joints between the opening cover and the rest open mid-move: drop their bends (they'd draw as curls)
    cov = set(res["cover"])
    out["kin_bends"] = [b for b in out["kin_bends"] if (b["f"] in cov) == (b["o"] in cov)]
    polys = [np.asarray(p["orig"], float) for p in out["pieces"]]
    prof = _path_profile(out["pieces"], out["kin_order"], polys, _RETIME_SAMPLES)
    out["retime"] = _retime_table(prof)
    out["path_len"] = round(float(prof.sum()), 4)


def _compact_heights(recs: list[dict], before: PaperState, after: PaperState) -> list[dict]:
    """Copy of recs with bi/ai = compact stacking heights (layer_heights) instead of stack indices."""
    hb, ha = layer_heights(before), layer_heights(after)
    return [dict(r, bi=hb[r["bi"]], ai=ha[r["ai"]]) for r in recs]


def _attach_kin(out: dict, recs: list[dict], before: PaperState, after: PaperState, *, nphases: int,
                motions, step_sign: int, drive: int | None = None, solve: bool = True) -> None:
    """Connected animation data (see kinematics.py): per piece parent / hinge / angles, plus bends.
    Heights are the compact ones (layer_heights), so pieces move between the drawn stack heights."""
    recs = _compact_heights(recs, before, after)
    for pj, r in zip(out["pieces"], recs):
        pj["z_before"], pj["z_after"] = r["bi"], r["ai"]
    kin = _kinematics(recs, nphases=nphases, motions=motions, step_sign=step_sign, drive=drive, solve=solve)
    for pj, kp in zip(out["pieces"], kin["pieces"]):
        pj.update(kp)
    out["kin_order"] = kin["order"]
    out["kin_bends"] = kin["bends"]
    out["kin_gap"] = round(float(kin["gap"]), 6)  # worst mid-motion loop gap / drift (paper units)
    if kin["retime"]:
        out["retime"] = kin["retime"]  # path parameter at even steps of motion: u = retime(ease(t))
    out["path_len"] = round(kin["path_len"], 4)  # distance the fastest-moving point travels (paper units)


# ---- multi-part steps (technique macros) -------------------------------------------

AREA_TOL = 1e-10


def _split_by_region(poly: np.ndarray, region: np.ndarray) -> list[tuple[np.ndarray, bool]]:
    """Split convex `poly` into convex parts that are each fully inside / outside convex `region`."""
    A = g.area(poly)
    inter = g.clip_convex(poly, region)
    ia = g.area(inter) if len(inter) >= 3 else 0.0
    if ia < AREA_TOL:
        return [(poly, False)]
    if ia > A - AREA_TOL:
        return [(poly, True)]
    out, rest = [], poly
    region = g.ccw(np.asarray(region, float))
    for i in range(len(region)):
        p, q = region[i], region[(i + 1) % len(region)]
        if np.linalg.norm(q - p) < 1e-12:
            continue
        inside, outside = g.split_convex(rest, p, q - p)  # ccw region: interior is left
        if outside is not None and len(outside) >= 3 and g.area(outside) > AREA_TOL:
            out.append((outside, False))
        if inside is None or len(inside) < 3 or g.area(inside) <= AREA_TOL:
            return out
        rest = inside
    out.append((rest, True))
    return out


def _clip_line_to(polys: list[np.ndarray], a, u, size: float) -> list[list[float]] | None:
    a, u = np.asarray(a, float), np.asarray(u, float)
    big = 10 * size
    ts = []
    for poly in polys:
        seg = g.clip_segment(a - big * u, a + big * u, poly)
        if seg is not None:
            ts += [float(np.dot(p - a, u)) for p in seg]
    if not ts:
        return None
    pad = 0.08 * size
    t0, t1 = min(ts) - pad, max(ts) + pad
    return [[float(v) for v in a + t0 * u], [float(v) for v in a + t1 * u]]


def _multi_step_json(k: int, step: dict, before: PaperState, after: PaperState, raw: list[dict], motions) -> dict:
    """A step made of several sub-motions (`state.motions`, see engine._move).

    Every piece of the before/after refinement is split further until each part is either fully
    inside or fully outside every sub-motion's moving region. Each part then gets a `chain`: the
    sub-motions it takes part in, in order, with the side of the axis it starts on (`dir`). The
    chain applied to `Tb` must reproduce `Ta` exactly (checked here).
    """
    size = before.size
    parts = [dict(orig=p["orig"], bi=p["bi"], ai=p["ai"], Tb=p["Tb"], Ta=p["Ta"], T=p["Tb"].copy(), chain=[])
             for p in raw]
    axes_polys: list[list[np.ndarray]] = []
    for mi, m in enumerate(motions):
        a, u = np.asarray(m["a"], float), g.unit(np.asarray(m["u"], float))
        regions = [np.asarray(r, float) for r in m["regions"]]
        nxt, moved_polys = [], []
        for pc in parts:
            pending, done = [pc["orig"]], []
            for R in regions:
                still = []
                for poly in pending:
                    for sub, inside in _split_by_region(poly, R):
                        (done if inside else still).append(sub)
                pending = still
            for poly, inside in [(q, False) for q in pending] + [(q, True) for q in done]:
                part = dict(pc, orig=poly, chain=list(pc["chain"]), T=pc["T"].copy())
                if inside:
                    folded = g.apply(part["T"], poly)
                    moved_polys.append(folded)
                    c = folded.mean(axis=0)
                    part["chain"].append({"m": mi, "dir": 1 if g.side(a, u, c) > 0 else -1})
                    part["T"] = g.reflection(a, u) @ part["T"]
                nxt.append(part)
        parts = nxt
        axes_polys.append(moved_polys)
    for pc in parts:
        if not np.allclose(pc["T"], pc["Ta"], atol=1e-6):
            raise ValueError(f"step {k + 1}: sub-motions don't reproduce the folded result")

    axes = []
    for mi, m in enumerate(motions):
        seg = _clip_line_to(axes_polys[mi], m["a"], m["u"], size)
        if seg is None:
            continue
        dup = any(ax["phase"] == m["phase"]  # same line in one phase (e.g. inside reverse): show once
                  and abs(g.side(np.array(ax["seg"][0]), np.array(ax["seg"][1]) - ax["seg"][0], m["a"])) < 1e-6
                  and abs(g.cross(np.array(ax["seg"][1]) - ax["seg"][0], m["u"])) < 1e-6 for ax in axes)
        if not dup:
            axes.append({"seg": seg, "sign": int(m["sign"]), "phase": int(m["phase"])})
    phases = sorted({int(m["phase"]) for m in motions})
    remap = {p: i for i, p in enumerate(phases)}  # phases numbered 0..P-1
    for ax in axes:
        ax["phase"] = remap[ax["phase"]]
    moving = [pc for pc in parts if pc["chain"]]
    first = axes[0] if axes else None
    out = {
        "index": k + 1,
        "op": step.get("op"),
        "arrow": None,
        "flip_axis": None,
        "degrees": None,
        "description": step.get("description", step.get("op")),
        "motion": None,  # several sub-motions: see `motions`
        "motions": [
            {"a": [float(v) for v in m["a"]], "u": [float(v) for v in g.unit(np.asarray(m["u"], float))],
             "sign": int(m["sign"]), "phase": remap[int(m["phase"])], "tuck": m.get("tuck")}
            for m in motions
        ],
        "phases": len(phases),
        "axes": axes,  # fold lines per sub-motion (folded space of the state when it starts)
        "axis": first["seg"] if first else None,
        "sign": first["sign"] if first else 1,
        "moving_before": sorted({pc["bi"] for pc in moving}),
        "moving_after": sorted({pc["ai"] for pc in moving}),
        "pieces": [
            {
                "orig": _poly(pc["orig"]),
                "Tb": _T23(pc["Tb"]),
                "Ta": _T23(pc["Ta"]),
                "layer_before": pc["bi"],
                "layer_after": pc["ai"],
                "moving": bool(pc["chain"]),
                "chain": pc["chain"],  # [{m: sub-motion index, dir: start side of its axis}]
                "lines": _face_lines(before, pc["orig"]),
            }
            for pc in parts
        ],
    }
    # petal: scripted motion driven by the lift (first phase-0 valley with no tuck); others: path solve
    drive = None
    if out["op"] == "petal":
        drive = next((i for i, m in enumerate(out["motions"])
                      if m["phase"] == 0 and m["sign"] == 1 and not m["tuck"]), None)
    # inside reverse: scripted "fake spread" frames instead of the rigid-panel path solve
    spread = (_spread_keys(_compact_heights(parts, before, after), out["motions"])
              if out["op"] == "inside_reverse" else None)
    _attach_kin(out, parts, before, after, nphases=len(phases), motions=out["motions"], step_sign=out["sign"],
                drive=drive, solve=spread is None)
    if spread is not None:
        _attach_spread(out, parts, spread)
    return out


def _attach_spread(out: dict, parts: list[dict], res: dict) -> None:
    """Inside reverse (reverse_anim.py): the flap halves open a little, the tip turns through the gap
    between them, they close. Paper joints at the reverse crease open slightly mid-move (`spread`)."""
    for pj, fk in zip(out["pieces"], res["fkeys"]):
        pj["fkeys"] = fk
        pj.pop("keys", None)
    out["spread"] = {"hinge_gap": round(res["hinge_gap"], 4), "hits": res["hits"]}
    out["kin_gap"] = 0.0   # no loop solve; the scripted gaps are reported in `spread`
    # joints that open mid-move (tip vs rest, tip half vs tip half) would draw as wide curls: drop them
    half = [tuple(sorted({c["m"] for c in pc["chain"]})) for pc in parts]
    out["kin_bends"] = [b for b in out["kin_bends"] if half[b["f"]] == half[b["o"]]]
    polys = [np.asarray(p["orig"], float) for p in out["pieces"]]
    prof = _path_profile(out["pieces"], out["kin_order"], polys, _RETIME_SAMPLES)
    out["retime"] = _retime_table(prof)
    out["path_len"] = round(float(prof.sum()), 4)


class TimelineBuilder:
    """Builds a viewer timeline one step at a time (live folding, agent runs) with undo.

        tb = TimelineBuilder(initial_state, name=..., title=..., colored_side=...)
        tb.append(step, before, after)   # -> the new step's JSON; tb.states[-1] = after's state JSON
        tb.pop()                          # drop the last step (undo)
        tb.to_dict()                      # same dict build_timeline returns for the same steps

    `before` must be the state the last append ended in (or the initial state). Each append only
    computes the new step and the new state, so appending is O(step), not O(timeline).
    The returned dict shares its lists with the builder (no copies); don't mutate it.
    """

    def __init__(self, initial: PaperState, *, name: str = "model", title: str | None = None,
                 colored_side: str = "up"):
        self.name = name
        self.title = name if title is None else title
        self.size = initial.size
        self.colored_side = colored_side
        self.original_faces = [{"id": i, "orig": _poly(f.orig)} for i, f in enumerate(initial.faces)]
        self._world = [(np.eye(3), 1, 0.0)]  # (M, s, zoff) per state, see _world_for
        self.states: list[dict] = [self._state_with_world(initial)]
        self.steps: list[dict] = []

    @classmethod
    def for_model(cls, model: dict, initial: PaperState) -> "TimelineBuilder":
        return cls(initial, name=model.get("name", "model"), title=model.get("title", model.get("name", "")),
                   colored_side=model.get("paper", {}).get("colored_side", "up"))

    def __len__(self) -> int:
        return len(self.steps)

    def _state_with_world(self, state: PaperState) -> dict:
        M, s, zoff = self._world[-1]
        sj = state_json(state)
        sj["world"] = {"M": _T23(M), "s": s, "zoff": float(zoff)}
        return sj

    def append(self, step: dict, before: PaperState, after: PaperState) -> dict:
        M, s, zoff = self._world[-1]
        if step.get("op") == "flip":  # same rule as _world_for
            lo, hi = before.bbox()
            M = M @ g.mirror((lo + hi) / 2, step.get("axis", "vertical"))
            zoff += s * max(layer_heights(before))
            s = -s
        k = len(self.steps)
        st = step_json(k, step, before, after)  # may raise: then nothing is changed
        self._world.append((M, s, zoff))
        try:
            sj = self._state_with_world(after)
        except Exception:
            self._world.pop()
            raise
        st["side_before"], st["side_after"] = self._world[-2][1], s
        self.states.append(sj)
        self.steps.append(st)
        return st

    def pop(self, n: int = 1) -> None:
        if n < 0 or n > len(self.steps):
            raise IndexError(f"can't pop {n} of {len(self.steps)} steps")
        for _ in range(n):
            self.steps.pop()
            self.states.pop()
            self._world.pop()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "title": self.title,
            "size": self.size,
            "colored_side": self.colored_side,
            "original_faces": self.original_faces,
            "states": self.states,
            "steps": self.steps,
        }


def build_timeline(model: dict) -> dict:
    states = run_model(model)
    tb = TimelineBuilder.for_model(model, states[0])
    for k, step in enumerate(model["steps"]):
        tb.append(step, states[k], states[k + 1])
    return tb.to_dict()


def round_floats(obj, ndigits: int = 5):
    """Copy of a JSON-like object with every float rounded (smaller timeline JSON). -0.0 becomes 0.0."""
    if isinstance(obj, float):
        return round(obj, ndigits) + 0.0
    if isinstance(obj, dict):
        return {k: round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_floats(v, ndigits) for v in obj]
    return obj


def timeline_to_json(timeline: dict, ndigits: int | None = None) -> str:
    """Compact JSON. ndigits: round floats first (design: 5 for shipped timelines); None = exact."""
    if ndigits is not None:
        timeline = round_floats(timeline, ndigits)
    return json.dumps(timeline, separators=(",", ":"))
