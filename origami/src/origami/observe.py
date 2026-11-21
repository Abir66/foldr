"""What the AI agent sees each turn.

    obs = observe(state, history, goal, colored_side="down")
    obs.text        compact text state (named points, outline, flaps, layers, colours, history, errors)
    obs.cp_png      crease-pattern image (PNG bytes): currently folded creases M (red) / V (blue), point names
    obs.folded_png  folded paper, front + back side by side, with point names
    inspect(state, "layers_at tl")   exact details on demand (layers, under, creases_at, face, fold)

Point names are derived from the ORIGINAL (unfolded) sheet position, so a point keeps its name in
every step: corners `bl br tr tl`, edge midpoints `bottom_mid right_mid top_mid left_mid`, `center`,
and every other crease vertex `pXXX_YYY` = sheet point (0.XXX, 0.YYY) in thousandths of the sheet size.
In step JSON such a point can be written `{"orig": [0.XXX, 0.YYY]}` (see `name_to_orig`).

Positions in the text are folded-space coordinates of the FRONT view (x right, y up); the back view
is the paper turned over left-right (x mirrored about the paper's bounding-box centre).
"""

from __future__ import annotations

import dataclasses
import io
import json
import math
import re
from dataclasses import dataclass, field

import numpy as np

from . import geometry as g
from .fold_export import to_fold
from .landmarks import NAMED_POINTS, resolve_point
from .paper import PaperState
from .validate import folded_creases

_NAME_RE = re.compile(r"^p(\d{3,4})_(\d{3,4})$")
_PROBE = 1e-3  # probe distance (fraction of sheet size) for "what is around this point"


# ---- point names ----------------------------------------------------------------------

def point_name(orig, size: float = 1.0) -> str:
    """Stable name of a sheet point (original-sheet coords)."""
    x, y = float(orig[0]) / size, float(orig[1]) / size
    for name, (fx, fy) in NAMED_POINTS.items():
        if abs(x - fx) < 1e-6 and abs(y - fy) < 1e-6:
            return name
    return f"p{round(x * 1000):03d}_{round(y * 1000):03d}"


def name_to_orig(name: str, size: float = 1.0, state: PaperState | None = None) -> np.ndarray:
    """Sheet coords of a point name. With `state`, a `pXXX_YYY` name snaps to the exact vertex."""
    if name in NAMED_POINTS:
        return np.array(NAMED_POINTS[name], dtype=float) * size
    m = _NAME_RE.match(name)
    if not m:
        raise ValueError(f"unknown point name {name!r} (want a corner/mid/center name or pXXX_YYY)")
    approx = np.array([int(m.group(1)), int(m.group(2))], dtype=float) / 1000 * size
    if state is not None:
        for nm, p in sheet_points(state).items():
            if nm == name:
                return p
    return approx


def sheet_points(state: PaperState) -> dict[str, np.ndarray]:
    """All named sheet points of the current state: the 9 landmarks + every face vertex."""
    s = state.size
    pts = {n: np.array(v, dtype=float) * s for n, v in NAMED_POINTS.items()}
    for f in state.faces:
        for p in f.orig:
            nm = point_name(p, s)
            if nm not in pts:
                pts[nm] = np.array(p, dtype=float)
    return pts


# ---- vectorised geometry helpers ---------------------------------------------------------

def _inside(poly: np.ndarray, pts: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Mask of points strictly inside a convex polygon (by more than tol)."""
    poly = g.ccw(poly)
    m = np.ones(len(pts), dtype=bool)
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        d = b - a
        L = math.hypot(*d)
        if L < 1e-12:
            continue
        m &= (d[0] * (pts[:, 1] - a[1]) - d[1] * (pts[:, 0] - a[0])) / L > tol
    return m


class _Geo:
    """Folded polygons + coverage queries for one state (faces are in draw order, 0 = bottom)."""

    def __init__(self, state: PaperState, colored_side: str = "up"):
        self.state = state
        self.size = state.size
        self.folded = [f.folded for f in state.faces]
        self.colored_up = colored_side != "down"

    def cover(self, pts: np.ndarray) -> np.ndarray:
        """Boolean matrix [face, point]: face strictly covers point."""
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        out = np.zeros((len(self.folded), len(pts)), dtype=bool)
        for i, poly in enumerate(self.folded):
            lo, hi = poly.min(0), poly.max(0)
            near = (pts[:, 0] > lo[0]) & (pts[:, 0] < hi[0]) & (pts[:, 1] > lo[1]) & (pts[:, 1] < hi[1])
            if near.any():
                out[i, near] = _inside(poly, pts[near])
        return out

    def colour(self, face: int, from_back: bool = False) -> str:
        """'c' (coloured) or 'w' (white): the side of `face` seen from the front (or back) view."""
        shows_front_of_sheet = self.state.faces[face].front_up != from_back
        return "c" if shows_front_of_sheet == self.colored_up else "w"


# ---- per-point info ----------------------------------------------------------------------

@dataclass
class PointInfo:
    name: str
    orig: tuple[float, float]
    pos: tuple[float, float]       # folded (front view)
    depth: int                     # max number of layers right around the point
    front: bool                    # visible from the front
    back: bool                     # visible from the back
    colours_front: str             # colours on top around the point seen from the front ("c", "w", "cw")
    colours_back: str


def _point_infos(geo: _Geo, pts: dict[str, np.ndarray]) -> dict[str, PointInfo]:
    st, s = geo.state, geo.size
    probes, owner = [], []  # one probe just inside each face around each point
    for nm, p in pts.items():
        for i, f in enumerate(st.faces):
            if not g.contains(f.orig, p, 1e-7):
                continue
            c = f.orig.mean(axis=0)
            d = c - p
            L = float(np.linalg.norm(d))
            r = min(_PROBE * s, 0.3 * L)
            q = p + d / L * r if L > 1e-12 else p
            probes.append(g.apply(f.T, q))
            owner.append((nm, i))
    cov = geo.cover(np.array(probes)) if probes else np.zeros((0, 0), bool)
    acc: dict[str, dict] = {}
    for k, (nm, i) in enumerate(owner):
        col = np.nonzero(cov[:, k])[0]
        if len(col) == 0:
            col = np.array([i])
        top, bot = int(col.max()), int(col.min())
        a = acc.setdefault(nm, {"depth": 0, "front": False, "back": False, "cf": set(), "cb": set(), "pos": None})
        a["depth"] = max(a["depth"], len(col))
        a["front"] |= top <= i
        a["back"] |= bot >= i
        a["cf"].add(geo.colour(top))
        a["cb"].add(geo.colour(bot, from_back=True))
        if a["pos"] is None:
            a["pos"] = g.apply(st.faces[i].T, pts[nm])
    out = {}
    for nm, a in acc.items():
        p = pts[nm]
        out[nm] = PointInfo(nm, (float(p[0]), float(p[1])), (float(a["pos"][0]), float(a["pos"][1])),
                            a["depth"], a["front"], a["back"],
                            "".join(sorted(a["cf"])), "".join(sorted(a["cb"])))
    return out


# ---- outline (union boundary of the folded faces) ---------------------------------------

def outline(state: PaperState) -> list[np.ndarray]:
    """Boundary loops of the folded paper's silhouette (front view). Outer loops CCW, holes CW;
    largest first. Collinear vertices are merged."""
    geo = _Geo(state)
    s = state.size
    key = lambda p: (round(float(p[0]) / s, 6), round(float(p[1]) / s, 6))
    segs: dict = {}
    for poly in geo.folded:
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]
            if np.linalg.norm(b - a) < 1e-9 * s:
                continue
            k = tuple(sorted((key(a), key(b))))
            segs.setdefault(k, (a, b))
    segs = list(segs.values())
    if not segs:
        return []
    A = np.array([a for a, _ in segs])
    B = np.array([b for _, b in segs])
    D = B - A
    verts = np.unique(np.round(np.vstack([A, B]), 9), axis=0)
    pieces = []
    for k in range(len(segs)):
        a, d = A[k], D[k]
        L2 = float(d @ d)
        ts = [0.0, 1.0]
        # crossings with other segments
        den = D[:, 0] * d[1] - D[:, 1] * d[0]  # cross(D_j, d)
        w = A - a
        ok = np.abs(den) > 1e-14
        t = np.where(ok, (w[:, 0] * D[:, 1] - w[:, 1] * D[:, 0]) / np.where(ok, -den, 1), -1)
        u = np.where(ok, (w[:, 0] * d[1] - w[:, 1] * d[0]) / np.where(ok, -den, 1), -1)
        sel = ok & (t > 1e-9) & (t < 1 - 1e-9) & (u > -1e-9) & (u < 1 + 1e-9)
        ts += list(t[sel])
        # vertices lying on the segment (T-junctions, collinear overlaps)
        rel = verts - a
        dist = (rel[:, 0] * d[1] - rel[:, 1] * d[0]) / math.sqrt(L2)
        tv = (rel @ d) / L2
        on = (np.abs(dist) < 1e-9 * s) & (tv > 1e-9) & (tv < 1 - 1e-9)
        ts += list(tv[on])
        # snap split points to the key grid so pieces chain up exactly
        cut: dict = {}
        for t in sorted(ts):
            p = a + t * d
            cut.setdefault(key(p), p)
        cps = list(cut.values())
        for p, q in zip(cps, cps[1:]):
            pieces.append((p, q))
    uniq = {}
    for p, q in pieces:
        uniq.setdefault(tuple(sorted((key(p), key(q)))), (p, q))
    pieces = list(uniq.values())
    eps = 1e-6 * s
    probes = []
    for p, q in pieces:
        m = (p + q) / 2
        d = q - p
        n = np.array([-d[1], d[0]]) / float(np.linalg.norm(d))
        probes += [m + eps * n, m - eps * n]
    cov = geo.cover(np.array(probes)).any(axis=0)
    nxt: dict = {}
    for k, (p, q) in enumerate(pieces):
        left, right = cov[2 * k], cov[2 * k + 1]
        if left == right:
            continue
        a, b = (p, q) if left else (q, p)  # paper on the left -> CCW outer boundary
        nxt.setdefault(key(a), []).append((key(b), a))
    loops = []
    while nxt:
        start = next(iter(nxt))
        cur, loop = start, []
        while cur in nxt:
            kb, a = nxt[cur].pop()
            if not nxt[cur]:
                del nxt[cur]
            loop.append(a)
            cur = kb
            if cur == start:
                break
        if len(loop) >= 3:
            loops.append(_merge_collinear(np.array(loop)))
    loops = [lp for lp in loops if len(lp) >= 3]
    loops.sort(key=lambda lp: -abs(g.signed_area(lp)))
    return loops


def _merge_collinear(loop: np.ndarray) -> np.ndarray:
    # drop coincident consecutive vertices (zero-length edges) so directions are defined
    keep = [i for i in range(len(loop)) if np.linalg.norm(loop[i] - loop[i - 1]) > 10 * g.EPS]
    if len(keep) < len(loop):
        loop = loop[keep]
    changed = True
    while changed and len(loop) > 3:
        changed = False
        n = len(loop)
        for i in range(n):
            p, v, q = loop[i - 1], loop[i], loop[(i + 1) % n]
            if abs(g.cross(g.unit(v - p), g.unit(q - v))) < 1e-7 and np.dot(v - p, q - v) > 0:
                loop = np.delete(loop, i, axis=0)
                changed = True
                break
    return loop


def _angles(loop: np.ndarray) -> list[float]:
    out = []
    n = len(loop)
    for i in range(n):
        p, v, q = loop[i - 1], loop[i], loop[(i + 1) % n]
        a, b = q - v, p - v
        th = math.degrees(math.atan2(g.cross(a, b), float(a @ b))) % 360
        out.append(th)
    return out


# ---- colours per side --------------------------------------------------------------------

def side_colours(geo: _Geo, n: int = 70) -> dict[str, float]:
    """Fraction of the silhouette showing the coloured side, front view and back view."""
    lo, hi = geo.state.bbox()
    xs = np.linspace(lo[0], hi[0], n + 2)[1:-1]
    ys = np.linspace(lo[1], hi[1], n + 2)[1:-1]
    P = np.array([(x, y) for y in ys for x in xs])
    cov = geo.cover(P)
    hit = cov.any(axis=0)
    if not hit.any():
        return {"front": 0.0, "back": 0.0}
    idx = np.arange(len(geo.folded))[:, None]
    top = np.where(cov, idx, -1).max(axis=0)[hit]
    bot = np.where(cov, idx, 10**9).min(axis=0)[hit]
    front = np.mean([geo.colour(int(i)) == "c" for i in top])
    back = np.mean([geo.colour(int(i), from_back=True) == "c" for i in bot])
    return {"front": float(front), "back": float(back)}


# ---- history one-liners ------------------------------------------------------------------

def _fmt(v) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)):
        return f"{v:g}"
    if isinstance(v, (list, tuple)):
        return "(" + ",".join(_fmt(x) for x in v) + ")"
    if isinstance(v, dict):
        items = list(v.items())
        if items and isinstance(items[0][1], (list, tuple, dict, str)) and items[0][0] not in ("top", "bottom"):
            k0, v0 = items[0]
            inner = _fmt(v0)
            inner = inner[1:-1] if inner.startswith("(") else inner
            rest = "".join(f",{k}={_fmt(x)}" for k, x in items[1:])
            return f"{k0}({inner}{rest})"
        return "{" + ",".join(f"{k}={_fmt(x)}" for k, x in items) + "}"
    return str(v)


def step_line(step) -> str:
    """One-line rendering of a step dict: `valley onto(tr,bl) move=bl`."""
    if isinstance(step, str):
        return step
    parts = [str(step.get("op", "?"))]
    for k, v in step.items():
        if k in ("op", "description"):
            continue
        if k == "line":
            parts.append(_fmt(v))
        elif k == "degrees":
            parts.append(f"{v:g}")
        else:
            parts.append(f"{k}={_fmt(v)}")
    return " ".join(parts)


# ---- the observation ---------------------------------------------------------------------

@dataclass
class Observation:
    text: str
    points: dict[str, PointInfo]
    outline: list[np.ndarray]
    cp_png: bytes | None = None
    folded_png: bytes | None = None
    extra: dict = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        """Rough token estimate of the text part (chars / 4)."""
        return len(self.text) // 4

    def save(self, out_dir, prefix: str = "obs") -> list:
        from pathlib import Path

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = [out / f"{prefix}_summary.txt"]
        paths[0].write_text(self.text + "\n")
        for nm, data in (("cp", self.cp_png), ("folded", self.folded_png)):
            if data:
                p = out / f"{prefix}_{nm}.png"
                p.write_bytes(data)
                paths.append(p)
        return paths


def _r(x: float) -> str:
    s = f"{x:.2f}"
    return "0.00" if s == "-0.00" else s


def _xy(p) -> str:
    return f"({_r(p[0])},{_r(p[1])})"


def _cluster(infos: dict[str, PointInfo], size: float) -> list[tuple[tuple, list[PointInfo]]]:
    groups: dict[tuple, list[PointInfo]] = {}
    for pi in infos.values():
        k = (round(pi.pos[0] / size, 4), round(pi.pos[1] / size, 4))
        groups.setdefault(k, []).append(pi)
    return sorted(groups.items(), key=lambda kv: (-kv[0][1], kv[0][0]))  # top to bottom, left to right


def _goal_name(goal) -> str:
    if goal is None:
        return "-"
    if isinstance(goal, dict):
        return str(goal.get("title") or goal.get("name") or goal)
    return str(goal)


_ORDER = {n: i for i, n in enumerate(NAMED_POINTS)}


def _sort_names(names):
    return sorted(names, key=lambda n: (_ORDER.get(n, 99), len(n), n))


def _names(pis: list[PointInfo], which: str | None = None, cap: int = 4) -> str:
    ns = _sort_names([p.name for p in pis if which is None or getattr(p, which)])
    if len(ns) > cap:
        return ",".join(ns[:cap]) + f",+{len(ns) - cap}"
    return ",".join(ns)


def observe(state: PaperState, history=None, goal=None, *, colored_side: str = "up",
            last_errors=None, step: int | None = None, budget: int | None = None,
            images: bool = True, labeler=None) -> Observation:
    """Build the agent's view of `state`.

    history: list of committed steps (step dicts or ready one-line strings).
    goal: name / dict with "title" or "name".
    colored_side: model's paper.colored_side ("up" = front of sheet coloured).
    last_errors: checker issues (validate.Issue or strings) of the last attempted move.
    """
    history = list(history or [])
    s = state.size
    geo = _Geo(state, colored_side)
    pts = sheet_points(state)
    infos = _point_infos(geo, pts)
    if labeler is not None:  # agent layer: show neutral sticker labels instead of internal names
        labeler.register(infos)
        infos = {labeler.label(k): dataclasses.replace(v, name=labeler.label(k)) for k, v in infos.items()}
    loops = outline(state)
    cols = side_colours(geo)
    n_faces = len(state.faces)
    max_depth = max((pi.depth for pi in infos.values()), default=1)
    lo, hi = state.bbox()
    k = step if step is not None else len(history)

    L = []
    L.append(f"GOAL: {_goal_name(goal)} | step {k}" + (f"/{budget}" if budget else "") +
             f" | coloured side of sheet: {'front' if geo.colored_up else 'back'} (starts {'up' if geo.colored_up else 'down'})")
    if labeler is not None:
        L.append(labeler.names_line() + " Positions = front view (x right, y up).")
    else:
        L.append("Names: corners bl,br,tr,tl; mids bottom_mid,right_mid,top_mid,left_mid; center; "
                 "pXXX_YYY = sheet point (0.XXX,0.YYY) -> {\"orig\":[0.XXX,0.YYY]}. Positions = front view (x right, y up).")
    L.append(f"Paper: bbox {_xy(lo)}-{_xy(hi)} ({_r(hi[0] - lo[0])}x{_r(hi[1] - lo[1])}), "
             f"{n_faces} faces, max {max_depth} layers, {len(pts)} named points.")
    L.append(f"Colours: front view {round(100 * cols['front'])}% coloured; back view {round(100 * cols['back'])}% coloured.")

    # outline with names at the corners
    by_pos = _cluster(infos, s)
    pos_index = {kk: v for kk, v in by_pos}

    def at(p):
        return pos_index.get((round(p[0] / s, 4), round(p[1] / s, 4)), [])

    tips = []
    for li, lp in enumerate(loops[:3]):
        ang = _angles(lp)
        kind = "outline" if g.signed_area(lp) > 0 else "hole"
        parts = []
        for v, a in zip(lp, ang):
            here = at(v)
            nm = _names(here, None, cap=2)
            parts.append(f"{_xy(v)}{round(a)}°" + (f"[{nm}]" if nm else ""))
            if kind == "outline" and a < 75:
                tips.append((v, a, here))
        L.append(f"{kind.capitalize()} (CCW, corner angle°, [points]): " + " ".join(parts))
    if len(loops) > 3:
        L.append(f"(+{len(loops) - 3} more outline loops)")

    if tips:
        L.append("Flaps/tips (sharp outline corners):")
        for v, a, here in tips:
            depth = max((p.depth for p in here), default=0)
            best = _sort_names([p.name for p in here])
            label = f"flap({best[0]})" if best else "flap(?)"
            L.append(f"  {label} tip {_xy(v)} {round(a)}° layers={depth} points={_names(here, None, cap=6) or '-'}")

    L.append("Visible points (pos d=layers F/B=colour on top from front/back c=coloured w=white; "
             "names: f=seen from front, b=from back):")
    hidden = 0
    for kk, grp in by_pos:
        vis = [p for p in grp if p.front or p.back]
        hidden += len(grp) - len(vis)
        if not vis:
            continue
        d = max(p.depth for p in grp)
        cf = "".join(sorted(set("".join(p.colours_front for p in grp))))
        cb = "".join(sorted(set("".join(p.colours_back for p in grp))))
        fn = _names([p for p in vis if p.front], cap=5)
        bn = _names([p for p in vis if p.back and not p.front], cap=5)
        both = [p for p in vis if p.front and p.back]
        tag = f"f:{fn}" if fn else ""
        if bn:
            tag += (" " if tag else "") + f"b:{bn}"
        if both and len(both) == len(vis):
            tag = f"fb:{_names(both, cap=5)}"
        L.append(f"  {_xy(grp[0].pos)} d={d} F={cf} B={cb} {tag}")
    if hidden:
        L.append(f"  (+{hidden} hidden points inside the stack; use inspect)")

    if history:
        L.append("History:")
        for i, h in enumerate(history, 1):
            L.append(f"  {i}. {labeler.sub(step_line(h)) if labeler else step_line(h)}")
    if last_errors:
        L.append("Last result: REJECTED")
        for e in last_errors:
            msg = getattr(e, "message", None) or str(e)
            L.append(f"  - {getattr(e, 'kind', 'error')}: {labeler.sub(msg) if labeler else msg}")
    text = "\n".join(L)

    obs = Observation(text=text, points=infos, outline=loops,
                      extra={"colours": cols, "faces": n_faces, "max_depth": max_depth})
    if images:
        obs.cp_png = render_cp_png(state, infos)
        obs.folded_png = render_folded_png(state, infos, colored_side)
    return obs


# ---- images ------------------------------------------------------------------------------

def _label_groups(infos: dict[str, PointInfo], size: float, side: str):
    out = []
    for _, grp in _cluster(infos, size):
        vis = [p for p in grp if getattr(p, side)]
        if vis:
            out.append((np.array(grp[0].pos), _names(vis, cap=3)))
    return out


def render_folded_png(state: PaperState, infos: dict[str, PointInfo], colored_side: str = "up",
                      width_px: int = 768) -> bytes:
    import matplotlib.pyplot as plt

    from . import render
    from .engine import flip

    old = (render.FRONT_COLOR, render.BACK_COLOR, render.LAYER_OFFSET)
    render.set_colored_side(colored_side)
    render.LAYER_OFFSET = 0.0015  # small layer shift so labels sit on the drawn paper
    try:
        dpi = 96
        fig, axes = plt.subplots(1, 2, figsize=(width_px / dpi, width_px / dpi * 0.55), dpi=dpi)
        lo, hi = state.bbox()
        c = (lo + hi) / 2
        half = float(max(hi - lo)) / 2 * 1.15 + 0.03 * state.size
        back = state.copy()
        flip(back)  # mirror about the bbox centre (x -> 2c - x), layer order reversed
        for ax, st, side, title in ((axes[0], state, "front", "front"),
                                    (axes[1], back, "back", "back (turned over left-right)")):
            render.draw_state(ax, st)
            for pos, lab in _label_groups(infos, state.size, side):
                if side == "back":
                    pos = np.array([2 * c[0] - pos[0], pos[1]])
                ax.plot(*pos, "o", color="black", markersize=2.5, zorder=50)
                ax.annotate(lab, pos, xytext=(3, 3), textcoords="offset points", fontsize=6.5, zorder=51,
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
            ax.set_xlim(c[0] - half, c[0] + half)
            ax.set_ylim(c[1] - half, c[1] + half)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=6)
            ax.grid(alpha=0.15)
            ax.set_title(title, fontsize=9)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return buf.getvalue()
    finally:
        render.FRONT_COLOR, render.BACK_COLOR, render.LAYER_OFFSET = old


def render_cp_png(state: PaperState, infos: dict[str, PointInfo] | None = None, width_px: int = 768) -> bytes:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    from .render import MOUNTAIN_COLOR, VALLEY_COLOR

    s = state.size
    dpi = 96
    fig, ax = plt.subplots(figsize=(width_px / dpi, width_px / dpi), dpi=dpi)
    ax.add_patch(Polygon([[0, 0], [s, 0], [s, s], [0, s]], closed=True, facecolor="white",
                         edgecolor="black", linewidth=1.5))
    for c in state.creases:  # every crease ever made (faint)
        ax.plot([c.p[0], c.q[0]], [c.p[1], c.q[1]], color="#bbbbbb", linewidth=0.8, zorder=1)
    for p, q, asg in folded_creases(state):  # folded right now
        if asg not in ("M", "V"):
            continue
        color, style = (VALLEY_COLOR, (0, (6, 3))) if asg == "V" else (MOUNTAIN_COLOR, (0, (6, 2, 1, 2)))
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, linestyle=style, linewidth=1.6, zorder=2)
    pts = sheet_points(state)
    for nm, p in pts.items():
        ax.plot(*p, "o", color="black", markersize=2, zorder=3)
        ax.annotate(nm, p, xytext=(2, 2), textcoords="offset points", fontsize=5.5, zorder=4)
    ax.set_xlim(-0.06 * s, 1.06 * s)
    ax.set_ylim(-0.06 * s, 1.06 * s)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Crease pattern now (blue dashed = valley, red dash-dot = mountain, grey = flat/unfolded)",
                 fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


# ---- inspect tool -----------------------------------------------------------------------

def _resolve(state: PaperState, spec):
    """-> (orig or None, folded point)."""
    if isinstance(spec, str):
        o = name_to_orig(spec, state.size, state)
        return o, state.to_folded(o)
    if isinstance(spec, dict) and set(spec) == {"orig"}:
        o = np.asarray(spec["orig"], dtype=float)
        return o, state.to_folded(o)
    return None, resolve_point(state, spec)


def _parse(query):
    if isinstance(query, dict):
        if len(query) != 1:
            raise ValueError("inspect query dict must have exactly one key")
        return next(iter(query.items()))
    parts = str(query).strip().split(None, 1)
    op = parts[0]
    arg = parts[1].strip() if len(parts) > 1 else None
    if arg is not None:
        try:
            arg = json.loads(arg)
        except json.JSONDecodeError:
            pass
    return op, arg


def inspect(state: PaperState, query, colored_side: str = "up"):
    """Exact details on demand. `query` is a dict {op: arg} or a string "op arg". Ops:

    layers_at P   faces stacked at point P (name, {"orig":..}, [x, y] folded, or any point spec),
                  top to bottom, with colour up; `depth` = layers right around P
    under X       faces below face X (int id) or below the top face holding point X, nearest first
    creases_at P  creases through sheet point P: assignment when made + folded state now
    face F        one face: sheet polygon (point names), folded polygon, colour, neighbours above/below
    fold          raw FOLD export (escape hatch; large)
    Face ids are indices in the current state (bottom = 0) and change after every move.
    """
    op, arg = _parse(query)
    geo = _Geo(state, colored_side)
    s = state.size
    if op == "fold":
        return to_fold(state)
    if op == "layers_at":
        o, X = _resolve(state, arg)
        return _layers_at(geo, X, o)
    if op == "under":
        if isinstance(arg, int):
            fid = arg
        else:
            o, X = _resolve(state, arg)
            cand = ([i for i, f in enumerate(state.faces) if g.contains(f.orig, o, 1e-7)] if o is not None
                    else [i for i, f in enumerate(geo.folded) if g.contains(f, X, 1e-7)])
            if not cand:
                raise ValueError(f"no paper at {arg!r}")
            fid = max(cand)
        below = sorted((j for i, j in state.above if i == fid), reverse=True)
        return {"face": fid, "colour_up": geo.colour(fid),
                "under": [{"face": j, "colour_up": geo.colour(j),
                           "overlap": round(g.overlap_area(geo.folded[fid], geo.folded[j]), 4),
                           "sheet_points": _poly_names(state.faces[j].orig, s)} for j in below]}
    if op == "creases_at":
        o, _ = _resolve(state, arg)
        if o is None:
            raise ValueError("creases_at needs a sheet point (name or {\"orig\": [x, y]})")
        return {"point": point_name(o, s), "creases": _creases_at(state, o)}
    if op == "face":
        fid = int(arg)
        f = state.faces[fid]
        return {"face": fid, "colour_up": geo.colour(fid), "front_up": f.front_up,
                "sheet_points": _poly_names(f.orig, s),
                "folded": [[round(float(x), 3), round(float(y), 3)] for x, y in f.folded],
                "above_it": sorted(i for i, j in state.above if j == fid),
                "below_it": sorted(j for i, j in state.above if i == fid)}
    raise ValueError(f"unknown inspect op {op!r}; known: layers_at, under, creases_at, face, fold")


def _poly_names(poly, size) -> list[str]:
    return [point_name(p, size) for p in poly]


def _layers_at(geo: _Geo, X, o=None) -> dict:
    st, s = geo.state, geo.size
    X = np.asarray(X, dtype=float)
    touching = [i for i, poly in enumerate(geo.folded) if g.contains(poly, X, 1e-7)]
    r = _PROBE * s
    dirs = [np.array([math.cos(t), math.sin(t)]) for t in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
    probes = [X] + [X + r * d for d in dirs]
    for i in touching:  # one probe into each face around X (catches thin wedges at sharp tips)
        c = geo.folded[i].mean(axis=0)
        L = float(np.linalg.norm(c - X))
        if L > 1e-12:
            probes.append(X + (c - X) / L * min(r, 0.3 * L))
    cov = geo.cover(np.array(probes))
    depth = int(cov.sum(axis=0).max())
    faces = []
    for i in sorted(touching, reverse=True):
        f = st.faces[i]
        here = g.apply(np.linalg.inv(f.T), X)
        faces.append({"face": i, "colour_up": geo.colour(i), "front_up": f.front_up,
                      "sheet_point": point_name(here, s), "interior": bool(cov[i, 0])})
    return {"at": [round(float(X[0]), 4), round(float(X[1]), 4)],
            "point": point_name(o, s) if o is not None else None,
            "depth": depth, "faces_top_to_bottom": faces}


def _creases_at(state: PaperState, o) -> list[dict]:
    s = state.size
    fc = folded_creases(state)

    def now_at(pt):
        for p, q, asg in fc:
            a, b = np.asarray(p, float), np.asarray(q, float)
            d = b - a
            L = float(np.linalg.norm(d))
            if L > 0 and abs(g.cross(d / L, pt - a)) < 1e-6 and -1e-9 <= (pt - a) @ d / L**2 <= 1 + 1e-9:
                return asg
        return "flat"

    out = []
    for c in state.creases:
        a, b = np.asarray(c.p, float), np.asarray(c.q, float)
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 1e-12 or abs(g.cross(d / L, o - a)) > 1e-6:
            continue
        t = float((o - a) @ d) / L**2
        if t < -1e-6 or t > 1 + 1e-6:
            continue
        u = d / L
        nows = set()
        for sgn in (1, -1):
            pt = o + sgn * 2e-3 * s * u
            tt = float((pt - a) @ d) / L**2
            if 0 <= tt <= 1:
                nows.add(now_at(pt))
        out.append({"from": point_name(a, s), "to": point_name(b, s), "made_as": c.assignment,
                    "now": "/".join(sorted(nows)) or "flat"})
    return out


def _main(argv=None) -> int:
    """`uv run python -m origami.observe models/crane.json --step 9 --out out/observe_examples/`"""
    import argparse
    from pathlib import Path

    from .cli import run_model

    ap = argparse.ArgumentParser(description="Write the agent observation (summary + PNGs) for one step.")
    ap.add_argument("model")
    ap.add_argument("--step", type=int, default=None, help="step number (default: last)")
    ap.add_argument("--out", default="out/observe_examples")
    args = ap.parse_args(argv)
    model = json.loads(Path(args.model).read_text())
    states = run_model(model)
    k = len(states) - 1 if args.step is None else args.step
    obs = observe(states[k], model["steps"][:k], model.get("title") or model.get("name"),
                  colored_side=model.get("paper", {}).get("colored_side", "up"),
                  step=k, budget=len(model["steps"]))
    for p in obs.save(args.out, f"{model.get('name', 'model')}_step{k:02d}"):
        print(p)
    print(f"~{obs.tokens} tokens (chars/4)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
