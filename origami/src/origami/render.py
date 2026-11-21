"""Matplotlib rendering of folded states and crease patterns."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon  # noqa: E402

from . import geometry as g  # noqa: E402
from .engine import FOLD_OPS, fold_geometry  # noqa: E402
from .landmarks import resolve_line_ref, resolve_point  # noqa: E402
from .paper import PaperState  # noqa: E402
from .bends import bends as _bends  # noqa: E402

FRONT_COLOR = "#e07a3f"  # colored side
BACK_COLOR = "#fbf6ec"  # white side
_COLORED, _WHITE = FRONT_COLOR, BACK_COLOR


def set_colored_side(side: str = "up") -> None:
    """Model option paper.colored_side: "up" (default) = front (+det) side colored; "down" = back colored.
    Only swaps the two side colors; geometry is unchanged (like starting the real sheet colored side down)."""
    global FRONT_COLOR, BACK_COLOR
    FRONT_COLOR, BACK_COLOR = (_WHITE, _COLORED) if side == "down" else (_COLORED, _WHITE)
EDGE_COLOR = "#3a2a1a"
CREASE_COLOR = "#5c3d24"
VALLEY_COLOR = "#1f5fbf"
MOUNTAIN_COLOR = "#c0392b"
LAYER_OFFSET = 0.011  # fraction of paper size; shows stacked layers (0 = flat, no shifted copies)


def set_layer_offset(v: float) -> None:
    """Per-layer shift used to show stacked layers. 0 draws all layers in place (cleanest look)."""
    global LAYER_OFFSET
    LAYER_OFFSET = float(v)


def _layer_depths(state: PaperState, order: list[int] | None = None) -> list[int]:
    """For each face: how many faces painted before it (default: below it) overlap it."""
    polys = [f.folded for f in state.faces]
    order = list(range(len(polys))) if order is None else order
    d = [0] * len(polys)
    for n, i in enumerate(order):
        d[i] = sum(g.overlap_area(polys[i], polys[j]) > 1e-6 for j in order[:n])
    return d


def _draw_bend_cap(ax, state: PaperState, bend: dict, shifts, z: float, back: bool = False) -> None:
    """Join two stacked layers along a fold edge: a small rounded cap of paper wrapping round it."""
    fa, fb = state.faces[bend["a"]], state.faces[bend["b"]]
    p, q = (g.apply(fa.T, np.array(pt)) for pt in bend["seg"])
    sa, sb = shifts[bend["a"]], shifts[bend["b"]]
    d = sb - sa
    if np.linalg.norm(d) < 1e-9:
        return
    u = g.unit(q - p)
    n = np.array([-u[1], u[0]])
    cen = np.mean(fa.folded, axis=0)
    if np.dot(cen - p, n) > 0:
        n = -n  # point away from the paper body
    h = 0.5 * np.linalg.norm(d)
    ts = np.linspace(0, 1, 13)
    arc = lambda e: [e + sa + t * d + n * h * np.sin(np.pi * t) for t in ts]
    poly = np.array(arc(p) + arc(q)[::-1])
    color = FRONT_COLOR if fb.front_up != back else BACK_COLOR
    ax.add_patch(Polygon(poly, closed=True, facecolor=color, edgecolor=EDGE_COLOR, linewidth=1.2,
                         joinstyle="round", zorder=z))


def draw_state(ax, state: PaperState, back: bool = False) -> None:
    """Paint faces bottom-up so upper layers hide what is under them. back=True: seen from below
    (layers in reverse, other side's colour); the caller mirrors the x axis."""
    order = list(range(len(state.faces)))
    if back:
        order.reverse()
    off_dir = np.array([-1.0, 1.0]) / np.sqrt(2) * LAYER_OFFSET * state.size
    depths = _layer_depths(state, order)
    shifts = [off_dir * min(d, 6) for d in depths]
    rank = {k: i for i, k in enumerate(order)}
    bends = _bends(state)
    for i, k in enumerate(order):
        f, shift = state.faces[k], shifts[k]
        z = 1 + i * 0.01
        poly = f.folded + shift
        ax.add_patch(Polygon(poly, closed=True, facecolor=FRONT_COLOR if f.front_up != back else BACK_COLOR,
                             edgecolor=EDGE_COLOR, linewidth=1.2, joinstyle="round", zorder=z))
        for b in bends:  # cap goes with whichever of its two faces is painted later
            if k in (b["a"], b["b"]) and rank[k] == max(rank[b["a"]], rank[b["b"]]):
                _draw_bend_cap(ax, state, b, shifts, z + 0.005, back)
        for c in state.creases:  # crease lines visible on this face
            seg = g.clip_segment(np.array(c.p), np.array(c.q), f.orig)
            if seg is None:
                continue
            p, q = g.apply(f.T, np.array(seg)) + shift
            ax.plot([p[0], q[0]], [p[1], q[1]], color=CREASE_COLOR, linewidth=0.7, zorder=z + 0.002)


def _line_extent(state: PaperState, a, u, pad: float):
    pts = np.vstack([f.folded for f in state.faces])
    t = (pts - a) @ u
    return a + (t.min() - pad) * u, a + (t.max() + pad) * u


def draw_next_step(ax, state: PaperState, step: dict) -> None:
    op = step.get("op")
    lo, hi = state.bbox()
    center = (lo + hi) / 2
    span = float(max(hi - lo))
    if op in FOLD_OPS:
        a, u, move = fold_geometry(state, step)
        color = VALLEY_COLOR if op == "valley" else MOUNTAIN_COLOR
        style = (0, (6, 4)) if op == "valley" else (0, (6, 3, 1, 3, 1, 3))
        p, q = _line_extent(state, a, u, 0.08 * span)
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, linestyle=style, linewidth=2.2, zorder=5)
        target = g.apply(g.reflection(a, u), move)
        arrow = FancyArrowPatch(
            move, target, connectionstyle="arc3,rad=0.35", color=color, linewidth=2,
            arrowstyle="-|>,head_length=8,head_width=5",
            linestyle="-" if op == "valley" else "--", zorder=6,
        )
        ax.add_patch(arrow)
        ax.plot(*move, "o", color=color, markersize=5, zorder=6)
    elif op in ("inside_reverse", "petal"):
        a, u, move = fold_geometry(state, step)
        color = "#7b3fa0"
        p, q = _line_extent(state, a, u, 0.08 * span)
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, linestyle=(0, (6, 3, 1, 3)), linewidth=2.2, zorder=5)
        target = g.apply(g.reflection(a, u), move)
        ax.add_patch(FancyArrowPatch(move, target, connectionstyle="arc3,rad=0.35", color=color, linewidth=2,
                                     arrowstyle="-|>,head_length=8,head_width=5", zorder=6))
        ax.plot(*move, "o", color=color, markersize=5, zorder=6)
        label = "inside reverse" if op == "inside_reverse" else "petal fold"
        ax.text(center[0], hi[1] + 0.06 * span, label, ha="center", fontsize=11, color=color)
    elif op == "squash":
        color = "#7b3fa0"
        a, u, _ = resolve_line_ref(state, step["hinge"])
        p, q = _line_extent(state, a, u, 0.08 * span)
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, linestyle=(0, (2, 3)), linewidth=2.0, zorder=5)
        move = resolve_point(state, step["move"])
        ax.plot(*move, "o", color=color, markersize=6, zorder=6)
        ax.text(center[0], hi[1] + 0.06 * span, "squash: lift, open, flatten", ha="center", fontsize=11, color=color)
    elif op == "flip":
        y = lo[1] - 0.12 * span
        ax.add_patch(FancyArrowPatch((center[0] - 0.25 * span, y), (center[0] + 0.25 * span, y),
                                     connectionstyle="arc3,rad=-0.5", arrowstyle="<|-|>,head_length=8,head_width=5",
                                     color="black", linewidth=1.8))
        ax.text(center[0], y - 0.06 * span, "turn over", ha="center", va="top", fontsize=11)
    elif op == "rotate":
        deg = float(step["degrees"])
        r = 0.6 * span
        start = center + r * np.array([np.cos(np.radians(60)), np.sin(np.radians(60))])
        end = center + r * np.array([np.cos(np.radians(60 + deg)), np.sin(np.radians(60 + deg))])
        ax.add_patch(FancyArrowPatch(start, end, connectionstyle=f"arc3,rad={0.4 if deg > 0 else -0.4}",
                                     arrowstyle="-|>,head_length=8,head_width=5", color="black", linewidth=1.8))
        ax.text(*(center + np.array([0, 0.7 * span])), f"rotate {deg:g}°", ha="center", fontsize=11)
    elif op == "unfold":
        ax.text(center[0], hi[1] + 0.08 * span, "unfold", ha="center", fontsize=11)


def render_step_png(path, state: PaperState, title: str, next_step: dict | None, limits, sides: bool = True) -> None:
    """One step picture. sides=True: front (with the next-step hint) and back side by side."""
    (x0, y0), (x1, y1) = limits
    if not sides:
        fig, ax = plt.subplots(figsize=(6, 6.4), dpi=110)
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11, 6.2), dpi=110)
    for i, ax in enumerate(axes):
        back = i == 1
        draw_state(ax, state, back)
        if next_step is not None and not back:
            draw_next_step(ax, state, next_step)
        ax.set_xlim(x1, x0) if back else ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.axis("off")
        if sides:
            ax.set_title("back" if back else "front", fontsize=9, color="#6b5a4a")
    if sides:
        fig.suptitle(title, fontsize=11, wrap=True)
    else:
        axes[0].set_title(title, fontsize=11, wrap=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def render_crease_pattern(path, state: PaperState) -> None:
    s = state.size
    fig, ax = plt.subplots(figsize=(5, 5), dpi=110)
    ax.add_patch(Polygon([[0, 0], [s, 0], [s, s], [0, s]], closed=True, facecolor="white",
                         edgecolor="black", linewidth=1.5))
    for c in state.creases:
        color, style = (VALLEY_COLOR, (0, (6, 4))) if c.assignment == "V" else (MOUNTAIN_COLOR, (0, (6, 3, 1, 3)))
        ax.plot([c.p[0], c.q[0]], [c.p[1], c.q[1]], color=color, linestyle=style, linewidth=1.5)
    ax.set_xlim(-0.05 * s, 1.05 * s)
    ax.set_ylim(-0.05 * s, 1.05 * s)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Crease pattern (blue dashed = valley, red dash-dot = mountain)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
