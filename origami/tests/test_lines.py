"""Huzita-Justin style line specs: bisect (O3), perpendicular (O4), point_onto_line (O5/O7)."""

import json
from pathlib import Path

import numpy as np
import pytest

from origami import geometry as g
from origami.engine import FoldError, Simulator
from origami.landmarks import resolve_line, resolve_point
from origami.paper import PaperState

MODELS = Path(__file__).parent.parent / "models"
SQ = PaperState.square()


def run(steps):
    sim = Simulator()
    for s in steps:
        sim.apply(s)
    return sim


def same_line(a, u, p, q):
    """Line (a, u) passes through p and q."""
    return abs(g.side(a, u, np.array(p, float))) < 1e-9 and abs(g.side(a, u, np.array(q, float))) < 1e-9


# ---- bisect (O3: line onto line) ---------------------------------------------

def test_bisect_edges_meeting_at_corner_gives_diagonal():
    a, u, move = resolve_line(SQ, {"bisect": [["bl", "br"], ["bl", "tl"]]})
    assert same_line(a, u, [0, 0], [1, 1])
    assert np.allclose(move, [1, 0])  # br (farther from crease than bl)
    sim = run([{"op": "valley", "line": {"bisect": [["bl", "br"], ["bl", "tl"]]}}])
    assert np.allclose(sim.state.to_folded([1, 0]), [0, 1])  # bottom edge now lies on left edge


def test_bisect_direction_picks_the_other_bisector():
    # Direction bl->br mapped onto tl->bl (pointing down): the bisector is y = -x.
    a, u, _ = resolve_line(SQ, {"bisect": [["bl", "br"], ["tl", "bl"]]})
    assert same_line(a, u, [0, 0], [1, -1])


def test_bisect_parallel_edges_gives_midline():
    a, u, move = resolve_line(SQ, {"bisect": [["bl", "br"], ["tl", "tr"]]})
    assert same_line(a, u, [0, 0.5], [1, 0.5])
    sim = run([{"op": "valley", "line": {"bisect": [["bl", "br"], ["tl", "tr"]]}}])
    assert np.allclose(sim.state.to_folded([0, 0]), [0, 1])


def test_bisect_uses_current_folded_positions():
    # After folding in half (bottom up), "bl" is at the top-left: bl-br is the top edge.
    sim = run([{"op": "valley", "line": {"through": ["left_mid", "right_mid"]}, "move": "bl"}])
    a, u, _ = resolve_line(sim.state, {"bisect": [["bl", "br"], ["left_mid", "right_mid"]]})
    assert same_line(a, u, [0, 0.75], [1, 0.75])


def test_bisect_same_line_errors():
    with pytest.raises(ValueError):
        resolve_line(SQ, {"bisect": [["bl", "br"], ["bottom_mid", "br"]]})


# ---- perpendicular (O4) --------------------------------------------------------

def test_perpendicular_through_point():
    a, u, _ = resolve_line(SQ, {"perpendicular": ["bl", "br"], "at": "center"})
    assert same_line(a, u, [0.5, 0], [0.5, 1])


# ---- point_onto_line with pivot (O5) ----------------------------------------------

def test_point_onto_line_through_pivot():
    spec = {"point_onto_line": "br", "line": ["bl", "tl"], "pivot": "bl"}
    a, u, move = resolve_line(SQ, spec)
    assert same_line(a, u, [0, 0], [1, 1])  # br lands on tl (inside the edge segment)
    assert np.allclose(move, [1, 0])
    sim = run([{"op": "valley", "line": spec}])
    assert np.allclose(sim.state.to_folded([1, 0]), [0, 1])


def test_point_onto_line_near_picks_other_solution():
    spec = {"point_onto_line": "br", "line": ["bl", "tl"], "pivot": "bl", "near": [0, -1]}
    a, u, _ = resolve_line(SQ, spec)
    assert same_line(a, u, [0, 0], [1, -1])  # br lands on (0, -1)


def test_point_onto_line_crease_passes_through_pivot_and_lands_on_line():
    spec = {"point_onto_line": "tr", "line": ["bl", "br"], "pivot": "left_mid"}
    a, u, _ = resolve_line(SQ, spec)
    assert abs(g.side(a, u, np.array([0, 0.5]))) < 1e-9
    landed = g.apply(g.reflection(a, u), np.array([1.0, 1.0]))
    assert abs(landed[1]) < 1e-9 and 0 <= landed[0] <= 1


def test_point_onto_line_no_solution():
    with pytest.raises(ValueError):
        resolve_line(SQ, {"point_onto_line": "center", "line": ["bl", "br"], "pivot": "top_mid"})


# ---- point_onto_line perpendicular to a line (O7) ------------------------------------

def test_point_onto_line_perpendicular_to():
    spec = {"point_onto_line": "br", "line": ["tl", "tr"], "perpendicular_to": ["bl", "tl"]}
    a, u, move = resolve_line(SQ, spec)
    assert same_line(a, u, [0, 0.5], [1, 0.5])
    assert np.allclose(move, [1, 0])


# ---- intersect accepts line specs ----------------------------------------------

def test_intersect_with_line_specs():
    p = resolve_point(SQ, {"intersect": [{"bisect": [["bl", "br"], ["bl", "tl"]]}, ["br", "tr"]]})
    assert np.allclose(p, [1, 1])
    p = resolve_point(SQ, {"intersect": [{"horizontal": "left_mid"}, ["bl", "tr"]]})
    assert np.allclose(p, [0.5, 0.5])


# ---- cup: new specs give the same shape as the old magic-number version ----------

OLD_CUP_C = 0.5857864376  # 2 - sqrt(2)
OLD_CUP_STEPS = [
    {"op": "rotate", "degrees": 45},
    {"op": "valley", "line": {"through": ["tl", "br"]}, "move": "bl"},
    {"op": "valley", "line": {"onto": ["br", {"lerp": ["tl", "tr"], "t": OLD_CUP_C}]}},
    {"op": "valley", "line": {"onto": ["tl", {"orig": [1.0, OLD_CUP_C]}]}},
    {"op": "valley", "line": {"horizontal": {"orig": [1.0, OLD_CUP_C]}}, "move": "bl", "layers": {"top": 1}},
    {"op": "flip"},
    {"op": "valley", "line": {"horizontal": {"orig": [1.0, OLD_CUP_C]}}, "move": "tr"},
]


def _faces(state):
    return sorted(
        (tuple(np.round(f.orig.mean(0), 6)), tuple(np.round(f.folded.mean(0), 6)), f.front_up)
        for f in state.faces
    )


def test_cup_without_magic_numbers_matches_old_shape():
    model = json.loads((MODELS / "cup.json").read_text())
    assert len(model["steps"]) == len(OLD_CUP_STEPS)
    old, new = Simulator(), Simulator()
    for so, sn in zip(OLD_CUP_STEPS, model["steps"]):
        old.apply(so)
        new.apply(sn)
        assert _faces(old.state) == _faces(new.state)
    assert [f.front_up for f in old.state.faces] == [f.front_up for f in new.state.faces]
    for fo, fn in zip(old.state.faces, new.state.faces):
        assert np.allclose(fo.folded, fn.folded, atol=1e-6)


# ---- "flap" layer option ---------------------------------------------------------

BLINTZ_2 = [
    {"op": "valley", "line": {"onto": ["bl", "center"]}},
    {"op": "valley", "line": {"onto": ["br", "center"]}},
]


def test_flap_option_moves_only_the_flap_under_move_point():
    # Two blintz flaps, tips at the center. Fold along x + y = 0.8: on the moving side
    # the left flap's tip is free, the right flap is reached too (and drags the base).
    fold = {"op": "valley", "line": {"through": [[0.3, 0.5], [0.5, 0.3]]}, "move": [0.45, 0.45],
            "layers": {"top": 1}}
    both = run(BLINTZ_2 + [fold])
    one = run(BLINTZ_2 + [{**fold, "layers": {"top": 1, "flap": True}}])
    assert np.allclose(both.state.to_folded([0, 0]), [0.3, 0.3])
    assert np.allclose(both.state.to_folded([1, 0]), [0.3, 0.3])
    assert np.allclose(one.state.to_folded([0, 0]), [0.3, 0.3])  # left flap (under move) folded
    assert np.allclose(one.state.to_folded([1, 0]), [0.5, 0.5])  # right flap untouched


def test_flap_option_errors_when_move_point_not_on_chosen_layer():
    with pytest.raises(FoldError):
        run(BLINTZ_2 + [{"op": "valley", "line": {"horizontal": [0, 0.4]}, "move": [0.5, 0.95],
                         "layers": {"top": 1, "flap": True}}])
