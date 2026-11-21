import json
from pathlib import Path

import numpy as np
import pytest

from origami import geometry as g
from origami.engine import FoldError, Simulator
from origami.landmarks import resolve_point

MODELS = Path(__file__).parent.parent / "models"


def run(steps):
    sim = Simulator()
    for s in steps:
        sim.apply(s)
    return sim


HALF_UP = {"op": "valley", "line": {"through": ["left_mid", "right_mid"]}, "move": "bl"}


# ---- reflection math ---------------------------------------------------------

def test_reflect_across_diagonal_swaps_xy():
    T = g.reflection(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    assert np.allclose(g.apply(T, [0.3, 0.7]), [0.7, 0.3])


def test_reflect_across_horizontal_line():
    T = g.reflection(np.array([0.0, 0.5]), np.array([1.0, 0.0]))
    assert np.allclose(g.apply(T, [0.2, 0.1]), [0.2, 0.9])


def test_reflection_is_involution_and_flips_orientation():
    T = g.reflection(np.array([0.3, -0.2]), np.array([0.4, 1.3]))
    assert np.allclose(T @ T, np.eye(3))
    assert np.linalg.det(T[:2, :2]) < 0


def test_split_convex_square():
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    left, right = g.split_convex(sq, np.array([0.5, 0]), np.array([0, 1.0]))
    assert g.area(left) == pytest.approx(0.5)
    assert g.area(right) == pytest.approx(0.5)


# ---- valley vs mountain layer order -------------------------------------------

def test_valley_puts_moving_part_on_top():
    sim = run([HALF_UP])
    faces = sim.state.faces
    assert len(faces) == 2
    top, bottom = faces[-1], faces[0]
    assert top.orig[:, 1].max() <= 0.5 + 1e-9  # top layer came from the bottom half
    assert not top.front_up and bottom.front_up
    assert np.allclose(sorted(top.folded[:, 1]), sorted(bottom.folded[:, 1]))


def test_mountain_puts_moving_part_underneath():
    sim = run([{**HALF_UP, "op": "mountain"}])
    bottom = sim.state.faces[0]
    assert bottom.orig[:, 1].max() <= 0.5 + 1e-9
    assert not bottom.front_up


def test_flip_reverses_layers_and_sides():
    sim = run([HALF_UP])
    before = [f.front_up for f in sim.state.faces]
    sim.apply({"op": "flip"})
    after = [f.front_up for f in sim.state.faces]
    assert after == [not x for x in reversed(before)]


# ---- partial layers ---------------------------------------------------------

def test_top_only_folds_one_layer_when_layers_not_attached():
    # Diagonal fold: two layers meet at the open corner (tr), not attached there.
    sim = run([
        {"op": "valley", "line": {"through": ["tl", "br"]}, "move": "bl"},
        {"op": "valley", "line": {"through": [[0.5, 1.0], [1.0, 0.5]]}, "move": [1.0, 1.0],
         "layers": {"top": 1}},
    ])
    faces = sim.state.faces
    assert len(faces) == 3  # only the top layer got cut
    tip = faces[-1]
    assert np.allclose(g.apply(tip.T, [0.0, 0.0]), [0.5, 0.5])  # bl corner folded down
    assert np.allclose(sim.state.to_folded([1.0, 1.0]), [1.0, 1.0])  # tr stayed


def test_attached_layers_are_dragged_along():
    # After folding in half, both right-hand layers are joined by the fold edge, so
    # "top 1" must still move both (paper can't tear).
    sim = run([
        HALF_UP,
        {"op": "valley", "line": {"vertical": "center"}, "move": "tr", "layers": {"top": 1}},
    ])
    assert len(sim.state.faces) == 4


def test_bottom_layer_mountain():
    sim = run([
        {"op": "valley", "line": {"through": ["tl", "br"]}, "move": "bl"},
        {"op": "mountain", "line": {"through": [[0.5, 1.0], [1.0, 0.5]]}, "move": [1.0, 1.0],
         "layers": {"bottom": 1}},
    ])
    assert len(sim.state.faces) == 3
    assert np.allclose(sim.state.to_folded([1.0, 1.0]), [0.5, 0.5])
    assert np.allclose(sim.state.faces[0].T @ np.array([1.0, 1.0, 1]), [0.5, 0.5, 1])


# ---- area & creases ---------------------------------------------------------

@pytest.mark.parametrize("name", ["triangle", "cup", "hat", "fortune_teller"])
def test_area_conserved_through_models(name):
    model = json.loads((MODELS / f"{name}.json").read_text())
    sim = Simulator()
    for step in model["steps"]:
        sim.apply(step)
        assert sim.state.total_area() == pytest.approx(1.0)
        assert sum(f.area for f in sim.state.faces) == pytest.approx(1.0)


def test_crease_recorded_on_original_sheet():
    sim = run([HALF_UP])
    (c,) = sim.state.creases
    assert c.assignment == "V"
    assert {round(c.p[1], 6), round(c.q[1], 6)} == {0.5}


def test_folding_two_layers_gives_opposite_assignments():
    sim = run([HALF_UP, {"op": "valley", "line": {"vertical": "center"}, "move": "tr"}])
    new = sim.state.creases[1:]
    assert len(new) == 2
    assert sorted(c.assignment for c in new) == ["M", "V"]


def test_mountain_on_front_is_M():
    sim = run([{**HALF_UP, "op": "mountain"}])
    assert sim.state.creases[0].assignment == "M"


def test_unfold_restores_shape_and_keeps_crease():
    sim = run([HALF_UP, {"op": "unfold"}])
    assert len(sim.state.faces) == 1
    assert sim.state.total_area() == pytest.approx(1.0)
    assert len(sim.state.creases) == 1


def test_unfold_replays_later_flip():
    sim = run([HALF_UP, {"op": "flip"}, {"op": "unfold"}])
    (f,) = sim.state.faces
    assert not f.front_up


# ---- landmarks --------------------------------------------------------------

def test_landmarks_follow_the_paper():
    sim = run([HALF_UP])
    assert np.allclose(resolve_point(sim.state, "bl"), [0, 1])
    assert np.allclose(resolve_point(sim.state, {"lerp": ["tl", "tr"], "t": 0.25}), [0.25, 1])
    assert np.allclose(resolve_point(sim.state, {"orig": [0.5, 0.25]}), [0.5, 0.75])


def test_onto_moves_first_point_onto_second():
    sim = run([{"op": "valley", "line": {"onto": ["br", "tl"]}}])
    assert np.allclose(sim.state.to_folded([1, 0]), [0, 1])


def test_reserved_macro_errors():
    with pytest.raises(FoldError):
        run([{"op": "sink"}])
