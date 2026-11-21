"""Layer selection: {"top"/"bottom": n, "local": true} counts layers at the move point only."""

import json
from pathlib import Path

import numpy as np
import pytest

from origami import engine as E
from origami import geometry as g
from origami.engine import FoldError, Simulator, fold
from origami.paper import Face, PaperState

ROOT = Path(__file__).resolve().parents[1]


def square(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


def shift(dx, dy):
    T = np.eye(3)
    T[:2, 2] = (dx, dy)
    return T


def stray_flap_state():
    """Bottom layer B and top layer A, both covering [0,1]^2 in folded space, plus a small
    piece C (not joined to A or B by any paper edge) lying on A at the top-right corner.
    A is the top layer at (0.7, 0.3) but C is above A elsewhere, like the crane's neck vs wing."""
    B = Face(square(0, 0, 1, 1))
    A = Face(square(2, 0, 3, 1), shift(-2, 0))
    C = Face(square(5, 5, 5.2, 5.2), shift(-4.2, -4.2))  # -> [0.8,1]^2
    return PaperState([B, A, C], creases=[])


LINE = (np.array([0.5, 0.0]), np.array([0.0, 1.0]))  # x = 0.5
MOVE = np.array([0.7, 0.3])


def cut_cand(state):
    st = E._cut(E._begin(state), *LINE)
    side = 1 if g.side(*LINE, MOVE) > 0 else -1
    cand = [k for k, f in enumerate(st.faces) if E._side_of(f, *LINE) == side]
    return st, cand


def names(st, ks):
    """Which source face (0=B, 1=A, 2=C) each selected piece comes from."""
    return sorted(st.faces[k].src for k in ks)


def test_local_top_counts_only_the_stack_at_the_move_point():
    st, cand = cut_cand(stray_flap_state())
    assert names(st, E._select_layers(st, cand, {"top": 1, "local": True}, MOVE)) == [1]
    assert names(st, E._select_layers(st, cand, {"top": 2, "local": True}, MOVE)) == [0, 1]
    assert names(st, E._select_layers(st, cand, {"bottom": 1, "local": True}, MOVE)) == [0]


def test_global_top_is_fooled_by_an_unrelated_flap():
    """The old (non-local) count: C over A's corner pushes A out of the 'top 1'."""
    st, cand = cut_cand(stray_flap_state())
    assert names(st, E._select_layers(st, cand, {"top": 1}, MOVE)) == [2]
    with pytest.raises(FoldError):
        E._select_layers(st, cand, {"top": 1, "flap": True}, MOVE)


def test_local_fold_moves_the_top_layer_and_leaves_the_stray_piece():
    out = fold(stray_flap_state(), *LINE, MOVE, "valley", {"top": 1, "local": True})
    lo, hi = out.bbox()
    assert np.allclose(lo, [0, 0]) and np.allclose(hi, [1, 1])
    # A's right half now lies over A's left half on the left; B and C did not move
    left_top = max((k for k, f in enumerate(out.faces) if g.contains(f.folded, np.array([0.3, 0.3]))))
    assert np.isclose(g.area(out.faces[left_top].folded), 0.5) and not out.faces[left_top].front_up
    assert any(np.allclose(f.folded.mean(0), [0.9, 0.9]) and f.front_up for f in out.faces)  # C
    assert any(np.allclose(f.folded.mean(0), [0.5, 0.5]) and np.isclose(f.area, 1.0)
               for f in out.faces)  # B: whole, unmoved


def test_local_drags_along_paper_attached_to_the_seed():
    """Triangle (2 layers joined along the diagonal). A fold line crossing that folded edge:
    'top 1 local' seeds only the top layer, but the bottom layer is joined to it by the
    diagonal (not on the fold line), so it must come along (no tearing)."""
    sim = Simulator()
    sim.apply({"op": "valley", "line": {"onto": ["tr", "bl"]}})  # triangle below the diagonal
    a, u = np.array([0.5, 0.0]), np.array([0.0, 1.0])
    st = E._cut(E._begin(sim.state), a, u)
    pt = np.array([0.8, 0.1])
    side = 1 if g.side(a, u, pt) > 0 else -1
    cand = [k for k, f in enumerate(st.faces) if E._side_of(f, a, u) == side]
    seeds = E._local_layers(st, cand, {"top": 1, "local": True}, pt)
    full = E._select_layers(st, cand, {"top": 1, "local": True}, pt, line=(a, u))
    assert len(seeds) == 1 and sorted(full) == sorted(cand) and len(cand) == 2


def test_local_point_on_a_crease_takes_both_sides_as_one_level():
    """At a point on the line where two pieces of one layer meet, both are 'the top layer'."""
    B = Face(square(0, 0, 1, 1))
    A1 = Face(square(2, 0, 2.5, 1), shift(-2, 0))
    A2 = Face(square(2.5, 0, 3, 1), shift(-2, 0))
    st0 = PaperState([B, A1, A2], creases=[])
    line = (np.array([0.0, 0.6]), np.array([1.0, 0.0]))
    st = E._cut(E._begin(st0), *line)
    pt = np.array([0.5, 0.8])
    side = 1 if g.side(*line, pt) > 0 else -1
    cand = [k for k, f in enumerate(st.faces) if E._side_of(f, *line) == side]
    assert names(st, E._select_layers(st, cand, {"top": 1, "local": True}, pt)) == [1, 2]


def test_local_needs_a_count_and_a_point_on_the_moving_side():
    st, cand = cut_cand(stray_flap_state())
    with pytest.raises(FoldError):
        E._select_layers(st, cand, {"local": True}, MOVE)
    with pytest.raises(FoldError):
        E._select_layers(st, cand, {"top": 1, "local": True}, np.array([5.0, 5.0]))


# ---- crane ----------------------------------------------------------------------------

def crane_states():
    model = json.loads((ROOT / "models" / "crane.json").read_text())
    sim = Simulator()
    out = [sim.state.copy()]
    for s in model["steps"]:
        sim.apply(s)
        out.append(sim.state.copy())
    return model, out


def stack_at(st, p):
    return [k for k, f in enumerate(st.faces) if g.contains(f.folded, np.asarray(p, float))]


def test_crane_neck_goes_between_the_wings():
    """After the neck reverse fold, the neck lies between the front and back wing layers."""
    model, states = crane_states()
    k = next(i for i, s in enumerate(model["steps"], 1) if "neck" in s.get("description", "")
             and s["op"] == "inside_reverse")
    st = states[k]
    stack = stack_at(st, [0.75, 0.5])  # neck crosses the wings here (bottom -> top)

    def wing_layer(i):  # wing faces lie in the corner triangles of (1,0) / (0,1)
        c = st.faces[i].orig.mean(0)
        return min(np.linalg.norm(c - np.array(w)) for w in ([1, 0], [0, 1])) < 0.45

    kinds = [wing_layer(i) for i in stack]
    assert kinds[:2] == [True, True] and kinds[-2:] == [True, True]  # back wing, front wing
    assert not any(kinds[2:-2]) and len(stack) > 4  # neck layers in between


def test_crane_wings_are_folded_down_and_symmetric():
    model, states = crane_states()
    st = states[-1]
    assert st.acyclic and np.isclose(st.total_area(), 1.0)
    wing_tips = [st.to_folded(p) for p in [(1, 0), (0, 1)]]
    lo, hi = st.bbox()
    assert np.allclose(wing_tips[0], wing_tips[1], atol=1e-6)
    assert np.isclose(wing_tips[0][1], lo[1])  # wing tips are the lowest point
    # neck/tail tips (sheet corners (0,0), (1,1)) stick up above the folded wings
    hinge_y = st.to_folded((0.32442335, 0.86561945))[1]
    for c in [(0, 0), (1, 1)]:
        assert st.to_folded(c)[1] > hinge_y
    # symmetric outline: the body (wings) is mirror-symmetric about its centre line
    x0 = wing_tips[0][0]
    assert np.isclose(st.to_folded((0.32442335, 0.86561945))[0] + st.to_folded((0.13438055, 0.67557665))[0],
                      2 * x0, atol=1e-6)


def test_tuck_inside_follows_the_hinge_layer_past_its_creases():
    """_same_layer: H plus faces continuing it flat (shared edge, same transform)."""
    A1 = Face(square(0, 0, 0.5, 1))
    A2 = Face(square(0.5, 0, 1, 1))
    A3 = Face(square(1, 0, 1.5, 1), shift(-1, 0))  # touches A2 in orig, but moved: a fold
    B = Face(square(2, 0, 3, 1), shift(-2, 0))
    st = PaperState([A1, A2, A3, B], creases=[])
    assert E._same_layer(st, {0}, exclude=set()) == {0, 1}
    assert E._same_layer(st, {0}, exclude={1}) == {0}
