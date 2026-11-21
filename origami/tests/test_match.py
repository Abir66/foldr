"""Goal matcher (M1-M4) and canonical state hash (src/origami/match.py)."""

import json
from pathlib import Path

import pytest

from origami.cli import run_model
from origami.engine import Simulator
from origami.match import VisitedTable, match, state_hash, transformed

MODELS = ["bird_base", "crane", "cup", "fortune_teller", "hat", "triangle"]
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runs():
    return {m: run_model(json.loads((ROOT / "models" / f"{m}.json").read_text())) for m in MODELS}


def _sim(steps):
    sim = Simulator()
    for s in steps:
        sim.apply(s)
    return sim.state


def _all(r):
    return r.m1 and r.m2 and r.m3 and r.m4


@pytest.mark.parametrize("name", MODELS)
def test_self_and_symmetric_copies(runs, name):
    A = runs[name][-1]
    h = state_hash(A)
    r = match(A, A.copy())
    assert _all(r) and r.iou == 1.0 and r.scores["M4"] == 1.0
    variants = [
        transformed(A, rotate=37, shift=(0.3, -1.0)),
        transformed(A, mirror=True, rotate=100),
        transformed(A, flip=True, rotate=-71.3),
    ]
    if name == "crane":  # slowest model: one combined copy is enough
        variants = [transformed(A, mirror=True, flip=True, rotate=23.5, shift=(2, 0))]
    for B in variants:
        r = match(A, B)
        assert r.m1 and r.m2, (name, r)
        assert r.iou > 0.99 and min(r.colour_front, r.colour_back) > 0.99
        assert _all(r), (name, r)  # exact same folded state, so all levels pass
        assert state_hash(B) == h


@pytest.mark.parametrize("name", ["bird_base", "crane", "cup", "hat", "triangle"])
def test_stopped_early_fails(runs, name):
    states = runs[name]
    A = states[-1]
    h = state_hash(A)
    # last two earlier states that are really different (view ops / fold+unfold give the same state)
    earlier = []
    for st in reversed(states[:-1]):
        if state_hash(st) != h and all(state_hash(st) != state_hash(e) for e in earlier):
            earlier.append(st)
        if len(earlier) == 2:
            break
    assert earlier
    for B in earlier:
        r = match(A, B)
        assert not r.m1 and not r.m2 and not r.m4, (name, r)
        assert r.shape_score < 0.9


def test_different_models_fail(runs):
    pairs = [("cup", "hat"), ("bird_base", "crane"), ("triangle", "fortune_teller")]
    for a, b in pairs:
        r = match(runs[a][-1], runs[b][-1])
        assert not r.m1 and not r.m2, (a, b, r)
        assert state_hash(runs[a][-1]) != state_hash(runs[b][-1])


BOOK = [{"op": "valley", "line": {"vertical": [0.5, 0.5]}, "move": "bl"}]
CUPBOARD = [{"op": "valley", "line": {"vertical": [0.25, 0.5]}, "move": "bl"},
            {"op": "valley", "line": {"vertical": [0.75, 0.5]}, "move": "br"}]
CUPBOARD_MIXED = [{"op": "valley", "line": {"vertical": [0.25, 0.5]}, "move": "bl"},
                  {"op": "mountain", "line": {"vertical": [0.75, 0.5]}, "move": "br"}]


def test_same_look_different_fold_is_m2_not_m3():
    # book fold and cupboard fold: same half-width outline, white on both sides; different creases
    r = match(_sim(BOOK), _sim(CUPBOARD))
    assert r.m1 and r.m2 and not r.m3 and not r.m4
    assert r.crease_f1 < 0.5


def test_same_outline_different_colours_is_m1_not_m2():
    # cupboard with one flap folded behind: same outline, half the front shows the coloured side
    r = match(_sim(BOOK), _sim(CUPBOARD_MIXED))
    assert r.m1 and not r.m2
    assert r.colour_front < 0.95


CORNER_A = {"op": "valley", "line": {"onto": ["bl", "center"]}}
CORNER_B = {"op": "valley", "line": {"onto": ["tr", "center"]}}


def test_move_order_gives_same_hash():
    s1 = _sim([CORNER_A, CORNER_B])
    s2 = _sim([CORNER_B, CORNER_A])
    s3 = _sim([CORNER_A, CORNER_B, {"op": "flip"}, {"op": "rotate", "degrees": 45}])
    assert state_hash(s1) == state_hash(s2) == state_hash(s3)
    assert _all(match(s1, s2))
    # the same two folds on the other diagonal = a D4 symmetry of the sheet
    s4 = _sim([{"op": "valley", "line": {"onto": ["br", "center"]}},
               {"op": "valley", "line": {"onto": ["tl", "center"]}}])
    assert state_hash(s4) == state_hash(s1)
    # only one corner folded: different state
    assert state_hash(_sim([CORNER_A])) != state_hash(s1)


def test_visited_table():
    t = VisitedTable()
    s1 = _sim([CORNER_A, CORNER_B])
    h, new = t.add(s1, 2)
    assert new and len(t) == 1 and s1 in t
    h2, new2 = t.add(_sim([CORNER_B, CORNER_A]), 5, note="dead end: no flaps left")
    assert h2 == h and not new2 and len(t) == 1
    e = t.get(s1)
    assert e.step == 2 and e.count == 2 and e.notes == ["dead end: no flaps left"]
    assert _sim([CORNER_A]) not in t
