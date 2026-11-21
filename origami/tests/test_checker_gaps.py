"""Checker gaps found by the audit (research/08-physics-checker-audit.md, faults 1,2,3,6,8,9,11)."""

import json
from pathlib import Path

import numpy as np
import pytest

from origami import engine
from origami import geometry as g
from origami.cli import run_model
from origami.paper import Face, PaperState
from origami.validate import check_invariants, check_state, check_step, validate_model

MODELS = Path(__file__).resolve().parents[1] / "models"
ALL = ["triangle", "cup", "hat", "fortune_teller", "bird_base", "crane"]


def load(name):
    return json.loads((MODELS / f"{name}.json").read_text())


def sq(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


def errors(issues, kind=None):
    return [i for i in issues if i.severity == "error" and (kind is None or i.kind == kind)]


# a | b: left / right half of one flat piece (same placement, joined by a flat seam at x = 0.5);
# c: a separate piece of paper lying across the seam, at [0.25, 0.75] x [0.2, 0.8]
C_ORIG = sq(0, 1.2, 0.5, 1.8)
C_T = np.array([[1, 0, 0.25], [0, 1, 0], [0, 0, 1]]) @ g.reflection(np.array([0, 1.0]), np.array([1.0, 0]))


# ---- fault 1: face across a flat seam, ordered between the seam's two sides ---------------

def test_face_between_two_sides_of_flat_seam_is_rejected():
    a, b, c = Face(sq(0, 0, 0.5, 1)), Face(sq(0.5, 0, 1, 1)), Face(C_ORIG, C_T)
    bad = PaperState([a, c, b], [], size=2.0, above={(1, 0), (2, 1)})  # a < c < b
    iss = errors(check_state(bad, 1), "tortilla_tortilla")
    assert iss and set(iss[0].faces) == {0, 1, 2}
    assert "step 1:" in iss[0].message and "pass through" in iss[0].message
    ok = PaperState([a, b, c], [], size=2.0, above={(2, 0), (2, 1)})  # c on top of both
    assert not errors(check_state(ok, 1), "tortilla_tortilla")


# ---- fault 2: merging pieces of one face must not hide a contradictory order -------------

def test_finish_keeps_contradictory_pieces_visible():
    p1 = Face(sq(0, 0, 0.5, 1)); p1.src = 0
    p2 = Face(sq(0.5, 0, 1, 1)); p2.src = 0
    q = Face(C_ORIG, C_T); q.src = 1
    before = PaperState([Face(sq(0, 0, 1, 1)), Face(C_ORIG, C_T)], [], size=2.0, above={(1, 0)})
    mid = PaperState([p1, q, p2], [], size=2.0, above={(0, 1), (1, 2)})  # p1 > q > p2
    out = engine._finish(mid, before)
    # either the engine refuses, or the pieces stay apart and the checker sees the problem
    assert len(out.faces) == 3
    assert errors(check_state(out, 1), "tortilla_tortilla")


def test_finish_still_merges_consistent_pieces():
    p1 = Face(sq(0, 0, 0.5, 1)); p1.src = 0
    p2 = Face(sq(0.5, 0, 1, 1)); p2.src = 0
    q = Face(C_ORIG, C_T); q.src = 1
    before = PaperState([Face(sq(0, 0, 1, 1)), Face(C_ORIG, C_T)], [], size=2.0, above={(1, 0)})
    mid = PaperState([p1, q, p2], [], size=2.0, above={(1, 0), (1, 2)})  # q above both
    out = engine._finish(mid, before)
    assert len(out.faces) == 2 and len(out.above) == 1


# ---- fault 3: the faces must still make up exactly one unstretched sheet --------------------

def test_missing_paper_is_rejected():
    s = run_model(load("cup"))[-1]
    s.faces = s.faces[:-1]
    s.above = {(i, j) for i, j in s.above if i < len(s.faces) and j < len(s.faces)}
    iss = errors(check_state(s, 7), "paper_lost")
    assert iss and "missing" in iss[0].message


def test_duplicated_paper_is_rejected():
    s = run_model(load("triangle"))[-1]
    s.faces = s.faces + [Face(s.faces[0].orig.copy(), s.faces[0].T @ np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]], float))]
    s.faces[-1].T = s.faces[-1].T.copy()
    s.faces[-1].T[0, 2] += 3.0  # moved off to the side so no layer order is involved
    s.above = set(s.above)
    kinds = {i.kind for i in errors(check_state(s, 2))}
    assert "paper_overlap" in kinds


def test_stretched_paper_is_rejected():
    s = run_model(load("cup"))[-1]
    s.faces[0].T = s.faces[0].T @ np.diag([1.05, 1, 1])
    iss = errors(check_state(s, 7), "stretch")
    assert iss and iss[0].faces == [0] and "stretch" in iss[0].message


# ---- fault 6: overlapping faces without a layer order are an error -------------------------

def test_unordered_overlap_is_an_error():
    a = Face(sq(0, 0, 0.5, 1))
    b = Face(sq(0.5, 0, 1, 1), g.reflection(np.array([0.5, 0.0]), np.array([0.0, 1.0])))
    st = PaperState([a, b], [], size=1.0, above=set())
    assert errors(check_state(st, 1), "unordered_overlap")


# ---- fault 8: drag-along widening a layer selection is reported ---------------------------

BOOK_THEN_BOTTOM = {"name": "t", "steps": [
    {"op": "valley", "line": {"vertical": "center"}, "move": "bl"},
    {"op": "valley", "line": {"horizontal": "center"}, "move": [0.8, 0.8], "layers": {"bottom": 1}}]}


def test_selection_growth_is_a_warning():
    rep = validate_model(BOOK_THEN_BOTTOM)
    iss = [i for i in rep.issues if i.kind == "selection_grew"]
    assert iss and iss[0].severity == "warning" and iss[0].step == 2
    assert "1 to 2 layers" in iss[0].message
    strict = validate_model(BOOK_THEN_BOTTOM, strict=True)
    assert [i.severity for i in strict.issues if i.kind == "selection_grew"] == ["error"]


def test_no_selection_growth_when_layers_are_free():
    model = {"name": "t", "steps": [
        {"op": "valley", "line": {"vertical": "center"}, "move": "bl"},
        {"op": "valley", "line": {"through": [[0.8, 1.0], [1.0, 0.8]]}, "move": [0.95, 0.95],
         "layers": {"top": 1}}]}
    assert not [i for i in validate_model(model).issues if i.kind == "selection_grew"]


# ---- fault 9: step-to-step invariants ----------------------------------------------------

def test_invariant_static_pair_swapping_is_caught():
    sim = engine.Simulator()
    sim.apply({"op": "valley", "line": {"horizontal": "center"}, "move": "bl"})
    before = sim.state.copy()
    step = {"op": "valley", "line": {"vertical": "center"}, "move": [0.8, 0.8]}
    sim.apply(step)
    after = sim.state.copy()
    assert not check_invariants(before, after, step, 2)
    # corrupt: swap the order of every pair (the static left half included)
    after.above = {(j, i) for i, j in after.above}
    iss = check_invariants(before, after, step, 2)
    assert iss and iss[0].kind == "order_changed" and "step 2" in iss[0].message


@pytest.mark.parametrize("name", ALL)
def test_models_keep_invariants(name):
    rep = validate_model(load(name))
    assert not [i for i in rep.issues if i.kind == "order_changed"], rep.to_text()


# ---- fault 11: bad step data is a reported fold error, not a crash --------------------------

@pytest.mark.parametrize("step", [
    {"op": "squash", "spine": {"through": ["bl", "tr"]}, "move": "bl"},  # no hinge
    {"op": "valley", "move": "bl"},                                      # no line
    {"op": "rotate"},                                                    # no degrees
])
def test_bad_step_json_is_reported(step):
    rep = validate_model({"name": "t", "steps": [step]})
    assert [i.kind for i in rep.errors] == ["fold_error"]
    assert rep.errors[0].message.startswith("step 1")


# ---- the models stay clean -----------------------------------------------------------------

@pytest.mark.parametrize("name", ALL)
def test_models_still_error_free(name):
    rep = validate_model(load(name))
    assert rep.ok, rep.to_text()


def test_check_step_accepts_strict():
    st = run_model(load("triangle"))
    step = load("triangle")["steps"][0]
    assert not errors(check_step(st[0], st[1], step, 1, strict=True))
