"""Physical validity checker (validate.py): impossible steps are caught, the models pass."""

import json
from pathlib import Path

import numpy as np
import pytest

from origami import geometry as g
from origami.cli import main, run_model
from origami.engine import FoldError, PhysicsError
from origami.paper import Crease, Face, PaperState
from origami.validate import (check_crease_pattern, check_state, dangling_creases, validate_model)

MODELS = Path(__file__).resolve().parents[1] / "models"
CLEAN = ["triangle", "cup", "hat", "fortune_teller", "bird_base", "crane"]
PETAL_VERTICES = [(0.793, 0.5), (0.5, 0.207), (0.5, 0.793), (0.207, 0.5)]

def load(name):
    base = MODELS.parent / "tests" / "data" / "models" if name == "hat_tuck" else MODELS
    return json.loads((base / f"{name}.json").read_text())


def kinds(rep, severity="error"):
    return {i.kind for i in rep.issues if i.severity == severity}


# ---- the models ---------------------------------------------------------------------

@pytest.mark.parametrize("name", CLEAN)
def test_clean_models_have_no_errors(name):
    rep = validate_model(load(name))
    assert rep.ok, rep.to_text()


def test_hat_tuck_is_only_a_warning():
    rep = validate_model(load("hat_tuck"))
    assert [(i.step, i.kind) for i in rep.issues] == [(10, "tuck")]


def test_bird_base_and_crane_have_no_warnings():
    assert not validate_model(load("bird_base")).issues
    # crane narrowing folds say {"top": 1, "flap": true}, but the leg's edge is a folded edge, so
    # the paper behind it is dragged along (real paper does the same): honest-selection warning only
    rep = validate_model(load("crane"))
    assert [(i.step, i.kind, i.severity) for i in rep.issues] == \
        [(k, "selection_grew", "warning") for k in (10, 11, 13, 14)]


def test_petal_vertices_are_three_to_one():
    """Each kite point of the bird base: petal hinge line one way, midline (to the centre) + both
    kite lines the other way (Maekawa 3/1: the side triangles lie on top of the lifted petal, so
    the kite creases fold opposite to the hinge); the midline from the kite point to the edge is
    flat again."""
    from origami.engine import _folded_edges
    st = run_model(load("bird_base"))[-1]
    edges = _folded_edges(st)
    for P in PETAL_VERTICES:
        rays = {}
        for p, q, a in edges:
            for x, y in ((p, q), (q, p)):
                if np.allclose(x, P, atol=2e-3):
                    rays[tuple(np.round(y, 3))] = a
        asg = sorted(rays.values())
        assert len(asg) == 4 and asg.count("M") in (1, 3), (P, rays)
        mid = rays[(0.5, 0.5)]
        assert asg.count(mid) == 3, (P, rays)  # midline goes with the kites
        hinge = [a for y, a in rays.items() if not np.isclose(y[0] % 1, 0) and y != (0.5, 0.5)]
        assert hinge == [a for a in "MV" if a != mid], (P, rays)  # hinge is the odd one out
        assert not any(np.isclose(y[0] % 1, 0) and np.isclose(y[1], 0.5) or
                       np.isclose(y[1] % 1, 0) and np.isclose(y[0], 0.5) for y in rays), rays


# ---- motion: folding through other layers ------------------------------------------

FOUR_LAYERS = [{"op": "valley", "line": {"horizontal": "center"}, "move": "bl"},
               {"op": "valley", "line": {"vertical": "center"}, "move": [0.8, 0.8]}]


def corner_fold(op, layers):
    """All four raw corners sit at (0, 1): fold one layer's corner across the others."""
    return {"name": "t", "steps": FOUR_LAYERS + [
        {"op": op, "line": {"through": [[0, 0.8], [0.2, 1]]}, "move": [0.05, 0.95], "layers": layers}]}


@pytest.mark.parametrize("op,layers", [("valley", {"bottom": 1}), ("mountain", {"top": 1})])
def test_fold_through_other_layers_is_rejected(op, layers):
    rep = validate_model(corner_fold(op, layers))
    col = [i for i in rep.errors if i.kind == "collision"]
    assert len(col) == 1 and col[0].step == 3
    assert "pass through 3 stationary" in col[0].message and "near (" in col[0].message
    assert "taco_tortilla" in kinds(rep)  # the result is impossible too


@pytest.mark.parametrize("op,layers", [("valley", {"top": 1}), ("mountain", {"bottom": 1}),
                                       ("valley", "all")])
def test_fold_of_outer_layer_is_fine(op, layers):
    assert validate_model(corner_fold(op, layers)).ok


def test_strict_mode_raises_physics_error():
    model = corner_fold("valley", {"bottom": 1})
    assert len(run_model(model)) == 4  # default: unchanged behaviour
    with pytest.raises(PhysicsError) as e:
        run_model(model, strict=True)
    assert isinstance(e.value, FoldError)
    assert e.value.issues and "step 3" in str(e.value)
    run_model(load("cup"), strict=True)  # valid models pass strict mode


def test_fold_error_is_reported():
    rep = validate_model({"name": "t", "steps": [{"op": "unfold"}]})
    assert [i.kind for i in rep.errors] == ["fold_error"]


# ---- flat-state layer constraints on hand-made states -------------------------------

def sq(x0, y0, w=1.0, h=1.0):
    return np.array([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]], float)


def shift(dx, dy):
    T = np.eye(3)
    T[:2, 2] = (dx, dy)
    return T


def test_cyclic_order_is_rejected():
    # three separate pieces moved onto one spot, ordered 0 > 1 > 2 > 0
    faces = [Face(sq(0, 0), np.eye(3)), Face(sq(3, 0), shift(-3.2, 0.1)), Face(sq(6, 0), shift(-6.1, 0.2))]
    st = PaperState(faces, [], size=10, above={(0, 1), (1, 2), (2, 0)})
    iss = [i for i in check_state(st, 1) if i.kind == "cyclic_order"]
    assert iss and iss[0].severity == "error" and sorted(iss[0].faces) == [0, 1, 2]


def test_taco_tortilla_is_rejected():
    # A = left half, B = right half folded onto A (crease x = 0.5); C straddles the crease line
    A = Face(sq(0, 0, 0.5, 1))
    B = Face(sq(0.5, 0, 0.5, 1), g.reflection(np.array([0.5, 0.0]), np.array([0.0, 1.0])))
    C = Face(sq(5, 0, 0.5, 1), shift(-4.75, 0))  # lands on [0.25, 0.75] x [0, 1]
    ok = PaperState([A, C, B], [], size=1, above={(1, 0), (2, 1), (2, 0)})  # C under nothing: B > C > A
    bad = [i for i in check_state(ok, 1) if i.kind == "taco_tortilla"]
    assert bad and set(bad[0].faces) == {0, 1, 2}
    fine = PaperState([C, A, B], [], size=1, above={(1, 0), (2, 1), (2, 0)})  # C below the taco
    # (C is a stray piece, not part of a 1x1 sheet: ignore the paper-area check)
    assert not [i for i in check_state(fine, 1) if i.severity == "error" and i.kind != "paper_lost"]


def test_taco_taco_interleaving_is_rejected():
    st = run_model({"steps": FOUR_LAYERS})[-1]
    assert not [i for i in check_state(st, 2) if i.severity == "error"]
    # reorder the 4 faces into the stack bottom->top: L1, L2, R1, R2 (folds interleave)
    rot = st.faces
    left = sorted([k for k, f in enumerate(rot) if np.allclose(f.T[:2, :2], np.diag([1, -1]))
                   or np.allclose(f.T[:2, :2], np.eye(2))], key=lambda k: rot[k].orig[:, 1].min())
    right = [k for k in range(4) if k not in left]
    # pair each right face with the left face it is hinged to (shares an edge on the sheet)
    pair = {r: next(l for l in left if g.shared_edge(rot[r].orig, rot[l].orig) is not None) for r in right}
    L1, L2 = left
    R1 = next(r for r in right if pair[r] == L1)
    R2 = next(r for r in right if pair[r] == L2)
    stack = [L1, L2, R1, R2]
    st.above = {(stack[b], stack[a]) for a in range(4) for b in range(a + 1, 4)}
    assert "taco_taco" in {i.kind for i in check_state(st, 2) if i.severity == "error"}


def test_tear_is_rejected():
    A, B = Face(sq(0, 0, 0.5, 1)), Face(sq(0.5, 0, 0.5, 1), shift(0.3, 0))
    st = PaperState([A, B], [], above=set())
    assert "tear" in {i.kind for i in check_state(st, 1)}


# ---- crease pattern -----------------------------------------------------------------

def rays(center, angles_deg, asg):
    c = np.array(center)
    return [(tuple(c), tuple(c + 1.0 * np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])), s)
            for a, s in zip(angles_deg, asg)]


def test_flat_foldable_vertex_passes():
    assert not check_crease_pattern(rays((0.5, 0.5), [0, 90, 180, 270], "MVVV"))
    assert not check_crease_pattern(rays((0.5, 0.5), [0, 60, 180, 300], "VMMM"))


@pytest.mark.parametrize("angles,asg,kind", [
    ([0, 90, 180, 270], "MMVV", "maekawa"),   # 2 M / 2 V
    ([0, 90, 180, 270], "VVVV", "maekawa"),   # 4 valleys
    ([0, 80, 180, 270], "MVVV", "kawasaki"),  # angles don't alternate to 180
    ([0, 120, 240], "VVV", "kawasaki"),       # odd degree
])
def test_non_flat_foldable_vertex_is_rejected(angles, asg, kind):
    iss = check_crease_pattern(rays((0.5, 0.5), angles, asg))
    assert kind in {i.kind for i in iss} and all(i.severity == "error" for i in iss)
    assert iss[0].sheet_point == [0.5, 0.5]


def test_dangling_crease_end():
    creases = [Crease((0.0, 0.5), (0.5, 0.5), "V")]  # stops in the middle of the sheet
    (v,) = dangling_creases(creases)
    assert np.allclose(v, (0.5, 0.5))
    creases.append(Crease((0.5, 0.0), (0.5, 1.0), "M"))  # now it meets another crease
    assert not dangling_creases(creases)
    # a collinear crease only counts if it carries the line on past the point
    assert len(dangling_creases([Crease((0.1, 0.5), (0.5, 0.5), "V"), Crease((0.0, 0.5), (0.3, 0.5), "V")])) == 1


# ---- CLI ----------------------------------------------------------------------------

def test_cli_check_exit_codes(tmp_path, capsys):
    assert main(["check", str(MODELS / "triangle.json")]) == 0
    assert "physically valid" in capsys.readouterr().out
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(corner_fold("valley", {"bottom": 1})))
    assert main(["check", str(bad), "--json"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False and out["issues"][0]["kind"] == "collision"


def test_sampled_mode_on_simple_model():
    rep = validate_model(load("cup"), sampled=True)
    assert rep.ok and not rep.issues
