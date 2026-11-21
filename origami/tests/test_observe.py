"""Agent observation + inspect tool (src/origami/observe.py)."""

import json
from pathlib import Path

import numpy as np
import pytest

from origami.cli import run_model
from origami.engine import Simulator
from origami.geometry import signed_area
from origami.observe import inspect, name_to_orig, observe, outline, point_name, sheet_points

MODELS = sorted((Path(__file__).parent.parent / "models").glob("*.json"))
_cache: dict = {}


def _run(path):
    if path not in _cache:
        model = json.loads(path.read_text())
        _cache[path] = (model, run_model(model))
    return _cache[path]


def _cs(model):
    return model.get("paper", {}).get("colored_side", "up")


@pytest.mark.parametrize("path", MODELS, ids=lambda p: p.stem)
def test_observe_every_step(path):
    model, states = _run(path)
    for k, st in enumerate(states):
        # images only on first / middle / last step (keeps the suite fast); text on every step
        img = k in (0, len(states) // 2, len(states) - 1)
        obs = observe(st, model["steps"][:k], model["name"], colored_side=_cs(model), images=img)
        assert obs.text and obs.points
        assert obs.outline, f"step {k}: no outline"
        # outline area = silhouette area <= paper area, > 0
        area = abs(signed_area(obs.outline[0]))
        assert 0 < area <= st.size**2 + 1e-9
        if img:
            assert obs.cp_png[:4] == b"\x89PNG" and obs.folded_png[:4] == b"\x89PNG"
        # every landmark resolves through its name
        for nm, pi in obs.points.items():
            assert np.allclose(name_to_orig(nm, st.size, st), pi.orig, atol=1e-3)


@pytest.mark.parametrize("path", MODELS, ids=lambda p: p.stem)
def test_point_names_stable(path):
    """A name always means the same sheet point, and a sheet point keeps its name across steps."""
    model, states = _run(path)
    seen: dict[str, np.ndarray] = {}
    for st in states:
        for nm, p in sheet_points(st).items():
            assert point_name(p, st.size) == nm
            if nm in seen:
                assert np.allclose(seen[nm], p, atol=1e-6), nm
            seen[nm] = p
    # points of step k that still exist at k+1 keep their names (creases only add points)
    for a, b in zip(states, states[1:]):
        na, nb = sheet_points(a), sheet_points(b)
        for nm in na:
            if nm in nb:
                assert np.allclose(na[nm], nb[nm])


def test_crane_final_summary_budget():
    model, states = _run(next(p for p in MODELS if p.stem == "crane"))
    obs = observe(states[-1], model["steps"], "crane", colored_side="down", images=False)
    assert obs.tokens < 2000, obs.tokens
    assert "flap(bl)" in obs.text and "flap(tr)" in obs.text  # neck and tail tips
    assert "History:" in obs.text and "20. " in obs.text


def test_last_errors_shown():
    st = Simulator().state
    obs = observe(st, [], "x", last_errors=["step 1: paper passes through itself"], images=False)
    assert "REJECTED" in obs.text and "passes through" in obs.text


def _half():
    sim = Simulator()
    sim.apply({"op": "valley", "line": {"onto": ["tl", "bl"]}})  # top half down onto bottom half
    return sim.state


def test_inspect_half_fold_two_layers():
    st = _half()
    for spec in ({"orig": [0.3, 0.2]}, {"orig": [0.7, 0.8]}, [0.5, 0.25], "bl", "center", "left_mid"):
        r = inspect(st, {"layers_at": spec})
        assert r["depth"] == 2, (spec, r)
    r = inspect(st, "layers_at [0.25,0.25]")
    top, bot = r["faces_top_to_bottom"]
    assert top["face"] > bot["face"]
    assert top["colour_up"] == "w" and bot["colour_up"] == "c"  # folded flap shows its white back
    assert {top["sheet_point"], bot["sheet_point"]} == {"p250_250", "p250_750"}


def test_inspect_under_creases_face_fold():
    st = _half()
    top = inspect(st, "layers_at [0.25,0.25]")["faces_top_to_bottom"][0]["face"]
    r = inspect(st, {"under": top})
    assert [u["face"] for u in r["under"]] == [1 - top] and r["under"][0]["overlap"] == pytest.approx(0.5)
    assert inspect(st, "under tl")["face"] == top  # tl is on the folded-down flap
    assert inspect(st, "under bl")["under"] == []  # bottom layer: nothing below
    c = inspect(st, "creases_at left_mid")["creases"]
    assert len(c) == 1 and c[0]["made_as"] == "V" and c[0]["now"] == "V"
    assert inspect(st, "creases_at bl")["creases"] == []
    f = inspect(st, {"face": top})
    assert set(f["sheet_points"]) == {"left_mid", "right_mid", "tr", "tl"} and f["below_it"] == [1 - top]
    fold = inspect(st, "fold")
    assert fold["file_spec"] == 1.1 and len(fold["faces_vertices"]) == 2
    # unfold: crease stays but is flat now
    sim = Simulator()
    sim.apply({"op": "valley", "line": {"onto": ["tl", "bl"]}})
    sim.apply({"op": "unfold"})
    assert inspect(sim.state, "creases_at left_mid")["creases"][0]["now"] == "flat"


def test_outline_square_and_half():
    sq = outline(Simulator().state)
    assert len(sq) == 1 and len(sq[0]) == 4
    h = outline(_half())
    assert len(h) == 1 and len(h[0]) == 4
    assert signed_area(h[0]) == pytest.approx(0.5)  # CCW, half the sheet
