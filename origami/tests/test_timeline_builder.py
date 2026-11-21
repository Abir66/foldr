"""TimelineBuilder (incremental timeline) must give exactly what the old batch build_timeline gave."""

import json
from functools import lru_cache
from pathlib import Path

import pytest

from origami.cli import run_model
from origami.steps_export import (
    TimelineBuilder,
    _poly,
    _world_for,
    build_timeline,
    round_floats,
    state_json,
    step_json,
    timeline_to_json,
)

MODELS = Path(__file__).parent.parent / "models"
NAMES = sorted(p.stem for p in MODELS.glob("*.json"))


def _load(name):
    return json.loads((MODELS / f"{name}.json").read_text())


def _reference(model: dict, states) -> dict:
    """The batch build_timeline as it was before TimelineBuilder (kept here as the oracle)."""
    steps = model["steps"]
    world = _world_for(states, steps)
    sts = [state_json(s) for s in states]
    for sj, w in zip(sts, world):
        sj["world"] = w
    tsteps = [step_json(k, steps[k], states[k], states[k + 1]) for k in range(len(steps))]
    for k, st in enumerate(tsteps):
        st["side_before"], st["side_after"] = world[k]["s"], world[k + 1]["s"]
    return {
        "name": model.get("name", "model"),
        "title": model.get("title", model.get("name", "")),
        "size": states[0].size,
        "colored_side": model.get("paper", {}).get("colored_side", "up"),
        "original_faces": [{"id": i, "orig": _poly(f.orig)} for i, f in enumerate(states[0].faces)],
        "states": sts,
        "steps": tsteps,
    }


@lru_cache(maxsize=None)
def _case(name):
    model = _load(name)
    states = run_model(model)
    return model, states, timeline_to_json(_reference(model, states))


@pytest.mark.parametrize("name", NAMES)
def test_build_timeline_matches_reference(name):
    model, _, ref = _case(name)
    assert timeline_to_json(build_timeline(model)) == ref


@pytest.mark.parametrize("name", NAMES)
def test_incremental_with_undo_matches_reference(name):
    """Append step by step; at a few points (incl. right after flips) undo and redo. Same bytes."""
    model, states, ref = _case(name)
    steps = model["steps"]
    n = len(steps)
    redo_at = {0, n // 2, n - 1} | {k for k, s in enumerate(steps) if s["op"] == "flip"}
    tb = TimelineBuilder.for_model(model, states[0])
    for k in range(n):
        st = tb.append(steps[k], states[k], states[k + 1])
        assert st is tb.steps[-1] and len(tb) == k + 1
        if k in redo_at:
            tb.pop()
            assert len(tb) == k and len(tb.states) == k + 1
            tb.append(steps[k], states[k], states[k + 1])
    if n >= 3:  # multi-step undo, then redo
        tb.pop(3)
        for k in range(n - 3, n):
            tb.append(steps[k], states[k], states[k + 1])
    assert timeline_to_json(tb.to_dict()) == ref


def test_pop_gives_prefix():
    model, states, ref = _case("cup")
    full = json.loads(ref)
    tb = TimelineBuilder.for_model(model, states[0])
    for k, step in enumerate(model["steps"]):
        tb.append(step, states[k], states[k + 1])
    for keep in (4, 1, 0):
        tb.pop(len(tb) - keep)
        d = json.loads(timeline_to_json(tb.to_dict()))
        assert d["steps"] == full["steps"][:keep]
        assert d["states"] == full["states"][: keep + 1]
    with pytest.raises(IndexError):
        tb.pop()


def test_blank_sheet_timeline():
    _, states, ref = _case("triangle")
    d = TimelineBuilder(states[0], name="mine").to_dict()
    assert d["name"] == d["title"] == "mine" and d["colored_side"] == "up"
    assert d["steps"] == [] and len(d["states"]) == 1
    assert json.loads(timeline_to_json(d))["states"][0] == json.loads(ref)["states"][0]


def test_round_floats():
    assert round_floats({"a": [1.234567891, -0.000000001, 2, True, None, "x"], "b": (0.5,)}, 5) == {
        "a": [1.23457, 0.0, 2, True, None, "x"], "b": [0.5]}
    assert timeline_to_json({"x": -1e-9}, 5) == '{"x":0.0}'
    _, _, ref = _case("cup")
    small = timeline_to_json(json.loads(ref), ndigits=5)
    assert len(small) < len(ref)
    a, b = json.loads(ref), json.loads(small)
    assert len(a["steps"]) == len(b["steps"])
    assert abs(a["states"][-1]["faces"][0]["T"][0][2] - b["states"][-1]["faces"][0]["T"][0][2]) <= 5e-6
