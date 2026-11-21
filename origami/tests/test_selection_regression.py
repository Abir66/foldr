"""Freeze the results of the finished models so engine changes (layer selection, tuck rules)
can't silently change them. Regenerate the fixture ONLY for an intended change:

    uv run python tests/test_selection_regression.py
"""

import json
from pathlib import Path

import numpy as np
import pytest

from origami.engine import Simulator

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "data" / "frozen_models.json"
FROZEN = ["triangle", "cup", "hat", "fortune_teller", "bird_base"]


def states_of(name):
    model = json.loads((ROOT / "models" / f"{name}.json").read_text())
    sim = Simulator(float(model.get("paper", {}).get("size", 1.0)))
    out = [sim.state.copy()]
    for step in model["steps"]:
        sim.apply(step)
        out.append(sim.state.copy())
    return out


def snapshot(st):
    return {
        "orig": [np.round(f.orig, 7).tolist() for f in st.faces],
        "T": [np.round(f.T, 7).tolist() for f in st.faces],
        "above": sorted([int(i), int(j)] for i, j in st.above),
        "creases": [[list(map(float, c.p)), list(map(float, c.q)), c.assignment]
                    for c in st.creases],
    }


@pytest.mark.parametrize("name", FROZEN)
def test_model_results_are_frozen(name):
    frozen = json.loads(FIXTURE.read_text())[name]
    states = states_of(name)
    assert len(states) == len(frozen)
    for k, (st, fr) in enumerate(zip(states, frozen)):
        snap = snapshot(st)
        assert len(snap["orig"]) == len(fr["orig"]), f"{name} state {k}: face count"
        for a, b in zip(snap["orig"], fr["orig"]):
            assert np.allclose(a, b, atol=1e-6), f"{name} state {k}: face polygon"
        for a, b in zip(snap["T"], fr["T"]):
            assert np.allclose(a, b, atol=1e-6), f"{name} state {k}: face transform"
        assert snap["above"] == fr["above"], f"{name} state {k}: layer order"
        assert len(snap["creases"]) == len(fr["creases"]), f"{name} state {k}: creases"
        for a, b in zip(snap["creases"], fr["creases"]):
            assert a[2] == b[2] and np.allclose(a[:2], b[:2], atol=1e-6), f"{name} state {k}"


if __name__ == "__main__":
    FIXTURE.parent.mkdir(exist_ok=True)
    FIXTURE.write_text(json.dumps({n: [snapshot(s) for s in states_of(n)] for n in FROZEN}))
    print(f"wrote {FIXTURE}")
