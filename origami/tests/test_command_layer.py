"""Command layer v2: sticker labels, snapping, sticker positions, same-spot help (offline)."""

import json
import re

import pytest

from origami.agents.command_layer import CommandLayer, Labeler
from origami.agents.llm_agent import SYSTEM_PROMPT
from origami.env import FoldEnv
from origami.landmarks import NAMED_POINTS


def _to_labels(o, lab):
    if isinstance(o, str):
        return lab.label(o) if o in NAMED_POINTS else o
    if isinstance(o, dict):
        return {k: (v if k in ("op", "description") else _to_labels(v, lab)) for k, v in o.items()}
    if isinstance(o, list):
        return [_to_labels(v, lab) for v in o]
    return o


@pytest.mark.parametrize("model", ["cup", "hat", "bird_base"])
def test_scripted_agent_label_only(model):
    """Reference solution replayed through the command layer with label-only, LLM-style (JSON string) args."""
    env = FoldEnv(f"models/{model}.json")
    layer = CommandLayer(env)
    for st in env.ref_steps:
        args = {k: (json.dumps(_to_labels(v, layer.labels)) if isinstance(v, (dict, list)) else v)
                for k, v in st.items() if k not in ("op", "description")}
        res = layer.run(st["op"], args)
        if st["op"] not in ("flip", "rotate"):
            assert res["ok"], (st["op"], res["text"])
        text = layer.stickers_text()
        assert text.startswith("Where stickers are now")
    sc = env.submit(compute_deletable=False)
    assert sc.m2, sc.scores


def test_labels_stable_and_neutral():
    env = FoldEnv("models/cup.json")
    layer = CommandLayer(env)
    first = env.observe(images=False).text
    assert "A=(0,0)" in first and not re.search(r"\b(bl|br|tl|tr|center)\b", first)
    assert layer.labels.label("bl") == "A" and layer.labels.label("center") == "I"
    layer.run("valley", {"line": {"onto": ["C", "A"]}})  # tr onto bl
    t2 = env.observe(images=False).text
    assert "p250" not in t2 and "p500" not in t2
    assert layer.labels.label("bl") == "A"
    # old names still accepted
    assert layer.run("unfold", {})["ok"]
    r = layer.run("valley", {"line": {"onto": ["tr", "bl"]}})
    assert r["ok"] and not re.search(r"\b(tr|bl)\b", r["text"])


def test_stickers_stacked_and_same_spot_help():
    env = FoldEnv("models/cup.json")
    layer = CommandLayer(env)
    assert layer.run("valley", {"line": {"onto": ["C", "A"]}})["ok"]  # C lands on A
    txt = layer.stickers_text()
    assert "stacked on the same spot" in txt and "A, C" in txt
    r = layer.run("valley", {"line": {"onto": ["A", "C"]}})
    assert not r["ok"] and r["code"] == "SAME_POINT"
    assert "A and C are on the same spot" in r["text"] and "Pick labels on different spots" in r["text"]


def test_snap_xy_and_lerp():
    env = FoldEnv("models/cup.json")
    layer = CommandLayer(env)
    r = layer.run("valley", {"line": {"onto": ["C", [0.01, 0.005]]}})  # near A
    assert r["ok"] and "snapped to A" in r["text"]
    env2 = FoldEnv("models/cup.json")
    l2 = CommandLayer(env2)
    r = l2.run("valley", {"line": {"onto": ["C", {"lerp": ["A", "B"], "t": 0.01}]}})
    assert "snapped to A" in r["text"]


def test_prompt_documents_helpers_and_plan():
    for w in ("intersect", "perpendicular", "near", "perpendicular_to", "150 words"):
        assert w in SYSTEM_PROMPT
    assert "Think briefly" not in SYSTEM_PROMPT and "do not need to work out" not in SYSTEM_PROMPT


def test_labeler_order_and_reserved():
    lab = Labeler()
    labs = [lab.label(f"p{i:03d}_500") for i in range(100, 100 + 70)]
    assert labs[0] == "J" and "AA" in labs and not {"BL", "BR", "TL", "TR"} & set(labs)


def test_view_turns_do_not_eat_action_budget(tmp_path):
    from test_agents import FakeClient, _reply
    from origami.agents.llm_agent import run_episode

    replies = [_reply(("rotate", {"degrees": 10, "plan": "p"})) for _ in range(5)]
    replies += [_reply(("valley", {"line": json.dumps({"onto": ["C", "A"]}), "plan": "p"})),
                _reply(("submit", {"plan": "p"}))]
    s = run_episode("models/cup.json", "fake/m", images=False, max_turns=3, out_root=tmp_path,
                    client=FakeClient(replies), log=lambda *_: None)
    assert s["stop_reason"] == "submitted" and s["legal"] == 6
