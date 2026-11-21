"""MENU action mode: candidate list + scripted agent that only picks menu options (no API calls)."""

import os

import pytest

from origami.agents.menu_agent import MENU_TOOLS, _execute, reference_policy, run_menu_episode
from origami.env import FoldEnv
from origami.match import state_hash
import json

from origami.menu import DEFAULT_CAP, FAMILIES, MACRO_CAP, REPEAT_CAP, candidates, menu_text, thumbnails_png


def test_flat_sheet_menu_basics():
    env = FoldEnv("models/cup.json")
    opts = candidates(env)
    assert 2 <= len(opts) <= DEFAULT_CAP + MACRO_CAP + REPEAT_CAP + 1
    hashes = [o.hash for o in opts]
    assert len(set(hashes)) == len(hashes)  # deduped
    assert state_hash(env.state) not in hashes  # no no-ops
    for o in opts:  # every option is legal and its id is its position
        assert env.preview(o.step).ok
        assert o.description
    assert [o.id for o in opts] == list(range(1, len(opts) + 1))
    # diagonal fold and book fold are both on the menu of a flat square
    names = menu_text(opts)
    assert "onto" in names


def test_thumbnails_png():
    env = FoldEnv("models/cup.json")
    png = thumbnails_png(candidates(env)[:4])
    assert png[:4] == b"\x89PNG"


def test_choose_and_custom_logged():
    env = FoldEnv("models/cup.json")
    opts = candidates(env)
    counts = {"menu_moves": 0, "custom_moves": 0, "illegal": 0, "bad_choice": 0, "undo": 0, "view_ops": 0}
    r = _execute(env, opts, "choose", {"option": 1}, counts, "F1")
    assert r["ok"] and r["source"] == "menu"
    r = _execute(env, opts, "choose", {"option": 999}, counts, "F1")
    assert not r["ok"] and counts["bad_choice"] == 1
    r = _execute(env, opts, "valley", {"line": '{"onto":["tr","bl"]}', "plan": "x"}, counts, "F1")
    assert r["source"] == "custom"
    assert counts["menu_moves"] == 1 and counts["custom_moves"] + counts["illegal"] == 1
    assert {t["function"]["name"] for t in MENU_TOOLS} >= {"choose", "valley", "flip", "undo", "submit"}


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("ORIGAMI_SLOW"), reason="~8 min; set ORIGAMI_SLOW=1")
@pytest.mark.parametrize("model", ["cup", "hat", "bird_base"])
def test_reference_reachable_through_menu(model):
    """Scripted agent: every reference fold is picked from the (capped) menu; reaches M2."""
    path = f"models/{model}.json"
    stats = {}
    s = run_menu_episode(path, reference_policy(path, stats), images=False, log=lambda *a: None,
                         compute_deletable=False)
    assert s["custom_moves"] == 0 and s["illegal"] == 0
    assert s["score"]["m2"], s["score"]
    # most reference folds are on the first menu page; the rest need at most a few `more` calls
    assert sum(n == 0 for n in stats["more_calls"]) >= len(stats["more_calls"]) - 2


def test_menu_survives_degenerate_outline_after_two_folds():
    """Regression: zero-length outline edge crashed candidates() (ValueError: zero-length direction)."""
    env = FoldEnv("models/cup.json")
    for line, move in [({"through": [[0.25, -1.0], [0.25, 1.0]]}, [0.125, 0.5]),
                       ({"through": [[1.425, -0.1], [-0.175, 1.1]]}, [0.625, 0.75])]:
        assert env.apply({"op": "valley", "line": line, "move": move}).ok
    assert len(candidates(env, more={f: 3 for f in FAMILIES})) > 0


def test_menu_survives_replayed_opus_cup_actions():
    """Regression: edges shorter than geometry.EPS (but > 1e-9) crashed outline() in menu building."""
    env = FoldEnv("models/cup.json")
    acts = [
        {"op": "valley", "line": {"through": [[1.2071067811865475, -0.20710678118654757], [-0.20710678118654746, 1.2071067811865475]]}, "move": [0.6666666666666666, 0.6666666666666666]},
        {"op": "valley", "line": {"onto": ["tl", [0.414214, 0]]}, "move": "tl"},
        {"op": "valley", "line": {"through": [[0.11731671058177595, -0.7167729011708042], [0.882683289418224, 1.1309862822728043]]}, "move": [0.6666666916949726, 0.13807118745770222]},
    ]
    for a in acts:
        candidates(env)
        assert env.apply(a).ok
    candidates(env)


@pytest.mark.parametrize("model", ["cup", "hat"])
def test_menu_random_walk_fuzz(model):
    import random
    seeds = range(6) if os.environ.get("ORIGAMI_SLOW") else range(1)
    for seed in seeds:
        rng = random.Random(seed)
        env = FoldEnv(f"models/{model}.json")
        for _ in range(4):
            opts = candidates(env)
            if not opts:
                break
            assert env.apply(rng.choice(opts).step).ok
        candidates(env)


def _page1(env, step):
    r = env.apply(step)
    assert r.ok, r.reason
    h = state_hash(env.state)
    env.undo(1)
    return next((o for o in candidates(env) if o.hash == h), None)


def test_hat_horns_and_brim_on_first_page_in_sonnet_state():
    """Sonnet's plan run: right first 5 moves (square view), then horns / brim were not on the first page
    (horn: flap section page 2; 2nd horn: missing; brim point-up: onto page 2 in Opus' state)."""
    d = json.load(open("tests/data/menu_sonnet_hat_state.json"))
    env = FoldEnv("models/hat.json")
    for s in d["moves"]:
        assert env.apply(s).ok
    o = _page1(env, d["horn1"])
    assert o is not None and o.fam == "flap"
    assert env.apply(d["horn1"]).ok
    o = _page1(env, d["horn2"])  # the other horn: mirror of the last fold
    assert o is not None and o.fam == "repeat"
    assert env.apply(d["horn2"]).ok
    assert _page1(env, d["brim_point_up"]) is not None


def test_repeat_is_not_chained():
    env = FoldEnv("models/hat.json")
    env.apply({"op": "valley", "line": {"onto": ["br", "bl"]}})
    rep = [o for o in candidates(env) if o.fam == "repeat"]
    assert rep
    assert env.apply(rep[0].step).ok
    assert not [o for o in candidates(env) if o.fam == "repeat"]


def test_repeat_uses_the_paper_mirror_line_not_any_silhouette_line():
    """Hat after horn 1: the diamond silhouette has 4 mirror lines, but only the vertical one is a mirror
    line of the folded paper -> exactly one 'repeat', and it makes the reference's second horn."""
    from origami.engine import Simulator

    steps = json.load(open("models/hat.json"))["steps"]
    env = FoldEnv("models/hat.json")
    for s in steps[:6]:
        assert env.apply(s).ok
    sim = Simulator(1.0)
    for s in steps[:7]:
        sim.apply(s)
    rep = [o for o in candidates(env) if o.fam == "repeat"]
    assert len(rep) == 1 and rep[0].hash == state_hash(sim.state)


class _FakeClient:
    model = "fake/model"

    def __init__(self):
        self.seen = []

    def chat(self, messages, tools):
        self.seen.append(messages)
        call = ({"name": "choose", "arguments": json.dumps({"option": 1, "plan": "1. fold"})} if len(self.seen) == 1
                else {"name": "submit", "arguments": "{}"})
        return {"choices": [{"message": {"tool_calls": [{"function": call}]}}], "usage": {}}


def _texts_and_images(messages):
    texts, imgs = [], []
    for m in messages:
        parts = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
        for p in parts:
            (texts.append(p["text"]) if p["type"] == "text" else imgs.append(p["image_url"]["url"]))
    return "\n".join(texts), imgs


def test_text_goal_mode_leaks_nothing_from_the_target(tmp_path):
    import base64

    from origami.agents.menu_agent import run_episode_menu

    goal = "Fold a samurai helmet (kabuto) from one square sheet of paper (coloured on one side)."
    env = FoldEnv("models/hat.json")
    target_img = "data:image/png;base64," + base64.b64encode(env.target_png()).decode()
    leaks = ["par ", "par)", "(par", "known solution", "Target", "target outline", env.title, "simple version",
             "What it is for", "outline + visible colours"]
    leaks += [ln.split(":", 1)[1].strip() for ln in env.goal_text().splitlines() if ln.startswith("Target outline")]
    fake = _FakeClient()
    s = run_episode_menu("models/hat.json", "fake/model", out_root=tmp_path, client=fake, goal_text=goal)
    assert len(fake.seen) == 2 and s["text_goal"]
    for msgs in fake.seen:
        text, imgs = _texts_and_images(msgs)
        assert "GOAL\n" + goal in text
        for bad in leaks:
            assert bad not in text, bad
        assert target_img not in imgs
        assert len(imgs) == 2  # own paper + menu thumbnails only
    assert set(s["variants"]) == {"hat", "hat_behind"}
    # control: the default (image) goal does show the target picture and par
    fake2 = _FakeClient()
    run_episode_menu("models/hat.json", "fake/model", out_root=tmp_path, client=fake2)
    text, imgs = _texts_and_images(fake2.seen[0])
    assert "par " in text and "known solution" in text and target_img in imgs
