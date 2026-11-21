"""Time-limited menus, the no-target env and the JSON bridge."""

import json
import subprocess
import sys

import pytest

from origami.env import FoldEnv, summarize_score
from origami.menu import candidates
from origami.web import bridge


def _cup_env(n=4):
    env = FoldEnv("models/cup.json")
    for s in env.ref_steps[:n]:
        assert env.apply(s).ok
    return env


def _sig(opts):
    return [(o.description, o.hash, o.fam, o.fam_rank, round(o.cost, 9)) for o in opts]


@pytest.fixture(scope="module")
def cup_full():
    env = _cup_env()
    return env, candidates(env)


def test_deadline_menu_is_a_prefix_of_the_full_menu(cup_full, monkeypatch):
    env, full = cup_full
    import origami.menu as M

    clock = iter(range(1000))  # every monotonic() call = 1 more second
    monkeypatch.setattr("time.monotonic", lambda: float(next(clock)))
    part = candidates(env, deadline_s=3.5)
    monkeypatch.undo()
    assert part.skipped, "deadline should have cut some sections"
    body = [o for o in part if o.fam != "unfold"]
    assert _sig(body) == _sig(full[: len(body)])
    for fam in part.skipped:
        assert part.truncated[fam] is True
        assert not any(o.fam == fam for o in part)
    assert part[-1].fam == "unfold" and full[-1].fam == "unfold"
    assert set(part.skipped) | {o.fam for o in part} >= set(M.FAMILIES) - {"repeat"}


def test_no_deadline_is_unchanged_and_on_section_reports_each_section(cup_full):
    env, full = cup_full
    seen = []
    again = candidates(env, deadline_s=None, on_section=lambda fam, opts: seen.append((fam, len(opts))))
    assert _sig(again) == _sig(full)
    assert again.skipped == []
    assert [f for f, _ in seen] == ["repeat", "onto", "through", "edges", "align", "flap", "tuck", "special"]
    assert sum(n for _, n in seen) == sum(1 for o in full if o.fam != "unfold")


def test_one_section_alone_dedupes_like_the_full_menu(cup_full):
    env, full = cup_full
    for fam in ("through", "align"):
        others = {o.hash for o in full if o.fam != fam}
        alone = candidates(env, sections={fam}, exclude_hashes=others)
        assert _sig(alone) == _sig([o for o in full if o.fam == fam])
        assert all(o.fam == fam for o in alone)


def test_free_env_apply_undo_submit():
    env = FoldEnv(None, size=2.0, colored_side="down")
    assert env.par is None and env.target_state is None and env.goal_text() == ""
    assert env.size == 2.0 and env.colored_side == "down"
    r = env.apply({"op": "valley", "line": {"onto": ["tr", "bl"]}})
    assert r.ok and r.progress is None
    assert not env.apply({"op": "valley", "line": {"onto": ["tr", "tr"]}}).ok
    assert env.undo().ok and env.trajectory == []
    env.apply({"op": "valley", "line": {"onto": ["tl", "br"]}})
    s = env.submit()
    assert s.passed is None and s.target is None and s.moves == 1 and s.scores == {}
    assert s.to_dict()["passed"] is None
    json.dumps(s.to_dict())
    assert "no target" in summarize_score(s)
    with pytest.raises(ValueError):
        env.target_png()


def _call(method, params=None):
    out = json.loads(bridge.call(method, json.dumps(params or {})))
    assert "exception" not in out, out
    return out


def test_bridge_session_round_trip():
    r = _call("session.new", {"size": 1, "colored_side": "up"})
    assert r["rev"] == 0 and len(r["points"]) == 9 and r["timeline"]["steps"] == []
    lab = {p["name"]: p["label"] for p in r["points"]}
    r = _call("session.custom", {"p": lab["tr"], "q": lab["bl"], "kind": "onto"})
    assert r["ok"] and r["rev"] == 1 and r["timeline_step"] and r["state"]["faces"]
    r = _call("session.apply", {"step": {"op": "valley", "line": {"onto": ["tr", "tr"]}}})
    assert not r["ok"] and r["rev"] == 1 and "REJECTED" in r["text"]
    pushed = []
    out = json.loads(bridge.call("session.menu", json.dumps({}), lambda s: pushed.append(json.loads(s))))
    assert out["options"] and not out["skipped"]
    assert [p["section"] for p in pushed][:2] == ["repeat", "onto"]
    opt = out["options"][0]
    assert {"id", "description", "section", "step", "result_state"} <= set(opt)
    r = _call("session.apply", {"step": opt["step"]})
    assert r["ok"] and r["rev"] == 2
    assert len(_call("session.steps")["model"]["steps"]) == 2
    r = _call("session.undo", {"n": 1})
    assert r["ok"] and r["timeline_len"] == 1 and r["rev"] == 3
    assert _call("session.preview", {"step": opt["step"]})["ok"]
    files = _call("export", {"kinds": ["fold", "model_json", "timeline_json"]})
    assert [f["name"].split(".", 1)[1] for f in files["files"]] == ["fold", "json", "timeline.json"]
    assert not files["errors"]


def test_bridge_timeline_partial_and_errors():
    model = json.load(open("models/cup.json"))
    bad = dict(model, steps=model["steps"][:2] + [{"op": "nope"}])
    r = _call("timeline", {"model": bad})
    assert r["failed_step"] == 3 and r["shown_steps"] == 2 and len(r["timeline"]["steps"]) == 2
    assert json.loads(bridge.call("no.such"))["exception"]["type"] == "ValueError"


def test_core_imports_without_matplotlib():
    code = ("import sys, origami.web.bridge as b, origami.menu, origami.env, origami.steps_export, origami.cli;"
            "b.call('session.new'); b.call('session.menu', '{\"deadline_ms\": 1}');"
            "print('matplotlib' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
