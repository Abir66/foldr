"""Agent runs with an event stream: Goal kinds, schema-valid events, live timeline, cancel, budget."""

import base64
import json
import threading
from pathlib import Path

import jsonschema
import pytest

from origami.agents.events import RunEvents
from origami.agents.goal import Goal
from origami.agents.llm_agent import run_episode
from origami.agents.menu_agent import reference_policy, run_episode_menu
from origami.agents.runner import AgentRun, config_from_request, run_agent
from origami.env import FoldEnv
from origami.llm import ScriptedClient, Usage, reply
from origami.steps_export import build_timeline, round_floats

SCHEMA = json.loads((Path(__file__).resolve().parents[2] / "docs" / "events.schema.json").read_text())
VALIDATOR = jsonschema.Draft202012Validator(SCHEMA)
NOLOG = dict(log=lambda *a, **k: None)


def check_stream(events: list[dict]) -> list[dict]:
    """Schema-valid, numbered, images defined before use, one run_started first / run_finished last.
    Returns the timeline steps a viewer ends with (append on committed actions, truncate on undo)."""
    for ev in events:
        errs = sorted(VALIDATOR.iter_errors(ev), key=lambda e: e.path)
        assert not errs, (ev["type"], [e.message[:200] for e in errs[:3]])
    assert [e["seq"] for e in events] == list(range(len(events)))
    assert events[0]["type"] == "run_started" and events[-1]["type"] == "run_finished"
    assert sum(e["type"] in ("run_started", "run_finished") for e in events) == 2
    seen_imgs, steps = set(), list(events[0]["timeline"]["steps"])

    def refs(o):
        if isinstance(o, dict):
            if "img" in o:
                yield o["img"]
            for v in o.values():
                yield from refs(v)
        elif isinstance(o, list):
            for v in o:
                yield from refs(v)

    seen_imgs.update(p["img"] for p in events[0]["goal_parts"] if "data" in p)
    for ev in events:
        if ev["type"] == "image":
            seen_imgs.add(ev["id"])
        for r in refs({k: v for k, v in ev.items() if k != "final_images"}):
            assert r in seen_imgs, (ev["type"], r)
        for r in ev.get("final_images", []):
            assert r in seen_imgs
        if ev["type"] == "action" and "timeline_step" in ev:
            steps.append(ev["timeline_step"])
            assert len(steps) == ev["timeline_len"]
        if ev["type"] == "undo":
            del steps[ev["timeline_len"]:]
    assert len(steps) == events[-1]["timeline_len"]
    return steps


def assert_timeline_matches_solution(events, steps):
    sol = events[-1]["solution"]
    ref = round_floats(build_timeline(sol), 5)["steps"]
    assert len(ref) == len(steps)
    for a, b in zip(ref, steps):
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def menu_client(ev: RunEvents, model_path: str) -> ScriptedClient:
    """Menu-mode scripted LLM: follows the model's own steps through the menu (reference_policy)."""
    from types import SimpleNamespace

    policy = reference_policy(model_path)

    def respond(messages, tools):
        view = SimpleNamespace(options=ev.last_menu, turn=None)
        return reply(*policy(view, {}), content="following the reference")

    return ScriptedClient(responder=respond)


# ---- end to end on cup ------------------------------------------------------------------------

def test_command_mode_cup_event_stream():
    cfg = config_from_request({"goal": {"kind": "example", "example": "cup"}, "actions": "command",
                               "model": "scripted", "images": True}, models_dir="models")
    got = []
    summary, ev = run_agent(cfg, ScriptedClient.from_model("models/cup.json"), emit=got.append)
    assert got == ev.events
    assert summary["stop_reason"] == "submitted" and summary["score"]["m2"]
    steps = check_stream(ev.events)
    assert_timeline_matches_solution(ev.events, steps)
    types = [e["type"] for e in ev.events]
    for t in ("image", "status", "llm_request", "llm_response", "action"):
        assert t in types
    fin = ev.events[-1]
    assert fin["score"]["passed"] and fin["stop_reason"] == "submitted"
    assert fin["totals"]["turns"] == types.count("llm_response")
    req = next(e for e in ev.events if e["type"] == "llm_request")
    assert req["messages"][0] == {"role": "system", "ref": "system_prompt"}
    assert [p["kind"] for p in req["messages"][1]["parts"]] == ["goal", "goal", "turn", "image"]


def test_menu_mode_cup_event_stream():
    cfg = config_from_request({"goal": {"kind": "example", "example": "cup"}, "actions": "menu",
                               "model": "scripted", "images": False}, models_dir="models")
    ev = RunEvents()
    summary, _ = run_agent(cfg, menu_client(ev, "models/cup.json"), events=ev)
    assert summary["stop_reason"] == "submitted" and summary["score"]["m2"], summary
    assert summary["custom_moves"] == 0 and summary["illegal"] == 0
    steps = check_stream(ev.events)
    assert_timeline_matches_solution(ev.events, steps)
    reqs = [e for e in ev.events if e["type"] == "llm_request"]
    assert all(r["menu"] for r in reqs)
    chosen = [e for e in ev.events if e["type"] == "action" and e["tool"] == "choose"]
    assert chosen and all(e["source"] == "menu" and "timeline_step" in e for e in chosen)
    assert any(e["type"] == "status" and e["phase"] == "menu" for e in ev.events)


def test_undo_truncates_the_live_timeline():
    replies = [reply(("valley", {"line": '{"onto":["C","A"]}', "plan": "p"})),
               reply(("valley", {"line": '{"onto":["B","D"]}', "plan": "p"})),
               reply(("undo", {"plan": "p"})),
               reply(("submit", {"plan": "p"}))]
    ev = RunEvents()
    run_episode("models/cup.json", "scripted", images=False, client=ScriptedClient(replies), events=ev, out=None,
                **NOLOG)
    steps = check_stream(ev.events)
    undo = [e for e in ev.events if e["type"] == "undo"]
    assert len(undo) == 1 and undo[0]["n"] == 1 and undo[0]["timeline_len"] == 1
    assert len(steps) == 1
    assert_timeline_matches_solution(ev.events, steps)


# ---- cancel / budget -------------------------------------------------------------------------

@pytest.mark.parametrize("actions", ["command", "menu"])
def test_cancel_mid_run(actions):
    cfg = config_from_request({"goal": {"kind": "example", "example": "cup"}, "actions": actions,
                               "model": "scripted", "images": False}, models_dir="models")
    at_turn3, release = threading.Event(), threading.Event()
    run = AgentRun(cfg, None, max_usd=None)
    inner = (ScriptedClient.from_model("models/cup.json") if actions == "command"
             else menu_client(run.events, "models/cup.json"))

    def respond(messages, tools):
        if inner.calls == 2:  # third call: hold it until the test has pressed stop
            at_turn3.set()
            assert release.wait(30)
        return inner.chat(messages, tools)

    run._client = ScriptedClient(responder=respond)
    run.start()
    assert at_turn3.wait(60)
    run.stop()
    release.set()
    assert run.wait(60)
    evs = run.events.events
    steps = check_stream(evs)
    assert evs[-1]["stop_reason"] == "stopped" and run.summary["stop_reason"] == "stopped"
    assert sum(e["type"] == "llm_response" for e in evs) == 3  # the call in flight finishes, no 4th
    assert len(steps) >= 1 and evs[-1]["score"] is not None and not evs[-1]["score"]["passed"]


def test_cancel_before_start_still_finishes_cleanly():
    cfg = config_from_request({"goal": {"kind": "text", "text": "a cup"}, "actions": "command",
                               "model": "scripted", "images": False}, models_dir="models")
    ev = RunEvents()
    ev.cancel.set()
    summary, _ = run_agent(cfg, ScriptedClient([]), events=ev)
    assert summary["stop_reason"] == "stopped"
    check_stream(ev.events)
    assert not any(e["type"] == "llm_request" for e in ev.events)


def test_run_budget_stops_before_the_next_call():
    usage = Usage(input=100, output=10, cost_usd=0.4)
    client = ScriptedClient(responder=lambda m, t: reply(("flip", {"plan": "look"}), usage=usage))
    cfg = config_from_request({"goal": {"kind": "example", "example": "cup"}, "actions": "command",
                               "model": "scripted", "images": False}, models_dir="models")
    summary, ev = run_agent(cfg, client, max_usd=1.0)
    assert summary["stop_reason"] == "budget" and client.calls == 3
    check_stream(ev.events)
    assert ev.events[-1]["totals"]["cost_usd"] == pytest.approx(1.2)
    assert any(e["type"] == "error" and e["kind"] == "budget" for e in ev.events)


# ---- goals -------------------------------------------------------------------------------------

def _texts_and_images(messages):
    texts, imgs = [], []
    for m in messages:
        c = m["content"]
        for p in c if isinstance(c, list) else [{"type": "text", "text": c}]:
            (imgs.append(p["image_url"]["url"]) if p["type"] == "image_url" else texts.append(p["text"]))
    return "\n".join(texts), imgs


def _leaks(env):
    bad = ["par ", "par)", "(par", "Target", "target outline", "target model", env.title, env.name,
           "outline AND the visible colours"]
    bad += [ln.split(":", 1)[1].strip() for ln in env.goal_text().splitlines() if ln.startswith("Target outline")]
    return bad


@pytest.mark.parametrize("actions", ["command", "menu"])
@pytest.mark.parametrize("kind", ["text", "image"])
def test_text_and_image_goals_hide_the_target(actions, kind):
    env = FoldEnv("models/cup.json")
    target_url = "data:image/png;base64," + base64.b64encode(env.target_png()).decode()
    photo = FoldEnv("models/hat.json").target_png()
    goal = (Goal.from_text("A paper drinking vessel.", check_against="models/cup.json") if kind == "text"
            else Goal.from_image(photo, text="fold this", check_against="models/cup.json"))
    client = ScriptedClient([reply(("flip", {"plan": "look at the back"}))])
    ev = RunEvents()
    run = run_episode if actions == "command" else run_episode_menu
    s = run(None, "scripted", images=True, client=client, goal=goal, events=ev, out=None, **NOLOG)
    assert client.calls == 2
    for msgs in client.seen:
        text, imgs = _texts_and_images(msgs)
        for bad in _leaks(env):
            assert bad not in text, bad
        assert target_url not in imgs
        if kind == "text":
            assert "GOAL\nA paper drinking vessel." in text
        else:
            assert "The picture shows the object to fold" in text and "fold this" in text
            assert "data:image/png;base64," + base64.b64encode(photo).decode() in imgs
        assert "/30 " in text or " of 30 " in text
    check_stream(ev.events)
    st = ev.events[0]
    assert st["goal"]["kind"] == kind and st["goal"]["target_name"] is None and st["goal"]["check_against"] == "cup"
    assert "cup" not in json.dumps(st["goal_parts"]).lower()
    assert ev.events[-1]["score"] is not None  # scored against the hidden reference
    assert s["stop_reason"] == "submitted"


def test_unscored_image_goal_without_reference():
    png = FoldEnv("models/hat.json").target_png()
    cfg = config_from_request({"goal": {"kind": "image", "image_b64": base64.b64encode(png).decode()},
                               "actions": "menu", "model": "scripted", "images": True}, models_dir="models")
    assert cfg.request["goal"]["image_b64"].startswith("<")
    summary, ev = run_agent(cfg, ScriptedClient([]))
    check_stream(ev.events)
    fin = ev.events[-1]
    assert fin["stop_reason"] == "submitted" and fin["score"] is None and fin["variants"] == {}
    assert summary["score"]["passed"] is None
    first_parts = ev.events[0]["goal_parts"]
    assert [p["kind"] for p in first_parts] == ["goal", "image"]


def test_image_goal_needs_a_picture_model():
    png = FoldEnv("models/cup.json").target_png()
    cfg = config_from_request({"goal": {"kind": "image", "image_b64": base64.b64encode(png).decode()},
                               "actions": "command", "model": "scripted"}, models_dir="models")
    summary, ev = run_agent(cfg, ScriptedClient([], supports_images=False))
    assert summary["stop_reason"] == "error" and "pictures" in summary["error"]
    check_stream(ev.events)


def test_example_goal_parts_match_the_old_goal_parts():
    from origami.agents.llm_agent import goal_parts

    env = FoldEnv("models/hat.json")
    g = Goal.example("models/hat.json")
    assert g.parts(g.make_env(), True) == goal_parts(env, True, env.target_png())
    assert not g.hides_target() and Goal.from_text("x").hides_target()


# ---- request / plan / keys ---------------------------------------------------------------------

@pytest.mark.parametrize("req, msg", [
    ({"goal": {"kind": "example", "example": "../cup"}, "model": "m"}, "bad example"),
    ({"goal": {"kind": "example", "example": "nope"}, "model": "m"}, "unknown example"),
    ({"goal": {"kind": "text", "text": "  "}, "model": "m"}, "needs some text"),
    ({"goal": {"kind": "example", "example": "cup"}, "model": "m", "actions": "decision"}, "decision"),
    ({"goal": {"kind": "example", "example": "cup"}, "model": "m", "max_turns": 99}, "capped"),
    ({"goal": {"kind": "example", "example": "cup"}}, "model"),
    ({"goal": {"kind": "image", "image_b64": "%%%"}, "model": "m"}, "base64"),
    ({"goal": {"kind": "image", "image_b64": base64.b64encode(b"GIF89a").decode()}, "model": "m"}, "PNG"),
])
def test_bad_requests_are_refused(req, msg):
    with pytest.raises(ValueError, match=msg):
        config_from_request(req, models_dir="models", max_turns_cap=40)


def test_planner_model_and_key_hygiene():
    key = "sk-test-SECRET-123"
    cfg = config_from_request({"goal": {"kind": "text", "text": "a paper cup", "check_against": "cup"},
                               "actions": "command", "model": "scripted", "images": False,
                               "plan": {"kind": "model", "model": "planner/x"},
                               "user_key": {"provider": "openrouter", "key": key}}, models_dir="models")
    planner = ScriptedClient([reply(content="1. Fold the bottom corner up.\n2. Fold the sides in.",
                                    usage=Usage(input=50, output=20, cost_usd=0.01))])
    client = ScriptedClient([reply(("submit", {"plan": "done"}), usage=Usage(input=10, output=5, cost_usd=0.0))])
    summary, ev = run_agent(cfg, client, planner_client=planner)
    check_stream(ev.events)
    plan = [e for e in ev.events if e["type"] == "plan"]
    assert len(plan) == 1 and plan[0]["source"] == "model" and plan[0]["text"].startswith("1. Fold")
    assert "a paper cup" in json.dumps(planner.seen[0]) and "Paper cup (traditional)" not in json.dumps(planner.seen[0])
    req = next(e for e in ev.events if e["type"] == "llm_request")
    assert {"kind": "plan", "ref": "plan"} in req["messages"][1]["parts"]
    assert "Fold the sides in" in json.dumps(client.seen[0])
    assert ev.events[-1]["totals"]["cost_usd"] == pytest.approx(0.01)
    assert key not in json.dumps(ev.events)
    assert ev.events[0]["config"]["user_key"]["key"] == "<hidden>"


def test_written_plan_is_sent_once_as_a_plan_event(tmp_path):
    ev = RunEvents()
    run_episode_menu("models/cup.json", "scripted", images=False, client=ScriptedClient([]), events=ev,
                     plan_text="1. Fold in half.", out=tmp_path / "run", **NOLOG)
    check_stream(ev.events)
    plans = [e for e in ev.events if e["type"] == "plan"]
    assert len(plans) == 1 and plans[0] == {**plans[0], "source": "text", "text": "1. Fold in half."}
    assert (tmp_path / "run" / "solution.json").exists() and (tmp_path / "run" / "score.json").exists()
