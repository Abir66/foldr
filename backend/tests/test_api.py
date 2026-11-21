import base64
import io
import json
import logging
import threading

import pytest

from conftest import CUP_REQ, KEY_ENV, CostlyClient, make_settings, sse_events, wait_finished
from foldr_backend.config import ConfigError, load_settings
from origami.llm import ProviderError, ScriptedClient


# ---- AI off ------------------------------------------------------------------------------

def test_ai_off_config_and_no_spending(make_client_for):
    c = make_client_for(enabled=False, allow_user_keys=True)
    cfg = c.get("/api/config").json()
    assert cfg["ai"]["enabled"] is False and cfg["ai"]["allow_user_keys"] is False
    assert cfg["models"] == [] and cfg["providers"] == [] and "cup" in cfg["examples"]
    assert c.get("/api/health").json()["ok"] is True
    r = c.post("/api/runs", json=CUP_REQ)
    assert r.status_code == 403
    r = c.post("/api/runs", json={**CUP_REQ, "user_key": {"provider": "paid", "key": "sk-abcdef"}})
    assert r.status_code == 403


def test_no_config_file_means_off(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = load_settings(environ={})
    assert s.enabled is False and s.models == []


def test_env_overrides(tmp_path):
    f = tmp_path / "c.toml"
    f.write_text('[ai]\nenabled = true\nper_run_usd = 2\ncors_origins = ["https://a.github.io/"]\n')
    s = load_settings(f, environ={"FOLDR_AI_ENABLED": "0", "FOLDR_DAILY_USD": "7",
                                  "FOLDR_CORS_ORIGINS": "https://b.example, https://a.github.io"})
    assert s.enabled is False and s.daily_usd == 7.0 and s.per_run_usd == 2.0
    assert s.cors_origins == ["https://a.github.io", "https://b.example"]
    assert load_settings(f, environ={"FOLDR_AI_ENABLED": "yes"}).enabled is True


def test_example_config_loads():
    from foldr_backend.config import BACKEND_DIR

    s = load_settings(BACKEND_DIR / "config.example.toml", environ={})
    assert s.models and any(m.key == "demo" for m in s.models)
    assert all(m.provider in s.providers for m in s.models)
    assert any(o.startswith("https://") for o in s.cors_origins)


def test_bad_config(tmp_path):
    f = tmp_path / "c.toml"
    f.write_text('[[models]]\nid = "x"\nprovider = "nope"\n')
    with pytest.raises(ConfigError):
        load_settings(f, environ={})


def test_config_hides_unusable_models(make_client_for, monkeypatch):
    c = make_client_for()
    ids = {m["id"] for m in c.get("/api/config").json()["models"]}
    assert ids == {"demo", "paid-model"}  # "unpriced" can't be budgeted
    monkeypatch.delenv(KEY_ENV)
    ids = {m["id"] for m in c.get("/api/config").json()["models"]}
    assert ids == {"demo"}
    r = c.post("/api/runs", json={**CUP_REQ, "model": "paid-model"})
    assert r.status_code == 422 and KEY_ENV in r.json()["detail"]
    c2 = make_client_for(allow_user_keys=True)
    models = {m["id"]: m for m in c2.get("/api/config").json()["models"]}
    assert models["paid-model"]["user_key"] and not models["paid-model"]["owner_key"]
    assert "demo" in models and not models["demo"]["user_key"]
    assert KEY_ENV not in json.dumps(c2.get("/api/config").json())


# ---- runs over SSE -------------------------------------------------------------------------

def test_scripted_cup_run_streams_to_finish(make_client_for):
    c = make_client_for()
    r = c.post("/api/runs", json=CUP_REQ)
    assert r.status_code == 202
    rid = r.json()["run_id"]
    events = sse_events(c, rid)
    assert [e["seq"] for e in events] == list(range(len(events)))
    assert events[0]["type"] == "run_started" and events[-1]["type"] == "run_finished"
    assert events[-1]["stop_reason"] == "submitted" and events[-1]["score"]["passed"]
    assert any(e["type"] == "action" and e["ok"] for e in events)
    info = c.get(f"/api/runs/{rid}").json()
    assert info["status"] == "finished" and info["summary"]["stop_reason"] == "submitted"
    assert info["events"] == events
    # reconnect: only events after Last-Event-ID, in order
    tail = sse_events(c, rid, {"Last-Event-ID": "3"})
    assert [e["seq"] for e in tail] == list(range(4, len(events)))
    assert c.get("/api/runs/nope").status_code == 404


def test_menu_mode_runs(make_client_for):
    c = make_client_for()
    rid = c.post("/api/runs", json={**CUP_REQ, "actions": "menu"}).json()["run_id"]
    info = wait_finished(c, rid)
    assert info["events"][0]["type"] == "run_started" and info["events"][-1]["type"] == "run_finished"


def test_stop(make_client_for):
    inner = ScriptedClient.from_model(make_settings().models[0].raw["script"])
    at_turn2, release = threading.Event(), threading.Event()

    def respond(messages, tools):
        if inner.calls == 1:
            at_turn2.set()
            assert release.wait(30)
        return inner.chat(messages, tools)

    c = make_client_for(lambda *a, **k: ScriptedClient(responder=respond))
    rid = c.post("/api/runs", json=CUP_REQ).json()["run_id"]
    assert at_turn2.wait(30)
    assert c.post(f"/api/runs/{rid}/stop").json()["status"] == "running"
    release.set()
    info = wait_finished(c, rid)
    assert info["summary"]["stop_reason"] == "stopped"
    assert c.post("/api/runs/nope/stop").status_code == 404


def test_one_run_at_a_time_and_cooldown(make_client_for):
    release = threading.Event()
    inner = ScriptedClient.from_model(make_settings().models[0].raw["script"])

    def respond(messages, tools):
        assert release.wait(30)
        return inner.chat(messages, tools)

    c = make_client_for(lambda *a, **k: ScriptedClient(responder=respond))
    rid = c.post("/api/runs", json=CUP_REQ).json()["run_id"]
    assert c.post("/api/runs", json=CUP_REQ).status_code == 409
    c.post(f"/api/runs/{rid}/stop")
    release.set()
    wait_finished(c, rid)

    c = make_client_for(ip_cooldown_s=60.0)
    assert c.post("/api/runs", json=CUP_REQ).status_code == 202
    assert c.post("/api/runs", json=CUP_REQ).status_code == 429


# ---- budgets -------------------------------------------------------------------------------

def test_per_run_cap_stops_run(make_client_for):
    c = make_client_for(per_run_usd=1.0)
    rid = c.post("/api/runs", json={**CUP_REQ, "model": "paid-model"}).json()["run_id"]
    info = wait_finished(c, rid)
    fin = info["events"][-1]
    assert fin["stop_reason"] == "budget"
    calls = sum(e["type"] == "llm_response" for e in info["events"])
    assert calls == 2  # 0.6 then 1.2 >= 1.0: the third call never happens
    assert fin["totals"]["cost_usd"] == pytest.approx(1.2)


def test_daily_cap(make_client_for):
    c = make_client_for(per_run_usd=1.0, daily_usd=1.5)
    assert c.get("/api/config").json()["ai"]["daily_left_usd"] == 1.5
    rid = c.post("/api/runs", json={**CUP_REQ, "model": "paid-model"}).json()["run_id"]
    wait_finished(c, rid)  # spends 1.2
    assert c.get("/api/config").json()["ai"]["daily_left_usd"] == pytest.approx(0.3)
    rid = c.post("/api/runs", json={**CUP_REQ, "model": "paid-model"}).json()["run_id"]
    info = wait_finished(c, rid)  # first call 0.6 would pass the 1.5 total: stops before the second
    assert info["events"][-1]["stop_reason"] in ("daily_limit", "budget")
    assert sum(e["type"] == "llm_response" for e in info["events"]) <= 1
    r = c.post("/api/runs", json={**CUP_REQ, "model": "paid-model"})
    assert r.status_code == 402
    assert c.post("/api/runs", json=CUP_REQ).status_code == 402  # the demo model too: one budget switch
    

def test_unknown_cost_stops_owner_run(make_client_for):
    class NoCost(CostlyClient):
        cost = None

    c = make_client_for(lambda m, p, k, log=None: NoCost.from_model(m["script"]))
    rid = c.post("/api/runs", json={**CUP_REQ, "model": "paid-model"}).json()["run_id"]
    assert wait_finished(c, rid)["events"][-1]["stop_reason"] == "budget"


# ---- validation ----------------------------------------------------------------------------

@pytest.mark.parametrize("patch", [
    {"model": "nope"}, {"model": None}, {"max_turns": 41}, {"actions": "decision"},
    {"goal": {"kind": "example", "example": "../etc"}}, {"goal": {"kind": "text", "text": ""}},
    {"model": "unpriced"},
])
def test_bad_requests_422(make_client_for, patch):
    c = make_client_for()
    assert c.post("/api/runs", json={**CUP_REQ, **patch}).status_code == 422


def test_bad_body(make_client_for):
    c = make_client_for()
    assert c.post("/api/runs", content=b"{nope", headers={"content-type": "application/json"}).status_code == 422
    assert c.post("/api/runs", json=[1]).status_code == 422
    assert c.post("/api/runs", content=b"x" * 4_100_000).status_code == 413


def _png(size, fmt="PNG"):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, (200, 30, 30)).save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def test_image_goal_resized_and_text_goal_capped(make_client_for):
    def factory(m, p, k, log=None):
        return ScriptedClient()

    c = make_client_for(factory, max_turns=7)
    big = _png((2000, 1500), "JPEG")
    r = c.post("/api/runs", json={**CUP_REQ, "goal": {"kind": "image", "image_b64": big}})
    assert r.status_code == 202
    info = wait_finished(c, r.json()["run_id"])
    start = info["events"][0]
    pic = next(p for p in start["goal_parts"] if p.get("kind") == "image")
    from PIL import Image

    im = Image.open(io.BytesIO(base64.b64decode(pic["data"].split(",", 1)[1])))
    assert max(im.size) <= 512  # events downscale again; the loop got <= 1024
    assert start["limits"]["max_turns"] == 7  # text/image default (30) is capped by the server
    assert "image_b64" not in json.dumps(start["config"]) or "base64 chars" in json.dumps(start["config"])
    assert c.post("/api/runs", json={**CUP_REQ, "goal": {"kind": "image", "image_b64": "AAAA"}}).status_code == 422
    huge = base64.b64encode(b"\0" * 2_100_000).decode()
    assert c.post("/api/runs", json={**CUP_REQ, "goal": {"kind": "image", "image_b64": huge}}).status_code == 422


# ---- user keys -----------------------------------------------------------------------------

USER_KEY = "sk-user-SECRET-9f8e7d6c5b4a"
HTTPS = {"x-forwarded-proto": "https"}


def test_user_key_rules(make_client_for):
    body = {**CUP_REQ, "model": "paid-model", "user_key": {"provider": "paid", "key": USER_KEY}}
    off = make_client_for(allow_user_keys=False)
    assert off.post("/api/runs", json=body, headers=HTTPS).status_code == 403
    c = make_client_for(allow_user_keys=True)
    assert c.post("/api/runs", json=body).status_code == 403  # plain http from a non-local host
    assert c.post("/api/runs", json={**body, "user_key": {"provider": "paid"}}, headers=HTTPS).status_code == 422
    assert c.post("/api/runs", json={**body, "model": "demo"}, headers=HTTPS).status_code == 422  # not its provider
    assert c.post("/api/runs", json={**body, "user_key": {"provider": "scripted", "key": USER_KEY}},
                  headers=HTTPS).status_code == 422  # provider takes no key
    local = make_client_for(allow_user_keys=True)
    assert local.base_url  # localhost over plain http is fine:
    assert local.post("/api/runs", json=body, headers={"host": "localhost:8000"}).status_code == 202


def test_user_key_used_once_never_stored_or_logged(make_client_for, caplog, monkeypatch, capfd):
    caplog.set_level(logging.DEBUG)
    keys_seen = []

    def factory(model_cfg, provider_cfg, api_key, log=None):
        keys_seen.append(api_key)

        class Leaky(CostlyClient):
            def chat(self, messages, tools=None, **kw):
                if len(self.seen) == 1:  # provider error text that echoes the key back
                    self.calls += 1
                    self.seen.append(messages)
                    raise ProviderError(f"401 invalid key {api_key} for provider")
                return super().chat(messages, tools, **kw)

        return Leaky.from_model(model_cfg["script"])

    monkeypatch.delenv(KEY_ENV)
    c = make_client_for(factory, allow_user_keys=True, daily_usd=1.0)
    body = {**CUP_REQ, "model": "paid-model", "user_key": {"provider": "paid", "key": USER_KEY}}
    r = c.post("/api/runs", json=body, headers=HTTPS)
    assert r.status_code == 202, r.text
    rid = r.json()["run_id"]
    info = wait_finished(c, rid)
    assert keys_seen == [USER_KEY]
    blob = json.dumps(info) + json.dumps(c.get("/api/config").json()) + "".join(
        json.dumps(e) for e in sse_events(c, rid))
    assert USER_KEY not in blob and "SECRET" not in blob
    assert any(e["type"] == "error" for e in info["events"])  # the leak attempt happened
    assert info["events"][0]["config"]["user_key"]["key"] == "<hidden>"
    out, err = capfd.readouterr()
    assert USER_KEY not in caplog.text + out + err
    run = c.app.state.manager.get(rid)
    assert run.secret is None and run.agent._client is None
    assert c.app.state.manager.daily.spent() == 0.0  # their money, not the owner's budget
    assert c.get("/api/config").json()["ai"]["daily_left_usd"] == 1.0


def test_user_key_run_with_owner_ai_budget_spent(make_client_for, monkeypatch):
    monkeypatch.delenv(KEY_ENV)
    c = make_client_for(allow_user_keys=True, daily_usd=0.0)
    assert c.post("/api/runs", json=CUP_REQ).status_code == 402
    body = {**CUP_REQ, "model": "paid-model", "user_key": {"provider": "paid", "key": USER_KEY}}
    assert c.post("/api/runs", json=body, headers=HTTPS).status_code == 202


# ---- CORS and tools ------------------------------------------------------------------------

def test_cors(make_client_for):
    c = make_client_for()
    for origin, ok in [("https://me.github.io", True), ("http://localhost:5173", True),
                       ("https://evil.example", False)]:
        r = c.get("/api/config", headers={"Origin": origin})
        assert (r.headers.get("access-control-allow-origin") == origin) is ok
    pre = c.options("/api/runs", headers={"Origin": "https://me.github.io", "Access-Control-Request-Method": "POST",
                                          "Access-Control-Request-Headers": "content-type"})
    assert pre.status_code == 200


def test_timeline_endpoint(make_client_for):
    from foldr_backend.config import DEFAULT_MODELS_DIR

    c = make_client_for(enabled=False)  # model tools never cost money, so they work with AI off
    model = json.loads((DEFAULT_MODELS_DIR / "cup.json").read_text())
    r = c.post("/api/timeline", json={"model": model})
    assert r.status_code == 200 and r.json()["timeline"]["steps"]
    r = c.post("/api/export", json={"model": model, "kinds": ["fold"]})
    assert r.status_code == 200 and r.json()["files"]
    assert c.post("/api/timeline", json={"model": 3}).json()["timeline"] is None
