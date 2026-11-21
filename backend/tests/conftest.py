import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from foldr_backend.app import create_app
from foldr_backend.config import DEFAULT_MODELS_DIR, ModelCfg, Settings
from origami.llm import DEFAULT_PROVIDERS, Price, ScriptedClient, Usage, make_client, register_provider
from origami.llm.types import as_result

CUP = str(DEFAULT_MODELS_DIR / "cup.json")
KEY_ENV = "FOLDR_TEST_KEY"
CUP_REQ = {"goal": {"kind": "example", "example": "cup"}, "actions": "command", "model": "demo", "images": False}


class CostlyClient(ScriptedClient):
    """The cup script, but every reply reports a cost, like a paid provider would."""

    cost = 0.6

    def chat(self, messages, tools=None, **kw):
        res = as_result(super().chat(messages, tools, **kw))
        res.usage = Usage(input=100, output=10, cost_usd=self.cost)
        return res


def _costly(model_cfg, provider_cfg, api_key, log):
    return CostlyClient.from_model(model_cfg["script"], model=model_cfg["id"])


register_provider("costly", _costly)


def make_settings(**kw) -> Settings:
    providers = {pid: {**p, "id": pid} for pid, p in DEFAULT_PROVIDERS.items()}
    providers["paid"] = {"id": "paid", "kind": "costly", "api_key_env": KEY_ENV}
    models = [
        ModelCfg(id="demo", provider="scripted", label="Demo", key="demo", raw={"id": "demo", "script": CUP},
                 default=True),
        ModelCfg(id="paid-model", provider="paid", label="Paid", key="paid-model", price=Price(1.0, 2.0),
                 raw={"id": "paid-model", "script": CUP, "price": {"input": 1.0, "output": 2.0}}),
        ModelCfg(id="unpriced", provider="paid", label="Unpriced", key="unpriced",
                 raw={"id": "unpriced", "script": CUP}),
    ]
    base = dict(enabled=True, providers=providers, models=models, ip_cooldown_s=0.0,
                cors_origins=["https://me.github.io"], per_run_usd=1.0, daily_usd=3.0, max_turns=40)
    base.update(kw)
    return Settings(**base)


@pytest.fixture(autouse=True)
def _key_env(monkeypatch):
    monkeypatch.setenv(KEY_ENV, "owner-key-xyz")


@pytest.fixture
def make_client_for():
    def build(factory=None, **kw):
        return TestClient(create_app(make_settings(**kw), client_factory=factory or make_client))

    return build


def wait_finished(client, run_id, timeout=60):
    import time

    t0 = time.time()
    while time.time() - t0 < timeout:
        info = client.get(f"/api/runs/{run_id}").json()
        if info["status"] == "finished":
            return info
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def sse_events(client, run_id, headers=None):
    import json

    out = []
    with client.stream("GET", f"/api/runs/{run_id}/events", headers=headers or {}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if line.startswith("data: "):
                out.append(json.loads(line[6:]))
    return out
