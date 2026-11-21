"""origami.llm: provider adapters against recorded-style HTTP fixtures (fake transport, no network)."""

import io
import json
import urllib.error

import pytest

from origami.llm import (AnthropicClient, ChatResult, NoCredit, OpenAICompatClient, OpenRouterClient, Price,
                         ProviderError, RateLimited, ScriptedClient, Usage, as_result, compute_cost, make_client,
                         register_provider, resolve_api_key)
from origami.llm.anthropic import to_anthropic_messages
from origami.llm.config import DEFAULT_PROVIDERS, PROVIDER_KINDS

KEY = "sk-test-SECRET-123456"
PNG_URL = "data:image/png;base64,iVBORw0KGgo="
TOOLS = [{"type": "function", "function": {"name": "choose", "description": "Pick an option.", "parameters": {
    "type": "object", "properties": {"option": {"type": "integer"}}, "required": ["option"]}}}]
MESSAGES = [
    {"role": "system", "content": "You are an origami agent."},
    {"role": "user", "content": [{"type": "text", "text": "GOAL\na cup"}, {"type": "image_url", "image_url": {"url": PNG_URL}}]},
    {"role": "assistant", "content": "Folding.", "tool_calls": [
        {"id": "toolu_1", "type": "function", "function": {"name": "choose", "arguments": '{"option": 2}'}},
        {"id": "toolu_2", "type": "function", "function": {"name": "choose", "arguments": '{"option": 3}'}}]},
    {"role": "tool", "tool_call_id": "toolu_1", "content": "OK"},
    {"role": "tool", "tool_call_id": "toolu_2", "content": "rejected"},
    {"role": "user", "content": [{"type": "text", "text": "=== TURN 2"}]},
]

OPENROUTER_OK = {
    "id": "gen-1", "provider": "Anthropic",
    "choices": [{"finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None, "reasoning": "The triangle first.",
        "tool_calls": [{"id": "c0", "type": "function", "function": {"name": "choose", "arguments": '{"option": 1}'}}]}}],
    "usage": {"prompt_tokens": 1000, "completion_tokens": 50, "prompt_tokens_details": {"cached_tokens": 400},
              "completion_tokens_details": {"reasoning_tokens": 20}, "cost": 0.0123}}

ANTHROPIC_OK = {
    "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
    "content": [{"type": "thinking", "thinking": "Corner C onto A.", "signature": "sig"},
                {"type": "text", "text": "I'll fold."},
                {"type": "tool_use", "id": "toolu_9", "name": "choose", "input": {"option": 4}}],
    "stop_reason": "tool_use", "stop_details": None,
    "usage": {"input_tokens": 600, "cache_read_input_tokens": 400, "cache_creation_input_tokens": 0,
              "output_tokens": 80}}


class FakeTransport:
    """Replays (status, headers, body) tuples or raises exceptions; records requests."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, url, data, headers, timeout):
        self.requests.append({"url": url, "body": json.loads(data), "headers": headers})
        r = self.responses.pop(0)
        if isinstance(r, BaseException):
            raise r
        status, h, body = r
        return status, h, json.dumps(body).encode() if not isinstance(body, (bytes, str)) else (
            body.encode() if isinstance(body, str) else body)


class Sleeps(list):
    def __call__(self, s):
        self.append(s)


def _oa(transport, **kw):
    kw.setdefault("dialect", "openrouter")
    return OpenAICompatClient("anthropic/claude-opus-5.5", base_url="https://openrouter.ai/api/v1", api_key=KEY,
                              transport=transport, sleep=Sleeps(), log=lambda *_: None, **kw)


def _an(transport, **kw):
    return AnthropicClient("claude-opus-5-5", api_key=KEY, transport=transport, sleep=Sleeps(),
                           log=lambda *_: None, **kw)


# ---- OpenAI-compatible ------------------------------------------------------------------------

def test_openai_compat_request_and_parse():
    t = FakeTransport((200, {}, OPENROUTER_OK))
    c = _oa(t, reports_cost=True, extra_headers={"X-Title": "foldr"}, extra_body={"usage": {"include": True}})
    r = c.chat(MESSAGES, TOOLS, reasoning="high")
    req = t.requests[0]
    assert req["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert req["headers"]["Authorization"] == f"Bearer {KEY}" and req["headers"]["X-Title"] == "foldr"
    b = req["body"]
    assert b["messages"] == MESSAGES and b["tools"] == TOOLS and b["tool_choice"] == "auto"
    assert b["reasoning"] == {"effort": "high"} and b["usage"] == {"include": True} and b["temperature"] == 0.3
    assert r.finish_reason == "tool_calls" and r.reasoning == "The triangle first." and r.provider == "Anthropic"
    assert r.tool_calls[0]["function"]["name"] == "choose"
    assert r.usage == Usage(input=1000, output=50, cached=400, reasoning=20, cost_usd=0.0123)
    assert r.message() == {"content": None, "reasoning": "The triangle first.", "tool_calls": r.tool_calls}


@pytest.mark.parametrize("dialect,reasoning,expect", [
    ("openrouter", "off", {"reasoning": {"enabled": False}}),
    ("openrouter", "medium", {"reasoning": {"effort": "medium"}}),
    ("openai", "xhigh", {"reasoning_effort": "high"}),
    ("openai", "off", {}),
    ("none", "high", {}),
])
def test_reasoning_dialects(dialect, reasoning, expect):
    c = _oa(FakeTransport(), dialect=dialect)
    b = c.build_body(MESSAGES, None, reasoning=reasoning)
    got = {k: b[k] for k in ("reasoning", "reasoning_effort") if k in b}
    assert got == expect and "tools" not in b


def test_model_without_reasoning_sends_nothing():
    c = _oa(FakeTransport(), supports_reasoning=False)
    assert "reasoning" not in c.build_body(MESSAGES, None, reasoning="high")


def test_cost_from_price_when_not_reported():
    t = FakeTransport((200, {}, OPENROUTER_OK))
    r = _oa(t, reports_cost=False, price=Price(input=2.0, output=10.0, cached_input=0.2)).chat(MESSAGES, TOOLS)
    assert r.usage.cost_usd == pytest.approx((600 * 2.0 + 400 * 0.2 + 50 * 10.0) / 1e6)
    r2 = _oa(FakeTransport((200, {}, OPENROUTER_OK)), reports_cost=False).chat(MESSAGES, TOOLS)
    assert r2.usage.cost_usd is None


def test_openrouter_error_inside_200_is_retried_then_ok():
    t = FakeTransport((200, {}, {"error": {"code": 502, "message": "upstream"}}), (200, {}, OPENROUTER_OK))
    c = _oa(t)
    assert c.chat(MESSAGES, TOOLS).finish_reason == "tool_calls" and c.calls == 2


# ---- retries / errors -------------------------------------------------------------------------

def test_429_honours_retry_after_then_succeeds():
    t = FakeTransport((429, {"retry-after": "7"}, {"error": {"type": "rate_limit_error"}}),
                      (529, {}, {"error": {"type": "overloaded_error"}}), (200, {}, ANTHROPIC_OK))
    c = _an(t)
    r = c.chat(MESSAGES, TOOLS)
    assert r.finish_reason == "tool_calls" and c.calls == 3
    assert c._sleep == [7.0, 16.0]  # Retry-After, then backoff (8s doubled)


def test_429_exhausted_raises_rate_limited():
    t = FakeTransport(*[(429, {}, {"error": "slow down"})] * 3)
    c = _an(t, max_retries=2)
    with pytest.raises(RateLimited):
        c.chat(MESSAGES, TOOLS)
    assert c.calls == 3


@pytest.mark.parametrize("status,body", [
    (402, {"error": {"type": "billing_error", "message": "no credit"}}),
    (429, {"error": {"type": "rate_limit_error", "details": {"error_code": "enforced_spend_limit_reached"}}}),
])
def test_no_credit_stops_without_retry(status, body):
    t = FakeTransport((status, {}, body))
    c = _an(t)
    with pytest.raises(NoCredit):
        c.chat(MESSAGES, TOOLS)
    assert c.calls == 1 and c._sleep == []


def test_4xx_error_text_never_contains_key():
    t = FakeTransport((401, {}, {"error": {"message": f"invalid x-api-key {KEY}"}}))
    with pytest.raises(ProviderError) as ei:
        _an(t).chat(MESSAGES, TOOLS)
    assert KEY not in str(ei.value) and "***" in str(ei.value)


def test_network_errors_retry():
    t = FakeTransport(urllib.error.URLError("down"), TimeoutError(), (200, {}, OPENROUTER_OK))
    c = _oa(t)
    assert c.chat(MESSAGES, TOOLS).tool_calls and c.calls == 3


def test_real_urllib_path(monkeypatch):
    calls = []

    class Resp(io.BytesIO):
        status = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout):
        calls.append(req)
        if len(calls) == 1:
            raise urllib.error.HTTPError(req.full_url, 503, "busy", {"Retry-After": "1"},
                                         io.BytesIO(b'{"error":"busy"}'))
        return Resp(json.dumps(ANTHROPIC_OK).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    c = AnthropicClient("claude-opus-5-5", api_key=KEY, sleep=lambda s: None, log=lambda *_: None)
    r = c.chat(MESSAGES, TOOLS)
    assert r.content == "I'll fold." and len(calls) == 2
    assert calls[1].full_url == "https://api.anthropic.com/v1/messages"
    assert calls[1].get_header("X-api-key") == KEY and calls[1].get_header("Anthropic-version") == "2026-10-21"


# ---- Anthropic conversion ---------------------------------------------------------------------

def test_anthropic_request_conversion():
    t = FakeTransport((200, {}, ANTHROPIC_OK))
    _an(t).chat(MESSAGES, TOOLS, reasoning="high", temperature=0.3)
    req = t.requests[0]
    h, b = req["headers"], req["body"]
    assert h["x-api-key"] == KEY and h["anthropic-version"] == "2026-10-21" and "Authorization" not in h
    assert b["system"] == "You are an origami agent."
    assert "temperature" not in b
    assert b["thinking"] == {"type": "adaptive", "display": "summarized"} and b["output_config"] == {"effort": "high"}
    assert b["tools"] == [{"name": "choose", "description": "Pick an option.",
                           "input_schema": TOOLS[0]["function"]["parameters"]}]
    m = b["messages"]
    assert [x["role"] for x in m] == ["user", "assistant", "user"]
    assert m[0]["content"][1] == {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                              "data": "iVBORw0KGgo="}}
    assert m[1]["content"][0] == {"type": "text", "text": "Folding."}
    assert m[1]["content"][1] == {"type": "tool_use", "id": "toolu_1", "name": "choose", "input": {"option": 2}}
    # both tool results in ONE user message, before the next turn's text
    assert [blk["type"] for blk in m[2]["content"]] == ["tool_result", "tool_result", "text"]
    assert m[2]["content"][1] == {"type": "tool_result", "tool_use_id": "toolu_2", "content": "rejected"}


@pytest.mark.parametrize("mode,reasoning,expect_thinking,expect_effort", [
    ("adaptive", "off", None, "low"),
    ("adaptive", None, None, None),
    ("budget", "medium", {"type": "enabled", "budget_tokens": 6000}, None),
    ("budget", "off", None, None),
    ("none", "high", None, None),
])
def test_anthropic_thinking_modes(mode, reasoning, expect_thinking, expect_effort):
    b = _an(FakeTransport(), thinking=mode).build_body(MESSAGES, None, reasoning=reasoning)
    assert b.get("thinking") == expect_thinking
    assert (b.get("output_config") or {}).get("effort") == expect_effort
    if expect_thinking and expect_thinking.get("budget_tokens"):
        assert b["max_tokens"] > expect_thinking["budget_tokens"]


def test_anthropic_response_parse_and_cost():
    t = FakeTransport((200, {}, ANTHROPIC_OK))
    r = _an(t, price=Price(input=4.0, output=20.0, cached_input=0.2)).chat(MESSAGES, TOOLS)
    assert r.content == "I'll fold." and r.reasoning == "Corner C onto A." and r.finish_reason == "tool_calls"
    assert r.tool_calls == [{"id": "toolu_9", "type": "function",
                             "function": {"name": "choose", "arguments": '{"option": 4}'}}]
    assert (r.usage.input, r.usage.cached, r.usage.output) == (1000, 400, 80)
    assert r.usage.cost_usd == pytest.approx((600 * 4.0 + 400 * 0.2 + 80 * 20.0) / 1e6)
    assert r.provider == "anthropic"


def test_anthropic_stop_reasons():
    for sr, want in [("end_turn", "stop"), ("max_tokens", "length"), ("refusal", "error"), ("tool_use", "tool_calls")]:
        out = {**ANTHROPIC_OK, "stop_reason": sr, "content": [{"type": "text", "text": "x"}]}
        assert _an(FakeTransport((200, {}, out))).chat(MESSAGES).finish_reason == want


def test_anthropic_edge_messages():
    sys, m = to_anthropic_messages([{"role": "assistant", "content": ""},
                                    {"role": "user", "content": [{"type": "image_url",
                                                                  "image_url": {"url": "https://x/y.png"}}]}])
    assert sys is None and m[0]["role"] == "user" and m[1]["content"] == [{"type": "text", "text": "(empty reply)"}]
    assert m[2]["content"][0]["source"] == {"type": "url", "url": "https://x/y.png"}


# ---- config / keys ----------------------------------------------------------------------------

def test_make_client_from_config(monkeypatch):
    pcfg = {"id": "openrouter", **DEFAULT_PROVIDERS["openrouter"]}
    c = make_client({"id": "anthropic/claude-opus-5.5", "images": True, "price": {"input": 1, "output": 2}}, pcfg, KEY)
    assert isinstance(c, OpenAICompatClient) and c.dialect == "openrouter" and c.reports_cost and c.price.input == 1
    a = make_client({"id": "claude-haiku-4-5", "images": False, "thinking": "budget"},
                    {"id": "anthropic", "kind": "anthropic", "api_key_env": "ANTHROPIC_API_KEY"}, KEY)
    assert isinstance(a, AnthropicClient) and a.thinking == "budget" and not a.supports_images
    o = make_client({"id": "llama3"}, {"kind": "openai", "base_url": "http://localhost:11434/v1", "api_key_env": ""})
    assert o.dialect == "none" and o._key is None
    for c in (c, a):
        assert KEY not in repr(c) and KEY not in str(c)
    with pytest.raises(ProviderError) as ei:
        make_client({"id": "m"}, {"kind": "anthropic", "api_key_env": "ANTHROPIC_API_KEY"}, None)
    assert "ANTHROPIC_API_KEY" in str(ei.value)
    with pytest.raises(ProviderError):
        make_client({"id": "m"}, {"kind": "nope"}, KEY)


def test_register_provider(monkeypatch):
    monkeypatch.setitem(PROVIDER_KINDS, "echo", None)
    register_provider("echo", lambda m, p, k, log: ScriptedClient(model=m["id"]))
    c = make_client({"id": "e1"}, {"kind": "echo"})
    assert isinstance(c, ScriptedClient) and c.model == "e1"


def test_resolve_api_key(tmp_path):
    env = tmp_path / ".env"
    env.write_text('export ANTHROPIC_API_KEY="from-file"\nOTHER=1\n')
    assert resolve_api_key("ANTHROPIC_API_KEY", environ={}, dotenv_paths=[env]) == "from-file"
    assert resolve_api_key("ANTHROPIC_API_KEY", environ={"ANTHROPIC_API_KEY": " env "}, dotenv_paths=[env]) == "env"
    assert resolve_api_key("MISSING", environ={}, dotenv_paths=[env]) is None
    assert resolve_api_key("", environ={}) is None


def test_openrouter_client_keeps_old_signature(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    c = OpenRouterClient("m", reasoning_tokens=3000, reasoning_effort=None)
    assert c.build_body([], None)["reasoning"] == {"max_tokens": 3000}
    assert KEY not in repr(c)


# ---- scripted + agent loops -------------------------------------------------------------------

def test_as_result_accepts_old_shapes():
    r = as_result(OPENROUTER_OK)
    assert isinstance(r, ChatResult) and r.usage.cost_usd == 0.0123
    assert as_result({"content": "hi"}).content == "hi"
    assert compute_cost(Usage(input=10, output=0), None) is None


def test_scripted_command_and_menu_runs_and_replay(tmp_path):
    from pathlib import Path

    from origami.agents.llm_agent import run_episode
    from origami.agents.menu_agent import run_episode_menu

    s = run_episode("models/cup.json", "scripted", images=False, out_root=tmp_path, log=lambda *_: None,
                    client=ScriptedClient.from_model("models/cup.json"))
    assert s["stop_reason"] == "submitted" and s["score"]["m2"]
    replay = ScriptedClient.from_turns_jsonl(Path(s["dir"]) / "turns.jsonl")
    s2 = run_episode("models/cup.json", "scripted", images=False, out_root=tmp_path, log=lambda *_: None, client=replay)
    assert s2["score"] == s["score"]
    sol1 = json.loads((Path(s["dir"]) / "solution.json").read_text())["steps"]
    sol2 = json.loads((Path(s2["dir"]) / "solution.json").read_text())["steps"]
    assert sol1 == sol2
    m = run_episode_menu("models/cup.json", "scripted", images=False, out_root=tmp_path, log=lambda *_: None,
                         client=ScriptedClient.from_model("models/cup.json"))
    assert m["stop_reason"] == "submitted" and m["score"]["m2"]


def test_key_never_in_logs_or_run_files(tmp_path):
    """Full command-mode run through the real OpenAI-compat adapter (fake transport) incl. a 4xx that echoes
    the key: the key must not appear in log lines, turns.jsonl or any run file."""
    from origami.agents.llm_agent import run_episode

    steps = json.load(open("models/cup.json"))["steps"]
    script = ScriptedClient.from_model({"steps": steps})
    responses = [(500, {}, {"error": {"message": f"oops key={KEY}"}})]
    for r in script.replies:
        responses.append((200, {}, {"choices": [{"finish_reason": "tool_calls", "message": r.message()}],
                                    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "cost": 0.001}}))
    t = FakeTransport(*responses)
    logs = []
    c = OpenAICompatClient("m", base_url="https://example.invalid/v1", api_key=KEY, reports_cost=True,
                           dialect="openrouter", transport=t, sleep=lambda s: None, log=logs.append)
    s = run_episode("models/cup.json", "m", images=False, out_root=tmp_path, client=c, log=logs.append)
    assert s["stop_reason"] == "submitted" and s["cost_usd"] == pytest.approx(0.008)
    from pathlib import Path

    blob = "\n".join(map(str, logs)) + "".join(p.read_text(errors="replace") for p in Path(s["dir"]).iterdir()
                                               if p.suffix in (".json", ".jsonl", ".txt"))
    assert KEY not in blob and any("HTTP 500" in str(x) for x in logs)
