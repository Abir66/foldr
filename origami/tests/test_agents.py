"""Baselines + LLM agent loop (offline: a fake client replays scripted replies; no network)."""

import json

import pytest

from origami.agents import llm_agent
from origami.agents.baselines import oracle_agent, random_agent
from origami.agents.llm_agent import TOOLS, call_to_action, extract_calls, run_episode


def test_oracle_and_random_baselines():
    _, s = oracle_agent("models/triangle.json")
    assert s.m2 and s.moves == s.par
    env, s = random_agent("models/triangle.json", seed=1, max_moves=3)
    assert s.moves <= 3 and env.done


def _reply(*calls, content="thinking"):
    return {"choices": [{"message": {"content": content, "tool_calls": [
        {"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
        for i, (n, a) in enumerate(calls)]}, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10, "prompt_tokens_details": {"cached_tokens": 50}}}


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.seen = []

    def chat(self, messages, tools):
        self.calls += 1
        self.seen.append(messages)
        return self.replies.pop(0)


CUP = [s for s in json.load(open("models/cup.json"))["steps"]]


def _cup_replies():
    out = [{"choices": [{"message": {"content": "I will fold."}}]}]  # format error first
    for s in CUP:
        args = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in s.items()
                if k not in ("op", "description")}
        out.append(_reply((s["op"], {**args, "plan": f"do {s['op']}"})))
    out.append(_reply(("submit", {"plan": "done"})))
    return out


@pytest.mark.parametrize("context", ["rebuilt", "chat"])
def test_llm_loop_with_fake_client(tmp_path, context):
    fc = FakeClient(_cup_replies())
    s = run_episode("models/cup.json", "fake/model", images=(context == "rebuilt"), max_turns=12,
                    context=context, out_root=tmp_path, client=fc, log=lambda *_: None)
    assert s["stop_reason"] == "submitted" and s["score"]["m2"]
    assert s["format_errors"] == 1 and s["legal"] == 7 and s["illegal"] == 0
    assert s["tokens"]["cached_tokens"] > 0
    from pathlib import Path

    d = Path(s["dir"])
    lines = (d / "turns.jsonl").read_text().splitlines()
    assert len(lines) == 9 and "FORMAT ERROR" in json.loads(lines[1])["prompt_text"]
    assert "Your current plan: do rotate" in json.loads(lines[2])["prompt_text"]
    sol = json.loads((d / "solution.json").read_text())
    assert len(sol["steps"]) == 7
    # stable prefix: system + goal part identical every turn (rebuilt mode)
    if context == "rebuilt":
        first = [fc.seen[0][0], fc.seen[0][1]["content"][:2]]
        assert all([m[0], m[1]["content"][:2]] == first for m in fc.seen)
    else:
        assert sum(1 for m in fc.seen[-1] if m["role"] == "tool") == 8


def test_extract_calls_fallbacks():
    msg = {"content": 'ok {"name": "valley", "arguments": {"line": "{\\"onto\\":[\\"tr\\",\\"bl\\"]}", "plan": "x"}}'}
    (name, args, _), = extract_calls(msg)
    assert name == "valley"
    a = call_to_action(name, args)
    assert a == {"op": "valley", "line": '{"onto":["tr","bl"]}'}
    with pytest.raises(ValueError):
        extract_calls({"content": "no call here"})
    with pytest.raises(ValueError):
        extract_calls({"tool_calls": [{"function": {"name": "valley", "arguments": "{bad"}}]})
    assert all("plan" in t["function"]["parameters"]["required"] for t in TOOLS)


def test_key_never_in_repr(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-SECRET")
    c = llm_agent.OpenRouterClient("m")
    assert "SECRET" not in repr(c) and "SECRET" not in str(vars(c).get("model"))


def test_out_of_tokens_and_budget(tmp_path):
    cut = {"choices": [{"message": {"content": None, "reasoning": "hmm " * 50}, "finish_reason": "length"}],
           "usage": {"prompt_tokens": 10, "completion_tokens": 10, "cost": 0.6}}
    led = llm_agent.CostLedger(tmp_path / "ledger.json", 1.0)
    fc = FakeClient([cut, cut, cut])
    s = run_episode("models/cup.json", "fake/m", images=False, max_turns=5, out_root=tmp_path, client=fc,
                    log=lambda *_: None, ledger=led)
    assert s["ran_out_of_tokens"] == 2 and s["format_errors"] == 0
    assert s["stop_reason"] == "budget" and fc.calls == 2 and abs(led.total() - 1.2) < 1e-9
