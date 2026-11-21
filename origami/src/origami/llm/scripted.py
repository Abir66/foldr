"""ScriptedClient: an LLMClient with no network, for tests, dry runs and offline demos.

    ScriptedClient([reply, ...])               # each reply: ChatResult | OpenAI response dict | message dict
    ScriptedClient(responder=fn)               # fn(messages, tools) -> reply (e.g. closes over an env/policy)
    ScriptedClient.from_turns_jsonl(path)      # replay the replies of a logged run (runs/**/turns.jsonl)
    ScriptedClient.from_model(model)           # command-mode tool calls that follow a model's own steps

When the script runs out it calls `submit` (if that tool is offered) or replies with empty text.
"""

from __future__ import annotations

import json
from pathlib import Path

from .types import UNSET, ChatResult, Usage, as_result


def tool_call(name: str, args: dict | None = None, call_id: str | None = None) -> dict:
    return {"id": call_id or f"call_{name}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args or {})}}


def reply(*calls, content: str | None = None, reasoning: str | None = None, usage: Usage | None = None,
          finish_reason: str | None = None) -> ChatResult:
    """reply(("valley", {...}), ("submit", {})) -> ChatResult with those tool calls."""
    tcs = [tool_call(n, a, f"call_{i}") for i, (n, a) in enumerate(calls)]
    return ChatResult(content=content, reasoning=reasoning, tool_calls=tcs,
                      finish_reason=finish_reason or ("tool_calls" if tcs else "stop"),
                      usage=usage or Usage(), provider="scripted")


class ScriptedClient:
    provider_name = "scripted"

    def __init__(self, replies=None, *, responder=None, model: str = "scripted", supports_images: bool = True):
        self.model = model
        self.supports_images = supports_images
        self.replies = list(replies or [])
        self.responder = responder
        self.calls = 0
        self.seen: list[list[dict]] = []    # messages of every call
        self.seen_kwargs: list[dict] = []   # tools / max_tokens / reasoning of every call

    def __repr__(self) -> str:
        return f"ScriptedClient(model={self.model!r}, left={len(self.replies)})"

    def chat(self, messages, tools=None, *, max_tokens=UNSET, temperature=UNSET, reasoning=UNSET) -> ChatResult:
        self.calls += 1
        self.seen.append(messages)
        self.seen_kwargs.append({"tools": tools, "max_tokens": max_tokens, "reasoning": reasoning})
        if self.responder is not None:
            r = self.responder(messages, tools)
        elif self.replies:
            r = self.replies.pop(0)
        elif any((t.get("function") or {}).get("name") == "submit" for t in tools or []):
            r = reply(("submit", {"plan": "script finished"}), content="(script finished)")
        else:
            r = ChatResult(content="", reasoning=None, finish_reason="stop", provider="scripted")
        if isinstance(r, BaseException):
            raise r
        res = as_result(r, provider="scripted")
        if res.provider is None:
            res.provider = "scripted"
        return res

    # ---- builders ----

    @classmethod
    def from_turns_jsonl(cls, path, **kw) -> "ScriptedClient":
        """Replies (content, reasoning, tool_calls, usage) of a logged run, in order. Turns that ended in an
        API error (no reply) are skipped."""
        out = []
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            rp = rec.get("reply")
            if rp is None:
                continue
            u = rec.get("usage") or {}
            usage = Usage(input=int(u.get("prompt") or u.get("input") or 0),
                          output=int(u.get("completion") or u.get("output") or 0),
                          cached=int(u.get("cached") or 0), reasoning=int(u.get("reasoning") or 0),
                          cost_usd=u.get("cost") if u.get("cost") is not None else u.get("cost_usd"))
            out.append(ChatResult(content=rp.get("content"), reasoning=rp.get("reasoning"),
                                  tool_calls=list(rp.get("tool_calls") or []), finish_reason=rec.get("finish_reason"),
                                  usage=usage, provider="scripted"))
        return cls(out, **kw)

    @classmethod
    def from_model(cls, model_json, **kw) -> "ScriptedClient":
        """Command-mode replies that make each step of a model JSON (path or dict), then submit."""
        m = json.loads(Path(model_json).read_text()) if isinstance(model_json, (str, Path)) else model_json
        out = []
        for s in m["steps"]:
            args = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in s.items()
                    if k not in ("op", "description")}
            out.append(reply((s["op"], {**args, "plan": f"step: {s.get('description') or s['op']}"}),
                             content=f"Next: {s['op']}"))
        out.append(reply(("submit", {"plan": "done"}), content="Done."))
        return cls(out, **kw)
