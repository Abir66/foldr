"""Anthropic native Messages API (POST /v1/messages) over urllib, no SDK dependency.

Request/response mapping (API version 2026-10-21):
- Header `x-api-key` + `anthropic-version`. temperature/top_p/top_k are never sent.
- Thinking, mode "adaptive" (default; Opus 5.5 / Sonnet 5.5 / Haiku 5.5 / Opus 4.6+):
  `thinking: {type: adaptive, display: summarized}` + `output_config: {effort}`.
  "off" -> thinking omitted + effort "low". None -> nothing sent.
- Thinking, mode "budget" (older models such as claude-haiku-4-5): `{type: enabled, budget_tokens}` with
  low 2k / medium 6k / high 16k. Mode "none": never send thinking fields.
- tool_choice is not sent (auto).
- stop_reason: end_turn/stop_sequence/pause_turn -> stop, tool_use -> tool_calls, max_tokens -> length,
  refusal -> error.
"""

from __future__ import annotations

import json
import re
import time

from .http import HTTPClientBase
from .types import UNSET, ChatResult, Price, Usage, compute_cost

API_VERSION = "2026-10-21"
BUDGETS = {"low": 2000, "medium": 6000, "high": 16000, "xhigh": 24000, "max": 32000}
THINKING_MODES = ("adaptive", "budget", "none")
_STOP = {"end_turn": "stop", "stop_sequence": "stop", "pause_turn": "stop", "tool_use": "tool_calls",
         "max_tokens": "length", "model_context_window_exceeded": "length", "refusal": "error"}
_DATA_URL = re.compile(r"^data:(?P<mt>[\w/+.-]+);base64,(?P<data>.*)$", re.S)
_ID_OK = re.compile(r"^[A-Za-z0-9_-]+$")


# ---- OpenAI-format messages -> Anthropic ---------------------------------------------------------

def _text_of(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "\n".join(p.get("text", "") for p in content if p.get("type") == "text")


def _part(p: dict) -> dict:
    t = p.get("type")
    if t == "text":
        return {"type": "text", "text": p.get("text", "")}
    if t == "image_url":
        url = (p.get("image_url") or {}).get("url", "")
        m = _DATA_URL.match(url)
        if m:
            return {"type": "image", "source": {"type": "base64", "media_type": m["mt"], "data": m["data"]}}
        return {"type": "image", "source": {"type": "url", "url": url}}
    raise ValueError(f"unsupported content part type {t!r}")


def _blocks(content) -> list[dict]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    return [_part(p) for p in content]


def _tool_id(tc_id, i: int) -> str:
    if tc_id and _ID_OK.match(str(tc_id)):
        return str(tc_id)
    return f"call_{i}" if not tc_id else "call_" + re.sub(r"[^A-Za-z0-9_-]", "_", str(tc_id))


def _args(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        v = json.loads(raw or "{}")
        return v if isinstance(v, dict) else {"value": v}
    except ValueError:
        return {"_raw": str(raw)}


def to_anthropic_messages(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """-> (system text, messages). Consecutive same-role turns are merged (all tool results of one
    assistant turn go back in ONE user message, before any text)."""
    system: list[str] = []
    out: list[dict] = []
    n = 0

    def push(role: str, blocks: list[dict]):
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": list(blocks)})

    for m in messages:
        role = m.get("role")
        if role in ("system", "developer"):
            system.append(_text_of(m.get("content")))
        elif role == "user":
            push("user", _blocks(m.get("content")))
        elif role == "assistant":
            blocks = _blocks(m.get("content"))
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                n += 1
                blocks.append({"type": "tool_use", "id": _tool_id(tc.get("id"), n), "name": fn.get("name", ""),
                               "input": _args(fn.get("arguments"))})
            push("assistant", blocks or [{"type": "text", "text": "(empty reply)"}])
        elif role == "tool":
            n += 1
            push("user", [{"type": "tool_result", "tool_use_id": _tool_id(m.get("tool_call_id"), n),
                           "content": _text_of(m.get("content")) or "(no output)"}])
        else:
            raise ValueError(f"unsupported message role {role!r}")
    if out and out[0]["role"] != "user":
        out.insert(0, {"role": "user", "content": [{"type": "text", "text": "(start)"}]})
    for msg in out:  # tool_result blocks must come first in a user message
        if msg["role"] == "user":
            msg["content"].sort(key=lambda b: b["type"] != "tool_result")
    return ("\n\n".join(s for s in system if s) or None), out


def to_anthropic_tools(tools: list[dict] | None) -> list[dict]:
    res = []
    for t in tools or []:
        fn = t.get("function", t)
        d = {"name": fn["name"], "input_schema": fn.get("parameters") or {"type": "object", "properties": {}}}
        if fn.get("description"):
            d["description"] = fn["description"]
        res.append(d)
    return res


# ---- Anthropic response -> ChatResult ------------------------------------------------------------

def parse_anthropic_usage(u: dict | None, price: Price | None = None) -> Usage:
    u = u or {}
    plain = int(u.get("input_tokens") or 0)
    read = int(u.get("cache_read_input_tokens") or 0)
    write = int(u.get("cache_creation_input_tokens") or 0)
    usage = Usage(input=plain + read + write, output=int(u.get("output_tokens") or 0), cached=read, cache_write=write)
    usage.cost_usd = compute_cost(usage, price)
    return usage


def parse_anthropic_response(out: dict, *, price: Price | None = None, latency_s: float = 0.0,
                             provider: str = "anthropic") -> ChatResult:
    texts, thoughts, calls = [], [], []
    for b in out.get("content") or []:
        t = b.get("type")
        if t == "text":
            texts.append(b.get("text", ""))
        elif t == "thinking":
            if b.get("thinking"):
                thoughts.append(b["thinking"])
        elif t == "tool_use":
            calls.append({"id": b.get("id"), "type": "function",
                          "function": {"name": b.get("name"), "arguments": json.dumps(b.get("input") or {})}})
    sr = out.get("stop_reason")
    return ChatResult(content="".join(texts) or None, reasoning="\n\n".join(thoughts) or None, tool_calls=calls,
                      finish_reason=_STOP.get(sr, sr), usage=parse_anthropic_usage(out.get("usage"), price),
                      latency_s=latency_s, provider=provider, raw=out)


class AnthropicClient(HTTPClientBase):
    provider_name = "anthropic"

    def __init__(self, model: str, *, api_key: str | None, price: Price | None = None, timeout: float = 300,
                 max_retries: int = 4, log=print, base_url: str = "https://api.anthropic.com",
                 api_version: str = API_VERSION, thinking: str = "adaptive", display: str = "summarized",
                 supports_images: bool = True, supports_reasoning: bool = True, max_tokens: int = 16000,
                 reasoning: str | None = "medium", betas: list[str] | None = None,
                 extra_headers: dict | None = None, extra_body: dict | None = None, min_interval: float = 0.0,
                 provider: str | None = None, transport=None, sleep=time.sleep):
        super().__init__(model, api_key=api_key, timeout=timeout, max_retries=max_retries,
                         min_interval=min_interval, log=log, transport=transport, sleep=sleep)
        if thinking not in THINKING_MODES:
            raise ValueError(f"unknown thinking mode {thinking!r} (use one of {THINKING_MODES})")
        self.base_url = base_url.rstrip("/")
        self.api_version = api_version
        self.price = Price.from_cfg(price)
        self.thinking = thinking
        self.display = display
        self.supports_images = supports_images
        self.supports_reasoning = supports_reasoning
        self.max_tokens = max_tokens
        self.reasoning = reasoning
        self.betas = list(betas or [])
        self.extra_headers = dict(extra_headers or {})
        self.extra_body = dict(extra_body or {})
        self.provider_name = provider or self.provider_name

    def _thinking_fields(self, reasoning, max_tokens: int) -> tuple[dict, int]:
        if not self.supports_reasoning or self.thinking == "none" or reasoning is None:
            return {}, max_tokens
        if self.thinking == "adaptive":
            if reasoning == "off":
                return {"output_config": {"effort": "low"}}, max_tokens
            th = {"type": "adaptive"}
            if self.display:
                th["display"] = self.display
            return {"thinking": th, "output_config": {"effort": reasoning}}, max_tokens
        if reasoning == "off":  # budget mode
            return {}, max_tokens
        budget = BUDGETS.get(reasoning, BUDGETS["medium"])
        return {"thinking": {"type": "enabled", "budget_tokens": budget}}, max(max_tokens, budget + 4000)

    def build_body(self, messages, tools=None, *, max_tokens=UNSET, temperature=UNSET, reasoning=UNSET) -> dict:
        # temperature is ignored
        max_tokens = self.max_tokens if max_tokens is UNSET else max_tokens
        reasoning = self.reasoning if reasoning is UNSET else reasoning
        system, msgs = to_anthropic_messages(messages)
        extra, max_tokens = self._thinking_fields(reasoning, max_tokens)
        body = {"model": self.model, "max_tokens": max_tokens, "messages": msgs}
        if system:
            body["system"] = system
        if tools:
            body["tools"] = to_anthropic_tools(tools)
        body.update(extra)
        body.update(self.extra_body)
        return body

    def _headers(self) -> dict:
        h = {"content-type": "application/json", "anthropic-version": self.api_version, **self.extra_headers}
        if self._key:
            h["x-api-key"] = self._key
        if self.betas:
            h["anthropic-beta"] = ",".join(self.betas)
        return h

    def chat(self, messages, tools=None, *, max_tokens=UNSET, temperature=UNSET, reasoning=UNSET) -> ChatResult:
        body = self.build_body(messages, tools, max_tokens=max_tokens, temperature=temperature, reasoning=reasoning)
        t0 = time.time()
        out = self._post(self.base_url + "/v1/messages", body, self._headers())
        return parse_anthropic_response(out, price=self.price, latency_s=round(time.time() - t0, 3),
                                        provider=self.provider_name)
