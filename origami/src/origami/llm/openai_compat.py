"""OpenAI-compatible chat completions: OpenRouter, OpenAI, Ollama, LM Studio, Together, vLLM, ...

Reasoning is sent per provider "dialect" (never guessed per model):
  "openrouter": body.reasoning = {"effort": e} | {"enabled": false} (off) | {"max_tokens": n} (reasoning_tokens)
  "openai":     body.reasoning_effort = e (xhigh/max clamp to high; off/None -> not sent)
  "none":       nothing sent (Ollama, most local servers)
"""

from __future__ import annotations

import time

from .http import HTTPClientBase
from .types import UNSET, ChatResult, Price, Usage, compute_cost

DIALECTS = ("openrouter", "openai", "none")
_FINISH = {"stop": "stop", "end_turn": "stop", "tool_calls": "tool_calls", "function_call": "tool_calls",
           "tool_use": "tool_calls", "length": "length", "max_tokens": "length", "error": "error",
           "content_filter": "error"}


def normalise_finish(fr) -> str | None:
    if fr is None:
        return None
    return _FINISH.get(str(fr).lower(), str(fr).lower())


def parse_openai_usage(u: dict | None, *, price: Price | None = None, reports_cost: bool = False) -> Usage:
    u = u or {}
    usage = Usage(input=int(u.get("prompt_tokens") or 0), output=int(u.get("completion_tokens") or 0),
                  cached=int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
                  reasoning=int((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0))
    if reports_cost and u.get("cost") is not None:
        usage.cost_usd = float(u["cost"])
    else:
        usage.cost_usd = compute_cost(usage, price)
    return usage


def parse_openai_response(out: dict, *, price: Price | None = None, reports_cost: bool = False,
                          provider: str | None = None, latency_s: float = 0.0) -> ChatResult:
    choice = (out.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    reasoning = msg.get("reasoning") or msg.get("reasoning_content")
    return ChatResult(content=msg.get("content"), reasoning=reasoning, tool_calls=list(msg.get("tool_calls") or []),
                      finish_reason=normalise_finish(choice.get("finish_reason")),
                      usage=parse_openai_usage(out.get("usage"), price=price, reports_cost=reports_cost),
                      latency_s=latency_s, provider=out.get("provider") or provider, raw=out)


class OpenAICompatClient(HTTPClientBase):
    provider_name = "openai"

    def __init__(self, model: str, *, base_url: str, api_key: str | None = None, price: Price | None = None,
                 reports_cost: bool = False, extra_headers: dict | None = None, extra_body: dict | None = None,
                 timeout: float = 300, max_retries: int = 4, min_interval: float = 0.0, log=print,
                 dialect: str = "none", supports_images: bool = True, supports_reasoning: bool = True,
                 max_tokens: int = 16000, temperature: float | None = 0.3, reasoning: str | None = "medium",
                 reasoning_tokens: int = 0, provider: str | None = None, transport=None, sleep=time.sleep):
        super().__init__(model, api_key=api_key, timeout=timeout, max_retries=max_retries,
                         min_interval=min_interval, log=log, transport=transport, sleep=sleep)
        if dialect not in DIALECTS:
            raise ValueError(f"unknown reasoning dialect {dialect!r} (use one of {DIALECTS})")
        self.base_url = base_url.rstrip("/")
        self.price = Price.from_cfg(price)
        self.reports_cost = reports_cost
        self.extra_headers = dict(extra_headers or {})
        self.extra_body = dict(extra_body or {})
        self.dialect = dialect
        self.supports_images = supports_images
        self.supports_reasoning = supports_reasoning
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning = reasoning
        self.reasoning_tokens = reasoning_tokens
        self.provider_name = provider or self.provider_name

    @property
    def url(self) -> str:
        return self.base_url + "/chat/completions"

    def _reasoning_fields(self, reasoning) -> dict:
        if not self.supports_reasoning or self.dialect == "none":
            return {}
        if self.dialect == "openrouter":
            if reasoning == "off":
                return {"reasoning": {"enabled": False}}
            if reasoning:
                return {"reasoning": {"effort": reasoning}}
            if self.reasoning_tokens:
                return {"reasoning": {"max_tokens": self.reasoning_tokens}}
            return {"reasoning": {"enabled": True}}
        if reasoning and reasoning != "off":  # openai
            return {"reasoning_effort": {"xhigh": "high", "max": "high"}.get(reasoning, reasoning)}
        return {}

    def build_body(self, messages, tools=None, *, max_tokens=UNSET, temperature=UNSET, reasoning=UNSET) -> dict:
        max_tokens = self.max_tokens if max_tokens is UNSET else max_tokens
        temperature = self.temperature if temperature is UNSET else temperature
        reasoning = self.reasoning if reasoning is UNSET else reasoning
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens}
        if temperature is not None:
            body["temperature"] = temperature
        if tools:  # no tools = plain text reply (e.g. the expert-plan call)
            body.update(tools=tools, tool_choice="auto")
        body.update(self._reasoning_fields(reasoning))
        body.update(self.extra_body)
        return body

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json", **self.extra_headers}
        if self._key:
            h["Authorization"] = f"Bearer {self._key}"
        return h

    def _body_error(self, out: dict):
        if "error" in out and not out.get("choices"):  # OpenRouter: error inside a 200
            err = out["error"] if isinstance(out["error"], dict) else {"message": str(out["error"])}
            code = int(err["code"]) if str(err.get("code", "")).isdigit() else 500
            return code, str(err.get("message") or err)[:500]
        return None

    def chat(self, messages, tools=None, *, max_tokens=UNSET, temperature=UNSET, reasoning=UNSET) -> ChatResult:
        body = self.build_body(messages, tools, max_tokens=max_tokens, temperature=temperature, reasoning=reasoning)
        t0 = time.time()
        out = self._post(self.url, body, self._headers())
        return parse_openai_response(out, price=self.price, reports_cost=self.reports_cost,
                                     provider=self.provider_name, latency_s=round(time.time() - t0, 3))


OPENROUTER_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient(OpenAICompatClient):
    """OpenRouter preset: OpenRouterClient(model, key=None, ..., reasoning_tokens, reasoning_effort).
    Key from OPENROUTER_API_KEY or .env when not given."""

    def __init__(self, model: str, key: str | None = None, max_tokens: int = 16000, timeout: float = 300,
                 max_retries: int = 4, log=print, reasoning_tokens: int = 0, reasoning_effort: str | None = "medium",
                 min_interval: float = 3.5, **kw):
        from .config import resolve_api_key
        from .errors import ProviderError

        key = key or resolve_api_key("OPENROUTER_API_KEY")
        if not key:
            raise ProviderError("OPENROUTER_API_KEY not found (env var or .env)")
        super().__init__(model, base_url=OPENROUTER_URL, api_key=key,
                         reports_cost=True, extra_headers={"X-Title": "foldr"},
                         extra_body={"usage": {"include": True}}, timeout=timeout, max_retries=max_retries,
                         min_interval=min_interval, log=log, dialect="openrouter", max_tokens=max_tokens,
                         reasoning=reasoning_effort, reasoning_tokens=reasoning_tokens, provider="openrouter", **kw)

    @property
    def reasoning_effort(self):
        return self.reasoning
