"""Provider-neutral result types + the LLMClient protocol.

Messages inside the package are OpenAI chat format (system / user parts `text` + `image_url` data URLs /
assistant `tool_calls` / `tool` results). Adapters convert at the edge.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol, runtime_checkable

REASONING_LEVELS = ("low", "medium", "high", "xhigh", "max")  # plus "off" and None (= provider default)
FINISH_REASONS = ("stop", "tool_calls", "length", "error")


class _Unset:
    def __repr__(self) -> str:
        return "UNSET"


UNSET = _Unset()  # "use the client's default" for per-call kwargs


@dataclass
class Price:
    """USD per 1M tokens."""
    input: float = 0.0
    output: float = 0.0
    cached_input: float | None = None  # cache reads; None = billed as input
    cache_write: float | None = None   # cache writes (Anthropic); None = 1.25 x input

    @classmethod
    def from_cfg(cls, d) -> "Price | None":
        if d is None:
            return None
        if isinstance(d, Price):
            return d
        return cls(**{k: (float(v) if v is not None else None) for k, v in dict(d).items()
                      if k in ("input", "output", "cached_input", "cache_write")})

    def is_zero(self) -> bool:
        return not (self.input or self.output or self.cached_input or self.cache_write)


@dataclass
class Usage:
    input: int = 0      # all prompt tokens, cached ones included
    output: int = 0     # all completion tokens, reasoning included
    cached: int = 0     # prompt tokens read from cache
    reasoning: int = 0  # part of `output` (0 when the provider doesn't say)
    cost_usd: float | None = None  # None = provider didn't say and no price known
    cache_write: int = 0  # prompt tokens written to cache (Anthropic)

    def to_dict(self) -> dict:
        return asdict(self)


def compute_cost(u: Usage, price: Price | None) -> float | None:
    if price is None:
        return None
    cached_rate = price.input if price.cached_input is None else price.cached_input
    write_rate = 1.25 * price.input if price.cache_write is None else price.cache_write
    plain = max(0, u.input - u.cached - u.cache_write)
    return (plain * price.input + u.cached * cached_rate + u.cache_write * write_rate
            + u.output * price.output) / 1e6


@dataclass
class ChatResult:
    content: str | None
    reasoning: str | None
    tool_calls: list[dict] = field(default_factory=list)  # {"id","type":"function","function":{"name","arguments"}}
    finish_reason: str | None = None  # normalised: "stop" | "tool_calls" | "length" | "error"
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0
    provider: str | None = None
    raw: dict | None = None  # provider response body (keys are only ever in headers, never here)

    def message(self) -> dict:
        """What extract_calls() / extract_menu_calls() eat."""
        return {"content": self.content, "reasoning": self.reasoning, "tool_calls": self.tool_calls or None}


@runtime_checkable
class LLMClient(Protocol):
    model: str
    supports_images: bool

    def chat(self, messages: list[dict], tools: list[dict] | None = None, *,
             max_tokens=UNSET, temperature=UNSET, reasoning=UNSET) -> ChatResult: ...


def as_result(obj, provider: str | None = None) -> ChatResult:
    """Accept a ChatResult, a raw OpenAI chat-completions body, or a bare message dict."""
    if isinstance(obj, ChatResult):
        return obj
    if isinstance(obj, dict) and "choices" in obj:
        from .openai_compat import parse_openai_response

        return parse_openai_response(obj, reports_cost=True, provider=provider)
    if isinstance(obj, dict):
        return ChatResult(content=obj.get("content"), reasoning=obj.get("reasoning"),
                          tool_calls=list(obj.get("tool_calls") or []), finish_reason=obj.get("finish_reason"),
                          provider=provider)
    raise TypeError(f"not a chat result: {type(obj).__name__}")
