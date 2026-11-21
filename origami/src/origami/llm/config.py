"""Build clients from config dicts (the [providers.X] / [[models]] tables of foldr.toml).

    client = make_client(model_cfg, provider_cfg, api_key=resolve_api_key(provider_cfg.get("api_key_env")))

Add a provider kind: write a factory `fn(model_cfg, provider_cfg, api_key, log) -> LLMClient` and call
`register_provider("mykind", fn)`; then `kind = "mykind"` works in config. Any OpenAI-compatible server
needs no code: `kind = "openai"` + `base_url` (+ `dialect` for how reasoning is sent).
Keys come only from the caller / env vars / .env; they are never put in repr, logs or error text.
"""

from __future__ import annotations

import os
from pathlib import Path

from .anthropic import AnthropicClient
from .errors import ProviderError
from .openai_compat import OPENROUTER_URL, OpenAICompatClient
from .scripted import ScriptedClient
from .types import Price

_PKG_ROOT = Path(__file__).resolve().parents[3]  # .../origami (uv project)

# Built-in provider presets for the CLI; same keys as the [providers.X] tables.
DEFAULT_PROVIDERS: dict[str, dict] = {
    "openrouter": {"kind": "openai", "base_url": OPENROUTER_URL, "api_key_env": "OPENROUTER_API_KEY",
                   "reports_cost": True, "dialect": "openrouter", "extra_headers": {"X-Title": "foldr"},
                   "extra_body": {"usage": {"include": True}}, "min_interval": 3.5},
    "openai": {"kind": "openai", "base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY",
               "dialect": "openai"},
    "anthropic": {"kind": "anthropic", "api_key_env": "ANTHROPIC_API_KEY"},
    "ollama": {"kind": "openai", "base_url": "http://localhost:11434/v1", "api_key_env": "", "dialect": "none"},
    "lmstudio": {"kind": "openai", "base_url": "http://localhost:1234/v1", "api_key_env": "", "dialect": "none"},
    "scripted": {"kind": "scripted", "api_key_env": ""},
}

# USD per 1M tokens, Anthropic first-party list prices (cached 2026-11-03; CHECK the pricing page).
# client_from_cli uses these when no price is given.
KNOWN_PRICES: dict[str, Price] = {
    "claude-opus-5-5": Price(input=4.0, output=20.0, cached_input=0.20),
    "claude-sonnet-5-5": Price(input=2.0, output=10.0, cached_input=0.10),
    "claude-haiku-5-5": Price(input=0.10, output=0.50, cached_input=0.01),
}


def resolve_api_key(env_name: str | None, *, environ=None, dotenv_paths=None) -> str | None:
    """Key from the env var `env_name`, else from a `.env` file (cwd, origami/, repo root). None if absent."""
    if not env_name:
        return None
    environ = os.environ if environ is None else environ
    v = environ.get(env_name)
    if v and v.strip():
        return v.strip()
    paths = dotenv_paths if dotenv_paths is not None else [Path.cwd() / ".env", _PKG_ROOT / ".env",
                                                          _PKG_ROOT.parent / ".env"]
    for p in paths:
        p = Path(p)
        if not p.is_file():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[7:].strip()
            if line.startswith(env_name + "="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                if val:
                    return val
    return None


def _common(model_cfg: dict, provider_cfg: dict, log) -> dict:
    kw = {"log": log, "supports_images": bool(model_cfg.get("images", True)),
          "supports_reasoning": bool(model_cfg.get("reasoning", True)),
          "price": Price.from_cfg(model_cfg.get("price")), "provider": provider_cfg.get("id")}
    for k in ("timeout", "max_retries", "min_interval"):
        if k in provider_cfg:
            kw[k] = provider_cfg[k]
    if "max_tokens" in model_cfg:
        kw["max_tokens"] = int(model_cfg["max_tokens"])
    if "reasoning_effort" in model_cfg:
        kw["reasoning"] = model_cfg["reasoning_effort"]
    return kw


def _make_openai(model_cfg, provider_cfg, api_key, log):
    kw = _common(model_cfg, provider_cfg, log)
    base_url = provider_cfg.get("base_url")
    if not base_url:
        raise ProviderError("provider config needs base_url for kind 'openai'")
    dialect = provider_cfg.get("dialect") or provider_cfg.get("reasoning_dialect") or "none"
    extra_body = {**(provider_cfg.get("extra_body") or {}), **(model_cfg.get("extra_body") or {})}
    if "temperature" in model_cfg:
        kw["temperature"] = model_cfg["temperature"]
    if "reasoning_tokens" in model_cfg:
        kw["reasoning_tokens"] = int(model_cfg["reasoning_tokens"])
    return OpenAICompatClient(model_cfg["id"], base_url=base_url, api_key=api_key,
                              reports_cost=bool(provider_cfg.get("reports_cost", False)),
                              extra_headers=provider_cfg.get("extra_headers"), extra_body=extra_body,
                              dialect=dialect, **kw)


def _make_anthropic(model_cfg, provider_cfg, api_key, log):
    kw = _common(model_cfg, provider_cfg, log)
    for k in ("base_url", "api_version", "betas", "extra_headers"):
        if k in provider_cfg:
            kw[k] = provider_cfg[k]
    kw["thinking"] = model_cfg.get("thinking") or provider_cfg.get("thinking") or "adaptive"
    kw["extra_body"] = {**(provider_cfg.get("extra_body") or {}), **(model_cfg.get("extra_body") or {})}
    return AnthropicClient(model_cfg["id"], api_key=api_key, **kw)


def _make_scripted(model_cfg, provider_cfg, api_key, log):
    script = model_cfg.get("script") or provider_cfg.get("script")
    kw = {"model": model_cfg.get("id", "scripted"), "supports_images": bool(model_cfg.get("images", True))}
    if script and str(script).endswith(".jsonl"):
        return ScriptedClient.from_turns_jsonl(script, **kw)
    if script:
        return ScriptedClient.from_model(script, **kw)
    return ScriptedClient(**kw)


PROVIDER_KINDS = {"openai": _make_openai, "anthropic": _make_anthropic, "scripted": _make_scripted}


def register_provider(kind: str, factory) -> None:
    """factory(model_cfg, provider_cfg, api_key, log) -> LLMClient"""
    PROVIDER_KINDS[kind] = factory


def make_client(model_cfg: dict, provider_cfg: dict, api_key: str | None = None, *, log=print):
    """model_cfg: {id, images?, reasoning?, price?, max_tokens?, temperature?, thinking?, extra_body?, ...}
    provider_cfg: {kind, base_url?, api_key_env?, reports_cost?, dialect?, extra_headers?, extra_body?, ...}
    api_key: the key to use (None + a non-empty api_key_env -> error, so a missing key fails early)."""
    kind = provider_cfg.get("kind", "openai")
    if kind not in PROVIDER_KINDS:
        raise ProviderError(f"unknown provider kind {kind!r} (known: {sorted(PROVIDER_KINDS)})")
    if not api_key and provider_cfg.get("api_key_env"):
        raise ProviderError(f"no API key: set the {provider_cfg['api_key_env']} environment variable (or .env)")
    return PROVIDER_KINDS[kind](model_cfg, provider_cfg, api_key, log)


def client_from_cli(provider: str, model: str, *, base_url: str | None = None, images: bool = True,
                    reasoning: str | None = "medium", reasoning_tokens: int = 0, script: str | None = None,
                    log=print):
    """The `origami agent --provider P --model M` path: built-in preset + key from env/.env."""
    if provider not in DEFAULT_PROVIDERS:
        raise SystemExit(f"unknown --provider {provider!r} (known: {', '.join(DEFAULT_PROVIDERS)})")
    pcfg = {"id": provider, **DEFAULT_PROVIDERS[provider]}
    if base_url:
        pcfg["base_url"] = base_url
    mcfg = {"id": model, "images": images, "reasoning_effort": reasoning}
    if reasoning_tokens:
        mcfg["reasoning_tokens"] = reasoning_tokens
    if model in KNOWN_PRICES and pcfg["kind"] == "anthropic":
        mcfg["price"] = KNOWN_PRICES[model]
    if script:
        mcfg["script"] = script
    return make_client(mcfg, pcfg, resolve_api_key(pcfg.get("api_key_env")), log=log)
