"""Settings: a TOML file (FOLDR_CONFIG, default backend/foldr.toml) plus env overrides. Keys live only in env vars."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from origami.llm import DEFAULT_PROVIDERS, Price

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
DEFAULT_CONFIG = BACKEND_DIR / "foldr.toml"
DEFAULT_MODELS_DIR = REPO_ROOT / "examples" / "models"
LOCAL_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8000", "http://127.0.0.1:8000"]
TRUE = ("1", "true", "yes", "on")


class ConfigError(ValueError):
    pass


@dataclass
class ModelCfg:
    id: str                       # id sent to the provider
    provider: str
    label: str
    key: str = ""                 # what the API/UI calls it (defaults to id); lets two providers serve one id
    images: bool = True
    reasoning: bool = True
    price: Price | None = None
    default: bool = False
    raw: dict = field(default_factory=dict)


@dataclass
class Settings:
    enabled: bool = False
    allow_user_keys: bool = False
    require_https_for_user_keys: bool = True
    max_concurrent_runs: int = 1
    max_user_key_runs: int = 2
    max_turns: int = 40
    per_run_usd: float = 1.0
    daily_usd: float = 3.0
    max_run_minutes: float = 20.0
    ip_cooldown_s: float = 60.0
    retain_s: float = 3600.0
    max_kept_runs: int = 20
    cors_origins: list[str] = field(default_factory=list)
    providers: dict[str, dict] = field(default_factory=dict)
    models: list[ModelCfg] = field(default_factory=list)
    models_dir: Path = DEFAULT_MODELS_DIR
    source: str = "defaults (no config file: AI is off)"

    def model(self, key: str) -> ModelCfg | None:
        return next((m for m in self.models if m.key == key), None)

    # --- what a model can be used for -----------------------------------------------------
    def owner_key(self, provider: str) -> str | None:
        from origami.llm import resolve_api_key

        return resolve_api_key(self.providers[provider].get("api_key_env"))

    def free_provider(self, provider: str) -> bool:
        """No key needed (local server, offline demo): costs nothing, so no price or budget is needed."""
        return not self.providers[provider].get("api_key_env")

    def owner_problem(self, m: ModelCfg) -> str | None:
        """Why the server's own key can't run this model (None = it can)."""
        p = self.providers[m.provider]
        if not self.free_provider(m.provider):
            if not self.owner_key(m.provider):
                return f"the server has no {p['api_key_env']} set"
            if not p.get("reports_cost") and (m.price is None or m.price.is_zero()):
                return "price missing: the budget can't be enforced without it"
        return None

    def user_key_ok(self, m: ModelCfg) -> bool:
        return self.allow_user_keys and bool(self.providers[m.provider].get("api_key_env"))


def _resolve_script(v: str, base: Path) -> str:
    p = Path(v)
    if p.is_absolute():
        return str(p)
    for root in (base, REPO_ROOT):
        if (root / p).exists():
            return str(root / p)
    return str(base / p)


def _env_bool(environ, name: str) -> bool | None:
    v = environ.get(name)
    return None if v is None or not v.strip() else v.strip().lower() in TRUE


def _env_num(environ, name: str) -> float | None:
    v = environ.get(name)
    if v is None or not v.strip():
        return None
    try:
        return float(v)
    except ValueError:
        raise ConfigError(f"{name} must be a number, got {v!r}") from None


def load_settings(path: str | os.PathLike | None = None, environ=None) -> Settings:
    """Read the config file (None -> $FOLDR_CONFIG -> backend/foldr.toml if it exists) and apply env overrides."""
    environ = os.environ if environ is None else environ
    return _load(path or environ.get("FOLDR_CONFIG"), environ)


def _load(path, environ) -> Settings:
    data: dict = {}
    s = Settings()
    cfg_path = Path(path) if path else (DEFAULT_CONFIG if DEFAULT_CONFIG.is_file() else None)
    base = BACKEND_DIR
    if cfg_path is not None:
        if not cfg_path.is_file():
            raise ConfigError(f"config file not found: {cfg_path}")
        try:
            data = tomllib.loads(cfg_path.read_text())
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(f"{cfg_path}: {e}") from None
        s.source = str(cfg_path)
        base = cfg_path.resolve().parent

    ai = data.get("ai") or {}
    for name in ("enabled", "allow_user_keys", "require_https_for_user_keys", "max_concurrent_runs",
                 "max_user_key_runs", "max_turns", "per_run_usd", "daily_usd", "max_run_minutes",
                 "ip_cooldown_s", "retain_s", "max_kept_runs"):
        if name in ai:
            setattr(s, name, type(getattr(s, name))(ai[name]))
    s.cors_origins = [str(o).rstrip("/") for o in ai.get("cors_origins", [])]

    for env_name, attr in (("FOLDR_AI_ENABLED", "enabled"), ("FOLDR_ALLOW_USER_KEYS", "allow_user_keys")):
        v = _env_bool(environ, env_name)
        if v is not None:
            setattr(s, attr, v)
    for env_name, attr in (("FOLDR_PER_RUN_USD", "per_run_usd"), ("FOLDR_DAILY_USD", "daily_usd"),
                           ("FOLDR_MAX_TURNS", "max_turns")):
        v = _env_num(environ, env_name)
        if v is not None:
            setattr(s, attr, type(getattr(s, attr))(v))
    extra = environ.get("FOLDR_CORS_ORIGINS", "")
    s.cors_origins += [o.strip().rstrip("/") for o in extra.split(",") if o.strip()]
    s.cors_origins = list(dict.fromkeys(s.cors_origins))
    if environ.get("FOLDR_MODELS_DIR"):
        s.models_dir = Path(environ["FOLDR_MODELS_DIR"])

    if s.per_run_usd < 0 or s.daily_usd < 0 or s.max_turns < 1:
        raise ConfigError("per_run_usd / daily_usd must be >= 0 and max_turns >= 1")

    # providers: every built-in preset is available by name; [providers.X] adds or overrides
    for pid, preset in DEFAULT_PROVIDERS.items():
        s.providers[pid] = {**preset, "id": pid}
    for pid, p in (data.get("providers") or {}).items():
        if not isinstance(p, dict):
            raise ConfigError(f"[providers.{pid}] must be a table")
        merged = {**s.providers.get(pid, {}), **p, "id": pid}
        if "kind" not in merged:
            raise ConfigError(f"[providers.{pid}] needs kind")
        if "api_key_env" not in merged:
            merged["api_key_env"] = ""
        s.providers[pid] = merged

    seen = set()
    for i, m in enumerate(data.get("models") or []):
        if not m.get("id") or not m.get("provider"):
            raise ConfigError(f"[[models]] #{i + 1} needs id and provider")
        if m["provider"] not in s.providers:
            raise ConfigError(f"model {m['id']!r}: unknown provider {m['provider']!r}")
        raw = dict(m)
        if raw.get("script"):
            raw["script"] = _resolve_script(raw["script"], base)
        mc = ModelCfg(id=m["id"], provider=m["provider"], label=m.get("label") or m["id"],
                      key=m.get("key") or m["id"], images=bool(m.get("images", True)),
                      reasoning=bool(m.get("reasoning", True)), price=Price.from_cfg(m.get("price")),
                      default=bool(m.get("default", False)), raw=raw)
        if mc.key in seen:
            raise ConfigError(f"duplicate model key {mc.key!r} (set `key` to tell them apart)")
        seen.add(mc.key)
        s.models.append(mc)
    for pid, p in s.providers.items():
        bu = p.get("base_url")
        if bu and urlparse(bu).scheme not in ("http", "https"):
            raise ConfigError(f"[providers.{pid}] base_url must be http(s)")
    return s
