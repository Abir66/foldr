"""Run manager: validates a start request, builds the client, runs the agent on a thread, keeps the events."""

from __future__ import annotations

import base64
import io
import json
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from origami.agents.menu_agent import TEXT_GOAL_MAX_TURNS
from origami.agents.runner import AgentRun, config_from_request
from origami.llm import BudgetExceeded, DailyLimit, ProviderError, make_client

from .config import ModelCfg, Settings

MAX_IMAGE_BYTES = 2_000_000
MAX_IMAGE_PX = 1024
MAX_KEY_CHARS = 400


class ApiError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


class DailyBudget:
    """Money spent today (UTC) by finished owner-key runs. In memory: a restart resets it, so the provider's own
    spend limit stays the real backstop."""

    def __init__(self, limit: float):
        self.limit = limit
        self._lock = threading.Lock()
        self._day = self._today()
        self._spent = 0.0

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def spent(self) -> float:
        with self._lock:
            if self._day != self._today():
                self._day, self._spent = self._today(), 0.0
            return self._spent

    def add(self, usd: float) -> None:
        self.spent()
        with self._lock:
            self._spent += usd


@dataclass
class Run:
    id: str
    uses_user_key: bool
    ip: str
    created: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    status: str = "running"
    lines: list[tuple[int, str]] = field(default_factory=list)  # (seq, serialised event)
    agent: AgentRun | None = None
    secret: str | None = None   # the visitor's key; set to None when the run ends
    counted: bool = False       # spend already added to the daily total

    def spent_usd(self) -> float:
        return self.agent.events.spent_usd if self.agent else 0.0


def prepare_image(req: dict) -> None:
    """Decode the goal picture, check its size, and replace it by a PNG of at most 1024px."""
    g = req.get("goal")
    if not isinstance(g, dict) or g.get("kind") != "image":
        return
    b64 = g.get("image_b64") or ""
    if not isinstance(b64, str):
        raise ValueError("image_b64 must be a string")
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[-1]
    if len(b64) > MAX_IMAGE_BYTES * 4 // 3 + 16:
        raise ValueError("the picture is too big (max 2 MB)")
    try:
        raw = base64.b64decode(b64, validate=True)
        from PIL import Image

        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception:  # noqa: BLE001
        raise ValueError("the goal picture could not be read as an image") from None
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA" if "transparency" in img.info or img.mode in ("LA", "PA") else "RGB")
    img.thumbnail((MAX_IMAGE_PX, MAX_IMAGE_PX))
    out = io.BytesIO()
    img.save(out, format="PNG")
    req["goal"] = {**g, "image_b64": base64.b64encode(out.getvalue()).decode()}


class RunManager:
    def __init__(self, settings: Settings, client_factory=make_client):
        self.s = settings
        self.client_factory = client_factory
        self.daily = DailyBudget(settings.daily_usd)
        self.runs: dict[str, Run] = {}
        self._lock = threading.Lock()
        self._last_start: dict[str, float] = {}

    # ---- queries ------------------------------------------------------------------------
    def daily_left(self) -> float:
        active = sum(r.spent_usd() for r in self.runs.values() if r.status == "running" and not r.uses_user_key)
        return max(0.0, self.s.daily_usd - self.daily.spent() - active)

    def get(self, run_id: str) -> Run | None:
        return self.runs.get(run_id)

    def stop(self, run: Run) -> None:
        if run.agent is not None:
            run.agent.stop()

    # ---- start --------------------------------------------------------------------------
    def start(self, req: dict, *, ip: str, key_channel_ok: bool) -> Run:
        s = self.s
        if not s.enabled:
            raise ApiError(403, "AI runs are turned off on this server")
        if not isinstance(req, dict):
            raise ApiError(422, "request must be a JSON object")
        uk = req.get("user_key")
        secret = None
        if uk is not None:
            if not s.allow_user_keys:
                raise ApiError(403, "this server does not accept your own API key")
            if not isinstance(uk, dict) or not isinstance(uk.get("key"), str) or not uk["key"].strip() \
                    or not isinstance(uk.get("provider"), str):
                raise ApiError(422, "user_key must be {provider, key}")
            if len(uk["key"]) > MAX_KEY_CHARS:
                raise ApiError(422, "that key is too long")
            if s.require_https_for_user_keys and not key_channel_ok:
                raise ApiError(403, "your key is only accepted over https")
            secret = uk["key"].strip()

        model = self._pick_model(req, uk["provider"] if secret else None, secret is not None)
        try:
            prepare_image(req)
            cfg = config_from_request(req, models_dir=s.models_dir, max_turns_cap=s.max_turns)
        except ValueError as e:
            raise ApiError(422, str(e)) from None
        if cfg.max_turns is None:
            cfg.max_turns = min(TEXT_GOAL_MAX_TURNS, s.max_turns) if cfg.goal.hides_target() else None
        cfg.images = cfg.images and model.images
        planner = None
        if cfg.plan_model:
            planner = self._pick_model({"model": cfg.plan_model}, uk["provider"] if secret else None,
                                       secret is not None)

        owner = secret is None
        if owner and self.daily_left() <= 0:
            raise ApiError(402, "today's AI budget is used up; try again tomorrow")

        with self._lock:
            self._prune()
            running = [r for r in self.runs.values() if r.status == "running"]
            limit = s.max_user_key_runs if not owner else s.max_concurrent_runs
            now = time.monotonic()
            if s.ip_cooldown_s and now - self._last_start.get(ip, -1e9) < s.ip_cooldown_s:
                raise ApiError(429, "please wait a minute between runs")
            if sum(r.uses_user_key == (not owner) for r in running) >= limit:
                raise ApiError(409, "a run is already in progress; wait for it to finish")
            try:
                client = self._client(model, cfg, secret)
                planner_client = self._client(planner, cfg, secret) if planner else None
            except ProviderError as e:
                raise ApiError(422, str(e)[:300].replace(secret or "\0", "***")) from None
            run = Run(id=secrets.token_urlsafe(12), uses_user_key=not owner, ip=ip, secret=secret)
            run.agent = AgentRun(cfg, client, emit=lambda ev, r=run: self._emit(r, ev),
                                 max_usd=s.per_run_usd if not self._free(model) else None,
                                 before_call=self._before_call(run, model, owner),
                                 planner_client=planner_client)
            self.runs[run.id] = run
            self._last_start[ip] = now
            run.agent.start()
        return run

    def _pick_model(self, req: dict, user_provider: str | None, by_user: bool) -> ModelCfg:
        key = req.get("model")
        if not isinstance(key, str) or not key.strip():
            raise ApiError(422, "model is required")
        m = self.s.model(key)
        if m is None:
            raise ApiError(422, f"unknown model {key!r}")
        if by_user:
            if m.provider != user_provider:
                raise ApiError(422, f"your {user_provider} key can't run {key!r} ({m.provider})")
            if not self.s.user_key_ok(m):
                raise ApiError(422, f"{m.provider} does not take a pasted key")
        else:
            why = self.s.owner_problem(m)
            if why:
                raise ApiError(422, f"{key!r} can't run here: {why}")
        return m

    def _free(self, m: ModelCfg) -> bool:
        return self.s.free_provider(m.provider)

    def _client(self, m: ModelCfg, cfg, secret: str | None):
        mcfg = dict(m.raw)
        mcfg["reasoning_effort"] = cfg.reasoning if m.reasoning else "off"
        pcfg = self.s.providers[m.provider]
        key = secret or self.s.owner_key(m.provider)
        return self.client_factory(mcfg, pcfg, key, log=lambda *a, **k: None)

    # ---- guards and events --------------------------------------------------------------
    def _before_call(self, run: Run, model: ModelCfg, owner: bool):
        s = self.s
        free = self._free(model)

        def check() -> None:
            if time.monotonic() - run.created > s.max_run_minutes * 60:
                raise BudgetExceeded(f"time limit of {s.max_run_minutes:g} minutes reached")
            if owner and not free:
                ev = run.agent.events
                if not ev.cost_known:
                    raise BudgetExceeded("the provider did not report a cost, so the budget can't be enforced")
                if self.daily.spent() + ev.spent_usd >= s.daily_usd:
                    raise DailyLimit("today's AI budget is used up")

        return check

    def _emit(self, run: Run, ev: dict) -> None:
        line = json.dumps(ev, allow_nan=False, separators=(",", ":"))
        if run.secret and run.secret in line:
            line = line.replace(run.secret, "***")
        run.lines.append((ev["seq"], line))
        if ev.get("type") == "run_finished":
            if not run.uses_user_key and not run.counted:
                run.counted = True
                self.daily.add(run.spent_usd())
            run.secret = None
            run.finished_at = time.monotonic()
            run.status = "finished"

    def _prune(self) -> None:
        now = time.monotonic()
        for rid, r in list(self.runs.items()):
            if r.finished_at is not None and now - r.finished_at > self.s.retain_s:
                del self.runs[rid]
        done = sorted((r for r in self.runs.values() if r.finished_at is not None), key=lambda r: r.finished_at)
        for r in done[: max(0, len(self.runs) - self.s.max_kept_runs)]:
            del self.runs[r.id]
        for ip, t in list(self._last_start.items()):
            if now - t > max(self.s.ip_cooldown_s, 1):
                del self._last_start[ip]
