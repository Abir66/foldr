"""Start an agent run from a start request and stream its events.

    cfg = config_from_request({"goal": {"kind": "example", "example": "cup"}, "actions": "menu",
                               "model": "scripted"}, models_dir="models")
    run = AgentRun(cfg, client, emit=on_event, max_usd=1.0).start()   # worker thread
    run.stop()                       # cancel: stops before the next LLM call ("stopped")
    run.wait(); run.events.events    # the full event list; run.summary = the loop's summary

    summary, ev = run_agent(cfg, client, emit=on_event)               # same, in this thread

The client is built by the caller (origami.llm.make_client / ScriptedClient); the request's
`reasoning` is a client setting, so the caller passes it to make_client.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

from ..llm import as_result, redact
from .events import Cancelled, RunEvents
from .goal import Goal

ACTIONS = ("menu", "command", "decision")
REASONING = (None, "off", "low", "medium", "high", "xhigh", "max")


@dataclass
class RunConfig:
    goal: Goal
    actions: str = "menu"
    model: str = "scripted"
    images: bool = True
    reasoning: str | None = "medium"
    max_turns: int | None = None       # None = default (3 x par; 30 for text/image goals)
    plan_text: str | None = None
    plan_model: str | None = None      # "Ask a planner model": the caller passes planner_client
    feedback: str = "F1"
    request: dict = field(default_factory=dict)  # sanitised request (no key, no picture bytes)


def sanitize_request(req: dict) -> dict:
    out = {k: v for k, v in req.items() if k != "user_key"}
    if req.get("user_key"):
        out["user_key"] = {"provider": (req["user_key"] or {}).get("provider"), "key": "<hidden>"}
    g = dict(req.get("goal") or {})
    if g.get("image_b64"):
        g["image_b64"] = f"<{len(g['image_b64'])} base64 chars>"
    out["goal"] = g
    return out


def config_from_request(req: dict, *, models_dir, max_turns_cap: int | None = None) -> RunConfig:
    """Validate a start request -> RunConfig. Raises ValueError with a short reason (-> HTTP 422)."""
    if not isinstance(req, dict):
        raise ValueError("request must be a JSON object")
    goal = Goal.from_request(req.get("goal") or {}, models_dir)
    actions = req.get("actions", "menu")
    if actions not in ACTIONS:
        raise ValueError(f"actions must be one of {ACTIONS}")
    if actions == "decision":
        raise ValueError("decision mode is not available in this runner")
    model = req.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model is required")
    reasoning = req.get("reasoning", "medium")
    if reasoning not in REASONING:
        raise ValueError(f"reasoning must be one of {REASONING[1:]}")
    max_turns = req.get("max_turns")
    if max_turns is not None:
        if not isinstance(max_turns, int) or isinstance(max_turns, bool) or max_turns < 1:
            raise ValueError("max_turns must be a positive integer")
        if max_turns_cap is not None and max_turns > max_turns_cap:
            raise ValueError(f"max_turns is capped at {max_turns_cap}")
    elif max_turns_cap is not None and not goal.hides_target():
        max_turns = min(3 * sum(1 for s in goal.target["steps"] if s.get("op") not in ("flip", "rotate")),
                        max_turns_cap)
    plan = req.get("plan") or {"kind": "none"}
    plan_text = plan_model = None
    if plan.get("kind") == "text":
        plan_text = (plan.get("text") or "").strip() or None
        if plan_text and len(plan_text) > 4000:
            raise ValueError("plan text is too long (max 4000 characters)")
    elif plan.get("kind") == "model":
        plan_model = plan.get("model")
        if not plan_model:
            raise ValueError("plan.model is required for a planner")
    elif plan.get("kind") != "none":
        raise ValueError("plan.kind must be none / text / model")
    images = req.get("images", True)
    if not isinstance(images, bool):
        raise ValueError("images must be true or false")
    return RunConfig(goal=goal, actions=actions, model=model, images=images, reasoning=reasoning,
                     max_turns=max_turns, plan_text=plan_text, plan_model=plan_model,
                     request=sanitize_request(req))


def _limits(cfg: RunConfig, env, ev: RunEvents) -> dict:
    mt = cfg.max_turns or cfg.goal.default_max_turns(env)
    lim = {"max_turns": mt, "usd": ev.max_usd}
    if cfg.actions == "command":
        lim["max_free_turns"] = mt
    return lim


def run_agent(cfg: RunConfig, client, *, events: RunEvents | None = None, emit=None, cancel=None,
              max_usd: float | None = None, before_call=None, planner_client=None, out=None,
              log=lambda *a, **k: None) -> tuple[dict, RunEvents]:
    """Run one episode, sending events as it goes. Always ends with a `run_finished` event (also on
    cancel and on errors: stop_reason "stopped" / "error"). out: a dir for the usual run files, or None."""
    from .llm_agent import TOOLS, run_episode
    from .menu_agent import MENU_TOOLS, run_episode_menu

    ev = events or RunEvents(emit, cancel=cancel, max_usd=max_usd, before_call=before_call)
    ev.meta = {"config": cfg.request or {"actions": cfg.actions, "model": cfg.model, "images": cfg.images},
               "goal": cfg.goal.describe()}
    summary: dict = {}
    try:
        env = cfg.goal.make_env(cfg.feedback)
        tools = MENU_TOOLS if cfg.actions == "menu" else TOOLS
        ev.started(env, system_prompt=cfg.goal.system_prompt(cfg.actions), tools=tools,
                   goal_parts=cfg.goal.parts(env, cfg.images), limits=_limits(cfg, env, ev))
        if cfg.goal.kind == "image" and not getattr(client, "supports_images", True):
            raise ValueError("an image goal needs a model that accepts pictures")
        plan_text = cfg.plan_text
        if cfg.plan_model:
            if planner_client is None:
                raise ValueError("plan.kind=model needs a planner client")
            from .expert_plan import make_plan

            ev.status("planning", f"asking {cfg.plan_model} for a plan")
            ev.guard()
            usage = {}

            class _Counting:  # forwards to the planner and books its usage on this run
                model = getattr(planner_client, "model", cfg.plan_model)

                def chat(self, messages, tools=None, **kw):
                    res = as_result(planner_client.chat(messages, tools, **kw))
                    ev.add_cost(res.usage)
                    usage.update(res.usage.to_dict())
                    return res

            plan_text, _ = make_plan(None, cfg.plan_model, client=_Counting(), goal=cfg.goal)
            ev.plan("model", plan_text, model=cfg.plan_model, usage=usage)
            plan_text = plan_text or None
        elif plan_text:
            ev.plan("text", plan_text)
        if ev.cancelled:
            raise Cancelled("stopped by the user")
        common = dict(images=cfg.images, max_turns=cfg.max_turns, feedback=cfg.feedback, client=client,
                      goal=cfg.goal, events=ev, out=Path(out) if out else None, plan_text=plan_text, log=log)
        if cfg.actions == "menu":
            summary = run_episode_menu(None, cfg.model, **common)
        else:
            summary = run_episode(None, cfg.model, **common)
    except Cancelled:
        summary = {"stop_reason": "stopped"}
        ev.finished("stopped")
    except Exception as e:  # noqa: BLE001  (any failure ends the run with an event, never silently)
        from ..llm import ProviderError, stop_reason_for

        reason = stop_reason_for(e) if isinstance(e, ProviderError) else "error"
        msg = redact(f"{type(e).__name__}: {e}")[:500]
        if not ev._started:  # failed before run_started: send a minimal one so the stream stays well-formed
            from ..env import FoldEnv

            ev.started(FoldEnv(None), system_prompt="", tools=[], goal_parts=[], limits={"max_turns": 0})
        ev.error("provider_error" if isinstance(e, ProviderError) else "internal", msg)
        summary = {"stop_reason": reason, "error": msg}
        ev.finished(reason, error=msg)
    return summary, ev


class AgentRun:
    """One run on a worker thread. emit(event) is called from that thread."""

    def __init__(self, cfg: RunConfig, client, *, emit=None, max_usd: float | None = None, before_call=None,
                 planner_client=None, out=None, log=lambda *a, **k: None):
        self.cfg = cfg
        self.cancel = threading.Event()
        self.events = RunEvents(emit, cancel=self.cancel, max_usd=max_usd, before_call=before_call)
        self.summary: dict | None = None
        self._kw = dict(planner_client=planner_client, out=out, log=log)
        self._client = client
        self.thread = threading.Thread(target=self._run, name="agent-run", daemon=True)

    def _run(self) -> None:
        try:
            self.summary, _ = run_agent(self.cfg, self._client, events=self.events, **self._kw)
        finally:
            self._client = None  # drop the client (and any user key it holds) when done

    def start(self) -> "AgentRun":
        self.thread.start()
        return self

    def stop(self) -> None:
        self.cancel.set()

    def wait(self, timeout: float | None = None) -> bool:
        self.thread.join(timeout)
        return not self.thread.is_alive()

    @property
    def done(self) -> bool:
        return self.events.finished_event is not None
