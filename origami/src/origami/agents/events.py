"""Event stream of one agent run (the SSE events; schema in docs/events.schema.json).

The agent loops call hooks on a RunEvents object; it numbers the events, de-duplicates pictures
(an `image` event is sent once, later events refer to {"img": id}), keeps a live viewer timeline in
step with env.trajectory (action -> timeline_step + state, undo -> truncate), adds up usage/cost, and
is where cancel and the per-run money limit are checked before every LLM call.

    ev = RunEvents(emit=print_or_queue, cancel=threading.Event(), max_usd=1.0)
    run_episode(..., events=ev)      # or run_menu_episode / run_agent
    ev.events                        # every event so far (JSON-safe dicts)
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import threading
import time

from ..llm import BudgetExceeded, ChatResult

MAX_IMAGE_PX = 512
ROUND_DIGITS = 5


class Cancelled(Exception):
    """The run's cancel flag was set (checked between LLM calls)."""


def _json_default(o):
    import numpy as np

    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, bytes):
        return f"<{len(o)} bytes>"
    return str(o)


def _plain(obj):
    """Deep, JSON-safe copy (numpy -> lists; later mutation of the source can't change the event)."""
    return json.loads(json.dumps(obj, default=_json_default))


def downscale_png(png: bytes, max_px: int = MAX_IMAGE_PX) -> bytes:
    try:
        from PIL import Image
    except ImportError:
        return png
    try:
        im = Image.open(io.BytesIO(png))
        if max(im.size) <= max_px:
            return png
        im.thumbnail((max_px, max_px))
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        return buf.getvalue()
    except Exception:  # noqa: BLE001  (an unreadable picture is shown as sent)
        return png


def _parse_args(raw):
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError):
        return raw  # shown as-is (the loop reports the format error)


class RunEvents:
    def __init__(self, emit=None, *, cancel: threading.Event | None = None, max_usd: float | None = None,
                 before_call=None, max_image_px: int = MAX_IMAGE_PX, round_digits: int | None = ROUND_DIGITS):
        """emit(event): called with each event as it is made (e.g. push to SSE subscribers).
        before_call(): extra check before each LLM call (e.g. a daily budget); may raise BudgetExceeded."""
        self._emit = emit
        self.cancel = cancel or threading.Event()
        self.max_usd = max_usd
        self.before_call = before_call
        self.max_image_px = max_image_px
        self.round_digits = round_digits
        self.events: list[dict] = []
        self.meta: dict = {}  # run_started fields set by the caller (config, goal)
        self._t0 = time.monotonic()
        self._lock = threading.Lock()
        self._img_ids: dict[str, str] = {}
        self.spent_usd = 0.0
        self.totals = {"turns": 0, "input": 0, "output": 0, "cached": 0, "reasoning": 0, "cost_usd": 0.0}
        self.cost_known = True
        self.system_prompt: str | None = None
        self.goal_parts: list = []
        self._goal_n = 0
        self._started = False
        self._plan_sent = False
        self.env = None
        self.tb = None
        self._states: list = []
        self._traj: list = []
        self.finished_event: dict | None = None
        self.last_menu = None  # Options of the latest menu request (scripted test policies read it)

    # -- basics ------------------------------------------------------------------------------
    def send(self, type_: str, turn: int | None = None, **fields) -> dict:
        with self._lock:
            ev = {"seq": len(self.events), "t": round(time.monotonic() - self._t0, 3), "type": type_}
            if turn is not None:
                ev["turn"] = turn
            ev.update(_plain(fields))
            self.events.append(ev)
        if self._emit is not None:
            self._emit(ev)
        return ev

    def image(self, png: bytes, caption: str | None = None, inline: list | None = None) -> str:
        """Id of this picture; sends an `image` event (downscaled data URL) the first time it is seen.
        inline: instead of an event, append {"id", "data", "caption"} to this list (pictures carried
        inside run_started, which must be the first event)."""
        key = hashlib.sha1(png).hexdigest()
        if key in self._img_ids:
            return self._img_ids[key]
        iid = f"i{len(self._img_ids) + 1}"
        self._img_ids[key] = iid
        small = downscale_png(png, self.max_image_px)
        img = {"id": iid, "data": "data:image/png;base64," + base64.b64encode(small).decode(), "caption": caption}
        if inline is not None:
            inline.append(img)
        else:
            self.send("image", **img)
        return iid

    def _image_url(self, url: str, caption: str | None = None, inline: list | None = None) -> str:
        if url.startswith("data:") and ";base64," in url:
            return self.image(base64.b64decode(url.split(",", 1)[1]), caption, inline)
        return url

    def status(self, phase: str, detail: str | None = None, turn: int | None = None) -> None:
        self.send("status", turn, phase=phase, detail=detail)

    def error(self, kind: str, text: str, turn: int | None = None) -> None:
        self.send("error", turn, kind=kind, text=str(text)[:2000])

    def guard(self) -> None:
        """Before each LLM call: stop if cancelled or out of money."""
        if self.cancel.is_set():
            raise Cancelled("stopped by the user")
        if self.max_usd is not None and self.spent_usd >= self.max_usd:
            raise BudgetExceeded(f"run limit ${self.max_usd:.2f} reached (spent ${self.spent_usd:.4f})")
        if self.before_call is not None:
            self.before_call()

    @property
    def cancelled(self) -> bool:
        return self.cancel.is_set()

    # -- run start ---------------------------------------------------------------------------
    def _parts(self, parts, kind: str = "text", caption: str | None = None) -> list[dict]:
        out = []
        for p in parts:
            if p.get("type") == "image_url":
                inline: list = []
                iid = self._image_url(p["image_url"]["url"], caption, inline)
                out.append({"kind": "image", "img": iid, **({"data": inline[0]["data"], "caption": caption}
                                                           if inline else {})})
            else:
                out.append({"kind": kind, "text": p.get("text", "")})
        return out

    def started(self, env, *, system_prompt: str, tools: list, goal_parts: list, limits: dict,
                plan_parts: int = 0, goal=None) -> None:
        """Attach the run's env (the live timeline starts from its current state) and send `run_started`.
        goal_parts: the stable parts every request starts with; the last `plan_parts` of them are a plan
        (sent as a `plan` event, unless one was sent already). Calling it again (the runner sends
        run_started before planning, then the loop makes its own env) only re-attaches."""
        from ..steps_export import TimelineBuilder

        self.env = env
        self.system_prompt = system_prompt
        self.goal_parts = list(goal_parts)
        self._goal_n = len(goal_parts) - plan_parts
        self.tb = TimelineBuilder.for_model(env.model, env.state)
        self._states = [env.state.copy()]
        self._traj = []
        if goal is not None:
            self.meta.setdefault("goal", goal.describe())
        if not self._started:
            self._started = True
            hidden = getattr(env, "hide_target", False)
            gp = self._parts(goal_parts[:self._goal_n], "goal", "goal picture" if hidden else "target, front and back")
            tl = self.tb.to_dict()
            tl = {**tl, "states": list(tl["states"]), "steps": list(tl["steps"])}
            self.send("run_started", config=self.meta.get("config", {}), goal=self.meta.get("goal", {}),
                      system_prompt=system_prompt,
                      tools=[{"name": t["function"]["name"], "summary": t["function"].get("description", "")}
                             for t in tools],
                      goal_parts=gp, timeline=self._round(tl), limits=limits)
        if plan_parts and not self._plan_sent:
            txt = "\n\n".join(p.get("text", "") for p in goal_parts[self._goal_n:])
            from .menu_agent import PLAN_HEADER

            self.plan("text", txt.removeprefix(PLAN_HEADER).strip())

    def plan(self, source: str, text: str, model: str | None = None, usage: dict | None = None) -> None:
        self._plan_sent = True
        self.send("plan", source=source, model=model, text=text, usage=usage)

    def add_cost(self, usage) -> None:
        """Usage of a call that isn't a folding turn (the planner): counts toward totals and the limit."""
        for k in ("input", "output", "cached", "reasoning"):
            self.totals[k] += int(getattr(usage, k) or 0)
        if usage.cost_usd is None:
            self.cost_known = False
        else:
            self.spent_usd += float(usage.cost_usd)
            self.totals["cost_usd"] = round(self.spent_usd, 6)

    # -- one LLM turn ------------------------------------------------------------------------
    def request(self, turn: int, messages: list, menu=None) -> None:
        out_msgs, chars, n_img = [], 0, 0
        for m in messages:
            role, content = m.get("role"), m.get("content")
            if role == "system":
                chars += len(content or "")
                out_msgs.append({"role": "system", "ref": "system_prompt"} if content == self.system_prompt
                                else {"role": "system", "text": content})
                continue
            if role == "tool":
                out_msgs.append({"role": "tool", "text": content})
                continue
            if role == "assistant":
                out_msgs.append({"role": "assistant", "content": content,
                                 "tool_calls": [{"name": (tc.get("function") or {}).get("name"),
                                                 "args": _parse_args((tc.get("function") or {}).get("arguments"))}
                                                for tc in m.get("tool_calls") or []]})
                continue
            parts = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
            ev_parts, first_text = [], True
            for p in parts:
                if p.get("type") == "image_url":
                    n_img += 1
                if p in self.goal_parts:
                    i = self.goal_parts.index(p)
                    ev_parts.append({"kind": "plan", "ref": "plan"} if i >= self._goal_n else {"kind": "goal", "ref": i})
                    chars += len(p.get("text", ""))
                elif p.get("type") == "image_url":
                    ev_parts.append({"kind": "image", "img": self._image_url(p["image_url"]["url"])})
                else:
                    t = p.get("text", "")
                    chars += len(t)
                    ev_parts.append({"kind": "turn" if first_text else "text", "text": t, "chars": len(t)})
                    first_text = False
            out_msgs.append({"role": role, "parts": ev_parts})
        fields = {"messages": out_msgs, "prompt_chars": chars, "images": n_img}
        self.last_menu = menu
        if menu is not None:
            fields["menu"] = [{"id": o.id, "section": o.fam, "description": o.description, "after": o.after,
                               "step": o.step} for o in menu]
            fields["menu_truncated"] = sorted(k for k, v in (getattr(menu, "truncated", None) or {}).items() if v)
        self.send("llm_request", turn, **fields)

    def response(self, turn: int, res: ChatResult) -> None:
        u = res.usage
        self.totals["turns"] += 1
        self.add_cost(u)
        calls = [{"name": (tc.get("function") or {}).get("name"),
                  "args": _parse_args((tc.get("function") or {}).get("arguments"))} for tc in res.tool_calls or []]
        self.send("llm_response", turn, latency_s=round(float(res.latency_s or 0.0), 2),
                  finish_reason=res.finish_reason, provider=res.provider, reasoning=res.reasoning,
                  content=res.content, tool_calls=calls,
                  usage={"input": u.input, "output": u.output, "cached": u.cached, "reasoning": u.reasoning,
                         "cost_usd": u.cost_usd},
                  spent_usd=round(self.spent_usd, 6))

    # -- results -----------------------------------------------------------------------------
    def _round(self, obj):
        if self.round_digits is None:
            return obj
        from ..steps_export import round_floats

        return round_floats(obj, self.round_digits)

    def action(self, turn: int, res: dict) -> None:
        """One executed tool call (its result dict from the loop). Syncs the timeline to env.trajectory:
        steps that were taken back -> `undo` event; a newly committed step -> in this action event."""
        new_steps, popped = self._sync()
        fields = {"tool": res.get("tool"), "ok": res.get("ok", True), "code": res.get("code"),
                  "text": res.get("text") or res.get("tool")}
        for k in ("option", "source", "query", "section"):
            if res.get(k) is not None:
                fields[k] = res[k]
        if res.get("action") is not None:
            fields["step"] = res["action"]
        if new_steps:
            fields["timeline_step"] = self._round(self.tb.steps[-1])
            fields["state"] = self._round(self.tb.states[-1])
            fields["timeline_len"] = len(self.tb)
            if len(new_steps) > 1:
                fields["timeline_steps"] = [self._round(s) for s in self.tb.steps[-len(new_steps):]]
                fields["states"] = [self._round(s) for s in self.tb.states[-len(new_steps):]]
        self.send("action", turn, **fields)
        if popped:
            self.send("undo", turn, n=popped, timeline_len=len(self.tb))

    def _sync(self) -> tuple[list, int]:
        if self.env is None:
            return [], 0
        traj = self.env.trajectory
        p = 0
        while p < min(len(traj), len(self._traj)) and traj[p] == self._traj[p]:
            p += 1
        popped = len(self._traj) - p
        if popped:
            self.tb.pop(popped)
            del self._traj[p:], self._states[p + 1:]
        new = list(traj[p:])
        from ..observe import step_line
        for i, step in enumerate(new):
            before = self._states[-1]
            if i == len(new) - 1:
                after = self.env.state.copy()
            else:  # several steps at once: rebuild the in-between states
                from ..engine import Simulator

                sim = Simulator(self.env.size)
                sim.state = before.copy()
                sim.apply(step)
                after = sim.state.copy()
            self.tb.append(step if step.get("description") else {**step, "description": step_line(step)},
                           before, after)
            self._states.append(after)
            self._traj.append(step)
        return new, popped

    def finished(self, stop_reason: str, *, score=None, variants: dict | None = None, solution: dict | None = None,
                 final_pngs=(), error: str | None = None) -> dict:
        sc = None
        if score is not None and score.target is not None:
            sc = {"passed": bool(score.passed), "m1": bool(score.m1), "m2": bool(score.m2),
                  "scores": score.scores, "target": score.target, "moves": score.moves, "par": score.par}
        if variants:
            sc = sc or {}
            sc["pass_any"] = any(v["m2"] for v in variants.values())
        imgs = [self.image(p, c) for p, c in final_pngs]
        totals = {**self.totals, "cost_usd": round(self.spent_usd, 6) if self.cost_known else None,
                  "seconds": round(time.monotonic() - self._t0, 1)}
        self.finished_event = self.send("run_finished", stop_reason=stop_reason, score=sc, variants=variants or {},
                                        final_images=imgs, solution=solution, totals=totals, error=error,
                                        timeline_len=len(self.tb) if self.tb is not None else 0)
        return self.finished_event


class NullEvents(RunEvents):
    """No-op hooks (loops run without an event stream); guard() still honours the cancel flag."""

    def send(self, type_, turn=None, **fields):
        return {}

    def image(self, png, caption=None):
        return ""

    def started(self, env, **kw):
        pass

    def plan(self, *a, **kw):
        pass

    def request(self, turn, messages, menu=None):
        pass

    def response(self, turn, res):
        pass

    def action(self, turn, res):
        pass

    def finished(self, stop_reason, **kw):
        return {}
