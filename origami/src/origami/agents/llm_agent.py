"""LLM agent loop over FoldEnv via OpenRouter (OpenAI-compatible chat completions + tool calling).

    uv run origami agent --target models/cup.json --model google/gemma-4-31b-it:free [--no-images]
                         [--max-steps N] [--context rebuilt|chat] [--feedback F1]

Prompt design:
- Prompt layout = STABLE PREFIX then CHANGING TAIL (so provider prompt caching can hit):
  system (rules, action format; tool definitions are sent identically every call) ->
  goal text + target image (identical within an episode) -> per-turn part: observation text
  (incl. one-line history, last errors, tabu notes), current image only, latest plan, last 2 short reasonings.
- context "rebuilt" (default): the prompt is rebuilt from the env every turn (no growing chat).
  context "chat": classic growing transcript; images of older turns are stripped.
- Every tool call carries a `plan` (<= 60 words); the latest plan is shown next turn.
- A turn = one API call. Several tool calls in one reply are executed in order (max 3); the first
  rejected move stops the rest. No / malformed tool call = format error -> re-prompt with the error.
- 429 / 5xx -> backoff; daily-limit / credit errors -> stop cleanly.
- The API key is read from OPENROUTER_API_KEY (env) or `.env` in the repo root and is NEVER logged.
Logs: runs/<date>/<model>/<target>[_n]/turns.jsonl + system.txt, target.png, final_*.png, score.json,
solution.json (committed steps as a model JSON = tutorial).
"""

from __future__ import annotations

import base64
import datetime as _dt
import io
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..env import FoldEnv, summarize_score
from ..observe import render_cp_png, step_line
from .command_layer import CommandLayer, call_to_action  # noqa: F401  (call_to_action re-exported)

REPO = Path(__file__).resolve().parents[3]
MAX_CALLS_PER_REPLY = 3


# re-exported for the agent modules
from ..llm import (APIError, BudgetExceeded, DailyLimit, LLMClient, NoCredit, OpenRouterClient,  # noqa: F401,E402
                   ProviderError, as_result, resolve_api_key)


class CostLedger:
    """Cumulative $ cost across runs (and parallel processes): a small JSON file, updated under a lock."""

    def __init__(self, path, cap_usd: float | None):
        self.path = Path(path)
        self.cap = cap_usd

    def _rw(self, add: float = 0.0, label: str = "") -> float:
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            txt = f.read()
            d = json.loads(txt) if txt.strip() else {"total_usd": 0.0, "by": {}}
            if add:
                d["total_usd"] += add
                d["by"][label] = d["by"].get(label, 0.0) + add
                f.seek(0)
                f.truncate()
                f.write(json.dumps(d, indent=1))
            fcntl.flock(f, fcntl.LOCK_UN)
        return float(d["total_usd"])

    def total(self) -> float:
        return self._rw()

    def add(self, usd: float, label: str) -> float:
        return self._rw(usd, label)

    def check(self) -> None:
        if self.cap is not None and self.total() >= self.cap:
            raise BudgetExceeded(f"cost ledger {self.total():.4f} >= cap {self.cap}")


# ---- API key + client -----------------------------------------------------------------------

def load_api_key() -> str:
    key = resolve_api_key("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not found (env var or .env)")
    return key


# ---- tools ---------------------------------------------------------------------------------

_PLAN = {"type": "string", "description": "Your plan, <= 150 words. First reply: the full step plan for the target (every fold in order, with labels). Later: the remaining steps, updated."}
_LINE = {"type": "string", "description": 'Fold line as JSON, e.g. {"onto":["C","A"]} or {"horizontal":"I"} (labels such as A, B, ...).'}
_MOVE = {"type": "string", "description": 'A point on the side that moves: a label like "D", or JSON like {"mid":["D","I"]}.'}
_LAYERS = {"type": "string", "description": 'Optional. "all" (default) or JSON like {"top":1} or {"top":2,"flap":true}.'}


def _tool(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": {**props, "plan": _PLAN}, "required": [*required, "plan"]}}}


TOOLS = [
    _tool("valley", "Valley fold: the moving part turns over along the line and lands ON TOP (toward you).",
          {"line": _LINE, "move": _MOVE, "layers": _LAYERS,
           "tuck": {"type": "string", "description": 'Optional: "inside" = tuck the flap into the pocket.'}}, ["line"]),
    _tool("mountain", "Mountain fold: the moving part turns over along the line and goes UNDERNEATH.",
          {"line": _LINE, "move": _MOVE, "layers": _LAYERS}, ["line"]),
    _tool("squash", "Squash fold: lift a flap upright on its hinge, open it and flatten it symmetrically.",
          {"move": _MOVE, "hinge": {"type": "string", "description": 'Edge the flap is attached by, JSON pair, e.g. ["I","A"].'},
           "spine": {"type": "string", "description": 'The flap\'s folded edge that gets opened, JSON pair, e.g. ["I","B"].'},
           "layers": _LAYERS}, ["move", "hinge", "spine"]),
    _tool("petal", "Petal fold: lift the bottom corner of the top layer up along a horizontal line; the sides fold in.",
          {"move": _MOVE, "line": _LINE, "layers": _LAYERS}, ["move", "line"]),
    _tool("inside_reverse", "Inside reverse fold: push the tip of a flap (beyond the line) inside between its own layers.",
          {"line": _LINE, "move": _MOVE, "layers": _LAYERS}, ["line", "move"]),
    _tool("flip", "Turn the paper over (not a fold).",
          {"axis": {"type": "string", "description": '"vertical" (default, left-right) or "horizontal".'}}, []),
    _tool("rotate", "Rotate the paper counter-clockwise (not a fold).",
          {"degrees": {"type": "number", "description": "Degrees, counter-clockwise."}}, ["degrees"]),
    _tool("unfold", "Open the most recent fold again (its crease stays). Counts as a move.", {}, []),
    _tool("undo", "Take back your last action(s) completely (back to the earlier paper state).",
          {"n": {"type": "integer", "description": "How many actions to take back (default 1)."}}, []),
    _tool("inspect", "Ask for exact details (not a fold). Queries: 'layers_at P', 'under P', 'creases_at P' (P = a label).",
          {"query": {"type": "string", "description": "e.g. layers_at I"}}, ["query"]),
    _tool("submit", "Finish: the current paper is judged against the target. Only when you think it matches.", {}, []),
]
MOVE_TOOLS = {"valley", "mountain", "squash", "petal", "inside_reverse", "flip", "rotate", "unfold"}

SYSTEM_PROMPT = """You are an origami agent. You fold a virtual square sheet of paper (side 1) step by step until it looks like a target model. You act only through the tools.

COORDINATES: x to the right, y up, as you look at the paper now. The sheet starts flat at (0,0)-(1,1), coloured side up ("front" view = what you see; "back" view = the paper turned over).

STICKER LABELS (the names of points; stable for the whole episode):
- Every spot of the ORIGINAL sheet has a short label (A, B, C, ...). A label is a sticker glued to that spot of paper: it follows the spot wherever the paper moves, so after folding, labels end up in new places and several labels can be stacked on the same spot.
- Start: A bottom-left corner, B bottom-right, C top-right, D top-left, E bottom-middle, F right-middle, G top-middle, H left-middle, I centre. New labels (J, K, ...) appear when creases cross; the observation lists every label with its original-sheet coordinates.
- Each turn "Where stickers are now" tells you where the visible labels currently are (top/bottom/left/right/centre of the current outline) and which labels are stacked on one spot. Always check it: a label does NOT mean "the corner that is bottom-left now".
- Other ways to give a point: {"mid":["D","I"]} (midpoint), {"lerp":["A","C"],"t":0.25} (25% of the way), [x,y] (current view coordinates; if it lands within 0.02 of a label spot it snaps to it and the result tells you).
- {"intersect":[L1,L2]}: the point where two lines cross. Each L is a pair ["P","Q"] (the line through P and Q) or any fold-line spec below. Example, the centre: {"intersect":[["A","C"],["B","D"]]}. Example, where an angle bisector meets an edge: {"intersect":[{"bisect":[["D","B"],["D","C"]]},["B","C"]]}. Use it for points like "sqrt(2)-1 along an edge": typed decimals are too inexact for the physics check.

FOLD LINES (argument "line", as JSON):
- {"onto":["P","Q"]}: crease that puts point P exactly onto point Q (P's side moves; no "move" needed).
- {"through":["P","Q"]}: crease through P and Q. Give "move" = a point on the side that should move.
- {"horizontal":"P"} / {"vertical":"P"}: horizontal / vertical crease through P (give "move").
- {"perpendicular":["P","Q"],"at":"X"}: crease through X at right angles to line PQ (give "move"). Example: {"perpendicular":["A","B"],"at":"I"} = vertical crease through the centre.
- {"bisect":[["P","Q"],["R","S"]]}: crease that lays line P->Q onto line R->S (angle bisector at the point where they meet).
- {"point_onto_line":"P","line":["R","S"],"pivot":"X"}: crease through X that puts P onto line RS (the "line" may also be a line spec, e.g. {"horizontal":"B"}). Add "near":"T" to choose, when there are two solutions, the one where P lands closest to T. Example: {"point_onto_line":"B","line":["D","C"],"pivot":"D","near":"C"}.
- {"point_onto_line":"P","line":["R","S"],"perpendicular_to":["U","V"]}: instead of a pivot, the crease is perpendicular to line UV, so P slides parallel to UV onto line RS. Example: {"point_onto_line":"B","line":["D","C"],"perpendicular_to":["A","B"]}.

FOLDS:
- valley: the moving part lands ON TOP (toward you). mountain: it goes UNDERNEATH.
- layers: default "all" layers on the moving side; {"top":1} = only the top layer (e.g. fold down the front flap only); {"top":2,"flap":true} = only the flap under "move".
- tuck "inside" (valley): tuck the flap into the pocket instead of on top.
- squash, petal, inside_reverse: classic techniques (see tool descriptions). For squash and petal, "move" must be a point inside the flap, not on its hinge.
- flip / rotate are not folds. unfold re-opens the last fold. undo takes back your last action.

RULES:
- Every move is checked physically (no stretching, tearing or passing paper through paper). An illegal move is REJECTED with a reason code and the paper stays unchanged: read the reason and change the move.
- Only the final state is judged, when you call submit: the outline AND the visible colours seen from the front AND from the back must match the target. Any rotation, turning over, mirroring or position is fine; the size must match (same sheet).
- Turn limits are shown in each observation; flip / rotate / inspect-only replies have their own separate limit.
- If the memory says you were here before, do not repeat moves that failed.

EXAMPLES (labels as at the start):
- valley line={"onto":["C","A"]}: fold the top-right corner onto the bottom-left corner (diagonal triangle).
- valley line={"horizontal":"I"} move="D": fold the top half down onto the bottom half.
- valley line={"onto":["D","I"]} layers={"top":1}: fold only the top layer's D corner to the centre.
- mountain line={"vertical":"I"} move="B": fold the right half behind.

OBSERVATION: each turn you get the current paper as text (outline with corner angles and labels, flaps, visible points with layer count d and colour on top seen from front F / back B: c = coloured, w = white), where the stickers are now, history, last result and memory. Images (if given) show front and back views with the same labels.

REPLY: call one tool (you may call up to 3 in order if you are sure). Every call must include "plan" (<= 150 words).
- On your FIRST reply: think carefully about the target. Recall how this model is folded in real life and write the full step plan in "plan" (every fold in order, using labels, incl. which layers move and where flips/rotations are needed), then make the first move.
- On later replies: check the observation against your plan, correct the plan if the paper is not where you expected, and keep "plan" up to date (remaining steps)."""


# ---- prompt building -----------------------------------------------------------------------

def _img_part(png: bytes) -> dict:
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}


def _words(s: str, n: int) -> str:
    w = (s or "").split()
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")


@dataclass
class AgentMemory:
    plan: str = ""
    reasonings: list = field(default_factory=list)  # last 2 short reasonings
    notes: list = field(default_factory=list)  # results of the last reply's calls (inspect output, ...)
    format_error: str | None = None


def build_turn_text(env: FoldEnv, mem: AgentMemory, turn: int, max_turns: int, images: bool,
                    layer: CommandLayer | None = None, free_used: int = 0, max_free: int | None = None) -> str:
    """turn / max_turns count ACTION turns (replies that are not only flip/rotate/inspect);
    free_used / max_free count the flip/rotate/inspect-only replies separately."""
    obs = env.observe(images=False)
    free = f" | flip/rotate/inspect-only replies so far {free_used}" + (f" of {max_free}" if max_free else "")
    L = [f"=== ACTION TURN {turn} of {max_turns}{free} | folds so far {env.moves} ===",
         "Current paper:", obs.text]
    if layer is not None:
        L.append(layer.stickers_text(obs))
    if mem.notes:
        L.append("Results of your last reply:")
        L += [f"  {n}" for n in mem.notes]
    if mem.format_error:
        L.append(f"FORMAT ERROR in your last reply: {mem.format_error}. Reply with a tool call.")
    if mem.plan:
        L.append(f"Your current plan: {mem.plan}")
    elif turn == 1:
        what = "goal" if getattr(env, "hide_target", False) else "target"
        L.append(f"You have no plan yet: write the full step plan for the {what} in the plan field of your first call.")
    if mem.reasonings:
        L.append("Your recent reasoning (short):")
        L += [f"  - {r}" for r in mem.reasonings]
    if images:
        L.append("Image: current paper, front and back views, with the same labels.")
    L.append("Now call the next tool.")
    return "\n".join(L)


def goal_parts(env: FoldEnv, images: bool, target_png: bytes | None) -> list:
    parts = [{"type": "text", "text": "GOAL\n" + env.goal_text() +
              ("\nImage: the TARGET, front and back views." if images else "")}]
    if images and target_png:
        parts.append(_img_part(target_png))
    return parts


# ---- reply parsing -------------------------------------------------------------------------

def _parse_args(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    raw = (raw or "").strip()
    if not raw:
        return {}
    return json.loads(raw)


_TEXT_CALL = re.compile(r"\{[\s\S]*\}")


def extract_calls(msg: dict) -> list[tuple[str, dict, str | None]]:
    """-> [(tool name, args, tool_call_id)]. Falls back to a JSON object in the text content
    ({"name"/"tool": ..., "arguments"/"args": {...}}). Raises ValueError on malformed output."""
    calls = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        name = fn.get("name")
        try:
            args = _parse_args(fn.get("arguments"))
        except json.JSONDecodeError as e:
            raise ValueError(f"tool call arguments are not valid JSON ({e.msg})") from None
        if name not in {t["function"]["name"] for t in TOOLS}:
            raise ValueError(f"unknown tool {name!r}")
        calls.append((name, args, tc.get("id")))
    if calls:
        return calls
    content = msg.get("content") or ""
    m = _TEXT_CALL.search(content if isinstance(content, str) else "")
    if m:
        try:
            obj = json.loads(m.group(0))
            name = obj.get("name") or obj.get("tool")
            args = obj.get("arguments") or obj.get("args") or obj.get("parameters") or {}
            if name in {t["function"]["name"] for t in TOOLS}:
                return [(name, _parse_args(args), None)]
        except (json.JSONDecodeError, AttributeError):
            pass
    raise ValueError("no tool call in the reply")


# ---- the loop ------------------------------------------------------------------------------

def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def run_dir(model: str, target: str, root: Path | None = None) -> Path:
    base = (root or REPO / "runs") / _dt.date.today().isoformat() / _safe(model) / _safe(target)
    d, n = base, 1
    while d.exists():
        n += 1
        d = base.with_name(f"{base.name}_{n}")
    d.mkdir(parents=True)
    return d


def _strip_images(messages: list) -> None:
    for m in messages:
        if isinstance(m.get("content"), list):
            m["content"] = [p if p.get("type") != "image_url" else {"type": "text", "text": "[older image removed]"}
                            for p in m["content"]]


def run_episode(target: str | None, model: str, images: bool = True, max_turns: int | None = None,
                context: str = "rebuilt", feedback: str = "F1", out_root: Path | None = None,
                client: LLMClient | None = None, log=print, reasoning_tokens: int = 0,
                ledger: CostLedger | None = None, reasoning_effort: str | None = "medium",
                max_free_turns: int | None = None, goal=None, events=None, out="auto",
                plan_text: str | None = None) -> dict:
    """goal: agents.goal.Goal (example / text / image); default = example goal of `target`.
    events: agents.events.RunEvents (event stream, cancel flag, per-run $ limit).
    out: "auto" = a new dir under out_root (or runs/), a Path = that dir, None = write no files."""
    from .events import Cancelled, NullEvents
    from .goal import Goal
    from .menu_agent import plan_part, score_variants

    goal = goal or Goal.example(target)
    ev = events or NullEvents()
    env = goal.make_env(feedback)
    sysp = goal.system_prompt("command")
    default_turns = goal.default_max_turns(env)
    max_turns = max_turns or default_turns  # ACTION turns; flip/rotate/inspect-only replies have their own limit
    max_free_turns = max_free_turns if max_free_turns is not None else default_turns
    layer = CommandLayer(env)  # neutral sticker labels, snapping, clearer errors
    client = client or OpenRouterClient(model, log=log, reasoning_tokens=reasoning_tokens,
                                        reasoning_effort=reasoning_effort)
    gparts = goal.parts(env, images)
    pparts = plan_part(plan_text)
    if out == "auto":
        out = run_dir(model, env.name, out_root)
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "system.txt").write_text(sysp + "\n\nTOOLS:\n" + json.dumps(TOOLS, indent=1))
        if goal.kind == "example" and images:
            (out / "target.png").write_bytes(env.target_png())
        (out / "goal.txt").write_text(gparts[0]["text"])
        if plan_text:
            (out / "plan.txt").write_text(plan_text)
    gparts = gparts + pparts
    ev.started(env, system_prompt=sysp, tools=TOOLS, goal_parts=gparts, plan_parts=len(pparts), goal=goal,
               limits={"max_turns": max_turns, "max_free_turns": max_free_turns, "usd": ev.max_usd})
    mem = AgentMemory()
    chat: list = []
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "calls": 0}
    counts = {"legal": 0, "illegal": 0, "format_errors": 0, "ran_out_of_tokens": 0, "undo": 0, "inspect": 0}
    cost = 0.0
    stop_reason, consecutive_fmt = "turn_limit", 0
    logf = (out / "turns.jsonl").open("w") if out is not None else io.StringIO()
    log(f"run {model} on {env.name} (par {env.par}, max action turns {max_turns}, free turns {max_free_turns}, "
        f"images={images}, context={context}) -> {out}")

    action_turns = free_turns = turn = 0
    while action_turns < max_turns and free_turns <= max_free_turns:
        turn += 1
        action_turns += 1  # given back below if the reply was only flip/rotate/inspect
        turn_text = build_turn_text(env, mem, action_turns, max_turns, images, layer, free_turns, max_free_turns)
        cur_png = env.observe(images=True).folded_png if images else None
        if cur_png:
            ev.image(cur_png, "current paper, front and back")
        tail = [{"type": "text", "text": turn_text}] + ([_img_part(cur_png)] if cur_png else [])
        if context == "chat":
            _strip_images(chat)
            if not chat:
                chat = [{"role": "system", "content": sysp},
                        {"role": "user", "content": gparts + tail}]
            else:
                chat.append({"role": "user", "content": tail})
            messages = chat
        else:
            messages = [{"role": "system", "content": sysp},
                        {"role": "user", "content": gparts + tail}]
        rec = {"turn": turn, "prompt_text": turn_text, "prompt_chars": sum(
            len(p.get("text", "")) for m in messages for p in (m["content"] if isinstance(m["content"], list) else [{"text": m["content"]}])),
            "images": sum(1 for m in messages if isinstance(m["content"], list) for p in m["content"] if p.get("type") == "image_url")}
        t0 = time.time()
        try:
            ev.guard()
            if ledger is not None:
                ledger.check()
            ev.status("thinking", turn=turn)
            ev.request(turn, messages)
            resp = as_result(client.chat(messages, TOOLS))
        except Cancelled as e:
            stop_reason = "stopped"
            rec["error"] = str(e)
            logf.write(json.dumps(rec) + "\n")
            log("  STOP (stopped by the user)")
            break
        except BudgetExceeded as e:
            stop_reason = "budget"
            rec["error"] = str(e)
            logf.write(json.dumps(rec) + "\n")
            ev.error("budget", str(e), turn)
            log(f"  STOP (budget): {e}")
            break
        except DailyLimit as e:
            stop_reason = "no_credit" if isinstance(e, NoCredit) else "daily_limit"
            rec["error"] = str(e)
            logf.write(json.dumps(rec) + "\n")
            ev.error(stop_reason, str(e), turn)
            log(f"  STOP (daily limit / credits): {str(e)[:200]}")
            break
        except APIError as e:
            stop_reason = "api_error"
            rec["error"] = str(e)
            logf.write(json.dumps(rec) + "\n")
            ev.error("provider_error", str(e), turn)
            log(f"  STOP (api error): {str(e)[:200]}")
            break
        rec["latency_s"] = round(time.time() - t0, 1)
        ev.response(turn, resp)
        totals["calls"] += 1
        u = resp.usage
        rec["usage"] = {"prompt": u.input, "completion": u.output, "cached": u.cached,
                        "reasoning": u.reasoning, "cost": u.cost_usd}
        totals["prompt_tokens"] += u.input
        totals["completion_tokens"] += u.output
        totals["cached_tokens"] += u.cached
        c = float(u.cost_usd or 0.0)
        cost += c
        if ledger is not None and c:
            ledger.add(c, f"{model}/{env.name}")
        rec["provider"] = resp.provider
        msg = resp.message()
        rec["finish_reason"] = resp.finish_reason
        rec["reply"] = dict(msg)
        mem.notes, mem.format_error = [], None
        try:
            calls = extract_calls(msg)
        except ValueError as e:
            consecutive_fmt += 1
            if resp.finish_reason == "length":
                counts["ran_out_of_tokens"] += 1
                rec["error_kind"] = "ran_out_of_tokens"
                e = ValueError("your reply was cut off before any tool call (you thought too long); "
                               "keep your thinking shorter and call a tool")
            else:
                counts["format_errors"] += 1
                rec["error_kind"] = "format_error"
            mem.format_error = str(e)
            rec["format_error"] = str(e)
            ev.error(rec["error_kind"], str(e), turn)
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            log(f"  turn {turn}: {rec['error_kind'].upper()} ({e}); content: {_words(str(msg.get('content')), 30)}")
            if context == "chat":
                chat.append({"role": "assistant", "content": msg.get("content") or ""})
            if consecutive_fmt >= 3:
                stop_reason = "no_action_3x"
                break
            continue
        consecutive_fmt = 0
        reason_txt = (msg.get("content") or "").strip()
        if reason_txt:
            reason_txt = _words(reason_txt, 100)
        elif (msg.get("reasoning") or "").strip():  # thinking models: keep the conclusion (last words)
            w = msg["reasoning"].split()
            reason_txt = ("... " if len(w) > 100 else "") + " ".join(w[-100:])
        if reason_txt:
            mem.reasonings = (mem.reasonings + [reason_txt])[-2:]
        results, submitted = [], False
        tool_msgs = []
        for name, args, tcid in calls[:MAX_CALLS_PER_REPLY]:
            if args.get("plan"):
                mem.plan = _words(str(args["plan"]), 150)
            if name == "submit":
                submitted = True
                res = {"tool": "submit"}
            else:
                ev.status("folding", turn=turn)
                res = layer.run(name, args)
                ev.action(turn, res)
                if name == "undo":
                    counts["undo"] += 1
                elif name == "inspect":
                    counts["inspect"] += 1
                else:
                    counts["legal" if res["ok"] else "illegal"] += 1
            results.append(res)
            note = res.get("text") or res["tool"]
            if res["tool"] == "inspect":
                note = f"inspect {res['query']}: {res['text']}"
            mem.notes.append(note)
            tool_msgs.append({"role": "tool", "tool_call_id": tcid or f"call_{turn}_{len(tool_msgs)}",
                              "content": note})
            if submitted or (res.get("ok") is False):
                break
        rec["calls"] = results
        rec["plan"] = mem.plan
        if calls and all(n in ("flip", "rotate", "inspect") for n, _, _ in calls[:MAX_CALLS_PER_REPLY]):
            action_turns -= 1  # view / inspect-only reply: counted against the separate free limit
            free_turns += 1
        logf.write(json.dumps(rec, default=str) + "\n")
        logf.flush()
        log(f"  turn {turn}: " + " | ".join(r.get("text") or r["tool"] for r in results)[:300])
        if context == "chat":
            am = {"role": "assistant", "content": msg.get("content") or ""}
            if msg.get("tool_calls"):
                am["tool_calls"] = [tc for tc in msg["tool_calls"]][:len(tool_msgs)]
                chat.append(am)
                chat.extend(tool_msgs)
            else:
                chat.append(am)
        if submitted:
            stop_reason = "submitted"
            break

    score = env.submit()
    variants = score_variants(env) if goal.scored else {}
    logf.close()
    st = env.state
    final_png, cp_png = env.observe(images=True).folded_png, render_cp_png(st)
    name = env.name if goal.scored else goal.kind
    sol = {"name": f"{name}_by_{_safe(model)}", "title": f"{env.title if goal.kind == 'example' else name} (agent: {model})",
           "paper": env.model.get("paper", {"size": 1.0}),
           "steps": [{**s, "description": step_line(s)} for s in env.trajectory]}
    summary = {"model": model, "target": env.name if goal.scored else None, "goal": goal.kind, "context": context,
               "images": images, "feedback": feedback,
               "stop_reason": stop_reason, "turns": totals["calls"], "max_turns": max_turns, **counts,
               "tokens": totals, "cost_usd": round(cost, 6), "http_requests": getattr(client, "calls", None),
               "score": score.to_dict(), "variants": variants,
               "pass_any": any(v["m2"] for v in variants.values()) if variants else score.passed,
               "dir": str(out) if out is not None else None}
    if out is not None:
        (out / "final_folded.png").write_bytes(final_png)
        (out / "final_cp.png").write_bytes(cp_png)
        (out / "solution.json").write_text(json.dumps(sol, indent=1))
        (out / "score.json").write_text(json.dumps(summary, indent=1, default=str))
    ev.finished(stop_reason, score=score, variants=variants, solution=sol,
                final_pngs=[(final_png, "final paper, front and back"), (cp_png, "crease pattern")])
    log(f"  done ({stop_reason}): {summarize_score(score)} | legal {counts['legal']} illegal {counts['illegal']} "
        f"format {counts['format_errors']} out-of-tokens {counts['ran_out_of_tokens']} | ${cost:.4f} | tokens in {totals['prompt_tokens']} out {totals['completion_tokens']} "
        f"cached {totals['cached_tokens']}")
    return summary


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="origami agent")
    add_args(ap)
    return cmd_agent(ap.parse_args(argv))


def add_args(ap) -> None:
    ap.add_argument("--target", required=True, help="target model JSON, e.g. models/cup.json")
    ap.add_argument("--model", default=None, help="model id as the provider names it (e.g. anthropic/claude-opus-5.5 "
                    "on openrouter, claude-opus-5-5 on anthropic)")
    ap.add_argument("--provider", default="openrouter",
                    help="openrouter (default) | anthropic | openai | ollama | lmstudio | scripted; keys from env/.env "
                         "(OPENROUTER_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY)")
    ap.add_argument("--base-url", default=None, help="override the provider's base URL (any OpenAI-compatible server)")
    ap.add_argument("--dry-run", action="store_true",
                    help="no API: a scripted client replays the target model's own steps (checks the loop end to end)")
    ap.add_argument("--no-images", action="store_true", help="text-only prompts")
    ap.add_argument("--max-steps", type=int, default=None, help="max action turns (replies that are not only flip/rotate/inspect); default 3 x par")
    ap.add_argument("--context", choices=("rebuilt", "chat"), default="rebuilt")
    ap.add_argument("--feedback", choices=("F0", "F1", "F2", "F3"), default="F1")
    ap.add_argument("--runs-dir", default=None, help="log root (default runs/)")
    ap.add_argument("--budget-usd", type=float, default=None,
                    help="hard cap on cumulative cost in the ledger (stop when reached)")
    ap.add_argument("--ledger", default=None, help="cost ledger JSON (default runs/cost_ledger.json)")
    ap.add_argument("--reasoning-effort", choices=("low", "medium", "high", "xhigh", "max", "off"), default="medium",
                    help="thinking effort, sent in the provider's own format (default medium; off = none/minimal)")
    ap.add_argument("--reasoning-tokens", type=int, default=0,
                    help="if > 0: send reasoning.max_tokens instead of effort (some providers ignore it)")
    ap.add_argument("--plan-file", default=None,
                    help="menu/decision modes: text file with an expert fold plan, shown every turn")
    ap.add_argument("--max-free-turns", type=int, default=None,
                    help="separate limit for flip/rotate/inspect-only replies; default 3 x par")


def make_cli_client(args, log=print):
    """--provider / --model / --base-url / --dry-run -> LLMClient (origami.llm presets, key from env/.env)."""
    from ..llm import client_from_cli

    if getattr(args, "dry_run", False):
        args.model = args.model or "scripted"
        return client_from_cli("scripted", args.model, script=args.target, log=log)
    if not args.model:
        raise SystemExit("--model is required (or use --dry-run)")
    return client_from_cli(getattr(args, "provider", "openrouter"), args.model, base_url=getattr(args, "base_url", None),
                           images=not args.no_images, reasoning=args.reasoning_effort,
                           reasoning_tokens=args.reasoning_tokens, log=log)


def cmd_agent(args) -> int:
    from .menu_agent import goal_from_args, read_plan

    client = make_cli_client(args)
    s = run_episode(args.target, args.model, client=client, images=not args.no_images, max_turns=args.max_steps,
                    goal=goal_from_args(args), plan_text=read_plan(args),
                    context=args.context, feedback=args.feedback,
                    out_root=Path(args.runs_dir) if args.runs_dir else None,
                    reasoning_tokens=args.reasoning_tokens, reasoning_effort=args.reasoning_effort,
                    max_free_turns=args.max_free_turns,
                    ledger=CostLedger(args.ledger or REPO / "runs" / "cost_ledger.json", args.budget_usd))
    return 0 if s["stop_reason"] not in ("daily_limit", "no_credit", "api_error", "budget") else 2


if __name__ == "__main__":
    raise SystemExit(main())
