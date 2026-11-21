"""MENU action mode: each turn the simulator lists sensible legal folds (origami.menu) and the agent
picks a number. Separates "can it plan a fold sequence" from "can it describe fold geometry".

    uv run origami agent --target models/cup.json --model <openrouter id> --actions menu

Tools: choose(option) + flip / rotate / undo / submit, plus an escape hatch: the command-mode fold
tools (valley / mountain / squash / petal / inside_reverse with line args, raw [x,y] points allowed)
for a "custom fold" that is not in the list. Each move is logged with its source (menu / custom).
Turn 1 asks for a full plan (whole fold sequence); it is shown again every turn.

The loop is policy-driven so it can be tested without an API: `run_menu_episode(target, policy)`
where policy(view) -> list of (tool name, args). `llm_policy(...)` wraps an OpenRouter model.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..env import FoldEnv, summarize_score
from ..menu import THUMB_MAX, candidates, menu_text, thumbnails_png
from .command_layer import CommandLayer, stickers_text
from ..observe import render_cp_png, step_line
from ..llm import LLMClient, as_result
from .llm_agent import (REPO, TOOLS, APIError, BudgetExceeded, CostLedger, DailyLimit, NoCredit,
                        OpenRouterClient, _img_part, _parse_args, _safe, _tool, _words, call_to_action,
                        goal_parts, make_cli_client, run_dir)

CUSTOM_TOOLS = ("valley", "mountain", "squash", "petal", "inside_reverse")

MENU_TOOLS = [
    _tool("choose", "Make the fold with this number from the menu (counts as a move).",
          {"option": {"type": "integer", "description": "Option number from the current menu."}}, ["option"]),
    _tool("more", "Show more options of one menu section next turn (free, not a move).",
          {"section": {"type": "string", "description": "Section name in [brackets], e.g. \"flap\"."}}, ["section"]),
    *[t for t in TOOLS if t["function"]["name"] in CUSTOM_TOOLS],
    *[t for t in TOOLS if t["function"]["name"] in ("flip", "rotate", "undo", "submit")],
]
for _t in MENU_TOOLS:  # turn-1 plan may be long
    if _t["function"]["name"] == "choose":
        _t["function"]["parameters"]["properties"]["plan"] = {
            "type": "string",
            "description": "Turn 1: your FULL plan, the whole fold sequence from flat sheet to finished model (numbered). "
                           "Later turns: <= 60 words, what you do next and why (update the plan if needed)."}

MENU_SYSTEM_PROMPT = """You are an origami agent. You fold a virtual square sheet of paper (side 1) step by step until it looks like a target model.

EACH TURN you see the current paper (text + images) and a numbered MENU of legal next folds, computed by the simulator from the current paper (corners, edges, creases). Every menu option is physically possible. Descriptions use the current view: "top", "left", ... ("now at top-left") are as you look at the paper now. Capital letters (A, B, C, ...) are stickers on fixed spots of the ORIGINAL sheet; they follow the paper and are drawn on the images. Each option has an "after:" line = a short fact summary of the paper after that fold (outline, colours, layers, flaps). "all layers" folds every layer on the moving side together; "front layer only" just the front layer; "only this one flap" a single flap; "open the pocket and slide ... inside (between the layers)" puts the flap between layers. Pictures are given for the first 10 options only; the others are text only.

ACTIONS:
- choose(option=N): make fold N from the menu. This is the normal way to fold.
- more(section="flap"): the menu shows only the simplest options of each [section]; this lists more of that section next turn (free).
- If the fold you want is NOT listed, make a custom fold with valley / mountain / squash / petal / inside_reverse (line JSON like {"onto":["C","K"]}, {"through":["P","Q"]}, {"horizontal":"P"}; points may be sticker letters or [x,y] current coordinates). Custom folds are checked too and may be rejected.
- flip (turn over) and rotate are free and do not count as moves; the menu is rebuilt after them.
- undo takes back your last action. submit ends the episode.

RULES: only the final paper is judged (outline + visible colours, front AND back; rotation / turning over / mirroring are fine). Use few moves (par = best known move count). Mirror-image options are listed once (folding the left or the right side first gives the same result up to mirroring).

TURN 1: before acting, write your FULL plan: the whole sequence of folds from the flat sheet to the target, in plain words, numbered. Put it in the "plan" argument of your first call. Then follow it, adapting when the paper does not look as expected.

REPLY: think, then call ONE tool (up to 3 in order if you are sure). Every call includes "plan"."""

# Text-goal mode: the goal is only a sentence of text (no target picture / outline / par / closeness score).
_TEXT_SWAPS = [
    ("step by step until it looks like a target model.",
     "step by step until it becomes the object described in the GOAL text. There is no picture of the result: "
     "you decide what it should look like."),
    ("RULES: only the final paper is judged (outline + visible colours, front AND back; rotation / turning over / "
     "mirroring are fine). Use few moves (par = best known move count).",
     "RULES: only the final paper is judged: is it the object described in the GOAL (rotation / turning over / "
     "mirroring are fine)? Use few moves."),
    ("the whole sequence of folds from the flat sheet to the target", "the whole sequence of folds from the flat sheet to the finished object"),
]
MENU_SYSTEM_PROMPT_TEXT = MENU_SYSTEM_PROMPT
for _a, _b in _TEXT_SWAPS:
    assert _a in MENU_SYSTEM_PROMPT_TEXT, _a
    MENU_SYSTEM_PROMPT_TEXT = MENU_SYSTEM_PROMPT_TEXT.replace(_a, _b)
TEXT_GOAL_MAX_TURNS = 30  # fixed (3 x par would leak par through "TURN k/N")


def text_goal_parts(goal_text: str) -> list:
    """Goal context in text-goal mode: ONLY the given sentence (nothing computed from the target)."""
    return [{"type": "text", "text": "GOAL\n" + goal_text.strip()}]


@dataclass
class MenuView:
    """What a policy sees each turn."""
    env: FoldEnv
    turn: int
    max_turns: int
    options: list
    text: str  # full turn text incl. menu
    images: list = field(default_factory=list)  # PNG bytes: current paper, menu thumbnails


def build_menu_text(env: FoldEnv, opts, turn: int, max_turns: int, plan: str, full_plan: str,
                    notes: list, format_error: str | None, images: bool) -> str:
    obs = env.observe(images=False)
    par = "" if getattr(env, "hide_target", False) else f" (par {env.par})"
    L = [f"=== TURN {turn}/{max_turns} | moves used {env.moves}{par} ===", "Current paper:", obs.text]
    if env.labeler is not None:
        L.append(stickers_text(obs.points, env.state))
    if notes:
        L.append("Results of your last reply:")
        L += [f"  {n}" for n in notes]
    if format_error:
        L.append(f"FORMAT ERROR in your last reply: {format_error}. Reply with a tool call.")
    if full_plan:
        L.append(f"Your overall plan (turn 1): {full_plan}")
    if plan and plan != full_plan:
        L.append(f"Your latest plan: {plan}")
    L.append("MENU (choose one with choose(option=N); or make a custom fold if what you want is not listed):")
    L.append(menu_text(opts, with_after=True, colored_side=env.colored_side))
    if images:
        L.append("Images: (1) current paper, front and back views with sticker labels; "
                 f"(2) the result of options 1-{min(len(opts), THUMB_MAX)}, each labelled with its number "
                 "(front view; orange = coloured side). Later options: text only.")
    if turn == 1:
        L.append("This is turn 1: write your FULL plan (all folds, numbered) in the plan argument, then act.")
    else:
        L.append("Now call the next tool.")
    return "\n".join(L)


def _execute(env: FoldEnv, opts, name: str, args: dict, counts: dict, feedback: str,
             layer: CommandLayer | None = None) -> dict:
    if name == "submit":
        return {"tool": "submit"}
    if name == "undo":
        r = env.undo(int(args.get("n") or 1))
        counts["undo"] += 1
        return {"tool": "undo", "ok": r.ok, "text": r.to_text(feedback) if not r.ok else r.reason}
    if name == "more":
        sec = str(args.get("section", "")).strip().strip("[]\"'")
        return {"tool": "more", "section": sec, "ok": True, "text": f"more options of section [{sec}] are listed below"}
    if name == "choose":
        try:
            n = int(args.get("option"))
        except (TypeError, ValueError):
            n = None
        opt = next((o for o in opts if o.id == n), None)
        if opt is None:
            counts["bad_choice"] += 1
            return {"tool": "choose", "ok": False, "text": f"REJECTED: there is no option {args.get('option')!r} "
                                                             f"(menu has 1..{len(opts)})"}
        r = env.apply(opt.step)
        counts["menu_moves" if r.ok else "illegal"] += 1
        return {"tool": "choose", "option": n, "source": "menu", "action": r.action, "ok": r.ok, "code": r.code,
                "text": f"option {n}: {opt.description} -> " + (env.labeler.sub(r.to_text(feedback)) if env.labeler else r.to_text(feedback))}
    if layer is not None:  # sticker labels -> sheet names, point snapping, label-aware errors
        res = layer.run(name, args)
        if name in ("flip", "rotate"):
            counts["view_ops"] += 1
        else:
            counts["custom_moves" if res["ok"] else "illegal"] += 1
        res["source"] = "custom" if name not in ("flip", "rotate") else "view"
        return res
    action = call_to_action(name, args)
    r = env.apply(action)
    if name in ("flip", "rotate"):
        counts["view_ops"] += 1
    else:
        counts["custom_moves" if r.ok else "illegal"] += 1
    return {"tool": name, "source": "custom" if name not in ("flip", "rotate") else "view", "action": r.action,
            "ok": r.ok, "code": r.code, "text": r.to_text(feedback)}


def run_menu_episode(target: str | None, policy, max_turns: int | None = None, feedback: str = "F1",
                     images: bool = True, thumbs: bool = True, out: Path | None = None, log=print,
                     compute_deletable: bool = True, text_goal: bool = False, goal=None, events=None,
                     started: dict | None = None) -> dict:
    """policy(view: MenuView, memory: dict) -> list[(tool name, args)] or raises ValueError (format error).
    memory keys the policy may set: 'plan', 'full_plan'.
    text_goal: hide everything derived from the target in the observations (name, par); F0/F1 only.
    goal: agents.goal.Goal (overrides target / text_goal). events: agents.events.RunEvents; `started` =
    its run_started arguments (system_prompt, tools, goal_parts, plan_parts), sent once the env exists."""
    from .events import Cancelled, NullEvents
    from .goal import Goal

    if goal is None:
        goal = Goal.from_text("(text goal)", target) if text_goal else Goal.example(target)
    text_goal = goal.hides_target()
    ev = events or NullEvents()
    env = goal.make_env(feedback)
    layer = CommandLayer(env)  # neutral sticker labels (same as command mode)
    max_turns = max_turns or goal.default_max_turns(env)
    if started is not None:
        ev.started(env, limits={"max_turns": max_turns, "usd": ev.max_usd}, **started)
    counts = {"menu_moves": 0, "custom_moves": 0, "illegal": 0, "bad_choice": 0, "undo": 0, "view_ops": 0,
              "format_errors": 0, "more": 0}
    mem = {"plan": "", "full_plan": "", "notes": [], "format_error": None}
    more: dict = {}
    logf = (out / "turns.jsonl").open("w") if out else None
    stop_reason, consecutive_fmt, menu_sizes = "turn_limit", 0, []
    for turn in range(1, max_turns + 1):
        if ev.cancelled:
            stop_reason = "stopped"
            log("  STOP (stopped by the user)")
            break
        t0 = time.time()
        ev.status("menu", "building suggestions", turn=turn)
        opts = candidates(env, more=more, labeler=layer.labels)
        menu_sizes.append(len(opts))
        text = build_menu_text(env, opts, turn, max_turns, mem["plan"], mem["full_plan"], mem["notes"],
                               mem["format_error"], images)
        imgs = []
        if images:
            imgs.append(env.observe(images=True).folded_png)
            ev.image(imgs[-1], "current paper, front and back")
            if thumbs and opts:
                imgs.append(thumbnails_png(opts, env.colored_side, max_n=THUMB_MAX))
                ev.image(imgs[-1], f"menu options 1-{min(len(opts), THUMB_MAX)}")
        view = MenuView(env, turn, max_turns, opts, text, imgs)
        rec = {"turn": turn, "menu_size": len(opts), "menu_s": round(time.time() - t0, 1),
               "menu": [o.description for o in opts], "prompt_text": text}
        mem["notes"], mem["format_error"] = [], None
        try:
            calls = policy(view, mem)
        except ValueError as e:
            counts["format_errors"] += 1
            consecutive_fmt += 1
            mem["format_error"] = str(e)
            rec["format_error"] = str(e)
            ev.error("format_error", str(e), turn)
            if logf:
                logf.write(json.dumps(rec, default=str) + "\n")
            log(f"  turn {turn}: FORMAT ERROR ({e})")
            if consecutive_fmt >= 3:
                stop_reason = "no_action_3x"
                break
            continue
        except Cancelled:
            stop_reason = "stopped"
            log("  STOP (stopped by the user)")
            break
        except (BudgetExceeded, DailyLimit, APIError) as e:
            stop_reason = ("budget" if isinstance(e, BudgetExceeded) else "no_credit" if isinstance(e, NoCredit)
                           else "daily_limit" if isinstance(e, DailyLimit) else "api_error")
            rec["error"] = str(e)
            ev.error("provider_error" if stop_reason == "api_error" else stop_reason, str(e), turn)
            if logf:
                logf.write(json.dumps(rec, default=str) + "\n")
            log(f"  STOP ({stop_reason}): {str(e)[:200]}")
            break
        consecutive_fmt = 0
        results, submitted = [], False
        for name, args in calls[:3]:
            if args.get("plan"):
                if turn == 1 and not mem["full_plan"]:
                    mem["full_plan"] = _words(str(args["plan"]), 250)
                mem["plan"] = _words(str(args["plan"]), 60)
            if name not in ("more", "submit"):
                ev.status("folding", turn=turn)
            res = _execute(env, opts, name, args, counts, feedback, layer)
            ev.action(turn, res)
            results.append(res)
            if name == "more":
                more[res["section"]] = more.get(res["section"], 1) + 1
                counts["more"] += 1
            elif res.get("ok") and name != "submit":
                more = {}
            mem["notes"].append(res.get("text") or res["tool"])
            if name == "submit":
                submitted = True
                break
            if res.get("ok") is False:
                break
            if name not in ("undo", "more") and res.get("ok") and len(calls) > 1:
                ev.status("menu", "building suggestions", turn=turn)
                opts = candidates(env, more=more, labeler=layer.labels)  # menu numbers refer to the new paper
        rec["calls"] = results
        rec["plan"] = mem["plan"]
        rec.update(mem.get("_rec") or {})
        if logf:
            logf.write(json.dumps(rec, default=str) + "\n")
            logf.flush()
        log(f"  turn {turn}: " + " | ".join(r.get("text") or r["tool"] for r in results)[:300])
        if submitted:
            stop_reason = "submitted"
            break
    score = env.submit(compute_deletable)
    if logf:
        logf.close()
    variants = score_variants(env) if goal.scored else {}
    moves = counts["menu_moves"] + counts["custom_moves"]
    summary = {"target": env.name, "actions": "menu", "stop_reason": stop_reason, **counts,
               "menu_share": round(counts["menu_moves"] / moves, 3) if moves else None,
               "menu_size_mean": round(sum(menu_sizes) / len(menu_sizes), 1) if menu_sizes else None,
               "menu_size_max": max(menu_sizes, default=0), "score": score.to_dict(),
               "full_plan": mem["full_plan"], "variants": variants,
               "pass_any": any(v["m2"] for v in variants.values()) if variants else score.m2,
               "text_goal": text_goal, "goal": goal.kind}
    name = env.name if goal.scored else goal.kind
    sol = {"name": f"{name}_menu", "title": f"{env.title if goal.kind == 'example' else name} (menu agent)",
           "paper": env.model.get("paper", {"size": 1.0}),
           "steps": [{**s, "description": s.get("description") or step_line(s)} for s in env.trajectory]}
    if out or events is not None:
        final_png, cp_png = env.observe(images=True).folded_png, render_cp_png(env.state)
    if out:
        (out / "final_folded.png").write_bytes(final_png)
        (out / "final_cp.png").write_bytes(cp_png)
        (out / "solution.json").write_text(json.dumps(sol, indent=1))
    if events is not None:
        ev.finished(stop_reason, score=score, variants=variants, solution=sol,
                    final_pngs=[(final_png, "final paper, front and back"), (cp_png, "crease pattern")])
    for vn, v in variants.items():
        log(f"  variant {vn}: M1={v['m1']}({v['M1']:.2f}) M2={v['m2']}({v['M2']:.2f})")
    log(f"  done ({stop_reason}): {summarize_score(score)} | menu {counts['menu_moves']} custom "
        f"{counts['custom_moves']} illegal {counts['illegal']}")
    return summary


def score_variants(env: FoldEnv) -> dict:
    """Match the final paper against the target and its accepted alternative endings (model JSON
    "alternatives": [paths relative to the model file], e.g. hat: last flap tucked inside / folded behind)."""
    from ..env import run_checked
    from ..match import match

    alts = env.model.get("alternatives") or []
    if not alts:
        return {}
    out = {}
    base = Path(env.target_path).parent if env.target_path else REPO / "models"
    states = [(env.name, env.target_state)]
    for a in alts:
        m = json.loads((base / a).read_text())
        st = run_checked(list(m["steps"]), float(m.get("paper", {}).get("size", 1.0)))
        if st is None:
            raise ValueError(f"alternative {a} does not replay cleanly")
        states.append((m.get("name", a), st))
    for nm, st in states:
        r = match(st, env.state)
        out[nm] = {"m1": bool(r.m1), "m2": bool(r.m2), "M1": float(r.scores["M1"]), "M2": float(r.scores["M2"])}
    return out


# ---- LLM policy ----------------------------------------------------------------------------

_NAMES = {t["function"]["name"] for t in MENU_TOOLS}

PLAN_HEADER = "A plan from an expert folder (a guide, not orders — if a step doesn't work, adapt):"


def plan_part(plan_text: str | None) -> list:
    """Goal-context part with an expert plan (shown every turn), or [] if no plan."""
    return [{"type": "text", "text": f"{PLAN_HEADER}\n{plan_text.strip()}"}] if plan_text and plan_text.strip() else []


def extract_menu_calls(msg: dict) -> list[tuple[str, dict]]:
    calls = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        try:
            args = _parse_args(fn.get("arguments"))
        except json.JSONDecodeError as e:
            raise ValueError(f"tool call arguments are not valid JSON ({e.msg})") from None
        if fn.get("name") not in _NAMES:
            raise ValueError(f"unknown tool {fn.get('name')!r}")
        calls.append((fn["name"], args))
    if not calls:
        raise ValueError("no tool call in the reply")
    return calls


def build_messages(view: MenuView, env_goal_parts: list, system_prompt: str = MENU_SYSTEM_PROMPT) -> list:
    tail = [{"type": "text", "text": view.text}] + [_img_part(p) for p in view.images]
    return [{"role": "system", "content": system_prompt},
            {"role": "user", "content": env_goal_parts + tail}]


def llm_policy(client: LLMClient, env_goal_parts: list, ledger: CostLedger | None, stats: dict,
               system_prompt: str = MENU_SYSTEM_PROMPT, events=None):
    from .events import NullEvents

    ev = events or NullEvents()

    def policy(view: MenuView, mem: dict):
        messages = build_messages(view, env_goal_parts, system_prompt)
        mem["_messages"] = messages  # for tests / inspection (not logged)
        ev.guard()
        if ledger is not None:
            ledger.check()
        ev.status("thinking", turn=view.turn)
        ev.request(view.turn, messages, menu=view.options)
        resp = as_result(client.chat(messages, MENU_TOOLS))
        ev.response(view.turn, resp)
        usage = resp.usage
        c = float(usage.cost_usd or 0.0)
        stats["cost"] += c
        stats["prompt_tokens"] += usage.input
        stats["completion_tokens"] += usage.output
        if ledger is not None and c:
            ledger.add(c, f"{getattr(client, 'model', '?')}/menu")
        msg = resp.message()
        mem["_rec"] = {"reply": dict(msg), "usage": usage.to_dict(), "finish_reason": resp.finish_reason,
                       "provider": resp.provider}
        if not mem["full_plan"] and view.turn == 1:  # keep a written-out plan even if not in the args
            txt = (msg.get("content") or "").strip()
            if txt:
                mem["full_plan"] = _words(txt, 250)
        return extract_menu_calls(msg)

    return policy


def run_episode_menu(target: str | None, model: str, images: bool = True, max_turns: int | None = None,
                     feedback: str = "F1", out_root: Path | None = None, log=print,
                     reasoning_tokens: int = 3000, reasoning_effort: str | None = None,
                     ledger: CostLedger | None = None, plan_text: str | None = None, client=None,
                     goal_text: str | None = None, goal=None, events=None, out="auto") -> dict:
    """goal_text: text-goal mode — the agent gets ONLY this sentence as the goal (no target image /
    outline / par / name); scoring still uses the target model (+ its alternatives).
    goal: agents.goal.Goal (overrides target / goal_text). events: agents.events.RunEvents.
    out: "auto" = a new dir under out_root (or runs/), a Path = that dir, None = write no files."""
    from .goal import Goal

    if goal is None:
        goal = Goal.from_text(goal_text, target) if goal_text else Goal.example(target)
    env0 = goal.make_env(feedback)
    sysp = goal.system_prompt("menu")
    gparts = goal.parts(env0, images)
    pparts = plan_part(plan_text)
    if out == "auto":
        out = run_dir(model + "_menu", env0.name, out_root)
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "system.txt").write_text(sysp + "\n\nTOOLS:\n" + json.dumps(MENU_TOOLS, indent=1))
        if goal.kind == "text":
            (out / "goal.txt").write_text(goal.text)
        else:
            (out / "goal.txt").write_text(gparts[0]["text"])
        if goal.kind == "example" and images:
            (out / "target.png").write_bytes(env0.target_png())
        if plan_text:
            (out / "plan.txt").write_text(plan_text)
    gparts = gparts + pparts
    client = client or OpenRouterClient(model, log=log, reasoning_tokens=reasoning_tokens,
                                        reasoning_effort=reasoning_effort)
    stats = {"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0}
    log(f"run {model} on {env0.name} (menu mode, par {env0.par}) -> {out}")
    started = {"system_prompt": sysp, "tools": MENU_TOOLS, "goal_parts": gparts, "plan_parts": len(pparts),
               "goal": goal}
    s = run_menu_episode(None, llm_policy(client, gparts, ledger, stats, sysp, events=events), max_turns=max_turns,
                         feedback=feedback, images=images, out=out, log=log, goal=goal, events=events,
                         started=started)
    s.update(model=model, cost_usd=round(stats["cost"], 6), tokens=stats, dir=str(out) if out else None,
             plan_file=bool(plan_text))
    if out is not None:
        (out / "score.json").write_text(json.dumps(s, indent=1, default=str))
    return s


# ---- scripted reference policy (tests) -------------------------------------------------------

def reference_policy(model_path: str, stats: dict | None = None):
    """Follows the target model's own steps, but only through the menu: choose the option whose
    result equals (canonical state_hash, so up to symmetry) the reference state after that step.
    A reference sim is kept separately (the menu may pick the mirror-image fold). If the step is not
    on the first page, it asks `more` for the cut sections. stats["more_calls"] = per-step count."""
    from ..engine import Simulator
    from ..match import state_hash

    stats = stats if stats is not None else {}
    stats.setdefault("more_calls", [])
    ref = [{k: v for k, v in st.items() if k != "description"}
           for st in json.loads(Path(model_path).read_text())["steps"]]
    sim = Simulator(float(json.loads(Path(model_path).read_text()).get("paper", {}).get("size", 1.0)))
    state = {"i": 0, "more": 0, "target": None}

    def policy(view: MenuView, mem: dict):
        while state["target"] is None:
            if state["i"] >= len(ref):
                return [("submit", {})]
            s = ref[state["i"]]
            state["i"] += 1
            sim.apply(s)
            if s["op"] in ("flip", "rotate"):
                return [(s["op"], s)]  # free; keeps the agent's view like the diagrams
            state["target"] = state_hash(sim.state)
        opt = next((o for o in view.options if o.hash == state["target"]), None)
        if opt is not None:
            stats["more_calls"].append(state["more"])
            state["more"], state["target"] = 0, None
            return [("choose", {"option": opt.id, "plan": "follow the reference"})]
        todo = [f for f, t in getattr(view.options, "truncated", {}).items() if t]
        state["more"] += 1
        if not todo or state["more"] > 15:
            raise AssertionError(f"reference step {ref[state['i'] - 1]} not in the menu")
        return [("more", {"section": todo[(state["more"] - 1) % len(todo)]})]

    return policy


def cmd_agent_menu(args) -> int:
    client = make_cli_client(args)
    s = run_episode_menu(args.target, args.model, client=client, images=not args.no_images, max_turns=args.max_steps,
                         feedback=args.feedback, out_root=Path(args.runs_dir) if args.runs_dir else None,
                         reasoning_tokens=args.reasoning_tokens, reasoning_effort=args.reasoning_effort,
                         ledger=CostLedger(args.ledger or REPO / "runs" / "cost_ledger.json", args.budget_usd),
                         plan_text=read_plan(args), goal=goal_from_args(args))
    return 0 if s["stop_reason"] not in ("daily_limit", "no_credit", "api_error", "budget") else 2


def goal_from_args(args):
    """--goal-image PATH -> image goal; --goal text -> text goal; otherwise None (example goal of --target)."""
    from .goal import Goal

    if getattr(args, "goal_image", None):
        return Goal.from_image(Path(args.goal_image).read_bytes(), getattr(args, "goal_text", None), args.target)
    t = read_goal(args)
    return Goal.from_text(t, args.target) if t else None


def read_goal(args) -> str | None:
    """--goal text: the goal sentence from --goal-text (required)."""
    if getattr(args, "goal", "image") != "text":
        return None
    t = (getattr(args, "goal_text", None) or "").strip()
    if not t:
        raise SystemExit("--goal text needs --goal-text \"...\"")
    return t


def read_plan(args) -> str | None:
    p = getattr(args, "plan_file", None)
    return Path(p).read_text() if p else None
