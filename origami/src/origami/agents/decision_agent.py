"""DECISION action mode: an OpenRouter "decision model" (returns probabilities, not text) picks the next fold.

    uv run origami agent --actions decision --model typesafe/jev-1.14 --target models/cup.json

Each turn: menu (origami.menu) + state text (+ target/current images if the model reads images) go to
POST /api/alpha/decisions as one `choice` question; options "1".."N" = menu folds, "submit" = stop.
Greedy: take the most probable option.
The loop is the menu episode loop (run_menu_episode) with a decision policy.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from pathlib import Path

from ..env import FoldEnv
from .llm_agent import REPO, APIError, BudgetExceeded, CostLedger, NoCredit, load_api_key, run_dir
from ..observe import step_line
from .menu_agent import PLAN_HEADER, MenuView, read_plan, run_menu_episode

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
TEXT_ONLY = {"microsoft/microsoft-decision-1"}  # fallback if the models API can't be reached
QUESTION = "Which fold should be done next to reach the target?"
PLAN_QUESTION = ("Which fold should be done next to reach the target? Pick the option that carries out the next "
                 "plan step (the first plan step not yet done by the moves so far).")
SUBMIT_KEY = "submit"
LEDGER_PATH = REPO / "runs" / "cost_ledger_decision.json"
CAP_USD = 0.50


def supports_images(model: str) -> bool:
    try:
        url = "https://openrouter.ai/api/v1/models?output_modalities=decisions"
        with urllib.request.urlopen(url, timeout=20) as r:
            for m in json.load(r)["data"]:
                if m["id"] == model:
                    return "image" in m["architecture"]["input_modalities"]
    except Exception:
        pass
    return model not in TEXT_ONLY


def _png_part(png: bytes) -> dict:
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}


def call_decisions(model: str, state, questions: dict, key: str, timeout: float = 120) -> dict:
    body = json.dumps({"model": model, "state": state, "questions": questions}).encode()
    req = urllib.request.Request(DECISIONS_URL, data=body, method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="replace")[:400]
        raise (NoCredit if e.code == 402 else APIError)(f"HTTP {e.code}: {msg}") from None
    except (urllib.error.URLError, TimeoutError) as e:
        raise APIError(f"network error: {e}") from None


def moves_done(env: FoldEnv) -> str:
    """Numbered descriptions of the moves made so far (flip / rotate included)."""
    rows = [s.get("description") or step_line(s) for s in env.trajectory]
    return "\n".join(f"{i}. {r}" for i, r in enumerate(rows, 1)) or "(none yet: flat sheet)"


def build_state(env: FoldEnv, view: MenuView, target_png: bytes | None, use_images: bool,
                plan_text: str | None = None):
    """State for the API: plain string (text-only) or a list of text / image_url parts."""
    obs = env.observe(images=False)
    plan = (f"{PLAN_HEADER}\n{plan_text.strip()}\n\nMoves done so far:\n{moves_done(env)}\n\n"
            if plan_text else "")
    text = (f"GOAL\n{env.goal_text()}\n\n{plan}"
            f"=== TURN {view.turn}/{view.max_turns} | moves used {env.moves} (par {env.par}) ===\n"
            f"Current paper:\n{obs.text}")
    if not use_images:
        return text
    parts = [{"type": "text", "text": text + "\nImage 1: the TARGET (front and back). Image 2: the CURRENT paper."}]
    if target_png:
        parts.append(_png_part(target_png))
    if view.images:
        parts.append(_png_part(view.images[0]))
    return parts


def make_question(opts, plan: bool = False) -> dict:
    criteria = {str(o.id): o.description for o in opts}
    criteria[SUBMIT_KEY] = "submit: the paper already matches the target, stop folding"
    return {"fold": {"type": "choice", "instructions": PLAN_QUESTION if plan else QUESTION, "criteria": criteria}}


def pick(answer: dict) -> tuple[str, dict]:
    """(chosen key, probabilities) from one choice answer; argmax of probabilities if present."""
    if answer.get("type") != "choice":
        raise ValueError(f"decision model gave no choice (answer type {answer.get('type')!r})")
    probs = {str(k): float(v) for k, v in (answer.get("probabilities") or {}).items()}
    key = max(probs, key=probs.get) if probs else str(answer.get("choice"))
    return key, probs


def decision_policy(model: str, target_png: bytes | None, use_images: bool, ledger: CostLedger | None,
                    stats: dict, key: str | None = None, caller=call_decisions, plan_text: str | None = None):
    key = key or (load_api_key() if caller is call_decisions else "")

    def policy(view: MenuView, mem: dict):
        if ledger is not None:
            ledger.check()
        if not view.options:
            return [("submit", {})]
        resp = caller(model, build_state(view.env, view, target_png, use_images, plan_text),
                      make_question(view.options, plan=bool(plan_text)), key)
        usage = resp.get("usage") or {}
        c = float(usage.get("cost") or 0.0)
        stats["cost"] += c
        stats["prompt_tokens"] += usage.get("input_tokens") or 0
        if ledger is not None and c:
            ledger.add(c, f"{model}/decision")
        chosen, probs = pick((resp.get("answers") or {}).get("fold") or {})
        mem["_rec"] = {"probabilities": probs, "chosen_key": chosen, "usage": usage}
        if chosen == SUBMIT_KEY:
            return [("submit", {})]
        try:
            return [("choose", {"option": int(chosen)})]
        except ValueError:
            raise ValueError(f"unexpected choice key {chosen!r}") from None

    return policy


def run_episode_decision(target: str, model: str, images: bool | None = None, max_turns: int | None = None,
                         feedback: str = "F1", out_root: Path | None = None, log=print,
                         ledger: CostLedger | None = None, caller=call_decisions, key: str | None = None,
                         plan_text: str | None = None) -> dict:
    env0 = FoldEnv(target, feedback=feedback)
    use_images = supports_images(model) if images is None else images
    out = (out_root or REPO / "runs" / "decision_v1") / model.replace("/", "_") / env0.name
    if out.exists():
        out = run_dir(model, env0.name, out_root or REPO / "runs" / "decision_v1")  # fresh dir, don't overwrite
    out.mkdir(parents=True, exist_ok=True)
    tpng = env0.target_png()
    (out / "target.png").write_bytes(tpng)
    stats = {"cost": 0.0, "prompt_tokens": 0}
    log(f"run {model} on {env0.name} (decision mode, greedy, images={use_images}, par {env0.par}) -> {out}")
    if plan_text:
        (out / "plan.txt").write_text(plan_text)
    pol = decision_policy(model, tpng, use_images, ledger, stats, key=key, caller=caller, plan_text=plan_text)
    s = run_menu_episode(target, pol, max_turns=max_turns, feedback=feedback, images=use_images, thumbs=False,
                         out=out, log=log)
    s.update(actions="decision", model=model, images=use_images, cost_usd=round(stats["cost"], 6),
             tokens=stats, dir=str(out), plan_file=bool(plan_text))
    (out / "score.json").write_text(json.dumps(s, indent=1, default=str))
    return s


def cmd_agent_decision(args) -> int:
    ledger = CostLedger(args.ledger or LEDGER_PATH, args.budget_usd if args.budget_usd is not None else CAP_USD)
    s = run_episode_decision(args.target, args.model, images=False if args.no_images else None,
                             max_turns=args.max_steps, feedback=args.feedback,
                             out_root=Path(args.runs_dir) if args.runs_dir else None, ledger=ledger,
                             plan_text=read_plan(args))
    print(json.dumps({k: s[k] for k in ("stop_reason", "menu_moves", "illegal", "cost_usd", "score")}, default=str))
    return 0 if s["stop_reason"] not in ("no_credit", "api_error", "budget") else 2
