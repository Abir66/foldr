"""One-off "expert plan" call: a strong model writes a plain-words fold plan for a target (no tools).
The plan is then given to cheaper executors with `origami agent --plan-file PATH` (menu / decision modes).

    python -m origami.agents.expert_plan --target models/hat.json --model anthropic/claude-opus-5.5 \
        --out runs/plan_exec/hat_plan.txt --ledger runs/cost_ledger_plan_opus.json --budget-usd 3.20

Input = what the menu agent sees as goal on turn 1: goal text (incl. purpose line) + target image.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..env import FoldEnv
from ..llm import as_result
from .llm_agent import CostLedger, OpenRouterClient, goal_parts

PLAN_REQUEST = ("Write a numbered step plan, in plain words, for folding this target from a flat square sheet. "
                "At most 150 words. Each step is ONE physical fold (say valley or mountain, what goes onto what, "
                "and which layers or which flap move: all layers, front layer only, one flap, tucked inside). "
                "Say when to turn the paper over. No other text.")


def make_plan(target: str | None, model: str, ledger: CostLedger | None = None, client=None,
              reasoning_effort: str = "medium", goal=None) -> tuple[str, dict]:
    """goal: agents.goal.Goal; the planner sees the same goal parts as the folding agent (default: example
    goal of `target`, with its picture)."""
    if goal is not None:
        env = goal.make_env()
        request = PLAN_REQUEST if goal.kind == "example" else PLAN_REQUEST.replace("this target", "the object in the GOAL")
        parts = goal.parts(env, True) + [{"type": "text", "text": request}]
    else:
        env = FoldEnv(target)
        parts = goal_parts(env, True, env.target_png()) + [{"type": "text", "text": PLAN_REQUEST}]
    client = client or OpenRouterClient(model, reasoning_effort=reasoning_effort)
    if ledger is not None:
        ledger.check()
    resp = as_result(client.chat([{"role": "user", "content": parts}], None))
    usage = resp.usage.to_dict()
    if ledger is not None and usage.get("cost_usd"):
        ledger.add(float(usage["cost_usd"]), f"{model}/plan")
    return (resp.content or "").strip(), usage


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--budget-usd", type=float, default=None)
    a = ap.parse_args(argv)
    ledger = CostLedger(a.ledger, a.budget_usd) if a.ledger else None
    text, usage = make_plan(a.target, a.model, ledger)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(text)
    print(json.dumps({"usage": usage}))
    return 0 if text else 1


if __name__ == "__main__":
    raise SystemExit(main())
