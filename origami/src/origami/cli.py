"""Command line:
  `origami fold models/<name>.json --out out/<name>/`   PNG steps + .fold
  `origami check models/<name>.json [--sampled] [--json]`   physical validity report (exit 1 on errors)
  `origami agent --target models/cup.json --model <openrouter id> [--no-images] [--max-steps N] [--actions menu]`   LLM agent episode
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .engine import FoldError, PhysicsError, Simulator
from .fold_export import write_fold


def run_model(model: dict, strict: bool = False):
    """Apply all steps; return list of states (state[0] = flat sheet).

    strict=True: also run the physical validity checks (validate.check_step) after every step and
    raise PhysicsError on the first step with errors (warnings are ignored).
    """
    sim = Simulator(float(model.get("paper", {}).get("size", 1.0)))
    states = [sim.state.copy()]
    for i, step in enumerate(model["steps"], 1):
        try:
            sim.apply(step)
        except (FoldError, ValueError) as e:
            raise FoldError(f"step {i} ({step.get('op')}): {e}") from e
        states.append(sim.state.copy())
        if strict:
            from .validate import check_step

            errors = [x for x in check_step(states[-2], states[-1], step, i) if x.severity == "error"]
            if errors:
                raise PhysicsError("\n".join(x.message for x in errors), errors)
    return states


def _half_span(states, pad=0.18) -> float:
    """One drawing scale for all steps, so the paper visibly shrinks as it is folded."""
    spans = [float(max(np.subtract(*reversed(s.bbox())))) for s in states]
    return max(spans) / 2 * (1 + pad) + 0.05


def _limits(state, half):
    lo, hi = state.bbox()
    c = (lo + hi) / 2
    return (c - half, c + half)


def cmd_fold(args) -> int:
    # matplotlib only here: the rest of the package (and the browser build) stays matplotlib-free
    from .render import render_crease_pattern, render_step_png, set_colored_side, set_layer_offset

    model = json.loads(Path(args.model).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        states = run_model(model)
    except FoldError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    steps = model["steps"]
    set_colored_side(model.get("paper", {}).get("colored_side", "up"))
    set_layer_offset(args.layer_offset)
    half = _half_span(states)
    for k, state in enumerate(states):
        if k < len(steps):
            nxt = steps[k]
            title = f"Step {k + 1}: {nxt.get('description', nxt['op'])}"
        else:
            nxt, title = None, f"Finished: {model.get('title', model.get('name', ''))}"
        render_step_png(out / f"step_{k:02d}.png", state, title, nxt, _limits(state, half))
    render_crease_pattern(out / "crease_pattern.png", states[-1])
    write_fold(out / f"{model.get('name', 'model')}.fold", states[-1], model.get("title", ""))
    print(f"{len(states)} images + crease_pattern.png + .fold written to {out}/")
    return 0


def cmd_check(args) -> int:
    from .validate import validate_model

    model = json.loads(Path(args.model).read_text())
    rep = validate_model(model, sampled=args.sampled)
    print(json.dumps(rep.to_dict(), indent=1) if args.json else rep.to_text())
    return 0 if rep.ok else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="origami")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fold", help="simulate a model's fold steps")
    f.add_argument("model", help="path to model JSON")
    f.add_argument("--out", required=True, help="output directory")
    f.add_argument("--layer-offset", type=float, default=0.0,
                   help="shift per stacked layer, as a fraction of paper size (default 0 = flat; try 0.004 to hint at stacked layers)")
    c = sub.add_parser("check", help="check that every step is physically possible")
    c.add_argument("model", help="path to model JSON")
    c.add_argument("--sampled", action="store_true",
                   help="also sample the 3D animation for self-intersections (slow, warnings only)")
    c.add_argument("--json", action="store_true", help="print the report as JSON")
    a = sub.add_parser("agent", help="run an LLM agent (any provider: --provider) on a target model")
    from .agents.llm_agent import add_args

    add_args(a)
    a.add_argument("--actions", choices=("command", "menu", "decision"), default="command",
                   help="command = describe folds as JSON (default); menu = pick from a list of legal folds; decision = decision model picks (probabilities)")
    a.add_argument("--goal", choices=("image", "text"), default="image",
                   help="image = target picture + computed facts + par (default); text = ONLY --goal-text "
                        "(menu mode; nothing derived from the target is shown; scored vs the target + its alternatives)")
    a.add_argument("--goal-text", default=None, help="the goal sentence for --goal text (a hint with --goal-image)")
    a.add_argument("--goal-image", default=None,
                   help="PNG picture of the object to fold: image goal (nothing derived from the target is shown; "
                        "the target is used for scoring only)")
    args = p.parse_args(argv)
    if args.cmd == "agent":
        if (args.goal == "text" or args.goal_image) and args.actions == "decision":
            p.error("text / image goals work with --actions command or menu")
        if args.actions == "menu":
            from .agents.menu_agent import cmd_agent_menu

            return cmd_agent_menu(args)
        if args.actions == "decision":
            if args.provider != "openrouter" or args.dry_run:
                p.error("--actions decision uses OpenRouter's /decisions API (only --provider openrouter, no --dry-run)")
            if not args.model:
                p.error("--model is required")
            from .agents.decision_agent import cmd_agent_decision

            return cmd_agent_decision(args)
        from .agents.llm_agent import cmd_agent

        return cmd_agent(args)
    if args.cmd == "check":
        return cmd_check(args)
    return cmd_fold(args)


if __name__ == "__main__":
    raise SystemExit(main())
