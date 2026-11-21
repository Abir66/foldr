"""No-LLM baselines.

- `random_agent`: each move, sample random landmark folds (valley/mountain, onto/through two visible
  named points, all layers or top 1) until one is legal; stop after `max_moves` moves. Shows what
  "flailing" scores and that the env rejects / accepts sensibly.
- `oracle_agent`: replay the target model's own steps (sanity check: must pass M2 with moves == par).

    uv run python -m origami.agents.baselines models/cup.json [--seed 0] [--moves 10]
"""

from __future__ import annotations

import argparse
import random

from ..env import FoldEnv, Score, replay_model, summarize_score


def random_action(env: FoldEnv, rng: random.Random) -> dict:
    pts = [n for n, p in env.observe(images=False).points.items() if p.front or p.back]
    if len(pts) < 2:
        pts = list(env.observe(images=False).points)
    if rng.random() < 0.1:
        return {"op": "flip"}
    p, q = rng.sample(pts, 2)
    kind = rng.choice(["onto", "onto", "through"])
    a = {"op": rng.choice(["valley", "valley", "mountain"]), "line": {kind: [p, q]}}
    if kind == "through":
        others = [n for n in pts if n not in (p, q)]
        if not others:
            return {"op": "flip"}
        a["move"] = rng.choice(others)
    if rng.random() < 0.3:
        a["layers"] = {"top": 1}
    return a


def random_agent(target, seed: int = 0, max_moves: int | None = None, tries_per_move: int = 30) -> tuple[FoldEnv, Score]:
    rng = random.Random(seed)
    env = FoldEnv(target)
    budget = max_moves if max_moves is not None else 2 * env.par
    while env.moves < budget:
        for _ in range(tries_per_move):
            if env.apply(random_action(env, rng)).ok:
                break
        else:
            break  # stuck: nothing legal found
    return env, env.submit(compute_deletable=False)


def oracle_agent(target) -> tuple[FoldEnv, Score]:
    env, _, score = replay_model(target)
    return env, score


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="origami.agents.baselines")
    ap.add_argument("target")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--moves", type=int, default=None)
    ap.add_argument("--runs", type=int, default=5)
    a = ap.parse_args(argv)
    _, s = oracle_agent(a.target)
    print("oracle:", summarize_score(s))
    for i in range(a.runs):
        _, s = random_agent(a.target, a.seed + i, a.moves)
        print(f"random seed {a.seed + i}:", summarize_score(s), f"shape={s.shape_score:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
