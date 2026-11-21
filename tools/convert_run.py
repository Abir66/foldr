"""Turn a run folder written by the origami agent loops (turns.jsonl, score.json, system.txt, ...) into a replay.

Run from the repo root:
  uv run --project origami python tools/convert_run.py --all [--runs-dir DIR]
  uv run --project origami python tools/convert_run.py SLUG [--runs-dir DIR]
  uv run --project origami python tools/convert_run.py --check          # validate committed replays only

Writes examples/runs/<slug>/events.jsonl (one event per line, docs/events.schema.json) and meta.json.
The log has no pictures, so they are rebuilt by replaying the logged actions on a fresh paper: the
"current paper" picture of every turn, and (menu mode) the option thumbnails when the menu rebuilt
today matches the logged one text for text. Timestamps are made up from the logged latencies.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "examples" / "models"
OUT = ROOT / "examples" / "runs"
DEFAULT_RUNS = ROOT.parent / "origami" / "runs"
SCHEMA = ROOT / "docs" / "events.schema.json"

# slug -> run folder (relative to the runs dir) and how the replay is presented.
CURATED = {
    "opus-hat-text": dict(
        path="textgoal/opus/run1/2026-11-21/anthropic_claude-opus-5.5_menu/hat",
        title="Folds a samurai helmet from words only", model="Claude Opus", example="hat",
        blurb="No picture of the result: only one sentence. It writes its own plan, then picks folds from the menu."),
    "opus-hat-menu": dict(
        path="opus_hat/2026-11-20/anthropic_claude-opus-5.5_menu/hat",
        title="Folds a samurai helmet", model="Claude Opus", example="hat",
        blurb="Knows the plan from the first turn and recognises each fold in the menu."),
    "opus-cup-menu": dict(
        path="big_v1b/2026-11-20/anthropic_claude-opus-5.5_menu/cup",
        title="Folds a paper cup", model="Claude Opus", example="cup",
        blurb="Five folds from the menu, no undo."),
    "opus-cup-command": dict(
        path="big_v1/2026-11-20/anthropic_claude-opus-5.5/cup",
        title="Folds a paper cup, describing each fold", model="Claude Opus", example="cup",
        blurb="No menu: it describes every fold line itself (for example with an angle bisector)."),
    "sonnet-hat-menu": dict(
        path="menu_fast/sonnet/run1/2026-11-21/anthropic_claude-sonnet-5.5_menu/hat",
        title="Tries a samurai helmet", model="Claude Sonnet", example="hat",
        blurb="Follows a written plan, keeps undoing a correct base and ends with wrong folds."),
    "haiku-cup-command": dict(
        path="cmd_v2/run1/2026-11-20/anthropic_claude-haiku-5.5/cup",
        title="A small model guesses a paper cup", model="Claude Haiku", example="cup",
        blurb="Does not know the fold sequence: it guesses, undoes and submits a wrong shape."),
}

PRICE_KEYS = ("cost", "cost_usd")


def _usage(u: dict):
    """Logged usage (command logs: prompt/completion; menu logs: raw provider usage) -> Usage."""
    from origami.llm import Usage

    if "prompt" in u:
        return Usage(input=int(u.get("prompt") or 0), output=int(u.get("completion") or 0),
                     cached=int(u.get("cached") or 0), reasoning=int(u.get("reasoning") or 0),
                     cost_usd=u.get("cost"))
    ptd, ctd = u.get("prompt_tokens_details") or {}, u.get("completion_tokens_details") or {}
    return Usage(input=int(u.get("prompt_tokens") or 0), output=int(u.get("completion_tokens") or 0),
                 cached=int(ptd.get("cached_tokens") or 0), reasoning=int(ctd.get("reasoning_tokens") or 0),
                 cost_usd=u.get("cost"))


def _img_part(png: bytes) -> dict:
    import base64

    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}


def _small_png(png: bytes, max_px: int = 512) -> bytes:
    """Replay pictures: <= max_px and a 64-colour palette (flat drawings), to keep the committed logs small."""
    import io

    from PIL import Image

    im = Image.open(io.BytesIO(png)).convert("RGB")
    if max(im.size) > max_px:
        im.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    im.quantize(64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


class _Menu(list):
    truncated: dict = {}


def _menu_options(opts, logged: list[str]):
    """Rebuilt options if they match the logged menu exactly, else options made from the logged text."""
    from types import SimpleNamespace

    rebuilt = [o.description for o in opts]
    if rebuilt == logged:
        return _Menu(opts), True
    out = _Menu()
    for i, d in enumerate(logged, 1):
        out.append(SimpleNamespace(id=i, fam="menu", description=d, after="", step={}))
    return out, False


def convert(slug: str, runs_dir: Path, out_root: Path = OUT) -> dict:
    from origami.agents import events as events_mod
    from origami.agents.command_layer import CommandLayer
    from origami.agents.events import RunEvents
    from origami.agents.goal import Goal
    from origami.agents.menu_agent import plan_part, score_variants
    from origami.llm import ChatResult
    from origami.menu import THUMB_MAX, candidates, thumbnails_png
    from origami.observe import render_cp_png

    cfg = CURATED[slug]
    run = runs_dir / cfg["path"]
    score_json = json.loads((run / "score.json").read_text())
    turns = [json.loads(line) for line in (run / "turns.jsonl").read_text().splitlines() if line.strip()]
    system_txt = (run / "system.txt").read_text()
    sysp, tools_json = system_txt.split("\n\nTOOLS:\n", 1)
    tools = json.loads(tools_json)
    menu_mode = score_json.get("actions") == "menu"
    text_goal = bool(score_json.get("text_goal"))
    target_path = MODELS / f"{score_json['target']}.json"
    goal_txt = (run / "goal.txt").read_text() if (run / "goal.txt").exists() else None
    plan_txt = (run / "plan.txt").read_text() if (run / "plan.txt").exists() else None

    goal = Goal.from_text(goal_txt, target_path) if text_goal else Goal.example(target_path)
    env = goal.make_env("F1")
    layer = CommandLayer(env)
    gparts = goal.parts(env, True)
    if goal_txt and not text_goal:
        gparts[0] = {**gparts[0], "text": goal_txt}
    pparts = plan_part(plan_txt)
    first = turns[0]["prompt_text"]
    m = re.search(r"TURN 1/(\d+)|TURN 1 of (\d+)", first)
    max_turns = int(m.group(1) or m.group(2)) if m else goal.default_max_turns(env)
    limits = {"max_turns": max_turns, "usd": None}
    if not menu_mode:
        limits["max_free_turns"] = max_turns

    clock = [0.0]

    def stamp(e: dict) -> None:
        e["t"] = round(clock[0], 3)

    events_mod.downscale_png = _small_png
    ev = RunEvents(emit=stamp, max_usd=None)
    ev.meta["config"] = {"actions": "menu" if menu_mode else "command", "model": score_json["model"], "images": True,
                         "plan": {"kind": "text" if plan_txt else "none"}, "max_turns": max_turns,
                         "replay": True}

    def at(dt: float = 0.0):
        clock[0] += dt

    ev.started(env, system_prompt=sysp, tools=tools, goal_parts=gparts + pparts, plan_parts=len(pparts),
               goal=goal, limits=limits)
    more: dict = {}
    mismatches: list[str] = []
    thumbs_ok = 0
    stop_reason = score_json.get("stop_reason", "submitted")
    for rec in turns:
        turn = rec["turn"]
        if menu_mode:
            at(float(rec.get("menu_s") or 0.0))
            ev.status("menu", "building suggestions", turn=turn)
        imgs = []
        cur = env.observe(images=True).folded_png
        ev.image(cur, "current paper, front and back")
        imgs.append(cur)
        menu = None
        if menu_mode:
            opts = candidates(env, more=more, labeler=layer.labels)
            menu, matched = _menu_options(opts, rec["menu"])
            if matched and opts:
                th = thumbnails_png(opts, env.colored_side, max_n=THUMB_MAX)
                ev.image(th, f"menu options 1-{min(len(opts), THUMB_MAX)}")
                imgs.append(th)
                thumbs_ok += 1
        messages = [{"role": "system", "content": sysp},
                    {"role": "user", "content": gparts + pparts + [{"type": "text", "text": rec["prompt_text"]}]
                     + [_img_part(p) for p in imgs]}]
        reply = rec.get("reply")
        if reply is None:
            if rec.get("error"):
                ev.error("provider_error", rec["error"], turn)
                break
            continue
        ev.status("thinking", turn=turn)
        ev.request(turn, messages, menu=menu)
        latency = float(rec.get("latency_s") or 0.0)
        at(latency)
        res = ChatResult(content=reply.get("content"), reasoning=reply.get("reasoning"),
                         tool_calls=reply.get("tool_calls") or [], finish_reason=rec.get("finish_reason"),
                         usage=_usage(rec.get("usage") or {}), latency_s=latency, provider=rec.get("provider"))
        ev.response(turn, res)
        if rec.get("format_error"):
            ev.error(rec.get("error_kind") or "format_error", rec["format_error"], turn)
            continue
        replied = [(tc["function"]["name"], _args(tc["function"].get("arguments")))
                   for tc in reply.get("tool_calls") or []]
        for i, call in enumerate(rec.get("calls") or []):
            name = call["tool"]
            args = replied[i][1] if i < len(replied) and replied[i][0] == name else {}
            if name == "submit":
                if menu_mode:
                    ev.action(turn, {"tool": "submit"})
                continue
            if name != "more":
                ev.status("folding", turn=turn)
            if name == "undo":
                env.undo(int(args.get("n") or 1))
            elif name in ("more", "inspect", "choose") and not call.get("action"):
                pass
            elif call.get("action") and call.get("ok"):
                r = env.apply(call["action"])
                if not r.ok:
                    mismatches.append(f"turn {turn} {name}: replay rejected ({r.code})")
            ev.action(turn, {k: call[k] for k in call if k not in ("warnings", "snapped")})
            if name == "more":
                more[call["section"]] = more.get(call["section"], 1) + 1
            elif call.get("ok"):
                more = {}
    at(0.0)
    score = env.submit(False)
    variants = score_variants(env) if goal.scored else {}
    sol = json.loads((run / "solution.json").read_text()) if (run / "solution.json").exists() else None
    ev.finished(stop_reason, score=score, variants=variants, solution=sol,
                final_pngs=[(env.observe(images=True).folded_png, "final paper, front and back"),
                            (render_cp_png(env.state), "crease pattern")])
    fin = ev.events[-1]
    fin["totals"]["seconds"] = round(clock[0], 1)
    fin["t"] = round(clock[0], 3)

    logged = score_json["score"]
    passed = bool(score_json.get("pass_any", logged.get("m2")))
    replay_pass = bool(fin["score"].get("pass_any", fin["score"]["m2"])) if fin.get("score") else None
    if replay_pass is not None and replay_pass != passed:
        mismatches.append(f"outcome: logged pass={passed}, replay pass={replay_pass}")

    meta = {
        "slug": slug, "title": cfg["title"], "blurb": cfg["blurb"],
        "model": cfg["model"], "modelId": score_json["model"],
        "mode": "menu" if menu_mode else "command", "plan": bool(plan_txt),
        "goal": {"kind": goal.kind, "text": goal.text if goal.kind == "text" else None,
                 "target": score_json["target"]},
        "example": cfg["example"],
        "outcome": {"passed": passed, "stopReason": stop_reason, "m1": bool(logged.get("m1")),
                    "m2": bool(logged.get("m2")), "moves": logged.get("moves"), "par": logged.get("par"),
                    "undos": score_json.get("undo", 0)},
        "costUsd": fin["totals"]["cost_usd"], "turns": fin["totals"]["turns"],
        "seconds": fin["totals"]["seconds"], "events": len(ev.events),
        "date": re.search(r"20\d\d-\d\d-\d\d", cfg["path"]).group(0) if re.search(r"20\d\d-\d\d-\d\d", cfg["path"]) else None,
    }
    d = out_root / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "events.jsonl").write_text("".join(json.dumps(e, separators=(",", ":")) + "\n" for e in ev.events))
    (d / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"{slug}: {len(ev.events)} events, {meta['turns']} turns, ${meta['costUsd']}, pass={passed}, "
          f"thumbs {thumbs_ok}/{len(turns)}, {(d / 'events.jsonl').stat().st_size // 1024} KB")
    for msg in mismatches:
        print(f"  WARNING {msg}")
    return meta


def _args(raw):
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}


SECRET_PATTERNS = [r"sk-or-[A-Za-z0-9_-]{8,}", r"sk-ant-[A-Za-z0-9_-]{8,}", r"sk-[A-Za-z0-9]{20,}",
                   r"/Users/[A-Za-z0-9_.-]+", r"/home/[A-Za-z0-9_.-]+", r"Bearer\s+[A-Za-z0-9._-]{12,}",
                   r"OPENROUTER_API_KEY\s*=", r"\bapi[_-]?key\b\s*[:=]\s*\S{8,}", r"\.env\b"]


def scan(d: Path) -> list[str]:
    hits = []
    for f in sorted(d.glob("*/*")):
        text = f.read_text()
        for pat in SECRET_PATTERNS:
            for m in re.finditer(pat, text):
                hits.append(f"{f.relative_to(ROOT)}: {m.group(0)[:40]}")
    return hits


def validate(d: Path) -> list[str]:
    import jsonschema

    schema = json.loads(SCHEMA.read_text())
    v = jsonschema.Draft202012Validator(schema)
    problems = []
    for f in sorted(d.glob("*/events.jsonl")):
        events = [json.loads(line) for line in f.read_text().splitlines()]
        for i, e in enumerate(events):
            if e["seq"] != i:
                problems.append(f"{f.parent.name}: seq {e['seq']} at line {i}")
            for err in list(v.iter_errors(e))[:2]:
                problems.append(f"{f.parent.name}: event {i} ({e['type']}): {err.message[:160]}")
        if events[0]["type"] != "run_started" or events[-1]["type"] != "run_finished":
            problems.append(f"{f.parent.name}: must start with run_started and end with run_finished")
        ts = [e["t"] for e in events]
        if ts != sorted(ts):
            problems.append(f"{f.parent.name}: t not increasing")
        imgs = {e["id"] for e in events if e["type"] == "image"} | {
            p["id"] for p in events[0]["goal_parts"] if "data" in p and "id" in p}
        for e in events:
            for ref in _img_refs(e):
                if ref not in imgs:
                    problems.append(f"{f.parent.name}: seq {e['seq']} refers to unknown image {ref}")
    return problems


def _img_refs(e: dict):
    if e["type"] == "llm_request":
        for m in e["messages"]:
            for p in m.get("parts", []):
                if p.get("kind") == "image":
                    yield p["img"]
    elif e["type"] == "run_finished":
        yield from e["final_images"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slugs", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--check", action="store_true", help="only validate and scan the committed replays")
    ap.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    if not args.check:
        slugs = list(CURATED) if args.all or not args.slugs else args.slugs
        for s in slugs:
            convert(s, args.runs_dir, args.out)
    problems = validate(args.out)
    hits = scan(args.out)
    for p in problems:
        print("INVALID", p)
    for h in hits:
        print("SECRET?", h)
    print(f"{'schema OK' if not problems else str(len(problems)) + ' schema problems'}; "
          f"{'no secrets/personal paths found' if not hits else str(len(hits)) + ' suspicious strings'}")
    return 1 if problems or hits else 0


if __name__ == "__main__":
    sys.exit(main())
