"""Build the files the frontend loads for every example model.

Run from the repo root:  uv run --project origami python tools/build_examples.py [--out DIR] [NAME ...]
Writes, per examples/models/<name>.json, into frontend/public/examples/:
  <name>.timeline.json  <name>.model.json  <name>.fold  <name>.thumb.svg
  <name>.crease_pattern.png  <name>.steps.zip (step_NN.png + crease_pattern.png)
examples/runs/<slug>/ replays are copied as run_<slug>.events.jsonl with exports of their solution
(run_<slug>.*), and one index.json lists examples and runs.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

from origami import render  # noqa: E402
from origami.cli import _half_span, _limits, run_model  # noqa: E402
from origami.fold_export import to_fold  # noqa: E402
from origami.steps_export import build_timeline, timeline_to_json  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "examples" / "models"
RUNS_DIR = ROOT / "examples" / "runs"
DEFAULT_OUT = ROOT / "frontend" / "public" / "examples"

# Listing order and short names for the library; models not listed here sort after, by name.
LABELS = {
    "crane": "Crane",
    "cup": "Paper cup",
    "hat": "Samurai helmet",
    "fortune_teller": "Fortune teller",
    "bird_base": "Bird base",
    "triangle": "Triangle",
}

# Paper colours of the thumbnails (SVG files are shown as images, so no CSS variables).
FRONT, BACK, INK = "#e07a3f", "#fbf6ec", "#3a2a1a"


def thumb_svg(state: dict) -> str:
    """Top-down silhouette of a finished state, as the viewer would show it from above."""
    M = np.array(state["world"]["M"], float)
    polys = []
    for f in sorted(state["faces"], key=lambda f: f["layer"]):
        T = np.array(f["T"], float)
        pts = np.array(f["orig"], float) @ T[:, :2].T + T[:, 2]
        pts = pts @ M[:, :2].T + M[:, 2]
        polys.append((pts, f["front_up"]))
    allp = np.vstack([p for p, _ in polys])
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    scale = 84 / max(float((hi - lo).max()), 1e-9)
    off = 50 - (lo + hi) / 2 * scale
    paths = []
    for pts, front in polys:
        q = pts * scale + off
        d = "M" + " ".join(f"{x:.1f},{100 - y:.1f}" for x, y in q) + "Z"
        paths.append(f'<path d="{d}" fill="{FRONT if front else BACK}"/>')
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        f'<g stroke="{INK}" stroke-opacity=".55" stroke-width="1.1" stroke-linejoin="round">'
        + "".join(paths)
        + "</g></svg>\n"
    )


def step_images(model: dict, states) -> list[tuple[str, bytes]]:
    render.set_colored_side(model.get("paper", {}).get("colored_side", "up"))
    render.set_layer_offset(0.0)
    steps = model["steps"]
    half = _half_span(states)
    out = []
    for k, state in enumerate(states):
        if k < len(steps):
            nxt = steps[k]
            title = f"Step {k + 1}: {nxt.get('description', nxt['op'])}"
        else:
            nxt, title = None, f"Finished: {model.get('title', model.get('name', ''))}"
        buf = io.BytesIO()
        render.render_step_png(buf, state, title, nxt, _limits(state, half))
        out.append((f"step_{k:02d}.png", buf.getvalue()))
    return out


def crease_png(states) -> bytes:
    buf = io.BytesIO()
    render.render_crease_pattern(buf, states[-1])
    return buf.getvalue()


def write_zip(path: Path, files: list[tuple[str, bytes]]) -> None:
    # Fixed timestamps keep the zip identical between builds.
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files:
            zf.writestr(zipfile.ZipInfo(name, (2000, 1, 1, 0, 0, 0)), data, zipfile.ZIP_DEFLATED)


def build_one(path: Path, out: Path) -> dict:
    raw = path.read_bytes()
    return build_model(json.loads(raw), path.stem, LABELS.get(path.stem), raw, out)


def build_model(model: dict, name: str, label: str | None, raw: bytes, out: Path) -> dict:
    states = run_model(model)
    tl = build_timeline(model)
    n = len(tl["steps"])
    assert len(tl["states"]) == n + 1 == len(states)

    (out / f"{name}.timeline.json").write_text(timeline_to_json(tl, ndigits=5))
    (out / f"{name}.model.json").write_text(json.dumps(model, indent=2) + "\n")
    (out / f"{name}.fold").write_text(json.dumps(to_fold(states[-1], model.get("title", ""))))
    (out / f"{name}.thumb.svg").write_text(thumb_svg(tl["states"][-1]))

    images = step_images(model, states)
    cp = crease_png(states)
    (out / f"{name}.crease_pattern.png").write_bytes(cp)
    write_zip(out / f"{name}.steps.zip", images + [("crease_pattern.png", cp)])

    return {
        "name": name,
        "label": label or model.get("title") or name,
        "title": model.get("title", name),
        "description": model.get("description", ""),
        "steps": [{"op": s["op"], "description": s.get("description", "")} for s in tl["steps"]],
        "files": {
            "timeline": f"{name}.timeline.json",
            "model": f"{name}.model.json",
            "fold": f"{name}.fold",
            "thumb": f"{name}.thumb.svg",
            "creasePattern": f"{name}.crease_pattern.png",
            "stepsZip": f"{name}.steps.zip",
        },
        # changes with the model, so the browser refetches after a rebuild
        "version": hashlib.sha1(raw).hexdigest()[:8],
    }


def build_runs(out: Path) -> list[dict]:
    """Curated replays (examples/runs/<slug>/): the event log, plus the exports of the run's own solution."""
    runs = []
    for meta_path in sorted(RUNS_DIR.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        slug = meta["slug"]
        events_raw = (meta_path.parent / "events.jsonl").read_bytes()
        (out / f"run_{slug}.events.jsonl").write_bytes(events_raw)
        files = {"events": f"run_{slug}.events.jsonl"}
        last = json.loads(events_raw.splitlines()[-1])
        if last.get("solution") and last["solution"].get("steps"):
            solution = json.dumps(last["solution"], indent=2).encode()
            built = build_model(last["solution"], f"run_{slug}", meta["title"], solution, out)
            files.update({k: v for k, v in built["files"].items() if k != "model"}, model=built["files"]["model"])
        runs.append({
            "slug": slug, "title": meta["title"], "blurb": meta.get("blurb", ""), "model": meta["model"],
            "example": meta["example"], "goal": "text" if meta["goal"]["kind"] == "text" else "example",
            "goalText": meta["goal"].get("text"), "mode": meta["mode"], "passed": meta["outcome"]["passed"],
            "costUsd": meta["costUsd"], "turns": meta["turns"], "files": files,
            "version": hashlib.sha1(events_raw).hexdigest()[:8],
        })
        print(f"run {slug}: {len(files)} files")
    return runs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="only these models (default: all)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    paths = sorted(MODELS.glob("*.json"))
    if args.names:
        paths = [p for p in paths if p.stem in args.names]
    if not paths:
        print("no models found", file=sys.stderr)
        return 1
    order = {n: i for i, n in enumerate(LABELS)}
    paths.sort(key=lambda p: (order.get(p.stem, len(order)), p.stem))

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    entries = []
    for p in paths:
        entry = build_one(p, args.out)
        entries.append(entry)
        print(f"{entry['name']}: {len(entry['steps'])} steps")
    runs = [] if args.names else build_runs(args.out)
    (args.out / "index.json").write_text(json.dumps({"examples": entries, "runs": runs}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
