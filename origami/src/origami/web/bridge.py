"""JSON-in / JSON-out API of the folding engine for the browser (Pyodide worker) and other hosts.

Every method takes one JSON object (params) and returns one JSON object. Nothing else crosses the
boundary: no Python objects, no files. Binary results are base64 strings. Another engine can stand in
for this module by implementing the same methods with the same shapes.

    from origami.web import bridge
    out_json = bridge.call("session.new", '{"size": 1, "colored_side": "up"}')
    out_json = bridge.call("session.menu", '{"deadline_ms": 2500}', progress=lambda s: print(s))

Methods (params -> result). `state` = steps_export state JSON (with "world"), `timeline_step` = one
timeline step JSON, `points` = tappable points, `rev` = session revision (bumped by every change):

  init            {stage: "core"|"export"}            -> {versions, stage}
  timeline        {model}                             -> {timeline|null, error|null, failed_step|null, shown_steps}
  session.new     {size?, colored_side?}              -> {rev, state, timeline, points}
  session.apply   {step}                              -> {rev, ok, code, text, timeline_step?, state?, points?}
  session.preview {step}                              -> {rev, ok, code, text, state?, line?, move?}
                  line = fold line clipped to the paper's bbox [[x,y],[x,y]], move = the point that moves
  session.undo    {n?}                                -> {rev, ok, text, timeline_len, state, points}
  session.menu    {more?, deadline_ms?, sections?}    -> {rev, options, truncated, skipped, seconds}
                  progress pushes (JSON string): {rev, section, options}
  session.custom  {p, q, kind: "through"|"onto", op?, side_point?, side?, layers?, preview?}
                  -> same as session.apply (or session.preview when preview=true), plus the built `step`
  session.steps   {}                                  -> {model}
  session.state   {}                                  -> {rev, state, points, timeline_len}
  export          {model? (default: the session's steps), kinds: [...]} -> {files: [{name, mime, b64}], errors}
                  kinds: "fold", "model_json", "timeline_json" (core); "png_zip", "cp_png" (export stage)

Point JSON: {label, name, xy, orig, depth, front, back, stacked: [labels at the same spot]}.
Menu option JSON: {id, description, section, rank, kind, step, result_state}.
Errors: call() returns {"exception": {"type", "message"}} instead of raising.
"""

from __future__ import annotations

import base64
import io
import json
import re
import time
import zipfile

API_VERSION = 1

_SESSION = None


def _versions() -> dict:
    import sys

    import numpy

    out = {"api": API_VERSION, "python": sys.version.split()[0], "numpy": numpy.__version__,
           "platform": sys.platform}
    try:
        from importlib.metadata import version

        out["origami"] = version("origami")
    except Exception:  # noqa: BLE001
        out["origami"] = None
    return out


# ---- stateless methods ------------------------------------------------------------------------

def init(params: dict) -> dict:
    import os

    stage = params.get("stage", "core")
    os.environ["ORIGAMI_MENU_WORKERS"] = "0"
    from .. import env, menu, steps_export  # noqa: F401  (import cost paid here)

    if stage == "export":
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from .. import render  # noqa: F401

        fig = plt.figure(figsize=(1, 1))  # builds the font cache once
        fig.text(0.5, 0.5, "x")
        fig.savefig(io.BytesIO(), format="png")
        plt.close(fig)
    elif stage != "core":
        raise ValueError(f"unknown stage {stage!r}")
    return {"versions": _versions(), "stage": stage}


def timeline(params: dict) -> dict:
    """Viewer timeline of a model; on a failing step, the timeline of the steps before it."""
    from ..engine import FoldError
    from ..steps_export import build_timeline

    model = params.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("steps"), list):
        return {"timeline": None, "error": 'not a step file: expected a JSON object with a "steps" list',
                "failed_step": None, "shown_steps": 0}
    try:
        return {"timeline": build_timeline(model), "error": None, "failed_step": None,
                "shown_steps": len(model["steps"])}
    except (FoldError, ValueError, KeyError, TypeError) as e:
        msg = str(e)
        m = re.match(r"step (\d+)", msg)
        k = int(m.group(1)) if m else None
        out = {"timeline": None, "error": f"{type(e).__name__}: {msg}", "failed_step": k, "shown_steps": 0}
        if k is not None and k >= 1:
            try:
                out["timeline"] = build_timeline(dict(model, steps=model["steps"][: k - 1]))
                out["shown_steps"] = k - 1
            except Exception:  # noqa: BLE001
                pass
        return out


def _states_of(model: dict):
    from ..cli import run_model

    return run_model(model)


def export(params: dict) -> dict:
    kinds = list(params.get("kinds") or [])
    model = params.get("model")
    if model is None:
        model = _session().steps_model()
    name = str(model.get("name") or "model")
    title = str(model.get("title") or name)
    files, errors = [], []
    states = None

    def add(fname, mime, data):
        files.append({"name": fname, "mime": mime,
                      "b64": base64.b64encode(data if isinstance(data, bytes) else data.encode()).decode()})

    for kind in kinds:
        try:
            if kind == "model_json":
                add(f"{name}.json", "application/json", json.dumps(model, indent=1))
            elif kind == "timeline_json":
                from ..steps_export import build_timeline, timeline_to_json

                add(f"{name}.timeline.json", "application/json", timeline_to_json(build_timeline(model), 5))
            elif kind == "fold":
                from ..fold_export import to_fold

                states = states or _states_of(model)
                add(f"{name}.fold", "application/json", json.dumps(to_fold(states[-1], title), indent=1))
            elif kind in ("png_zip", "cp_png"):
                states = states or _states_of(model)
                add(*_render(kind, model, states, name, title))
            else:
                errors.append({"kind": kind, "message": f"unknown export kind {kind!r}"})
        except Exception as e:  # noqa: BLE001
            errors.append({"kind": kind, "message": f"{type(e).__name__}: {e}"})
    return {"files": files, "errors": errors}


def _render(kind, model, states, name, title):
    import matplotlib

    matplotlib.use("Agg")
    from .. import render
    from ..cli import _half_span, _limits

    render.set_colored_side(model.get("paper", {}).get("colored_side", "up"))
    render.set_layer_offset(0.0)
    if kind == "cp_png":
        buf = io.BytesIO()
        render.render_crease_pattern(buf, states[-1])
        return f"{name}_crease_pattern.png", "image/png", buf.getvalue()
    steps = model["steps"]
    half = _half_span(states)
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
        for k, state in enumerate(states):
            if k < len(steps):
                nxt = steps[k]
                t = f"Step {k + 1}: {nxt.get('description', nxt['op'])}"
            else:
                nxt, t = None, f"Finished: {title}"
            buf = io.BytesIO()
            render.render_step_png(buf, state, t, nxt, _limits(state, half))
            zf.writestr(f"step_{k:02d}.png", buf.getvalue())
        buf = io.BytesIO()
        render.render_crease_pattern(buf, states[-1])
        zf.writestr("crease_pattern.png", buf.getvalue())
    return f"{name}_steps.zip", "application/zip", zbuf.getvalue()


# ---- the folding session (one at a time) ----------------------------------------------------

class Session:
    def __init__(self, size: float = 1.0, colored_side: str = "up"):
        from ..agents.command_layer import Labeler
        from ..env import FoldEnv
        from ..steps_export import TimelineBuilder

        if colored_side not in ("up", "down"):
            raise ValueError("colored_side must be 'up' or 'down'")
        self.env = FoldEnv(None, size=float(size), colored_side=colored_side)
        self.env.labeler = Labeler(self.env.size)
        self.builder = TimelineBuilder(self.env.state, name="my_fold", title="My fold", colored_side=colored_side)
        self.rev = 0
        self._menu = None  # (rev, {section: [option JSON]}, {hash of every listed option})

    # -- views --------------------------------------------------------------------------
    def points(self) -> list[dict]:
        from ..observe import observe

        lab = self.env.labeler
        infos = observe(self.env.state, colored_side=self.env.colored_side, images=False, labeler=lab).points
        out = []
        for label, pi in infos.items():
            out.append({"label": label, "name": lab.internal(label), "xy": [float(pi.pos[0]), float(pi.pos[1])],
                        "orig": [float(pi.orig[0]), float(pi.orig[1])], "depth": int(pi.depth),
                        "front": bool(pi.front), "back": bool(pi.back)})
        spots: dict = {}
        for p in out:
            spots.setdefault((round(p["xy"][0], 6), round(p["xy"][1], 6)), []).append(p["label"])
        for p in out:
            p["stacked"] = [x for x in spots[(round(p["xy"][0], 6), round(p["xy"][1], 6))] if x != p["label"]]
        return out

    def text(self, r) -> str:
        t = r.to_text(self.env.feedback)
        return self.env.labeler.sub(t)

    def steps_model(self) -> dict:
        from ..observe import step_line

        steps = []
        for st in self.env.trajectory:
            st = dict(st)
            st.setdefault("description", self.env.labeler.sub(step_line(st)))
            steps.append(st)
        return {"name": "my_fold", "title": "My fold",
                "paper": {"size": self.env.size, "colored_side": self.env.colored_side}, "steps": steps}

    # -- actions ------------------------------------------------------------------------
    def apply(self, step) -> dict:
        before = self.env.state
        r = self.env.apply(step)
        out = {"ok": bool(r.ok), "code": r.code, "text": self.text(r)}
        if r.ok:
            try:
                ts = self.builder.append(r.action, before, self.env.state)
            except Exception as e:  # noqa: BLE001  (the viewer can't show this step: take it back)
                self.env.undo(1)
                out.update(ok=False, code="VIEWER", text=f"REJECTED [VIEWER]: {type(e).__name__}: {e}")
            else:
                self.rev += 1
                out.update(timeline_step=ts, state=self.builder.states[-1], points=self.points())
        out["rev"] = self.rev
        return out

    def preview(self, step) -> dict:
        from ..steps_export import state_json

        self.env.stats["previews"] += 1
        r, new, _ = self.env._evaluate(step, commit=False)
        out = {"rev": self.rev, "ok": bool(r.ok), "code": r.code, "text": self.text(r)}
        if new is not None:
            out["state"] = state_json(new.state)
        seg = self._fold_segment(step)
        if seg is not None:
            out["line"], out["move"] = seg
        return out

    def _fold_segment(self, step):
        """The fold line clipped to the paper's bounding box, plus the point that moves (engine coords)."""
        import numpy as np

        from ..engine import fold_geometry
        from ..env import normalize_action

        if not isinstance(step, dict) or "line" not in step:
            return None
        st = self.env.state
        try:
            a, u, mv = fold_geometry(st, normalize_action(step, st.size))
        except Exception:  # noqa: BLE001
            return None
        pts = np.vstack([f.folded for f in st.faces])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        ts = []
        for k in (0, 1):
            if abs(u[k]) > 1e-12:
                for edge in (lo[k], hi[k]):
                    t = (edge - a[k]) / u[k]
                    p = a + t * u
                    if np.all(p >= lo - 1e-9) and np.all(p <= hi + 1e-9):
                        ts.append(float(t))
        if len(ts) < 2:
            return None
        p0, p1 = a + min(ts) * u, a + max(ts) * u
        return [[float(p0[0]), float(p0[1])], [float(p1[0]), float(p1[1])]], [float(mv[0]), float(mv[1])]

    def undo(self, n: int = 1) -> dict:
        have = len(self.env._undo_stack)
        r = self.env.undo(n)
        if r.ok:
            self.builder.pop(min(max(1, int(n)), have))
            self.rev += 1
        return {"rev": self.rev, "ok": bool(r.ok), "text": self.text(r) if not r.ok else self.env.labeler.sub(r.reason),
                "timeline_len": len(self.builder), "state": self.builder.states[-1], "points": self.points()}

    def custom(self, params: dict) -> dict:
        """Fold along a line through two points (or bringing p onto q). preview=true only checks it."""
        kind = params.get("kind", "through")
        if kind not in ("through", "onto"):
            return {"rev": self.rev, "ok": False, "code": "BAD_ARGS", "text": "kind must be 'through' or 'onto'"}
        try:
            step = self._custom_step(params, kind)
        except ValueError as e:
            return {"rev": self.rev, "ok": False, "code": "BAD_ARGS", "text": f"REJECTED [BAD_ARGS]: {e}"}
        out = self.preview(step) if params.get("preview") else self.apply(step)
        out["step"] = step
        return out

    def _custom_step(self, params: dict, kind: str) -> dict:
        p, q = self._pt(params.get("p")), self._pt(params.get("q"))
        step = {"op": params.get("op", "valley"), "line": {kind: [p, q]}}
        mv = params.get("side_point")
        if mv is not None:
            step["move"] = self._pt(mv)
        elif kind == "onto":
            step["move"] = p
        else:
            step["move"] = self._side_point(p, q, int(params.get("side", 1)))
        layers = params.get("layers", "all")
        spec = {"all": "all", "top": {"top": 1}, "flap": {"top": 1, "flap": True}}.get(layers, layers) \
            if isinstance(layers, str) else layers
        if spec != "all":
            step["layers"] = spec
        return step

    def _pt(self, spec):
        """Point label (from `points`) -> its sheet name; [x, y] (folded coords) passes through."""
        if isinstance(spec, str):
            nm = self.env.labeler.internal(spec)
            return nm if nm is not None else spec
        if isinstance(spec, (list, tuple)) and len(spec) == 2:
            return [float(spec[0]), float(spec[1])]
        raise ValueError(f"bad point {spec!r}")

    def _side_point(self, p, q, side: int):
        """A point of the paper on the given side of line p -> q (+1 = left), far from the line."""
        import numpy as np

        from .. import geometry as g
        from ..landmarks import resolve_point

        from ..env import normalize_action

        st = self.env.state

        a = resolve_point(st, normalize_action(p, st.size))
        b = resolve_point(st, normalize_action(q, st.size))
        u = b - a
        best, far = None, -1.0
        for f in st.faces:
            m = f.folded.mean(axis=0)
            for v in f.folded:  # halfway from a corner to the face centre: inside the face, off the line
                c = 0.5 * (v + m)
                d = float(g.side(a, u, c)) * (1 if side >= 0 else -1)
                if d > far:
                    best, far = c, d
        if best is None or far <= 0:
            raise ValueError("no paper on that side of the line")
        return [float(best[0]), float(best[1])]

    def menu(self, params: dict, progress=None) -> dict:
        from ..menu import candidates
        from ..steps_export import state_json

        more = params.get("more") or None
        deadline = params.get("deadline_ms")
        sections = params.get("sections")
        rev = self.rev
        exclude = ()
        if sections is not None and self._menu is not None and self._menu[0] == rev:
            listed = self._menu[1]
            exclude = {h for fam, hs in listed.items() if fam not in sections for h in hs}

        def opt_json(o, idx):
            return {"id": idx, "description": o.description, "section": o.fam, "rank": o.fam_rank, "kind": o.kind,
                    "step": o.step, "result_state": state_json(o.state) if o.state is not None else None}

        counter = [0]

        def on_section(fam, opts):
            if progress is None:
                return
            js = []
            for o in opts:
                counter[0] += 1
                js.append(opt_json(o, counter[0]))
            progress(json.dumps({"rev": rev, "section": fam, "options": js}))

        t0 = time.monotonic()
        m = candidates(self.env, more=more, labeler=self.env.labeler,
                       deadline_s=None if deadline is None else float(deadline) / 1000.0,
                       on_section=on_section, sections=None if sections is None else set(sections),
                       exclude_hashes=exclude)
        options = [opt_json(o, o.id) for o in m]
        listed = dict(self._menu[1]) if (self._menu is not None and self._menu[0] == rev) else {}
        for fam in {o.fam for o in m}:
            listed[fam] = {o.hash for o in m if o.fam == fam}
        self._menu = (rev, listed)
        return {"rev": rev, "options": options, "truncated": dict(m.truncated), "skipped": list(m.skipped),
                "seconds": round(time.monotonic() - t0, 3)}

    def state(self) -> dict:
        return {"rev": self.rev, "state": self.builder.states[-1], "points": self.points(),
                "timeline_len": len(self.builder)}


def _session() -> Session:
    if _SESSION is None:
        raise RuntimeError("no session: call session.new first")
    return _SESSION


def session_new(params: dict) -> dict:
    global _SESSION
    _SESSION = Session(params.get("size", 1.0), params.get("colored_side", "up"))
    s = _SESSION
    return {"rev": s.rev, "state": s.builder.states[-1], "timeline": s.builder.to_dict(), "points": s.points()}


METHODS = {
    "init": lambda p, _: init(p),
    "timeline": lambda p, _: timeline(p),
    "export": lambda p, _: export(p),
    "session.new": lambda p, _: session_new(p),
    "session.apply": lambda p, _: _session().apply(p.get("step")),
    "session.preview": lambda p, _: _session().preview(p.get("step")),
    "session.undo": lambda p, _: _session().undo(int(p.get("n", 1))),
    "session.custom": lambda p, _: _session().custom(p),
    "session.menu": lambda p, prog: _session().menu(p, prog),
    "session.steps": lambda p, _: {"model": _session().steps_model()},
    "session.state": lambda p, _: _session().state(),
}


def handle(method: str, params: dict | None = None, progress=None) -> dict:
    """Python-side entry: dict in, dict out (raises on errors)."""
    fn = METHODS.get(method)
    if fn is None:
        raise ValueError(f"unknown method {method!r}")
    return fn(params or {}, progress)


def call(method: str, params_json: str = "{}", progress=None) -> str:
    """Host entry: JSON string in, JSON string out. Errors come back as {"exception": {...}}.
    progress: optional callable taking one JSON string (menu sections as they finish)."""
    try:
        params = json.loads(params_json) if params_json else {}
        out = handle(method, params, progress)
    except Exception as e:  # noqa: BLE001
        out = {"exception": {"type": type(e).__name__, "message": str(e)}}
    return json.dumps(out, allow_nan=False)


__all__ = ["call", "handle", "METHODS", "API_VERSION", "Session"]
