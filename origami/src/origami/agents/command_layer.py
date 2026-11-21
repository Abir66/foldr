"""Command layer between an LLM (or a scripted agent) and FoldEnv: neutral sticker labels,
point snapping, "where are my stickers now" text, clearer same-spot errors.

- Labeler: internal point names (bl, p250_500, ...) <-> short neutral labels A, B, ... AA, AB ...
  Labels are assigned in order of first appearance within an episode and never change; the 9 landmarks
  are always A..I (bl, br, tr, tl, bottom_mid, right_mid, top_mid, left_mid, center).
  Old internal names are still accepted as arguments.
- CommandLayer.run(tool, args): translate labels -> internal names, snap loose points, apply on the env,
  and return a result dict whose text uses labels only.
"""

from __future__ import annotations

import json
import re

import numpy as np

from ..landmarks import NAMED_POINTS, resolve_point
from ..observe import _NAME_RE, name_to_orig

SNAP_TOL = 0.02  # fraction of the sheet size
_NAME_TOKEN = re.compile(r"\b(?:bl|br|tr|tl|bottom_mid|right_mid|top_mid|left_mid|center|p\d{3,4}_\d{3,4})\b")
_RESERVED = {n.upper() for n in ("bl", "br", "tr", "tl")}  # would look like the old names


def _gen_labels():
    n = 0
    while True:
        n += 1
        s, k = "", n
        while k:
            k, r = divmod(k - 1, 26)
            s = chr(65 + r) + s
        if s not in _RESERVED:
            yield s


class Labeler:
    def __init__(self, size: float = 1.0):
        self.size = size
        self._gen = _gen_labels()
        self.to_label: dict[str, str] = {}
        self.to_name: dict[str, str] = {}
        for nm in NAMED_POINTS:
            self.label(nm)

    def label(self, name: str) -> str:
        if name not in self.to_label:
            lab = next(self._gen)
            self.to_label[name] = lab
            self.to_name[lab] = name
        return self.to_label[name]

    def register(self, names) -> None:
        """Give labels to newly seen names, in a deterministic order (by sheet position)."""
        new = [n for n in names if n not in self.to_label]
        def key(n):
            try:
                o = name_to_orig(n, self.size)
                return (float(o[0]), float(o[1]), n)
            except ValueError:
                return (9e9, 9e9, n)
        for n in sorted(new, key=key):
            self.label(n)

    def internal(self, label: str) -> str | None:
        return self.to_name.get(label)

    def sub(self, text: str) -> str:
        """Replace internal names in free text by labels."""
        return _NAME_TOKEN.sub(lambda m: self.label(m.group(0)), text) if isinstance(text, str) else text

    def names_line(self) -> str:
        parts = []
        for nm, lab in self.to_label.items():
            o = name_to_orig(nm, self.size) / self.size
            parts.append(f"{lab}=({o[0]:g},{o[1]:g})")
        return ("Labels (a label marks a fixed spot of the ORIGINAL sheet, shown as its sheet coordinates, "
                "and follows that spot as the paper folds): " + " ".join(parts) + ".")


# ---- argument translation ---------------------------------------------------------------------

def translate(obj, lab: Labeler):
    """Labels -> internal names inside tool arguments; JSON-encoded string args are decoded."""
    if isinstance(obj, str):
        s = obj.strip()
        if len(s) >= 2 and s[0] == s[-1] == '"':
            try:
                return translate(json.loads(s), lab)
            except json.JSONDecodeError:
                pass
        if s[:1] in "{[":
            try:
                return translate(json.loads(s), lab)
            except json.JSONDecodeError:
                return obj
        return lab.internal(s) or obj
    if isinstance(obj, dict):
        return {k: (v if k in ("op", "description") else translate(v, lab)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [translate(v, lab) for v in obj]
    return obj


def _is_num_pair(o) -> bool:
    return isinstance(o, (list, tuple)) and len(o) == 2 and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in o)


def snap_points(action, state, infos: dict, lab: Labeler, tol: float = SNAP_TOL):
    """Snap point specs given as [x,y] or lerp to a known sheet point (landmark / crease vertex)
    within tol. -> (new action, notes). `infos`: label -> PointInfo (folded positions)."""
    notes: list[str] = []
    tol_abs = tol * state.size
    cand = [(np.array(pi.pos), k, pi) for k, pi in infos.items()]

    def near(p):
        best = None
        for pos, k, pi in cand:
            d = float(np.linalg.norm(pos - p))
            if d <= tol_abs and (best is None or d < best[0]):
                best = (d, k, pi)
        return best

    def walk(o):
        if isinstance(o, dict):
            if "lerp" in o and "t" in o:
                try:
                    hit = near(resolve_point(state, o))
                except Exception:  # noqa: BLE001 - bad spec: let the env report it
                    hit = None
                if hit:
                    notes.append(f"a lerp point was snapped to {hit[1]}")
                    return {"orig": [float(hit[2].orig[0]), float(hit[2].orig[1])]}
            return {k: (v if k in ("op", "description") else walk(v)) for k, v in o.items()}
        if _is_num_pair(o):
            hit = near(np.array(o, dtype=float))
            if hit and hit[0] > 1e-9:
                notes.append(f"[{o[0]:g},{o[1]:g}] was snapped to {hit[1]}")
                return {"orig": [float(hit[2].orig[0]), float(hit[2].orig[1])]}
            return o
        if isinstance(o, (list, tuple)):
            return [walk(v) for v in o]
        return o

    return walk(action), notes


# ---- where the stickers are -------------------------------------------------------------------

def where_words(pos, lo, hi) -> str:
    w = np.maximum(np.asarray(hi) - np.asarray(lo), 1e-9)
    fx, fy = (pos[0] - lo[0]) / w[0], (pos[1] - lo[1]) / w[1]
    h = "left" if fx < 0.2 else "right" if fx > 0.8 else ""
    v = "bottom" if fy < 0.2 else "top" if fy > 0.8 else ""
    return "-".join(p for p in (v, h) if p) or "center"


def sticker_groups(infos: dict, state) -> list[tuple[str, tuple, list[str]]]:
    """Visible labelled points grouped by spot: [(words, (x,y), [labels])], top to bottom."""
    lo, hi = state.bbox()
    groups: dict[tuple, list[str]] = {}
    for k, pi in infos.items():
        if pi.front or pi.back:
            groups.setdefault((round(pi.pos[0] / state.size, 3), round(pi.pos[1] / state.size, 3)), []).append(k)
    out = []
    for key, labs in sorted(groups.items(), key=lambda kv: (-kv[0][1], kv[0][0])):
        pos = (key[0] * state.size, key[1] * state.size)
        out.append((where_words(pos, lo, hi), pos, sorted(labs, key=lambda x: (len(x), x))))
    return out


def stickers_text(infos: dict, state, cap: int = 24) -> str:
    gs = sticker_groups(infos, state)
    L = ["Where stickers are now (current front view; words are relative to the current outline):"]
    for words, pos, labs in gs[:cap]:
        extra = "  <- stacked on the same spot" if len(labs) > 1 else ""
        L.append(f"  {words} ({pos[0]:.2f},{pos[1]:.2f}): {', '.join(labs)}{extra}")
    if len(gs) > cap:
        L.append(f"  (+{len(gs) - cap} more spots)")
    return "\n".join(L)


# ---- the layer --------------------------------------------------------------------------------

def call_to_action(name: str, args: dict) -> dict:
    a = {"op": name}
    for k, v in args.items():
        if k in ("plan",) or v in (None, "", "all") and k == "layers":
            continue
        a[k] = v
    if name == "rotate" and "degrees" in a:
        a["degrees"] = float(a["degrees"])
    return a


def _leaf_names(o, out: list):
    if isinstance(o, str):
        if o in NAMED_POINTS or _NAME_RE.match(o):
            out.append(o)
    elif isinstance(o, dict):
        for k, v in o.items():
            if k not in ("op", "description"):
                _leaf_names(v, out)
    elif isinstance(o, (list, tuple)):
        for v in o:
            _leaf_names(v, out)


class CommandLayer:
    def __init__(self, env, snap_tol: float = SNAP_TOL):
        self.env = env
        self.labels = Labeler(env.size)
        env.labeler = self.labels
        self.snap_tol = snap_tol

    def infos(self) -> dict:
        return self.env.observe(images=False).points  # keyed by label

    def stickers_text(self, obs=None) -> str:
        return stickers_text((obs or self.env.observe(images=False)).points, self.env.state)

    def _same_spot_help(self, action, infos) -> str:
        names: list[str] = []
        _leaf_names(action, names)
        spots: dict[tuple, list[str]] = {}
        for n in dict.fromkeys(names):
            try:
                p = resolve_point(self.env.state, n)
            except Exception:  # noqa: BLE001
                continue
            spots.setdefault((round(float(p[0]), 4), round(float(p[1]), 4)), []).append(self.labels.label(n))
        same = [v for v in spots.values() if len(v) > 1]
        if not same:
            return ""
        msg = " ".join(f"{' and '.join(v)} are on the same spot right now (stacked)." for v in same)
        gs = sticker_groups(infos, self.env.state)
        alts = [f"{labs[0]} ({w})" for w, _, labs in gs][:8]
        return msg + " Pick labels on different spots, e.g. " + ", ".join(alts) + "."

    def run(self, name: str, args: dict) -> dict:
        env, lab = self.env, self.labels
        if name == "undo":
            r = env.undo(int(args.get("n") or 1))
            return {"tool": "undo", "ok": r.ok, "text": lab.sub(r.to_text(env.feedback) if not r.ok else r.reason)}
        if name == "inspect":
            q = args.get("query", "")
            qt = translate(q, lab)
            if isinstance(qt, str):
                qt = " ".join(str(lab.internal(t) or t) for t in qt.split())
            out = env.inspect(qt)
            return {"tool": "inspect", "query": q, "text": lab.sub(json.dumps(out, default=str))[:1500]}
        action = translate(call_to_action(name, args), lab)
        notes: list[str] = []
        if name not in ("flip", "rotate", "unfold"):
            action, notes = snap_points(action, env.state, {k: v for k, v in self.infos().items()}, lab, self.snap_tol)
        r = env.apply(action)
        text = lab.sub(r.to_text(env.feedback))
        if notes:
            text += " [snapped: " + "; ".join(notes) + "]"
        if not r.ok:
            if r.code == "SAME_POINT":
                help_ = self._same_spot_help(action, self.infos())
                if help_:
                    text += " " + help_
            elif r.code == "BAD_REF":
                text += " Labels known so far: " + ", ".join(lab.to_name) + "."
        return {"tool": name, "action": r.action, "ok": r.ok, "code": r.code, "text": text,
                "warnings": r.warnings, "snapped": notes}
