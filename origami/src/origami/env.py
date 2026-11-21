"""Agent environment: one episode = fold a fresh sheet until it matches a target state.

Pure Python, no LLM.

    env = FoldEnv("models/cup.json")          # target = that model's final state
    obs = env.reset()                          # Observation (observe.py) + memory notes
    r = env.apply({"op": "valley", "line": {"onto": ["tr", "bl"]}})
    r.ok, r.code, r.reason                     # rejected moves leave the state unchanged
    env.undo(); env.checkpoint("a"); env.restore("a"); env.preview(action); env.inspect("layers_at center")
    score = env.submit()                       # match levels vs target + efficiency

    FoldEnv(None, size=1.0, colored_side="up")  # free folding: no target, par, goal text or score
                                               # (submit() -> Score with passed=None)

Rules (decided 2026-11-20):
- every move is physics-checked (validate.check_step); any error -> rejected, state unchanged;
- flips / rotations are free (not moves; logged separately as view ops);
- par = number of non-view steps of the target model's own sequence;
- goal judged on the final state only: headline = M2 (outline + visible colours both sides);
- feedback tier F0 (just "rejected") / F1 (codes + reasons, default) / F2 (+ shape score after each
  move) / F3 (+ per-part scores: outline IoU, colour front/back);
- tabu memory: canonical state hash -> notes of what was tried from there (rejected / undone).

Sheet point names from the observation (`p250_500` = sheet point (0.25, 0.5)) are accepted anywhere
a point is expected; they are turned into {"orig": [x, y]} before the engine sees them.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .engine import FoldError, Simulator, VIEW_OPS
from .match import VisitedTable, match
from .observe import _NAME_RE, Observation, inspect as _inspect, name_to_orig, observe, step_line
from .validate import STATE_KINDS, _same_spot, check_step

FEEDBACK_TIERS = ("F0", "F1", "F2", "F3")
NO_PROGRESS_K = 5  # moves without reaching a new state -> nudge

# ---- error codes ---------------------------------------------------------------------------

_MSG_CODES = [  # (substring of engine / landmark message, code, hint)
    ("unknown landmark", "BAD_REF", "use a point name from the observation (corners, mids, center, pXXX_YYY)"),
    ("bad point spec", "BAD_REF", "a point is a name, [x,y], {\"orig\":[x,y]}, {\"mid\":[P,Q]}, {\"lerp\":[P,Q],\"t\":..}"),
    ("bad line", "BAD_ARGS", "a line is {\"onto\":[P,Q]}, {\"through\":[P,Q]}, {\"bisect\":[L1,L2]}, ..."),
    ("unknown op", "BAD_ARGS", "use one of the listed tools"),
    ("lies on the fold line", "MOVE_ON_LINE", "pick a 'move' point strictly on the side that should move"),
    ("nothing to fold", "LINE_MISSES_PAPER", "the line must cross the paper with paper on the moving side"),
    ("nothing to unfold", "PRECONDITION", "there is no fold to unfold"),
    ("reserved", "NOT_IMPLEMENTED", "this technique is not available yet"),
    ("point_onto_line", "NO_SOLUTION", "that point can't reach that line with this pivot / direction"),
    ("bisect", "NO_SOLUTION", "the two lines must be different"),
    ("needs a 'move' point", "BAD_ARGS", "give a 'move' point"),
    ("zero-length", "SAME_POINT", "the two points are at the same place now; pick points that are apart"),
]


def _code_for_exc(e: Exception) -> tuple[str, str]:
    if isinstance(e, KeyError):
        return "BAD_ARGS", f"missing field {e}"
    if not isinstance(e, (FoldError, ValueError)):
        return "BAD_ARGS", f"bad step data ({type(e).__name__})"
    msg = str(e)
    for sub, code, hint in _MSG_CODES:
        if sub in msg:
            return code, hint
    return "PRECONDITION", "this move is not possible on the current paper"


# ---- results -------------------------------------------------------------------------------

@dataclass
class StepResult:
    ok: bool
    action: dict
    line: str  # one-line form of the action
    code: str = "OK"  # OK or an error code (BAD_REF, COLLISION, TACO_TORTILLA, ...)
    reason: str = ""
    hint: str = ""
    warnings: list[str] = field(default_factory=list)
    affected: list = field(default_factory=list)  # sheet points involved in the errors
    revisited: int | None = None  # this move led to a state first seen at step k
    progress: dict | None = None  # F2/F3 only
    committed: bool = False  # False for preview / rejected

    def to_text(self, tier: str = "F1") -> str:
        if self.ok:
            s = f"OK: {self.line}"
            if self.revisited is not None:
                s += f" (this state was already reached at step {self.revisited})"
            if self.progress:
                s += " | progress " + ", ".join(f"{k}={v:.2f}" for k, v in self.progress.items())
            return s
        if tier == "F0":
            return f"REJECTED: {self.line} (invalid move)"
        s = f"REJECTED [{self.code}]: {self.line} -- {self.reason}"
        if self.hint:
            s += f" Hint: {self.hint}"
        return s

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Score:
    target: str | None  # None: no target (free folding); then every match field below is None too
    m1: bool | None
    m2: bool | None
    m3: bool | None
    m4: bool | None
    scores: dict
    shape_score: float | None
    iou: float | None
    colour_front: float | None
    colour_back: float | None
    moves: int  # committed non-view moves
    view_ops: int
    par: int | None
    length_ratio: float | None  # moves / par (successes only)
    spl: float  # success * par / max(par, moves)
    deletable: int | None  # moves removable without losing M2 (only computed on success)
    undone: int  # moves taken back by undo / restore
    rejected: int  # illegal attempts
    executed: int  # all committed-at-some-point moves (incl. undone)
    previews: int
    inspects: int
    solution: list  # committed steps (= tutorial)

    @property
    def passed(self) -> bool | None:
        return self.m2

    def to_dict(self) -> dict:
        d = asdict(self)
        d["passed"] = self.passed
        return d


# ---- helpers -------------------------------------------------------------------------------

def is_view(step: dict) -> bool:
    return isinstance(step, dict) and step.get("op") in VIEW_OPS


def normalize_action(action, size: float = 1.0):
    """Copy of `action` with sheet point names (`p250_500`) replaced by {"orig": [x, y]} and
    JSON-encoded string values (LLM tool args) decoded."""
    if isinstance(action, str):
        s = action.strip()
        if len(s) >= 2 and s[0] == s[-1] == '"':  # JSON-quoted string from a tool arg: '"bl"'
            try:
                return normalize_action(json.loads(s), size)
            except json.JSONDecodeError:
                pass
        if _NAME_RE.match(s):
            return {"orig": [float(v) for v in name_to_orig(s, size)]}
        if s[:1] in "{[":
            try:
                return normalize_action(json.loads(s), size)
            except json.JSONDecodeError:
                return action
        return action
    if isinstance(action, dict):
        return {k: (v if k in ("op", "description") else normalize_action(v, size)) for k, v in action.items()}
    if isinstance(action, (list, tuple)):
        return [normalize_action(v, size) for v in action]
    return action


def _sheet_pts(issues) -> list:
    return [i.sheet_point for i in issues if getattr(i, "sheet_point", None) is not None]


def try_step(sim: Simulator, step: dict, k: int, open_issues: list, strict: bool = False):
    """Apply `step` to a COPY of `sim` and physics-check it.
    -> (new_sim or None, errors(list[Issue]) or exc-code tuple, warnings, new_open_issues)."""
    new = copy.deepcopy(sim)
    try:
        new.apply(step)
    except Exception as e:  # noqa: BLE001 - any bad step data is a rejected move
        code, hint = _code_for_exc(e)
        return None, (code, str(e), hint), [], open_issues
    if is_view(step):
        return new, [], [], open_issues
    issues = check_step(sim.state, new.state, step, k, strict)
    errors, warns, still = [], [], []
    for iss in issues:
        if iss.kind in STATE_KINDS and any(_same_spot(o, iss) for o in open_issues):
            still.append(iss)  # carried over from before this move: not this move's fault
            continue
        (errors if iss.severity == "error" else warns).append(iss)
    if errors:
        return None, errors, warns, open_issues
    nxt = [i for i in issues if i.kind in STATE_KINDS]
    return new, [], warns, nxt


def run_checked(steps: list[dict], size: float = 1.0):
    """Replay steps from a flat sheet with the per-step checker. -> final state or None if any step fails."""
    sim, open_ = Simulator(size), []
    for k, st in enumerate(steps, 1):
        sim2, err, _, open_ = try_step(sim, st, k, open_)
        if sim2 is None:
            return None
        sim = sim2
    return sim.state


# ---- the environment -----------------------------------------------------------------------

@dataclass
class _Snap:
    sim: Simulator
    trajectory: list
    open_issues: list


class FoldEnv:
    def __init__(self, target=None, feedback: str = "F1", tabu_hard: bool = False, strict: bool = False,
                 max_moves: int | None = None, *, size: float = 1.0, colored_side: str = "up"):
        """target: model JSON (dict) or a path to one; None = free folding on a `size` sheet.
        size / colored_side are only used without a target (a target model sets its own paper)."""
        self.has_target = target is not None
        if isinstance(target, (str, Path)):
            self.target_path = str(target)
            model = json.loads(Path(target).read_text())
        elif target is None:
            self.target_path = None
            model = {"name": "free", "title": "Free folding",
                     "paper": {"size": float(size), "colored_side": colored_side}, "steps": []}
        else:
            self.target_path = None
            model = target
        if feedback not in FEEDBACK_TIERS:
            raise ValueError(f"feedback must be one of {FEEDBACK_TIERS}")
        self.model = model
        self.name = model.get("name", "target")
        self.title = model.get("title", self.name)
        self.size = float(model.get("paper", {}).get("size", 1.0))
        self.colored_side = model.get("paper", {}).get("colored_side", "up")
        self.feedback = feedback
        self.tabu_hard = tabu_hard
        self.strict = strict
        self.ref_steps = list(model["steps"])
        self.par = sum(1 for s in self.ref_steps if not is_view(s)) if self.has_target else None
        self.max_moves = max_moves
        self.labeler = None  # optional agents.command_layer.Labeler: neutral sticker labels in observations
        self.target_state = None
        if self.has_target:
            tgt = run_checked(self.ref_steps, self.size)
            if tgt is None:
                raise ValueError(f"target model {self.name} does not replay cleanly")
            self.target_state = tgt
        self.reset()

    # -- goal -----------------------------------------------------------------------------
    @property
    def goal(self) -> dict | None:
        if not self.has_target:
            return None
        if getattr(self, "hide_target", False):  # text-goal mode: no target name in observations
            return {"name": "the object described in the GOAL text"}
        return {"name": self.name, "title": self.title}

    def goal_text(self) -> str:
        """Computed facts about the target (no step info): size, outline, colours. "" without a target."""
        if not self.has_target:
            return ""
        t = observe(self.target_state, goal=self.goal, colored_side=self.colored_side, images=False)
        keep = [ln for ln in t.text.splitlines()
                if ln.startswith(("Paper:", "Colours:", "Outline", "Hole"))]
        out = [f"Target: {self.title}. Fold a fresh square sheet (side {self.size:g}) until it looks like this "
               f"(outline + visible colours, front and back; turning over / rotating / mirroring is fine).",
               f"A known solution uses {self.par} folds (flips and rotations are free)."]
        if self.model.get("purpose"):
            out.append("What it is for: " + self.model["purpose"])
        for ln in keep:
            ln = re.sub(r", \[points\]|\[[^\]]*\]", "", ln)  # drop point names: they would leak the solution
            ln = re.sub(r", \d+ named points", "", ln)
            out.append("Target " + ln[0].lower() + ln[1:])
        return "\n".join(out)

    def target_png(self) -> bytes:
        if not self.has_target:
            raise ValueError("no target model")
        from .observe import render_folded_png

        return render_folded_png(self.target_state, {}, self.colored_side)

    # -- episode ---------------------------------------------------------------------------
    def reset(self) -> Observation:
        self.sim = Simulator(self.size)
        self.trajectory: list[dict] = []  # committed steps
        self.open_issues: list = []
        self.checkpoints: dict[str, _Snap] = {}
        self._undo_stack: list[_Snap] = []
        self.visited = VisitedTable()
        self.visited.add(self.sim.state, 0)
        self.log: list[dict] = []  # every env call
        self.last_result: StepResult | None = None
        self.stats = {"rejected": 0, "undone": 0, "executed": 0, "previews": 0, "inspects": 0, "view_ops": 0}
        self._failed: set[tuple[str, str]] = set()
        self._no_new = 0
        self.done = False
        self.final: Score | None = None
        return self.observe(images=False)

    @property
    def state(self):
        return self.sim.state

    @property
    def moves(self) -> int:
        return sum(1 for s in self.trajectory if not is_view(s))

    def _snap(self) -> _Snap:
        return _Snap(copy.deepcopy(self.sim), list(self.trajectory), list(self.open_issues))

    def _load(self, s: _Snap) -> None:
        self.sim = copy.deepcopy(s.sim)
        self.trajectory = list(s.trajectory)
        self.open_issues = list(s.open_issues)

    def _note_here(self, note: str) -> None:
        if self.state in self.visited:
            self.visited.note(self.state, note)
        else:
            self.visited.add(self.state, len(self.trajectory), note)

    def _akey(self, action) -> str:
        return json.dumps(action, sort_keys=True, default=str)

    def _progress(self, state) -> dict | None:
        if self.feedback not in ("F2", "F3") or not self.has_target:
            return None
        r = match(self.target_state, state)
        p = {"shape": r.shape_score}
        if self.feedback == "F3":
            p.update(outline_iou=r.iou, colour_front=r.colour_front, colour_back=r.colour_back)
        return p

    def _evaluate(self, action, commit: bool) -> tuple[StepResult, Simulator | None, list]:
        if not isinstance(action, dict):
            r = StepResult(False, {"raw": str(action)}, str(action), "BAD_ARGS", "action must be a JSON object")
            return r, None, self.open_issues
        act = normalize_action(action, self.size)
        line = step_line(act)
        if self.done:
            return StepResult(False, act, line, "EPISODE_OVER", "the episode was submitted"), None, self.open_issues
        if self.max_moves is not None and not is_view(act) and self.moves >= self.max_moves:
            return StepResult(False, act, line, "BUDGET", "move budget used up"), None, self.open_issues
        h = self.visited.key(self.state)
        if self.tabu_hard and (h, self._akey(act)) in self._failed:
            return (StepResult(False, act, line, "TABU_REPEAT", "this exact move was already rejected here",
                               "try something different"), None, self.open_issues)
        k = len(self.trajectory) + 1
        new, err, warns, open_ = try_step(self.sim, act, k, self.open_issues, self.strict)
        wtxt = [w.message for w in warns]
        if new is None:
            if isinstance(err, tuple):
                code, reason, hint = err
                r = StepResult(False, act, line, code, reason, hint, wtxt)
            else:
                first = err[0]
                r = StepResult(False, act, line, first.kind.upper(),
                               " ".join(e.message for e in err[:3]),
                               "the paper can't do this physically; try other layers, another line or another order",
                               wtxt, _sheet_pts(err))
            return r, None, self.open_issues
        r = StepResult(True, act, line, warnings=wtxt, committed=commit)
        r.progress = self._progress(new.state) if not is_view(act) else None
        return r, new, open_

    # -- tools -----------------------------------------------------------------------------
    def apply(self, action) -> StepResult:
        r, new, open_ = self._evaluate(action, commit=True)
        if not r.ok:
            if r.code not in ("EPISODE_OVER", "BUDGET"):
                self.stats["rejected"] += 1
                if self.feedback != "F0":
                    self._note_here(f"tried `{r.line}` -> rejected ({r.code})")
                self._failed.add((self.visited.key(self.state), self._akey(r.action)))
        else:
            self._undo_stack.append(self._snap())
            self.sim, self.open_issues = new, open_
            self.trajectory.append(r.action)
            if is_view(r.action):
                self.stats["view_ops"] += 1
            else:
                self.stats["executed"] += 1
                e = self.visited.get(self.state)
                _, is_new = self.visited.add(self.state, len(self.trajectory))
                if not is_new and e is not None:
                    r.revisited = e.step
                self._no_new = 0 if is_new else self._no_new + 1
        self.last_result = r
        self.log.append({"tool": "apply", **r.to_dict()})
        return r

    def preview(self, action) -> StepResult:
        """Dry run: checker result (+ progress at F2/F3) without committing."""
        self.stats["previews"] += 1
        r, new, _ = self._evaluate(action, commit=False)
        self.log.append({"tool": "preview", **r.to_dict()})
        return r

    def undo(self, n: int = 1) -> StepResult:
        if not self._undo_stack:
            r = StepResult(False, {"op": "undo"}, "undo", "PRECONDITION", "nothing to undo")
            self.last_result = r
            self.log.append({"tool": "undo", **r.to_dict()})
            return r
        undone = []
        for _ in range(max(1, int(n))):
            if not self._undo_stack:
                break
            undone.append(self.trajectory[-1])
            self._load(self._undo_stack.pop())
        for st in undone:
            if not is_view(st):
                self.stats["undone"] += 1
        # note on the state we came back to: what was tried from here
        first = undone[-1]
        if not is_view(first):
            self._note_here(f"tried `{step_line(first)}` -> later undone")
        r = StepResult(True, {"op": "undo", "n": len(undone)}, f"undo {len(undone)}",
                       reason="undid: " + "; ".join(step_line(s) for s in undone))
        self.last_result = r
        self.log.append({"tool": "undo", **r.to_dict()})
        return r

    def checkpoint(self, name: str = "cp") -> StepResult:
        self.checkpoints[name] = self._snap()
        r = StepResult(True, {"op": "checkpoint", "name": name}, f"checkpoint {name}")
        self.log.append({"tool": "checkpoint", **r.to_dict()})
        return r

    def restore(self, name: str = "cp") -> StepResult:
        if name not in self.checkpoints:
            r = StepResult(False, {"op": "restore", "name": name}, f"restore {name}", "BAD_REF",
                           f"no checkpoint {name!r}; have {sorted(self.checkpoints)}")
        else:
            snap = self.checkpoints[name]
            n_cp = len(snap.trajectory)
            dropped = [s for s in self.trajectory[n_cp:] if not is_view(s)] \
                if self.trajectory[:n_cp] == snap.trajectory else [s for s in self.trajectory if not is_view(s)]
            self.stats["undone"] += len(dropped)
            self._undo_stack.append(self._snap())
            self._load(snap)
            r = StepResult(True, {"op": "restore", "name": name}, f"restore {name}",
                           reason=f"back to checkpoint {name} ({len(dropped)} moves dropped)")
        self.last_result = r
        self.log.append({"tool": "restore", **r.to_dict()})
        return r

    def inspect(self, query):
        self.stats["inspects"] += 1
        try:
            if isinstance(query, dict):
                query = {k: normalize_action(v, self.size) for k, v in query.items()}
            out = _inspect(self.state, query, self.colored_side)
            ok = True
        except Exception as e:  # noqa: BLE001
            out, ok = {"error": f"{type(e).__name__}: {e}"}, False
        self.log.append({"tool": "inspect", "query": query, "ok": ok})
        return out

    # -- observation ----------------------------------------------------------------------
    def memory_notes(self) -> list[str]:
        notes = []
        e = self.visited.get(self.state)
        if e is not None and e.notes:
            notes.append(f"You were at this exact paper state before (first at step {e.step}). From here you already tried:")
            seen = []
            for n in e.notes:
                if n not in seen:
                    seen.append(n)
            notes += [f"  - {n}" for n in seen[-8:]]
        if self._no_new >= NO_PROGRESS_K:
            notes.append(f"No new paper state in the last {self._no_new} moves: step back (undo) and rethink the plan.")
        return notes

    def observe(self, images: bool = True) -> Observation:
        errs = None
        lr = self.last_result
        if lr is not None and not lr.ok:
            errs = ["invalid move"] if self.feedback == "F0" else [f"[{lr.code}] {lr.line}: {lr.reason}"]
            if self.labeler is not None:
                errs = [self.labeler.sub(e) for e in errs]
        obs = observe(self.state, history=self.trajectory, goal=self.goal, colored_side=self.colored_side,
                      last_errors=errs, step=self.moves, budget=self.max_moves, images=images,
                      labeler=self.labeler)
        mem = self.memory_notes()
        if mem:
            txt = "\nMemory:\n" + "\n".join(mem)
            obs.text += self.labeler.sub(txt) if self.labeler is not None else txt
        return obs

    # -- scoring ---------------------------------------------------------------------------
    def score(self, compute_deletable: bool = True) -> Score:
        if not self.has_target:
            return Score(None, None, None, None, None, {}, None, None, None, None, self.moves,
                         sum(1 for s in self.trajectory if is_view(s)), None, None, 0.0, None,
                         self.stats["undone"], self.stats["rejected"], self.stats["executed"],
                         self.stats["previews"], self.stats["inspects"], list(self.trajectory))
        r = match(self.target_state, self.state)
        moves = self.moves
        success = r.m2
        deletable = None
        if success and compute_deletable:
            deletable = self._deletable()
        return Score(self.name, bool(r.m1), bool(r.m2), bool(r.m3), bool(r.m4),
                     {k: float(v) for k, v in r.scores.items()}, float(r.shape_score), float(r.iou),
                     float(r.colour_front), float(r.colour_back), moves, sum(1 for s in self.trajectory if is_view(s)), self.par,
                     (moves / self.par) if success and self.par else None,
                     (self.par / max(self.par, moves)) if success and self.par else 0.0,
                     deletable, self.stats["undone"], self.stats["rejected"], self.stats["executed"],
                     self.stats["previews"], self.stats["inspects"], list(self.trajectory))

    def _deletable(self) -> int:
        """Greedy: drop each non-view move in turn; keep the drop if the rest still replays legally
        and still reaches M2."""
        steps = list(self.trajectory)
        removed = 0
        # prefix[j] = (sim, open issues) after the first j kept steps (all legal by construction)
        prefix = [(Simulator(self.size), [])]
        i = 0
        while i < len(steps):
            sim, open_ = prefix[i]
            if not is_view(steps[i]):
                ok = True
                for k, st in enumerate(steps[i + 1:], i + 1):  # k = 1-based index in the trial
                    sim, err, _, open_ = try_step(sim, st, k, open_)
                    if sim is None:
                        ok = False
                        break
                if ok and match(self.target_state, sim.state).m2:
                    steps = steps[:i] + steps[i + 1:]
                    removed += 1
                    continue  # prefix[i] still valid; try the new steps[i]
            sim, err, _, open_ = try_step(*prefix[i][:1], steps[i], i + 1, prefix[i][1])
            prefix.append((sim, open_))
            i += 1
        return removed

    def submit(self, compute_deletable: bool = True) -> Score:
        self.final = self.score(compute_deletable)
        self.done = True
        self.log.append({"tool": "submit", "score": self.final.to_dict()})
        return self.final


def replay_model(path, env_kwargs=None, compute_deletable: bool = True) -> tuple[FoldEnv, list[StepResult], Score]:
    """Oracle sanity check: feed a model's own steps through a fresh env aimed at itself."""
    env = FoldEnv(path, **(env_kwargs or {}))
    res = [env.apply({k: v for k, v in s.items() if k != "description"}) for s in env.ref_steps]
    return env, res, env.submit(compute_deletable)


def summarize_score(s: Score) -> str:
    if s.target is None:
        return f"(no target): moves={s.moves} rejected={s.rejected} undone={s.undone}"
    return (f"{s.target}: M1={'Y' if s.m1 else 'n'}({s.scores['M1']:.2f}) M2={'Y' if s.m2 else 'n'}"
            f"({s.scores['M2']:.2f}) M3={'Y' if s.m3 else 'n'} M4={'Y' if s.m4 else 'n'} "
            f"moves={s.moves}/par {s.par} rejected={s.rejected} undone={s.undone} deletable={s.deletable}")


__all__ = ["FoldEnv", "StepResult", "Score", "replay_model", "run_checked", "normalize_action", "is_view",
           "summarize_score"]
