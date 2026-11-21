"""What the agent is asked to fold: an example model, a sentence of text, or a picture.

    Goal.example("models/cup.json")               # target picture + computed outline/colours, scored
    Goal.from_text("a paper cup", check_against="models/cup.json")   # only the sentence; scored if given
    Goal.from_image(png_bytes, text="optional hint")                 # only the picture; scored if given

Text and image goals hide everything derived from a reference model (name, par, outline, picture,
closeness feedback); the reference is used for scoring only.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ..env import FoldEnv

IMAGE_GOAL_TEXT = ("GOAL\nThe picture shows the object to fold from one square sheet. It may be a photo or a "
                   "drawing; match its shape and visible colours.")


class GoalEnv(FoldEnv):
    """FoldEnv whose observations name the goal generically when the goal hides its reference."""

    goal_label = "the object in the GOAL"

    @property
    def goal(self):
        if getattr(self, "hide_target", False):
            return {"name": self.goal_label}
        return super().goal


def _load_model(m) -> tuple[dict, str | None]:
    if isinstance(m, dict):
        return m, None
    p = Path(m)
    return json.loads(p.read_text()), str(p)


@dataclass
class Goal:
    kind: Literal["example", "text", "image"]
    target: dict | None = None        # reference model JSON: required for "example", optional for text/image
    text: str | None = None
    image_png: bytes | None = None
    target_path: str | None = field(default=None, repr=False)  # lets the scorer find the model's alternatives

    def __post_init__(self):
        if self.kind not in ("example", "text", "image"):
            raise ValueError(f"unknown goal kind {self.kind!r}")
        if self.kind == "example" and self.target is None:
            raise ValueError("an example goal needs a target model")
        if self.kind == "text" and not (self.text or "").strip():
            raise ValueError("a text goal needs some text")
        if self.kind == "image" and not self.image_png:
            raise ValueError("an image goal needs a picture")
        if self.kind == "image" and not self.image_png.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("the goal picture must be a PNG")

    # -- builders ----------------------------------------------------------------------------
    @classmethod
    def example(cls, model) -> "Goal":
        m, p = _load_model(model)
        return cls("example", target=m, target_path=p)

    @classmethod
    def from_text(cls, text: str, check_against=None) -> "Goal":
        m, p = _load_model(check_against) if check_against is not None else (None, None)
        return cls("text", target=m, text=text.strip(), target_path=p)

    @classmethod
    def from_image(cls, png: bytes, text: str | None = None, check_against=None) -> "Goal":
        m, p = _load_model(check_against) if check_against is not None else (None, None)
        return cls("image", target=m, text=(text or "").strip() or None, image_png=png, target_path=p)

    @classmethod
    def from_request(cls, d: dict, models_dir) -> "Goal":
        """The start request's goal object. Example names resolve to models_dir/<name>.json."""

        def model_path(name):
            if name is None:
                return None
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
                raise ValueError(f"bad example name {name!r}")
            p = Path(models_dir) / f"{name}.json"
            if not p.is_file():
                raise ValueError(f"unknown example {name!r}")
            return p

        kind = d.get("kind")
        if kind == "example":
            return cls.example(model_path(d.get("example")))
        if kind == "text":
            return cls.from_text(d.get("text") or "", model_path(d.get("check_against")))
        if kind == "image":
            b64 = d.get("image_b64") or ""
            if b64.startswith("data:"):
                b64 = b64.split(",", 1)[-1]
            try:
                png = base64.b64decode(b64, validate=True)
            except ValueError:
                raise ValueError("image_b64 is not valid base64") from None
            return cls.from_image(png, d.get("text"), model_path(d.get("check_against")))
        raise ValueError(f"unknown goal kind {kind!r}")

    # -- what the agent sees -----------------------------------------------------------------
    def hides_target(self) -> bool:
        return self.kind != "example"

    @property
    def scored(self) -> bool:
        return self.target is not None

    @property
    def target_name(self) -> str | None:
        return self.target.get("name") if self.target is not None else None

    def make_env(self, feedback: str = "F1") -> FoldEnv:
        if self.hides_target() and feedback not in ("F0", "F1"):
            raise ValueError("text/image goal: feedback F2/F3 would leak a closeness score; use F0 or F1")
        src = self.target_path or self.target
        env = GoalEnv(src, feedback=feedback)
        env.hide_target = self.hides_target()
        return env

    def parts(self, env: FoldEnv, images: bool) -> list[dict]:
        """LLM content parts (OpenAI format) of the goal. An image goal's picture is always included."""
        from .llm_agent import _img_part, goal_parts
        from .menu_agent import text_goal_parts

        if self.kind == "example":
            return goal_parts(env, images, env.target_png() if images else None)
        if self.kind == "text":
            return text_goal_parts(self.text)
        txt = IMAGE_GOAL_TEXT + (f"\nHint from the person: {self.text}" if self.text else "")
        return [{"type": "text", "text": txt}, _img_part(self.image_png)]

    def default_max_turns(self, env: FoldEnv) -> int:
        from .menu_agent import TEXT_GOAL_MAX_TURNS  # fixed, so "TURN k/N" can't leak the reference's par

        return TEXT_GOAL_MAX_TURNS if self.hides_target() else 3 * env.par

    def system_prompt(self, actions: str) -> str:
        """System prompt for 'command' or 'menu' mode, reworded for text / image goals."""
        from .llm_agent import SYSTEM_PROMPT
        from .menu_agent import MENU_SYSTEM_PROMPT, MENU_SYSTEM_PROMPT_TEXT

        if actions == "menu":
            if self.kind == "example":
                return MENU_SYSTEM_PROMPT
            if self.kind == "text":
                return MENU_SYSTEM_PROMPT_TEXT
            return _swap(MENU_SYSTEM_PROMPT, _MENU_IMAGE_SWAPS)
        if actions == "command":
            if self.kind == "example":
                return SYSTEM_PROMPT
            return _swap(SYSTEM_PROMPT, _COMMAND_SWAPS[self.kind])
        raise ValueError(f"no system prompt for actions={actions!r}")

    def describe(self) -> dict:
        """JSON-safe summary for events (no picture bytes)."""
        return {"kind": self.kind, "text": self.text, "target_name": None if self.hides_target() else self.target_name,
                "check_against": self.target_name if self.hides_target() else None, "scored": self.scored}


def _swap(prompt: str, swaps) -> str:
    for a, b in swaps:
        if a not in prompt:
            raise AssertionError(f"prompt text to replace not found: {a[:60]!r}")
        prompt = prompt.replace(a, b)
    return prompt


_GOAL_IS = {
    "text": ("step by step until it becomes the object described in the GOAL text. There is no picture of the "
             "result: you decide what it should look like."),
    "image": ("step by step until it looks like the object in the GOAL picture (a photo or a drawing: match its "
              "shape and visible colours)."),
}
_COMMAND_SWAPS = {
    kind: [
        ("step by step until it looks like a target model.", _GOAL_IS[kind]),
        ("- Only the final state is judged, when you call submit: the outline AND the visible colours seen from the "
         "front AND from the back must match the target. Any rotation, turning over, mirroring or position is fine; "
         "the size must match (same sheet).",
         "- Only the final state is judged, when you call submit: is it the object in the GOAL? Any rotation, "
         "turning over, mirroring or position is fine."),
        ("think carefully about the target. Recall how this model is folded",
         "think carefully about the goal. Recall how such a model is folded"),
    ]
    for kind in ("text", "image")
}
_MENU_IMAGE_SWAPS = [
    ("step by step until it looks like a target model.", _GOAL_IS["image"]),
    ("RULES: only the final paper is judged (outline + visible colours, front AND back; rotation / turning over / "
     "mirroring are fine). Use few moves (par = best known move count).",
     "RULES: only the final paper is judged: is it the object in the GOAL picture (rotation / turning over / "
     "mirroring are fine)? Use few moves."),
    ("the whole sequence of folds from the flat sheet to the target",
     "the whole sequence of folds from the flat sheet to the finished object"),
]
