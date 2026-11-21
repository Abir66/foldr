"""Tuck-inside animation (tuck_anim.py): the flap slides in under the opened cover, never in front of it."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from origami import geometry as g
from origami.kinematics import frames
from origami.steps_export import build_timeline

MODELS = Path(__file__).resolve().parent.parent / "models"
DATA = Path(__file__).resolve().parent / "data" / "models"


@lru_cache(maxsize=None)
def _tl(name):
    path = DATA / f"{name}.json" if name == "hat_tuck" else MODELS / f"{name}.json"
    return build_timeline(json.loads(path.read_text()))


def _tuck_step(name):
    return next(s for s in _tl(name)["steps"] if s.get("tuck_anim"))


def test_hat_tuck_uses_scripted_slide_in():
    s = _tuck_step("hat_tuck")
    assert s["index"] == 10 and s["tuck_anim"]["cover"]
    assert all(p.get("fkeys") for p in s["pieces"])


def test_crane_narrowing_tucks_keep_plain_path():
    # nothing ends over the narrowed flap: no cover, no scripted motion
    assert not any(s.get("tuck_anim") for s in _tl("crane")["steps"])


def test_hat_tuck_flap_stays_hinged_and_under_the_cover():
    s = _tuck_step("hat_tuck")
    P = s["pieces"]
    cover = set(s["tuck_anim"]["cover"])
    mov = [i for i, p in enumerate(P) if p["moving"]]
    origs = [g.ccw(np.array(p["orig"], float)) for p in P]
    hinge = [(i, j, g.shared_edge(origs[i], origs[j])) for i in mov for j in range(len(P))
             if j not in mov and g.shared_edge(origs[i], origs[j]) is not None]
    assert hinge
    for t in np.linspace(0.05, 0.95, 19):
        G = frames(P, s["kin_order"], float(t), retime=s.get("retime"))
        for i, j, seg in hinge:  # flap stays on its hinge
            for v in seg:
                a = G[i] @ np.array([v[0], v[1], 0.0, 1.0])
                b = G[j] @ np.array([v[0], v[1], 0.0, 1.0])
                assert np.linalg.norm(a - b) < 1e-6, (t, i, j)
        for c in cover:  # every flap vertex under a cover piece is below that piece's plane
            Gc = G[c]
            inv = np.linalg.inv(Gc)
            for m in mov:
                for v in origs[m]:
                    x = G[m] @ np.array([v[0], v[1], 0.0, 1.0])
                    # cover-plane point straight "above" x (along world z): solve in the cover's frame
                    n = Gc[:3, 2] / np.linalg.norm(Gc[:3, 2])
                    if abs(n[2]) < 1e-6:
                        continue
                    o = Gc[:3, 3]
                    zc = x[2] - np.dot(x[:3] - o, n) / n[2]       # cover plane height at x's (x, y)
                    loc = inv @ np.array([x[0], x[1], zc, 1.0])
                    if not g.contains(origs[c], loc[:2], -1e-6):
                        continue
                    assert x[2] <= zc + 1e-9, (t, m, c, x[2], zc)
