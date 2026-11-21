"""Agent environment (env.py): replay, rejection, undo, checkpoints, tabu memory, scoring."""

import json

import pytest

from origami.env import FoldEnv, normalize_action, replay_model

MODELS = ["triangle", "cup", "hat", "fortune_teller", "bird_base", "crane"]


@pytest.mark.parametrize("name", MODELS)
def test_replay_own_steps_reaches_m2(name):
    env, res, score = replay_model(f"models/{name}.json", compute_deletable=(name == "cup"))
    assert all(r.ok for r in res), [r.to_text() for r in res if not r.ok]
    assert score.m2 and score.m1 and score.passed
    assert score.moves == score.par and score.spl == 1.0
    assert score.rejected == 0 and score.undone == 0
    if name == "cup":
        assert score.deletable == 0


def _cup():
    return FoldEnv("models/cup.json")


def test_par_counts_only_moves():
    env = _cup()
    assert env.par == 5  # 7 steps minus rotate and flip


def test_illegal_move_rejected_state_unchanged():
    env = _cup()
    env.apply({"op": "valley", "line": {"onto": ["tr", "bl"]}})
    before = env.state.copy()
    traj = list(env.trajectory)
    r = env.apply({"op": "valley", "line": {"onto": ["tl", "br"]}, "move": "center"})
    assert not r.ok and r.code == "MOVE_ON_LINE"
    r = env.apply({"op": "valley", "line": {"onto": ["tr", "bl"]}})
    assert not r.ok and r.code == "SAME_POINT"  # tr now lies on bl
    r = env.apply({"op": "valley", "line": {"onto": ["nowhere", "bl"]}})
    assert not r.ok and r.code == "BAD_REF"
    r = env.apply({"op": "sink", "line": {"onto": ["tr", "bl"]}})
    assert not r.ok and r.code == "NOT_IMPLEMENTED"
    assert env.trajectory == traj
    assert len(env.state.faces) == len(before.faces)
    assert env.stats["rejected"] == 4
    assert "REJECTED" in env.observe(images=False).text


def test_physics_rejection_has_checker_code():
    # second fold tucked under the first fold's own layer: layers interleave on one crease line
    env = FoldEnv("models/triangle.json")
    env.apply({"op": "valley", "line": {"onto": ["tr", "bl"]}})
    r = env.apply({"op": "valley", "line": {"onto": ["tl", "br"]}, "move": "tl", "tuck": {"under": 1}})
    assert not r.ok and r.code == "TACO_TACO" and r.affected
    assert env.moves == 1


def test_undo_and_tabu_note():
    env = _cup()
    env.apply({"op": "valley", "line": {"onto": ["tr", "bl"]}})
    n_faces = len(env.state.faces)
    env.apply({"op": "valley", "line": {"onto": ["tl", "br"]}})
    assert len(env.state.faces) > n_faces
    env.undo()
    assert len(env.state.faces) == n_faces and len(env.trajectory) == 1
    assert env.stats["undone"] == 1
    mem = "\n".join(env.memory_notes())
    assert "later undone" in mem
    env.undo()
    assert env.trajectory == [] and len(env.state.faces) == 1
    assert not env.undo().ok  # nothing left


def test_rejected_move_noted_and_hard_tabu():
    env = FoldEnv("models/cup.json", tabu_hard=True)
    bad = {"op": "valley", "line": {"onto": ["tr", "bl"]}, "move": "center"}
    assert env.apply(bad).code == "MOVE_ON_LINE"
    assert "rejected (MOVE_ON_LINE)" in "\n".join(env.memory_notes())
    assert env.apply(bad).code == "TABU_REPEAT"


def test_revisit_detected_view_ops_free():
    env = _cup()
    env.apply({"op": "valley", "line": {"onto": ["tr", "bl"]}})
    env.apply({"op": "unfold"})
    r = env.apply({"op": "valley", "line": {"onto": ["tr", "bl"]}})
    assert r.ok and r.revisited == 1
    env.apply({"op": "flip"})
    env.apply({"op": "rotate", "degrees": 30})
    assert env.moves == 3 and env.stats["view_ops"] == 2


def test_checkpoint_restore_preview():
    env = _cup()
    env.apply({"op": "rotate", "degrees": 45})
    env.checkpoint("start")
    env.apply({"op": "valley", "line": {"through": ["tl", "br"]}, "move": "bl"})
    p = env.preview({"op": "valley", "line": {"onto": ["tl", "br"]}})
    assert p.ok and not p.committed and env.moves == 1
    env.restore("start")
    assert env.moves == 0 and env.stats["undone"] == 1
    assert not env.restore("nope").ok


def test_point_names_and_json_strings_accepted():
    a = normalize_action({"op": "valley", "line": '{"onto": ["p250_500", "tr"]}', "move": "p250_500"})
    assert a["line"] == {"onto": [{"orig": [0.25, 0.5]}, "tr"]}
    assert a["move"] == {"orig": [0.25, 0.5]}
    assert normalize_action({"op": "valley", "move": '"bl"'})["move"] == "bl"
    env = _cup()
    assert env.apply({"op": "valley", "line": {"vertical": "p250_500"}, "move": "bl"}).ok


def test_score_failure_and_deletable():
    env = _cup()
    env.apply({"op": "valley", "line": {"onto": ["tr", "bl"]}})
    s = env.submit()
    assert not s.m2 and s.spl == 0.0 and s.deletable is None and s.length_ratio is None
    assert type(s.m2) is bool and json.loads(json.dumps(s.to_dict()))["m2"] is False
    assert not env.apply({"op": "flip"}).ok  # episode over
    # padded solution: a fold + unfold that change nothing
    env = _cup()
    env.apply({"op": "valley", "line": {"onto": ["tr", "bl"]}})
    env.apply({"op": "unfold"})
    for st in env.ref_steps:
        assert env.apply({k: v for k, v in st.items() if k != "description"}).ok
    s = env.submit()
    assert s.m2 and s.moves == 7 and s.deletable == 2 and s.spl == pytest.approx(5 / 7)


def test_feedback_tiers():
    env = FoldEnv("models/cup.json", feedback="F2")
    r = env.apply({"op": "valley", "line": {"onto": ["tr", "bl"]}})
    assert r.progress and 0 <= r.progress["shape"] <= 1
    env = FoldEnv("models/cup.json", feedback="F0")
    r = env.apply({"op": "valley", "line": {"onto": ["tr", "bl"]}, "move": "center"})
    assert r.to_text("F0").endswith("(invalid move)")
    assert "MOVE_ON_LINE" not in env.observe(images=False).text


def test_goal_text_leaks_no_point_names():
    t = _cup().goal_text()
    assert "known solution" in t and "outline" in t
    for nm in ("bl", "tr", "center", "p"):
        assert f"[{nm}" not in t
