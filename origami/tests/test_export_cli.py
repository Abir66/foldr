import json
from pathlib import Path

from origami.cli import main, run_model
from origami.fold_export import to_fold
from origami.validate import folded_creases

MODELS = Path(__file__).parent.parent / "models"


def test_fold_export_half_fold():
    model = {"steps": [{"op": "valley", "line": {"through": ["left_mid", "right_mid"]}, "move": "bl"}]}
    state = run_model(model)[-1]
    d = to_fold(state)
    assert d["edges_assignment"].count("V") == 1
    assert d["edges_assignment"].count("B") == 6  # 4 sides, two split by the crease
    assert d["edges_assignment"].count("F") == 0
    frame = d["file_frames"][0]
    assert len(frame["faces_vertices"]) == 2
    # FOLD spec: [f, g, +1] = f lies on the side g's normal points to. Face 1 is on top of
    # face 0, and face 0 is front-up (normal toward the viewer), so +1.
    assert frame["faceOrders"] == [[1, 0, 1]]


def _on_seg(m, a, b, tol=1e-6):
    ax, ay, bx, by = a[0], a[1], b[0], b[1]
    cr = (bx - ax) * (m[1] - ay) - (by - ay) * (m[0] - ax)
    t = ((m[0] - ax) * (bx - ax) + (m[1] - ay) * (by - ay)) / ((bx - ax) ** 2 + (by - ay) ** 2)
    return abs(cr) < tol and -tol <= t <= 1 + tol


def test_fold_export_describes_current_state_not_history():
    state = run_model(json.loads((MODELS / "bird_base.json").read_text()))[-1]
    d = to_fold(state)
    assert d["origami:creases_history"]  # history kept as an extra field
    folded = [(p, q) for p, q, a in folded_creases(state) if a in ("M", "V")]
    v = d["vertices_coords"]
    mv = [(v[a], v[b]) for (a, b), s in zip(d["edges_vertices"], d["edges_assignment"]) if s in ("M", "V")]
    assert mv
    # every exported M/V edge lies on a currently folded crease (no stale history)
    for p, q in mv:
        m = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
        assert any(_on_seg(m, a, b) for a, b in folded)
    assert "F" in d["edges_assignment"]
    assert len(d["faces_vertices"]) == len(state.faces)
    # some historical creases are flat now: they must not be exported as M/V
    stale = [
        h for h in d["origami:creases_history"]
        if not any(_on_seg(((p[0] + q[0]) / 2, (p[1] + q[1]) / 2), h["p"], h["q"]) for p, q in mv)
    ]
    assert stale


def test_cli_writes_pngs_and_fold(tmp_path):
    assert main(["fold", str(MODELS / "cup.json"), "--out", str(tmp_path)]) == 0
    model = json.loads((MODELS / "cup.json").read_text())
    pngs = sorted(tmp_path.glob("step_*.png"))
    assert len(pngs) == len(model["steps"]) + 1
    assert (tmp_path / "crease_pattern.png").exists()
    json.loads((tmp_path / "cup.fold").read_text())
