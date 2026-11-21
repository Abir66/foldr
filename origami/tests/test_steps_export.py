import json
from pathlib import Path

import numpy as np

from origami.cli import main, run_model
from origami.steps_export import build_timeline

MODELS = Path(__file__).parent.parent / "models"


def _load(name):
    return json.loads((MODELS / f"{name}.json").read_text())


def _apply(T, pts):
    T = np.array(T)
    return np.array(pts) @ T[:, :2].T + T[:, 2]


def test_timeline_shape():
    model = _load("cup")
    tl = build_timeline(model)
    n = len(model["steps"])
    assert len(tl["states"]) == n + 1
    assert len(tl["steps"]) == n
    assert tl["original_faces"][0]["orig"][0] == [0.0, 0.0]
    for k, s in enumerate(tl["steps"]):
        assert s["index"] == k + 1
        assert s["op"] == model["steps"][k]["op"]
        assert s["description"] == model["steps"][k]["description"]
        assert s["pieces"]
    json.dumps(tl)  # serializable


def test_triangle_valley_moving_faces():
    tl = build_timeline(_load("triangle"))
    s1, s2 = tl["steps"]
    # Step 1: the one sheet splits; the new top face (index 1) is the moving one.
    assert s1["motion"]["type"] == "reflect"
    assert s1["sign"] == 1
    assert s1["moving_before"] == [0]
    assert s1["moving_after"] == [1]
    assert sum(p["moving"] for p in s1["pieces"]) == 1
    # fold axis is the tl-br diagonal (direction ±(1,-1)/√2)
    u = np.array(s1["motion"]["u"])
    assert abs(abs(u @ np.array([1, -1]) / np.sqrt(2)) - 1) < 1e-9
    # Step 2: both layers move, ending as the top two faces.
    assert s2["moving_before"] == [0, 1]
    assert s2["moving_after"] == [2, 3]


def test_after_state_matches_engine():
    model = _load("cup")
    states = run_model(model)
    tl = build_timeline(model)
    final = states[-1]
    assert len(tl["states"][-1]["faces"]) == len(final.faces)
    for fj, f in zip(tl["states"][-1]["faces"], final.faces):
        assert np.allclose(fj["T"], f.T[:2])
        assert fj["front_up"] == f.front_up
    # every step's pieces, mapped by Ta, sit inside the engine's after-state faces
    for k, s in enumerate(tl["steps"]):
        after = states[k + 1]
        area = 0.0
        for p in s["pieces"]:
            fa = after.faces[p["layer_after"]]
            assert np.allclose(p["Ta"], fa.T[:2])
            pts = np.array(p["orig"])
            x, y = pts[:, 0], pts[:, 1]
            area += 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        assert abs(area - 1.0) < 1e-6  # pieces tile the sheet


def test_reflection_motion_maps_before_to_after():
    tl = build_timeline(_load("cup"))
    for s in tl["steps"]:
        m = s["motion"]
        for p in s["pieces"]:
            if not p["moving"]:
                assert np.allclose(p["Tb"], p["Ta"])
                continue
            q = _apply(p["Tb"], p["orig"])
            if m["type"] == "reflect":
                a, u = np.array(m["a"]), np.array(m["u"])
                rel = q - a
                q2 = a + 2 * np.outer(rel @ u, u) - rel
            else:
                th = np.radians(m["degrees"])
                R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
                q2 = (q - m["center"]) @ R.T + m["center"]
            assert np.allclose(q2, _apply(p["Ta"], p["orig"]), atol=1e-8)


def test_mountain_and_unfold_swing_direction():
    model = {"steps": [
        {"op": "valley", "line": {"through": ["left_mid", "right_mid"]}, "move": "bl"},
        {"op": "unfold"},
        {"op": "mountain", "line": {"vertical": "center"}, "move": "bl"},
        {"op": "unfold"},
    ]}
    signs = [s["sign"] for s in build_timeline(model)["steps"]]
    assert signs == [1, 1, -1, -1]


def _world_pt(W, T, p, layer):
    q = _apply(T, [p])[0]
    M = np.array(W["M"])
    return np.array([*(M[:, :2] @ q + M[:, 2]), W["s"] * layer + W["zoff"]])


def test_flip_is_camera_move_world_continuous():
    """A flip must not move the paper in world space; camera side flips instead."""
    for name in ("cup", "fortune_teller"):
        tl = build_timeline(_load(name))
        for k, s in enumerate(tl["steps"]):
            b, a = tl["states"][k], tl["states"][k + 1]
            assert s["side_before"] == b["world"]["s"] and s["side_after"] == a["world"]["s"]
            if s["op"] != "flip":
                assert a["world"] == b["world"]
                continue
            assert s["side_after"] == -s["side_before"]
            n = len(b["faces"])
            assert len(a["faces"]) == n
            for fb in b["faces"]:
                fa = a["faces"][n - 1 - fb["id"]]  # engine reverses layers
                for p in fb["orig"]:
                    assert np.allclose(_world_pt(b["world"], fb["T"], p, fb["layer"]),
                                       _world_pt(a["world"], fa["T"], p, fa["layer"]), atol=1e-8)


def test_step_arrows():
    tl = build_timeline(_load("cup"))
    for s in tl["steps"]:
        if s["op"] == "valley":
            assert len(s["arrow"]["from"]) == 2 and len(s["arrow"]["to"]) == 2
        else:
            assert s["arrow"] is None
    assert tl["steps"][0]["degrees"] == 45
    assert tl["steps"][5]["flip_axis"] == "vertical"


def test_bends_exported():
    for name in ("cup", "hat", "fortune_teller"):
        tl = build_timeline(_load(name))
        for st in tl["states"]:
            nfaces = len(st["faces"])
            for b in st["bends"]:
                assert 0 <= b["a"] < b["b"] < nfaces
                (x0, y0), (x1, y1) = b["seg"]
                assert (x1 - x0) ** 2 + (y1 - y0) ** 2 > 1e-10
                fa, fb = st["faces"][b["a"]], st["faces"][b["b"]]
                # both faces map the shared crease segment to the same folded segment
                for p in b["seg"]:
                    assert np.allclose(_apply(fa["T"], [p]), _apply(fb["T"], [p]), atol=1e-6)
    assert tl["states"][0]["bends"] == []  # flat sheet: nothing is folded


def test_cup_final_has_bends_and_flat_has_none():
    tl = build_timeline(_load("triangle"))
    assert tl["states"][0]["bends"] == []
    assert len(tl["states"][1]["bends"]) >= 1  # after the first valley fold the flap is bent over


def _refl(a, u):
    a, u = np.asarray(a, float), np.asarray(u, float)
    L = 2 * np.outer(u, u) - np.eye(2)
    M = np.eye(3)
    M[:2, :2] = L
    M[:2, 2] = a - L @ a
    return M


def _T33(T):
    return np.vstack([np.array(T, float), [0, 0, 1]])


def test_multi_motion_steps_exported():
    """Squash / petal steps export ordered sub-motions grouped in phases; each piece's chain of
    reflections maps its before transform exactly onto its after transform."""
    model = _load("bird_base")
    tl = build_timeline(model)
    ops = {s["op"] for s in tl["steps"] if "motions" in s}
    assert ops == {"squash", "petal"}
    for s in tl["steps"]:
        if s["op"] in ("valley", "mountain", "flip", "rotate", "unfold"):
            assert "motions" not in s and "chain" not in s["pieces"][0]  # single-motion format unchanged
            continue
        assert s["phases"] == 2
        assert sorted({m["phase"] for m in s["motions"]}) == [0, 1]
        assert [m["phase"] for m in s["motions"]] == sorted(m["phase"] for m in s["motions"])
        assert s["motion"] is None and s["arrow"] is None
        assert any(ax["phase"] == 0 for ax in s["axes"]) and s["axis"] == s["axes"][0]["seg"]
        assert any(p["moving"] for p in s["pieces"]) and any(not p["moving"] for p in s["pieces"])
        area = 0.0
        for p in s["pieces"]:
            T = _T33(p["Tb"])
            for c in p["chain"]:
                m = s["motions"][c["m"]]
                assert c["dir"] in (1, -1) and m["sign"] in (1, -1)
                T = _refl(m["a"], m["u"]) @ T
            assert np.allclose(T[:2], p["Ta"], atol=1e-8)
            assert p["moving"] == bool(p["chain"])
            pts = np.array(p["orig"])
            x, y = pts[:, 0], pts[:, 1]
            area += 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        assert abs(area - 1.0) < 1e-6  # parts still tile the sheet


def test_petal_phases_and_signs():
    tl = build_timeline(_load("bird_base"))
    petal = next(s for s in tl["steps"] if s["op"] == "petal")
    # phase 0: lift (valley, first) + 2nd-layer sides turn in along the kites on top of their
    # own layer (valley); phase 1: top-layer sides fold over the lifted flap (valley). All plain.
    ph0 = [m for m in petal["motions"] if m["phase"] == 0]
    assert len(ph0) == 3 and all(m["sign"] == 1 and not m["tuck"] for m in ph0)
    ph1 = [m for m in petal["motions"] if m["phase"] == 1]
    assert len(ph1) == 2 and all(m["sign"] == 1 and not m["tuck"] for m in ph1)
    # some pieces move in both phases (lifted, then tucked)
    assert any({petal["motions"][c["m"]]["phase"] for c in p["chain"]} == {0, 1} for p in petal["pieces"])


def test_steps_carry_timing_and_creases():
    tl = build_timeline(_load("bird_base"))
    for s in tl["steps"]:
        if s["op"] == "flip":
            continue
        assert s["path_len"] > 0
        assert all("lines" in p for p in s["pieces"])
        if s["op"] in ("squash", "petal"):
            assert s["retime"][0] == 0.0 and s["retime"][-1] == 1.0
        if s["op"] in ("valley", "rotate"):
            assert "retime" not in s  # simple folds already move evenly
    # creases made by earlier steps show on the moving pieces
    assert any(p["lines"] for p in tl["steps"][2]["pieces"])
