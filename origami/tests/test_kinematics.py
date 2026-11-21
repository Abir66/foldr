"""Connected fold animation (kinematics.py): the paper stays joined while it moves."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from origami import geometry as g
from origami.kinematics import frames, path_profile, retime_at, retime_table
from origami.steps_export import build_timeline

MODELS = Path(__file__).parent.parent / "models"
NAMES = ["triangle", "cup", "hat", "fortune_teller", "bird_base", "crane"]
DZ = 0.004


@lru_cache(maxsize=None)
def _tl(name):
    return build_timeline(json.loads((MODELS / f"{name}.json").read_text()))


def _kin_steps(name):
    return [s for s in _tl(name)["steps"] if s["op"] != "flip"]


def _pos(G, p, z=0.0):
    return (G @ np.array([p[0], p[1], z, 1.0]))[:3]


def _apply(T, p):
    T = np.asarray(T, float)
    return T[:, :2] @ np.asarray(p, float) + T[:, 2]


@pytest.mark.parametrize("name", NAMES)
def test_ends_reproduce_before_and_after(name):
    """At t = 0 / 1 forward kinematics gives exactly the before / after state (incl. layer heights)."""
    for s in _kin_steps(name):
        assert s["kin_order"] and sorted(s["kin_order"]) == list(range(len(s["pieces"])))
        for t, T, zk in ((0.0, "Tb", "z_before"), (1.0, "Ta", "z_after")):
            G = frames(s["pieces"], s["kin_order"], t, dz=DZ, retime=s.get("retime"))
            for Gi, p in zip(G, s["pieces"]):
                for v in p["orig"]:
                    x = _pos(Gi, v)
                    assert np.allclose(x[:2], _apply(p[T], v), atol=1e-9), (name, s["index"], t)
                    assert abs(x[2] - DZ * p[zk]) < 1e-9


def _adjacent_pairs(s):
    polys = [g.ccw(np.array(p["orig"])) for p in s["pieces"]]
    out = []
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            seg = g.shared_edge(polys[i], polys[j])
            if seg is not None:
                out.append((i, j, seg))
    return out


@pytest.mark.parametrize("name", NAMES)
def test_paper_stays_connected_mid_motion(name):
    """Pieces that share an edge of the sheet keep meeting along it during the fold. Tree hinges meet
    exactly; loops close exactly for simple folds and up to a small solver gap for macros."""
    for s in _kin_steps(name):
        if s.get("spread") or s.get("tuck_anim"):
            continue  # scripted inside reverse / tuck: some joints open on purpose (own tests)
        pairs = _adjacent_pairs(s)
        tree = {(min(i, p["parent"]), max(i, p["parent"])) for i, p in enumerate(s["pieces"]) if p["parent"] >= 0}
        assert s["kin_gap"] < 5e-3, (name, s["index"], s["kin_gap"])
        for t in (0.25, 0.5, 0.75):
            G = frames(s["pieces"], s["kin_order"], t, retime=s.get("retime"))  # dz = 0: pure geometry
            for i, j, seg in pairs:
                d = max(np.linalg.norm(_pos(G[i], v) - _pos(G[j], v)) for v in seg)
                tol = 1e-9 if (i, j) in tree else 1e-6 if s["kin_gap"] == 0 else 5e-3
                assert d < tol, (name, s["index"], t, i, j, d)


def test_squash_actually_moves_out_of_plane_and_stays_joined():
    s = next(s for s in _tl("bird_base")["steps"] if s["op"] == "squash")
    G = frames(s["pieces"], s["kin_order"], 0.5, retime=s.get("retime"))
    z = [abs(_pos(Gi, v)[2]) for Gi, p in zip(G, s["pieces"]) for v in p["orig"]]
    assert max(z) > 0.2  # the flap stands up mid-squash
    assert s["kin_gap"] < 1e-4  # squash is rigidly foldable: loops stay closed


def test_simple_valley_swings_like_before():
    """Single valley folds: still pieces don't move, moving pieces swing up (+z) through 90 deg at t = 0.5."""
    for name in ("cup", "hat"):
        for s in _kin_steps(name):
            if s["op"] != "valley" or s.get("tuck_anim"):
                continue  # tuck inside: scripted slide-in (test_tuck_anim.py)
            G = frames(s["pieces"], s["kin_order"], 0.5, retime=s.get("retime"))
            for Gi, p in zip(G, s["pieces"]):
                for v in p["orig"]:
                    x = _pos(Gi, v)
                    if not p["moving"]:
                        assert np.allclose(x[:2], _apply(p["Tb"], v), atol=1e-9) and abs(x[2]) < 1e-9
                    else:
                        assert x[2] > -1e-9
                if p["moving"]:
                    assert abs(Gi[2, 2]) < 1e-6  # normal lies in the table plane: standing upright


def test_mountain_swings_under():
    model = {"steps": [{"op": "mountain", "line": {"vertical": "center"}, "move": "bl"}]}
    s = build_timeline(model)["steps"][0]
    G = frames(s["pieces"], s["kin_order"], 0.5, retime=s.get("retime"))
    zs = [_pos(Gi, v)[2] for Gi, p in zip(G, s["pieces"]) if p["moving"] for v in p["orig"]]
    assert min(zs) < -0.4 and max(zs) < 1e-9


def test_bends_follow_folded_hinges():
    """Joints folded flat at both ends stay folded (normals opposite) all the way through a fold, so the
    viewer keeps their rounded bend visible."""
    seen = 0
    for name in ("cup", "hat", "bird_base"):
        for s in _kin_steps(name):
            n = len(s["pieces"])
            for b in s["kin_bends"]:
                assert 0 <= b["f"] < n and 0 <= b["o"] < n and b["f"] != b["o"]
                if not (abs(abs(b["theta"][0]) - np.pi) < 1e-9 and b["theta"][0] == b["theta"][1]):
                    continue
                if s["kin_gap"] > 0:
                    continue  # loop-solved steps: constant joints may flex slightly
                G = frames(s["pieces"], s["kin_order"], 0.5, retime=s.get("retime"))
                tol = 1e-6 if s.get("tuck_anim") else 1e-9  # scripted key frames are rounded to 1e-7
                assert np.dot(G[b["f"]][:3, 2], G[b["o"]][:3, 2]) < -1 + tol
                seen += 1
    assert seen > 0


# ---- timing: even speed (retime) ----

def test_retime_table_monotone_with_exact_ends():
    rng = np.random.default_rng(1)
    for d in (np.ones(40), rng.random(64), np.r_[np.zeros(20), 5.0, 9.0, np.zeros(20)], np.zeros(10)):
        tab = retime_table(d, samples=50)
        assert tab[0] == 0.0 and tab[-1] == 1.0 and len(tab) == 51
        assert all(b > a for a, b in zip(tab, tab[1:]))  # strictly increasing
    tab = retime_table(np.ones(40))
    assert np.allclose(tab, np.linspace(0, 1, len(tab)), atol=1e-6)  # even path: identity
    # a burst of motion in the middle gets most of the time
    tab = retime_table(np.r_[np.full(20, 0.01), np.full(4, 1.0), np.full(20, 0.01)], samples=100)
    u = np.linspace(0, 1, 45)
    assert np.searchsorted(tab, u[20]) < 15 and np.searchsorted(tab, u[24]) > 85
    assert retime_at(None, 0.3) == 0.3 and retime_at(tab, 0.0) == 0.0 and retime_at(tab, 1.0) == 1.0


def _speed_ratio(s, n=60):
    """Peak / mean of the largest vertex displacement per time sample over the whole step."""
    polys = [np.array(p["orig"]) for p in s["pieces"]]
    V = [np.c_[P, np.zeros(len(P)), np.ones(len(P))] for P in polys]
    prev, d = None, []
    for t in np.linspace(0, 1, n + 1):
        G = frames(s["pieces"], s["kin_order"], t, retime=s.get("retime"))
        X = np.concatenate([(v @ Gi.T)[:, :3] for v, Gi in zip(V, G)])
        if prev is not None:
            d.append(np.max(np.linalg.norm(X - prev, axis=1)))
        prev = X
    return max(d) / np.mean(d)


def test_solved_steps_move_at_even_speed():
    """Solved (keyed) steps used to snap through the middle (reverse fold: peak speed ~10x the mean).
    Retimed, every step has the speed profile of the plain ease curve (peak/mean ~ 2.85)."""
    keyed = [s for s in _kin_steps("crane") if s.get("retime")]
    assert {s["op"] for s in keyed} >= {"squash", "petal", "inside_reverse"}
    for s in keyed:
        assert _speed_ratio(s) < 3.3, (s["index"], s["op"])
        tab = s["retime"]
        assert tab[0] == 0.0 and tab[-1] == 1.0 and all(b > a for a, b in zip(tab, tab[1:]))


def test_path_len_exported():
    for s in _kin_steps("crane"):
        assert s["path_len"] > 0
        prof = path_profile(s["pieces"], s["kin_order"], [p["orig"] for p in s["pieces"]], 16)
        assert abs(prof.sum() - s["path_len"]) < 0.05 * s["path_len"] + 1e-3


def test_still_layers_stay_put_in_squash_and_petal():
    """Joints between two pieces that do not move are frozen in the path solve, so still layers
    don't flex and pass through each other mid-motion."""
    for s in _kin_steps("bird_base"):
        if s["op"] not in ("squash", "petal"):
            continue
        for t in (0.3, 0.5, 0.7):
            G = frames(s["pieces"], s["kin_order"], t, retime=s.get("retime"))
            for Gi, p in zip(G, s["pieces"]):
                if p["moving"]:
                    continue
                for v in p["orig"]:
                    x = _pos(Gi, v)
                    assert np.allclose(x[:2], _apply(p["Tb"], v), atol=1e-6) and abs(x[2]) < 1e-6, (s["index"], t)


def test_petal_is_one_driven_motion():
    """Petal = scripted motion: the flap lifts evenly the whole time while the sides swing in with it
    (loops closed exactly, no snapping): every moving hinge turns monotonically, all together."""
    for name in ("bird_base", "crane"):
        for s in _kin_steps(name):
            if s["op"] != "petal":
                continue
            assert s["kin_gap"] < 1e-6, (name, s["index"])
            moving = [p["keys"] for p in s["pieces"] if p.get("keys") and abs(p["keys"][-1] - p["keys"][0]) > 1]
            assert len(moving) >= 5  # flap + 2 top-layer sides + 2 second-layer sides
            for k in moving:
                d = np.diff(k) * np.sign(k[-1] - k[0])
                assert np.all(d > -1e-9) and np.max(np.abs(np.diff(k))) < 0.2  # monotone, no jumps
                assert abs(k[len(k) // 4] - k[0]) > 0.05 * abs(k[-1] - k[0])  # already moving early on


# ---- inside reverse: scripted "fake spread" (reverse_anim.py) ----

def _spread_steps():
    out = [s for s in _kin_steps("crane") if s["op"] == "inside_reverse"]
    assert len(out) == 3 and all(s.get("spread") for s in out)
    return out


def _inside_convex(poly, p, margin):
    poly = g.ccw(np.asarray(poly, float))
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        e = b - a
        if np.linalg.norm(e) > 1e-12 and (e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0])) / np.linalg.norm(e) <= margin:
            return False
    return True


def test_inside_reverse_tip_does_not_pass_through_the_flap():
    """Crane neck / tail / head: mid-move the tip never pokes through the flap's own still layers (it
    stays on their inner side where the halves open), never lies inside flat still paper, and the
    front and back tip halves never cross each other."""
    for s in _spread_steps():
        P = s["pieces"]
        tip = [i for i, p in enumerate(P) if p["moving"]]
        halves = {}
        for i in tip:
            halves.setdefault(P[i]["chain"][0]["m"], []).append(i)
        lay = {m: np.mean([P[i]["layer_before"] for i in ids]) for m, ids in halves.items()}
        up = max(lay, key=lay.get)
        thr = 0.5 * (min(P[i]["layer_before"] for i in halves[up])
                     + max(P[i]["layer_before"] for m, ids in halves.items() if m != up for i in ids))
        for t in (0.2, 0.35, 0.5, 0.65, 0.8):
            G = frames(P, s["kin_order"], t, retime=s.get("retime"))
            pts = []
            for i in tip:
                o = np.array(P[i]["orig"])
                c = o.mean(axis=0)
                for q in list(o) + [c] + [0.5 * (v + c) for v in o]:
                    pts.append((i, _pos(G[i], q)))
            # tip halves stay on their own side of the middle plane (z = 0 here)
            for i, x in pts:
                assert (x[2] > -1e-6) if i in halves[up] else (x[2] < 1e-6), (s["index"], t, i, x)
            for j, p in enumerate(P):
                if p["moving"]:
                    continue
                Gi = np.linalg.inv(G[j])
                det = 1.0 if np.linalg.det(np.array(p["Tb"])[:, :2]) > 0 else -1.0
                spread = not np.allclose(G[j][:2, :2], _ext(p["Tb"])[:2, :2], atol=1e-9) or abs(G[j][2, 3]) > 1e-9
                inner = -1.0 if p["layer_before"] > thr else 1.0
                for i, x in pts:
                    L = Gi @ np.r_[x, 1.0]
                    if not _inside_convex(p["orig"], L[:2], 2e-3):
                        continue
                    if spread:   # opened half: the tip must be on its inner side
                        assert inner * det * L[2] > -1e-6, (s["index"], t, "through still layer", j, "tip", i)
                    else:        # flat still paper: the tip must not lie in it
                        assert abs(L[2]) > 1e-3, (s["index"], t, "inside flat paper", j, "tip", i)


def _ext(T):
    T = np.asarray(T, float)
    G = np.eye(4)
    G[:2, :2] = T[:2, :2]
    return G


def test_inside_reverse_spread_opens_a_little_and_keeps_the_tip_joined():
    """The flap halves open mid-move (not flat), the tip stays one piece of paper (its joints, incl. the
    tip spine, hold exactly) and the joints that do open (reverse crease, flap-body join) open only a bit."""
    for s in _spread_steps():
        P = s["pieces"]
        assert s["spread"]["hits"] == 0
        grp = [tuple(c["m"] for c in p.get("chain") or []) for p in P]
        for t in (0.25, 0.5, 0.75):
            G = frames(P, s["kin_order"], t, retime=s.get("retime"))
            for i, j, seg in _adjacent_pairs(s):
                d = max(np.linalg.norm(_pos(G[i], v) - _pos(G[j], v)) for v in seg)
                if grp[i] and grp[j]:
                    assert d < 1e-5, (s["index"], t, i, j, d)          # tip halves and tip spine
                else:
                    assert d < 0.06, (s["index"], t, i, j, d)          # small temporary gap
        # halves open: some still piece tilts, but by less than 60 degrees
        G = frames(P, s["kin_order"], 0.5, retime=s.get("retime"))
        tilt = [np.degrees(np.arccos(min(1.0, abs(G[j][2, 2])))) for j, p in enumerate(P) if not p["moving"]]
        assert 15 < max(tilt) < 60, (s["index"], max(tilt))
