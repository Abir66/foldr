"""Local layer order, tuck, inside reverse, squash, petal."""

import json
from pathlib import Path

import numpy as np
import pytest

from origami import geometry as g
from origami.cli import run_model
from origami.engine import Simulator
from origami.fold_export import to_fold

MODELS = Path(__file__).resolve().parents[1] / "models"


def run(steps):
    sim = Simulator()
    for s in steps:
        sim.apply(s)
    return sim.state


def load(name):
    base = MODELS.parent / "tests" / "data" / "models" if name == "hat_tuck" else MODELS
    return json.loads((base / f"{name}.json").read_text())


def face_at(state, orig_pt):
    return [k for k, f in enumerate(state.faces) if g.contains(f.orig, np.array(orig_pt))
            and g.area(f.orig) > 1e-9]


def assert_consistent(state):
    """Order only between overlapping faces, antisymmetric, and draw order respects it."""
    for i, j in state.above:
        assert (j, i) not in state.above
        assert g.overlap_area(state.faces[i].folded, state.faces[j].folded) > 1e-7
        assert i > j  # faces list is a linear extension (bottom -> top)
    assert state.acyclic


# ---- tuck ---------------------------------------------------------------------------

def test_hat_back_flap_is_tucked_inside():
    st = run_model(load("hat_tuck"))[-1]
    assert_consistent(st)
    (flap,) = face_at(st, (0.1, 0.1))  # the back bottom corner (bl), tucked up
    over = [i for i, j in st.above if j == flap]
    under = [j for i, j in st.above if i == flap]
    assert len(under) == 1  # only the back layer it is hinged to lies below it
    assert len(over) >= 3   # all the front layers stay in front of it
    assert np.isclose(st.total_area(), 1.0)


def test_tuck_under_keeps_top_layers_over_the_flap():
    # fold in half (bottom up), then fold the right part over with tuck under 1:
    st = run([{"op": "valley", "line": {"horizontal": "center"}, "move": "bl"},
              {"op": "valley", "line": {"vertical": [0.6, 0.5]}, "move": [0.9, 0.3],
               "layers": {"top": 1}, "tuck": {"under": 1}}])
    assert_consistent(st)
    moved = [k for k, f in enumerate(st.faces) if f.folded[:, 0].max() <= 0.6 + 1e-9
             and f.orig[:, 0].min() >= 0.6 - 1e-9]
    assert moved
    for m in moved:  # exactly one static layer stays over each moved face
        assert sum(1 for i, j in st.above if j == m and i not in moved) == 1


# ---- inside reverse -----------------------------------------------------------------

def reverse_triangle():
    return run([{"op": "valley", "line": {"through": ["bl", "tr"]}, "move": "tl"},
                {"op": "inside_reverse", "line": {"horizontal": {"orig": [1, 0.7]}}, "move": "tr"}])


def test_inside_reverse_geometry_and_order():
    st = reverse_triangle()
    assert_consistent(st)
    assert np.isclose(st.total_area(), 1.0)
    lo, hi = st.bbox()
    assert np.allclose(hi, [1.0, 0.7])  # the tip went inside: nothing above the line
    tip_a = face_at(st, (0.75, 0.95))[0]  # tip of the top layer (came from tl half)
    tip_b = face_at(st, (0.95, 0.75))[0]  # tip of the bottom layer
    body_a = face_at(st, (0.2, 0.5))[0]
    body_b = face_at(st, (0.5, 0.2))[0]
    # order (top -> bottom): A, A', B', B
    assert st.is_above(body_a, tip_a) and st.is_above(tip_a, tip_b) and st.is_above(tip_b, body_b)
    assert np.allclose(st.to_folded((1, 1)), [1.0, 0.4])
    # spine crease beyond the line is reversed (was V, now M)
    spine = [c for c in st.creases if np.allclose(sorted([c.p, c.q]), [[0.7, 0.7], [1, 1]], atol=1e-6)]
    assert spine and spine[-1].assignment == "M"


def test_inside_reverse_records_two_sub_motions():
    st = reverse_triangle()
    assert [m["sign"] for m in st.motions] == [-1, 1]
    assert all(m["type"] == "reflect" and m["phase"] == 0 for m in st.motions)


# ---- squash / petal -----------------------------------------------------------------

BB = None


def bird_states():
    global BB
    if BB is None:
        BB = run_model(load("bird_base"))
    return BB


def test_squash_is_symmetric_about_the_hinge():
    st = bird_states()[4]  # after the first squash
    assert_consistent(st)
    a, b = st.to_folded((0.5, 0.5)), st.to_folded((0, 0))  # hinge: centre -> bottom point
    u = b - a
    R = g.reflection(a, u)
    # the squashed flap = faces lying inside the diamond with diagonal a-b
    c = (a + b) / 2
    diamond = np.array([a, c + np.array([-u[1], u[0]]) / 2, b, c - np.array([-u[1], u[0]]) / 2])
    flap = [f.folded for f in st.faces if g.overlap_area(f.folded, diamond) > g.area(f.folded) - 1e-9]
    assert len(flap) == 4
    left = [p for p in flap if g.side(a, u, p.mean(0)) > 0]
    right = [p for p in flap if g.side(a, u, p.mean(0)) < 0]
    assert len(left) == len(right) == 2
    for p in left:  # every piece has a mirror image on the other side of the hinge
        q = g.apply(R, p)
        assert any(np.isclose(g.overlap_area(q, r), g.area(q)) for r in right)
    assert np.isclose(sum(map(g.area, flap)), g.area(diamond) * 2)  # 2 layers each side


def test_preliminary_base_is_a_four_layer_square():
    st = bird_states()[6]
    assert_consistent(st)
    lo, hi = st.bbox()
    assert np.allclose(hi - lo, [np.sqrt(0.5), np.sqrt(0.5)])
    assert np.isclose(st.total_area(), 1.0)
    assert len(st.faces) == 8
    corners = [st.to_folded(p) for p in [(0, 0), (1, 0), (1, 1), (0, 1)]]
    assert all(np.allclose(c, corners[0]) for c in corners)


def test_bird_base_shape_and_area():
    states = bird_states()
    for st in states:
        assert np.isclose(sum(f.area for f in st.faces), 1.0)
        assert_consistent(st)
    st = states[-1]
    lo, hi = st.bbox()
    assert np.isclose(hi[1] - lo[1], 1.0)            # tall diamond: petal tip to leg tips
    assert np.isclose(hi[0] - lo[0], np.sqrt(2) - 1)  # width 2 * (sqrt2 - 1) / 2
    tips = [st.to_folded(p) for p in [(1, 0), (0, 1)]]
    legs = [st.to_folded(p) for p in [(0, 0), (1, 1)]]
    assert np.allclose(tips[0], tips[1]) and np.isclose(tips[0][1], hi[1])
    assert np.allclose(legs[0], legs[1]) and np.isclose(legs[0][1], lo[1])


def test_petal_motions_recorded():
    st = bird_states()[7]  # after the front petal
    phases = [m["phase"] for m in st.motions]
    assert phases[0] == 0 and phases[-1] == 1 and len(st.motions) == 5


def _top_at(st, p):
    hits = [k for k, f in enumerate(st.faces) if g.contains(f.folded, np.array(p))
            and g.area(f.orig) > 1e-9]
    return hits[-1]


@pytest.mark.parametrize("name,step", [("bird_base", 7), ("bird_base", 9),
                                       ("crane", 7), ("crane", 9)])
def test_petal_sides_cover_the_flap_front(name, step):
    """Real paper: after a petal fold the side triangles swing in ON TOP of the lifted flap,
    showing the prelim's outside colour, with their raw edges meeting on the centre line (slit).
    Under them is the flap's inner face (the opposite colour)."""
    states = bird_states() if name == "bird_base" else run_model(load(name))
    before, st = states[step - 1], states[step]
    assert_consistent(st)
    lo, hi = st.bbox()
    cx = (lo[0] + hi[0]) / 2
    pre_top = before.faces[_top_at(before, ((lo[0] + hi[0]) / 2, 0.3))]
    for y in (0.4, 0.55):  # upper diamond (above the petal hinge line)
        left, right = _top_at(st, (cx - 0.02, y)), _top_at(st, (cx + 0.02, y))
        assert left != right  # two side triangles meet at the centre slit
        for k, x in ((left, cx - 0.02), (right, cx + 0.02)):
            f = st.faces[k]
            assert f.front_up == pre_top.front_up  # outside colour of the prelim
            # the side piece has a raw paper edge lying on the centre line
            assert any(abs(p[0] - cx) < 1e-6 and abs(q[0] - cx) < 1e-6
                       for p, q in zip(f.folded, np.roll(f.folded, -1, axis=0))), f.folded
            # underneath: the lifted flap, showing its inner face
            under = [j for j in range(k) if g.contains(st.faces[j].folded, np.array((x, y)))]
            assert st.faces[under[-1]].front_up != pre_top.front_up


def test_fold_export_face_orders_match_state():
    st = bird_states()[-1]
    fr = to_fold(st)["file_frames"][0]
    assert len(fr["faceOrders"]) == len(st.above)


def test_crane_runs_acyclic_with_area():
    st = run_model(load("crane"))[-1]
    assert_consistent(st)
    assert np.isclose(st.total_area(), 1.0)


@pytest.mark.parametrize("name", ["triangle", "cup", "hat", "fortune_teller"])
def test_existing_models_have_consistent_orders(name):
    for st in run_model(load(name)):
        assert_consistent(st)


CRANE = None


def crane_states():
    global CRANE
    if CRANE is None:
        CRANE = run_model(load("crane"))
    return CRANE


def _moved_faces(before, after):
    """Faces of `after` whose transform differs from the `before` face they came from."""
    out = []
    for k, f in enumerate(after.faces):
        c = f.orig.mean(0)
        src = next(j for j, h in enumerate(before.faces) if g.contains(h.orig, c))
        if not np.allclose(f.T, before.faces[src].T, atol=1e-7):
            out.append(k)
    return out


@pytest.mark.parametrize("step", [15, 16, 17])
def test_crane_inside_reverse_matches_real_paper(step):
    """Real inside reverse (neck, tail, head): the tip goes INSIDE, between the flap's own
    halves. The crane's two halves are the two sides of the sheet's diagonal x = y (the
    step-3 fold = spine of every flap). Checked facts:
      - every reversed face lies under all still front-half layers it overlaps and over all
        still back-half layers (hidden between them, so the outside keeps its colour);
      - the front half's reversed layers stay on the front side (above the back half's);
      - mirrored order: a deeper still layer has a shallower reversed tip (hinges nest);
      - the tip's spine (diagonal fold between the two halves) is reversed (V <-> M)."""
    before, st = crane_states()[step - 1], crane_states()[step]
    assert_consistent(st)
    moved = _moved_faces(before, st)
    assert len(moved) == 8
    half = {k: f.orig.mean(0)[0] > f.orig.mean(0)[1] for k, f in enumerate(st.faces)}
    front = half[max(moved)]  # topmost moved face belongs to the front half
    still = [k for k in range(len(st.faces)) if k not in moved]
    pairs = 0
    for m in moved:
        for s in still:
            r = st.is_above(s, m)
            if r is None:
                continue
            pairs += 1
            assert r == (half[s] == front), (m, s)
        for m2 in moved:
            if half[m] == front and half[m2] != front and st.is_above(m, m2) is not None:
                assert st.is_above(m, m2)
    assert pairs > 0
    # mirrored nesting: hinge partner (still face sharing an edge with the tip face)
    partner = {}
    for m in moved:
        for s in still:
            if g.shared_edge(st.faces[m].orig, st.faces[s].orig) is not None:
                partner[m] = s
    for m1 in partner:
        for m2 in partner:
            s1, s2 = partner[m1], partner[m2]
            if m1 < m2 and half[m1] == half[m2] and st.is_above(s1, s2) is not None \
                    and st.is_above(m1, m2) is not None:
                assert st.is_above(s1, s2) != st.is_above(m1, m2)
    # spine of the tip reversed
    def spine_asg(state, faces):
        for i in faces:
            for j in faces:
                fi, fj = state.faces[i], state.faces[j]
                if half[i] != front or half[j] == front:
                    continue
                e = g.shared_edge(fi.orig, fj.orig)
                e = None if e is None else np.array(e, float)
                if e is None or not np.allclose(e[:, 0], e[:, 1], atol=1e-6):
                    continue  # only the diagonal (x = y) edge
                return "V" if state.is_above(j, i) == fi.front_up else "M", e
        return None, None

    asg, e = spine_asg(st, moved)
    assert asg is not None
    mid = np.asarray(e).mean(0)
    src = [j for j, f in enumerate(before.faces) if g.contains(f.orig, mid + [1e-4, -1e-4])
           or g.contains(f.orig, mid + [-1e-4, 1e-4])]
    i = next(j for j in src if (before.faces[j].orig.mean(0)[0] > before.faces[j].orig.mean(0)[1]) == front)
    k = next(j for j in src if j != i)
    old = "V" if before.is_above(k, i) == before.faces[i].front_up else "M"
    assert asg != old
