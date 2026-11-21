"""Decision agent with a mocked API: picks the argmax option, 'submit' stops the episode."""
from origami.agents.decision_agent import SUBMIT_KEY, pick, run_episode_decision


def test_pick_argmax():
    key, probs = pick({"type": "choice", "choice": "1", "probabilities": {"1": 0.2, "3": 0.7, SUBMIT_KEY: 0.1}})
    assert key == "3" and probs["3"] == 0.7


def test_episode_mocked(tmp_path):
    calls = []

    def fake(model, state, questions, key):
        calls.append(questions)
        crit = questions["fold"]["criteria"]
        assert SUBMIT_KEY in crit and "1" in crit
        if len(calls) == 1:
            probs = {k: 0.0 for k in crit}
            probs["1"] = 1.0
        else:
            probs = {k: 0.0 for k in crit}
            probs[SUBMIT_KEY] = 1.0
        return {"answers": {"fold": {"type": "choice", "choice": "x", "probabilities": probs}},
                "usage": {"input_tokens": 10, "cost": 0.001}}

    s = run_episode_decision("models/cup.json", "fake/model", images=False, out_root=tmp_path, log=lambda *_: None,
                             caller=fake, key="k")
    assert s["stop_reason"] == "submitted" and s["menu_moves"] == 1
    assert abs(s["cost_usd"] - 0.002) < 1e-9
    import json
    rows = [json.loads(l) for l in (next(tmp_path.rglob("turns.jsonl"))).read_text().splitlines()]
    assert rows[0]["probabilities"]["1"] == 1.0


def test_plan_text_reaches_decision_model(tmp_path):
    """--plan-file: plan + moves done so far are in the state; instructions ask for the next plan step."""
    seen = []

    def fake(model, state, questions, key):
        seen.append((state, questions))
        crit = questions["fold"]["criteria"]
        probs = {k: 0.0 for k in crit}
        probs["1" if len(seen) == 1 else SUBMIT_KEY] = 1.0
        return {"answers": {"fold": {"type": "choice", "probabilities": probs}}, "usage": {"cost": 0.0}}

    run_episode_decision("models/cup.json", "fake/model", images=False, out_root=tmp_path, log=lambda *_: None,
                         caller=fake, key="k", plan_text="1. Fold the square in half to a triangle. ZEBRA")
    s1, q1 = seen[0]
    s2, _ = seen[1]
    assert "expert folder" in s1 and "ZEBRA" in s1 and "(none yet" in s1
    assert "next plan step" in q1["fold"]["instructions"]
    assert "ZEBRA" in s2 and "Moves done so far:\n1. " in s2


def test_plan_text_reaches_menu_model(tmp_path):
    import json

    from origami.agents.menu_agent import run_episode_menu

    class Fake:
        model = "fake/model"

        def __init__(self):
            self.seen = []

        def chat(self, messages, tools):
            self.seen.append(messages)
            tc = [{"id": "c0", "type": "function",
                   "function": {"name": "submit", "arguments": json.dumps({"plan": "x"})}}]
            return {"choices": [{"message": {"content": "", "tool_calls": tc}}], "usage": {"cost": 0.0}}

    fc = Fake()
    run_episode_menu("models/cup.json", "fake/model", images=False, out_root=tmp_path, log=lambda *_: None,
                     client=fc, plan_text="1. Valley fold corner to corner. ZEBRA")
    texts = " ".join(p["text"] for p in fc.seen[0][1]["content"] if p.get("type") == "text")
    assert "expert folder" in texts and "ZEBRA" in texts
