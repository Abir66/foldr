import json
from pathlib import Path

from origami.bends import bends
from origami.cli import run_model
from origami.render import render_step_png

MODEL = Path(__file__).parent.parent / "models" / "cup.json"


def test_bends_found_and_render_runs(tmp_path):
    states = run_model(json.loads(MODEL.read_text()))
    final = states[-1]
    assert len(bends(final)) > 0  # folded cup has fold-connected layers
    lo, hi = final.bbox()
    out = tmp_path / "s.png"
    render_step_png(out, final, "t", None, ((lo[0] - 0.2, lo[1] - 0.2), (hi[0] + 0.2, hi[1] + 0.2)))
    assert out.stat().st_size > 1000
