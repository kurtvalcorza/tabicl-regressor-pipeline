"""Tier C review findings T98-1, T98-2, T98-4, T98-5 (2026-10-25)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "tutorials" / "tabiclv2_regressor_colab.ipynb"


def _markdown():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    md = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    return " ".join(md.split())


def test_t98_1_baseline_numbers_name_their_run():
    assert "the hosted Colab T4 run of the previous blob (`426cd4d6`" in _markdown()


def test_t98_2_no_unprinted_random_forest_test_figure():
    md = _markdown()
    assert "44.67" not in md and "0.470" not in md


def test_t98_4_untuned_baselines_disclosed():
    md = _markdown()
    assert "so the comparison is fair" not in md and "None is tuned" in md


def test_t98_5_dat9_overlap_statement():
    md = _markdown()
    assert "**Overlap with the training data:**" in md and "has not been checked" in md
