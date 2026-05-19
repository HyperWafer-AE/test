from __future__ import annotations

from pathlib import Path


def test_round5_scripts_that_generate_figure_inputs_exist():
    root = Path("/home/duzc/data/agent_wafer")
    for rel in [
        "scripts/run_existing_cache_gap.py",
        "scripts/run_decode_cohort_sweep.py",
        "scripts/run_replication_tradeoff.py",
        "scripts/make_paper_figures.py",
        "waferagent/paper_figures.py",
    ]:
        assert (root / rel).exists(), rel
