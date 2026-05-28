from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from src.loaders.terminalbench import DATASET_NAME, mock_terminalbench_rows
from src.normalize.normalizer import normalize_rows


def test_mock_terminalbench_normalizes_to_canonical_tables() -> None:
    traces, steps, objects, warnings = normalize_rows(DATASET_NAME, mock_terminalbench_rows())

    assert warnings == []
    assert len(traces) >= 1
    assert len(steps) >= 1
    assert len(objects) >= 1
    assert {"trace_id", "dataset", "total_steps"}.issubset(traces.columns)
    assert {"trace_id", "step_id", "phase", "tool_name"}.issubset(steps.columns)
    assert {"trace_id", "step_id", "object_type", "object_id"}.issubset(objects.columns)
    assert "execute/test" in set(steps["phase"])


def test_run_all_offline_mock_outputs_required_artifacts(tmp_path: Path) -> None:
    cmd = [
        sys.executable,
        "scripts/run_all.py",
        "--datasets",
        "mock",
        "--sample-size",
        "10",
        "--offline-mock",
        "--outdir",
        str(tmp_path),
    ]
    subprocess.run(cmd, check=True)

    assert (tmp_path / "reports" / "agent_trace_profile.md").exists()
    assert (tmp_path / "data" / "normalized" / "traces.csv").exists()
    assert (tmp_path / "figures" / "tool_transition_heatmap.png").exists()
    assert (tmp_path / "tables" / "fingerprint_metrics.csv").exists()
