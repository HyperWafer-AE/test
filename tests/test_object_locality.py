from __future__ import annotations

import pandas as pd

from src.analysis.object_locality import run_object_locality_analysis


def test_stable_reuse_excludes_large_observation_bucket(tmp_path) -> None:
    objects = pd.DataFrame(
        [
            {"trace_id": "t", "step_id": 0, "object_type": "file", "object_id": "file:a.py", "stable_object": True},
            {"trace_id": "t", "step_id": 2, "object_type": "file", "object_id": "file:a.py", "stable_object": True},
            {
                "trace_id": "t",
                "step_id": 0,
                "object_type": "large_observation_bucket",
                "object_id": "large_obs:x",
                "stable_object": False,
            },
            {
                "trace_id": "t",
                "step_id": 1,
                "object_type": "large_observation_bucket",
                "object_id": "large_obs:x",
                "stable_object": False,
            },
        ]
    )
    out = run_object_locality_analysis(objects, tmp_path / "figures", tmp_path / "tables")
    summary = out["object_reuse_summary"].set_index("reuse_class")
    assert summary.loc["stable", "reuse_events"] == 1
    assert summary.loc["synthetic", "reuse_events"] == 1

