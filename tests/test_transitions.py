from __future__ import annotations

import pandas as pd

from src.analysis.transitions import run_transition_analysis


def test_transition_analysis_outputs_layered_views(tmp_path) -> None:
    steps = pd.DataFrame(
        [
            {"trace_id": "t1", "step_id": 0, "tool_wrapper": "bash_command", "semantic_tool": "ls", "phase": "explore/read"},
            {"trace_id": "t1", "step_id": 1, "tool_wrapper": "bash_command", "semantic_tool": "pytest", "phase": "execute/test"},
            {"trace_id": "t1", "step_id": 2, "tool_wrapper": "editor", "semantic_tool": "edit", "phase": "edit/write"},
            {"trace_id": "t2", "step_id": 0, "tool_wrapper": "bash_command", "semantic_tool": "grep", "phase": "explore/read"},
            {"trace_id": "t2", "step_id": 1, "tool_wrapper": "editor", "semantic_tool": "edit", "phase": "edit/write"},
        ]
    )
    out = run_transition_analysis(steps, tmp_path / "figures", tmp_path / "tables")
    recall = out["transition_recall_by_view"]
    assert {"wrapper_tool", "semantic_tool", "collapsed_semantic_tool", "phase"}.issubset(set(recall["view"]))
    assert (tmp_path / "tables" / "wrapper_tool_transition_top_pairs.csv").exists()
    assert (tmp_path / "tables" / "semantic_tool_transition_top_pairs.csv").exists()
    assert (tmp_path / "tables" / "collapsed_semantic_tool_transition_top_pairs.csv").exists()

