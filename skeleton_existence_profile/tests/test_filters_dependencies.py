from __future__ import annotations

import json

import pandas as pd

from skeleton.experiments import transition_rows
from skeleton.loader import load_terminalbench_strict_real
from skeleton.normalize import normalize_terminalbench_rows


def test_unknown_no_tool_not_in_tool_action_transition() -> None:
    rows = [
        {
            "trial_id": "t1",
            "task_name": "task",
            "agent": "h",
            "model": "m",
            "reward": 1,
            "steps": json.dumps(
                [
                    {"src": "agent", "msg": "thinking", "tools": None, "obs": None},
                    {"src": "agent", "tools": [{"fn": "Bash", "cmd": "cat a.py"}], "obs": "x"},
                    {"src": "agent", "tools": [{"fn": "Bash", "cmd": "pytest"}], "obs": "passed"},
                ]
            ),
        }
    ]
    _, steps, _, _, _ = normalize_terminalbench_rows(rows)
    trans = transition_rows(steps, "semantic_tool_tool_action_only")
    assert set(trans["current"]) == {"read_file"}
    assert set(trans["next"]) == {"test"}


def test_artifact_filtered() -> None:
    rows = [
        {
            "trial_id": "t1",
            "task_name": "task",
            "agent": "h",
            "model": "m",
            "reward": 1,
            "steps": json.dumps(
                [
                    {"src": "agent", "cmd": "{json artifact}", "obs": ""},
                    {"src": "agent", "cmd": "cat a.py", "obs": "a.py"},
                    {"src": "agent", "cmd": "pytest", "obs": "passed"},
                ]
            ),
        }
    ]
    _, steps, _, _, _ = normalize_terminalbench_rows(rows)
    trans = transition_rows(steps, "semantic_tool_tool_action_only")
    assert len(trans) == 1


def test_observation_path_requires_future_command_for_dependency() -> None:
    rows = [
        {
            "trial_id": "t1",
            "task_name": "task",
            "agent": "h",
            "model": "m",
            "reward": 1,
            "steps": json.dumps(
                [
                    {"src": "agent", "cmd": "grep foo .", "obs": "src/a.py: foo"},
                    {"src": "agent", "cmd": "echo hello", "obs": ""},
                ]
            ),
        }
    ]
    _, _, _, deps, _ = normalize_terminalbench_rows(rows)
    assert deps.empty
    rows[0]["steps"] = json.dumps(
        [
            {"src": "agent", "cmd": "grep foo .", "obs": "src/a.py: foo"},
            {"src": "agent", "cmd": "cat src/a.py", "obs": "foo"},
        ]
    )
    _, _, _, deps, _ = normalize_terminalbench_rows(rows)
    assert not deps.empty
    assert deps.iloc[0]["dependency_type"] == "file_path_from_output_to_next_arg"


def test_large_observation_bucket_not_stable_object() -> None:
    rows = [
        {
            "trial_id": "t1",
            "task_name": "task",
            "agent": "h",
            "model": "m",
            "reward": 1,
            "steps": json.dumps([{"src": "agent", "cmd": "cat a.py", "obs": "x" * 5001}]),
        }
    ]
    _, _, objects, _, _ = normalize_terminalbench_rows(rows)
    bucket = objects[objects["object_type"] == "large_observation_bucket"].iloc[0]
    assert not bool(bucket["stable_object"])


def test_exact_object_and_path_prefix_are_separate() -> None:
    rows = [
        {
            "trial_id": "t1",
            "task_name": "task",
            "agent": "h",
            "model": "m",
            "reward": 1,
            "steps": json.dumps([{"src": "agent", "cmd": "cat src/pkg/a.py", "obs": "src/pkg/a.py"}]),
        }
    ]
    _, _, objects, _, _ = normalize_terminalbench_rows(rows)
    file_obj = objects[objects["object_type"] == "file"].iloc[0]
    assert file_obj["object_id"] == "file:src/pkg/a.py"
    assert file_obj["object_prefix"] == "src/pkg"


def test_permutation_changes_temporal_order() -> None:
    rows = [
        {
            "trial_id": "t1",
            "task_name": "task",
            "agent": "h",
            "model": "m",
            "reward": 1,
            "steps": json.dumps(
                [
                    {"src": "agent", "cmd": "cat a.py", "obs": ""},
                    {"src": "agent", "cmd": "pytest", "obs": ""},
                    {"src": "agent", "cmd": "cat b.py", "obs": ""},
                    {"src": "agent", "cmd": "python x.py", "obs": ""},
                ]
            ),
        }
    ]
    _, steps, _, _, _ = normalize_terminalbench_rows(rows)
    real = transition_rows(steps, "semantic_tool_tool_action_only")
    shuffled = transition_rows(steps, "semantic_tool_tool_action_only", shuffle="within_trace", seed=2)
    assert real["next"].tolist() != shuffled["next"].tolist()


def test_strict_real_no_mock_fallback(tmp_path) -> None:
    try:
        load_terminalbench_strict_real(1, tmp_path / "raw.jsonl", base_url="https://invalid.local", max_retries=1)
    except RuntimeError as exc:
        assert "strict-real" in str(exc)
    else:
        raise AssertionError("strict-real loader should not return mock rows")
