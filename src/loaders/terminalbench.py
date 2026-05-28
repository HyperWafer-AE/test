"""Loader for ``yoonholee/terminalbench-trajectories``.

The real dataset is hosted on HuggingFace and stores one trajectory per row.
The ``steps`` field is a JSON-serialized list of step objects.  This loader is
deliberately thin: it returns raw rows and metadata, while normalization lives
in ``src.normalize.normalizer``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

LOGGER = logging.getLogger(__name__)

DATASET_NAME = "yoonholee/terminalbench-trajectories"


@dataclass(slots=True)
class LoadResult:
    dataset: str
    rows: list[dict[str, Any]]
    warnings: list[str]
    used_mock: bool = False


def _has_nonempty_steps(row: dict[str, Any]) -> bool:
    raw = row.get("steps")
    if raw in (None, "", "null"):
        return False
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception:
            return bool(raw.strip())
        return isinstance(parsed, list) and len(parsed) > 0
    return isinstance(raw, list) and len(raw) > 0


def mock_terminalbench_rows() -> list[dict[str, Any]]:
    """Small realistic-enough fallback for offline CI and smoke tests."""

    traces = [
        (
            "tb_mock_success_1",
            "fix-parser-edge-case",
            "codex",
            "gpt-5-mini",
            1,
            [
                ("assistant", "ls", "List repository", "README.md\nsrc/parser.py\ntests/test_parser.py"),
                ("assistant", "cat", "Open src/parser.py", "def parse(x):\n    return x.split(',')\n"),
                ("assistant", "grep", "Search parser tests", "tests/test_parser.py:12:test_empty"),
                ("assistant", "edit", "Patch parser empty input", "patched src/parser.py"),
                ("assistant", "pytest", "Run tests", "3 passed in 0.21s"),
                ("assistant", "final", "Submit answer", "resolved"),
            ],
        ),
        (
            "tb_mock_fail_loop",
            "repair-cli-timeout",
            "openhands",
            "claude-sonnet",
            0,
            [
                ("assistant", "ls", "Inspect files", "app.py\nrequirements.txt\ntests/test_cli.py"),
                ("assistant", "cat", "Read app.py", "def main():\n    while True:\n        pass\n"),
                ("assistant", "pytest", "Run failing tests", "FAILED tests/test_cli.py::test_timeout\nAssertionError"),
                ("assistant", "sed", "Try quick edit", "modified app.py"),
                ("assistant", "pytest", "Run failing tests again", "FAILED tests/test_cli.py::test_timeout\nAssertionError"),
                ("assistant", "sed", "Try second edit", "modified app.py"),
                ("assistant", "pytest", "Run failing tests again", "FAILED tests/test_cli.py::test_timeout\nAssertionError"),
                ("assistant", "final", "Give up", "not resolved"),
            ],
        ),
        (
            "tb_mock_success_2",
            "numpy-shape-bug",
            "mini-swe-agent",
            "qwen-coder",
            1,
            [
                ("assistant", "find", "Find python files", "./model.py\n./tests/test_model.py"),
                ("assistant", "cat", "Read tests/test_model.py", "assert output.shape == (2, 3)"),
                ("assistant", "cat", "Read model.py", "return x.reshape(3, 2)"),
                ("assistant", "edit", "Fix reshape", "patched model.py"),
                ("assistant", "python", "Run focused script", "shape ok"),
                ("assistant", "pytest", "Run suite", "5 passed"),
            ],
        ),
        (
            "tb_mock_browser",
            "api-docs-update",
            "factory-droid",
            "gpt-5-codex",
            1,
            [
                ("assistant", "search", "Search docs", "Result: API v2 changed status field"),
                ("assistant", "fetch", "Fetch docs", "status: completed|failed|pending\nretry_after: integer"),
                ("assistant", "cat", "Read client.py", "class Client:\n    def status(self): ..."),
                ("assistant", "edit", "Update field parser", "patched client.py"),
                ("assistant", "pytest", "Run tests", "7 passed"),
            ],
        ),
    ]
    rows: list[dict[str, Any]] = []
    for trace_id, task, agent, model, reward, steps in traces:
        step_dicts = [
            {
                "role": role,
                "tool_name": tool,
                "content": msg,
                "observation": obs,
                "arguments": msg,
            }
            for role, tool, msg, obs in steps
        ]
        rows.append(
            {
                "trial_id": trace_id,
                "trial_name": trace_id,
                "task_name": task,
                "agent": agent,
                "model": model,
                "reward": reward,
                "duration_seconds": 60.0 + 12.5 * len(step_dicts),
                "input_tokens": 1000 + 80 * len(step_dicts),
                "output_tokens": 250 + 35 * len(step_dicts),
                "cache_tokens": 0,
                "steps": json.dumps(step_dicts),
            }
        )
    return rows


def load_terminalbench(
    sample_size: int = 5000,
    split: str = "train",
    streaming: bool = True,
    cache_dir: str | None = None,
    seed: int = 0,
    offline_mock: bool = False,
    skip_empty_steps: bool = True,
) -> LoadResult:
    """Load or stream a bounded sample of Terminal-Bench trajectories."""

    warnings: list[str] = []
    if offline_mock:
        warnings.append("terminalbench: offline_mock requested; using bundled mock rows.")
        return LoadResult(DATASET_NAME, mock_terminalbench_rows()[:sample_size], warnings, used_mock=True)

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local env
        warnings.append(f"terminalbench: datasets import failed ({exc}); using mock rows.")
        return LoadResult(DATASET_NAME, mock_terminalbench_rows()[:sample_size], warnings, used_mock=True)

    rows: list[dict[str, Any]] = []
    skipped_empty = 0
    try:
        ds = load_dataset(DATASET_NAME, split=split, streaming=streaming, cache_dir=cache_dir)
        if streaming and hasattr(ds, "shuffle"):
            try:
                ds = ds.shuffle(seed=seed, buffer_size=max(1000, min(sample_size * 20, 20000)))
            except Exception as exc:
                warnings.append(f"terminalbench: streaming shuffle failed ({exc}); using dataset order.")

        for row in ds:
            raw_row = dict(row)
            if skip_empty_steps and not _has_nonempty_steps(raw_row):
                skipped_empty += 1
                if skipped_empty > sample_size * 10 and rows:
                    break
                continue
            rows.append(raw_row)
            if len(rows) >= sample_size:
                break
    except Exception as exc:
        LOGGER.exception("Terminal-Bench loading failed")
        warnings.append(f"terminalbench: load failed ({exc}); using mock rows.")
        return LoadResult(DATASET_NAME, mock_terminalbench_rows()[:sample_size], warnings, used_mock=True)

    if skipped_empty:
        warnings.append(f"terminalbench: skipped {skipped_empty} rows with empty/missing steps.")
    if not rows:
        warnings.append("terminalbench: no rows loaded; using mock rows.")
        return LoadResult(DATASET_NAME, mock_terminalbench_rows()[:sample_size], warnings, used_mock=True)
    return LoadResult(DATASET_NAME, rows, warnings, used_mock=False)
