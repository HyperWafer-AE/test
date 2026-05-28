"""Loader for ``nebius/SWE-agent-trajectories``.

The dataset stores SWE-agent style message trajectories in a list-like
``trajectory`` field, with task metadata such as ``instance_id``,
``model_name``, ``target`` and ``exit_status``.  Loading is bounded and can run
in HuggingFace streaming mode to avoid materializing the full corpus.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

LOGGER = logging.getLogger(__name__)

DATASET_NAME = "nebius/SWE-agent-trajectories"


@dataclass(slots=True)
class LoadResult:
    dataset: str
    rows: list[dict[str, Any]]
    warnings: list[str]
    used_mock: bool = False


def mock_swe_agent_rows() -> list[dict[str, Any]]:
    return [
        {
            "instance_id": "mock__project-101",
            "model_name": "swe-agent-llama-70b",
            "target": True,
            "exit_status": "submitted",
            "trajectory": [
                {"role": "system", "content": "You are fixing a bug in a repository."},
                {"role": "assistant", "tool_name": "open", "content": "open src/cache.py", "observation": "class Cache:\n    def get(self, key): ..."},
                {"role": "assistant", "tool_name": "search", "content": "search ttl", "observation": "tests/test_cache.py: test_ttl_expiry"},
                {"role": "assistant", "tool_name": "edit", "content": "patch ttl handling", "observation": "patch applied"},
                {"role": "assistant", "tool_name": "pytest", "content": "pytest tests/test_cache.py", "observation": "1 passed"},
            ],
            "generated_patch": "diff --git a/src/cache.py b/src/cache.py\n+ fix ttl",
            "eval_logs": "PASS",
        },
        {
            "instance_id": "mock__project-202",
            "model_name": "swe-agent-mixtral",
            "target": False,
            "exit_status": "error",
            "trajectory": [
                {"role": "system", "content": "Autonomous programmer command-line setting."},
                {"role": "assistant", "tool_name": "open", "content": "open server.py", "observation": "def handler(req):\n    raise RuntimeError('boom')"},
                {"role": "assistant", "tool_name": "pytest", "content": "run tests", "observation": "FAILED tests/test_server.py\nRuntimeError: boom"},
                {"role": "assistant", "tool_name": "edit", "content": "patch handler", "observation": "patch applied"},
                {"role": "assistant", "tool_name": "pytest", "content": "run tests", "observation": "FAILED tests/test_server.py\nAssertionError"},
            ],
            "generated_patch": "",
            "eval_logs": "FAIL",
        },
    ]


def _has_trajectory(row: dict[str, Any]) -> bool:
    traj = row.get("trajectory")
    return isinstance(traj, list) and len(traj) > 0


def load_swe_agent(
    sample_size: int = 5000,
    split: str = "train",
    streaming: bool = True,
    cache_dir: str | None = None,
    seed: int = 0,
    offline_mock: bool = False,
    sample_mode: bool = True,
) -> LoadResult:
    """Load a bounded sample of SWE-agent traces.

    ``sample_mode`` is kept explicit because this dataset can include very large
    patch/log fields.  The loader always bounds row count, and the normalizer
    truncates only for derived features rather than mutating raw rows.
    """

    warnings: list[str] = []
    if offline_mock:
        warnings.append("swe_agent: offline_mock requested; using bundled mock rows.")
        return LoadResult(DATASET_NAME, mock_swe_agent_rows()[:sample_size], warnings, used_mock=True)

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local env
        warnings.append(f"swe_agent: datasets import failed ({exc}); using mock rows.")
        return LoadResult(DATASET_NAME, mock_swe_agent_rows()[:sample_size], warnings, used_mock=True)

    rows: list[dict[str, Any]] = []
    skipped_empty = 0
    try:
        ds = load_dataset(DATASET_NAME, split=split, streaming=streaming, cache_dir=cache_dir)
        if streaming and hasattr(ds, "shuffle"):
            try:
                ds = ds.shuffle(seed=seed, buffer_size=max(500, min(sample_size * 10, 5000)))
            except Exception as exc:
                warnings.append(f"swe_agent: streaming shuffle failed ({exc}); using dataset order.")

        for row in ds:
            raw_row = dict(row)
            if not _has_trajectory(raw_row):
                skipped_empty += 1
                continue
            rows.append(raw_row)
            if len(rows) >= sample_size:
                break
            if sample_mode and len(rows) >= sample_size:
                break
    except Exception as exc:
        LOGGER.exception("SWE-agent loading failed")
        warnings.append(f"swe_agent: load failed ({exc}); using mock rows.")
        return LoadResult(DATASET_NAME, mock_swe_agent_rows()[:sample_size], warnings, used_mock=True)

    if skipped_empty:
        warnings.append(f"swe_agent: skipped {skipped_empty} rows with empty/missing trajectory.")
    if not rows:
        warnings.append("swe_agent: no rows loaded; using mock rows.")
        return LoadResult(DATASET_NAME, mock_swe_agent_rows()[:sample_size], warnings, used_mock=True)
    return LoadResult(DATASET_NAME, rows, warnings, used_mock=False)
