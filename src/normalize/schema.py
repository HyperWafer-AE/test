"""Canonical tabular schema for heterogeneous agent traces.

The pipeline writes three CSV/Parquet-friendly tables:

* Trace: one row per attempted task/session.
* Step: one row per interaction step/message/tool observation.
* ObjectAccess: approximate state-object accesses inferred from steps.

All fields are intentionally simple Python scalar types so the tables can be
loaded by pandas, DuckDB, Spark, or a future trace replay system.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class Trace:
    trace_id: str
    dataset: str
    task_id: str | None = None
    model: str | None = None
    agent_or_harness: str | None = None
    success: bool | None = None
    reward: float | None = None
    resolved: bool | None = None
    duration_s: float | None = None
    input_tokens: float | None = None
    output_tokens: float | None = None
    cache_tokens: float | None = None
    total_steps: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Step:
    trace_id: str
    step_id: int
    role: str | None = None
    phase: str = "unknown"
    tool_name: str | None = None
    message_text: str | None = None
    message_tokens_est: int = 0
    tool_args_len: int = 0
    observation_text: str | None = None
    observation_len_chars: int = 0
    observation_tokens_est: int = 0
    error_flag: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ObjectAccess:
    trace_id: str
    step_id: int
    object_type: str
    object_id: str
    size_chars: int = 0
    access_type: str = "read"
    phase: str = "unknown"
    tool_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


TRACE_COLUMNS = [
    "trace_id",
    "dataset",
    "task_id",
    "model",
    "agent_or_harness",
    "success",
    "reward",
    "resolved",
    "duration_s",
    "input_tokens",
    "output_tokens",
    "cache_tokens",
    "total_steps",
]

STEP_COLUMNS = [
    "trace_id",
    "step_id",
    "role",
    "phase",
    "tool_name",
    "message_text",
    "message_tokens_est",
    "tool_args_len",
    "observation_text",
    "observation_len_chars",
    "observation_tokens_est",
    "error_flag",
]

OBJECT_COLUMNS = [
    "trace_id",
    "step_id",
    "object_type",
    "object_id",
    "size_chars",
    "access_type",
    "phase",
    "tool_name",
]
