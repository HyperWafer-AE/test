from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TraceRecord:
    schema_version: str
    run_id: str
    job_id: str
    workload: str
    node_id: str
    node_type: str
    agent_id: str
    role: str
    round_id: int
    deps: list[str]
    model_name: str
    model_path: str
    engine: str
    gpu_id: int | None
    input_tokens: int
    output_tokens: int
    shared_prefix_ids: list[str]
    private_prefix_ids: list[str]
    prompt_hash: str
    start_time_unix: float
    end_time_unix: float
    ttft_ms: float
    decode_ms: float
    total_ms: float
    tool_latency_ms: float
    kv_bytes_estimated: int
    cache_hit_tag: str
    scheduler_tag: str
    output_hash: str
    quality_proxy: float | None = None
    shared_prefix_token_len: int = 0
    private_prefix_token_len: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceRecord":
        return cls(**data)


def write_traces(path: str | Path, traces: list[TraceRecord]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for tr in traces:
            f.write(json.dumps(tr.to_dict(), sort_keys=True) + "\n")


def read_traces(path: str | Path) -> list[TraceRecord]:
    traces = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traces.append(TraceRecord.from_dict(json.loads(line)))
    return traces
