from __future__ import annotations

from dataclasses import dataclass, field

from waferagent.graph_ir import AgentGraph, NodeType
from waferagent.trace_schema import TraceRecord


@dataclass
class Stage:
    stage_id: str
    parent_node_id: str
    job_id: str
    stage_type: str
    deps: list[str]
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: float = 0.0
    tile_pool: str = "none"
    shared_prefix_ids: list[str] = field(default_factory=list)
    shared_prefix_token_len: int = 0
    kv_bytes_estimated: int = 0
    tool_latency_ms: float = 0.0


@dataclass
class StageSchedule:
    stage_id: str
    parent_node_id: str
    job_id: str
    baseline: str
    stage_type: str
    start_ms: float
    end_ms: float
    assigned_tiles: int
    sram_read_bytes: int
    sram_write_bytes: int
    mesh_bytes: int
    mesh_wait_ms: float
    queue_wait_ms: float
    stall_reason: str

    def to_dict(self) -> dict[str, str | int | float]:
        return self.__dict__.copy()


def build_stages(graph: AgentGraph, traces: list[TraceRecord]) -> dict[str, Stage]:
    trace_map = {tr.node_id: tr for tr in traces}
    producer_stage: dict[str, str] = {}
    stages: dict[str, Stage] = {}

    for node_id in graph.topological_order():
        node = graph.nodes[node_id]
        tr = trace_map[node_id]
        dep_stage_ids = [producer_stage[d] for d in node.deps if d in producer_stage]
        if node.node_type == NodeType.TOOL_CALL:
            sid = f"{node_id}.tool"
            stages[sid] = Stage(
                stage_id=sid,
                parent_node_id=node_id,
                job_id=node.job_id,
                stage_type="tool",
                deps=dep_stage_ids,
                duration_ms=tr.tool_latency_ms,
                tool_latency_ms=tr.tool_latency_ms,
            )
            producer_stage[node_id] = sid
            continue

        prefill = f"{node_id}.prefill"
        decode = f"{node_id}.decode"
        stages[prefill] = Stage(
            stage_id=prefill,
            parent_node_id=node_id,
            job_id=node.job_id,
            stage_type="prefill",
            deps=dep_stage_ids,
            input_tokens=tr.input_tokens,
            duration_ms=max(0.0, tr.ttft_ms),
            tile_pool="prefill",
            shared_prefix_ids=list(tr.shared_prefix_ids),
            shared_prefix_token_len=tr.shared_prefix_token_len,
            kv_bytes_estimated=tr.kv_bytes_estimated,
        )
        stages[decode] = Stage(
            stage_id=decode,
            parent_node_id=node_id,
            job_id=node.job_id,
            stage_type="decode",
            deps=[prefill],
            input_tokens=tr.input_tokens,
            output_tokens=tr.output_tokens,
            duration_ms=max(0.0, tr.decode_ms),
            tile_pool="decode",
            shared_prefix_ids=list(tr.shared_prefix_ids),
            shared_prefix_token_len=tr.shared_prefix_token_len,
            kv_bytes_estimated=tr.kv_bytes_estimated,
        )
        producer_stage[node_id] = decode
    return stages
