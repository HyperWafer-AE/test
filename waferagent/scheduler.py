from __future__ import annotations

import heapq
from dataclasses import dataclass

from waferagent.baselines import BaselineConfig
from waferagent.graph_ir import AgentGraph, NodeType
from waferagent.mesh import Mesh, MeshConfig
from waferagent.placement import Placement
from waferagent.trace_schema import TraceRecord


@dataclass
class ScheduleRecord:
    node_id: str
    job_id: str
    baseline: str
    start_ms: float
    end_ms: float
    assigned_tiles: int
    placement_region: str
    prefill_ms: float
    decode_ms: float
    comm_ms: float
    queue_wait_ms: float
    cache_hit: str
    tile_r: int
    tile_c: int

    def to_dict(self) -> dict[str, float | str | int]:
        return self.__dict__.copy()


def _duration(
    tr: TraceRecord,
    graph: AgentGraph,
    baseline: BaselineConfig,
    mesh_cfg: MeshConfig,
    prefix_seen: set[str],
) -> tuple[float, float, str]:
    if tr.node_type == NodeType.TOOL_CALL.value:
        tool = tr.tool_latency_ms
        if baseline.tool_ttl:
            tool *= 0.92
        return tool, 0.0, "not_applicable"
    prefill = max(0.0, tr.ttft_ms)
    decode = max(0.0, tr.decode_ms)
    cache_hit = "not_applicable"
    if baseline.kv_sharing and tr.shared_prefix_ids:
        prior = any(pid in prefix_seen for pid in tr.shared_prefix_ids)
        cache_hit = "hit" if prior else "miss"
        shared_ratio = tr.shared_prefix_token_len / tr.input_tokens if tr.input_tokens else 0.0
        if prior:
            prefill *= max(0.18, 1.0 - 0.85 * shared_ratio)
        for pid in tr.shared_prefix_ids:
            prefix_seen.add(pid)
    prefill *= mesh_cfg.h100_prefill_to_wafer_scale * baseline.prefill_time_multiplier
    decode *= mesh_cfg.h100_decode_to_wafer_scale * baseline.decode_time_multiplier
    if baseline.dynamic_pd_partition:
        if tr.input_tokens > max(1, tr.output_tokens * 8):
            prefill *= 0.88
        else:
            decode *= 0.90
    return prefill, decode, cache_hit


def _priority(graph: AgentGraph, node_id: str, traces: dict[str, TraceRecord], baseline: BaselineConfig) -> float:
    node = graph.nodes[node_id]
    tr = traces[node_id]
    if baseline.critical_path:
        return -(
            1000.0 * node.criticality
            + 2.0 * len(node.shared_prefix_ids)
            - 0.01 * tr.total_ms
            + 0.1 * node.fan_out
        )
    if baseline.scheduling_policy == "shortest_job_first":
        return tr.total_ms
    if baseline.scheduling_policy == "round_robin_agents":
        return hash(node.agent_id) % 1024
    if baseline.scheduling_policy == "kvflow_like_steps_to_execution":
        return -(len(node.shared_prefix_ids) * 10 + node.criticality)
    return float(graph.topological_order().index(node_id))


def schedule_graph(
    graph: AgentGraph,
    traces: list[TraceRecord],
    mesh: Mesh,
    placements: dict[str, Placement],
    baseline: BaselineConfig,
) -> list[ScheduleRecord]:
    graph.critical_path_lengths()
    trace_map = {t.node_id: t for t in traces}
    indegree = {node_id: len(graph.nodes[node_id].deps) for node_id in graph.nodes}
    children: dict[str, list[str]] = {node_id: [] for node_id in graph.nodes}
    for edge in graph.edges:
        children[edge.src].append(edge.dst)
    ready: list[tuple[float, str]] = []
    for node_id, deg in indegree.items():
        if deg == 0:
            heapq.heappush(ready, (_priority(graph, node_id, trace_map, baseline), node_id))

    total_tiles = mesh.config.total_tiles
    base_lanes = max(1, int(total_tiles / 512))
    lanes = max(1, int(base_lanes * baseline.parallelism_multiplier))
    if graph.workload == "debate":
        lanes = max(lanes, 2)
    lane_available = [0.0 for _ in range(lanes)]
    end_times: dict[str, float] = {}
    records: list[ScheduleRecord] = []
    prefix_seen: set[str] = set()

    while ready:
        _, node_id = heapq.heappop(ready)
        tr = trace_map[node_id]
        dep_ready = max((end_times[d] for d in graph.nodes[node_id].deps), default=0.0)
        lane_idx = min(range(lanes), key=lambda i: lane_available[i])
        lane_ready = lane_available[lane_idx]
        start = max(dep_ready, lane_ready)
        prefill_ms, decode_ms, cache_hit = _duration(tr, graph, baseline, mesh.config, prefix_seen)
        placement = placements[node_id]
        comm_ms = 0.0
        for dep in graph.nodes[node_id].deps:
            dep_place = placements[dep]
            dep_trace = trace_map[dep]
            bytes_moved = max(1, dep_trace.output_tokens) * 4
            if baseline.kv_sharing and dep_trace.shared_prefix_ids:
                bytes_moved += max(0, dep_trace.shared_prefix_token_len) * 16
            comm_ms += mesh.route(dep_place.tile, placement.tile, bytes_moved)
        comm_ms *= baseline.comm_time_multiplier
        if baseline.mesh_congestion_penalty:
            stats = mesh.stats()
            penalty = min(1.25, 1.0 + max(0.0, stats["mesh_hotspot_ratio"] - 2.0) * 0.02)
            comm_ms *= penalty
        duration = prefill_ms + decode_ms + comm_ms
        end = start + duration
        lane_available[lane_idx] = end
        end_times[node_id] = end
        records.append(
            ScheduleRecord(
                node_id=node_id,
                job_id=tr.job_id,
                baseline=baseline.name,
                start_ms=start,
                end_ms=end,
                assigned_tiles=placement.assigned_tiles,
                placement_region=placement.placement_region,
                prefill_ms=prefill_ms,
                decode_ms=decode_ms,
                comm_ms=comm_ms,
                queue_wait_ms=max(0.0, start - dep_ready),
                cache_hit=cache_hit,
                tile_r=placement.tile[0],
                tile_c=placement.tile[1],
            )
        )
        for child in children[node_id]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, (_priority(graph, child, trace_map, baseline), child))

    if len(records) != len(graph.nodes):
        raise RuntimeError("Scheduler failed to schedule all nodes")
    return records
