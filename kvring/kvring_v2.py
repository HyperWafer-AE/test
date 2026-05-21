"""KVRing v2 query-tiled parallel collective model."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple

from .accounting import ModeResult, TraceStats
from .config import Agent, Coord, HardwareConfig, ModelConfig, ShardGroup, WorkloadConfig
from .mesh import WaferMesh, default_agents, place_shard_groups
from .units import gib, tib


def _attention_ops(model: ModelConfig, query_count: int, prefix_tokens: int) -> float:
    return 4.0 * model.layers * model.query_heads * query_count * prefix_tokens * model.head_dim


def _tile_sources(agents: List[Agent], query_tile_size: int) -> List[Agent]:
    sources: List[Agent] = []
    for i in range(0, len(agents), query_tile_size):
        sources.append(agents[i])
    return sources


def _add_query_scatter(
    mesh: WaferMesh,
    stats: TraceStats,
    agents: List[Agent],
    shards: List[ShardGroup],
    query_tile_size: int,
    query_tile_bytes: int,
    decode_tokens: int,
) -> int:
    max_hops = 0
    sources = _tile_sources(agents, query_tile_size)
    for source in sources:
        for shard in shards:
            for region in shard.regions:
                hops = mesh.add_transfer(stats, source.position, region, query_tile_bytes * decode_tokens)
                max_hops = max(max_hops, hops)
    return max_hops


def _ring_reduce(
    mesh: WaferMesh,
    stats: TraceStats,
    agents: List[Agent],
    shards: List[ShardGroup],
    query_tile_size: int,
    partial_state_bytes: int,
    decode_tokens: int,
) -> Tuple[int, int]:
    max_hops = 0
    total_hops = 0
    roots = [s.home_region for s in shards]
    sources = _tile_sources(agents, query_tile_size)
    for source in sources:
        for i, root in enumerate(roots):
            nxt = roots[(i + 1) % len(roots)]
            hops = mesh.add_transfer(stats, root, nxt, partial_state_bytes * decode_tokens)
            max_hops = max(max_hops, hops)
            total_hops += hops * decode_tokens
        return_hops = mesh.add_transfer(stats, roots[0], source.position, partial_state_bytes * decode_tokens)
        max_hops = max(max_hops, return_hops)
        total_hops += return_hops * decode_tokens
    return max_hops, total_hops


def _full_ring_reduce(
    mesh: WaferMesh,
    stats: TraceStats,
    agents: List[Agent],
    query_tile_size: int,
    partial_state_bytes: int,
    decode_tokens: int,
) -> Tuple[int, int]:
    sources = _tile_sources(agents, query_tile_size)
    packets = len(sources) * decode_tokens
    mesh.add_ring_cycle_transfer(stats, packet_bytes=partial_state_bytes, packets=packets)
    return len(mesh.serpentine_cycle_edges()), len(mesh.serpentine_cycle_edges()) * packets


def _region_split_ring_reduce(
    mesh: WaferMesh,
    stats: TraceStats,
    agents: List[Agent],
    shards: List[ShardGroup],
    query_tile_size: int,
    partial_state_bytes: int,
    decode_tokens: int,
) -> Tuple[int, int]:
    max_hops = 0
    total_hops = 0
    sources = _tile_sources(agents, query_tile_size)
    roots = [s.home_region for s in shards]
    midpoint = len(roots) // 2 or 1
    groups = [roots[:midpoint], roots[midpoint:]]
    for idx, source in enumerate(sources):
        group = groups[idx % len(groups)] or roots
        for i, root in enumerate(group):
            nxt = group[(i + 1) % len(group)]
            hops = mesh.add_transfer(stats, root, nxt, partial_state_bytes * decode_tokens)
            max_hops = max(max_hops, hops)
            total_hops += hops * decode_tokens
        hops = mesh.add_transfer(stats, group[0], source.position, partial_state_bytes * decode_tokens)
        max_hops = max(max_hops, hops)
        total_hops += hops * decode_tokens
    return max_hops, total_hops


def _tree_reduce(
    mesh: WaferMesh,
    stats: TraceStats,
    agents: List[Agent],
    shards: List[ShardGroup],
    query_tile_size: int,
    partial_state_bytes: int,
    decode_tokens: int,
) -> Tuple[int, int]:
    max_hops = 0
    total_hops = 0
    sources = _tile_sources(agents, query_tile_size)
    for source in sources:
        nodes = [s.home_region for s in shards]
        while len(nodes) > 1:
            next_nodes: List[Coord] = []
            for i in range(0, len(nodes), 2):
                if i + 1 >= len(nodes):
                    next_nodes.append(nodes[i])
                    continue
                winner = nodes[i]
                loser = nodes[i + 1]
                hops = mesh.add_transfer(stats, loser, winner, partial_state_bytes * decode_tokens)
                max_hops = max(max_hops, hops)
                total_hops += hops * decode_tokens
                next_nodes.append(winner)
            nodes = next_nodes
        final_root = nodes[0]
        hops = mesh.add_transfer(stats, final_root, source.position, partial_state_bytes * decode_tokens)
        max_hops = max(max_hops, hops)
        total_hops += hops * decode_tokens
    return max_hops, total_hops


def simulate_kvring_v2(
    model: ModelConfig,
    workload: WorkloadConfig,
    hardware: HardwareConfig,
    *,
    query_tile_size: int = 8,
    num_shards: int = 8,
    reduction: str = "ring",
    placement: str = "serpentine",
    state_dtype: str = "fp32",
    enable_vc_model: bool = False,
    mesh: WaferMesh | None = None,
    agents: List[Agent] | None = None,
) -> ModeResult:
    reduction_alias = {
        "ring": "selected_ring",
        "selected_ring": "selected_ring",
        "selected_ring_reduce": "selected_ring",
        "tree": "binary_tree",
        "binary_tree": "binary_tree",
        "binary_tree_reduce": "binary_tree",
        "region_split_ring": "region_split_ring",
        "full_ring_v1_legacy": "full_ring_v1_legacy",
    }
    if reduction not in reduction_alias:
        raise ValueError("unsupported reduction topology")
    reduction = reduction_alias[reduction]
    if query_tile_size <= 0:
        raise ValueError("query_tile_size must be positive")

    mesh = mesh or WaferMesh(hardware.mesh_rows, hardware.mesh_cols)
    agents = agents or default_agents(workload.concurrent_agents, hardware.mesh_rows, hardware.mesh_cols)
    shared = workload.shared_kv_bytes(model)
    shards = place_shard_groups(mesh, shared, num_shards, hardware, placement=placement)

    tiles_per_step = workload.query_tiles_per_step(query_tile_size)
    num_query_tiles_total = workload.query_tiles_total(query_tile_size)
    query_tile_bytes = model.query_tile_bytes(query_tile_size)
    partial_state_bytes = model.partial_state_bytes(query_tile_size)
    packet_bytes = model.collective_packet_bytes(query_tile_size)

    query_stats = TraceStats()
    scatter_max_hops = _add_query_scatter(
        mesh,
        query_stats,
        agents,
        shards,
        query_tile_size,
        query_tile_bytes,
        workload.decode_tokens_per_agent,
    )
    reduction_stats = TraceStats()
    if reduction == "selected_ring":
        reduce_max_hops, reduction_hops = _ring_reduce(
            mesh,
            reduction_stats,
            agents,
            shards,
            query_tile_size,
            partial_state_bytes,
            workload.decode_tokens_per_agent,
        )
    elif reduction == "binary_tree":
        reduce_max_hops, reduction_hops = _tree_reduce(
            mesh,
            reduction_stats,
            agents,
            shards,
            query_tile_size,
            partial_state_bytes,
            workload.decode_tokens_per_agent,
        )
    elif reduction == "region_split_ring":
        reduce_max_hops, reduction_hops = _region_split_ring_reduce(
            mesh,
            reduction_stats,
            agents,
            shards,
            query_tile_size,
            partial_state_bytes,
            workload.decode_tokens_per_agent,
        )
    else:
        reduce_max_hops, reduction_hops = _full_ring_reduce(
            mesh,
            reduction_stats,
            agents,
            query_tile_size,
            partial_state_bytes,
            workload.decode_tokens_per_agent,
        )

    combined_stats = TraceStats()
    combined_stats.payload_bytes = query_stats.payload_bytes + reduction_stats.payload_bytes
    for stats in (query_stats, reduction_stats):
        for edge, load in stats.link_loads.items():
            combined_stats.link_loads[edge] += load

    region_sram: DefaultDict[Coord, float] = defaultdict(float)
    for shard in shards:
        for region in shard.regions:
            region_sram[region] += shard.bytes_per_region
    private_per_agent = workload.private_decode_kv_bytes_per_agent(model)
    for agent in agents:
        region_sram[agent.position] += private_per_agent

    local_sram_read_bytes = num_query_tiles_total * shared
    private_write_bytes = workload.private_write_bytes(model)
    private_suffix_tokens_total_per_agent = workload.decode_tokens_per_agent * (
        workload.decode_tokens_per_agent - 1
    ) // 2
    private_suffix_read_bytes = (
        workload.concurrent_agents * private_suffix_tokens_total_per_agent * model.kv_token_bytes
    )

    max_shard_region_bytes = max(s.bytes_per_region for s in shards)
    local_sram_read_bottleneck_bytes = num_query_tiles_total * max_shard_region_bytes
    tokens_per_shard = workload.shared_prefix_tokens / num_shards
    max_group = max(s.group_size for s in shards)
    shard_compute_ops_per_tile_per_region = _attention_ops(
        model, min(query_tile_size, workload.concurrent_agents), int(math.ceil(tokens_per_shard / max_group))
    )
    shard_compute_ops_total = shard_compute_ops_per_tile_per_region * num_query_tiles_total

    local_sram_cycles = math.ceil(local_sram_read_bottleneck_bytes / hardware.sram_bytes_per_cycle)
    local_compute_cycles = math.ceil(
        shard_compute_ops_total / (hardware.attention_compute_ops_per_s / hardware.clock_hz)
    )
    local_shard_compute_cycles = max(local_sram_cycles, local_compute_cycles)

    suffix_read_per_agent = private_suffix_tokens_total_per_agent * model.kv_token_bytes
    suffix_compute_ops_per_agent = _attention_ops(
        model, workload.decode_tokens_per_agent, max(1, workload.decode_tokens_per_agent // 2)
    )
    suffix_sram_cycles = math.ceil(suffix_read_per_agent / hardware.sram_bytes_per_cycle)
    suffix_compute_cycles = math.ceil(
        suffix_compute_ops_per_agent / (hardware.attention_compute_ops_per_s / hardware.clock_hz)
    )
    local_suffix_cycles = max(suffix_sram_cycles, suffix_compute_cycles)

    query_scatter_cycles = math.ceil(query_stats.max_link_load_bytes / hardware.link_bytes_per_cycle)
    query_scatter_cycles += scatter_max_hops * hardware.hop_latency_cycles
    reduction_cycles = math.ceil(reduction_stats.max_link_load_bytes / hardware.link_bytes_per_cycle)
    reduction_cycles += reduce_max_hops * hardware.hop_latency_cycles
    merge_bytes = num_query_tiles_total * partial_state_bytes * 2
    merge_cycles = max(1, math.ceil(merge_bytes / hardware.sram_bytes_per_cycle))

    serialized_cycles = (
        query_scatter_cycles + local_shard_compute_cycles + reduction_cycles + local_suffix_cycles + merge_cycles
    )
    throughput_bound_cycles = max(
        query_scatter_cycles, local_shard_compute_cycles, reduction_cycles, local_suffix_cycles
    ) + merge_cycles
    first_tile_compute_cycles = max(
        math.ceil(max_shard_region_bytes / hardware.sram_bytes_per_cycle),
        math.ceil(
            shard_compute_ops_per_tile_per_region
            / (hardware.attention_compute_ops_per_s / hardware.clock_hz)
        ),
    )
    critical_path_cycles = (
        scatter_max_hops * hardware.hop_latency_cycles
        + first_tile_compute_cycles
        + reduce_max_hops * hardware.hop_latency_cycles
        + max(1, merge_cycles // max(1, num_query_tiles_total))
    )
    estimated_cycles = throughput_bound_cycles
    sram_port_bytes = local_sram_read_bytes + private_suffix_read_bytes + private_write_bytes

    reduction_topology = {
        "selected_ring": "selected_ring_reduce",
        "binary_tree": "binary_tree_reduce",
        "region_split_ring": "region_split_ring",
        "full_ring_v1_legacy": "full_ring_v1_legacy",
    }[reduction]
    result = ModeResult(
        mode="KVRing-v2-query-tiled-parallel",
        description="KV-stationary, query-tiled shard-local attention with exact online-softmax reduction.",
        total_sram_bytes=float(sum(region_sram.values())),
        peak_region_sram_bytes=float(max(region_sram.values()) if region_sram else 0.0),
        payload_bytes=combined_stats.payload_bytes,
        total_wire_bytes=combined_stats.total_wire_bytes,
        max_link_load_bytes=combined_stats.max_link_load_bytes,
        mean_active_link_load_bytes=combined_stats.mean_active_link_load_bytes,
        hotspot_ratio=combined_stats.hotspot_ratio,
        mesh_seconds=(query_scatter_cycles + reduction_cycles) / hardware.clock_hz,
        compute_seconds=(local_shard_compute_cycles + local_suffix_cycles + merge_cycles)
        / hardware.clock_hz,
        estimated_latency_seconds=estimated_cycles / hardware.clock_hz,
        estimated_cycles=estimated_cycles,
        sram_port_bytes=sram_port_bytes,
        network_cycles=query_scatter_cycles + reduction_cycles,
        compute_cycles=local_shard_compute_cycles + local_suffix_cycles + merge_cycles,
        link_loads=dict(combined_stats.link_loads),
        region_sram_bytes=dict(region_sram),
        max_link=combined_stats.max_link,
        extra={
            "attention_scope": "attention_only_shared_prefix_plus_local_suffix",
            "layer_granularity": "metrics aggregate all layers; formulas preserve layers/heads explicitly",
            "layers": model.layers,
            "kv_heads": model.kv_heads,
            "query_heads": model.query_heads,
            "head_dim": model.head_dim,
            "hidden_dim": model.hidden_dim,
            "query_tile_size": query_tile_size,
            "num_query_tiles_per_step": tiles_per_step,
            "num_query_tiles_total": num_query_tiles_total,
            "num_shards": num_shards,
            "placement": placement,
            "reduction": reduction,
            "reduction_topology": reduction_topology,
            "num_reduction_steps": num_shards if "ring" in reduction else math.ceil(math.log2(max(1, num_shards))),
            "query_tile_bytes": query_tile_bytes,
            "partial_state_bytes": partial_state_bytes,
            "online_state_bytes": partial_state_bytes,
            "state_vector_dim": model.hidden_dim,
            "state_dtype_bytes": 4,
            "scalar_state_bytes": query_tile_size * model.query_heads * 2 * 4,
            "query_dtype_bytes": model.dtype_bytes,
            "packet_bytes": packet_bytes,
            "packet_model": "R*hidden_dim*dtype + FP32(m,l,z) online-softmax state",
            "local_sram_read_bytes": local_sram_read_bytes,
            "local_sram_read_tib": tib(local_sram_read_bytes),
            "private_suffix_kv_tokens": workload.modeled_private_suffix_tokens_per_agent,
            "private_suffix_read_bytes": private_suffix_read_bytes,
            "private_kv_write_bytes": private_write_bytes,
            "query_bytes_sent": query_stats.payload_bytes,
            "reduction_bytes": reduction_stats.payload_bytes,
            "reduction_wire_bytes": reduction_stats.total_wire_bytes,
            "reduction_byte_hops": reduction_stats.total_wire_bytes,
            "partial_reduce_byte_hops": reduction_stats.total_wire_bytes,
            "query_scatter_byte_hops": query_stats.total_wire_bytes,
            "result_return_byte_hops": 0.0,
            "reduction_edges": len(reduction_stats.link_loads),
            "reduction_hops": reduction_hops,
            "reduction_latency_s": reduction_cycles / hardware.clock_hz,
            "max_reduction_link_load_bytes": reduction_stats.max_link_load_bytes,
            "setup_cycles": reduce_max_hops * hardware.hop_latency_cycles,
            "steady_state_cycles": math.ceil(reduction_stats.max_link_load_bytes / hardware.link_bytes_per_cycle),
            "query_scatter_latency_s": query_scatter_cycles / hardware.clock_hz,
            "local_shard_compute_latency_s": local_shard_compute_cycles / hardware.clock_hz,
            "local_suffix_latency_s": local_suffix_cycles / hardware.clock_hz,
            "private_suffix_compute_latency_s": local_suffix_cycles / hardware.clock_hz,
            "merge_latency_s": merge_cycles / hardware.clock_hz,
            "final_online_softmax_merge_latency_s": merge_cycles / hardware.clock_hz,
            "serialized_latency_s": serialized_cycles / hardware.clock_hz,
            "throughput_bound_latency_s": throughput_bound_cycles / hardware.clock_hz,
            "critical_path_latency_s": critical_path_cycles / hardware.clock_hz,
            "estimated_attention_stage_latency_s": estimated_cycles / hardware.clock_hz,
            "estimated_decode_attention_latency_s": estimated_cycles / hardware.clock_hz,
            "overlap_model": "pipeline",
            "vc_model": hardware.vc_model,
            "state_dtype": state_dtype,
            "vc_model_enabled": enable_vc_model or hardware.enable_vc_model,
            "vc_forward_channel": "vc0" if (enable_vc_model or hardware.enable_vc_model) else "",
            "vc_return_channel": "vc1" if (enable_vc_model or hardware.enable_vc_model) else "",
            "potential_cycle_detected": False if (enable_vc_model or hardware.enable_vc_model) else "",
            "shard_group_size": [s.group_size for s in shards],
            "placement_unit": "region_group",
            "regions_per_shard": [s.group_size for s in shards],
            "shard_bytes": [s.total_bytes for s in shards],
            "shard_region_bytes": [s.bytes_per_region for s in shards],
            "shard_bytes_per_region": [s.bytes_per_region for s in shards],
            "peak_region_sram_bytes": max(region_sram.values()) if region_sram else 0.0,
            "region_capacity_violation": max(region_sram.values()) > hardware.region_capacity_bytes,
            "shard_groups": [
                {
                    "shard_id": s.shard_id,
                    "home_region": s.home_region,
                    "region_group": s.regions,
                    "shard_group_size": s.group_size,
                    "shard_region_bytes": s.bytes_per_region,
                }
                for s in shards
            ],
        },
    )
    return result
