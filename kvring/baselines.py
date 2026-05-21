"""Replication and centralization baselines."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import DefaultDict, List

from .accounting import ModeResult, TraceStats, result_from_stats
from .config import Agent, Coord, HardwareConfig, ModelConfig, WorkloadConfig, actual_query_tile_sizes
from .mesh import WaferMesh, central_home, default_agents
from .units import KiB, MiB, gib


def _attention_ops(model: ModelConfig, query_count: int, prefix_tokens: int) -> float:
    # QK and PV matvecs. This is an attention-only, scalar-op accounting model.
    return 4.0 * model.layers * model.query_heads * query_count * prefix_tokens * model.head_dim


def simulate_replicate_all(
    model: ModelConfig,
    workload: WorkloadConfig,
    hardware: HardwareConfig,
    mesh: WaferMesh | None = None,
    agents: List[Agent] | None = None,
) -> ModeResult:
    mesh = mesh or WaferMesh(hardware.mesh_rows, hardware.mesh_cols)
    agents = agents or default_agents(workload.concurrent_agents, hardware.mesh_rows, hardware.mesh_cols)
    home = central_home(hardware)
    shared = workload.shared_kv_bytes(model)
    private_per_agent = workload.private_decode_kv_bytes_per_agent(model)
    private_write_bytes = workload.private_write_bytes(model)

    stats = TraceStats()
    max_setup_hops = 0
    for agent in agents:
        max_setup_hops = max(max_setup_hops, mesh.add_transfer(stats, home, agent.position, shared))

    region_sram: DefaultDict[Coord, float] = defaultdict(float)
    for agent in agents:
        region_sram[agent.position] += shared + private_per_agent

    steady_state_local_read_bytes = shared * workload.decode_tokens_per_agent
    sram_port_bytes = steady_state_local_read_bytes + private_write_bytes
    region_capacity_violation = max(region_sram.values()) > hardware.region_capacity_bytes
    return result_from_stats(
        mode="Replicate-All",
        description="Full shared KV cache copied to every agent region; steady-state decode reads are local.",
        stats=stats,
        region_sram=region_sram,
        sram_port_bytes=sram_port_bytes,
        propagation_hops=max_setup_hops,
        link_bytes_per_cycle=hardware.link_bytes_per_cycle,
        sram_bytes_per_cycle=hardware.sram_bytes_per_cycle,
        clock_hz=hardware.clock_hz,
        hop_latency_cycles=hardware.hop_latency_cycles,
        extra={
            "setup_replication_payload_gib": gib(shared * len(agents)),
            "setup_replication_wire_tib": stats.total_wire_bytes / (1024**4),
            "steady_state_local_decode_read_gib": gib(steady_state_local_read_bytes),
            "local_sram_read_bytes": steady_state_local_read_bytes,
            "private_kv_write_bytes": private_write_bytes,
            "private_kv_write_gib_total": gib(private_write_bytes),
            "replicas": len(agents),
            "shared_kv_per_replica_gib": gib(shared),
            "private_decode_kv_per_agent_mib": private_per_agent / MiB,
            "setup_route_max_hops": max_setup_hops,
            "logical_decode_queries": workload.total_decode_steps,
            "query_tiles": workload.total_decode_steps,
            "latency_bound_used": "throughput_bound",
            "region_capacity_violation": region_capacity_violation,
            "valid_capacity": not region_capacity_violation,
            "region_capacity_bytes": hardware.region_capacity_bytes,
            "peak_region_sram_bytes": max(region_sram.values()),
            "capacity_violation_reason": (
                f"peak region SRAM {max(region_sram.values())} exceeds capacity {hardware.region_capacity_bytes}"
                if region_capacity_violation
                else ""
            ),
        },
    )


def simulate_pull_kv_independent(
    model: ModelConfig,
    workload: WorkloadConfig,
    hardware: HardwareConfig,
    mesh: WaferMesh | None = None,
    agents: List[Agent] | None = None,
) -> ModeResult:
    mesh = mesh or WaferMesh(hardware.mesh_rows, hardware.mesh_cols)
    agents = agents or default_agents(workload.concurrent_agents, hardware.mesh_rows, hardware.mesh_cols)
    home = central_home(hardware)
    shared = workload.shared_kv_bytes(model)
    private_per_agent = workload.private_decode_kv_bytes_per_agent(model)
    private_write_bytes = workload.private_write_bytes(model)

    stats = TraceStats()
    max_decode_hops = 0
    request_bytes = model.query_bytes
    response_bytes = model.query_bytes
    for agent in agents:
        bytes_to_agent = shared * workload.decode_tokens_per_agent
        max_decode_hops = max(max_decode_hops, mesh.add_transfer(stats, home, agent.position, bytes_to_agent))
        mesh.add_transfer(stats, agent.position, home, request_bytes * workload.decode_tokens_per_agent)
        mesh.add_transfer(stats, home, agent.position, response_bytes * workload.decode_tokens_per_agent)

    region_sram: DefaultDict[Coord, float] = defaultdict(float)
    region_sram[home] += shared
    for agent in agents:
        region_sram[agent.position] += private_per_agent

    sram_port_bytes = shared * workload.total_decode_steps + private_write_bytes
    region_capacity_violation = max(region_sram.values()) > hardware.region_capacity_bytes
    return result_from_stats(
        mode="Pull-KV-Independent",
        description="One shared KV home; each agent remotely pulls the full shared KV every decode step.",
        stats=stats,
        region_sram=region_sram,
        sram_port_bytes=sram_port_bytes,
        propagation_hops=workload.decode_tokens_per_agent * max_decode_hops,
        link_bytes_per_cycle=hardware.link_bytes_per_cycle,
        sram_bytes_per_cycle=hardware.sram_bytes_per_cycle,
        clock_hz=hardware.clock_hz,
        hop_latency_cycles=hardware.hop_latency_cycles,
        extra={
            "home_region": home,
            "old_mode_alias": "Single-Home",
            "central_sram_read_bytes": shared * workload.total_decode_steps,
            "local_sram_read_bytes": shared * workload.total_decode_steps,
            "remote_read_gib_per_agent": gib(shared * workload.decode_tokens_per_agent),
            "query_bytes_sent": request_bytes * workload.total_decode_steps,
            "partial_bytes_returned": response_bytes * workload.total_decode_steps,
            "tiny_query_request_kib_per_agent": request_bytes * workload.decode_tokens_per_agent / KiB,
            "private_kv_write_bytes": private_write_bytes,
            "private_kv_write_gib_total": gib(private_write_bytes),
            "decode_route_max_hops": max_decode_hops,
            "logical_decode_queries": workload.total_decode_steps,
            "query_tiles": workload.total_decode_steps,
            "remote_kv_pull_bytes": shared * workload.total_decode_steps,
            "remote_kv_pull_wire_bytes": stats.total_wire_bytes,
            "home_sram_bytes": shared,
            "latency_bound_used": "throughput_bound",
            "region_capacity_violation": region_capacity_violation,
            "valid_capacity": not region_capacity_violation,
            "region_capacity_bytes": hardware.region_capacity_bytes,
            "peak_region_sram_bytes": max(region_sram.values()),
            "capacity_violation_reason": (
                f"peak region SRAM {max(region_sram.values())} exceeds capacity {hardware.region_capacity_bytes}"
                if region_capacity_violation
                else ""
            ),
        },
    )


def simulate_central_kv_stationary(
    model: ModelConfig,
    workload: WorkloadConfig,
    hardware: HardwareConfig,
    mesh: WaferMesh | None = None,
    agents: List[Agent] | None = None,
    query_tile_size: int = 1,
) -> ModeResult:
    mesh = mesh or WaferMesh(hardware.mesh_rows, hardware.mesh_cols)
    agents = agents or default_agents(workload.concurrent_agents, hardware.mesh_rows, hardware.mesh_cols)
    home = central_home(hardware)
    shared = workload.shared_kv_bytes(model)
    private_per_agent = workload.private_decode_kv_bytes_per_agent(model)
    private_write_bytes = workload.private_write_bytes(model)
    query_tile_size = max(1, query_tile_size)
    tile_sizes = actual_query_tile_sizes(len(agents), query_tile_size)
    query_payload_per_step = sum(model.query_tile_bytes(tile) for tile in tile_sizes)
    partial_payload_per_step = sum(model.partial_state_bytes(query_tile_size=tile) for tile in tile_sizes)
    max_query_tile_bytes = max((model.query_tile_bytes(tile) for tile in tile_sizes), default=0)
    max_partial_bytes = max(
        (model.partial_state_bytes(query_tile_size=tile) for tile in tile_sizes), default=0
    )
    tiles_per_step = len(tile_sizes)

    query_stats = TraceStats()
    return_stats = TraceStats()
    query_max_hops = 0
    return_max_hops = 0
    offset = 0
    for tile in tile_sizes:
        source = agents[offset]
        total_q = model.query_tile_bytes(tile) * workload.decode_tokens_per_agent
        total_out = model.partial_state_bytes(query_tile_size=tile) * workload.decode_tokens_per_agent
        query_max_hops = max(query_max_hops, mesh.add_transfer(query_stats, source.position, home, total_q))
        return_max_hops = max(return_max_hops, mesh.add_transfer(return_stats, home, source.position, total_out))
        offset += tile

    stats = TraceStats()
    stats.payload_bytes = query_stats.payload_bytes + return_stats.payload_bytes
    for partial_stats in (query_stats, return_stats):
        for edge, load in partial_stats.link_loads.items():
            stats.link_loads[edge] += load

    region_sram: DefaultDict[Coord, float] = defaultdict(float)
    region_sram[home] += shared
    for agent in agents:
        region_sram[agent.position] += private_per_agent

    central_sram_read_bytes = shared * tiles_per_step * workload.decode_tokens_per_agent
    central_compute_ops = _attention_ops(model, workload.total_decode_steps, workload.shared_prefix_tokens)
    sram_cycles = math.ceil(central_sram_read_bytes / hardware.sram_bytes_per_cycle)
    compute_cycles = math.ceil(central_compute_ops / (hardware.attention_compute_ops_per_s / hardware.clock_hz))
    central_queue_cycles = sram_cycles + compute_cycles
    query_mesh_cycles = math.ceil(query_stats.max_link_load_bytes / hardware.link_bytes_per_cycle)
    query_mesh_cycles += workload.decode_tokens_per_agent * query_max_hops * hardware.hop_latency_cycles
    return_mesh_cycles = math.ceil(return_stats.max_link_load_bytes / hardware.link_bytes_per_cycle)
    return_mesh_cycles += workload.decode_tokens_per_agent * return_max_hops * hardware.hop_latency_cycles
    network_cycles = query_mesh_cycles + return_mesh_cycles
    serialized_cycles = network_cycles + central_queue_cycles
    throughput_bound_cycles = max(query_mesh_cycles, central_queue_cycles, return_mesh_cycles)
    critical_path_cycles = query_max_hops * hardware.hop_latency_cycles + sram_cycles + compute_cycles + return_max_hops * hardware.hop_latency_cycles
    sram_port_bytes = central_sram_read_bytes + private_write_bytes
    region_capacity_violation = max(region_sram.values()) > hardware.region_capacity_bytes
    component_cycles = {
        "query_mesh": query_mesh_cycles,
        "result_mesh": return_mesh_cycles,
        "sram_read": sram_cycles,
        "compute": compute_cycles,
        "queue": central_queue_cycles,
    }
    central_bottleneck_component = max(component_cycles.items(), key=lambda kv: kv[1])[0]

    result = result_from_stats(
        mode="Central-KV-Stationary",
        description="Shared KV stays at the center; agents send small queries and receive attention states.",
        stats=stats,
        region_sram=region_sram,
        sram_port_bytes=sram_port_bytes,
        propagation_hops=0,
        link_bytes_per_cycle=hardware.link_bytes_per_cycle,
        sram_bytes_per_cycle=hardware.sram_bytes_per_cycle,
        clock_hz=hardware.clock_hz,
        hop_latency_cycles=hardware.hop_latency_cycles,
        network_cycles_override=network_cycles,
        compute_cycles_override=central_queue_cycles,
        extra={
            "home_region": home,
            "requested_query_tile_size": query_tile_size,
            "query_tile_size": query_tile_size,
            "actual_query_tile_sizes": tile_sizes,
            "num_query_tiles_per_step": tiles_per_step,
            "num_query_tiles_total": tiles_per_step * workload.decode_tokens_per_agent,
            "logical_decode_queries": workload.total_decode_steps,
            "query_tiles": tiles_per_step * workload.decode_tokens_per_agent,
            "central_sram_read_bytes": central_sram_read_bytes,
            "local_sram_read_bytes": central_sram_read_bytes,
            "central_compute_ops": central_compute_ops,
            "central_compute_bytes_or_ops": central_compute_ops,
            "central_compute_proxy_bytes": central_compute_ops,
            "central_query_payload_bytes": query_payload_per_step * workload.decode_tokens_per_agent,
            "central_result_payload_bytes": partial_payload_per_step * workload.decode_tokens_per_agent,
            "central_query_wire_bytes": query_stats.total_wire_bytes,
            "central_result_wire_bytes": return_stats.total_wire_bytes,
            "query_bytes_sent": query_payload_per_step * workload.decode_tokens_per_agent,
            "partial_bytes_returned": partial_payload_per_step * workload.decode_tokens_per_agent,
            "central_router_in_bytes": query_payload_per_step * workload.decode_tokens_per_agent,
            "central_router_out_bytes": partial_payload_per_step * workload.decode_tokens_per_agent,
            "central_max_link_load_bytes": stats.max_link_load_bytes,
            "central_max_directed_link_load_bytes": stats.max_link_load_bytes,
            "central_hotspot_ratio": stats.hotspot_ratio,
            "serialized_latency_s": serialized_cycles / hardware.clock_hz,
            "throughput_bound_latency_s": throughput_bound_cycles / hardware.clock_hz,
            "critical_path_latency_s": critical_path_cycles / hardware.clock_hz,
            "attention_stage_proxy_latency_s": throughput_bound_cycles / hardware.clock_hz,
            "latency_bound_used": "throughput_bound",
            "network_latency_s": network_cycles / hardware.clock_hz,
            "sram_latency_s": sram_cycles / hardware.clock_hz,
            "compute_latency_s": compute_cycles / hardware.clock_hz,
            "merge_latency_s": 0.0,
            "query_scatter_latency_s": query_mesh_cycles / hardware.clock_hz,
            "reduction_latency_s": 0.0,
            "local_suffix_latency_s": 0.0,
            "private_kv_write_bytes": private_write_bytes,
            "central_region_queue_time_s": central_queue_cycles / hardware.clock_hz,
            "central_query_mesh_latency_s": query_mesh_cycles / hardware.clock_hz,
            "central_sram_read_latency_s": sram_cycles / hardware.clock_hz,
            "central_compute_latency_s": compute_cycles / hardware.clock_hz,
            "central_return_mesh_latency_s": return_mesh_cycles / hardware.clock_hz,
            "central_result_mesh_latency_s": return_mesh_cycles / hardware.clock_hz,
            "central_queue_latency_s": central_queue_cycles / hardware.clock_hz,
            "central_total_latency_s": throughput_bound_cycles / hardware.clock_hz,
            "central_bottleneck_component": central_bottleneck_component,
            "central_sram_queue_time_s": sram_cycles / hardware.clock_hz,
            "central_compute_queue_time_s": compute_cycles / hardware.clock_hz,
            "private_kv_write_gib_total": gib(private_write_bytes),
            "packet_model": "query bytes in, exact FP32 online-softmax state out",
            "query_tile_bytes": max_query_tile_bytes,
            "partial_state_bytes": max_partial_bytes,
            "query_tile_payload_bytes_per_step": query_payload_per_step,
            "partial_state_payload_bytes_per_step": partial_payload_per_step,
            "region_capacity_violation": region_capacity_violation,
            "valid_capacity": not region_capacity_violation,
            "region_capacity_bytes": hardware.region_capacity_bytes,
            "peak_region_sram_bytes": max(region_sram.values()),
            "capacity_violation_reason": (
                f"peak region SRAM {max(region_sram.values())} exceeds capacity {hardware.region_capacity_bytes}"
                if region_capacity_violation
                else ""
            ),
        },
    )
    return result
