"""Replication and centralization baselines."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import DefaultDict, List

from .accounting import ModeResult, TraceStats, result_from_stats
from .config import Agent, Coord, HardwareConfig, ModelConfig, WorkloadConfig
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
    query_tile_bytes = model.query_tile_bytes(query_tile_size)
    partial_bytes = model.partial_state_bytes(query_tile_size=query_tile_size)
    tiles_per_step = workload.query_tiles_per_step(query_tile_size)

    stats = TraceStats()
    max_hops = 0
    for i in range(0, len(agents), query_tile_size):
        source = agents[i]
        total_q = query_tile_bytes * workload.decode_tokens_per_agent
        total_out = partial_bytes * workload.decode_tokens_per_agent
        max_hops = max(max_hops, mesh.add_transfer(stats, source.position, home, total_q))
        max_hops = max(max_hops, mesh.add_transfer(stats, home, source.position, total_out))

    region_sram: DefaultDict[Coord, float] = defaultdict(float)
    region_sram[home] += shared
    for agent in agents:
        region_sram[agent.position] += private_per_agent

    central_sram_read_bytes = shared * tiles_per_step * workload.decode_tokens_per_agent
    central_compute_ops = _attention_ops(
        model,
        tiles_per_step * workload.decode_tokens_per_agent * query_tile_size,
        workload.shared_prefix_tokens,
    )
    sram_cycles = math.ceil(central_sram_read_bytes / hardware.sram_bytes_per_cycle)
    compute_cycles = math.ceil(central_compute_ops / (hardware.attention_compute_ops_per_s / hardware.clock_hz))
    central_queue_cycles = sram_cycles + compute_cycles
    network_cycles = math.ceil(stats.max_link_load_bytes / hardware.link_bytes_per_cycle)
    network_cycles += workload.decode_tokens_per_agent * max_hops * hardware.hop_latency_cycles
    total_cycles = network_cycles + central_queue_cycles
    sram_port_bytes = central_sram_read_bytes + private_write_bytes

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
            "query_tile_size": query_tile_size,
            "num_query_tiles_per_step": tiles_per_step,
            "num_query_tiles_total": tiles_per_step * workload.decode_tokens_per_agent,
            "central_sram_read_bytes": central_sram_read_bytes,
            "local_sram_read_bytes": central_sram_read_bytes,
            "central_compute_ops": central_compute_ops,
            "central_compute_bytes_or_ops": central_compute_ops,
            "central_compute_proxy_bytes": central_compute_ops,
            "central_query_payload_bytes": query_tile_bytes * tiles_per_step * workload.decode_tokens_per_agent,
            "central_result_payload_bytes": partial_bytes * tiles_per_step * workload.decode_tokens_per_agent,
            "query_bytes_sent": query_tile_bytes * tiles_per_step * workload.decode_tokens_per_agent,
            "partial_bytes_returned": partial_bytes * tiles_per_step * workload.decode_tokens_per_agent,
            "central_router_in_bytes": query_tile_bytes * tiles_per_step * workload.decode_tokens_per_agent,
            "central_router_out_bytes": partial_bytes * tiles_per_step * workload.decode_tokens_per_agent,
            "central_max_link_load_bytes": stats.max_link_load_bytes,
            "central_hotspot_ratio": stats.hotspot_ratio,
            "attention_stage_proxy_latency_s": total_cycles / hardware.clock_hz,
            "private_kv_write_bytes": private_write_bytes,
            "central_region_queue_time_s": central_queue_cycles / hardware.clock_hz,
            "central_sram_queue_time_s": sram_cycles / hardware.clock_hz,
            "central_compute_queue_time_s": compute_cycles / hardware.clock_hz,
            "private_kv_write_gib_total": gib(private_write_bytes),
            "packet_model": "query bytes in, exact FP32 online-softmax state out",
        },
    )
    return result
