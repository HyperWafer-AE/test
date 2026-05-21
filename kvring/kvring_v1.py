"""Legacy KVRing v1 sequential/pipelined ring baseline."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import DefaultDict, List

from .accounting import ModeResult, TraceStats, result_from_stats
from .config import Agent, Coord, HardwareConfig, ModelConfig, WorkloadConfig
from .mesh import WaferMesh, default_agents, place_shard_groups
from .units import gib


def simulate_kvring_v1(
    model: ModelConfig,
    workload: WorkloadConfig,
    hardware: HardwareConfig,
    ring_shards: int = 8,
    mesh: WaferMesh | None = None,
    agents: List[Agent] | None = None,
) -> ModeResult:
    mesh = mesh or WaferMesh(hardware.mesh_rows, hardware.mesh_cols)
    agents = agents or default_agents(workload.concurrent_agents, hardware.mesh_rows, hardware.mesh_cols)
    shared = workload.shared_kv_bytes(model)
    private_per_agent = workload.private_decode_kv_bytes_per_agent(model)
    private_write_bytes = workload.private_write_bytes(model)
    packets = workload.total_decode_steps
    packet_bytes = model.legacy_ring_packet_bytes
    ring_edges = mesh.serpentine_cycle_edges()
    shards = place_shard_groups(mesh, shared, ring_shards, hardware, placement="serpentine")

    stats = TraceStats()
    mesh.add_ring_cycle_transfer(stats, packet_bytes=packet_bytes, packets=packets)

    region_sram: DefaultDict[Coord, float] = defaultdict(float)
    for shard in shards:
        for region in shard.regions:
            region_sram[region] += shard.bytes_per_region
    for agent in agents:
        region_sram[agent.position] += private_per_agent

    sram_port_bytes = max(shard.bytes_per_region for shard in shards) * packets + private_write_bytes
    ideal_throughput = min(1.0, hardware.link_bytes_per_cycle / packet_bytes)
    effective_throughput = ideal_throughput / hardware.ring_congestion_bubble_factor
    ring_setup_hops = len(ring_edges)
    steady_cycles = int(math.ceil(packets / effective_throughput))
    setup_cycles = ring_setup_hops * hardware.hop_latency_cycles
    network_cycles = steady_cycles + setup_cycles

    return result_from_stats(
        mode="KVRing-v1-sequential-pipeline",
        description="Legacy full-serpentine ring: one small packet per agent decode step circulates around all regions.",
        stats=stats,
        region_sram=region_sram,
        sram_port_bytes=sram_port_bytes,
        propagation_hops=ring_setup_hops,
        link_bytes_per_cycle=hardware.link_bytes_per_cycle,
        sram_bytes_per_cycle=hardware.sram_bytes_per_cycle,
        clock_hz=hardware.clock_hz,
        hop_latency_cycles=hardware.hop_latency_cycles,
        network_cycles_override=network_cycles,
        extra={
            "legacy_behavior_preserved": True,
            "ring_shards": len(shards),
            "ring_edges": len(ring_edges),
            "ring_setup_hops": ring_setup_hops,
            "ring_packet_kib": packet_bytes / 1024,
            "packet_model": "legacy fixed whole-query + aggregated online state",
            "total_packets": packets,
            "local_sram_read_bytes": max(shard.bytes_per_region for shard in shards) * packets,
            "private_kv_write_bytes": private_write_bytes,
            "private_kv_write_gib_total": gib(private_write_bytes),
            "ring_congestion_bubble_factor": hardware.ring_congestion_bubble_factor,
            "ideal_pipeline_throughput_packets_per_cycle": ideal_throughput,
            "effective_pipeline_throughput_packets_per_cycle": effective_throughput,
            "ring_steady_state_cycles": steady_cycles,
            "ring_setup_cycles": setup_cycles,
            "network_formula": "ceil(total_packets / (min(1, link_bytes_per_cycle / packet_bytes) / bubble_factor)) + ring_setup_hops * hop_latency_cycles",
            "shard_group_size": [s.group_size for s in shards],
            "shard_region_bytes": [s.bytes_per_region for s in shards],
            "region_capacity_violation": max(region_sram.values()) > hardware.region_capacity_bytes,
            "shard_positions": [
                {"id": s.shard_id, "home_region": s.home_region, "region_group": s.regions}
                for s in shards
            ],
        },
    )
