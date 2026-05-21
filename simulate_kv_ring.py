#!/usr/bin/env python3
"""
Trace-driven wafer-scale simulator for the KV Replication-Hotspot Dilemma.

The simulator compares three mappings for multi-agent long-context decoding:

1. Replicate-All: copy the full shared KV cache to every agent region.
2. Single-Home: keep the shared KV cache in one central home region and serve
   remote reads on every decode step.
3. KVRing: shard the shared KV cache across a spatial serpentine ring; packets
   carry only the query and online-softmax reduction state while the KV remains
   stationary.

Traffic accounting:
  * payload_bytes: logical data injected by producers.
  * total_wire_bytes: bytes accumulated over every directed physical mesh
    channel. This is byte-hop traffic and is the better proxy for NoC energy
    and contention.
  * max_link_load_bytes: total bytes crossing the hottest unidirectional
    physical channel. A->B and B->A are independent NoC resources.

The code intentionally uses a compact trace aggregation model: repeated decode
steps are folded into weighted transfer events, while physical routing and link
loads remain exact for the selected deterministic routes.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Mapping, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


KiB = 1024
MiB = 1024**2
GiB = 1024**3
TiB = 1024**4
GB_DEC = 10**9
TB_DEC = 10**12

Coord = Tuple[int, int]
Edge = Tuple[Coord, Coord]


def gib(x: float) -> float:
    return x / GiB


def tib(x: float) -> float:
    return x / TiB


def fmt_bytes(x: float) -> str:
    ax = abs(x)
    if ax >= TiB:
        return f"{x / TiB:.3f} TiB"
    if ax >= GiB:
        return f"{x / GiB:.3f} GiB"
    if ax >= MiB:
        return f"{x / MiB:.3f} MiB"
    if ax >= KiB:
        return f"{x / KiB:.3f} KiB"
    return f"{x:.0f} B"


def fmt_seconds(x: float) -> str:
    if x >= 1.0:
        return f"{x:.6f} s"
    if x >= 1e-3:
        return f"{x * 1e3:.6f} ms"
    if x >= 1e-6:
        return f"{x * 1e6:.6f} us"
    return f"{x * 1e9:.6f} ns"


def edge_key(edge: Edge) -> str:
    (r0, c0), (r1, c1) = edge
    return f"({r0},{c0})->({r1},{c1})"


@dataclass(frozen=True)
class ModelConfig:
    model_path: str = "/data1/duzc/model/model/LLM-Research/Meta-Llama-3___1-8B-Instruct"
    hidden_dim: int = 4096
    layers: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    dtype_bytes: int = 2

    @property
    def kv_token_bytes(self) -> int:
        # K and V for every layer, every GQA KV head, BF16/FP16.
        return 2 * self.layers * self.kv_heads * self.head_dim * self.dtype_bytes

    @property
    def query_bytes(self) -> int:
        return self.hidden_dim * self.dtype_bytes

    @property
    def online_scalar_bytes(self) -> int:
        # Per-layer/per-KV-head max and sum scalars, stored as FP32.
        return self.layers * self.kv_heads * 2 * 4

    @property
    def online_state_bytes(self) -> int:
        # sum(exp) * V is represented as one hidden-size vector plus scalars.
        return self.query_bytes + self.online_scalar_bytes

    @property
    def ring_packet_bytes(self) -> int:
        return self.query_bytes + self.online_state_bytes


@dataclass(frozen=True)
class WorkloadConfig:
    shared_prefix_tokens: int = 32768
    concurrent_agents: int = 8
    decode_tokens_per_agent: int = 256

    def shared_kv_bytes(self, model: ModelConfig) -> int:
        return self.shared_prefix_tokens * model.kv_token_bytes

    def private_decode_kv_bytes_per_agent(self, model: ModelConfig) -> int:
        return self.decode_tokens_per_agent * model.kv_token_bytes

    @property
    def total_decode_steps(self) -> int:
        return self.concurrent_agents * self.decode_tokens_per_agent


@dataclass(frozen=True)
class HardwareConfig:
    mesh_rows: int = 16
    mesh_cols: int = 16
    link_bandwidth_gbps: float = 100.0
    region_sram_bandwidth_tibps: float = 20.0
    clock_hz: float = 1.0e9
    hop_latency_cycles: int = 5
    ring_congestion_bubble_factor: float = 1.15

    @property
    def link_bandwidth_bytes_per_s(self) -> float:
        # Effective link bandwidth, interpreted as decimal GB/s.
        return self.link_bandwidth_gbps * GB_DEC

    @property
    def sram_bandwidth_bytes_per_s(self) -> float:
        return self.region_sram_bandwidth_tibps * TiB

    @property
    def link_bytes_per_cycle(self) -> float:
        return self.link_bandwidth_bytes_per_s / self.clock_hz

    @property
    def sram_bytes_per_cycle(self) -> float:
        return self.sram_bandwidth_bytes_per_s / self.clock_hz


@dataclass(frozen=True)
class Agent:
    agent_id: int
    position: Coord


@dataclass(frozen=True)
class KVShard:
    shard_id: int
    position: Coord
    bytes: int
    ring_index: int


@dataclass
class TraceStats:
    payload_bytes: float = 0.0
    link_loads: DefaultDict[Edge, float] = field(default_factory=lambda: defaultdict(float))

    def add_payload(self, bytes_: float) -> None:
        self.payload_bytes += bytes_

    @property
    def total_wire_bytes(self) -> float:
        return float(sum(self.link_loads.values()))

    @property
    def active_link_loads(self) -> List[float]:
        return [v for v in self.link_loads.values() if v > 0.0]


@dataclass
class ModeResult:
    mode: str
    description: str
    total_sram_bytes: float
    peak_region_sram_bytes: float
    payload_bytes: float
    total_wire_bytes: float
    max_link_load_bytes: float
    mean_active_link_load_bytes: float
    hotspot_ratio: float
    mesh_seconds: float
    compute_seconds: float
    estimated_latency_seconds: float
    estimated_cycles: int
    sram_port_bytes: float
    network_cycles: int
    compute_cycles: int
    link_loads: Mapping[Edge, float] = field(repr=False)
    region_sram_bytes: Mapping[Coord, float] = field(repr=False)
    max_link: Optional[Edge] = None
    extra: Dict[str, object] = field(default_factory=dict)

    def top_links(self, n: int = 10) -> List[Dict[str, object]]:
        items = sorted(self.link_loads.items(), key=lambda kv: kv[1], reverse=True)
        return [
            {
                "edge": edge_key(edge),
                "bytes": load,
                "gib": gib(load),
            }
            for edge, load in items[:n]
        ]

    def to_summary_dict(self) -> Dict[str, object]:
        return {
            "mode": self.mode,
            "total_sram_gib": gib(self.total_sram_bytes),
            "peak_region_sram_gib": gib(self.peak_region_sram_bytes),
            "payload_gib": gib(self.payload_bytes),
            "total_wire_tib": tib(self.total_wire_bytes),
            "max_link_load_gib": gib(self.max_link_load_bytes),
            "hotspot_ratio_max_over_mean_active": self.hotspot_ratio,
            "mesh_time_s": self.mesh_seconds,
            "compute_time_s": self.compute_seconds,
            "estimated_latency_s": self.estimated_latency_seconds,
            "estimated_cycles": self.estimated_cycles,
            "sram_port_gib": gib(self.sram_port_bytes),
            "network_cycles": self.network_cycles,
            "compute_cycles": self.compute_cycles,
            "max_link": edge_key(self.max_link) if self.max_link else None,
            "max_directed_link": edge_key(self.max_link) if self.max_link else None,
            "extra": self.extra,
        }

    def to_full_dict(self) -> Dict[str, object]:
        data = self.to_summary_dict()
        data["description"] = self.description
        data["top_links"] = self.top_links(16)
        data["region_sram_gib"] = {
            f"({r},{c})": gib(v)
            for (r, c), v in sorted(self.region_sram_bytes.items(), key=lambda kv: kv[0])
            if v > 0.0
        }
        data["link_load_gib"] = {
            edge_key(edge): gib(load)
            for edge, load in sorted(self.link_loads.items(), key=lambda kv: edge_key(kv[0]))
            if load > 0.0
        }
        return data


class WaferMesh:
    def __init__(self, rows: int, cols: int) -> None:
        if rows < 2 or cols < 2:
            raise ValueError("mesh must be at least 2x2")
        self.rows = rows
        self.cols = cols

    def validate(self, coord: Coord) -> None:
        r, c = coord
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise ValueError(f"coordinate {coord} is outside {self.rows}x{self.cols} mesh")

    def xy_route(self, src: Coord, dst: Coord) -> List[Edge]:
        self.validate(src)
        self.validate(dst)
        r, c = src
        dr, dc = dst
        edges: List[Edge] = []

        step_r = 1 if dr >= r else -1
        while r != dr:
            nr = r + step_r
            a, b = (r, c), (nr, c)
            edges.append((a, b))
            r = nr

        step_c = 1 if dc >= c else -1
        while c != dc:
            nc = c + step_c
            a, b = (r, c), (r, nc)
            edges.append((a, b))
            c = nc

        return edges

    def route_hops(self, src: Coord, dst: Coord) -> int:
        return abs(src[0] - dst[0]) + abs(src[1] - dst[1])

    def add_transfer(self, stats: TraceStats, src: Coord, dst: Coord, bytes_: float) -> None:
        stats.add_payload(bytes_)
        for edge in self.xy_route(src, dst):
            stats.link_loads[edge] += bytes_

    def serpentine_cycle_nodes(self) -> List[Coord]:
        """Hamiltonian serpentine cycle over all mesh nodes.

        The first column is reserved as the return rail. The path visits the
        top row, snakes through columns 1..cols-1, then returns upward through
        column 0. This gives one simple cycle with exactly rows*cols edges.
        """
        nodes: List[Coord] = []
        nodes.extend((0, c) for c in range(self.cols))

        for r in range(1, self.rows):
            if r % 2 == 1:
                nodes.extend((r, c) for c in range(self.cols - 1, 0, -1))
            else:
                nodes.extend((r, c) for c in range(1, self.cols))

        nodes.append((self.rows - 1, 0))
        nodes.extend((r, 0) for r in range(self.rows - 2, 0, -1))

        if len(nodes) != self.rows * self.cols:
            raise RuntimeError("serpentine cycle did not cover every mesh node")
        if len(set(nodes)) != len(nodes):
            raise RuntimeError("serpentine cycle has duplicate nodes")
        return nodes

    def serpentine_cycle_edges(self) -> List[Edge]:
        nodes = self.serpentine_cycle_nodes()
        edges = [
            (nodes[i], nodes[(i + 1) % len(nodes)])
            for i in range(len(nodes))
        ]
        if len(set(edges)) != len(edges):
            raise RuntimeError("serpentine cycle reused a directed physical channel")
        return edges

    def add_ring_cycle_transfer(self, stats: TraceStats, packet_bytes: float, packets: int) -> None:
        total_payload = packet_bytes * packets
        stats.add_payload(total_payload)
        per_edge = packet_bytes * packets
        for edge in self.serpentine_cycle_edges():
            stats.link_loads[edge] += per_edge

    def all_edges(self) -> Iterable[Edge]:
        for r in range(self.rows):
            for c in range(self.cols - 1):
                yield ((r, c), (r, c + 1))
                yield ((r, c + 1), (r, c))
        for r in range(self.rows - 1):
            for c in range(self.cols):
                yield ((r, c), (r + 1, c))
                yield ((r + 1, c), (r, c))


class Simulator:
    def __init__(
        self,
        model: ModelConfig,
        workload: WorkloadConfig,
        hardware: HardwareConfig,
        ring_shards: int,
    ) -> None:
        if ring_shards <= 0:
            raise ValueError("ring_shards must be positive")
        self.model = model
        self.workload = workload
        self.hardware = hardware
        self.mesh = WaferMesh(hardware.mesh_rows, hardware.mesh_cols)
        self.ring_shards = ring_shards
        self.home = (hardware.mesh_rows // 2 - 1, hardware.mesh_cols // 2 - 1)
        self.agents = self._default_agents(workload.concurrent_agents)

    def _default_agents(self, n: int) -> List[Agent]:
        rows, cols = self.hardware.mesh_rows, self.hardware.mesh_cols
        candidates = [
            (0, 0),
            (0, cols - 1),
            (rows - 1, 0),
            (rows - 1, cols - 1),
            (0, cols // 2),
            (rows // 2, 0),
            (rows - 1, cols // 2 - 1),
            (rows // 2 - 1, cols - 1),
        ]
        if n <= len(candidates):
            return [Agent(i, candidates[i]) for i in range(n)]

        # Deterministic fallback for larger N: walk the perimeter.
        perimeter: List[Coord] = []
        perimeter.extend((0, c) for c in range(cols))
        perimeter.extend((r, cols - 1) for r in range(1, rows))
        perimeter.extend((rows - 1, c) for c in range(cols - 2, -1, -1))
        perimeter.extend((r, 0) for r in range(rows - 2, 0, -1))
        seen = []
        for coord in candidates + perimeter:
            if coord not in seen:
                seen.append(coord)
        return [Agent(i, seen[i % len(seen)]) for i in range(n)]

    def _place_kv_shards(self, shared_kv_bytes: int) -> List[KVShard]:
        cycle = self.mesh.serpentine_cycle_nodes()
        if self.ring_shards > len(cycle):
            raise ValueError("ring_shards cannot exceed number of mesh regions")
        base = shared_kv_bytes // self.ring_shards
        rem = shared_kv_bytes % self.ring_shards
        shards: List[KVShard] = []
        for sid in range(self.ring_shards):
            idx = (sid * len(cycle)) // self.ring_shards
            shard_bytes = base + (1 if sid < rem else 0)
            shards.append(KVShard(sid, cycle[idx], shard_bytes, idx))
        return shards

    def _cycles_for_bytes(self, bytes_: float, bytes_per_cycle: float) -> int:
        if bytes_ <= 0.0:
            return 0
        return int(math.ceil(bytes_ / bytes_per_cycle))

    def _result(
        self,
        mode: str,
        description: str,
        stats: TraceStats,
        region_sram: Mapping[Coord, float],
        sram_bottleneck_bytes: float,
        propagation_hops: int,
        network_cycles_override: Optional[int] = None,
        extra: Optional[Dict[str, object]] = None,
    ) -> ModeResult:
        active = stats.active_link_loads
        max_link: Optional[Edge] = None
        max_link_load = 0.0
        if active:
            max_link, max_link_load = max(stats.link_loads.items(), key=lambda kv: kv[1])
        mean_active = float(np.mean(active)) if active else 0.0
        hotspot_ratio = max_link_load / mean_active if mean_active > 0.0 else 0.0

        if network_cycles_override is None:
            network_cycles = self._cycles_for_bytes(
                max_link_load, self.hardware.link_bytes_per_cycle
            ) + propagation_hops * self.hardware.hop_latency_cycles
        else:
            network_cycles = network_cycles_override
        compute_cycles = self._cycles_for_bytes(
            sram_bottleneck_bytes, self.hardware.sram_bytes_per_cycle
        )
        total_cycles = network_cycles + compute_cycles

        return ModeResult(
            mode=mode,
            description=description,
            total_sram_bytes=float(sum(region_sram.values())),
            peak_region_sram_bytes=float(max(region_sram.values()) if region_sram else 0.0),
            payload_bytes=stats.payload_bytes,
            total_wire_bytes=stats.total_wire_bytes,
            max_link_load_bytes=max_link_load,
            mean_active_link_load_bytes=mean_active,
            hotspot_ratio=hotspot_ratio,
            mesh_seconds=network_cycles / self.hardware.clock_hz,
            compute_seconds=compute_cycles / self.hardware.clock_hz,
            estimated_latency_seconds=total_cycles / self.hardware.clock_hz,
            estimated_cycles=total_cycles,
            sram_port_bytes=sram_bottleneck_bytes,
            network_cycles=network_cycles,
            compute_cycles=compute_cycles,
            link_loads=dict(stats.link_loads),
            region_sram_bytes=dict(region_sram),
            max_link=max_link,
            extra=extra or {},
        )

    def simulate_replicate_all(self) -> ModeResult:
        shared = self.workload.shared_kv_bytes(self.model)
        private_per_agent = self.workload.private_decode_kv_bytes_per_agent(self.model)
        private_write_bytes = self.model.kv_token_bytes * self.workload.total_decode_steps

        stats = TraceStats()
        max_setup_hops = 0
        for agent in self.agents:
            self.mesh.add_transfer(stats, self.home, agent.position, shared)
            max_setup_hops = max(max_setup_hops, self.mesh.route_hops(self.home, agent.position))

        region_sram: DefaultDict[Coord, float] = defaultdict(float)
        for agent in self.agents:
            region_sram[agent.position] += shared + private_per_agent

        sram_bottleneck = shared * self.workload.decode_tokens_per_agent + private_write_bytes
        return self._result(
            mode="Replicate-All",
            description="Full shared KV cache copied to every agent region; decode reads are local.",
            stats=stats,
            region_sram=region_sram,
            sram_bottleneck_bytes=sram_bottleneck,
            propagation_hops=max_setup_hops,
            extra={
                "replicas": len(self.agents),
                "shared_kv_per_replica_gib": gib(shared),
                "private_decode_kv_per_agent_mib": private_per_agent / MiB,
                "private_kv_write_gib_total": gib(private_write_bytes),
                "setup_route_max_hops": max_setup_hops,
            },
        )

    def simulate_single_home(self) -> ModeResult:
        shared = self.workload.shared_kv_bytes(self.model)
        private_per_agent = self.workload.private_decode_kv_bytes_per_agent(self.model)
        private_write_bytes = self.model.kv_token_bytes * self.workload.total_decode_steps
        t = self.workload.decode_tokens_per_agent

        stats = TraceStats()
        max_decode_hops = 0
        request_bytes = self.model.query_bytes
        response_bytes = self.model.query_bytes

        for agent in self.agents:
            bytes_to_agent = shared * t
            self.mesh.add_transfer(stats, self.home, agent.position, bytes_to_agent)

            tiny_request = request_bytes * t
            tiny_response = response_bytes * t
            self.mesh.add_transfer(stats, agent.position, self.home, tiny_request)
            self.mesh.add_transfer(stats, self.home, agent.position, tiny_response)
            max_decode_hops = max(max_decode_hops, self.mesh.route_hops(self.home, agent.position))

        region_sram: DefaultDict[Coord, float] = defaultdict(float)
        region_sram[self.home] += shared
        for agent in self.agents:
            region_sram[agent.position] += private_per_agent

        sram_bottleneck = shared * self.workload.total_decode_steps + private_write_bytes
        return self._result(
            mode="Single-Home",
            description="Shared KV cache stays at one center home; every decode step performs remote KV reads.",
            stats=stats,
            region_sram=region_sram,
            sram_bottleneck_bytes=sram_bottleneck,
            propagation_hops=t * max_decode_hops,
            extra={
                "home_region": self.home,
                "remote_read_gib_per_agent": gib(shared * t),
                "tiny_query_request_kib_per_agent": (request_bytes * t) / KiB,
                "private_kv_write_gib_total": gib(private_write_bytes),
                "decode_route_max_hops": max_decode_hops,
            },
        )

    def simulate_kv_ring(self) -> ModeResult:
        shared = self.workload.shared_kv_bytes(self.model)
        private_per_agent = self.workload.private_decode_kv_bytes_per_agent(self.model)
        private_write_bytes = self.model.kv_token_bytes * self.workload.total_decode_steps
        packets = self.workload.total_decode_steps
        packet_bytes = self.model.ring_packet_bytes
        ring_edges = self.mesh.serpentine_cycle_edges()
        shards = self._place_kv_shards(shared)

        stats = TraceStats()
        self.mesh.add_ring_cycle_transfer(stats, packet_bytes=packet_bytes, packets=packets)

        region_sram: DefaultDict[Coord, float] = defaultdict(float)
        for shard in shards:
            region_sram[shard.position] += shard.bytes
        for agent in self.agents:
            region_sram[agent.position] += private_per_agent

        sram_bottleneck = max(shard.bytes for shard in shards) * packets + private_write_bytes

        # Systolic ring pipeline latency bound:
        #
        #   Cycles_network =
        #       TotalPackets / PipelineThroughput
        #       + RingSetupHops * HopLatency
        #
        # A packet occupies a directed ring channel for
        # packet_bytes / link_bytes_per_cycle cycles.  The ideal steady-state
        # ring throughput is therefore min(1, link_bytes_per_cycle/packet_bytes)
        # packets/cycle.  Eight agents injecting concurrently create arbitration,
        # phase-alignment bubbles, and occasional structural hazards, modeled by
        # the congestion bubble factor (default 1.15), which reduces throughput.
        ideal_throughput_packets_per_cycle = min(
            1.0,
            self.hardware.link_bytes_per_cycle / packet_bytes,
        )
        effective_throughput_packets_per_cycle = (
            ideal_throughput_packets_per_cycle
            / self.hardware.ring_congestion_bubble_factor
        )
        ring_setup_hops = len(ring_edges)
        ring_steady_state_cycles = int(
            math.ceil(packets / effective_throughput_packets_per_cycle)
        )
        ring_setup_cycles = ring_setup_hops * self.hardware.hop_latency_cycles
        ring_network_cycles = ring_steady_state_cycles + ring_setup_cycles

        return self._result(
            mode="KVRing",
            description="Shared KV is sharded on a full-wafer serpentine ring; only query and online-softmax state move.",
            stats=stats,
            region_sram=region_sram,
            sram_bottleneck_bytes=sram_bottleneck,
            propagation_hops=ring_setup_hops,
            network_cycles_override=ring_network_cycles,
            extra={
                "ring_shards": len(shards),
                "ring_edges": len(ring_edges),
                "ring_setup_hops": ring_setup_hops,
                "ring_packet_kib": packet_bytes / KiB,
                "total_packets": packets,
                "private_kv_write_gib_total": gib(private_write_bytes),
                "ring_congestion_bubble_factor": self.hardware.ring_congestion_bubble_factor,
                "ideal_pipeline_throughput_packets_per_cycle": ideal_throughput_packets_per_cycle,
                "effective_pipeline_throughput_packets_per_cycle": effective_throughput_packets_per_cycle,
                "ring_steady_state_cycles": ring_steady_state_cycles,
                "ring_setup_cycles": ring_setup_cycles,
                "network_formula": "ceil(total_packets / (min(1, link_bytes_per_cycle / packet_bytes) / bubble_factor)) + ring_setup_hops * hop_latency_cycles",
                "shard_gib": [gib(s.bytes) for s in shards],
                "shard_positions": [
                    {"id": s.shard_id, "position": s.position, "ring_index": s.ring_index}
                    for s in shards
                ],
            },
        )

    def run_all(self) -> List[ModeResult]:
        return [
            self.simulate_replicate_all(),
            self.simulate_single_home(),
            self.simulate_kv_ring(),
        ]


def plot_comparison(results: List[ModeResult], out_path: Path) -> None:
    names = [r.mode for r in results]
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    x = np.arange(len(results))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("KV Replication-Hotspot Dilemma on a 16x16 Wafer Mesh", fontsize=14)

    ax = axes[0, 0]
    width = 0.36
    ax.bar(x - width / 2, [gib(r.total_sram_bytes) for r in results], width, label="Full wafer", color=colors)
    ax.bar(
        x + width / 2,
        [gib(r.peak_region_sram_bytes) for r in results],
        width,
        label="Peak region",
        color=["#9ECAE9", "#FDBF6F", "#A1D99B"],
    )
    ax.set_title("SRAM Occupancy")
    ax.set_ylabel("GiB")
    ax.set_xticks(x, names, rotation=15)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[0, 1]
    wire_tib = [max(tib(r.total_wire_bytes), 1e-9) for r in results]
    ax.bar(x, wire_tib, color=colors)
    ax.set_yscale("log")
    ax.set_title("Total Mesh Traffic")
    ax.set_ylabel("TiB, directed channel byte-hop")
    ax.set_xticks(x, names, rotation=15)
    ax.grid(axis="y", alpha=0.25, which="both")

    ax = axes[1, 0]
    max_link_gib = [max(gib(r.max_link_load_bytes), 1e-9) for r in results]
    bars = ax.bar(x, max_link_gib, color=colors)
    ax.set_yscale("log")
    ax.set_title("Max Directed Link Load")
    ax.set_ylabel("GiB on hottest unidirectional channel")
    ax.set_xticks(x, names, rotation=15)
    ax.grid(axis="y", alpha=0.25, which="both")
    for bar, result in zip(bars, results):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{result.hotspot_ratio:.1f}x",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax = axes[1, 1]
    lat = [max(r.estimated_latency_seconds, 1e-12) for r in results]
    ax.bar(x, lat, color=colors)
    ax.set_yscale("log")
    ax.set_title("Estimated JCT")
    ax.set_ylabel("Seconds, log scale")
    ax.set_xticks(x, names, rotation=15)
    ax.grid(axis="y", alpha=0.25, which="both")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_link_loads(
    results: List[ModeResult],
    mesh: WaferMesh,
    agents: List[Agent],
    home: Coord,
    shards: List[KVShard],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(5.1 * len(results), 5.2), constrained_layout=True)
    if len(results) == 1:
        axes = [axes]

    all_positive = [
        load for result in results for load in result.link_loads.values() if load > 0.0
    ]
    vmin = max(min(all_positive), 1.0) if all_positive else 1.0
    vmax = max(all_positive) if all_positive else 1.0
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("magma")

    agent_positions = np.array([a.position for a in agents])
    shard_positions = np.array([s.position for s in shards])
    channel_offset = 0.08

    for ax, result in zip(axes, results):
        ax.set_title(f"{result.mode}\ndirected max {fmt_bytes(result.max_link_load_bytes)}")
        for edge in mesh.all_edges():
            load = result.link_loads.get(edge, 0.0)
            (r0, c0), (r1, c1) = edge
            if load <= 0:
                color = "#E6E6E6"
                lw = 0.5
                alpha = 0.45
            else:
                color = cmap(norm(load))
                lw = 0.8 + 3.5 * (math.log(load / vmin + 1.0) / math.log(vmax / vmin + 2.0))
                alpha = 0.95
            x0, x1 = float(c0), float(c1)
            y0, y1 = float(r0), float(r1)
            if r0 == r1:
                # Draw eastbound and westbound channels on opposite sides of
                # the same physical axis so bidirectional capacity is not
                # visually collapsed into one undirected line.
                dy = -channel_offset if c1 > c0 else channel_offset
                y0 += dy
                y1 += dy
            elif c0 == c1:
                # Same idea for north/south channels.
                dx = channel_offset if r1 > r0 else -channel_offset
                x0 += dx
                x1 += dx
            ax.plot([x0, x1], [y0, y1], color=color, lw=lw, alpha=alpha, solid_capstyle="round")

        ax.scatter(agent_positions[:, 1], agent_positions[:, 0], marker="o", s=48, c="#1F77B4", edgecolors="white", linewidths=0.8, label="Agent")
        ax.scatter([home[1]], [home[0]], marker="s", s=58, c="#D62728", edgecolors="white", linewidths=0.8, label="Home")
        if result.mode == "KVRing" and len(shard_positions) > 0:
            ax.scatter(
                shard_positions[:, 1],
                shard_positions[:, 0],
                marker="D",
                s=38,
                c="#2CA02C",
                edgecolors="white",
                linewidths=0.6,
                label="KV shard",
            )

        ax.set_xlim(-0.8, mesh.cols - 0.2)
        ax.set_ylim(mesh.rows - 0.2, -0.8)
        ax.set_aspect("equal")
        ax.set_xticks(range(mesh.cols))
        ax.set_yticks(range(mesh.rows))
        ax.tick_params(labelsize=7)
        ax.grid(False)

    handles, labels = axes[-1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=3, frameon=False)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02, label="Link load bytes, log scale")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_text_report(
    path: Path,
    model: ModelConfig,
    workload: WorkloadConfig,
    hardware: HardwareConfig,
    simulator: Simulator,
    results: List[ModeResult],
) -> None:
    lines: List[str] = []
    shared = workload.shared_kv_bytes(model)
    lines.append("KV Replication-Hotspot Dilemma Trace Report")
    lines.append("=" * 56)
    lines.append(f"Model path: {model.model_path}")
    lines.append(f"KV/token: {fmt_bytes(model.kv_token_bytes)}")
    lines.append(f"Shared prefix: {workload.shared_prefix_tokens} tokens = {fmt_bytes(shared)}")
    lines.append(f"Agents: {workload.concurrent_agents}; decode tokens/agent: {workload.decode_tokens_per_agent}")
    lines.append(f"Query bytes: {fmt_bytes(model.query_bytes)}")
    lines.append(f"KVRing packet: {fmt_bytes(model.ring_packet_bytes)}")
    private_write_bytes = model.kv_token_bytes * workload.total_decode_steps
    lines.append(f"Private decode KV writes: {fmt_bytes(private_write_bytes)} included in SRAM port bytes")
    lines.append(f"Mesh: {hardware.mesh_rows}x{hardware.mesh_cols}; home: {simulator.home}")
    lines.append(f"Link BW: {hardware.link_bandwidth_gbps:.1f} GB/s; SRAM BW/region: {hardware.region_sram_bandwidth_tibps:.1f} TiB/s")
    lines.append(f"NoC channels: directed bidirectional links; max link load is single-direction")
    lines.append(f"KVRing congestion bubble factor: {hardware.ring_congestion_bubble_factor:.2f}")
    lines.append(f"Agents: {[a.position for a in simulator.agents]}")
    lines.append("")
    for result in results:
        lines.append(result.mode)
        lines.append("-" * len(result.mode))
        lines.append(f"SRAM total / peak region: {fmt_bytes(result.total_sram_bytes)} / {fmt_bytes(result.peak_region_sram_bytes)}")
        lines.append(f"SRAM port bytes read+write: {fmt_bytes(result.sram_port_bytes)}")
        lines.append(f"Payload / mesh byte-hop traffic: {fmt_bytes(result.payload_bytes)} / {fmt_bytes(result.total_wire_bytes)}")
        lines.append(f"Max directed link load: {fmt_bytes(result.max_link_load_bytes)} on {edge_key(result.max_link) if result.max_link else 'n/a'}")
        lines.append(f"Hotspot ratio max/mean-active: {result.hotspot_ratio:.3f}x")
        lines.append(f"Mesh time: {fmt_seconds(result.mesh_seconds)} ({result.network_cycles:,} cycles)")
        lines.append(f"Compute time: {fmt_seconds(result.compute_seconds)} ({result.compute_cycles:,} cycles)")
        lines.append(f"Estimated JCT: {fmt_seconds(result.estimated_latency_seconds)} ({result.estimated_cycles:,} cycles)")
        if result.mode == "KVRing":
            lines.append(
                "KVRing network model: total_packets / effective_pipeline_throughput "
                "+ ring_setup_hops * hop_latency"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def print_console_report(
    model: ModelConfig,
    workload: WorkloadConfig,
    hardware: HardwareConfig,
    simulator: Simulator,
    results: List[ModeResult],
    output_paths: Mapping[str, Path],
) -> None:
    shared = workload.shared_kv_bytes(model)
    print("\n=== KV Replication-Hotspot Dilemma Trace Simulator ===")
    print(f"Model path: {model.model_path}")
    print(
        "Model params: "
        f"hidden={model.hidden_dim}, layers={model.layers}, kv_heads={model.kv_heads}, "
        f"head_dim={model.head_dim}, dtype_bytes={model.dtype_bytes}"
    )
    print(f"KV/token: {fmt_bytes(model.kv_token_bytes)}")
    print(f"Shared KV: {workload.shared_prefix_tokens} tokens x {fmt_bytes(model.kv_token_bytes)} = {fmt_bytes(shared)}")
    print(f"Agents: {workload.concurrent_agents} at {[a.position for a in simulator.agents]}")
    print(f"Decode tokens/agent: {workload.decode_tokens_per_agent}")
    print(f"Query vector: {fmt_bytes(model.query_bytes)}; KVRing packet: {fmt_bytes(model.ring_packet_bytes)}")
    private_write_bytes = model.kv_token_bytes * workload.total_decode_steps
    print(f"Private decode KV writes included in SRAM model: {fmt_bytes(private_write_bytes)}")
    print(
        f"Mesh: {hardware.mesh_rows}x{hardware.mesh_cols}, center home={simulator.home}, "
        f"link BW={hardware.link_bandwidth_gbps:.1f} GB/s"
    )
    print(
        "NoC accounting: directed bidirectional channels; "
        f"KVRing bubble factor={hardware.ring_congestion_bubble_factor:.2f}"
    )

    print("\n--- Mode Summary ---")
    header = (
        f"{'Mode':<15} {'SRAM total':>12} {'Peak SRAM':>12} {'Payload':>12} "
        f"{'Wire traffic':>14} {'Max dir link':>12} {'Hotspot':>9} {'JCT':>12}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.mode:<15} "
            f"{fmt_bytes(result.total_sram_bytes):>12} "
            f"{fmt_bytes(result.peak_region_sram_bytes):>12} "
            f"{fmt_bytes(result.payload_bytes):>12} "
            f"{fmt_bytes(result.total_wire_bytes):>14} "
            f"{fmt_bytes(result.max_link_load_bytes):>12} "
            f"{result.hotspot_ratio:>8.2f}x "
            f"{fmt_seconds(result.estimated_latency_seconds):>12}"
        )

    print("\n--- Hottest Links ---")
    for result in results:
        top = result.top_links(3)
        top_s = ", ".join(f"{item['edge']}={fmt_bytes(item['bytes'])}" for item in top)
        print(f"{result.mode:<15}: {top_s}")

    print("\n--- Structured Summary JSON ---")
    summary = {r.mode: r.to_summary_dict() for r in results}
    print(json.dumps(summary, indent=2, sort_keys=True))

    print("\n--- Saved Artifacts ---")
    for label, path in output_paths.items():
        print(f"{label}: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace-driven KVRing wafer mesh simulator")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="directory for JSON, text, and PNG outputs",
    )
    parser.add_argument("--ring-shards", type=int, default=8, help="number of stationary KV shards on the ring")
    parser.add_argument("--mesh-rows", type=int, default=16)
    parser.add_argument("--mesh-cols", type=int, default=16)
    parser.add_argument("--link-bandwidth-gbps", type=float, default=100.0)
    parser.add_argument("--region-sram-bandwidth-tibps", type=float, default=20.0)
    parser.add_argument(
        "--ring-congestion-bubble-factor",
        type=float,
        default=1.15,
        help="KVRing systolic pipeline congestion/synchronization bubble factor",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    model = ModelConfig()
    workload = WorkloadConfig()
    hardware = HardwareConfig(
        mesh_rows=args.mesh_rows,
        mesh_cols=args.mesh_cols,
        link_bandwidth_gbps=args.link_bandwidth_gbps,
        region_sram_bandwidth_tibps=args.region_sram_bandwidth_tibps,
        ring_congestion_bubble_factor=args.ring_congestion_bubble_factor,
    )
    simulator = Simulator(
        model=model,
        workload=workload,
        hardware=hardware,
        ring_shards=args.ring_shards,
    )
    results = simulator.run_all()
    shards = simulator._place_kv_shards(workload.shared_kv_bytes(model))

    json_path = outdir / "kv_ring_results.json"
    report_path = outdir / "kv_ring_report.txt"
    comparison_path = outdir / "kv_ring_comparison.png"
    links_path = outdir / "kv_ring_link_loads.png"

    full = {
        "config": {
            "model": {
                "path": model.model_path,
                "hidden_dim": model.hidden_dim,
                "layers": model.layers,
                "kv_heads": model.kv_heads,
                "head_dim": model.head_dim,
                "dtype_bytes": model.dtype_bytes,
                "kv_token_bytes": model.kv_token_bytes,
                "query_bytes": model.query_bytes,
                "ring_packet_bytes": model.ring_packet_bytes,
            },
            "workload": {
                "shared_prefix_tokens": workload.shared_prefix_tokens,
                "shared_kv_bytes": workload.shared_kv_bytes(model),
                "concurrent_agents": workload.concurrent_agents,
                "decode_tokens_per_agent": workload.decode_tokens_per_agent,
                "total_decode_steps": workload.total_decode_steps,
            },
            "hardware": {
                "mesh_rows": hardware.mesh_rows,
                "mesh_cols": hardware.mesh_cols,
                "home_region": simulator.home,
                "link_bandwidth_gbps": hardware.link_bandwidth_gbps,
                "region_sram_bandwidth_tibps": hardware.region_sram_bandwidth_tibps,
                "clock_hz": hardware.clock_hz,
                "hop_latency_cycles": hardware.hop_latency_cycles,
                "ring_congestion_bubble_factor": hardware.ring_congestion_bubble_factor,
                "noc_channel_model": "directed_bidirectional",
            },
            "agents": [{"id": a.agent_id, "position": a.position} for a in simulator.agents],
        },
        "results": [r.to_full_dict() for r in results],
    }
    json_path.write_text(json.dumps(full, indent=2, sort_keys=True), encoding="utf-8")

    write_text_report(report_path, model, workload, hardware, simulator, results)
    plot_comparison(results, comparison_path)
    plot_link_loads(results, simulator.mesh, simulator.agents, simulator.home, shards, links_path)

    print_console_report(
        model,
        workload,
        hardware,
        simulator,
        results,
        {
            "JSON results": json_path,
            "Text report": report_path,
            "Comparison plot": comparison_path,
            "Link-load plot": links_path,
        },
    )


if __name__ == "__main__":
    main()
