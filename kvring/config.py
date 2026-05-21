"""Configuration and placement dataclasses."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

from .units import GB_DEC, GiB, TiB

Coord = Tuple[int, int]
Edge = Tuple[Coord, Coord]


def actual_query_tile_sizes(num_agents: int, query_tile_size: int) -> List[int]:
    """Return real query counts in each tile for one decode step."""
    if num_agents < 0:
        raise ValueError("num_agents must be non-negative")
    if query_tile_size <= 0:
        raise ValueError("query_tile_size must be positive")
    sizes: List[int] = []
    remaining = num_agents
    while remaining > 0:
        tile = min(query_tile_size, remaining)
        sizes.append(tile)
        remaining -= tile
    return sizes


@dataclass(frozen=True)
class ModelConfig:
    model_path: str = "/data1/duzc/model/model/LLM-Research/Meta-Llama-3___1-8B-Instruct"
    hidden_dim: int = 4096
    layers: int = 32
    kv_heads: int = 8
    query_heads: int = 32
    head_dim: int = 128
    dtype_bytes: int = 2
    query_tile_size: int = 8

    @property
    def kv_token_bytes(self) -> int:
        return 2 * self.layers * self.kv_heads * self.head_dim * self.dtype_bytes

    @property
    def query_bytes(self) -> int:
        return self.hidden_dim * self.dtype_bytes

    def query_tile_bytes(self, query_tile_size: int | None = None) -> int:
        r = self.query_tile_size if query_tile_size is None else query_tile_size
        return r * self.hidden_dim * self.dtype_bytes

    def partial_state_bytes(self, query_tile_size: int | None = None) -> int:
        r = self.query_tile_size if query_tile_size is None else query_tile_size
        m_bytes = r * self.query_heads * 4
        l_bytes = r * self.query_heads * 4
        z_bytes = r * self.hidden_dim * 4
        return m_bytes + l_bytes + z_bytes

    def collective_packet_bytes(self, query_tile_size: int | None = None) -> int:
        return self.query_tile_bytes(query_tile_size) + self.partial_state_bytes(query_tile_size)

    @property
    def legacy_online_scalar_bytes(self) -> int:
        return self.layers * self.kv_heads * 2 * 4

    @property
    def legacy_online_state_bytes(self) -> int:
        return self.query_bytes + self.legacy_online_scalar_bytes

    @property
    def legacy_ring_packet_bytes(self) -> int:
        return self.query_bytes + self.legacy_online_state_bytes


@dataclass(frozen=True)
class WorkloadConfig:
    shared_prefix_tokens: int = 32768
    concurrent_agents: int = 8
    decode_tokens_per_agent: int = 256
    private_suffix_kv_tokens: int | None = None

    def shared_kv_bytes(self, model: ModelConfig) -> int:
        return self.shared_prefix_tokens * model.kv_token_bytes

    def private_decode_kv_bytes_per_agent(self, model: ModelConfig) -> int:
        return self.decode_tokens_per_agent * model.kv_token_bytes

    def private_write_bytes(self, model: ModelConfig) -> int:
        return self.total_decode_steps * model.kv_token_bytes

    @property
    def total_decode_steps(self) -> int:
        return self.concurrent_agents * self.decode_tokens_per_agent

    def query_tiles_per_step(self, query_tile_size: int) -> int:
        return len(actual_query_tile_sizes(self.concurrent_agents, query_tile_size))

    def query_tiles_total(self, query_tile_size: int) -> int:
        return self.decode_tokens_per_agent * self.query_tiles_per_step(query_tile_size)

    @property
    def modeled_private_suffix_tokens_per_agent(self) -> int:
        if self.private_suffix_kv_tokens is not None:
            return self.private_suffix_kv_tokens
        return max(0, self.decode_tokens_per_agent - 1)


@dataclass(frozen=True)
class HardwareConfig:
    mesh_rows: int = 16
    mesh_cols: int = 16
    regions_per_shard: int = 1
    link_bandwidth_gbps: float = 100.0
    region_sram_bandwidth_tibps: float = 20.0
    region_sram_capacity_gib: float = 1.0
    region_capacity_gib: float = 1.0
    attention_compute_tops_per_region: float = 1024.0
    clock_hz: float = 1.0e9
    hop_latency_cycles: int = 5
    ring_congestion_bubble_factor: float = 1.15
    vc_model: str = "not_modeled_for_performance"
    enable_vc_model: bool = False

    @property
    def link_bandwidth_bytes_per_s(self) -> float:
        return self.link_bandwidth_gbps * GB_DEC

    @property
    def sram_bandwidth_bytes_per_s(self) -> float:
        return self.region_sram_bandwidth_tibps * TiB

    @property
    def region_capacity_bytes(self) -> float:
        return self.region_sram_capacity_gib * GiB

    @property
    def attention_compute_ops_per_s(self) -> float:
        return self.attention_compute_tops_per_region * 1e12

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
class ShardGroup:
    shard_id: int
    regions: List[Coord]
    total_bytes: int
    ring_index: int

    @property
    def home_region(self) -> Coord:
        return self.regions[0]

    @property
    def group_size(self) -> int:
        return len(self.regions)

    @property
    def bytes_per_region(self) -> float:
        return self.total_bytes / max(1, len(self.regions))


KVShard = ShardGroup
