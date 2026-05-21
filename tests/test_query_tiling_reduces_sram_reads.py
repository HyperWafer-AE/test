from __future__ import annotations

from kvring.config import HardwareConfig, ModelConfig, WorkloadConfig
from kvring.kvring_v2 import simulate_kvring_v2


def test_query_tiling_reduces_shared_sram_reads() -> None:
    workload = WorkloadConfig(concurrent_agents=8, decode_tokens_per_agent=16, shared_prefix_tokens=2048)
    hardware = HardwareConfig()
    r1 = simulate_kvring_v2(ModelConfig(query_tile_size=1), workload, hardware, query_tile_size=1)
    r8 = simulate_kvring_v2(ModelConfig(query_tile_size=8), workload, hardware, query_tile_size=8)
    assert r8.extra["local_sram_read_bytes"] == r1.extra["local_sram_read_bytes"] / 8
    assert r8.extra["num_query_tiles_total"] == workload.decode_tokens_per_agent
    assert r1.extra["num_query_tiles_total"] == workload.decode_tokens_per_agent * 8
