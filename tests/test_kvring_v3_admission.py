from __future__ import annotations

from kvring.config import HardwareConfig, ModelConfig, WorkloadConfig
from kvring.kvring_v3 import simulate_kvring_v3_adaptive


def test_kvring_v3_never_selects_invalid_capacity_candidate_when_valid_exists() -> None:
    model = ModelConfig(query_tile_size=8)
    workload = WorkloadConfig(shared_prefix_tokens=32768, concurrent_agents=8, decode_tokens_per_agent=256)
    hardware = HardwareConfig(region_sram_capacity_gib=1.0, region_capacity_gib=1.0)
    result = simulate_kvring_v3_adaptive(model, workload, hardware, query_tile_size=8, num_shards=8)
    assert result.extra["valid_candidate_count"] > 0
    assert result.extra["valid_capacity"] is True
    assert result.extra["selected_mode"].startswith("KVRing-v2")


def test_kvring_v3_selects_central_for_tiny_prefix_when_it_is_cheaper() -> None:
    model = ModelConfig(query_tile_size=8)
    workload = WorkloadConfig(shared_prefix_tokens=512, concurrent_agents=2, decode_tokens_per_agent=32)
    hardware = HardwareConfig(region_sram_capacity_gib=4.5, region_capacity_gib=4.5)
    result = simulate_kvring_v3_adaptive(model, workload, hardware, query_tile_size=8, num_shards=32)
    assert result.extra["selected_mode"] == "Central-KV-Stationary"


def test_kvring_v3_selects_kvring_for_large_prefix_high_agent_regime() -> None:
    model = ModelConfig(query_tile_size=8)
    workload = WorkloadConfig(shared_prefix_tokens=32768, concurrent_agents=16, decode_tokens_per_agent=512)
    hardware = HardwareConfig(region_sram_capacity_gib=4.5, region_capacity_gib=4.5)
    result = simulate_kvring_v3_adaptive(model, workload, hardware, query_tile_size=8, num_shards=8)
    assert result.extra["selected_mode"].startswith("KVRing-v2")
