from __future__ import annotations

from kvring.baselines import simulate_central_kv_stationary
from kvring.config import HardwareConfig, ModelConfig, WorkloadConfig, actual_query_tile_sizes
from kvring.kvring_v2 import simulate_kvring_v2


def test_actual_query_tile_sizes_examples() -> None:
    assert actual_query_tile_sizes(8, 1) == [1, 1, 1, 1, 1, 1, 1, 1]
    assert actual_query_tile_sizes(8, 2) == [2, 2, 2, 2]
    assert actual_query_tile_sizes(8, 3) == [3, 3, 2]
    assert actual_query_tile_sizes(8, 8) == [8]
    assert actual_query_tile_sizes(8, 16) == [8]
    assert actual_query_tile_sizes(10, 4) == [4, 4, 2]


def test_kvring_v2_r_greater_than_agents_does_not_double_count_payload() -> None:
    workload = WorkloadConfig(shared_prefix_tokens=2048, concurrent_agents=8, decode_tokens_per_agent=8)
    hardware = HardwareConfig()
    r8 = simulate_kvring_v2(ModelConfig(query_tile_size=8), workload, hardware, query_tile_size=8)
    r16 = simulate_kvring_v2(ModelConfig(query_tile_size=16), workload, hardware, query_tile_size=16)
    assert r8.extra["actual_query_tile_sizes"] == [8]
    assert r16.extra["actual_query_tile_sizes"] == [8]
    assert r8.extra["query_bytes_sent"] == r16.extra["query_bytes_sent"]
    assert r8.extra["reduction_bytes"] == r16.extra["reduction_bytes"]
    assert r8.payload_bytes == r16.payload_bytes
    assert r8.extra["local_sram_read_bytes"] == r16.extra["local_sram_read_bytes"]


def test_central_r_greater_than_agents_does_not_double_count_payload() -> None:
    workload = WorkloadConfig(shared_prefix_tokens=2048, concurrent_agents=8, decode_tokens_per_agent=8)
    hardware = HardwareConfig()
    r8 = simulate_central_kv_stationary(
        ModelConfig(query_tile_size=8), workload, hardware, query_tile_size=8
    )
    r16 = simulate_central_kv_stationary(
        ModelConfig(query_tile_size=16), workload, hardware, query_tile_size=16
    )
    assert r8.extra["actual_query_tile_sizes"] == [8]
    assert r16.extra["actual_query_tile_sizes"] == [8]
    assert r8.extra["central_query_payload_bytes"] == r16.extra["central_query_payload_bytes"]
    assert r8.extra["central_result_payload_bytes"] == r16.extra["central_result_payload_bytes"]
    assert r8.extra["central_sram_read_bytes"] == r16.extra["central_sram_read_bytes"]


def test_n10_r4_payloads_match_sum_over_actual_tiles() -> None:
    workload = WorkloadConfig(shared_prefix_tokens=1024, concurrent_agents=10, decode_tokens_per_agent=4)
    hardware = HardwareConfig()
    model = ModelConfig(query_tile_size=4)
    result = simulate_kvring_v2(model, workload, hardware, query_tile_size=4, num_shards=4)
    tiles = [4, 4, 2]
    assert result.extra["actual_query_tile_sizes"] == tiles
    expected_query_payload_per_step = sum(model.query_tile_bytes(tile) for tile in tiles)
    expected_partial_payload_per_step = sum(model.partial_state_bytes(tile) for tile in tiles)
    assert result.extra["query_tile_payload_bytes_per_step"] == expected_query_payload_per_step
    assert result.extra["partial_state_payload_bytes_per_step"] == expected_partial_payload_per_step
    assert result.extra["query_bytes_sent"] == (
        expected_query_payload_per_step * workload.decode_tokens_per_agent * 4
    )
    assert result.extra["reduction_bytes"] == (
        expected_partial_payload_per_step * workload.decode_tokens_per_agent * (4 + 1)
    )
