from __future__ import annotations

from kvring.config import HardwareConfig, ModelConfig, WorkloadConfig
from kvring.kvring_v1 import simulate_kvring_v1
from kvring.kvring_v2 import simulate_kvring_v2


def test_kvring_v2_tile_reads_can_reduce_v1_sram_reads() -> None:
    workload = WorkloadConfig(shared_prefix_tokens=32768, concurrent_agents=8, decode_tokens_per_agent=64)
    hardware = HardwareConfig()
    model = ModelConfig(query_tile_size=8)
    v1 = simulate_kvring_v1(model, workload, hardware, ring_shards=4)
    v2 = simulate_kvring_v2(model, workload, hardware, query_tile_size=8, num_shards=4)
    assert v2.extra["local_sram_read_bytes"] < v1.extra["local_sram_read_bytes"]


def test_kvring_v2_local_sram_reads_monotonic_with_actual_tiles() -> None:
    workload = WorkloadConfig(shared_prefix_tokens=2048, concurrent_agents=8, decode_tokens_per_agent=8)
    hardware = HardwareConfig()
    reads = []
    for tile in [1, 2, 3, 4, 5, 6, 7, 8, 16, 32]:
        result = simulate_kvring_v2(
            ModelConfig(query_tile_size=tile), workload, hardware, query_tile_size=tile
        )
        reads.append(result.extra["local_sram_read_bytes"])
    assert reads == sorted(reads, reverse=True)
    assert reads[-1] == reads[-2] == reads[-3]
