from __future__ import annotations

from kvring.baselines import simulate_central_kv_stationary
from kvring.config import HardwareConfig, ModelConfig, WorkloadConfig


def test_central_kv_stationary_reports_required_breakdown() -> None:
    result = simulate_central_kv_stationary(
        ModelConfig(query_tile_size=8),
        WorkloadConfig(shared_prefix_tokens=2048, concurrent_agents=8, decode_tokens_per_agent=8),
        HardwareConfig(region_sram_capacity_gib=4.5, region_capacity_gib=4.5),
        query_tile_size=16,
    )
    extra = result.extra
    assert extra["actual_query_tile_sizes"] == [8]
    assert extra["central_query_wire_bytes"] > 0
    assert extra["central_result_wire_bytes"] > 0
    assert extra["central_query_mesh_latency_s"] > 0
    assert extra["central_result_mesh_latency_s"] > 0
    assert extra["central_bottleneck_component"] in {
        "query_mesh",
        "result_mesh",
        "sram_read",
        "compute",
        "queue",
    }
