from __future__ import annotations

from kvring.baselines import simulate_central_kv_stationary, simulate_pull_kv_independent
from kvring.config import HardwareConfig, ModelConfig, WorkloadConfig


def test_central_kv_stationary_keeps_kv_off_mesh_but_reads_centrally() -> None:
    model = ModelConfig()
    workload = WorkloadConfig(shared_prefix_tokens=2048, concurrent_agents=4, decode_tokens_per_agent=8)
    hardware = HardwareConfig()
    central = simulate_central_kv_stationary(model, workload, hardware)
    pull = simulate_pull_kv_independent(model, workload, hardware)
    expected_read = workload.shared_kv_bytes(model) * workload.total_decode_steps
    assert central.extra["central_sram_read_bytes"] == expected_read
    assert central.total_wire_bytes < pull.total_wire_bytes
    assert central.extra["query_bytes_sent"] > 0
    assert central.extra["partial_bytes_returned"] > 0
