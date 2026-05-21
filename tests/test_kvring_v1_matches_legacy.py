from __future__ import annotations

from kvring.config import HardwareConfig, ModelConfig, WorkloadConfig
from kvring.kvring_v1 import simulate_kvring_v1
from kvring.units import GiB, MiB


def test_kvring_v1_matches_round1_legacy_numbers() -> None:
    result = simulate_kvring_v1(ModelConfig(), WorkloadConfig(), HardwareConfig())
    assert result.mode == "KVRing-v1-sequential-pipeline"
    assert abs(result.max_link_load_bytes - 36 * MiB) < 1
    assert abs(result.total_wire_bytes - 9 * GiB) < 1
    assert result.network_cycles == 435391
    assert result.extra["legacy_behavior_preserved"] is True
