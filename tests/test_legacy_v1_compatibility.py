from __future__ import annotations

from kvring.artifacts import legacy_v1_results
from kvring.units import GiB, MiB, TiB


def _rel_close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * max(abs(b), 1.0)


def test_legacy_default_order_and_counters() -> None:
    results = legacy_v1_results()
    assert [r.mode for r in results] == [
        "Replicate-All",
        "Pull-KV-Independent",
        "KVRing-v1-sequential-pipeline",
    ]
    expected = {
        "Replicate-All": {
            "total_sram_bytes": 32.25 * GiB,
            "total_wire_bytes": 368 * GiB,
            "max_link_load_bytes": 16 * GiB,
            "latency": 0.221810980,
        },
        "Pull-KV-Independent": {
            "total_sram_bytes": 4.25 * GiB,
            "total_wire_bytes": 92.00035095214844 * TiB,
            "max_link_load_bytes": 4096.0078125 * GiB,
            "latency": 44.380581686,
        },
        "KVRing-v1-sequential-pipeline": {
            "total_sram_bytes": 4.25 * GiB,
            "total_wire_bytes": 9 * GiB,
            "max_link_load_bytes": 36 * MiB,
            "latency": 0.050447599,
        },
    }
    for result in results:
        exp = expected[result.mode]
        assert _rel_close(result.total_sram_bytes, exp["total_sram_bytes"], 1e-6)
        assert _rel_close(result.total_wire_bytes, exp["total_wire_bytes"], 1e-6)
        assert _rel_close(result.max_link_load_bytes, exp["max_link_load_bytes"], 1e-6)
        assert _rel_close(result.estimated_latency_seconds, exp["latency"], 1e-3)
