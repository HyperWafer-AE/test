from __future__ import annotations

from waferagent.mesh import MeshConfig
from waferagent.mesh_network import MeshNetwork


def _wait(bw: float) -> float:
    cfg = MeshConfig(
        rows=4,
        cols=4,
        tile_sram_mb=1,
        tile_prefill_tflops=1,
        tile_decode_tflops=1,
        link_bandwidth_GBps=bw,
        link_latency_us=1,
        multicast_supported=True,
        energy_per_flop_pJ=1,
        energy_per_byte_pJ=1,
    )
    mesh = MeshNetwork(cfg, "unit", congestion_enabled=True)
    mesh.route("j", "s0", (0, 0), (0, 3), 10_000_000, 0)
    mesh.route("j", "s1", (0, 0), (0, 3), 10_000_000, 0)
    return mesh.stats()["mesh_wait_ms"]


def test_lower_bandwidth_increases_mesh_wait():
    assert _wait(1) > _wait(100)
