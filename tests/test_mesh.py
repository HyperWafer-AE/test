from __future__ import annotations

from waferagent.mesh import Mesh, MeshConfig


def test_mesh_manhattan_distance_and_traffic():
    cfg = MeshConfig(
        rows=4,
        cols=4,
        tile_sram_mb=1,
        tile_prefill_tflops=1,
        tile_decode_tflops=1,
        link_bandwidth_GBps=100,
        link_latency_us=1,
        multicast_supported=True,
        energy_per_flop_pJ=1,
        energy_per_byte_pJ=1,
    )
    mesh = Mesh(cfg)
    assert mesh.manhattan_distance((0, 0), (3, 2)) == 5
    mesh.route((0, 0), (0, 2), 100)
    stats = mesh.stats()
    assert stats["mesh_total_traffic_bytes"] == 200
    assert stats["mesh_hotspot_ratio"] >= 1.0
