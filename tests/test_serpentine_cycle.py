from __future__ import annotations

from kvring.mesh import WaferMesh


def test_serpentine_cycle_covers_each_region_once_and_edges_are_adjacent() -> None:
    mesh = WaferMesh(16, 16)
    nodes = mesh.serpentine_cycle_nodes()
    edges = mesh.serpentine_cycle_edges()
    assert len(nodes) == 16 * 16
    assert len(set(nodes)) == len(nodes)
    assert len(edges) == len(nodes)
    for src, dst in edges:
        assert mesh.route_hops(src, dst) == 1
