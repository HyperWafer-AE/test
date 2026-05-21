from __future__ import annotations

from kvring.accounting import TraceStats
from kvring.mesh import WaferMesh


def test_mesh_uses_independent_directed_channels() -> None:
    mesh = WaferMesh(2, 2)
    stats = TraceStats()
    mesh.add_transfer(stats, (0, 0), (0, 1), 10)
    mesh.add_transfer(stats, (0, 1), (0, 0), 3)
    assert stats.link_loads[((0, 0), (0, 1))] == 10
    assert stats.link_loads[((0, 1), (0, 0))] == 3
    assert stats.max_link_load_bytes == 10


def test_all_edges_generates_both_directions() -> None:
    mesh = WaferMesh(16, 16)
    edges = list(mesh.all_edges())
    assert len(edges) == 2 * (16 * 15 + 15 * 16)
    assert ((0, 0), (0, 1)) in edges
    assert ((0, 1), (0, 0)) in edges
