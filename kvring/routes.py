"""Small routing helpers kept separate for artifact scripts and tests."""

from __future__ import annotations

from .accounting import TraceStats
from .config import Coord
from .mesh import WaferMesh


def route_transfer(mesh: WaferMesh, src: Coord, dst: Coord, bytes_: float) -> TraceStats:
    stats = TraceStats()
    mesh.add_transfer(stats, src, dst, bytes_)
    return stats
