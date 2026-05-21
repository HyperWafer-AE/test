"""Region-level placement helpers for KVRing shard groups."""

from __future__ import annotations

from .config import Coord, HardwareConfig
from .mesh import WaferMesh, central_home


def balanced_link_load_regions(mesh: WaferMesh, num_shards: int, hardware: HardwareConfig) -> list[Coord]:
    """Small deterministic heuristic that spreads shard roots across quadrants and diagonals."""
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    anchors = [
        (0, 0),
        (0, mesh.cols - 1),
        (mesh.rows - 1, mesh.cols - 1),
        (mesh.rows - 1, 0),
        (mesh.rows // 2, mesh.cols // 2),
        (0, mesh.cols // 2),
        (mesh.rows // 2, mesh.cols - 1),
        (mesh.rows - 1, mesh.cols // 2),
        (mesh.rows // 2, 0),
        central_home(hardware),
    ]
    coords: list[Coord] = []
    for coord in anchors + mesh.serpentine_cycle_nodes():
        if coord not in coords:
            coords.append(coord)
        if len(coords) >= num_shards:
            break
    return coords[:num_shards]


def placement_options() -> list[str]:
    return ["serpentine", "diagonal", "balanced_link_load", "stripe", "centralized"]
