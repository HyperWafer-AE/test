"""Region-level wafer mesh model and state placement utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

from .ir import StateNode

Coord = tuple[int, int]


@dataclass(frozen=True)
class Region:
    region_id: str
    coord: Coord
    memory_capacity_bytes: int
    compute_capacity: float


@dataclass(frozen=True)
class StatePlacement:
    state_id: str
    policy: str
    regions: tuple[str, ...]
    memory_bytes: int
    placement_byte_hop: int
    reason: str = ""


@dataclass
class WaferTopology:
    mesh_x: int = 8
    mesh_y: int = 8
    link_bandwidth_bytes: float = 2.0e11
    hop_latency: float = 1.0e-7
    region_memory_capacity: int = 256 * 1024 * 1024
    region_compute_capacity: float = 1.0
    region_memory_used: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mesh_x <= 0 or self.mesh_y <= 0:
            raise ValueError("mesh dimensions must be positive")
        for region in self.regions:
            self.region_memory_used.setdefault(region.region_id, 0)

    @classmethod
    def from_mesh(cls, mesh: str, **kwargs: object) -> "WaferTopology":
        x_str, sep, y_str = mesh.lower().partition("x")
        if not sep:
            raise ValueError("mesh must be formatted like 32x32")
        return cls(mesh_x=int(x_str), mesh_y=int(y_str), **kwargs)

    @property
    def regions(self) -> list[Region]:
        return [
            Region(self.region_id((x, y)), (x, y), self.region_memory_capacity, self.region_compute_capacity)
            for x in range(self.mesh_x)
            for y in range(self.mesh_y)
        ]

    def region_id(self, coord: Coord) -> str:
        self._validate(coord)
        return f"R_{coord[0]}_{coord[1]}"

    def coord(self, region_id: str) -> Coord:
        _, x, y = region_id.split("_")
        coord = (int(x), int(y))
        self._validate(coord)
        return coord

    def manhattan(self, a: str | Coord, b: str | Coord) -> int:
        ca = self.coord(a) if isinstance(a, str) else a
        cb = self.coord(b) if isinstance(b, str) else b
        self._validate(ca)
        self._validate(cb)
        return abs(ca[0] - cb[0]) + abs(ca[1] - cb[1])

    def byte_hop(self, bytes_: int, src: str | Coord, dst: str | Coord) -> int:
        return int(bytes_ * self.manhattan(src, dst))

    def nearest_region_to_centroid(self, coords: Iterable[Coord]) -> str:
        coords = list(coords)
        if not coords:
            return self.region_id((self.mesh_x // 2, self.mesh_y // 2))
        cx = sum(coord[0] for coord in coords) / len(coords)
        cy = sum(coord[1] for coord in coords) / len(coords)
        best = min(
            ((x, y) for x in range(self.mesh_x) for y in range(self.mesh_y)),
            key=lambda coord: abs(coord[0] - cx) + abs(coord[1] - cy),
        )
        return self.region_id(best)

    def place_state(
        self,
        state: StateNode,
        policy: str,
        consumer_regions: list[str],
        max_replicas: int = 8,
        commit: bool = False,
    ) -> StatePlacement:
        consumer_coords = [self.coord(region_id) for region_id in consumer_regions]
        size = state.materialized_size_bytes
        if policy == "replicate":
            regions = tuple(dict.fromkeys(consumer_regions[:max_replicas]))
            if not regions:
                regions = (self.nearest_region_to_centroid(consumer_coords),)
            reason = "replicated near consumer regions"
        elif policy == "shard":
            shard_count = min(
                max(2, math.ceil(size / max(1, self.region_memory_capacity * 0.5))),
                min(max_replicas, self.mesh_x * self.mesh_y),
            )
            center = self.coord(self.nearest_region_to_centroid(consumer_coords))
            coords = sorted(
                ((x, y) for x in range(self.mesh_x) for y in range(self.mesh_y)),
                key=lambda coord: abs(coord[0] - center[0]) + abs(coord[1] - center[1]),
            )[:shard_count]
            regions = tuple(self.region_id(coord) for coord in coords)
            reason = "sharded around weighted consumer centroid"
        elif policy in {"pin", "cache_kv", "cache_text"}:
            regions = (self.nearest_region_to_centroid(consumer_coords),)
            reason = "pinned at consumer centroid"
        elif policy == "evict":
            regions = tuple()
            reason = "not placed"
        else:
            regions = (self.nearest_region_to_centroid(consumer_coords),)
            reason = "inline/recompute placement anchor"

        memory = size * len(regions)
        placement_cost = 0
        if regions:
            for consumer_region in consumer_regions:
                nearest = min(regions, key=lambda region: self.manhattan(region, consumer_region))
                placement_cost += self.byte_hop(size, nearest, consumer_region)
        if commit:
            per_region = math.ceil(size / max(1, len(regions))) if policy == "shard" and regions else size
            for region in regions:
                self.region_memory_used[region] = self.region_memory_used.get(region, 0) + per_region
        return StatePlacement(state.state_id, policy, regions, memory, placement_cost, reason)

    def movement_byte_hop(
        self,
        state: StateNode,
        placement: StatePlacement | None,
        consumer_region: str,
        source_region: str | None = None,
    ) -> int:
        if placement and placement.regions:
            nearest = min(placement.regions, key=lambda region: self.manhattan(region, consumer_region))
            return self.byte_hop(state.materialized_size_bytes, nearest, consumer_region)
        source = source_region or self.region_id((self.mesh_x // 2, self.mesh_y // 2))
        return self.byte_hop(state.materialized_size_bytes, source, consumer_region)

    def max_memory_pressure(self) -> float:
        if not self.region_memory_used:
            return 0.0
        return max(self.region_memory_used.values()) / max(1, self.region_memory_capacity)

    def _validate(self, coord: Coord) -> None:
        if not (0 <= coord[0] < self.mesh_x and 0 <= coord[1] < self.mesh_y):
            raise ValueError(f"coordinate {coord} outside {self.mesh_x}x{self.mesh_y}")
