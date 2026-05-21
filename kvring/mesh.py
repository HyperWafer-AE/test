"""Directed 2D mesh topology and region-level placement."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import DefaultDict, Iterable, List

from .accounting import TraceStats
from .config import Agent, Coord, Edge, HardwareConfig, ShardGroup


class WaferMesh:
    def __init__(self, rows: int, cols: int) -> None:
        if rows < 2 or cols < 2:
            raise ValueError("mesh must be at least 2x2")
        self.rows = rows
        self.cols = cols

    def validate(self, coord: Coord) -> None:
        r, c = coord
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise ValueError(f"coordinate {coord} is outside {self.rows}x{self.cols} mesh")

    def xy_route(self, src: Coord, dst: Coord) -> List[Edge]:
        self.validate(src)
        self.validate(dst)
        r, c = src
        dr, dc = dst
        edges: List[Edge] = []
        step_r = 1 if dr >= r else -1
        while r != dr:
            nr = r + step_r
            edges.append(((r, c), (nr, c)))
            r = nr
        step_c = 1 if dc >= c else -1
        while c != dc:
            nc = c + step_c
            edges.append(((r, c), (r, nc)))
            c = nc
        return edges

    def route_hops(self, src: Coord, dst: Coord) -> int:
        return abs(src[0] - dst[0]) + abs(src[1] - dst[1])

    def add_transfer(self, stats: TraceStats, src: Coord, dst: Coord, bytes_: float) -> int:
        stats.add_payload(bytes_)
        route = self.xy_route(src, dst)
        for edge in route:
            stats.link_loads[edge] += bytes_
        return len(route)

    def all_edges(self) -> Iterable[Edge]:
        for r in range(self.rows):
            for c in range(self.cols - 1):
                yield ((r, c), (r, c + 1))
                yield ((r, c + 1), (r, c))
        for r in range(self.rows - 1):
            for c in range(self.cols):
                yield ((r, c), (r + 1, c))
                yield ((r + 1, c), (r, c))

    def serpentine_cycle_nodes(self) -> List[Coord]:
        nodes: List[Coord] = []
        nodes.extend((0, c) for c in range(self.cols))
        for r in range(1, self.rows):
            if r % 2 == 1:
                nodes.extend((r, c) for c in range(self.cols - 1, 0, -1))
            else:
                nodes.extend((r, c) for c in range(1, self.cols))
        nodes.append((self.rows - 1, 0))
        nodes.extend((r, 0) for r in range(self.rows - 2, 0, -1))
        if len(nodes) != self.rows * self.cols or len(set(nodes)) != len(nodes):
            raise RuntimeError("invalid serpentine Hamiltonian cycle")
        return nodes

    def serpentine_cycle_edges(self) -> List[Edge]:
        nodes = self.serpentine_cycle_nodes()
        return [(nodes[i], nodes[(i + 1) % len(nodes)]) for i in range(len(nodes))]

    def add_ring_cycle_transfer(self, stats: TraceStats, packet_bytes: float, packets: int) -> None:
        total_payload = packet_bytes * packets
        stats.add_payload(total_payload)
        per_edge = packet_bytes * packets
        for edge in self.serpentine_cycle_edges():
            stats.link_loads[edge] += per_edge


def default_agents(n: int, rows: int, cols: int) -> List[Agent]:
    candidates = [
        (0, 0),
        (0, cols - 1),
        (rows - 1, 0),
        (rows - 1, cols - 1),
        (0, cols // 2),
        (rows // 2, 0),
        (rows - 1, cols // 2 - 1),
        (rows // 2 - 1, cols - 1),
    ]
    if n <= len(candidates):
        return [Agent(i, candidates[i]) for i in range(n)]
    perimeter: List[Coord] = []
    perimeter.extend((0, c) for c in range(cols))
    perimeter.extend((r, cols - 1) for r in range(1, rows))
    perimeter.extend((rows - 1, c) for c in range(cols - 2, -1, -1))
    perimeter.extend((r, 0) for r in range(rows - 2, 0, -1))
    seen: List[Coord] = []
    for coord in candidates + perimeter:
        if coord not in seen:
            seen.append(coord)
    return [Agent(i, seen[i % len(seen)]) for i in range(n)]


def central_home(hardware: HardwareConfig) -> Coord:
    return (hardware.mesh_rows // 2 - 1, hardware.mesh_cols // 2 - 1)


def place_shard_groups(
    mesh: WaferMesh,
    shared_kv_bytes: int,
    num_shards: int,
    hardware: HardwareConfig,
    placement: str = "serpentine",
) -> List[ShardGroup]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if num_shards > mesh.rows * mesh.cols:
        raise ValueError("num_shards cannot exceed number of regions")
    cycle = mesh.serpentine_cycle_nodes()
    base = shared_kv_bytes // num_shards
    rem = shared_kv_bytes % num_shards
    groups: List[ShardGroup] = []
    used: set[Coord] = set()
    for sid in range(num_shards):
        shard_bytes = base + (1 if sid < rem else 0)
        group_size = max(1, int(math.ceil(shard_bytes / hardware.region_capacity_bytes)))
        if placement == "central":
            start_idx = cycle.index(central_home(hardware))
        elif placement == "diagonal":
            diag = (sid * max(mesh.rows, mesh.cols)) // num_shards
            coord = (diag % mesh.rows, diag % mesh.cols)
            start_idx = cycle.index(coord) if coord in cycle else 0
        elif placement == "stripe":
            start_idx = sid
        elif placement == "random":
            start_idx = (sid * 37 + 11) % len(cycle)
        else:
            start_idx = (sid * len(cycle)) // num_shards
        regions: List[Coord] = []
        idx = start_idx
        while len(regions) < group_size:
            coord = cycle[idx % len(cycle)]
            if placement in {"central", "region_split"} or coord not in used:
                regions.append(coord)
                used.add(coord)
            idx += 1
            if idx - start_idx > len(cycle) * 2:
                regions.append(coord)
        groups.append(ShardGroup(sid, regions, shard_bytes, start_idx % len(cycle)))
    return groups


def region_sram_from_shards(shards: List[ShardGroup]) -> DefaultDict[Coord, float]:
    region_sram: DefaultDict[Coord, float] = defaultdict(float)
    for shard in shards:
        for region in shard.regions:
            region_sram[region] += shard.bytes_per_region
    return region_sram
