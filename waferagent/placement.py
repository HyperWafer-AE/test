from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from waferagent.graph_ir import AgentGraph
from waferagent.mesh import MeshConfig


@dataclass
class Placement:
    node_id: str
    tile: tuple[int, int]
    assigned_tiles: int = 1
    placement_region: str = "default"
    sram_need_bytes: int = 0
    sram_capacity_bytes: int = 0
    sram_overflow_bytes: int = 0
    metadata: dict[str, float | str] = field(default_factory=dict)


def _sram_need(graph: AgentGraph, node_id: str) -> int:
    node = graph.nodes[node_id]
    return int(node.kv_bytes_estimated or 0)


def _make_placement(graph: AgentGraph, cfg: MeshConfig, node_id: str, tile: tuple[int, int], region: str) -> Placement:
    need = _sram_need(graph, node_id)
    assigned = max(1, math.ceil(need / cfg.tile_sram_bytes)) if need else 1
    assigned = min(assigned, cfg.total_tiles)
    cap = assigned * cfg.tile_sram_bytes
    return Placement(
        node_id=node_id,
        tile=(max(0, min(cfg.rows - 1, tile[0])), max(0, min(cfg.cols - 1, tile[1]))),
        assigned_tiles=assigned,
        placement_region=region,
        sram_need_bytes=need,
        sram_capacity_bytes=cap,
        sram_overflow_bytes=max(0, need - cap),
    )


def round_robin_placement(graph: AgentGraph, cfg: MeshConfig) -> dict[str, Placement]:
    placements = {}
    for idx, node_id in enumerate(graph.topological_order()):
        r = idx % cfg.rows
        c = (idx // cfg.rows) % cfg.cols
        placements[node_id] = _make_placement(graph, cfg, node_id, (r, c), "round_robin")
    return placements


def random_seeded_placement(graph: AgentGraph, cfg: MeshConfig, seed: int = 0) -> dict[str, Placement]:
    rng = random.Random(seed)
    return {
        node_id: _make_placement(
            graph, cfg, node_id, (rng.randrange(cfg.rows), rng.randrange(cfg.cols)), "random_seeded"
        )
        for node_id in graph.topological_order()
    }


def layer_contiguous_placement(graph: AgentGraph, cfg: MeshConfig) -> dict[str, Placement]:
    placements = {}
    for node_id in graph.topological_order():
        node = graph.nodes[node_id]
        r = min(cfg.rows - 1, max(0, node.round_id % cfg.rows))
        c = len([p for p in placements.values() if p.tile[0] == r]) % cfg.cols
        placements[node_id] = _make_placement(graph, cfg, node_id, (r, c), "layer_contiguous")
    return placements


def communication_affinity_placement(
    graph: AgentGraph,
    cfg: MeshConfig,
    seed: int = 0,
    aggregator_aware: bool = True,
    avoid_hotspots: bool = True,
) -> dict[str, Placement]:
    graph.critical_path_lengths()
    prefix_groups: dict[str, list[str]] = {}
    for node_id, node in graph.nodes.items():
        key = node.shared_prefix_ids[0] if node.shared_prefix_ids else node.job_id
        prefix_groups.setdefault(key, []).append(node_id)

    placements: dict[str, Placement] = {}
    group_items = sorted(prefix_groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    grid_centers: list[tuple[int, int]] = []
    step_r = max(1, cfg.rows // max(1, int(math.sqrt(len(group_items))) + 1))
    step_c = max(1, cfg.cols // max(1, int(math.sqrt(len(group_items))) + 1))
    for gr in range(0, cfg.rows, step_r):
        for gc in range(0, cfg.cols, step_c):
            grid_centers.append((gr, gc))
    if not grid_centers:
        grid_centers = [(cfg.rows // 2, cfg.cols // 2)]

    for group_idx, (_, node_ids) in enumerate(group_items):
        center = grid_centers[group_idx % len(grid_centers)]
        radius = 0
        placed_in_group = 0
        for node_id in sorted(node_ids, key=lambda n: (-graph.nodes[n].criticality, n)):
            if aggregator_aware and graph.nodes[node_id].fan_in >= 2:
                deps = graph.nodes[node_id].deps
                dep_tiles = [placements[d].tile for d in deps if d in placements]
                if dep_tiles:
                    rr = round(sum(t[0] for t in dep_tiles) / len(dep_tiles))
                    cc = round(sum(t[1] for t in dep_tiles) / len(dep_tiles))
                    tile = (rr, cc)
                else:
                    tile = center
            else:
                if avoid_hotspots:
                    radius = placed_in_group // 4
                    offsets = [(0, 0), (0, radius), (radius, 0), (0, -radius), (-radius, 0)]
                    off = offsets[placed_in_group % len(offsets)]
                    tile = (center[0] + off[0], center[1] + off[1])
                else:
                    tile = center
            placements[node_id] = _make_placement(graph, cfg, node_id, tile, "communication_affinity")
            placed_in_group += 1
    return placements


def make_placement(
    policy: str,
    graph: AgentGraph,
    cfg: MeshConfig,
    seed: int = 0,
    aggregator_aware: bool = True,
    avoid_hotspots: bool = True,
) -> dict[str, Placement]:
    if policy == "random_seeded":
        return random_seeded_placement(graph, cfg, seed)
    if policy == "layer_contiguous":
        return layer_contiguous_placement(graph, cfg)
    if policy == "communication_affinity":
        return communication_affinity_placement(graph, cfg, seed, aggregator_aware, avoid_hotspots)
    return round_robin_placement(graph, cfg)
