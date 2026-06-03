from collections import deque
from typing import Iterable, List, Sequence, Set

from .types import Node


def compute_nodes(hardware_platform) -> List[Node]:
    """Return wafer compute nodes, excluding DDR nodes."""
    nodes = set(hardware_platform.nodes_set)
    if hasattr(hardware_platform, "ddr_set"):
        nodes -= set(hardware_platform.ddr_set)
    return sorted(nodes)


def dist(hardware_platform, a: Node, b: Node, dijkstra: bool = True) -> float:
    if a == b:
        return 0
    if dijkstra and hasattr(hardware_platform, "node_to_node_distance_dict"):
        return hardware_platform.node_to_node_distance_dict[a][b]
    if hasattr(hardware_platform, "node_to_node_manhattan_distance_dict"):
        return hardware_platform.node_to_node_manhattan_distance_dict[a][b]
    return hardware_platform.node_to_node_distance_dict[a][b]


def bfs_region(
    hardware_platform,
    anchor: Node,
    max_nodes: int,
    allowed_nodes: Set[Node],
) -> List[Node]:
    """Return a compact connected region around anchor."""
    queue = deque([anchor])
    visited = {anchor}
    region: List[Node] = []
    while queue and len(region) < max_nodes:
        node = queue.popleft()
        if node in allowed_nodes:
            region.append(node)
        for neighbor in sorted(hardware_platform.available_neighbors.get(node, [])):
            if neighbor not in visited and neighbor in allowed_nodes:
                visited.add(neighbor)
                queue.append(neighbor)
    return region


def weighted_medoid(
    hardware_platform,
    candidates: Iterable[Node],
    consumers: Sequence[Node],
    weights: Sequence[float],
    dijkstra: bool = True,
) -> Node:
    """Choose candidate r minimizing sum_i weights[i] * dist(r, consumers[i])."""
    candidate_list = sorted(candidates)
    if not candidate_list:
        raise ValueError("weighted_medoid requires at least one candidate")
    if not consumers:
        return candidate_list[0]
    if len(consumers) != len(weights):
        raise ValueError("consumers and weights must have the same length")
    return min(
        candidate_list,
        key=lambda candidate: (
            sum(weight * dist(hardware_platform, candidate, consumer, dijkstra) for consumer, weight in zip(consumers, weights)),
            candidate,
        ),
    )


def choose_spread_anchors(hardware_platform, k: int, allowed_nodes: List[Node]) -> List[Node]:
    """Choose k anchors that are far apart to reduce KRD overlap."""
    if k <= 0:
        return []
    nodes = sorted(allowed_nodes)
    if not nodes:
        raise ValueError("choose_spread_anchors requires at least one allowed node")

    first = weighted_medoid(hardware_platform, nodes, nodes, [1.0] * len(nodes), dijkstra=True)
    anchors = [first]
    while len(anchors) < min(k, len(nodes)):
        next_anchor = max(
            nodes,
            key=lambda candidate: (
                min(dist(hardware_platform, candidate, anchor, dijkstra=True) for anchor in anchors),
                candidate,
            ),
        )
        if next_anchor in anchors:
            break
        anchors.append(next_anchor)
    return anchors

