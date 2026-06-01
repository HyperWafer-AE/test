import math
from typing import Iterable, List, Optional, Tuple


def available_compute_locations(hardware_platform, device_type: str = "tensorcore") -> List[Tuple]:
    modules = getattr(hardware_platform, "modules_dict", {}).get(device_type, {})
    return sorted(modules.keys())


def node_of_module(module_loc: Tuple) -> Tuple:
    return tuple(module_loc[:-1])


def _coord_distance(coord_a, coord_b) -> float:
    return float(sum(abs(x - y) for x, y in zip(coord_a, coord_b)))


def node_distance(hardware_platform, a, b) -> float:
    a = tuple(a)
    b = tuple(b)
    if a == b:
        return 0.0

    distance_dict = getattr(hardware_platform, "node_to_node_distance_dict", None)
    if distance_dict and a in distance_dict and b in distance_dict[a]:
        return float(distance_dict[a][b])

    ch_hop = getattr(hardware_platform, "ch_to_ch_hop_dict", None)
    if ch_hop and a and b and a[0] in ch_hop and b[0] in ch_hop[a[0]]:
        return float(ch_hop[a[0]][b[0]])

    coord_dict = getattr(hardware_platform, "nodes_coordinate_dict", None)
    if coord_dict and a in coord_dict and b in coord_dict:
        return _coord_distance(coord_dict[a], coord_dict[b])

    return float("inf")


def _finite_distance(hardware_platform, a, b, fallback: float = 1_000_000.0) -> float:
    distance = node_distance(hardware_platform, a, b)
    if math.isinf(distance):
        return fallback
    return distance


def assign_agent_homes(
    agent_ids,
    hardware_platform,
    device_type: str = "tensorcore",
    spread: str = "round_robin",
) -> dict:
    locations = available_compute_locations(hardware_platform, device_type)
    if not locations:
        raise ValueError(f"No compute modules available for device_type={device_type}")
    sorted_agents = sorted(agent_ids)
    if spread == "compact":
        return {agent_id: locations[min(idx, len(locations) - 1)] for idx, agent_id in enumerate(sorted_agents)}
    if spread != "round_robin":
        raise ValueError(f"Unsupported agent placement spread: {spread}")
    return {agent_id: locations[idx % len(locations)] for idx, agent_id in enumerate(sorted_agents)}


def choose_shared_anchor(hardware_platform) -> Tuple:
    coord_dict = getattr(hardware_platform, "nodes_coordinate_dict", None)
    if coord_dict:
        coords = list(coord_dict.values())
        dims = len(coords[0])
        center = tuple(sum(coord[d] for coord in coords) / len(coords) for d in range(dims))
        return min(
            coord_dict,
            key=lambda node: (
                sum(abs(coord_dict[node][d] - center[d]) for d in range(dims)),
                node,
            ),
        )
    locations = available_compute_locations(hardware_platform)
    if not locations:
        raise ValueError("Cannot choose shared anchor without compute locations")
    return node_of_module(locations[0])


def choose_exec_location(
    input_states: Iterable,
    hardware_platform,
    agent_home: Optional[Tuple] = None,
    affinity_states: Optional[Iterable] = None,
    module_load: Optional[dict] = None,
    device_type: str = "tensorcore",
    locality_weight: float = 1.0,
    home_weight: float = 0.25,
    affinity_weight: float = 0.5,
    load_weight: float = 0.1,
) -> Tuple:
    candidates = available_compute_locations(hardware_platform, device_type)
    if not candidates:
        raise ValueError(f"No compute modules available for device_type={device_type}")

    module_load = module_load or {}
    affinity_states = list(affinity_states or [])
    agent_home_node = node_of_module(agent_home) if agent_home is not None else None
    best_loc = candidates[0]
    best_cost = float("inf")

    for module_loc in candidates:
        module_node = node_of_module(module_loc)
        locality_cost = 0.0
        for state in input_states:
            if state.resident and state.loc is not None:
                locality_cost += state.kv_bytes * _finite_distance(hardware_platform, module_node, state.loc)

        home_cost = 0.0
        if agent_home_node is not None:
            home_cost = _finite_distance(hardware_platform, module_node, agent_home_node)

        affinity_cost = 0.0
        for item in affinity_states:
            if isinstance(item, tuple):
                state, weight = item
            else:
                state, weight = item, 1.0
            if getattr(state, "loc", None) is not None:
                affinity_cost += float(weight) * _finite_distance(hardware_platform, module_node, state.loc)

        cost = (
            locality_weight * locality_cost
            + home_weight * home_cost
            + affinity_weight * affinity_cost
            + load_weight * module_load.get(module_loc, 0)
        )
        if cost < best_cost or (cost == best_cost and module_loc < best_loc):
            best_cost = cost
            best_loc = module_loc
    return best_loc


def _nearest_node_with_capacity(
    preferred_node,
    hardware_platform,
    state,
    per_node_used: Optional[dict],
    per_node_budget: Optional[int],
) -> Tuple:
    preferred_node = tuple(preferred_node)
    if per_node_budget is None or per_node_used is None:
        return preferred_node
    if per_node_used.get(preferred_node, 0) + state.kv_bytes <= per_node_budget:
        return preferred_node

    nodes = sorted(getattr(hardware_platform, "nodes_set", []))
    for node in sorted(
        nodes,
        key=lambda candidate: (
            _finite_distance(hardware_platform, preferred_node, candidate),
            candidate,
        ),
    ):
        if per_node_used.get(node, 0) + state.kv_bytes <= per_node_budget:
            return tuple(node)
    return preferred_node


def choose_state_location(
    state,
    predicted_exec_loc,
    hardware_platform,
    agent_home: Optional[Tuple] = None,
    shared_anchor: Optional[Tuple] = None,
    per_node_used: Optional[dict] = None,
    per_node_budget: Optional[int] = None,
) -> Tuple:
    if state.state_type in {"system_prefix", "task_prefix", "shared_prefix"}:
        preferred = shared_anchor if shared_anchor is not None else choose_shared_anchor(hardware_platform)
    elif state.state_type == "agent_role" and agent_home is not None:
        preferred = node_of_module(agent_home)
    elif predicted_exec_loc is not None:
        preferred = node_of_module(predicted_exec_loc)
    elif agent_home is not None:
        preferred = node_of_module(agent_home)
    else:
        locations = available_compute_locations(hardware_platform)
        if not locations:
            raise ValueError("Cannot place state without compute locations")
        preferred = node_of_module(locations[0])
    return _nearest_node_with_capacity(preferred, hardware_platform, state, per_node_used, per_node_budget)
