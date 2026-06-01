from typing import Iterable, List, Tuple


def available_compute_locations(hardware_platform, device_type: str = "tensorcore") -> List[Tuple]:
    modules = getattr(hardware_platform, "modules_dict", {}).get(device_type, {})
    return sorted(modules.keys())


def node_of_module(module_loc: Tuple) -> Tuple:
    return tuple(module_loc[:-1])


def _coord_distance(coord_a, coord_b) -> float:
    return float(sum(abs(x - y) for x, y in zip(coord_a, coord_b)))


def node_distance(hardware_platform, a, b) -> float:
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


def choose_exec_location(input_states: Iterable, hardware_platform, device_type: str = "tensorcore") -> Tuple:
    candidates = available_compute_locations(hardware_platform, device_type)
    if not candidates:
        raise ValueError(f"No compute modules available for device_type={device_type}")

    resident_inputs = [state for state in input_states if state.resident and state.loc is not None]
    if not resident_inputs:
        return candidates[0]

    best_loc = candidates[0]
    best_cost = float("inf")
    for module_loc in candidates:
        module_node = node_of_module(module_loc)
        cost = 0.0
        for state in resident_inputs:
            cost += state.kv_bytes * node_distance(hardware_platform, module_node, state.loc)
        if cost < best_cost or (cost == best_cost and module_loc < best_loc):
            best_cost = cost
            best_loc = module_loc
    return best_loc


def choose_state_location(state, predicted_exec_loc, hardware_platform) -> Tuple:
    if predicted_exec_loc is None:
        locations = available_compute_locations(hardware_platform)
        if not locations:
            raise ValueError("Cannot place state without compute locations")
        predicted_exec_loc = locations[0]
    if len(predicted_exec_loc) >= 3:
        return node_of_module(predicted_exec_loc)
    return tuple(predicted_exec_loc)

