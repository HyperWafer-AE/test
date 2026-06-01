from src.scheduling.event_notation import communication_notation

from .placement import node_distance


def estimate_comm_cycles(
    state,
    old_node,
    new_node,
    hardware_platform,
    effective_bandwidth_bytes_per_cycle: float = 64.0,
) -> int:
    distance = node_distance(hardware_platform, old_node, new_node)
    if distance == float("inf"):
        distance = 8.0
    distance = max(1.0, float(distance))
    bandwidth = max(1.0, float(effective_bandwidth_bytes_per_cycle))
    return int(state.kv_bytes * distance / bandwidth)


def estimate_comm_cycles_backend(state, old_node, new_node, hardware_platform) -> int:
    old_node = tuple(old_node)
    new_node = tuple(new_node)
    if old_node == new_node:
        return 0
    paths_dict = hardware_platform.xy_multicast_path(
        source_node_idx=old_node,
        target_nodes_set={new_node},
    )
    if paths_dict is None:
        return estimate_comm_cycles(state, old_node, new_node, hardware_platform)
    probe = communication_notation(
        comm_name="estimate",
        comm_tag=-1,
        source_location=old_node,
        target_location=new_node,
        comm_bytes=state.kv_bytes,
    )
    path_list = probe.get_paths(paths_dict)
    link_times = []
    for multicast_path in path_list:
        for link in multicast_path:
            link_obj = hardware_platform.links_dict.get(link)
            if link_obj is None:
                continue
            link_times.append(int(link_obj.working_time(data_bytes=state.kv_bytes)))
    if not link_times:
        return estimate_comm_cycles(state, old_node, new_node, hardware_platform)
    # The current collective_event_driver issues hops of one communication event at
    # the same scheduler time when links are free, so max-link time tracks the
    # single-event microbenchmark better than a serial sum.
    return max(link_times)


def estimate_comm_cycles_with_model(
    state,
    old_node,
    new_node,
    hardware_platform,
    effective_bandwidth_bytes_per_cycle: float = 64.0,
    comm_cost_model: str = "backend",
) -> int:
    if comm_cost_model == "backend":
        return estimate_comm_cycles_backend(state, old_node, new_node, hardware_platform)
    return estimate_comm_cycles(
        state,
        old_node,
        new_node,
        hardware_platform,
        effective_bandwidth_bytes_per_cycle,
    )


def estimate_future_accesses(state, phase_score, agent: str = None) -> float:
    expected = 1.0 + 3.0 * state.reuse_prob
    if state.next_use <= 2:
        expected += 1.0
    if agent is not None and state.owner == agent:
        expected += 0.5
    if phase_score.get("verify", 0.0) > 0.25 and state.state_type in {"edit_diff", "test_failure_summary"}:
        expected += 0.5
    if phase_score.get("failure", 0.0) > 0.25 and state.state_type in {"failure_summary", "test_failure_summary"}:
        expected += 0.5
    state.predicted_future_accesses = expected
    return expected


def should_migrate_state(
    state,
    old_node,
    new_node,
    hardware_platform,
    phase_score,
    expected_future_accesses,
    beta: float = 1.2,
    effective_bandwidth_bytes_per_cycle: float = 64.0,
    remote_read_factor: float = 0.35,
    comm_cost_model: str = "backend",
) -> bool:
    migration_cost = estimate_comm_cycles_with_model(
        state,
        old_node,
        new_node,
        hardware_platform,
        effective_bandwidth_bytes_per_cycle,
        comm_cost_model,
    )
    remote_read_cost = migration_cost * float(remote_read_factor)
    future_savings = max(0.0, expected_future_accesses - 1.0) * remote_read_cost
    return future_savings > beta * migration_cost
