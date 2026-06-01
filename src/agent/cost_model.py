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
) -> bool:
    migration_cost = estimate_comm_cycles(
        state,
        old_node,
        new_node,
        hardware_platform,
        effective_bandwidth_bytes_per_cycle,
    )
    remote_read_cost = migration_cost * float(remote_read_factor)
    future_savings = max(0.0, expected_future_accesses - 1.0) * remote_read_cost
    return future_savings > beta * migration_cost
