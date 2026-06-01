def state_policy(state) -> str:
    if state.state_type in {"system_prefix", "task_prefix", "shared_prefix"}:
        return "replicate"
    if state.state_type == "agent_role":
        return "pin_home"
    if state.state_type in {
        "skill_core",
        "file_context",
        "edit_diff",
        "test_failure_summary",
        "failure_summary",
        "assistant_delta",
        "dialogue_delta",
    }:
        return "cost_aware"
    if state.state_type == "tool_observation":
        return "summarize_or_cold" if state.token_len >= 2048 else "cost_aware"
    if state.state_type == "raw_error_log":
        return "summarize_or_cold"
    if state.state_type == "speculative_state" and not state.metadata.get("committed", False):
        return "evict_low_priority"
    if state.state_type == "speculative_state":
        return "cost_aware"
    return "cost_aware"


def is_static_shared(state) -> bool:
    return state.state_type in {"system_prefix", "task_prefix", "shared_prefix"}


def should_never_demand_migrate(state) -> bool:
    return is_static_shared(state) or state_policy(state) in {"replicate", "evict_low_priority"}
