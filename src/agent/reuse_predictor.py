from typing import Dict

from .state_node import StateNode


TYPE_BASE_SCORE = {
    "system_prefix": 1.0,
    "agent_role": 1.0,
    "task_prefix": 1.0,
    "shared_prefix": 0.95,
    "skill_core": 0.85,
    "file_context": 0.75,
    "edit_diff": 0.70,
    "test_failure_summary": 0.90,
    "failure_summary": 0.80,
    "tool_observation": 0.45,
    "raw_error_log": 0.15,
    "dialogue_delta": 0.50,
    "assistant_delta": 0.50,
    "speculative_state": 0.20,
}


PHASE_VALUE = {
    "system_prefix": {"explore": 1.0, "execute": 1.0, "verify": 1.0, "failure": 1.0, "finalize": 0.8},
    "agent_role": {"explore": 1.0, "execute": 1.0, "verify": 1.0, "failure": 1.0, "finalize": 0.8},
    "task_prefix": {"explore": 1.0, "execute": 1.0, "verify": 1.0, "failure": 1.0, "finalize": 0.8},
    "shared_prefix": {"explore": 0.9, "execute": 0.9, "verify": 0.9, "failure": 0.8, "finalize": 0.6},
    "skill_core": {"explore": 0.8, "execute": 0.9, "verify": 0.8, "failure": 0.7, "finalize": 0.4},
    "file_context": {"explore": 0.8, "execute": 1.0, "verify": 0.8, "failure": 0.7, "finalize": 0.3},
    "edit_diff": {"explore": 0.2, "execute": 1.0, "verify": 0.9, "failure": 0.8, "finalize": 0.4},
    "test_failure_summary": {"explore": 0.1, "execute": 0.7, "verify": 1.0, "failure": 1.0, "finalize": 0.2},
    "failure_summary": {"explore": 0.1, "execute": 0.6, "verify": 0.9, "failure": 1.0, "finalize": 0.2},
    "tool_observation": {"explore": 0.7, "execute": 0.5, "verify": 0.3, "failure": 0.3, "finalize": 0.1},
    "raw_error_log": {"explore": 0.1, "execute": 0.2, "verify": 0.2, "failure": 0.1, "finalize": 0.0},
    "dialogue_delta": {"explore": 0.5, "execute": 0.5, "verify": 0.5, "failure": 0.4, "finalize": 0.2},
    "assistant_delta": {"explore": 0.5, "execute": 0.5, "verify": 0.5, "failure": 0.4, "finalize": 0.2},
    "speculative_state": {"explore": 0.2, "execute": 0.2, "verify": 0.1, "failure": 0.0, "finalize": 0.0},
}


def _weighted_phase_value(state_type: str, phase_score: Dict[str, float]) -> float:
    values = PHASE_VALUE.get(state_type, {})
    return sum(phase_score.get(phase, 0.0) * values.get(phase, 0.2) for phase in phase_score)


def compute_reuse_score(state: StateNode, phase_score: Dict[str, float], current_step: int) -> float:
    type_score = TYPE_BASE_SCORE.get(state.state_type, 0.4)
    phase_value = _weighted_phase_value(state.state_type, phase_score)
    recency = 1.0 / (1.0 + max(0, current_step - state.last_access))
    sharing = min(1.0, state.access_count / 4.0)

    noise_penalty = 0.0
    if state.state_type == "raw_error_log":
        noise_penalty = 0.25
    elif state.state_type == "speculative_state" and not state.metadata.get("committed", False):
        noise_penalty = 0.20

    score = 0.35 * type_score + 0.30 * phase_value + 0.20 * recency + 0.15 * sharing - noise_penalty
    score = max(0.0, min(1.0, score))
    state.reuse_prob = score
    state.phase_value = phase_value
    return score


def estimate_next_use_distance(state: StateNode, graph, phase_score: Dict[str, float]) -> float:
    if state.state_type in {"system_prefix", "task_prefix", "agent_role", "shared_prefix"}:
        return 1.0
    if state.state_type == "test_failure_summary" and (
        phase_score.get("verify", 0.0) > 0.25 or phase_score.get("failure", 0.0) > 0.25
    ):
        return 1.0
    if state.state_type == "failure_summary" and phase_score.get("failure", 0.0) > 0.25:
        return 1.0
    if state.state_type == "file_context" and phase_score.get("execute", 0.0) > 0.25:
        return 1.5
    if state.state_type == "edit_diff" and (
        phase_score.get("execute", 0.0) > 0.25 or phase_score.get("verify", 0.0) > 0.25
    ):
        return 2.0
    if state.state_type == "raw_error_log":
        return float("inf")
    if state.state_type == "speculative_state" and not state.metadata.get("committed", False):
        return float("inf")
    return max(2.0, 8.0 * (1.0 - state.reuse_prob))

