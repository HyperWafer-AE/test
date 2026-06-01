from typing import Iterable, Optional

from .reuse_predictor import compute_reuse_score, estimate_next_use_distance
from .state_node import StateNode


class BaseStateManager:
    policy_name = "base"

    def __init__(self, memory_budget_bytes: int, per_node_budget_bytes: Optional[int] = None):
        self.memory_budget_bytes = int(memory_budget_bytes)
        self.per_node_budget_bytes = per_node_budget_bytes
        self.resident_state_ids = set()
        self.last_evicted_bytes = 0

    def update(self, graph, phase_score, current_step):
        raise NotImplementedError

    def is_resident(self, state_id: str) -> bool:
        return state_id in self.resident_state_ids

    def mark_access(self, state: StateNode):
        state.last_access += 0

    def _mark_selected(self, states: Iterable[StateNode]):
        selected = {state.state_id for state in states}
        self.last_evicted_bytes = 0
        for state in sorted(self._all_states, key=lambda s: s.state_id):
            was_resident = state.resident
            if state.state_id in selected:
                state.resident = True
                state.tier = "hot"
            else:
                state.resident = False
                state.tier = "evicted"
                if was_resident:
                    self.last_evicted_bytes += state.kv_bytes
        self.resident_state_ids = selected

    def _prepare(self, graph, phase_score, current_step):
        self._all_states = list(graph.states.values())
        for state in self._all_states:
            compute_reuse_score(state, phase_score, current_step)
            state.next_use = estimate_next_use_distance(state, graph, phase_score)


class NoCacheStateManager(BaseStateManager):
    policy_name = "nocache"

    def update(self, graph, phase_score, current_step):
        self._prepare(graph, phase_score, current_step)
        self._mark_selected([])


class LRUStateManager(BaseStateManager):
    policy_name = "lru"

    def update(self, graph, phase_score, current_step):
        self._prepare(graph, phase_score, current_step)
        selected = []
        used = 0
        candidates = sorted(
            self._all_states,
            key=lambda state: (-state.last_access, -state.access_count, state.state_id),
        )
        for state in candidates:
            if state.kv_bytes <= 0:
                continue
            if used + state.kv_bytes <= self.memory_budget_bytes:
                selected.append(state)
                used += state.kv_bytes
        self._mark_selected(selected)


class KVFlowLikeStateManager(BaseStateManager):
    policy_name = "kvflow"

    def update(self, graph, phase_score, current_step):
        self._prepare(graph, phase_score, current_step)
        selected = []
        used = 0
        candidates = sorted(
            self._all_states,
            key=lambda state: (state.next_use, -state.reuse_prob, state.state_id),
        )
        for state in candidates:
            if state.kv_bytes <= 0:
                continue
            if used + state.kv_bytes <= self.memory_budget_bytes:
                selected.append(state)
                used += state.kv_bytes
        self._mark_selected(selected)


class ASGStateManager(BaseStateManager):
    policy_name = "asg"

    def __init__(
        self,
        memory_budget_bytes: int,
        per_node_budget_bytes: Optional[int] = None,
        prefill_cycles_per_token: int = 1000,
        lambda_mem: float = 1.0,
    ):
        super().__init__(memory_budget_bytes, per_node_budget_bytes)
        self.prefill_cycles_per_token = int(prefill_cycles_per_token)
        self.lambda_mem = float(lambda_mem)

    def update(self, graph, phase_score, current_step):
        self._prepare(graph, phase_score, current_step)
        selected = []
        used = 0
        scored = []
        budget = max(1, self.memory_budget_bytes)
        for state in self._all_states:
            miss_penalty = self.prefill_cycles_per_token * state.token_len
            memory_cost = state.kv_bytes / budget
            state.retention_score = state.reuse_prob * miss_penalty - self.lambda_mem * memory_cost
            density = state.retention_score / max(1, state.kv_bytes)
            scored.append((density, state.retention_score, -state.next_use, state.state_id, state))

        candidates = [item[-1] for item in sorted(scored, reverse=True)]
        for state in candidates:
            if state.kv_bytes <= 0 or state.retention_score <= 0:
                continue
            if used + state.kv_bytes <= self.memory_budget_bytes:
                selected.append(state)
                used += state.kv_bytes
        self._mark_selected(selected)

