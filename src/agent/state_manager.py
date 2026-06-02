import math
from collections import defaultdict
from typing import Iterable, Optional

from .cost_model import estimate_future_accesses
from .reuse_predictor import compute_reuse_score, estimate_next_use_distance
from .state_node import StateNode


class BaseStateManager:
    policy_name = "base"

    def __init__(self, memory_budget_bytes: int, per_node_budget_bytes: Optional[int] = None):
        self.memory_budget_bytes = int(memory_budget_bytes)
        self.per_node_budget_bytes = per_node_budget_bytes
        self.resident_state_ids = set()
        self.last_evicted_bytes = 0

    def update(self, graph, phase_score, current_step, future_index=None, oracle_future: bool = False, demand_predictor=None):
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

    def update(self, graph, phase_score, current_step, future_index=None, oracle_future: bool = False, demand_predictor=None):
        self._prepare(graph, phase_score, current_step)
        self._mark_selected([])


class LRUStateManager(BaseStateManager):
    policy_name = "lru"

    def update(self, graph, phase_score, current_step, future_index=None, oracle_future: bool = False, demand_predictor=None):
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

    def update(self, graph, phase_score, current_step, future_index=None, oracle_future: bool = False, demand_predictor=None):
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

    def update(self, graph, phase_score, current_step, future_index=None, oracle_future: bool = False, demand_predictor=None):
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
        selected_ids = set()
        reserved_for_dynamic = int(self.memory_budget_bytes * 0.35)
        for state in candidates:
            if reserved_for_dynamic <= 0:
                break
            if state.state_type in {"system_prefix", "task_prefix", "shared_prefix", "agent_role"}:
                continue
            if state.kv_bytes <= 0 or state.retention_score <= 0:
                continue
            if used + state.kv_bytes <= self.memory_budget_bytes and state.kv_bytes <= reserved_for_dynamic:
                selected.append(state)
                selected_ids.add(state.state_id)
                used += state.kv_bytes
                reserved_for_dynamic -= state.kv_bytes
        for state in candidates:
            if state.state_id in selected_ids:
                continue
            if state.kv_bytes <= 0 or state.retention_score <= 0:
                continue
            if used + state.kv_bytes <= self.memory_budget_bytes:
                selected.append(state)
                used += state.kv_bytes
        self._mark_selected(selected)


class ASGRetentionStateManager(ASGStateManager):
    policy_name = "asg-retention"


class ASGPlacementStateManager(ASGStateManager):
    policy_name = "asg-placement"


class ASGPrefetchStateManager(ASGStateManager):
    policy_name = "asg-prefetch"


class ASGKnapsackStateManager(BaseStateManager):
    policy_name = "asg-retention-v2"

    def __init__(
        self,
        memory_budget_bytes: int,
        per_node_budget_bytes: Optional[int] = None,
        prefill_cycles_per_token: int = 1000,
        knapsack_granularity_bytes: int = 16 * 1024 * 1024,
        max_future_access_cap: int = 8,
        storage_penalty: float = 0.0,
        knapsack_max_candidates: int = 2048,
        current_prompt_bonus: float = 100.0,
        recency_bonus: float = 1.0,
    ):
        super().__init__(memory_budget_bytes, per_node_budget_bytes)
        self.prefill_cycles_per_token = int(prefill_cycles_per_token)
        self.knapsack_granularity_bytes = max(1, int(knapsack_granularity_bytes))
        self.max_future_access_cap = max(1, int(max_future_access_cap))
        self.storage_penalty = float(storage_penalty)
        self.knapsack_max_candidates = max(1, int(knapsack_max_candidates))
        self.current_prompt_bonus = float(current_prompt_bonus)
        self.recency_bonus = float(recency_bonus)

    def _criticality(self, state: StateNode, phase_score: dict) -> float:
        criticality = 1.0
        if state.state_type in {"system_prefix", "task_prefix", "agent_role", "shared_prefix"}:
            criticality += 0.5
        if state.state_type in {"test_failure_summary", "failure_summary"} and phase_score.get("failure", 0.0) > 0.2:
            criticality += 0.5
        if state.state_type in {"edit_diff", "file_context"} and phase_score.get("execute", 0.0) > 0.2:
            criticality += 0.25
        return criticality

    def _future_terms(self, state: StateNode, graph, phase_score, current_step, future_index, oracle_future, demand_predictor=None):
        if oracle_future and future_index is not None:
            future_accesses = future_index.future_access_count(
                current_step,
                state.state_id,
            )
            next_use = future_index.next_use_distance(current_step, state.state_id)
            if math.isinf(next_use):
                future_accesses = 0
            reuse_probability = 1.0 if future_accesses > 0 else 0.0
            expected_saved_cycles = None
        elif demand_predictor is not None:
            prediction = demand_predictor.predict(state, graph, phase_score, current_step)
            future_accesses = prediction.expected_future_accesses
            next_use = prediction.next_use_distance
            reuse_probability = prediction.reuse_probability
            expected_saved_cycles = prediction.expected_saved_cycles
        else:
            future_accesses = estimate_future_accesses(state, phase_score, agent=None)
            next_use = state.next_use
            reuse_probability = state.reuse_prob
            expected_saved_cycles = None
        state.next_use = next_use
        state.predicted_future_accesses = float(future_accesses)
        return future_accesses, reuse_probability, expected_saved_cycles

    def _live_prompt_profit(self, state: StateNode, current_step: int) -> float:
        """Keep online ASG from evicting states in the live prompt.

        Real agent traces often replay an accumulated prompt. The policy is
        refreshed immediately before the LLM event is costed, so LRU naturally
        protects just-touched prompt segments. Future-demand scoring still needs
        that admission credit, then uses its semantic and future terms to break
        ties better than plain recency.
        """
        age_since_access = max(0, int(current_step) - int(state.last_access))
        hot_bonus = self.current_prompt_bonus if age_since_access == 0 else 0.0
        recency_bonus = self.recency_bonus / (1.0 + age_since_access)
        if hot_bonus <= 0.0 and recency_bonus <= 0.0:
            return 0.0
        return state.token_len * self.prefill_cycles_per_token * (hot_bonus + recency_bonus)

    def _profit(self, state: StateNode, graph, phase_score, current_step, future_index, oracle_future, demand_predictor=None) -> float:
        future_accesses, reuse_probability, predicted_saved_cycles = self._future_terms(
            state,
            graph,
            phase_score,
            current_step,
            future_index,
            oracle_future,
            demand_predictor,
        )
        if state.kv_bytes <= 0:
            state.retention_score = 0.0
            return 0.0
        live_prompt_profit = self._live_prompt_profit(state, current_step)
        if future_accesses <= 0:
            state.retention_score = live_prompt_profit
            return state.retention_score
        recompute_cost = state.token_len * self.prefill_cycles_per_token
        storage_cost = self.storage_penalty * state.kv_bytes / max(1, self.memory_budget_bytes)
        if predicted_saved_cycles is None:
            expected_saved_cycles = (
                min(float(future_accesses), float(self.max_future_access_cap))
                * recompute_cost
                * float(reuse_probability)
                * self._criticality(state, phase_score)
            )
        else:
            expected_saved_cycles = float(predicted_saved_cycles) * self._criticality(state, phase_score)
        state.retention_score = expected_saved_cycles + live_prompt_profit - storage_cost
        return state.retention_score

    def update(self, graph, phase_score, current_step, future_index=None, oracle_future: bool = False, demand_predictor=None):
        self._prepare(graph, phase_score, current_step)
        capacity = self.memory_budget_bytes // self.knapsack_granularity_bytes
        if capacity <= 0:
            self._mark_selected([])
            return

        candidates = []
        for state in self._all_states:
            profit = self._profit(state, graph, phase_score, current_step, future_index, oracle_future, demand_predictor)
            if profit <= 0 or state.kv_bytes <= 0:
                continue
            if state.state_type in {"system_prefix", "task_prefix", "shared_prefix"}:
                weight = 0
            else:
                weight = max(1, math.ceil(state.kv_bytes / self.knapsack_granularity_bytes))
            if weight > capacity:
                continue
            candidates.append((profit, -weight, -state.access_count, state.state_id, state, weight))
        candidates.sort(reverse=True)
        candidates = candidates[: self.knapsack_max_candidates]

        # Sparse dynamic-programming knapsack over MB-ish buckets. This optimizes
        # saved cycles directly, avoiding the tiny-state density bias of Round3.
        dp = {0: (0.0, [])}
        for profit, _, _, _, state, weight in candidates:
            snapshot = list(dp.items())
            for used, (value, states) in snapshot:
                new_used = used + weight
                if new_used > capacity:
                    continue
                new_value = value + profit
                old_value = dp.get(new_used, (-1.0, []))[0]
                if new_value > old_value:
                    dp[new_used] = (new_value, states + [state])
        best_used, (_, selected) = max(dp.items(), key=lambda item: (item[1][0], -item[0]))
        _ = best_used
        self._mark_selected(selected)


class ASGRetentionV2StateManager(ASGKnapsackStateManager):
    policy_name = "asg-retention-v2"


class ASGPlacementV2StateManager(ASGKnapsackStateManager):
    policy_name = "asg-placement-v2"


class ASGPrefetchV2StateManager(ASGKnapsackStateManager):
    policy_name = "asg-prefetch-v2"


class ASGOracleRetentionStateManager(ASGKnapsackStateManager):
    policy_name = "asg-oracle-retention"


class ASGOraclePlacementStateManager(ASGKnapsackStateManager):
    policy_name = "asg-oracle-placement"


class ASGOraclePrefetchStateManager(ASGKnapsackStateManager):
    policy_name = "asg-oracle-prefetch"


class NodeMemoryTracker:
    def __init__(self, global_budget_bytes, per_node_budget_bytes=None):
        self.global_budget_bytes = int(global_budget_bytes)
        self.per_node_budget_bytes = (
            int(per_node_budget_bytes) if per_node_budget_bytes is not None else None
        )
        self.node_used = defaultdict(int)
        self.state_to_node = {}
        self.total_used = 0

    def can_place(self, state, node) -> bool:
        node = tuple(node)
        current_node = self.state_to_node.get(state.state_id)
        current_size = state.kv_bytes if current_node == node else 0
        projected_total = self.total_used
        if current_node is None:
            projected_total += state.kv_bytes
        elif current_node != node:
            projected_total += 0
        if projected_total > self.global_budget_bytes:
            return False
        if self.per_node_budget_bytes is None:
            return True
        projected_node = self.node_used[node] - current_size + state.kv_bytes
        return projected_node <= self.per_node_budget_bytes

    def place(self, state, node):
        node = tuple(node)
        self.remove(state)
        self.node_used[node] += state.kv_bytes
        self.state_to_node[state.state_id] = node
        self.total_used += state.kv_bytes
        state.loc = node

    def remove(self, state):
        old_node = self.state_to_node.pop(state.state_id, None)
        if old_node is None:
            return
        self.node_used[old_node] = max(0, self.node_used[old_node] - state.kv_bytes)
        self.total_used = max(0, self.total_used - state.kv_bytes)

    def move(self, state, old_node, new_node):
        self.remove(state)
        self.place(state, new_node)

    def used(self, node):
        return self.node_used[tuple(node)]
