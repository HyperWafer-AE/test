from collections import defaultdict
from typing import Dict, List, Optional

from src.scheduling.event_notation import (
    communication_notation,
    external_wait_notation,
    fixed_compute_notation,
)

from .cost_model import (
    estimate_comm_cycles_with_model,
    estimate_future_accesses,
    should_migrate_state,
)
from .agent_state_graph import AgentStateGraph
from .future_demand import FutureDemandIndex
from .metrics import AgentMetrics
from .phase_detector import detect_phase
from .placement import (
    assign_agent_homes,
    choose_exec_location,
    choose_shared_anchor,
    choose_state_location,
    node_of_module,
)
from .state_lifecycle import is_static_shared, should_never_demand_migrate
from .state_manager import NodeMemoryTracker
from .state_node import ExecNode, StateNode


STATIC_DEFAULT_TOKENS = {
    "system": 512,
    "task": 512,
}


TOPOLOGY_POLICIES = {
    "asg-placement",
    "asg-prefetch",
    "asg-placement-v2",
    "asg-prefetch-v2",
    "asg-oracle-placement",
    "asg-oracle-prefetch",
    "asg",
}
PREFETCH_POLICIES = {"asg-prefetch", "asg-prefetch-v2", "asg-oracle-prefetch", "asg"}
V2_POLICIES = {
    "asg-retention-v2",
    "asg-placement-v2",
    "asg-prefetch-v2",
    "asg-oracle-retention",
    "asg-oracle-placement",
    "asg-oracle-prefetch",
    "asg",
}
ORACLE_POLICIES = {"asg-oracle-retention", "asg-oracle-placement", "asg-oracle-prefetch"}


class AgentEventBuilder:
    def __init__(
        self,
        hardware_platform,
        model_profile,
        state_manager,
        enable_prefetch: bool = True,
        enable_topology_placement: bool = True,
        agent_placement: str = "round_robin",
        per_node_budget_bytes: Optional[int] = None,
        max_prefetch_states: int = 2,
        tool_latency_scale: int = 1_000_000,
        effective_bandwidth_bytes_per_cycle: float = 64.0,
        prefetch_reuse_threshold: float = 0.6,
        prefetch_next_use_threshold: float = 2.0,
        max_prefetch_bytes: int = 536870912,
        prefetch_wait_fraction: float = 0.8,
        enable_observation_compression: bool = False,
        large_observation_token_threshold: int = 2048,
        observation_compression_ratio: float = 0.25,
        future_horizon: int = 16,
        oracle_future: bool = False,
        comm_cost_model: str = "backend",
    ):
        self.hardware_platform = hardware_platform
        self.model_profile = model_profile
        self.state_manager = state_manager
        self.enable_prefetch = enable_prefetch
        self.enable_topology_placement = enable_topology_placement
        self.agent_placement = agent_placement
        self.max_prefetch_states = int(max_prefetch_states)
        self.tool_latency_scale = int(tool_latency_scale)
        self.effective_bandwidth_bytes_per_cycle = float(effective_bandwidth_bytes_per_cycle)
        self.prefetch_reuse_threshold = float(prefetch_reuse_threshold)
        self.prefetch_next_use_threshold = float(prefetch_next_use_threshold)
        self.max_prefetch_bytes = int(max_prefetch_bytes)
        self.prefetch_wait_fraction = float(prefetch_wait_fraction)
        self.enable_observation_compression = enable_observation_compression
        self.large_observation_token_threshold = int(large_observation_token_threshold)
        self.observation_compression_ratio = float(observation_compression_ratio)
        self.future_horizon = int(future_horizon)
        self.oracle_future = bool(oracle_future)
        self.comm_cost_model = comm_cost_model
        self.graph = AgentStateGraph()
        self.metrics = AgentMetrics()
        self.events_dict: Dict[int, object] = {}
        self._next_event_tag = 0
        self._next_exec_id = 0
        self._last_agent_event: Dict[str, int] = {}
        self._pending_prefetch_tags: Dict[str, Dict[str, int]] = defaultdict(dict)
        self._module_load = defaultdict(int)
        self.agent_homes: Dict[str, tuple] = {}
        self.shared_anchor = None
        self.future_index = None
        self.node_memory = NodeMemoryTracker(
            state_manager.memory_budget_bytes,
            per_node_budget_bytes,
        )

    @property
    def policy_name(self) -> str:
        return self.state_manager.policy_name

    @property
    def topology_enabled(self) -> bool:
        return self.enable_topology_placement and self.policy_name in TOPOLOGY_POLICIES

    @property
    def prefetch_enabled(self) -> bool:
        return self.enable_prefetch and self.policy_name in PREFETCH_POLICIES

    @property
    def v2_enabled(self) -> bool:
        return self.policy_name in V2_POLICIES

    @property
    def oracle_enabled(self) -> bool:
        return self.oracle_future or self.policy_name in ORACLE_POLICIES

    def build(self, trace: List[dict]):
        self.future_index = FutureDemandIndex(trace, horizon=self.future_horizon)
        agent_ids = sorted({event["agent"] for event in trace if "agent" in event})
        self.agent_homes = assign_agent_homes(
            agent_ids,
            self.hardware_platform,
            spread=self.agent_placement,
        )
        self.shared_anchor = choose_shared_anchor(self.hardware_platform)
        self.metrics.num_agents = len(agent_ids)

        for step, trace_event in enumerate(trace):
            self.graph.current_step = step
            self.graph.recent_events.append(trace_event)
            event_type = trace_event.get("type")
            if event_type == "state":
                self._handle_state(trace_event)
            elif event_type == "llm":
                self._handle_llm(trace_event)
            elif event_type == "tool":
                self._handle_tool(trace_event)
            else:
                raise ValueError(f"Unsupported trace event type: {event_type}")

        self.metrics.num_states_total = len(self.graph.states)
        self.metrics.num_states_resident_final = sum(1 for state in self.graph.states.values() if state.resident)
        return self.events_dict, self.graph, self.metrics

    def _new_tag(self) -> int:
        tag = self._next_event_tag
        self._next_event_tag += 1
        return tag

    def _new_exec_id(self, prefix: str) -> str:
        exec_id = f"{prefix}_{self._next_exec_id}"
        self._next_exec_id += 1
        return exec_id

    def _add_event(self, event):
        self.events_dict[event.event_tag] = event
        return event

    def _add_dependency(self, parent_tag: Optional[int], child_event):
        if parent_tag is None:
            return
        child_event.dependency_set.add(parent_tag)
        self.events_dict[parent_tag].issue_set.add(child_event.event_tag)

    def _phase_score(self) -> dict:
        return detect_phase(self.graph.recent_events)

    def _phase_name(self, phase_score: dict) -> str:
        return max(phase_score.items(), key=lambda item: (item[1], item[0]))[0]

    def _agent_home(self, agent: str):
        return self.agent_homes.get(agent) or next(iter(self.agent_homes.values()))

    def _refresh_policy(self, phase_score: dict):
        self.state_manager.update(
            self.graph,
            phase_score,
            self.graph.current_step,
            future_index=self.future_index,
            oracle_future=self.oracle_enabled,
        )
        self.metrics.evicted_kv_bytes += self.state_manager.last_evicted_bytes
        self._sync_node_memory()

    def _sync_node_memory(self):
        for state in sorted(self.graph.states.values(), key=lambda item: item.state_id):
            if not state.resident:
                self.node_memory.remove(state)
            elif state.loc is not None and state.state_id not in self.node_memory.state_to_node:
                if self.node_memory.can_place(state, state.loc):
                    self.node_memory.place(state, state.loc)
            if state.resident and is_static_shared(state):
                self._replicate_static_state(state)

    def _replicate_static_state(self, state: StateNode):
        if self.policy_name == "nocache" or not self.agent_homes:
            return
        state.pinned = True
        state.anchored = True
        for home in self.agent_homes.values():
            node = node_of_module(home)
            if node not in state.replica_locs:
                state.replica_locs.add(node)
                self.metrics.num_static_replicas += 1
                self.metrics.static_replica_bytes += state.kv_bytes
        if state.loc is None:
            state.loc = self.shared_anchor or next(iter(state.replica_locs))

    def _place_state_if_needed(self, state: StateNode, predicted_exec_loc, agent: str):
        if state.loc is not None:
            return
        node = choose_state_location(
            state,
            predicted_exec_loc,
            self.hardware_platform,
            agent_home=self._agent_home(agent),
            shared_anchor=self.shared_anchor,
            per_node_used=self.node_memory.node_used,
            per_node_budget=self.node_memory.per_node_budget_bytes,
        )
        if state.resident and self.node_memory.can_place(state, node):
            self.node_memory.place(state, node)
        else:
            state.loc = node

    def _infer_state(self, state_id: str, agent: str, token_len: int = 1) -> StateNode:
        if state_id in self.graph.states:
            return self.graph.states[state_id]
        if state_id in STATIC_DEFAULT_TOKENS:
            state_type = "system_prefix" if state_id == "system" else "task_prefix"
            token_len = STATIC_DEFAULT_TOKENS[state_id]
            owner = "shared"
        elif state_id.endswith("_role"):
            state_type = "agent_role"
            owner = agent
            token_len = max(token_len, 64)
        else:
            state_type = "dialogue_delta"
            owner = agent
        return self.graph.build_or_get_state(
            state_id=state_id,
            state_type=state_type,
            owner=owner,
            token_len=token_len,
            kv_bytes=self.model_profile.kv_bytes(token_len),
        )

    def _handle_state(self, trace_event: dict):
        token_len = int(trace_event.get("tokens", trace_event.get("token_len", 1)))
        state = self.graph.build_or_get_state(
            state_id=trace_event["state_id"],
            state_type=trace_event.get("state_type", "dialogue_delta"),
            owner=trace_event.get("owner", trace_event.get("agent", "shared")),
            token_len=token_len,
            kv_bytes=int(trace_event.get("kv_bytes", self.model_profile.kv_bytes(token_len))),
            semantic_id=trace_event.get("semantic_id"),
            exact_kv_id=trace_event.get("exact_kv_id"),
            metadata=trace_event.get("metadata"),
        )
        phase_score = self._phase_score()
        self._refresh_policy(phase_score)
        if is_static_shared(state):
            state.available_event_tag = None
            self._replicate_static_state(state)
        agent = state.owner if state.owner in self.agent_homes else next(iter(self.agent_homes), "agent_0")
        self._place_state_if_needed(state, self._agent_home(agent) if self.agent_homes else None, agent)

    def _handle_llm(self, trace_event: dict):
        agent = trace_event["agent"]
        input_ids = list(trace_event.get("input_state_ids", []))
        input_states = [self._infer_state(state_id, agent) for state_id in input_ids]

        for state in input_states:
            self.graph.touch_state(state.state_id)

        phase_score = self._phase_score()
        phase_name = self._phase_name(phase_score)
        exec_id = self._new_exec_id("llm")
        append_tokens = int(trace_event.get("append_tokens", 0))
        output_tokens = int(trace_event.get("output_tokens", 0))
        exec_node = ExecNode(
            exec_id=exec_id,
            exec_type="llm",
            agent_id=agent,
            phase=phase_name,
            input_states=input_ids,
            input_tokens=sum(state.token_len for state in input_states),
            append_tokens=append_tokens,
            output_tokens=output_tokens,
            status=trace_event.get("status", "ok"),
            metadata=dict(trace_event),
        )
        self.graph.add_exec(exec_node)
        for state_id in input_ids:
            self.graph.add_dependency(state_id, exec_id)
            self.graph.add_affinity(state_id, exec_id, 1.0)
        self._add_static_affinities(agent)

        self._refresh_policy(phase_score)
        agent_home = self._agent_home(agent)
        exec_loc = self._choose_llm_exec_location(agent, input_states, exec_id, agent_home)
        exec_node.assigned_loc = exec_loc
        target_node = node_of_module(exec_loc)

        parent_tag = self._last_agent_event.get(agent)
        pending_by_state = self._pending_prefetch_tags.pop(agent, {})
        input_id_set = set(input_ids)
        prefill_dependencies = set()
        for state_id, tag in pending_by_state.items():
            if state_id in input_id_set:
                prefill_dependencies.add(tag)
            else:
                self.metrics.unused_prefetch_events += 1
        resident_now = {state.state_id: state.resident for state in input_states}
        loc_before = {state.state_id: state.loc for state in input_states}
        movement_dependencies = set()

        if self.topology_enabled:
            for state in input_states:
                if not state.resident:
                    continue
                self._place_state_if_needed(state, exec_loc, agent)
                old_node = loc_before.get(state.state_id) or state.loc
                if old_node is None:
                    continue
                if old_node == target_node:
                    self.metrics.num_action_local += 1
                    continue
                if target_node in state.replica_locs:
                    if is_static_shared(state):
                        self.metrics.num_action_static_hit += 1
                    else:
                        self.metrics.num_action_local += 1
                    continue
                if self.v2_enabled:
                    action = self._select_state_action(state, old_node, target_node, agent, phase_score)
                    if action == "REMOTE_READ":
                        self.metrics.num_action_remote_read += 1
                        read_tag = self._emit_remote_read(state, old_node, target_node, parent_tag, agent)
                        movement_dependencies.add(read_tag)
                    elif action == "REPLICATE":
                        self.metrics.num_action_replicate += 1
                        replica_tag = self._emit_kv_migration(
                            state=state,
                            old_node=old_node,
                            new_node=target_node,
                            parent_tag=parent_tag,
                            reason="replicate",
                            blocking=True,
                            agent=agent,
                            copy_only=True,
                        )
                        movement_dependencies.add(replica_tag)
                    elif action == "MIGRATE":
                        self.metrics.num_action_migrate += 1
                        migration_tag = self._emit_kv_migration(
                            state=state,
                            old_node=old_node,
                            new_node=target_node,
                            parent_tag=parent_tag,
                            reason="demand",
                            blocking=True,
                            agent=agent,
                        )
                        movement_dependencies.add(migration_tag)
                    else:
                        self.metrics.num_action_local += 1
                    continue
                if should_never_demand_migrate(state):
                    self.metrics.num_action_remote_read += 1
                    read_tag = self._emit_remote_read(state, old_node, target_node, parent_tag, agent)
                    movement_dependencies.add(read_tag)
                    continue

                expected_accesses = estimate_future_accesses(state, phase_score, agent)
                if should_migrate_state(
                    state,
                    old_node,
                    target_node,
                    self.hardware_platform,
                    phase_score,
                    expected_accesses,
                    effective_bandwidth_bytes_per_cycle=self.effective_bandwidth_bytes_per_cycle,
                    comm_cost_model=self.comm_cost_model,
                ):
                    self.metrics.num_action_migrate += 1
                    migration_tag = self._emit_kv_migration(
                        state=state,
                        old_node=old_node,
                        new_node=target_node,
                        parent_tag=parent_tag,
                        reason="demand",
                        blocking=True,
                        agent=agent,
                    )
                    movement_dependencies.add(migration_tag)
                else:
                    self.metrics.migration_skipped_by_cost += 1
                    self.metrics.num_action_remote_read += 1
                    read_tag = self._emit_remote_read(state, old_node, target_node, parent_tag, agent)
                    movement_dependencies.add(read_tag)
        else:
            for state in input_states:
                if state.resident:
                    self._place_state_if_needed(state, agent_home, agent)

        total_input_tokens = sum(state.token_len for state in input_states)
        effective_prefill_tokens = self._effective_prefill_tokens(
            input_states=input_states,
            resident_now=resident_now,
            loc_before=loc_before,
            target_node=target_node,
            append_tokens=append_tokens,
        )

        prefill_tag = self._new_tag()
        prefill = fixed_compute_notation(
            comp_name=f"{agent}_prefill",
            comp_tag=prefill_tag,
            comp_device="tensorcore",
            comp_location=exec_loc,
            duration=max(1, self.model_profile.prefill_cycles(effective_prefill_tokens)),
            metadata={
                "agent": agent,
                "exec_id": exec_id,
                "effective_prefill_tokens": effective_prefill_tokens,
                "append_tokens": append_tokens,
            },
        )
        self._add_event(prefill)
        self._add_dependency(parent_tag, prefill)
        for dep_tag in sorted(prefill_dependencies):
            self._add_dependency(dep_tag, prefill)
        for dep_tag in sorted(movement_dependencies):
            self._add_dependency(dep_tag, prefill)
        for state in input_states:
            if state.available_event_tag is not None and state.available_event_tag in self.events_dict:
                self._add_dependency(state.available_event_tag, prefill)

        decode_tag = self._new_tag()
        decode = fixed_compute_notation(
            comp_name=f"{agent}_decode",
            comp_tag=decode_tag,
            comp_device="tensorcore",
            comp_location=exec_loc,
            duration=max(1, self.model_profile.decode_cycles(output_tokens)),
            metadata={"agent": agent, "exec_id": exec_id, "output_tokens": output_tokens},
        )
        self._add_event(decode)
        self._add_dependency(prefill_tag, decode)
        self._last_agent_event[agent] = decode_tag
        self._module_load[exec_loc] += 1

        output_state_id = trace_event.get("new_state_id") or f"{agent}_llm_{exec_id}_out"
        output_state_type = trace_event.get("new_state_type", "assistant_delta")
        output_state = self.graph.build_or_get_state(
            state_id=output_state_id,
            state_type=output_state_type,
            owner=agent,
            token_len=output_tokens,
            kv_bytes=self.model_profile.kv_bytes(output_tokens),
            metadata={"producer_exec": exec_id},
        )
        output_state.producer_exec_id = exec_id
        output_state.producer_event_tag = decode_tag
        output_state.available_event_tag = decode_tag
        output_state.loc = choose_state_location(
            output_state,
            exec_loc,
            self.hardware_platform,
            agent_home=agent_home,
            shared_anchor=self.shared_anchor,
            per_node_used=self.node_memory.node_used,
            per_node_budget=self.node_memory.per_node_budget_bytes,
        )
        self.graph.add_generation(exec_id, output_state.state_id)
        self._add_output_affinities(agent, output_state.state_id, input_ids)

        self.metrics.total_input_tokens += total_input_tokens
        self.metrics.append_tokens += append_tokens
        self.metrics.effective_prefill_tokens += effective_prefill_tokens
        self.metrics.decode_tokens += output_tokens
        self.metrics.llm_output_tokens += output_tokens
        self.metrics.num_llm_steps += 1
        self._refresh_policy(phase_score)
        if output_state.resident and self.node_memory.can_place(output_state, output_state.loc):
            self.node_memory.place(output_state, output_state.loc)

    def _choose_llm_exec_location(self, agent, input_states, exec_id, agent_home):
        if not self.topology_enabled:
            return agent_home
        affinity_states = self.graph.affinity_neighbor_states(exec_id, top_k=8)
        resident_bytes = sum(state.kv_bytes for state in input_states if state.resident)
        home_weight = max(1.0, resident_bytes)
        return choose_exec_location(
            input_states=input_states,
            hardware_platform=self.hardware_platform,
            agent_home=agent_home,
            affinity_states=affinity_states,
            module_load=self._module_load,
            locality_weight=1.0,
            home_weight=home_weight,
            affinity_weight=0.4,
            load_weight=max(1.0, home_weight * 0.05),
        )

    def _effective_prefill_tokens(self, input_states, resident_now: dict, loc_before: dict, target_node, append_tokens: int) -> int:
        if self.policy_name == "nocache":
            self.metrics.cache_misses += len(input_states)
            self.metrics.state_misses += len(input_states)
            return sum(state.token_len for state in input_states) + append_tokens

        missing_tokens = 0
        for state in input_states:
            if resident_now.get(state.state_id, False):
                self.metrics.cache_hits += 1
                old_node = loc_before.get(state.state_id)
                if old_node is None or old_node == target_node or target_node in state.replica_locs:
                    self.metrics.local_state_hits += 1
                else:
                    self.metrics.remote_state_hits += 1
                    self.metrics.num_remote_accesses += 1
                    self.metrics.remote_access_bytes += state.kv_bytes
            else:
                self.metrics.cache_misses += 1
                self.metrics.state_misses += 1
                missing_tokens += state.token_len
        return missing_tokens + append_tokens

    def _state_comm_cost(self, state: StateNode, old_node, new_node) -> int:
        return estimate_comm_cycles_with_model(
            state,
            old_node,
            new_node,
            self.hardware_platform,
            self.effective_bandwidth_bytes_per_cycle,
            self.comm_cost_model,
        )

    def _future_consumers(self, state: StateNode):
        if not self.oracle_enabled or self.future_index is None:
            return []
        return self.future_index.future_consumers(
            self.graph.current_step,
            state.state_id,
            self.future_horizon,
        )

    def _future_access_count(self, state: StateNode, phase_score: dict, agent: str = None) -> float:
        if self.oracle_enabled and self.future_index is not None:
            count = self.future_index.future_access_count(
                self.graph.current_step,
                state.state_id,
                self.future_horizon,
            )
            state.predicted_future_accesses = float(count)
            state.next_use = self.future_index.next_use_distance(self.graph.current_step, state.state_id)
            return float(count)
        return estimate_future_accesses(state, phase_score, agent)

    def _future_agents(self, state: StateNode, agent: str) -> set:
        if self.oracle_enabled and self.future_index is not None:
            return self.future_index.future_agents(
                self.graph.current_step,
                state.state_id,
                self.future_horizon,
            )
        if state.owner == "shared":
            return set(self.agent_homes)
        return {state.owner or agent}

    def _can_replicate_to(self, state: StateNode, target_node) -> bool:
        target_node = tuple(target_node)
        if target_node in state.replica_locs:
            return True
        if self.node_memory.total_used + state.kv_bytes > self.node_memory.global_budget_bytes:
            return False
        if self.node_memory.per_node_budget_bytes is None:
            return True
        return self.node_memory.node_used[target_node] + state.kv_bytes <= self.node_memory.per_node_budget_bytes

    def _select_state_action(self, state: StateNode, old_node, target_node, agent: str, phase_score: dict) -> str:
        if old_node == target_node:
            return "LOCAL_HIT"
        if target_node in state.replica_locs:
            return "STATIC_REPLICA_HIT" if is_static_shared(state) else "LOCAL_HIT"
        if is_static_shared(state) or should_never_demand_migrate(state):
            self._replicate_static_state(state)
            return "STATIC_REPLICA_HIT" if target_node in state.replica_locs else "REMOTE_READ"

        future_accesses = self._future_access_count(state, phase_score, agent)
        future_agents = self._future_agents(state, agent)
        same_agent_future = len([item for item in self._future_consumers(state) if item.get("agent") == agent])
        comm_cost = self._state_comm_cost(state, old_node, target_node)
        remote_read_cost = max(1, comm_cost)
        expected_future_local_savings = max(0.0, future_accesses - 1.0) * remote_read_cost

        if state.state_type in {"tool_observation", "raw_error_log"} and state.token_len >= self.large_observation_token_threshold:
            return "REMOTE_READ"
        if future_accesses <= 1:
            return "REMOTE_READ"
        if len(future_agents) > 1:
            if self._can_replicate_to(state, target_node) and expected_future_local_savings > comm_cost:
                return "REPLICATE"
            return "REMOTE_READ"
        if same_agent_future >= 2 or future_accesses >= 2:
            should_migrate = should_migrate_state(
                state,
                old_node,
                target_node,
                self.hardware_platform,
                phase_score,
                future_accesses,
                effective_bandwidth_bytes_per_cycle=self.effective_bandwidth_bytes_per_cycle,
                comm_cost_model=self.comm_cost_model,
            )
            if should_migrate and expected_future_local_savings > comm_cost:
                return "MIGRATE"
        self.metrics.migration_skipped_by_cost += 1
        return "REMOTE_READ"

    def _emit_kv_migration(
        self,
        state: StateNode,
        old_node,
        new_node,
        parent_tag: Optional[int],
        reason: str,
        blocking: bool,
        agent: str,
        associated_wait_tag: Optional[int] = None,
        copy_only: bool = False,
    ) -> int:
        old_node = tuple(old_node)
        new_node = tuple(new_node)
        if old_node == new_node:
            state.loc = new_node
            return parent_tag if parent_tag is not None else -1

        prior_available_tag = state.available_event_tag
        tag = self._new_tag()
        migration = communication_notation(
            comm_name="kv_migrate",
            comm_tag=tag,
            source_location=old_node,
            target_location=new_node,
            comm_bytes=state.kv_bytes,
        )
        migration.metadata = {
            "agent": agent,
            "state_id": state.state_id,
            "state_type": state.state_type,
            "reason": reason,
            "blocking": blocking,
            "bytes": state.kv_bytes,
            "associated_wait_tag": associated_wait_tag,
        }
        self._add_event(migration)
        self._add_dependency(parent_tag, migration)
        if prior_available_tag is not None and prior_available_tag in self.events_dict:
            self._add_dependency(prior_available_tag, migration)
        estimate = estimate_comm_cycles_with_model(
            state,
            old_node,
            new_node,
            self.hardware_platform,
            self.effective_bandwidth_bytes_per_cycle,
            self.comm_cost_model,
        )
        self.metrics.migration_cost_estimate_cycles += estimate
        self.metrics.kv_migration_bytes += state.kv_bytes
        self.metrics.num_kv_migrations += 1
        if reason == "prefetch":
            self.metrics.prefetch_migration_bytes += state.kv_bytes
            self.metrics.num_prefetch_migrations += 1
            self.metrics.num_prefetch_events += 1
            self.metrics.prefetch_kv_bytes += state.kv_bytes
        elif reason == "replicate":
            self.metrics.demand_migration_bytes += state.kv_bytes
            self.metrics.num_demand_migrations += 1
        else:
            self.metrics.demand_migration_bytes += state.kv_bytes
            self.metrics.num_demand_migrations += 1
        if copy_only:
            state.replica_locs.add(new_node)
        elif state.resident:
            self.node_memory.move(state, old_node, new_node)
        else:
            state.loc = new_node
        state.available_event_tag = tag
        return tag

    def _emit_remote_read(self, state: StateNode, old_node, new_node, parent_tag: Optional[int], agent: str) -> int:
        old_node = tuple(old_node)
        new_node = tuple(new_node)
        tag = self._new_tag()
        remote_read = communication_notation(
            comm_name="kv_remote_read",
            comm_tag=tag,
            source_location=old_node,
            target_location=new_node,
            comm_bytes=state.kv_bytes,
        )
        remote_read.metadata = {
            "agent": agent,
            "state_id": state.state_id,
            "state_type": state.state_type,
            "reason": "remote_read",
            "blocking": True,
            "bytes": state.kv_bytes,
        }
        self._add_event(remote_read)
        self._add_dependency(parent_tag, remote_read)
        if state.available_event_tag is not None and state.available_event_tag in self.events_dict:
            self._add_dependency(state.available_event_tag, remote_read)
        estimate = estimate_comm_cycles_with_model(
            state,
            old_node,
            new_node,
            self.hardware_platform,
            self.effective_bandwidth_bytes_per_cycle,
            self.comm_cost_model,
        )
        self.metrics.remote_read_cost_estimate_cycles += estimate
        self.metrics.remote_read_bytes += state.kv_bytes
        self.metrics.num_remote_reads += 1
        return tag

    def _handle_tool(self, trace_event: dict):
        agent = trace_event["agent"]
        tool_name = trace_event.get("tool", "tool")
        phase_score = self._phase_score()
        phase_name = self._phase_name(phase_score)
        exec_id = self._new_exec_id("tool")
        raw_latency = int(trace_event.get("latency", trace_event.get("tool_latency", 0)))
        latency = int(raw_latency * self.tool_latency_scale)
        output_tokens = int(trace_event.get("output_tokens", 0))
        exec_node = ExecNode(
            exec_id=exec_id,
            exec_type="tool",
            agent_id=agent,
            phase=phase_name,
            output_tokens=output_tokens,
            tool_name=tool_name,
            tool_latency=latency,
            status=trace_event.get("status", "ok"),
            metadata=dict(trace_event),
        )
        self.graph.add_exec(exec_node)

        parent_before_wait = self._last_agent_event.get(agent)
        wait_tag = self._new_tag()
        wait = external_wait_notation(
            wait_name=f"tool_{tool_name}",
            wait_tag=wait_tag,
            duration=max(0, latency),
            metadata={"agent": agent, "exec_id": exec_id, "tool": tool_name, "raw_latency": raw_latency},
        )
        self._add_event(wait)
        self._add_dependency(parent_before_wait, wait)
        self._last_agent_event[agent] = wait_tag

        state_id = trace_event.get("new_state_id") or f"{agent}_tool_{exec_id}_out"
        state_type = trace_event.get("new_state_type", "tool_observation")
        output_tokens, summarized, original_tokens = self._maybe_compress_observation(state_type, output_tokens)
        if summarized:
            self.metrics.num_compressed_observations += 1
            self.metrics.compressed_observation_tokens_saved += max(0, original_tokens - output_tokens)
        state = self.graph.build_or_get_state(
            state_id=state_id,
            state_type=state_type,
            owner=agent,
            token_len=output_tokens,
            kv_bytes=self.model_profile.kv_bytes(output_tokens),
            metadata={"producer_exec": exec_id, "tool": tool_name, "original_tokens": original_tokens},
        )
        state.producer_exec_id = exec_id
        state.producer_event_tag = wait_tag
        state.available_event_tag = wait_tag
        state.summarized = summarized
        state.loc = choose_state_location(
            state,
            self._agent_home(agent),
            self.hardware_platform,
            agent_home=self._agent_home(agent),
            shared_anchor=self.shared_anchor,
            per_node_used=self.node_memory.node_used,
            per_node_budget=self.node_memory.per_node_budget_bytes,
        )
        self.graph.add_generation(exec_id, state.state_id)
        self.graph.add_affinity(exec_id, state.state_id, 1.0)
        self._add_output_affinities(agent, state.state_id, ["task", f"{agent}_role"])

        self.metrics.tool_wait_cycles += latency
        self.metrics.tool_output_tokens += output_tokens
        self.metrics.num_tool_steps += 1
        self._refresh_policy(phase_score)
        if state.resident and self.node_memory.can_place(state, state.loc):
            self.node_memory.place(state, state.loc)
        self._maybe_emit_prefetches_during_tool_wait(agent, parent_before_wait, wait_tag, latency, trace_event, phase_score)

    def _maybe_compress_observation(self, state_type: str, token_len: int):
        original_tokens = token_len
        if not self.enable_observation_compression:
            return token_len, False, original_tokens
        compressible = {"tool_observation", "file_context", "raw_error_log", "failure_summary"}
        if state_type == "test_failure_summary" and token_len > self.large_observation_token_threshold:
            return self.large_observation_token_threshold, True, original_tokens
        if state_type not in compressible or token_len <= self.large_observation_token_threshold:
            return token_len, False, original_tokens
        compressed = max(1, int(token_len * self.observation_compression_ratio))
        return compressed, True, original_tokens

    def _maybe_emit_prefetches_during_tool_wait(self, agent, parent_tag, wait_tag, tool_latency_cycles, tool_event, phase_score):
        if not self.prefetch_enabled or self.max_prefetch_states <= 0:
            return
        if self.v2_enabled:
            self._maybe_emit_windowed_prefetches(agent, parent_tag, wait_tag, tool_latency_cycles, phase_score)
            return
        predicted_exec_loc = self._agent_home(agent)
        target_node = node_of_module(predicted_exec_loc)
        candidates = []
        for state in self.graph.states.values():
            if not state.resident or state.loc is None or state.loc == target_node:
                continue
            cross_agent_candidate = state.owner not in {agent, "shared"}
            if (
                cross_agent_candidate
                and state.access_count < 2
                and state.reuse_prob < self.prefetch_reuse_threshold + 0.1
            ):
                continue
            if is_static_shared(state) and target_node in state.replica_locs:
                continue
            if state.state_type == "raw_error_log":
                continue
            if state.state_type == "speculative_state" and not state.metadata.get("committed", False):
                continue
            if state.reuse_prob < self.prefetch_reuse_threshold:
                continue
            if state.next_use > self.prefetch_next_use_threshold:
                continue
            if state.kv_bytes > self.max_prefetch_bytes:
                continue
            estimated_cycles = estimate_comm_cycles_with_model(
                state,
                state.loc,
                target_node,
                self.hardware_platform,
                self.effective_bandwidth_bytes_per_cycle,
                self.comm_cost_model,
            )
            if estimated_cycles > self.prefetch_wait_fraction * max(1, tool_latency_cycles):
                continue
            priority = (
                state.reuse_prob
                + (0.3 if state.owner == agent else 0.0)
                + (0.2 if state.next_use <= 2 else 0.0)
                + phase_score.get("execute", 0.0) * 0.1
            )
            candidates.append((priority, state.state_id, state))

        for _, _, state in sorted(candidates, reverse=True)[: self.max_prefetch_states]:
            old_node = state.loc
            tag = self._emit_kv_migration(
                state=state,
                old_node=old_node,
                new_node=target_node,
                parent_tag=parent_tag,
                reason="prefetch",
                blocking=False,
                agent=agent,
                associated_wait_tag=wait_tag,
            )
            if tag >= 0:
                self._pending_prefetch_tags[agent][state.state_id] = tag

    def _prefetch_future_accesses_for_agent(self, state: StateNode, agent: str) -> int:
        if not self.oracle_enabled:
            if state.owner not in {agent, "shared"} and state.reuse_prob < self.prefetch_reuse_threshold:
                return 0
            return 1 if state.next_use <= self.prefetch_next_use_threshold else 0
        if self.future_index is None:
            return 0
        return len(
            [
                consumer
                for consumer in self.future_index.future_consumers(
                    self.graph.current_step,
                    state.state_id,
                    self.future_horizon,
                )
                if consumer.get("agent") == agent
            ]
        )

    def _maybe_emit_windowed_prefetches(self, agent, parent_tag, wait_tag, tool_latency_cycles, phase_score):
        target_node = node_of_module(self._agent_home(agent))
        window_budget = int(max(0, tool_latency_cycles) * self.prefetch_wait_fraction)
        if window_budget <= 0:
            return
        jobs = []
        for state in self.graph.states.values():
            if not state.resident or state.loc is None:
                continue
            if state.loc == target_node or target_node in state.replica_locs:
                continue
            if state.state_type == "raw_error_log":
                continue
            if state.state_type == "speculative_state" and not state.metadata.get("committed", False):
                continue
            if state.kv_bytes > self.max_prefetch_bytes:
                continue
            future_count = self._prefetch_future_accesses_for_agent(state, agent)
            if future_count <= 0:
                continue
            next_use = (
                self.future_index.next_use_distance(self.graph.current_step, state.state_id)
                if self.oracle_enabled and self.future_index
                else state.next_use
            )
            if next_use > self.future_horizon:
                continue
            cost_cycles = self._state_comm_cost(state, state.loc, target_node)
            if cost_cycles <= 0 or cost_cycles > window_budget:
                continue
            recompute_cost = state.token_len * self.model_profile.prefill_cycles_per_token
            benefit_cycles = max(cost_cycles * future_count, recompute_cost * min(future_count, 2) * 0.05)
            if state.owner == agent:
                benefit_cycles *= 1.15
            if state.state_type in {"file_context", "edit_diff", "test_failure_summary", "failure_summary"}:
                benefit_cycles *= 1.10
            if benefit_cycles <= cost_cycles:
                continue
            jobs.append(
                {
                    "density": benefit_cycles / max(1, cost_cycles),
                    "benefit": benefit_cycles,
                    "cost": cost_cycles,
                    "deadline": next_use,
                    "state": state,
                }
            )

        jobs.sort(key=lambda item: (-item["density"], item["deadline"], item["state"].state_id))
        used_budget = 0
        emitted = 0
        for job in jobs:
            if emitted >= self.max_prefetch_states:
                break
            if used_budget + job["cost"] > window_budget:
                continue
            state = job["state"]
            tag = self._emit_kv_migration(
                state=state,
                old_node=state.loc,
                new_node=target_node,
                parent_tag=parent_tag,
                reason="prefetch",
                blocking=False,
                agent=agent,
                associated_wait_tag=wait_tag,
            )
            if tag >= 0:
                self._pending_prefetch_tags[agent][state.state_id] = tag
                used_budget += job["cost"]
                emitted += 1

    def _add_static_affinities(self, agent: str):
        role = f"{agent}_role"
        if role in self.graph.states:
            for static_id in ("system", "task"):
                if static_id in self.graph.states:
                    self.graph.add_affinity(static_id, role, 1.5)

    def _add_output_affinities(self, agent: str, output_state_id: str, input_ids: List[str]):
        for anchor_id in ("task", f"{agent}_role"):
            if anchor_id in self.graph.states:
                self.graph.add_affinity(output_state_id, anchor_id, 1.0)
        for state_id in input_ids[-4:]:
            if state_id in self.graph.states:
                self.graph.add_affinity(output_state_id, state_id, 0.5)
