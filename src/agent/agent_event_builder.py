from typing import Dict, List, Optional

from src.scheduling.event_notation import (
    communication_notation,
    external_wait_notation,
    fixed_compute_notation,
)

from .agent_state_graph import AgentStateGraph
from .metrics import AgentMetrics
from .phase_detector import detect_phase
from .placement import choose_exec_location, choose_state_location, node_of_module
from .state_node import ExecNode, StateNode


STATIC_DEFAULT_TOKENS = {
    "system": 512,
    "task": 512,
}


class AgentEventBuilder:
    def __init__(self, hardware_platform, model_profile, state_manager, enable_prefetch: bool = True):
        self.hardware_platform = hardware_platform
        self.model_profile = model_profile
        self.state_manager = state_manager
        self.enable_prefetch = enable_prefetch
        self.graph = AgentStateGraph()
        self.metrics = AgentMetrics()
        self.events_dict: Dict[int, object] = {}
        self._next_event_tag = 0
        self._next_exec_id = 0
        self._last_agent_event: Dict[str, int] = {}

    def build(self, trace: List[dict]):
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

    def _refresh_policy(self, phase_score: dict):
        self.state_manager.update(self.graph, phase_score, self.graph.current_step)
        self.metrics.evicted_kv_bytes += self.state_manager.last_evicted_bytes

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
        self.graph.build_or_get_state(
            state_id=trace_event["state_id"],
            state_type=trace_event.get("state_type", "dialogue_delta"),
            owner=trace_event.get("owner", trace_event.get("agent", "shared")),
            token_len=token_len,
            kv_bytes=int(trace_event.get("kv_bytes", self.model_profile.kv_bytes(token_len))),
            semantic_id=trace_event.get("semantic_id"),
            exact_kv_id=trace_event.get("exact_kv_id"),
            metadata=trace_event.get("metadata"),
        )
        self._refresh_policy(self._phase_score())

    def _handle_llm(self, trace_event: dict):
        agent = trace_event["agent"]
        input_ids = list(trace_event.get("input_state_ids", []))
        input_states = [self._infer_state(state_id, agent) for state_id in input_ids]
        resident_before = {state.state_id: state.resident for state in input_states}
        loc_before = {state.state_id: state.loc for state in input_states}

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

        self._refresh_policy(phase_score)
        exec_loc = choose_exec_location(input_states, self.hardware_platform, "tensorcore")
        exec_node.assigned_loc = exec_loc
        target_node = node_of_module(exec_loc)

        parent_tag = self._last_agent_event.get(agent)
        if self.state_manager.policy_name != "nocache":
            for state in input_states:
                if not resident_before.get(state.state_id, False):
                    continue
                old_node = loc_before.get(state.state_id)
                if old_node is None:
                    state.loc = target_node
                    continue
                if old_node == target_node:
                    continue
                parent_tag = self._emit_migration(state, old_node, target_node, parent_tag)

        total_input_tokens = sum(state.token_len for state in input_states)
        effective_prefill_tokens = self._effective_prefill_tokens(
            input_states=input_states,
            resident_before=resident_before,
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
        output_state.loc = target_node
        self.graph.add_generation(exec_id, output_state.state_id)

        for state in input_states:
            if state.resident:
                state.loc = target_node

        self.metrics.total_input_tokens += total_input_tokens
        self.metrics.append_tokens += append_tokens
        self.metrics.effective_prefill_tokens += effective_prefill_tokens
        self.metrics.decode_tokens += output_tokens
        self.metrics.num_llm_steps += 1
        self._refresh_policy(phase_score)

    def _effective_prefill_tokens(self, input_states, resident_before: dict, append_tokens: int) -> int:
        if self.state_manager.policy_name == "nocache":
            self.metrics.cache_misses += len(input_states)
            return sum(state.token_len for state in input_states) + append_tokens

        missing_tokens = 0
        for state in input_states:
            if resident_before.get(state.state_id, False):
                self.metrics.cache_hits += 1
            else:
                self.metrics.cache_misses += 1
                missing_tokens += state.token_len
        return missing_tokens + append_tokens

    def _emit_migration(self, state: StateNode, old_node, new_node, parent_tag: Optional[int]) -> int:
        tag = self._new_tag()
        migration = communication_notation(
            comm_name="kv_migrate",
            comm_tag=tag,
            source_location=old_node,
            target_location=new_node,
            comm_bytes=state.kv_bytes,
        )
        self._add_event(migration)
        self._add_dependency(parent_tag, migration)
        self.metrics.kv_migration_bytes += state.kv_bytes
        state.loc = new_node
        return tag

    def _handle_tool(self, trace_event: dict):
        agent = trace_event["agent"]
        tool_name = trace_event.get("tool", "tool")
        phase_score = self._phase_score()
        phase_name = self._phase_name(phase_score)
        exec_id = self._new_exec_id("tool")
        latency = int(trace_event.get("latency", trace_event.get("tool_latency", 0)))
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

        wait_tag = self._new_tag()
        wait = external_wait_notation(
            wait_name=f"tool_{tool_name}",
            wait_tag=wait_tag,
            duration=max(0, latency),
            metadata={"agent": agent, "exec_id": exec_id, "tool": tool_name},
        )
        self._add_event(wait)
        self._add_dependency(self._last_agent_event.get(agent), wait)
        self._last_agent_event[agent] = wait_tag

        state_id = trace_event.get("new_state_id") or f"{agent}_tool_{exec_id}_out"
        state_type = trace_event.get("new_state_type", "tool_observation")
        state = self.graph.build_or_get_state(
            state_id=state_id,
            state_type=state_type,
            owner=agent,
            token_len=output_tokens,
            kv_bytes=self.model_profile.kv_bytes(output_tokens),
            metadata={"producer_exec": exec_id, "tool": tool_name},
        )
        if self.graph.hot_states():
            anchor = self.graph.hot_states()[0].loc
            state.loc = anchor
        self.graph.add_generation(exec_id, state.state_id)

        self.metrics.tool_wait_cycles += latency
        self.metrics.num_tool_steps += 1
        self._refresh_policy(phase_score)

