from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

from .state_node import ExecNode, StateNode


class AgentStateGraph:
    def __init__(self, history_window: int = 64):
        self.states: Dict[str, StateNode] = {}
        self.execs: Dict[str, ExecNode] = {}
        self.control_edges: Dict[str, Set[str]] = defaultdict(set)
        self.dep_edges: Dict[str, Set[str]] = defaultdict(set)
        self.gen_edges: Dict[str, Set[str]] = defaultdict(set)
        self.affinity_edges: Dict[Tuple[str, str], float] = {}
        self.recent_events = deque(maxlen=history_window)
        self.current_step: int = 0

    def add_state(self, state: StateNode) -> StateNode:
        self.states[state.state_id] = state
        return state

    def get_state(self, state_id: str) -> Optional[StateNode]:
        return self.states.get(state_id)

    def add_exec(self, exec_node: ExecNode) -> ExecNode:
        self.execs[exec_node.exec_id] = exec_node
        return exec_node

    def add_dependency(self, state_id: str, exec_id: str):
        self.dep_edges[state_id].add(exec_id)
        if exec_id in self.execs and state_id not in self.execs[exec_id].input_states:
            self.execs[exec_id].input_states.append(state_id)

    def add_generation(self, exec_id: str, state_id: str):
        self.gen_edges[exec_id].add(state_id)
        if exec_id in self.execs and state_id not in self.execs[exec_id].output_states:
            self.execs[exec_id].output_states.append(state_id)

    def add_control(self, src_exec_id: str, dst_exec_id: str):
        self.control_edges[src_exec_id].add(dst_exec_id)

    def add_affinity(self, a: str, b: str, weight: float):
        key = tuple(sorted((a, b)))
        self.affinity_edges[key] = self.affinity_edges.get(key, 0.0) + float(weight)

    def touch_state(self, state_id: str):
        state = self.states.get(state_id)
        if state is None:
            return
        state.last_access = self.current_step
        state.access_count += 1

    def input_states(self, exec_id: str) -> List[StateNode]:
        exec_node = self.execs.get(exec_id)
        if exec_node is None:
            return []
        return [self.states[sid] for sid in exec_node.input_states if sid in self.states]

    def hot_states(self) -> List[StateNode]:
        return [state for state in self.states.values() if state.resident and state.tier == "hot"]

    def build_or_get_state(
        self,
        state_id: str,
        state_type: str,
        owner: str,
        token_len: int,
        kv_bytes: int,
        semantic_id: Optional[str] = None,
        exact_kv_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> StateNode:
        state = self.states.get(state_id)
        if state is not None:
            state.last_access = self.current_step
            state.access_count += 1
            if semantic_id is not None:
                state.semantic_id = semantic_id
            if exact_kv_id is not None:
                state.exact_kv_id = exact_kv_id
            if metadata:
                state.metadata.update(metadata)
            return state

        state = StateNode(
            state_id=state_id,
            state_type=state_type,
            owner=owner,
            token_len=int(token_len),
            kv_bytes=int(kv_bytes),
            birth_step=self.current_step,
            last_access=self.current_step,
            access_count=1,
            semantic_id=semantic_id,
            exact_kv_id=exact_kv_id,
            metadata=metadata or {},
        )
        self.states[state_id] = state
        return state

