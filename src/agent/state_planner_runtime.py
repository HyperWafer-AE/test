from .state_lifecycle import is_static_shared


class StatePlannerRuntime:
    """Wrap persistent-state planning and node-memory synchronization."""

    def __init__(self, state_manager, node_memory, metrics, static_replicator):
        self.state_manager = state_manager
        self.node_memory = node_memory
        self.metrics = metrics
        self._static_replicator = static_replicator

    def update(self, graph, phase_score, current_step, future_index=None, oracle_future=False, demand_predictor=None):
        self.state_manager.update(
            graph,
            phase_score,
            current_step,
            future_index=future_index,
            oracle_future=oracle_future,
            demand_predictor=demand_predictor,
        )
        self.metrics.evicted_kv_bytes += self.state_manager.last_evicted_bytes
        self.sync_node_memory(graph)

    def sync_node_memory(self, graph):
        for state in sorted(graph.states.values(), key=lambda item: item.state_id):
            if not state.resident:
                self.node_memory.remove(state)
            elif state.loc is not None and state.state_id not in self.node_memory.state_to_node:
                if self.node_memory.can_place(state, state.loc):
                    self.node_memory.place(state, state.loc)
            if state.resident and is_static_shared(state):
                self._static_replicator(state)

    def resident_states(self, graph):
        return [state for state in graph.states.values() if state.resident]
