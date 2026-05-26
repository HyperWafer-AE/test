"""Static and dynamic state hotness estimation."""

from __future__ import annotations

from dataclasses import dataclass

from .ir import OperatorNode, StateAccessGraph, StateNode


@dataclass(frozen=True)
class HotnessConfig:
    alpha: float = 0.65
    promote_threshold: float = 2500.0
    demote_threshold: float = 900.0
    criticality_floor: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.promote_threshold <= self.demote_threshold:
            raise ValueError("promote_threshold must be greater than demote_threshold")


class HotnessTracker:
    def __init__(self, graph: StateAccessGraph, config: HotnessConfig | None = None) -> None:
        self.graph = graph
        self.config = config or HotnessConfig()
        for state in graph.states.values():
            state.static_hotness = self.static_hotness(state)
        self.dynamic_hotness: dict[str, float] = {
            state_id: state.dynamic_hotness for state_id, state in graph.states.items()
        }
        self.state_class: dict[str, str] = {
            state_id: "hot" if state.static_hotness >= self.config.promote_threshold else "cold"
            for state_id, state in graph.states.items()
        }

    def static_hotness(self, state: StateNode) -> float:
        consumer_ops = [self.graph.operators[op_id] for op_id in state.consumers]
        criticality = sum(max(self.config.criticality_floor, op.criticality) for op in consumer_ops)
        future_accesses = len(consumer_ops)
        return state.token_size * max(0, future_accesses - 1) * max(1.0, criticality / 2.0)

    def observe_access(
        self,
        state_id: str,
        operator: OperatorNode | None = None,
        access_cost: float = 1.0,
    ) -> float:
        state = self.graph.states[state_id]
        criticality = operator.criticality if operator is not None else 1.0
        observed = state.token_size * access_cost * max(self.config.criticality_floor, criticality)
        old = self.dynamic_hotness.get(state_id, 0.0)
        new = self.config.alpha * old + (1.0 - self.config.alpha) * observed
        self.dynamic_hotness[state_id] = new
        state.dynamic_hotness = new
        self._update_class(state_id, new)
        return new

    def cool(self, state_id: str) -> float:
        old = self.dynamic_hotness.get(state_id, 0.0)
        new = self.config.alpha * old
        self.dynamic_hotness[state_id] = new
        self.graph.states[state_id].dynamic_hotness = new
        self._update_class(state_id, new)
        return new

    def combined_hotness(self, state_id: str) -> float:
        state = self.graph.states[state_id]
        return max(self.static_hotness(state), self.dynamic_hotness.get(state_id, 0.0))

    def classify(self, state_id: str) -> str:
        return self.state_class[state_id]

    def _update_class(self, state_id: str, value: float) -> None:
        current = self.state_class.get(state_id, "cold")
        if current != "hot" and value >= self.config.promote_threshold:
            self.state_class[state_id] = "hot"
        elif current == "hot" and value <= self.config.demote_threshold:
            self.state_class[state_id] = "cold"


def initialize_static_hotness(graph: StateAccessGraph, config: HotnessConfig | None = None) -> None:
    tracker = HotnessTracker(graph, config)
    for state in graph.states.values():
        state.static_hotness = tracker.static_hotness(state)
