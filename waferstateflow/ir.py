"""State Access Graph IR for WaferStateFlow.

The IR keeps prompt-visible state as first-class graph nodes. Edges are
directional and preserve whether an operator consumes or produces a state.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass
class StateNode:
    state_id: str
    kind: str
    token_size: int
    kv_size_bytes: int = 0
    producer: Optional[str] = None
    consumers: list[str] = field(default_factory=list)
    lifetime_start: float | int = 0
    lifetime_end: float | int = 0
    materialized_form: str = "inline"
    location: Optional[str] = None
    prefix_compatible: bool = False
    kv_cacheable: bool = False
    prompt_position: Optional[int] = None
    static_hotness: float = 0.0
    dynamic_hotness: float = 0.0
    deterministic: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text_size_bytes(self) -> int:
        return int(self.token_size * 4)

    @property
    def materialized_size_bytes(self) -> int:
        return max(self.text_size_bytes, self.kv_size_bytes)


@dataclass
class OperatorNode:
    op_id: str
    kind: str
    role: str
    input_states: list[str] = field(default_factory=list)
    output_states: list[str] = field(default_factory=list)
    deps: list[str] = field(default_factory=list)
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    phase_profile: str = "mixed"
    deterministic: bool = False
    criticality: float = 1.0
    ready_time: float | int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AccessEdge:
    src: str
    dst: str
    edge_type: str
    token_size: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.edge_type not in {"consume", "produce"}:
            raise ValueError(f"unknown edge_type {self.edge_type!r}")


@dataclass
class WorkflowTrace:
    workflow_name: str
    seed: int
    graph: "StateAccessGraph"
    metadata: dict[str, Any] = field(default_factory=dict)


class StateAccessGraph:
    """Mutable bipartite graph of states and operators."""

    def __init__(self, graph_id: str = "workflow", metadata: Optional[dict[str, Any]] = None) -> None:
        self.graph_id = graph_id
        self.metadata = metadata or {}
        self.states: dict[str, StateNode] = {}
        self.operators: dict[str, OperatorNode] = {}
        self.edges: list[AccessEdge] = []

    def add_state(self, state: StateNode) -> StateNode:
        if state.state_id in self.states:
            raise ValueError(f"duplicate state_id {state.state_id}")
        if state.token_size < 0 or state.kv_size_bytes < 0:
            raise ValueError("state sizes must be non-negative")
        self.states[state.state_id] = state
        return state

    def add_operator(self, operator: OperatorNode) -> OperatorNode:
        if operator.op_id in self.operators:
            raise ValueError(f"duplicate op_id {operator.op_id}")
        if operator.estimated_input_tokens < 0 or operator.estimated_output_tokens < 0:
            raise ValueError("operator token estimates must be non-negative")
        self.operators[operator.op_id] = operator
        return operator

    def add_access_edge(self, edge: AccessEdge) -> AccessEdge:
        if edge.edge_type == "consume":
            if edge.src not in self.states:
                raise KeyError(f"missing state {edge.src}")
            if edge.dst not in self.operators:
                raise KeyError(f"missing operator {edge.dst}")
            state = self.states[edge.src]
            operator = self.operators[edge.dst]
            if edge.dst not in state.consumers:
                state.consumers.append(edge.dst)
            if edge.src not in operator.input_states:
                operator.input_states.append(edge.src)
        else:
            if edge.src not in self.operators:
                raise KeyError(f"missing operator {edge.src}")
            if edge.dst not in self.states:
                raise KeyError(f"missing state {edge.dst}")
            operator = self.operators[edge.src]
            state = self.states[edge.dst]
            if state.producer is not None and state.producer != edge.src:
                raise ValueError(f"state {edge.dst} already has producer {state.producer}")
            state.producer = edge.src
            if edge.dst not in operator.output_states:
                operator.output_states.append(edge.dst)
        self.edges.append(edge)
        return edge

    def connect_state_to_operator(self, state_id: str, op_id: str) -> AccessEdge:
        state = self.states[state_id]
        return self.add_access_edge(
            AccessEdge(src=state_id, dst=op_id, edge_type="consume", token_size=state.token_size)
        )

    def connect_operator_to_state(self, op_id: str, state_id: str) -> AccessEdge:
        state = self.states[state_id]
        return self.add_access_edge(
            AccessEdge(src=op_id, dst=state_id, edge_type="produce", token_size=state.token_size)
        )

    def state_consumers(self, state_id: str) -> list[str]:
        return list(self.states[state_id].consumers)

    def operator_inputs(self, op_id: str) -> list[str]:
        return list(self.operators[op_id].input_states)

    def operator_outputs(self, op_id: str) -> list[str]:
        return list(self.operators[op_id].output_states)

    def producer(self, state_id: str) -> Optional[str]:
        return self.states[state_id].producer

    def state_fanout(self, state_id: str) -> int:
        return len(self.states[state_id].consumers)

    def producer_consumer_fanout(self, op_id: str) -> int:
        return sum(self.state_fanout(state_id) for state_id in self.operators[op_id].output_states)

    def operator_dependencies(self, op_id: str) -> list[str]:
        deps = set(self.operators[op_id].deps)
        for state_id in self.operators[op_id].input_states:
            producer = self.states[state_id].producer
            if producer is not None and producer != op_id:
                deps.add(producer)
        return sorted(deps)

    def ready_operators(self, completed_ops: Iterable[str]) -> list[str]:
        done = set(completed_ops)
        ready = []
        for op_id in self.operator_topological_order():
            if op_id in done:
                continue
            if all(dep in done for dep in self.operator_dependencies(op_id)):
                ready.append(op_id)
        return ready

    def topological_order(self) -> list[str]:
        """Return a topological order across state and operator nodes."""

        nodes = set(self.states) | set(self.operators)
        indegree = {node: 0 for node in nodes}
        adjacency: dict[str, list[str]] = defaultdict(list)
        for edge in self.edges:
            adjacency[edge.src].append(edge.dst)
            indegree[edge.dst] += 1
        queue = deque(sorted(node for node, degree in indegree.items() if degree == 0))
        order: list[str] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for dst in sorted(adjacency[node]):
                indegree[dst] -= 1
                if indegree[dst] == 0:
                    queue.append(dst)
        if len(order) != len(nodes):
            raise ValueError("StateAccessGraph contains a cycle")
        return order

    def operator_topological_order(self) -> list[str]:
        ordered_nodes = self.topological_order()
        return [node for node in ordered_nodes if node in self.operators]

    def compute_lifetimes(self) -> None:
        op_rank = {op_id: i for i, op_id in enumerate(self.operator_topological_order())}
        for state in self.states.values():
            if state.producer is None:
                start = 0
            else:
                start = op_rank.get(state.producer, 0)
            end = start
            if state.consumers:
                end = max(op_rank[op_id] for op_id in state.consumers)
            state.lifetime_start = start
            state.lifetime_end = end

    def update_operator_input_tokens(self) -> None:
        for operator in self.operators.values():
            operator.estimated_input_tokens = sum(
                self.states[state_id].token_size for state_id in operator.input_states
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "metadata": self.metadata,
            "states": [asdict(state) for state in self.states.values()],
            "operators": [asdict(operator) for operator in self.operators.values()],
            "edges": [asdict(edge) for edge in self.edges],
        }

    def export_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    def export_csv(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._write_csv(out_dir / "state_nodes.csv", [asdict(s) for s in self.states.values()])
        self._write_csv(out_dir / "operator_nodes.csv", [asdict(o) for o in self.operators.values()])
        self._write_csv(out_dir / "access_edges.csv", [asdict(e) for e in self.edges])

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        normalized = []
        for row in rows:
            clean = {}
            for key, value in row.items():
                if isinstance(value, (list, dict)):
                    clean[key] = json.dumps(value, sort_keys=True)
                else:
                    clean[key] = value
            normalized.append(clean)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(normalized[0].keys()))
            writer.writeheader()
            writer.writerows(normalized)
