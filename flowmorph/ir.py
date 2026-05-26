"""PhaseDAG IR for FlowMorph characterization."""

from __future__ import annotations

import csv
import json
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PhaseOperator:
    op_id: str
    dependencies: list[str] = field(default_factory=list)
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    prefill_cost: float = 0.0
    decode_cost: float = 0.0
    local_tool_cost: float = 0.0
    criticality: float = 1.0
    earliest_ready_time: float = 0.0
    kind: str = "llm"
    role: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_cost(self) -> float:
        return self.prefill_cost + self.decode_cost + self.local_tool_cost


class PhaseDAG:
    def __init__(self, graph_id: str = "phase_dag", metadata: dict[str, Any] | None = None) -> None:
        self.graph_id = graph_id
        self.metadata = metadata or {}
        self.operators: dict[str, PhaseOperator] = {}

    def add_operator(self, operator: PhaseOperator) -> PhaseOperator:
        if operator.op_id in self.operators:
            raise ValueError(f"duplicate operator {operator.op_id}")
        missing = [dep for dep in operator.dependencies if dep not in self.operators]
        if missing:
            # Dependencies can be added out of order only if the caller later validates.
            operator.metadata = dict(operator.metadata)
            operator.metadata["unresolved_dependencies"] = missing
        self.operators[operator.op_id] = operator
        return operator

    def dependencies(self, op_id: str) -> list[str]:
        return list(self.operators[op_id].dependencies)

    def successors(self) -> dict[str, list[str]]:
        succ: dict[str, list[str]] = defaultdict(list)
        for op in self.operators.values():
            for dep in op.dependencies:
                succ[dep].append(op.op_id)
        return {key: sorted(value) for key, value in succ.items()}

    def topological_order(self) -> list[str]:
        indegree = {op_id: 0 for op_id in self.operators}
        succ: dict[str, list[str]] = defaultdict(list)
        for op in self.operators.values():
            for dep in op.dependencies:
                if dep not in self.operators:
                    raise KeyError(f"operator {op.op_id} depends on missing {dep}")
                indegree[op.op_id] += 1
                succ[dep].append(op.op_id)
        queue = deque(sorted(op_id for op_id, degree in indegree.items() if degree == 0))
        order: list[str] = []
        while queue:
            op_id = queue.popleft()
            order.append(op_id)
            for dst in sorted(succ[op_id]):
                indegree[dst] -= 1
                if indegree[dst] == 0:
                    queue.append(dst)
        if len(order) != len(self.operators):
            raise ValueError("PhaseDAG contains a cycle")
        return order

    def compute_earliest_ready_times(self) -> None:
        finish_times: dict[str, float] = {}
        for op_id in self.topological_order():
            op = self.operators[op_id]
            ready = max((finish_times[dep] for dep in op.dependencies), default=0.0)
            op.earliest_ready_time = ready
            finish_times[op_id] = ready + op.total_cost

    def critical_path_length(self) -> float:
        self.compute_earliest_ready_times()
        return max(
            (op.earliest_ready_time + op.total_cost for op in self.operators.values()),
            default=0.0,
        )

    def total_work(self) -> float:
        return sum(op.total_cost for op in self.operators.values())

    def to_operator_rows(self) -> list[dict[str, Any]]:
        return [asdict(self.operators[op_id]) | {"total_cost": self.operators[op_id].total_cost} for op_id in self.topological_order()]

    def export_csv(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        _write_csv(out / "phase_operators.csv", self.to_operator_rows())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )
