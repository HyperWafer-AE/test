"""Converters from synthetic State Access Graphs to FlowMorph PhaseDAGs."""

from __future__ import annotations

from dataclasses import dataclass

from waferstateflow.ir import StateAccessGraph
from waferstateflow.workflow_generators import generate_workflow

from .ir import PhaseDAG, PhaseOperator


@dataclass(frozen=True)
class PhaseCostModel:
    prefill_cost_per_token: float = 0.002
    decode_cost_per_token: float = 0.006
    local_cost_per_token: float = 0.0005
    base_local_tool_cost: float = 0.1


def phase_dag_from_workflow_name(
    workflow: str,
    cost_model: PhaseCostModel | None = None,
    **workflow_kwargs: object,
) -> PhaseDAG:
    graph = generate_workflow(workflow, **workflow_kwargs)
    return phase_dag_from_state_graph(graph, cost_model)


def phase_dag_from_state_graph(
    graph: StateAccessGraph,
    cost_model: PhaseCostModel | None = None,
) -> PhaseDAG:
    model = cost_model or PhaseCostModel()
    graph.update_operator_input_tokens()
    graph.compute_lifetimes()
    dag = PhaseDAG(graph.graph_id, metadata={"source": "waferstateflow_synthetic"})
    for op_id in graph.operator_topological_order():
        op = graph.operators[op_id]
        deps = graph.operator_dependencies(op_id)
        output_tokens = op.estimated_output_tokens
        if output_tokens <= 0:
            output_tokens = sum(graph.states[state_id].token_size for state_id in op.output_states)
        if op.kind == "llm":
            prefill_cost = op.estimated_input_tokens * model.prefill_cost_per_token
            decode_cost = output_tokens * model.decode_cost_per_token
            local_tool_cost = 0.0
        elif op.kind in {"retrieval", "tool"}:
            prefill_cost = 0.0
            decode_cost = 0.0
            local_tool_cost = model.base_local_tool_cost + op.estimated_input_tokens * model.local_cost_per_token
        else:
            prefill_cost = 0.0
            decode_cost = 0.0
            local_tool_cost = model.base_local_tool_cost + op.estimated_input_tokens * model.local_cost_per_token
        dag.add_operator(
            PhaseOperator(
                op_id=op_id,
                dependencies=deps,
                estimated_input_tokens=op.estimated_input_tokens,
                estimated_output_tokens=output_tokens,
                prefill_cost=prefill_cost,
                decode_cost=decode_cost,
                local_tool_cost=local_tool_cost,
                criticality=op.criticality,
                earliest_ready_time=float(op.ready_time),
                kind=op.kind,
                role=op.role,
                metadata={"source_role": op.role, "phase_profile": op.phase_profile},
            )
        )
    dag.compute_earliest_ready_times()
    return dag
