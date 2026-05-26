"""Residual redundancy analysis after existing cache/workflow baselines.

This module is deliberately scheduler-free. It asks whether duplicated state
tokens remain after accounting for mechanisms that existing systems plausibly
cover: exact prefix cache, deterministic operator-output cache, and
workflow-aware future-use cache under a capacity limit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .ir import StateAccessGraph, StateNode


@dataclass(frozen=True)
class ResidualAnalysisConfig:
    kvflow_capacity_bytes: int = 512 * 1024 * 1024
    residual_ratio_threshold: float = 1.5
    dynamic_hot_fraction_threshold: float = 0.2


@dataclass(frozen=True)
class ResidualSummary:
    workflow: str
    total_prompt_tokens: int
    unique_state_tokens: int
    raw_duplicated_state_tokens: int
    exact_prefix_cache_covered_tokens: int
    deterministic_operator_output_cache_covered_tokens: int
    future_cache_covered_tokens: int
    residual_non_prefix_non_deterministic_dynamic_state_tokens: int
    raw_redundancy_ratio: float
    residual_redundancy_ratio: float
    dynamic_hot_residual_fraction: float
    residual_token_weighted_fanout: int
    wafer_opportunity_score: float
    decision: str
    decision_reason: str


def analyze_residual_redundancy(
    graph: StateAccessGraph,
    config: ResidualAnalysisConfig | None = None,
) -> dict[str, Any]:
    cfg = config or ResidualAnalysisConfig()
    graph.update_operator_input_tokens()
    graph.compute_lifetimes()

    preliminary_rows = [_initial_state_row(graph, state) for state in graph.states.values()]
    remaining_for_kvflow = [
        row
        for row in preliminary_rows
        if row["raw_duplicated_state_tokens"] > 0
        and not row["prefix_covered"]
        and not row["operator_cache_covered"]
    ]
    kvflow_state_ids = _select_kvflow_states(graph, remaining_for_kvflow, cfg.kvflow_capacity_bytes)

    rows = []
    totals = {
        "raw_duplicated_state_tokens": 0,
        "exact_prefix_cache_covered_tokens": 0,
        "deterministic_operator_output_cache_covered_tokens": 0,
        "future_cache_covered_tokens": 0,
        "residual_non_prefix_non_deterministic_dynamic_state_tokens": 0,
    }
    dynamic_residual_tokens = 0
    for row in preliminary_rows:
        row = dict(row)
        raw_tokens = int(row["raw_duplicated_state_tokens"])
        if row["prefix_covered"]:
            prefix_tokens = raw_tokens
            operator_tokens = 0
            kvflow_tokens = 0
            residual_tokens = 0
            reason = "covered_by_exact_prefix_cache"
        elif row["operator_cache_covered"]:
            prefix_tokens = 0
            operator_tokens = raw_tokens
            kvflow_tokens = 0
            residual_tokens = 0
            reason = "covered_by_deterministic_operator_output_cache"
        elif row["state_id"] in kvflow_state_ids:
            prefix_tokens = 0
            operator_tokens = 0
            kvflow_tokens = raw_tokens
            residual_tokens = 0
            reason = "covered_by_kvflow_future_cache_capacity"
        elif raw_tokens > 0:
            prefix_tokens = 0
            operator_tokens = 0
            kvflow_tokens = 0
            residual_tokens = raw_tokens
            reason = _residual_reason(graph.states[str(row["state_id"])])
        else:
            prefix_tokens = 0
            operator_tokens = 0
            kvflow_tokens = 0
            residual_tokens = 0
            reason = "no_repeated_consumers"

        row.update(
            {
                "prefix_covered_tokens": prefix_tokens,
                "operator_cache_covered_tokens": operator_tokens,
                "kvflow_cache_covered": row["state_id"] in kvflow_state_ids,
                "kvflow_cache_covered_tokens": kvflow_tokens,
                "residual_candidate": residual_tokens > 0,
                "residual_token_weighted_fanout": residual_tokens,
                "reason": reason,
            }
        )
        for key in totals:
            if key == "exact_prefix_cache_covered_tokens":
                totals[key] += prefix_tokens
            elif key == "deterministic_operator_output_cache_covered_tokens":
                totals[key] += operator_tokens
            elif key == "future_cache_covered_tokens":
                totals[key] += kvflow_tokens
            elif key == "residual_non_prefix_non_deterministic_dynamic_state_tokens":
                totals[key] += residual_tokens
            else:
                totals[key] += raw_tokens
        if row["dynamic_hot_candidate"]:
            dynamic_residual_tokens += residual_tokens
        rows.append(row)

    total_prompt_tokens = sum(op.estimated_input_tokens for op in graph.operators.values())
    unique_state_tokens = sum(state.token_size for state in graph.states.values() if state.consumers)
    raw_ratio = total_prompt_tokens / unique_state_tokens if unique_state_tokens else 0.0
    residual_tokens = totals["residual_non_prefix_non_deterministic_dynamic_state_tokens"]
    residual_ratio = (unique_state_tokens + residual_tokens) / unique_state_tokens if unique_state_tokens else 0.0
    dynamic_fraction = dynamic_residual_tokens / residual_tokens if residual_tokens else 0.0
    opportunity = _wafer_opportunity_score(residual_ratio, dynamic_fraction, residual_tokens, unique_state_tokens)
    decision, decision_reason = _decision(
        residual_ratio,
        dynamic_fraction,
        cfg.residual_ratio_threshold,
        cfg.dynamic_hot_fraction_threshold,
    )

    summary = ResidualSummary(
        workflow=graph.graph_id,
        total_prompt_tokens=total_prompt_tokens,
        unique_state_tokens=unique_state_tokens,
        raw_duplicated_state_tokens=totals["raw_duplicated_state_tokens"],
        exact_prefix_cache_covered_tokens=totals["exact_prefix_cache_covered_tokens"],
        deterministic_operator_output_cache_covered_tokens=totals[
            "deterministic_operator_output_cache_covered_tokens"
        ],
        future_cache_covered_tokens=totals["future_cache_covered_tokens"],
        residual_non_prefix_non_deterministic_dynamic_state_tokens=residual_tokens,
        raw_redundancy_ratio=raw_ratio,
        residual_redundancy_ratio=residual_ratio,
        dynamic_hot_residual_fraction=dynamic_fraction,
        residual_token_weighted_fanout=residual_tokens,
        wafer_opportunity_score=opportunity,
        decision=decision,
        decision_reason=decision_reason,
    )
    rows.sort(
        key=lambda item: (
            int(item["residual_token_weighted_fanout"]),
            int(item["raw_duplicated_state_tokens"]),
        ),
        reverse=True,
    )
    return {"summary": asdict(summary), "state_rows": rows}


def _initial_state_row(graph: StateAccessGraph, state: StateNode) -> dict[str, Any]:
    consumer_count = len(state.consumers)
    raw_tokens = state.token_size * max(0, consumer_count - 1)
    producer = graph.operators.get(state.producer) if state.producer else None
    operator_cache_covered = bool(
        raw_tokens > 0
        and state.producer
        and state.deterministic
        and producer is not None
        and producer.deterministic
    )
    prefix_covered = bool(
        raw_tokens > 0
        and state.producer is None
        and state.prefix_compatible
        and state.prompt_position is not None
    )
    return {
        "workflow": graph.graph_id,
        "state_id": state.state_id,
        "kind": state.kind,
        "token_size": state.token_size,
        "kv_size_bytes": state.kv_size_bytes,
        "consumer_count": consumer_count,
        "producer": state.producer or "",
        "producer_deterministic": bool(producer.deterministic) if producer is not None else "",
        "state_deterministic": state.deterministic,
        "prefix_compatible": state.prefix_compatible,
        "kv_cacheable": state.kv_cacheable,
        "prompt_position": "" if state.prompt_position is None else state.prompt_position,
        "dynamic_hot_candidate": bool(state.metadata.get("dynamic_hot_candidate")),
        "raw_duplicated_state_tokens": raw_tokens,
        "prefix_covered": prefix_covered,
        "operator_cache_covered": operator_cache_covered,
        "kvflow_cache_covered": False,
        "residual_candidate": False,
        "reason": "",
    }


def _select_kvflow_states(
    graph: StateAccessGraph,
    rows: list[dict[str, Any]],
    capacity_bytes: int,
) -> set[str]:
    selected: set[str] = set()
    used = 0
    ordered = sorted(
        rows,
        key=lambda row: (
            int(row["raw_duplicated_state_tokens"]),
            -graph.states[str(row["state_id"])].materialized_size_bytes,
        ),
        reverse=True,
    )
    for row in ordered:
        state = graph.states[str(row["state_id"])]
        size = state.materialized_size_bytes
        if used + size <= capacity_bytes:
            selected.add(state.state_id)
            used += size
    return selected


def _residual_reason(state: StateNode) -> str:
    reasons = []
    if not state.prefix_compatible:
        reasons.append("non_prefix")
    if state.producer is not None and not state.deterministic:
        reasons.append("non_deterministic_output")
    if state.metadata.get("dynamic_hot_candidate"):
        reasons.append("dynamic_hot_candidate")
    if state.kv_size_bytes > 0:
        reasons.append("not_admitted_by_kvflow_capacity")
    return "+".join(reasons) if reasons else "not_covered_by_existing_cache_models"


def _wafer_opportunity_score(
    residual_ratio: float,
    dynamic_fraction: float,
    residual_tokens: int,
    unique_state_tokens: int,
) -> float:
    if unique_state_tokens <= 0:
        return 0.0
    residual_weight = residual_tokens / unique_state_tokens
    return max(0.0, residual_ratio - 1.0) * (0.5 + dynamic_fraction) * residual_weight


def _decision(
    residual_ratio: float,
    dynamic_fraction: float,
    residual_ratio_threshold: float,
    dynamic_fraction_threshold: float,
) -> tuple[str, str]:
    if residual_ratio < residual_ratio_threshold:
        return (
            "abandon_or_pivot",
            f"residual_redundancy_ratio={residual_ratio:.2f} is below {residual_ratio_threshold:.2f}",
        )
    if dynamic_fraction < dynamic_fraction_threshold:
        return (
            "weak_paper_direction",
            f"dynamic_hot_residual_fraction={dynamic_fraction:.2f} is below {dynamic_fraction_threshold:.2f}",
        )
    return (
        "continue_investigation",
        "residual redundancy and dynamic residual fraction exceed screening thresholds",
    )
