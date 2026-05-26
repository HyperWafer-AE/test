"""Redundancy and hot-state characterization for State Access Graphs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .ir import StateAccessGraph


@dataclass(frozen=True)
class RedundancySummary:
    workflow: str
    total_prompt_tokens: int
    unique_state_tokens: int
    input_redundancy_ratio: float
    duplicate_materialization_bytes: int
    dynamic_hot_state_count: int
    dynamic_hot_token_weighted_cost: int
    top_state_share: float
    h1_supported: bool
    h2_supported: bool


def analyze_redundancy(graph: StateAccessGraph, top_k: int = 10) -> dict[str, Any]:
    graph.update_operator_input_tokens()
    rows = state_fanout_rows(graph)
    total_prompt_tokens = sum(op.estimated_input_tokens for op in graph.operators.values())
    consumed_states = [state for state in graph.states.values() if state.consumers]
    unique_state_tokens = sum(state.token_size for state in consumed_states)
    duplicate_bytes = sum(
        state.text_size_bytes * max(0, len(state.consumers) - 1) for state in graph.states.values()
    )
    dynamic_rows = [
        row for row in rows if row["dynamic_hot_candidate"] and row["consumer_count"] > 1
    ]
    dynamic_cost = sum(row["token_weighted_state_fanout"] for row in dynamic_rows)
    hotness_rows = hotness_skew_rows(rows)
    top_share = hotness_rows[min(top_k, len(hotness_rows)) - 1]["cumulative_hotness_fraction"] if hotness_rows else 0.0
    ratio = total_prompt_tokens / unique_state_tokens if unique_state_tokens else 0.0
    summary = RedundancySummary(
        workflow=graph.graph_id,
        total_prompt_tokens=total_prompt_tokens,
        unique_state_tokens=unique_state_tokens,
        input_redundancy_ratio=ratio,
        duplicate_materialization_bytes=duplicate_bytes,
        dynamic_hot_state_count=len(dynamic_rows),
        dynamic_hot_token_weighted_cost=dynamic_cost,
        top_state_share=top_share,
        h1_supported=ratio >= 1.5,
        h2_supported=top_share >= 0.5 if len(hotness_rows) >= top_k else top_share >= 0.4,
    )
    return {
        "summary": asdict(summary),
        "state_fanout": rows,
        "state_hotness": hotness_rows,
        "top_hot_states": hotness_rows[:top_k],
    }


def state_fanout_rows(graph: StateAccessGraph) -> list[dict[str, Any]]:
    rows = []
    for state in graph.states.values():
        consumer_count = len(state.consumers)
        token_weighted = state.token_size * max(0, consumer_count - 1)
        hotness = max(state.static_hotness, token_weighted)
        rows.append(
            {
                "state_id": state.state_id,
                "kind": state.kind,
                "token_size": state.token_size,
                "kv_size_bytes": state.kv_size_bytes,
                "consumer_count": consumer_count,
                "producer": state.producer or "",
                "token_weighted_state_fanout": token_weighted,
                "static_hotness": hotness,
                "dynamic_hotness": state.dynamic_hotness,
                "dynamic_hot_candidate": bool(state.metadata.get("dynamic_hot_candidate")),
                "lifetime_start": state.lifetime_start,
                "lifetime_end": state.lifetime_end,
            }
        )
    rows.sort(key=lambda row: (row["token_weighted_state_fanout"], row["token_size"]), reverse=True)
    return rows


def hotness_skew_rows(fanout_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total_hotness = sum(float(row["static_hotness"]) for row in fanout_rows)
    cumulative = 0.0
    out = []
    for rank, row in enumerate(
        sorted(fanout_rows, key=lambda item: float(item["static_hotness"]), reverse=True), start=1
    ):
        hotness = float(row["static_hotness"])
        cumulative += hotness
        copied = dict(row)
        copied["rank"] = rank
        copied["hotness"] = hotness
        copied["cumulative_hotness"] = cumulative
        copied["cumulative_hotness_fraction"] = cumulative / total_hotness if total_hotness else 0.0
        out.append(copied)
    return out
