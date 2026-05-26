"""Phase/resource irregularity metrics for FlowMorph."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from .ir import PhaseDAG, PhaseOperator


@dataclass(frozen=True)
class FlowMorphConfig:
    fixed_worker_count: int = 8
    prefill_partition_workers: int = 4
    decode_partition_workers: int = 4
    frontier_cv_threshold: float = 0.35
    phase_variation_threshold: float = 0.25
    parallel_slack_threshold: float = 2.0


@dataclass(frozen=True)
class FlowMorphSummary:
    workflow: str
    operator_count: int
    max_frontier_width: int
    median_frontier_width: float
    mean_frontier_width: float
    frontier_width_cv: float
    width_drop_ratio: float
    wide_stage_work_fraction: float
    narrow_critical_stage_fraction: float
    mean_phase_mix_entropy: float
    phase_mix_variation: float
    critical_path_length: float
    total_work: float
    total_work_to_critical_path_ratio: float
    parallel_slack: float
    critical_path_serial_fraction: float
    fixed_worker_idle_fraction: float
    fixed_prefill_decode_partition_imbalance: float
    frontier_morphing_opportunity: str
    phase_morphing_opportunity: str
    combined_opportunity: str
    opportunity_taxonomy: str
    decision: str
    decision_reason: str


def characterize_phase_irregularity(
    dag: PhaseDAG,
    config: FlowMorphConfig | None = None,
) -> dict[str, Any]:
    cfg = config or FlowMorphConfig()
    dag.compute_earliest_ready_times()
    timeline = _frontier_timeline(dag)
    frontier_widths = [int(row["frontier_width"]) for row in timeline]
    phase_mix_vectors = [row["phase_mix_vector"] for row in timeline]
    critical_path = dag.critical_path_length()
    total_work = dag.total_work()
    phase_variation = _phase_mix_variation(phase_mix_vectors)
    frontier_cv = _coefficient_of_variation(frontier_widths)
    median_width = _median(frontier_widths)
    max_width = max(frontier_widths, default=0)
    width_drop_ratio = max_width / median_width if median_width > 0 else 0.0
    wide_stage_work_fraction = _wide_stage_work_fraction(timeline, median_width, total_work)
    narrow_critical_stage_fraction = _narrow_critical_stage_fraction(dag, median_width, critical_path)
    parallel_slack = total_work / critical_path if critical_path > 0 else 0.0
    serial_fraction = critical_path / total_work if total_work > 0 else 0.0
    opportunity = _opportunity_taxonomy(frontier_cv, phase_variation, parallel_slack, cfg)
    summary = FlowMorphSummary(
        workflow=dag.graph_id,
        operator_count=len(dag.operators),
        max_frontier_width=max_width,
        median_frontier_width=median_width,
        mean_frontier_width=_mean(frontier_widths),
        frontier_width_cv=frontier_cv,
        width_drop_ratio=width_drop_ratio,
        wide_stage_work_fraction=wide_stage_work_fraction,
        narrow_critical_stage_fraction=narrow_critical_stage_fraction,
        mean_phase_mix_entropy=_mean([float(row["phase_mix_entropy"]) for row in timeline]),
        phase_mix_variation=phase_variation,
        critical_path_length=critical_path,
        total_work=total_work,
        total_work_to_critical_path_ratio=parallel_slack,
        parallel_slack=parallel_slack,
        critical_path_serial_fraction=serial_fraction,
        fixed_worker_idle_fraction=_fixed_worker_idle_fraction(dag, cfg.fixed_worker_count),
        fixed_prefill_decode_partition_imbalance=_partition_imbalance(dag, cfg),
        frontier_morphing_opportunity=opportunity["frontier_morphing_opportunity"],
        phase_morphing_opportunity=opportunity["phase_morphing_opportunity"],
        combined_opportunity=opportunity["combined_opportunity"],
        opportunity_taxonomy=opportunity["opportunity_taxonomy"],
        decision=opportunity["opportunity_taxonomy"],
        decision_reason=opportunity["decision_reason"],
    )
    return {
        "summary": asdict(summary),
        "timeline": [
            {key: value for key, value in row.items() if key != "phase_mix_vector"}
            for row in timeline
        ],
        "operator_rows": dag.to_operator_rows(),
    }


def _frontier_timeline(dag: PhaseDAG) -> list[dict[str, Any]]:
    groups: dict[float, list[PhaseOperator]] = {}
    for op in dag.operators.values():
        groups.setdefault(round(op.earliest_ready_time, 9), []).append(op)
    rows: list[dict[str, Any]] = []
    for time in sorted(groups):
        ops = groups[time]
        prefill = sum(op.prefill_cost for op in ops)
        decode = sum(op.decode_cost for op in ops)
        local = sum(op.local_tool_cost for op in ops)
        total = prefill + decode + local
        vector = _phase_vector(prefill, decode, local)
        rows.append(
            {
                "time": time,
                "frontier_width": len(ops),
                "operator_ids": [op.op_id for op in sorted(ops, key=lambda item: item.op_id)],
                "prefill_demand": prefill,
                "decode_demand": decode,
                "local_tool_demand": local,
                "total_demand": total,
                "phase_mix_entropy": _entropy(vector),
                "phase_mix_vector": vector,
            }
        )
    return rows


def _fixed_worker_idle_fraction(dag: PhaseDAG, worker_count: int) -> float:
    if worker_count <= 0 or not dag.operators:
        return 0.0
    worker_available = [0.0 for _ in range(worker_count)]
    finish_times: dict[str, float] = {}
    for op_id in dag.topological_order():
        op = dag.operators[op_id]
        deps_done = max((finish_times[dep] for dep in op.dependencies), default=0.0)
        worker_idx = min(range(worker_count), key=lambda idx: worker_available[idx])
        start = max(deps_done, worker_available[worker_idx])
        finish = start + op.total_cost
        worker_available[worker_idx] = finish
        finish_times[op_id] = finish
    makespan = max(worker_available, default=0.0)
    capacity = makespan * worker_count
    if capacity <= 0:
        return 0.0
    return max(0.0, 1.0 - dag.total_work() / capacity)


def _partition_imbalance(dag: PhaseDAG, config: FlowMorphConfig) -> float:
    prefill_workers = max(1, config.prefill_partition_workers)
    decode_workers = max(1, config.decode_partition_workers)
    prefill_work = sum(op.prefill_cost for op in dag.operators.values())
    decode_work = sum(op.decode_cost for op in dag.operators.values())
    local_work = sum(op.local_tool_cost for op in dag.operators.values())
    phase_time = max(prefill_work / prefill_workers, decode_work / decode_workers)
    total_workers = prefill_workers + decode_workers
    ideal = (prefill_work + decode_work + local_work) / max(1, total_workers)
    if ideal <= 0:
        return 0.0
    return max(0.0, phase_time / ideal - 1.0)


def _opportunity_taxonomy(
    frontier_cv: float,
    phase_variation: float,
    parallel_slack: float,
    config: FlowMorphConfig,
) -> dict[str, str]:
    frontier_varies = (
        frontier_cv >= config.frontier_cv_threshold
        or parallel_slack >= config.parallel_slack_threshold
    )
    phase_varies = phase_variation >= config.phase_variation_threshold
    if frontier_varies and phase_varies:
        taxonomy = "frontier_and_phase"
    elif frontier_varies:
        taxonomy = "frontier_only"
    elif phase_varies:
        taxonomy = "phase_only"
    else:
        taxonomy = "weak"
    frontier_reason = (
        f"frontier_width_cv={frontier_cv:.2f} "
        f"(threshold {config.frontier_cv_threshold:.2f}) or parallel_slack={parallel_slack:.2f} "
        f"(threshold {config.parallel_slack_threshold:.2f})"
    )
    phase_reason = (
        f"phase_mix_variation={phase_variation:.2f} "
        f"(threshold {config.phase_variation_threshold:.2f})"
    )
    return {
        "frontier_morphing_opportunity": "yes" if frontier_varies else "no",
        "phase_morphing_opportunity": "yes" if phase_varies else "no",
        "combined_opportunity": "yes" if frontier_varies and phase_varies else "no",
        "opportunity_taxonomy": taxonomy,
        "decision_reason": f"{taxonomy}: {frontier_reason}; {phase_reason}",
    }


def _wide_stage_work_fraction(
    timeline: list[dict[str, Any]],
    median_width: float,
    total_work: float,
) -> float:
    if total_work <= 0:
        return 0.0
    wide_work = sum(
        float(row["total_demand"])
        for row in timeline
        if float(row["frontier_width"]) > median_width
    )
    return wide_work / total_work


def _narrow_critical_stage_fraction(
    dag: PhaseDAG,
    median_width: float,
    critical_path: float,
) -> float:
    if critical_path <= 0:
        return 0.0
    critical_ops = _critical_path_operator_ids(dag)
    widths_by_time: dict[float, int] = {}
    for op in dag.operators.values():
        ready_time = round(op.earliest_ready_time, 9)
        widths_by_time[ready_time] = widths_by_time.get(ready_time, 0) + 1
    narrow_work = 0.0
    for op_id in critical_ops:
        op = dag.operators[op_id]
        ready_time = round(op.earliest_ready_time, 9)
        if widths_by_time.get(ready_time, 0) <= median_width:
            narrow_work += op.total_cost
    return narrow_work / critical_path


def _critical_path_operator_ids(dag: PhaseDAG) -> set[str]:
    best_finish: dict[str, float] = {}
    predecessor: dict[str, str | None] = {}
    for op_id in dag.topological_order():
        op = dag.operators[op_id]
        best_dep = max(op.dependencies, key=lambda dep: best_finish[dep], default=None)
        best_start = best_finish[best_dep] if best_dep is not None else 0.0
        best_finish[op_id] = best_start + op.total_cost
        predecessor[op_id] = best_dep
    if not best_finish:
        return set()
    current: str | None = max(best_finish, key=best_finish.get)
    path: set[str] = set()
    while current is not None:
        path.add(current)
        current = predecessor[current]
    return path


def _phase_vector(prefill: float, decode: float, local: float) -> tuple[float, float, float]:
    total = prefill + decode + local
    if total <= 0:
        return (0.0, 0.0, 0.0)
    return (prefill / total, decode / total, local / total)


def _entropy(values: tuple[float, ...]) -> float:
    positive = [value for value in values if value > 0]
    if not positive:
        return 0.0
    return -sum(value * math.log(value, 2) for value in positive)


def _phase_mix_variation(vectors: list[tuple[float, float, float]]) -> float:
    if len(vectors) <= 1:
        return 0.0
    mean_vector = tuple(sum(vector[i] for vector in vectors) / len(vectors) for i in range(3))
    distances = [
        math.sqrt(sum((vector[i] - mean_vector[i]) ** 2 for i in range(3)))
        for vector in vectors
    ]
    return _mean(distances)


def _coefficient_of_variation(values: list[int]) -> float:
    mean = _mean(values)
    if mean <= 0:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance) / mean


def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _mean(values: list[float] | list[int]) -> float:
    return sum(values) / len(values) if values else 0.0
