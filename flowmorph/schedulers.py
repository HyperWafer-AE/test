"""Frontier-aware FlowMorph-v1 scheduler prototypes.

The schedulers in this module use an abstract worker-resource model. They do
not model wafer placement, NoC movement, or hardware-specific performance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

from .analyzer import FlowMorphConfig, characterize_phase_irregularity
from .ir import PhaseDAG, PhaseOperator


FIXED_WORKER_POOL = "fixed_worker_pool"
STATIC_FULL_RESOURCE = "static_full_resource"
STATIC_SPLIT_RESOURCE = "static_split_resource"
ALWAYS_PARALLEL = "always_parallel"
ALWAYS_CONSOLIDATED = "always_consolidated"
FRONTIER_AWARE_MORPHING = "frontier_aware_morphing"

DEFAULT_SCHEDULERS = [
    FIXED_WORKER_POOL,
    STATIC_FULL_RESOURCE,
    ALWAYS_PARALLEL,
    ALWAYS_CONSOLIDATED,
    STATIC_SPLIT_RESOURCE,
    FRONTIER_AWARE_MORPHING,
]


@dataclass(frozen=True)
class FrontierSchedulerConfig:
    worker_count: int = 8
    frontier_cv_threshold: float = 0.35
    phase_variation_threshold: float = 0.25
    parallel_slack_threshold: float = 2.0
    consolidated_speedup_exponent: float = 0.55
    split_consolidated_workers: int = 4
    criticality_threshold: float = 0.8


@dataclass(frozen=True)
class ScheduleMetrics:
    workflow: str
    scheduler: str
    policy: str
    opportunity_taxonomy: str
    workflow_latency: float
    worker_idle_fraction: float
    critical_path_delay: float
    mode_switch_count: int
    wide_stage_utilization: float
    narrow_stage_latency: float
    scheduled_operator_count: int
    wide_threshold: float
    median_frontier_width: float


@dataclass(frozen=True)
class RunningOp:
    op_id: str
    finish_time: float
    resource_units: int
    mode: str


def run_scheduler(
    dag: PhaseDAG,
    scheduler: str,
    config: FrontierSchedulerConfig | None = None,
    characterization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config or FrontierSchedulerConfig()
    if cfg.worker_count <= 0:
        raise ValueError("worker_count must be positive")
    analyzer_config = FlowMorphConfig(
        fixed_worker_count=cfg.worker_count,
        frontier_cv_threshold=cfg.frontier_cv_threshold,
        phase_variation_threshold=cfg.phase_variation_threshold,
        parallel_slack_threshold=cfg.parallel_slack_threshold,
    )
    characterization = characterization or characterize_phase_irregularity(dag, analyzer_config)
    summary = characterization["summary"]
    canonical = _canonical_scheduler_name(scheduler)
    critical_path = _critical_path_operator_ids(dag)
    median_width = float(summary["median_frontier_width"])
    wide_threshold = max(2.0, median_width + 1e-9)
    if canonical == FRONTIER_AWARE_MORPHING and summary["frontier_morphing_opportunity"] != "yes":
        policy = "fallback_fixed_worker_pool"
        rows, mode_switches = _simulate(
            dag,
            scheduler_name=canonical,
            policy=policy,
            config=cfg,
            summary=summary,
            critical_path=critical_path,
            select_actions=_fixed_worker_actions,
        )
    else:
        policy = canonical
        rows, mode_switches = _simulate(
            dag,
            scheduler_name=canonical,
            policy=policy,
            config=cfg,
            summary=summary,
            critical_path=critical_path,
            select_actions=_selector_for(canonical),
        )
    metrics = _metrics_from_rows(
        dag,
        rows,
        scheduler_name=canonical,
        policy=policy,
        summary=summary,
        config=cfg,
        critical_path=critical_path,
        mode_switch_count=mode_switches,
        wide_threshold=wide_threshold,
        median_width=median_width,
    )
    return {
        "summary": asdict(metrics),
        "schedule_rows": rows,
    }


def _simulate(
    dag: PhaseDAG,
    scheduler_name: str,
    policy: str,
    config: FrontierSchedulerConfig,
    summary: dict[str, Any],
    critical_path: set[str],
    select_actions: Callable[..., list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], int]:
    dag.compute_earliest_ready_times()
    unscheduled = set(dag.operators)
    running: list[RunningOp] = []
    finish_times: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    time = 0.0
    last_mode: str | None = None
    mode_switch_count = 0
    wave_id = 0
    while unscheduled or running:
        ready = _ready_ops(dag, unscheduled, finish_times)
        available = config.worker_count - sum(op.resource_units for op in running)
        actions = []
        if ready and available > 0:
            actions = select_actions(
                dag=dag,
                ready=ready,
                running=running,
                available=available,
                config=config,
                summary=summary,
                critical_path=critical_path,
            )
        if actions:
            wave_mode = actions[0]["mode"]
            if wave_mode != last_mode:
                if last_mode is not None:
                    mode_switch_count += 1
                last_mode = wave_mode
            for action in actions:
                op = dag.operators[action["op_id"]]
                dep_ready = max((finish_times[dep] for dep in op.dependencies), default=0.0)
                start = time
                duration = action["duration"]
                finish = start + duration
                row = {
                    "workflow": dag.graph_id,
                    "scheduler": scheduler_name,
                    "policy": policy,
                    "wave_id": wave_id,
                    "op_id": op.op_id,
                    "mode": action["mode"],
                    "start_time": start,
                    "finish_time": finish,
                    "duration": duration,
                    "resource_units": action["resource_units"],
                    "frontier_width": len(ready),
                    "dependency_ready_time": dep_ready,
                    "resource_wait": max(0.0, start - dep_ready),
                    "critical_path_operator": op.op_id in critical_path,
                    "criticality": op.criticality,
                    "prefill_cost": op.prefill_cost,
                    "decode_cost": op.decode_cost,
                    "local_tool_cost": op.local_tool_cost,
                }
                rows.append(row)
                running.append(
                    RunningOp(
                        op_id=op.op_id,
                        finish_time=finish,
                        resource_units=action["resource_units"],
                        mode=action["mode"],
                    )
                )
                unscheduled.remove(op.op_id)
            wave_id += 1
            continue
        if not running:
            raise RuntimeError(f"scheduler {scheduler_name} reached a dead end")
        next_time = min(op.finish_time for op in running)
        time = next_time
        done = [op for op in running if op.finish_time <= time + 1e-12]
        running = [op for op in running if op.finish_time > time + 1e-12]
        for op in done:
            finish_times[op.op_id] = op.finish_time
    return rows, mode_switch_count


def _fixed_worker_actions(**kwargs: Any) -> list[dict[str, Any]]:
    ready: list[PhaseOperator] = kwargs["ready"]
    available: int = kwargs["available"]
    actions = []
    for op in _priority_ready_ops(ready, kwargs["critical_path"])[:available]:
        actions.append(
            {
                "op_id": op.op_id,
                "mode": "fixed",
                "duration": op.total_cost,
                "resource_units": 1,
            }
        )
    return actions


def _always_parallel_actions(**kwargs: Any) -> list[dict[str, Any]]:
    ready: list[PhaseOperator] = kwargs["ready"]
    available: int = kwargs["available"]
    return [
        {
            "op_id": op.op_id,
            "mode": "parallel",
            "duration": op.total_cost,
            "resource_units": 1,
        }
        for op in _priority_ready_ops(ready, kwargs["critical_path"])[:available]
    ]


def _always_consolidated_actions(**kwargs: Any) -> list[dict[str, Any]]:
    running: list[RunningOp] = kwargs["running"]
    if running:
        return []
    config: FrontierSchedulerConfig = kwargs["config"]
    ready = _priority_ready_ops(kwargs["ready"], kwargs["critical_path"])
    op = ready[0]
    return [
        {
            "op_id": op.op_id,
            "mode": "consolidated",
            "duration": _scaled_duration(op, config.worker_count, config),
            "resource_units": config.worker_count,
        }
    ]


def _static_split_actions(**kwargs: Any) -> list[dict[str, Any]]:
    dag: PhaseDAG = kwargs["dag"]
    config: FrontierSchedulerConfig = kwargs["config"]
    ready = _priority_ready_ops(kwargs["ready"], kwargs["critical_path"])
    running: list[RunningOp] = kwargs["running"]
    critical_path: set[str] = kwargs["critical_path"]
    consolidated_units = min(config.worker_count, max(1, config.split_consolidated_workers))
    parallel_units = max(1, config.worker_count - consolidated_units)
    consolidated_busy = any(op.mode == "split_consolidated" for op in running)
    parallel_busy = sum(op.resource_units for op in running if op.mode == "split_parallel")
    actions: list[dict[str, Any]] = []
    already_chosen: set[str] = set()
    if not consolidated_busy:
        critical_ready = [op for op in ready if _is_critical(op, critical_path, config)]
        if critical_ready:
            op = critical_ready[0]
            actions.append(
                {
                    "op_id": op.op_id,
                    "mode": "split_consolidated",
                    "duration": _scaled_duration(op, consolidated_units, config),
                    "resource_units": consolidated_units,
                }
            )
            already_chosen.add(op.op_id)
    free_parallel = max(0, parallel_units - parallel_busy)
    for op in ready:
        if free_parallel <= 0:
            break
        if op.op_id in already_chosen:
            continue
        actions.append(
            {
                "op_id": op.op_id,
                "mode": "split_parallel",
                "duration": dag.operators[op.op_id].total_cost,
                "resource_units": 1,
            }
        )
        free_parallel -= 1
    return actions


def _frontier_morphing_actions(**kwargs: Any) -> list[dict[str, Any]]:
    ready: list[PhaseOperator] = kwargs["ready"]
    running: list[RunningOp] = kwargs["running"]
    available: int = kwargs["available"]
    config: FrontierSchedulerConfig = kwargs["config"]
    summary: dict[str, Any] = kwargs["summary"]
    critical_path: set[str] = kwargs["critical_path"]
    median_width = float(summary["median_frontier_width"])
    high_frontier = len(ready) > median_width and len(ready) >= 2
    if high_frontier:
        return [
            {
                "op_id": op.op_id,
                "mode": "parallel",
                "duration": op.total_cost,
                "resource_units": 1,
            }
            for op in _priority_ready_ops(ready, critical_path)[:available]
        ]
    critical_ready = [op for op in _priority_ready_ops(ready, critical_path) if _is_critical(op, critical_path, config)]
    if critical_ready and not running and available == config.worker_count:
        op = critical_ready[0]
        return [
            {
                "op_id": op.op_id,
                "mode": "consolidated_fast_lane",
                "duration": _scaled_duration(op, config.worker_count, config),
                "resource_units": config.worker_count,
            }
        ]
    return _fixed_worker_actions(**kwargs)


def _selector_for(scheduler: str) -> Callable[..., list[dict[str, Any]]]:
    if scheduler == FIXED_WORKER_POOL:
        return _fixed_worker_actions
    if scheduler in {ALWAYS_PARALLEL, STATIC_FULL_RESOURCE}:
        return _always_parallel_actions
    if scheduler == ALWAYS_CONSOLIDATED:
        return _always_consolidated_actions
    if scheduler == STATIC_SPLIT_RESOURCE:
        return _static_split_actions
    if scheduler == FRONTIER_AWARE_MORPHING:
        return _frontier_morphing_actions
    raise ValueError(f"unknown scheduler {scheduler}")


def _canonical_scheduler_name(name: str) -> str:
    normalized = name.strip()
    if normalized == STATIC_FULL_RESOURCE:
        return STATIC_FULL_RESOURCE
    if normalized in {
        FIXED_WORKER_POOL,
        ALWAYS_PARALLEL,
        ALWAYS_CONSOLIDATED,
        STATIC_SPLIT_RESOURCE,
        FRONTIER_AWARE_MORPHING,
    }:
        return normalized
    raise ValueError(f"unknown scheduler {name}")


def _ready_ops(
    dag: PhaseDAG,
    unscheduled: set[str],
    finish_times: dict[str, float],
) -> list[PhaseOperator]:
    return [
        dag.operators[op_id]
        for op_id in sorted(unscheduled)
        if all(dep in finish_times for dep in dag.operators[op_id].dependencies)
    ]


def _priority_ready_ops(ops: list[PhaseOperator], critical_path: set[str]) -> list[PhaseOperator]:
    return sorted(
        ops,
        key=lambda op: (
            op.op_id not in critical_path,
            -op.criticality,
            -op.total_cost,
            op.op_id,
        ),
    )


def _is_critical(
    op: PhaseOperator,
    critical_path: set[str],
    config: FrontierSchedulerConfig,
) -> bool:
    return op.op_id in critical_path or op.criticality >= config.criticality_threshold


def _scaled_duration(
    op: PhaseOperator,
    resource_units: int,
    config: FrontierSchedulerConfig,
) -> float:
    speedup = max(1.0, float(resource_units) ** config.consolidated_speedup_exponent)
    return op.total_cost / speedup


def _metrics_from_rows(
    dag: PhaseDAG,
    rows: list[dict[str, Any]],
    scheduler_name: str,
    policy: str,
    summary: dict[str, Any],
    config: FrontierSchedulerConfig,
    critical_path: set[str],
    mode_switch_count: int,
    wide_threshold: float,
    median_width: float,
) -> ScheduleMetrics:
    latency = max((float(row["finish_time"]) for row in rows), default=0.0)
    utilization = _interval_utilization(rows, config.worker_count)
    wide_utilization = _interval_utilization(
        [row for row in rows if float(row["frontier_width"]) >= wide_threshold],
        config.worker_count,
    )
    narrow_latency = _interval_union_length(
        [row for row in rows if float(row["frontier_width"]) <= median_width]
    )
    critical_delay = sum(
        float(row["resource_wait"])
        for row in rows
        if str(row["op_id"]) in critical_path
    )
    return ScheduleMetrics(
        workflow=dag.graph_id,
        scheduler=scheduler_name,
        policy=policy,
        opportunity_taxonomy=str(summary["opportunity_taxonomy"]),
        workflow_latency=latency,
        worker_idle_fraction=max(0.0, 1.0 - utilization),
        critical_path_delay=critical_delay,
        mode_switch_count=mode_switch_count,
        wide_stage_utilization=wide_utilization,
        narrow_stage_latency=narrow_latency,
        scheduled_operator_count=len(rows),
        wide_threshold=wide_threshold,
        median_frontier_width=median_width,
    )


def _interval_utilization(rows: list[dict[str, Any]], worker_count: int) -> float:
    if worker_count <= 0 or not rows:
        return 0.0
    intervals = _active_intervals(rows)
    if not intervals:
        return 0.0
    used = 0.0
    capacity = 0.0
    for start, finish, active_rows in intervals:
        duration = finish - start
        if duration <= 0:
            continue
        active_units = sum(int(row["resource_units"]) for row in active_rows)
        used += min(worker_count, active_units) * duration
        capacity += worker_count * duration
    return used / capacity if capacity > 0 else 0.0


def _interval_union_length(rows: list[dict[str, Any]]) -> float:
    return sum(finish - start for start, finish, _ in _active_intervals(rows))


def _active_intervals(rows: list[dict[str, Any]]) -> list[tuple[float, float, list[dict[str, Any]]]]:
    points = sorted(
        {
            float(row["start_time"])
            for row in rows
        }
        | {
            float(row["finish_time"])
            for row in rows
        }
    )
    intervals: list[tuple[float, float, list[dict[str, Any]]]] = []
    for start, finish in zip(points, points[1:]):
        if finish <= start:
            continue
        midpoint = (start + finish) / 2.0
        active = [
            row
            for row in rows
            if float(row["start_time"]) <= midpoint < float(row["finish_time"])
        ]
        if active:
            intervals.append((start, finish, active))
    return intervals


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
