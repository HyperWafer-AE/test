"""Audit FlowMorph-v1 scheduler robustness before wafer mapping."""

from __future__ import annotations

import argparse
from itertools import product

from flowmorph.analyzer import FlowMorphConfig, characterize_phase_irregularity
from flowmorph.converters import PhaseCostModel, phase_dag_from_workflow_name
from flowmorph.reporting import write_flowmorph_sensitivity_report
from flowmorph.schedulers import (
    ALWAYS_CONSOLIDATED,
    ALWAYS_PARALLEL,
    FIXED_WORKER_POOL,
    FRONTIER_AWARE_MORPHING,
    STATIC_SPLIT_RESOURCE,
    FrontierSchedulerConfig,
    run_scheduler,
)
from waferstateflow.workflow_generators import WORKFLOW_NAMES


STATIC_ORACLE_SCHEDULERS = [
    FIXED_WORKER_POOL,
    ALWAYS_PARALLEL,
    ALWAYS_CONSOLIDATED,
    STATIC_SPLIT_RESOURCE,
]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflows", default="all", help="Comma-separated workflows, or all.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--num-branches", type=int, default=4)
    parser.add_argument("--num-rounds", type=int, default=2)
    parser.add_argument("--shared-state-size", type=int, default=4000)
    parser.add_argument("--unique-state-size", type=int, default=400)
    parser.add_argument("--dynamic-hot-state-probability", type=float, default=0.25)
    parser.add_argument("--output-token-mean", type=int, default=250)
    parser.add_argument("--prefill-cost-per-token", type=float, default=0.002)
    parser.add_argument("--decode-cost-per-token", type=float, default=0.006)
    parser.add_argument("--local-cost-per-token", type=float, default=0.0005)
    parser.add_argument("--base-local-tool-cost", type=float, default=0.1)
    parser.add_argument("--consolidated-speedup-exponents", default="0.2,0.35,0.5,0.65,0.8")
    parser.add_argument("--mode-switch-overheads", default="0,0.5,1,2,5")
    parser.add_argument("--worker-counts", default="4,8,16")
    parser.add_argument("--criticality-thresholds", default="0.8,1.5,2.0")
    parser.add_argument("--split-consolidated-workers", type=int, default=4)
    parser.add_argument("--frontier-cv-threshold", type=float, default=0.35)
    parser.add_argument("--phase-variation-threshold", type=float, default=0.25)
    parser.add_argument("--parallel-slack-threshold", type=float, default=2.0)
    parser.add_argument("--regret-tolerance", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/flowmorph_scheduler_sensitivity")
    args = parser.parse_args(argv)

    workflows = _parse_workflows(args.workflows)
    exponents = _parse_float_list(args.consolidated_speedup_exponents)
    overheads = _parse_float_list(args.mode_switch_overheads)
    worker_counts = _parse_int_list(args.worker_counts)
    criticality_thresholds = _parse_float_list(args.criticality_thresholds)
    cost_model = PhaseCostModel(
        prefill_cost_per_token=args.prefill_cost_per_token,
        decode_cost_per_token=args.decode_cost_per_token,
        local_cost_per_token=args.local_cost_per_token,
        base_local_tool_cost=args.base_local_tool_cost,
    )

    dags = {}
    characterizations = {}
    statuses = {}
    for workflow in workflows:
        dag = phase_dag_from_workflow_name(
            workflow,
            cost_model,
            batch_size=args.batch_size,
            num_agents=args.num_agents,
            num_branches=args.num_branches,
            num_rounds=args.num_rounds,
            shared_state_size=args.shared_state_size,
            unique_state_size=args.unique_state_size,
            dynamic_hot_state_probability=args.dynamic_hot_state_probability,
            output_token_mean=args.output_token_mean,
            seed=args.seed,
        )
        characterization = characterize_phase_irregularity(
            dag,
            FlowMorphConfig(
                fixed_worker_count=max(worker_counts),
                frontier_cv_threshold=args.frontier_cv_threshold,
                phase_variation_threshold=args.phase_variation_threshold,
                parallel_slack_threshold=args.parallel_slack_threshold,
            ),
        )
        taxonomy = characterization["summary"]["opportunity_taxonomy"]
        frontier_positive = taxonomy in {"frontier_only", "frontier_and_phase"}
        negative_control = workflow == "iterative"
        if frontier_positive or negative_control:
            dags[workflow] = dag
            characterizations[workflow] = characterization
            statuses[workflow] = "frontier_positive" if frontier_positive else "weak"

    sensitivity_rows: list[dict[str, object]] = []
    for exponent, overhead, workers, criticality_threshold in product(
        exponents,
        overheads,
        worker_counts,
        criticality_thresholds,
    ):
        scheduler_config = FrontierSchedulerConfig(
            worker_count=workers,
            consolidated_speedup_exponent=exponent,
            mode_switch_overhead=overhead,
            split_consolidated_workers=min(args.split_consolidated_workers, workers),
            criticality_threshold=criticality_threshold,
            frontier_cv_threshold=args.frontier_cv_threshold,
            phase_variation_threshold=args.phase_variation_threshold,
            parallel_slack_threshold=args.parallel_slack_threshold,
        )
        for workflow, dag in dags.items():
            static_results = {
                scheduler: run_scheduler(
                    dag,
                    scheduler,
                    scheduler_config,
                    characterization=characterizations[workflow],
                )["summary"]
                for scheduler in STATIC_ORACLE_SCHEDULERS
            }
            flowmorph = run_scheduler(
                dag,
                FRONTIER_AWARE_MORPHING,
                scheduler_config,
                characterization=characterizations[workflow],
            )["summary"]
            best_static_scheduler, best_static = min(
                static_results.items(),
                key=lambda item: float(item[1]["workflow_latency"]),
            )
            flow_latency = float(flowmorph["workflow_latency"])
            static_latency = float(best_static["workflow_latency"])
            regret = flow_latency / static_latency - 1.0 if static_latency > 0 else 0.0
            winner = FRONTIER_AWARE_MORPHING if flow_latency <= static_latency else best_static_scheduler
            sensitivity_rows.append(
                {
                    "workflow": workflow,
                    "frontier_status": statuses[workflow],
                    "opportunity_taxonomy": characterizations[workflow]["summary"]["opportunity_taxonomy"],
                    "consolidated_speedup_exponent": exponent,
                    "mode_switch_overhead": overhead,
                    "worker_count": workers,
                    "criticality_threshold": criticality_threshold,
                    "flowmorph_latency": flow_latency,
                    "best_static_latency": static_latency,
                    "best_static_oracle_latency": static_latency,
                    "regret": regret,
                    "winner": winner,
                    "best_static_scheduler": best_static_scheduler,
                    "best_static_oracle_scheduler": best_static_scheduler,
                    "mode_switch_count": int(flowmorph["mode_switch_count"]),
                    "fixed_worker_pool_latency": float(static_results[FIXED_WORKER_POOL]["workflow_latency"]),
                    "always_parallel_latency": float(static_results[ALWAYS_PARALLEL]["workflow_latency"]),
                    "always_consolidated_latency": float(static_results[ALWAYS_CONSOLIDATED]["workflow_latency"]),
                    "static_split_resource_latency": float(static_results[STATIC_SPLIT_RESOURCE]["workflow_latency"]),
                }
            )

    winner_rows = _winner_counts(sensitivity_rows)
    regret_rows = _regret_by_workflow(sensitivity_rows, args.regret_tolerance)
    write_flowmorph_sensitivity_report(
        args.out,
        sensitivity_rows,
        winner_rows,
        regret_rows,
        vars(args)
        | {
            "resolved_workflows": workflows,
            "selected_workflows": sorted(dags),
            "static_oracle_schedulers": STATIC_ORACLE_SCHEDULERS,
            "consolidated_speedup_exponents": exponents,
            "mode_switch_overheads": overheads,
            "worker_counts": worker_counts,
            "criticality_thresholds": criticality_thresholds,
        },
    )


def _winner_counts(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (str(row["frontier_status"]), str(row["winner"]))
        counts[key] = counts.get(key, 0) + 1
    return [
        {"frontier_status": status, "winner": winner, "count": count}
        for (status, winner), count in sorted(counts.items())
    ]


def _regret_by_workflow(rows: list[dict[str, object]], tolerance: float) -> list[dict[str, object]]:
    by_workflow: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_workflow.setdefault(str(row["workflow"]), []).append(row)
    summaries = []
    for workflow, workflow_rows in sorted(by_workflow.items()):
        regrets = sorted(float(row["regret"]) for row in workflow_rows)
        nonzero = [row for row in workflow_rows if float(row["mode_switch_overhead"]) > 0]
        robust_nonzero = [
            row for row in nonzero
            if float(row["regret"]) <= tolerance
        ]
        flow_wins = sum(1 for row in workflow_rows if row["winner"] == FRONTIER_AWARE_MORPHING)
        summaries.append(
            {
                "workflow": workflow,
                "frontier_status": str(workflow_rows[0]["frontier_status"]),
                "case_count": len(workflow_rows),
                "mean_regret": sum(regrets) / len(regrets) if regrets else 0.0,
                "p95_regret": _percentile(regrets, 0.95),
                "max_regret": max(regrets, default=0.0),
                "nonzero_overhead_cases": len(nonzero),
                "robust_nonzero_overhead_cases": len(robust_nonzero),
                "robust_under_nonzero_overhead": (
                    "yes" if nonzero and len(robust_nonzero) > len(nonzero) / 2 else "no"
                ),
                "flowmorph_wins": flow_wins,
                "best_static_wins": len(workflow_rows) - flow_wins,
            }
        )
    return summaries


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    fraction = index - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _parse_workflows(spec: str) -> list[str]:
    if spec.strip().lower() == "all":
        return list(WORKFLOW_NAMES)
    workflows = _parse_str_list(spec)
    unknown = [workflow for workflow in workflows if workflow not in WORKFLOW_NAMES]
    if unknown:
        raise ValueError(f"unknown workflows {unknown}; expected names from {WORKFLOW_NAMES}")
    return workflows


def _parse_float_list(spec: str) -> list[float]:
    return [float(item) for item in _parse_str_list(spec)]


def _parse_int_list(spec: str) -> list[int]:
    return [int(item) for item in _parse_str_list(spec)]


def _parse_str_list(spec: str) -> list[str]:
    return [item.strip() for item in spec.split(",") if item.strip()]


if __name__ == "__main__":
    main()
