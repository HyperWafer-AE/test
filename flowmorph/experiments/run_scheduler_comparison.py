"""Run FlowMorph-v1 frontier-aware scheduler comparison."""

from __future__ import annotations

import argparse

from flowmorph.analyzer import FlowMorphConfig, characterize_phase_irregularity
from flowmorph.converters import PhaseCostModel, phase_dag_from_workflow_name
from flowmorph.reporting import write_flowmorph_scheduler_report
from flowmorph.schedulers import (
    DEFAULT_SCHEDULERS,
    FrontierSchedulerConfig,
    run_scheduler,
)
from waferstateflow.workflow_generators import WORKFLOW_NAMES


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflows", default="all", help="Comma-separated workflows, or all.")
    parser.add_argument("--schedulers", default=",".join(DEFAULT_SCHEDULERS))
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
    parser.add_argument("--worker-count", type=int, default=8)
    parser.add_argument("--split-consolidated-workers", type=int, default=4)
    parser.add_argument("--frontier-cv-threshold", type=float, default=0.35)
    parser.add_argument("--phase-variation-threshold", type=float, default=0.25)
    parser.add_argument("--parallel-slack-threshold", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/flowmorph_scheduler")
    args = parser.parse_args(argv)

    workflows = _parse_workflows(args.workflows)
    schedulers = _parse_csv(args.schedulers)
    cost_model = PhaseCostModel(
        prefill_cost_per_token=args.prefill_cost_per_token,
        decode_cost_per_token=args.decode_cost_per_token,
        local_cost_per_token=args.local_cost_per_token,
        base_local_tool_cost=args.base_local_tool_cost,
    )
    analyzer_config = FlowMorphConfig(
        fixed_worker_count=args.worker_count,
        frontier_cv_threshold=args.frontier_cv_threshold,
        phase_variation_threshold=args.phase_variation_threshold,
        parallel_slack_threshold=args.parallel_slack_threshold,
    )
    scheduler_config = FrontierSchedulerConfig(
        worker_count=args.worker_count,
        split_consolidated_workers=args.split_consolidated_workers,
        frontier_cv_threshold=args.frontier_cv_threshold,
        phase_variation_threshold=args.phase_variation_threshold,
        parallel_slack_threshold=args.parallel_slack_threshold,
    )
    dags = {}
    characterizations = {}
    selection_rows = []
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
        characterization = characterize_phase_irregularity(dag, analyzer_config)
        taxonomy = characterization["summary"]["opportunity_taxonomy"]
        frontier_positive = taxonomy in {"frontier_only", "frontier_and_phase"}
        negative_control = workflow == "iterative"
        selected = frontier_positive or negative_control
        selection_rows.append(
            {
                "workflow": workflow,
                "opportunity_taxonomy": taxonomy,
                "selected": "yes" if selected else "no",
                "selection_reason": (
                    "frontier_positive"
                    if frontier_positive
                    else "negative_control"
                    if negative_control
                    else "not_frontier_positive"
                ),
            }
        )
        if selected:
            dags[workflow] = dag
            characterizations[workflow] = characterization

    summary_rows = []
    schedule_rows = []
    for workflow, dag in dags.items():
        for scheduler in schedulers:
            result = run_scheduler(
                dag,
                scheduler,
                scheduler_config,
                characterization=characterizations[workflow],
            )
            summary_rows.append(result["summary"])
            schedule_rows.extend(result["schedule_rows"])

    write_flowmorph_scheduler_report(
        args.out,
        summary_rows,
        schedule_rows,
        selection_rows,
        vars(args) | {"resolved_workflows": workflows, "resolved_schedulers": schedulers},
    )


def _parse_workflows(spec: str) -> list[str]:
    if spec.strip().lower() == "all":
        return list(WORKFLOW_NAMES)
    workflows = _parse_csv(spec)
    unknown = [workflow for workflow in workflows if workflow not in WORKFLOW_NAMES]
    if unknown:
        raise ValueError(f"unknown workflows {unknown}; expected names from {WORKFLOW_NAMES}")
    return workflows


def _parse_csv(spec: str) -> list[str]:
    return [item.strip() for item in spec.split(",") if item.strip()]


if __name__ == "__main__":
    main()
