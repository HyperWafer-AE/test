"""Run FlowMorph phase/resource irregularity characterization."""

from __future__ import annotations

import argparse

from flowmorph.analyzer import FlowMorphConfig, characterize_phase_irregularity
from flowmorph.converters import PhaseCostModel, phase_dag_from_workflow_name
from flowmorph.reporting import write_flowmorph_report
from waferstateflow.workflow_generators import WORKFLOW_NAMES


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
    parser.add_argument("--fixed-worker-count", type=int, default=8)
    parser.add_argument("--prefill-partition-workers", type=int, default=4)
    parser.add_argument("--decode-partition-workers", type=int, default=4)
    parser.add_argument("--frontier-cv-threshold", type=float, default=0.35)
    parser.add_argument("--phase-variation-threshold", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/flowmorph_characterization")
    args = parser.parse_args(argv)

    workflows = _parse_workflows(args.workflows)
    cost_model = PhaseCostModel(
        prefill_cost_per_token=args.prefill_cost_per_token,
        decode_cost_per_token=args.decode_cost_per_token,
        local_cost_per_token=args.local_cost_per_token,
        base_local_tool_cost=args.base_local_tool_cost,
    )
    analyzer_config = FlowMorphConfig(
        fixed_worker_count=args.fixed_worker_count,
        prefill_partition_workers=args.prefill_partition_workers,
        decode_partition_workers=args.decode_partition_workers,
        frontier_cv_threshold=args.frontier_cv_threshold,
        phase_variation_threshold=args.phase_variation_threshold,
    )
    summaries = []
    timeline_rows = []
    operator_rows = []
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
        result = characterize_phase_irregularity(dag, analyzer_config)
        summaries.append(result["summary"])
        for row in result["timeline"]:
            row = dict(row)
            row["workflow"] = workflow
            timeline_rows.append(row)
        for row in result["operator_rows"]:
            row = dict(row)
            row["workflow"] = workflow
            operator_rows.append(row)

    write_flowmorph_report(
        args.out,
        summaries,
        timeline_rows,
        operator_rows,
        vars(args) | {"resolved_workflows": workflows},
    )


def _parse_workflows(spec: str) -> list[str]:
    if spec.strip().lower() == "all":
        return list(WORKFLOW_NAMES)
    workflows = [item.strip() for item in spec.split(",") if item.strip()]
    unknown = [workflow for workflow in workflows if workflow not in WORKFLOW_NAMES]
    if unknown:
        raise ValueError(f"unknown workflows {unknown}; expected names from {WORKFLOW_NAMES}")
    return workflows


if __name__ == "__main__":
    main()
