"""Run residual redundancy analysis across synthetic workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from waferstateflow.reporting import write_csv, write_json
from waferstateflow.residual_redundancy_analyzer import (
    ResidualAnalysisConfig,
    analyze_residual_redundancy,
)
from waferstateflow.workflow_generators import WORKFLOW_NAMES, generate_workflow


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflows", default="all", help="Comma-separated workflow names, or all.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--num-branches", type=int, default=4)
    parser.add_argument("--num-rounds", type=int, default=2)
    parser.add_argument("--shared-state-size", type=int, default=4000)
    parser.add_argument("--unique-state-size", type=int, default=400)
    parser.add_argument("--dynamic-hot-state-probability", type=float, default=0.25)
    parser.add_argument("--output-token-mean", type=int, default=250)
    parser.add_argument("--kvflow-capacity-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--residual-ratio-threshold", type=float, default=1.5)
    parser.add_argument("--dynamic-hot-fraction-threshold", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/waferstateflow_residual_analysis")
    args = parser.parse_args(argv)

    workflows = _parse_workflows(args.workflows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {"resolved_workflows": workflows}
    analysis_config = ResidualAnalysisConfig(
        kvflow_capacity_bytes=args.kvflow_capacity_bytes,
        residual_ratio_threshold=args.residual_ratio_threshold,
        dynamic_hot_fraction_threshold=args.dynamic_hot_fraction_threshold,
    )

    summaries = []
    state_rows = []
    for workflow in workflows:
        graph = generate_workflow(
            workflow,
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
        result = analyze_residual_redundancy(graph, analysis_config)
        summaries.append(result["summary"])
        state_rows.extend(result["state_rows"])

    write_json(out / "metadata.json", {"experiment": "residual_redundancy_analysis", "workflows": workflows})
    write_json(out / "config.json", config)
    write_csv(out / "residual_summary.csv", summaries)
    write_csv(out / "residual_state_table.csv", state_rows)
    _write_report(out / "report.md", summaries, state_rows, analysis_config)
    _write_figures(out / "figures", summaries)


def _parse_workflows(spec: str) -> list[str]:
    if spec.strip().lower() == "all":
        return list(WORKFLOW_NAMES)
    workflows = [item.strip() for item in spec.split(",") if item.strip()]
    unknown = [workflow for workflow in workflows if workflow not in WORKFLOW_NAMES]
    if unknown:
        raise ValueError(f"unknown workflows {unknown}; expected names from {WORKFLOW_NAMES}")
    return workflows


def _write_report(
    path: Path,
    summaries: list[dict[str, object]],
    state_rows: list[dict[str, object]],
    config: ResidualAnalysisConfig,
) -> None:
    decisions = {str(row["decision"]) for row in summaries}
    if decisions == {"continue_investigation"}:
        overall = "continue investigation"
    elif "abandon_or_pivot" in decisions:
        overall = "abandon or pivot the wafer hot-state mapping direction for these synthetic settings"
    else:
        overall = "weak paper direction without stronger real-trace evidence"

    lines = [
        "# WaferStateFlow Residual Redundancy Analysis",
        "",
        "## Executive Summary",
        "",
        "This analysis subtracts redundancy covered by exact prefix caching, deterministic "
        "operator-output caching, and a KVFlow-like future-use cache before estimating "
        "the residual opportunity for wafer hot-state mapping.",
        "",
        f"Overall decision: **{overall}**.",
        "",
        "## Workflow Summary",
        "",
        "| workflow | raw ratio | residual ratio | dynamic residual fraction | residual fanout | wafer score | decision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in sorted(summaries, key=lambda item: str(item["workflow"])):
        lines.append(
            f"| `{row['workflow']}` | {float(row['raw_redundancy_ratio']):.2f} | "
            f"{float(row['residual_redundancy_ratio']):.2f} | "
            f"{float(row['dynamic_hot_residual_fraction']):.2f} | "
            f"{int(row['residual_token_weighted_fanout'])} | "
            f"{float(row['wafer_opportunity_score']):.3f} | {row['decision']} |"
        )

    lines.extend(
        [
            "",
            "## Redundancy Decomposition",
            "",
            "| workflow | raw duplicated | prefix covered | deterministic output covered | KVFlow covered | residual |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(summaries, key=lambda item: str(item["workflow"])):
        lines.append(
            f"| `{row['workflow']}` | {int(row['raw_duplicated_state_tokens'])} | "
            f"{int(row['exact_prefix_cache_covered_tokens'])} | "
            f"{int(row['deterministic_operator_output_cache_covered_tokens'])} | "
            f"{int(row['future_cache_covered_tokens'])} | "
            f"{int(row['residual_non_prefix_non_deterministic_dynamic_state_tokens'])} |"
        )

    residual_states = [
        row for row in state_rows if bool(row["residual_candidate"])
    ]
    lines.extend(
        [
            "",
            "## Top Residual States",
            "",
            "| workflow | state | kind | consumers | residual fanout | reason |",
            "| --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for row in sorted(
        residual_states,
        key=lambda item: int(item["residual_token_weighted_fanout"]),
        reverse=True,
    )[:20]:
        lines.append(
            f"| `{row['workflow']}` | `{row['state_id']}` | {row['kind']} | "
            f"{int(row['consumer_count'])} | {int(row['residual_token_weighted_fanout'])} | "
            f"{row['reason']} |"
        )
    if not residual_states:
        lines.append("| - | - | - | 0 | 0 | no residual candidates after baseline coverage |")

    lines.extend(
        [
            "",
            "## Decision Logic",
            "",
            f"- If `residual_redundancy_ratio < {config.residual_ratio_threshold:.2f}`, report abandon/pivot.",
            f"- If dynamic residual fraction is below {config.dynamic_hot_fraction_threshold:.2f}, report weak direction.",
            "- Negative results are retained; the generator was not tuned for this analysis.",
            "",
            "## What This Means",
            "",
        ]
    )
    weak = [
        row
        for row in summaries
        if row["decision"] in {"abandon_or_pivot", "weak_paper_direction"}
    ]
    if weak:
        lines.append(
            "Residual redundancy beyond existing cache/workflow baselines is weak for at least one "
            "workflow. Hot-state wafer mapping is therefore not a strong standalone paper direction "
            "under these synthetic settings."
        )
    else:
        lines.append(
            "All workflows pass the residual screening thresholds in this synthetic run. This would "
            "justify only continued investigation, not a paper-grade claim."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_figures(out_dir: Path, summaries: list[dict[str, object]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    rows = sorted(summaries, key=lambda row: str(row["workflow"]))
    if not rows:
        return
    x = range(len(rows))
    plt.figure(figsize=(8, 4))
    plt.bar([i - 0.2 for i in x], [float(row["raw_redundancy_ratio"]) for row in rows], width=0.4, label="raw")
    plt.bar(
        [i + 0.2 for i in x],
        [float(row["residual_redundancy_ratio"]) for row in rows],
        width=0.4,
        label="residual",
    )
    plt.xticks(list(x), [str(row["workflow"]) for row in rows], rotation=35, ha="right")
    plt.ylabel("Redundancy ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "residual_redundancy_ratio.png")
    plt.close()


if __name__ == "__main__":
    main()
