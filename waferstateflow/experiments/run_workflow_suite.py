"""Run characterization and scheduler comparison across multiple workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from waferstateflow.hotness import initialize_static_hotness
from waferstateflow.redundancy_analyzer import analyze_redundancy
from waferstateflow.reporting import (
    write_characterization_outputs,
    write_csv,
    write_json,
    write_scheduler_outputs,
)
from waferstateflow.simulator import BASELINES, SimulationConfig, run_all_baselines
from waferstateflow.state_policy import decide_policies
from waferstateflow.wafer_topology import WaferTopology
from waferstateflow.workflow_generators import WORKFLOW_NAMES, generate_workflow


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workflows",
        default=",".join(WORKFLOW_NAMES),
        help="Comma-separated workflow names, or 'all'.",
    )
    parser.add_argument("--mode", choices=["characterization", "scheduler", "both"], default="both")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--num-branches", type=int, default=4)
    parser.add_argument("--num-rounds", type=int, default=2)
    parser.add_argument("--shared-state-size", type=int, default=4000)
    parser.add_argument("--unique-state-size", type=int, default=400)
    parser.add_argument("--dynamic-hot-state-probability", type=float, default=0.25)
    parser.add_argument("--output-token-mean", type=int, default=250)
    parser.add_argument("--mesh", default="16x16")
    parser.add_argument("--state-policy", choices=["static", "dynamic"], default="dynamic")
    parser.add_argument("--worker-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    workflows = _parse_workflows(args.workflows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {"resolved_workflows": workflows}
    write_json(out / "metadata.json", {"experiment": "workflow_suite", "workflows": workflows})
    write_json(out / "config.json", config)

    characterization_rows = []
    scheduler_rows = []
    topology = WaferTopology.from_mesh(args.mesh)
    sim_config = SimulationConfig(worker_count=args.worker_count)

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
        initialize_static_hotness(graph)
        analysis = analyze_redundancy(graph)
        characterization_rows.append(dict(analysis["summary"]))

        if args.mode in {"characterization", "both"}:
            policies = decide_policies(list(graph.states.values()), memory_pressure=0.0)
            write_characterization_outputs(
                graph,
                analysis,
                policies,
                out / "characterization" / workflow,
                config | {"workflow": workflow},
            )

        if args.mode in {"scheduler", "both"}:
            runs = run_all_baselines(
                graph,
                topology=topology,
                config=sim_config,
                state_policy=args.state_policy,
                baselines=BASELINES,
            )
            for run in runs:
                row = run.result.to_row()
                row["workflow"] = workflow
                scheduler_rows.append(row)
            write_scheduler_outputs(
                graph,
                analysis,
                runs,
                out / "scheduler" / workflow,
                config | {"workflow": workflow},
            )

    write_csv(out / "characterization_summary.csv", characterization_rows)
    write_csv(out / "scheduler_summary.csv", scheduler_rows)
    _write_suite_report(out / "report.md", characterization_rows, scheduler_rows)
    _write_suite_figures(out / "figures", characterization_rows, scheduler_rows)


def _parse_workflows(spec: str) -> list[str]:
    if spec.strip().lower() == "all":
        return list(WORKFLOW_NAMES)
    workflows = [item.strip() for item in spec.split(",") if item.strip()]
    unknown = [workflow for workflow in workflows if workflow not in WORKFLOW_NAMES]
    if unknown:
        raise ValueError(f"unknown workflows {unknown}; expected names from {WORKFLOW_NAMES}")
    return workflows


def _write_suite_report(
    path: Path,
    characterization_rows: list[dict[str, object]],
    scheduler_rows: list[dict[str, object]],
) -> None:
    lines = [
        "# WaferStateFlow Workflow Suite",
        "",
        "## Executive Summary",
        "",
        f"Characterized {len(characterization_rows)} workflows and collected "
        f"{len(scheduler_rows)} scheduler rows.",
        "",
        "## Characterization",
        "",
        "| workflow | redundancy ratio | duplicate bytes | H1 | H2 |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for row in sorted(characterization_rows, key=lambda item: str(item["workflow"])):
        lines.append(
            f"| `{row['workflow']}` | {float(row['input_redundancy_ratio']):.2f} | "
            f"{int(row['duplicate_materialization_bytes'])} | {row['h1_supported']} | {row['h2_supported']} |"
        )

    if scheduler_rows:
        best_by_workflow: dict[str, dict[str, object]] = {}
        for row in scheduler_rows:
            workflow = str(row["workflow"])
            if workflow not in best_by_workflow or float(row["workflow_latency"]) < float(
                best_by_workflow[workflow]["workflow_latency"]
            ):
                best_by_workflow[workflow] = row
        lines.extend(
            [
                "",
                "## Scheduler Winners",
                "",
                "| workflow | winner | latency | WaferStateFlow latency |",
                "| --- | --- | ---: | ---: |",
            ]
        )
        for workflow in sorted(best_by_workflow):
            winner = best_by_workflow[workflow]
            proposed = next(
                row
                for row in scheduler_rows
                if row["workflow"] == workflow and row["baseline"] == "WaferStateFlow"
            )
            lines.append(
                f"| `{workflow}` | `{winner['baseline']}` | {float(winner['workflow_latency']):.3f} | "
                f"{float(proposed['workflow_latency']):.3f} |"
            )
        negative = [
            (workflow, row)
            for workflow, row in best_by_workflow.items()
            if row["baseline"] != "WaferStateFlow"
        ]
        lines.extend(["", "## Failure Cases", ""])
        if negative:
            workflow, row = sorted(negative)[0]
            lines.append(
                f"`{workflow}` has `{row['baseline']}` no worse than WaferStateFlow in this suite, "
                "so the prototype preserves counterexamples and ties instead of forcing WaferStateFlow "
                "to win every setting."
            )
        else:
            lines.append(
                "No non-WaferStateFlow winner appeared in this suite. Use low-fanout and cache-rich sweeps "
                "before making platform necessity claims."
            )

    lines.extend(
        [
            "",
            "## What This Means for the Paper",
            "",
            "The suite is useful for screening hypotheses across workflow shapes. It remains synthetic and should be followed by real trace replay.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_suite_figures(
    out_dir: Path,
    characterization_rows: list[dict[str, object]],
    scheduler_rows: list[dict[str, object]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    if characterization_rows:
        rows = sorted(characterization_rows, key=lambda row: str(row["workflow"]))
        plt.figure(figsize=(8, 4))
        plt.bar(
            [str(row["workflow"]) for row in rows],
            [float(row["input_redundancy_ratio"]) for row in rows],
        )
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("Input redundancy ratio")
        plt.tight_layout()
        plt.savefig(out_dir / "suite_redundancy.png")
        plt.close()

    if scheduler_rows:
        workflows = sorted({str(row["workflow"]) for row in scheduler_rows})
        proposed = [
            float(
                next(
                    row["workflow_latency"]
                    for row in scheduler_rows
                    if row["workflow"] == workflow and row["baseline"] == "WaferStateFlow"
                )
            )
            for workflow in workflows
        ]
        best = [
            min(
                float(row["workflow_latency"])
                for row in scheduler_rows
                if row["workflow"] == workflow
            )
            for workflow in workflows
        ]
        plt.figure(figsize=(8, 4))
        x = range(len(workflows))
        plt.bar([i - 0.2 for i in x], best, width=0.4, label="best")
        plt.bar([i + 0.2 for i in x], proposed, width=0.4, label="WaferStateFlow")
        plt.xticks(list(x), workflows, rotation=35, ha="right")
        plt.ylabel("Workflow latency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "suite_scheduler_latency.png")
        plt.close()


if __name__ == "__main__":
    main()
