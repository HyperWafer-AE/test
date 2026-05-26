"""Compare static and dynamic policies as runtime-hot states appear."""

from __future__ import annotations

import argparse
from pathlib import Path

from waferstateflow.hotness import initialize_static_hotness
from waferstateflow.redundancy_analyzer import analyze_redundancy
from waferstateflow.reporting import write_csv, write_json
from waferstateflow.simulator import SimulationConfig, simulate_workflow
from waferstateflow.wafer_topology import WaferTopology
from waferstateflow.workflow_generators import WORKFLOW_NAMES, generate_workflow


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", choices=WORKFLOW_NAMES, default="trading")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--num-branches", type=int, default=4)
    parser.add_argument("--num-rounds", type=int, default=2)
    parser.add_argument("--shared-state-size", type=int, default=4000)
    parser.add_argument("--unique-state-size", type=int, default=400)
    parser.add_argument("--output-token-mean", type=int, default=250)
    parser.add_argument("--mesh", default="32x32")
    parser.add_argument("--worker-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    probabilities = [0.0, 0.25, 0.5, 0.75]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    topology = WaferTopology.from_mesh(args.mesh)
    sim_config = SimulationConfig(worker_count=args.worker_count)
    summary_rows = []
    wave_rows = []
    policy_rows = []
    access_rows = []
    last_graph = None
    last_analysis = None

    for probability in probabilities:
        graph = generate_workflow(
            args.workflow,
            batch_size=args.batch_size,
            num_agents=args.num_agents,
            num_branches=args.num_branches,
            num_rounds=args.num_rounds,
            shared_state_size=args.shared_state_size,
            unique_state_size=args.unique_state_size,
            dynamic_hot_state_probability=probability,
            output_token_mean=args.output_token_mean,
            seed=args.seed,
        )
        initialize_static_hotness(graph)
        analysis = analyze_redundancy(graph)
        last_graph = graph
        last_analysis = analysis
        for state_policy in ("static", "dynamic"):
            run = simulate_workflow(
                graph,
                "WaferStateFlow",
                topology=topology,
                config=sim_config,
                state_policy=state_policy,
            )
            row = run.result.to_row()
            row["dynamic_hot_state_probability"] = probability
            row["state_policy"] = state_policy
            row["dynamic_hot_state_count"] = analysis["summary"]["dynamic_hot_state_count"]
            summary_rows.append(row)
            for wave in run.wave_schedule:
                wave = dict(wave)
                wave["dynamic_hot_state_probability"] = probability
                wave["state_policy"] = state_policy
                wave_rows.append(wave)
            for event in run.state_access_events:
                event = dict(event)
                event["dynamic_hot_state_probability"] = probability
                event["state_policy"] = state_policy
                access_rows.append(event)
            for decision in run.policy_decisions:
                prow = decision.to_row()
                prow["dynamic_hot_state_probability"] = probability
                prow["state_policy"] = state_policy
                policy_rows.append(prow)

    assert last_graph is not None and last_analysis is not None
    last_graph.export_csv(out)
    last_graph.export_json(out / "state_access_graph.json")
    write_json(out / "metadata.json", {"experiment": "dynamic_hotness_sweep", "workflow": args.workflow})
    write_json(out / "config.json", vars(args) | {"probabilities": probabilities})
    write_csv(out / "state_hotness.csv", last_analysis["state_hotness"])
    write_csv(out / "policy_decisions.csv", policy_rows)
    write_csv(out / "wave_schedule.csv", wave_rows)
    write_csv(out / "state_access_events.csv", access_rows)
    write_csv(out / "simulation_summary.csv", summary_rows)
    _write_report(out / "report.md", summary_rows)
    _write_figures(out / "figures", summary_rows)


def _write_report(path: Path, rows: list[dict[str, object]]) -> None:
    by_probability: dict[float, dict[str, dict[str, object]]] = {}
    for row in rows:
        p = float(row["dynamic_hot_state_probability"])
        policy = str(row["state_policy"])
        by_probability.setdefault(p, {})[policy] = row
    lines = [
        "# WaferStateFlow Dynamic Hotness Sweep",
        "",
        "## Executive Summary",
        "",
        "This sweep compares a static policy that cannot see runtime-hot candidate states with a dynamic policy that can promote them after observation.",
        "",
        "| p(dynamic) | static latency | dynamic latency | materialization delta | H3 signal |",
        "| ---: | ---: | ---: | ---: | --- |",
    ]
    any_signal = False
    for p in sorted(by_probability):
        static = by_probability[p]["static"]
        dynamic = by_probability[p]["dynamic"]
        latency_delta = float(static["workflow_latency"]) - float(dynamic["workflow_latency"])
        materialization_delta = int(static["state_materialization_bytes"]) - int(dynamic["state_materialization_bytes"])
        signal = "yes" if latency_delta > 1e-6 or materialization_delta > 0 else "no"
        any_signal = any_signal or signal == "yes"
        lines.append(
            f"| {p:.2f} | {float(static['workflow_latency']):.3f} | "
            f"{float(dynamic['workflow_latency']):.3f} | {materialization_delta} | {signal} |"
        )
    lines.extend(
        [
            "",
            "## Failure Cases",
            "",
            "When `p(dynamic)=0`, dynamic policy should not help. If it does, the simulator is over-crediting dynamic adaptation.",
            "",
            "## What This Means for the Paper",
            "",
            f"H3 is {'partially supported' if any_signal else 'not supported'} in this synthetic sweep. "
            "A real trace is still needed before claiming dynamic hotness as a core contribution.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_figures(out_dir: Path, rows: list[dict[str, object]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    probabilities = sorted({float(row["dynamic_hot_state_probability"]) for row in rows})
    for metric, ylabel, filename in [
        ("workflow_latency", "Workflow latency", "dynamic_latency.png"),
        ("state_materialization_bytes", "Materialization bytes", "dynamic_materialization.png"),
    ]:
        plt.figure(figsize=(7, 4))
        for policy in ("static", "dynamic"):
            values = [
                float(
                    next(
                        row[metric]
                        for row in rows
                        if float(row["dynamic_hot_state_probability"]) == p and row["state_policy"] == policy
                    )
                )
                for p in probabilities
            ]
            plt.plot(probabilities, values, marker="o", label=policy)
        plt.xlabel("Dynamic hot-state probability")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / filename)
        plt.close()


if __name__ == "__main__":
    main()
