"""Run a compact wafer sensitivity sweep."""

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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    meshes = ["8x8", "16x16", "32x32"]
    memory_caps = [64 * 1024 * 1024, 256 * 1024 * 1024]
    shared_sizes = [200, 1000, 4000, 12000]
    branch_widths = [1, 2, 8]
    baselines = ["request_parallel_gpu_like", "helium_like_operator_schedule", "wafer_request_centric", "WaferStateFlow"]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    wave_rows = []
    policy_rows = []
    last_graph = None
    last_analysis = None

    for mesh in meshes:
        for memory_cap in memory_caps:
            for shared_size in shared_sizes:
                for branches in branch_widths:
                    graph = generate_workflow(
                        args.workflow,
                        batch_size=max(1, branches),
                        num_branches=branches,
                        shared_state_size=shared_size,
                        unique_state_size=400,
                        dynamic_hot_state_probability=0.25,
                        seed=args.seed,
                    )
                    initialize_static_hotness(graph)
                    analysis = analyze_redundancy(graph)
                    topology = WaferTopology.from_mesh(mesh, region_memory_capacity=memory_cap)
                    sim_config = SimulationConfig(worker_count=min(8, max(1, branches)))
                    last_graph = graph
                    last_analysis = analysis
                    for baseline in baselines:
                        run = simulate_workflow(
                            graph,
                            baseline,
                            topology=topology,
                            config=sim_config,
                            state_policy="dynamic",
                        )
                        row = run.result.to_row()
                        row.update(
                            {
                                "mesh": mesh,
                                "region_memory_capacity": memory_cap,
                                "shared_state_size": shared_size,
                                "branch_width": branches,
                                "input_redundancy_ratio": analysis["summary"]["input_redundancy_ratio"],
                            }
                        )
                        rows.append(row)
                        for wave in run.wave_schedule:
                            wave = dict(wave)
                            wave.update(
                                {
                                    "mesh": mesh,
                                    "region_memory_capacity": memory_cap,
                                    "shared_state_size": shared_size,
                                    "branch_width": branches,
                                }
                            )
                            wave_rows.append(wave)
                        for decision in run.policy_decisions:
                            prow = decision.to_row()
                            prow.update(
                                {
                                    "baseline": baseline,
                                    "mesh": mesh,
                                    "region_memory_capacity": memory_cap,
                                    "shared_state_size": shared_size,
                                    "branch_width": branches,
                                }
                            )
                            policy_rows.append(prow)

    assert last_graph is not None and last_analysis is not None
    last_graph.export_csv(out)
    last_graph.export_json(out / "state_access_graph.json")
    write_json(out / "metadata.json", {"experiment": "wafer_sensitivity", "workflow": args.workflow})
    write_json(
        out / "config.json",
        {
            "workflow": args.workflow,
            "seed": args.seed,
            "meshes": meshes,
            "memory_caps": memory_caps,
            "shared_sizes": shared_sizes,
            "branch_widths": branch_widths,
            "baselines": baselines,
        },
    )
    write_csv(out / "state_hotness.csv", last_analysis["state_hotness"])
    write_csv(out / "policy_decisions.csv", policy_rows)
    write_csv(out / "wave_schedule.csv", wave_rows)
    write_csv(out / "simulation_summary.csv", rows)
    _write_report(out / "report.md", rows)
    _write_figures(out / "figures", rows)


def _write_report(path: Path, rows: list[dict[str, object]]) -> None:
    wins: dict[str, int] = {}
    cases: dict[tuple[str, int, int, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            str(row["mesh"]),
            int(row["region_memory_capacity"]),
            int(row["shared_state_size"]),
            int(row["branch_width"]),
        )
        cases.setdefault(key, []).append(row)
    for case_rows in cases.values():
        best = min(case_rows, key=lambda row: float(row["workflow_latency"]))
        wins[str(best["baseline"])] = wins.get(str(best["baseline"]), 0) + 1
    negative = [
        case_rows
        for case_rows in cases.values()
        if min(case_rows, key=lambda row: float(row["workflow_latency"]))["baseline"] != "WaferStateFlow"
    ]
    lines = [
        "# WaferStateFlow Wafer Sensitivity",
        "",
        "## Executive Summary",
        "",
        f"Swept {len(cases)} mesh/memory/state-size/branch-width cases. Win counts: {wins}.",
        "",
        "## Failure Cases",
        "",
    ]
    if negative:
        example = min(negative[0], key=lambda row: float(row["workflow_latency"]))
        lines.append(
            f"Example non-WaferStateFlow winner: `{example['baseline']}` on mesh {example['mesh']}, "
            f"memory {example['region_memory_capacity']}, shared size {example['shared_state_size']}, "
            f"branch width {example['branch_width']}. This indicates wafer-aware waves are not always necessary."
        )
    else:
        lines.append(
            "No non-WaferStateFlow winner appeared in this compact sweep; expand low-fanout and cache-rich regimes before making a platform necessity claim."
        )
    lines.extend(
        [
            "",
            "## What This Means for the Paper",
            "",
            "The sensitivity sweep is a prototype sanity check, not a calibrated hardware study. It is useful for finding regimes and counterexamples.",
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
    labels = []
    values = []
    for baseline in sorted({str(row["baseline"]) for row in rows}):
        subset = [float(row["workflow_latency"]) for row in rows if row["baseline"] == baseline]
        labels.append(baseline)
        values.append(sum(subset) / len(subset))
    plt.figure(figsize=(8, 4))
    plt.bar(labels, values)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Mean latency across sweep")
    plt.tight_layout()
    plt.savefig(out_dir / "sensitivity_mean_latency.png")
    plt.close()


if __name__ == "__main__":
    main()
