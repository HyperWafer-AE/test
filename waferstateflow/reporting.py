"""Output helpers for WaferStateFlow experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .ir import StateAccessGraph
from .simulator import SimulationRun
from .state_policy import PolicyDecision


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        path.write_text("", encoding="utf-8")
        return
    names = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _cell(row.get(name, "")) for name in names})


def write_characterization_outputs(
    graph: StateAccessGraph,
    analysis: dict[str, Any],
    policy_decisions: list[PolicyDecision],
    out_dir: str | Path,
    config: dict[str, Any],
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    graph.export_csv(out)
    graph.export_json(out / "state_access_graph.json")
    write_json(out / "metadata.json", {"experiment": "problem_characterization", "workflow": graph.graph_id})
    write_json(out / "config.json", config)
    write_csv(out / "state_hotness.csv", analysis["state_hotness"])
    write_csv(out / "policy_decisions.csv", [decision.to_row() for decision in policy_decisions])
    write_csv(
        out / "wave_schedule.csv",
        [],
        fieldnames=[
            "wave_id",
            "baseline",
            "scheduler",
            "seed_state_id",
            "operator_ids",
            "region_ids",
            "start_time",
            "end_time",
            "batch_size",
            "materialization_bytes",
            "movement_byte_hop",
            "max_link_load_delta",
            "benefit",
            "wait_penalty",
        ],
    )
    write_csv(
        out / "state_access_events.csv",
        [],
        fieldnames=[
            "baseline",
            "wave_id",
            "time",
            "operator_id",
            "state_id",
            "region_id",
            "policy_before",
            "policy_after",
            "dynamic_hotness_before",
            "dynamic_hotness_after",
            "materialization_bytes",
            "movement_byte_hop",
            "cache_hit",
        ],
    )
    write_csv(out / "simulation_summary.csv", [analysis["summary"]])
    _write_characterization_report(out / "report.md", analysis)
    _write_characterization_figures(out / "figures", analysis)


def write_scheduler_outputs(
    graph: StateAccessGraph,
    analysis: dict[str, Any],
    runs: list[SimulationRun],
    out_dir: str | Path,
    config: dict[str, Any],
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    graph.export_csv(out)
    graph.export_json(out / "state_access_graph.json")
    write_json(out / "metadata.json", {"experiment": "scheduler_comparison", "workflow": graph.graph_id})
    write_json(out / "config.json", config)
    write_csv(out / "state_hotness.csv", analysis["state_hotness"])
    all_policy_rows = []
    seen: set[tuple[str, str]] = set()
    for run in runs:
        for decision in run.policy_decisions:
            key = (run.result.baseline, decision.state_id)
            if key in seen:
                continue
            row = decision.to_row()
            row["baseline"] = run.result.baseline
            all_policy_rows.append(row)
            seen.add(key)
    write_csv(out / "policy_decisions.csv", all_policy_rows)
    wave_rows = []
    for run in runs:
        wave_rows.extend(run.wave_schedule)
    write_csv(out / "wave_schedule.csv", wave_rows)
    access_rows = []
    for run in runs:
        access_rows.extend(run.state_access_events)
    write_csv(out / "state_access_events.csv", access_rows)
    write_csv(out / "simulation_summary.csv", [run.result.to_row() for run in runs])
    _write_scheduler_report(out / "report.md", analysis, runs)
    _write_scheduler_figures(out / "figures", analysis, runs)


def _write_characterization_report(path: Path, analysis: dict[str, Any]) -> None:
    summary = analysis["summary"]
    hot = analysis["top_hot_states"][:10]
    fanout = analysis["state_fanout"][:10]
    lines = [
        "# WaferStateFlow Report",
        "",
        "## 1. Executive Summary",
        "",
        f"Workflow `{summary['workflow']}` has input redundancy ratio "
        f"{summary['input_redundancy_ratio']:.2f}. H1 is "
        f"{'supported' if summary['h1_supported'] else 'not supported'} under this synthetic setting. "
        f"Top-state cumulative hotness share is {summary['top_state_share']:.2f}; H2 is "
        f"{'supported' if summary['h2_supported'] else 'not supported'}.",
        "",
        "## 2. Problem Characterization",
        "",
        f"- Input redundancy ratio: {summary['input_redundancy_ratio']:.2f}",
        f"- Total prompt tokens: {summary['total_prompt_tokens']}",
        f"- Unique consumed state tokens: {summary['unique_state_tokens']}",
        f"- Duplicate materialization bytes: {summary['duplicate_materialization_bytes']}",
        f"- Dynamic hot states: {summary['dynamic_hot_state_count']}",
        "",
        "### Top Hot States",
        "",
        "| rank | state | kind | consumers | hotness | cumulative share |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in hot:
        lines.append(
            f"| {row['rank']} | `{row['state_id']}` | {row['kind']} | {row['consumer_count']} | "
            f"{row['hotness']:.1f} | {row['cumulative_hotness_fraction']:.2f} |"
        )
    lines.extend(["", "### State Fanout Table", "", "| state | kind | tokens | consumers | token-weighted fanout |", "| --- | --- | ---: | ---: | ---: |"])
    for row in fanout:
        lines.append(
            f"| `{row['state_id']}` | {row['kind']} | {row['token_size']} | "
            f"{row['consumer_count']} | {row['token_weighted_state_fanout']} |"
        )
    lines.extend(
        [
            "",
            "## 3. Method",
            "",
            "This run builds a State Access Graph and measures state-level fanout before any wafer claim is made.",
            "",
            "## 7. Failure Cases",
            "",
            "If the redundancy ratio is close to 1.0 or hotness share is flat, state-centric scheduling is not a strong paper claim for this workflow.",
            "",
            "## 8. What This Means for the Paper",
            "",
            "Use this characterization as a gate before scheduler results. Unsupported hypotheses should be reported rather than hidden.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_scheduler_report(path: Path, analysis: dict[str, Any], runs: list[SimulationRun]) -> None:
    summary = analysis["summary"]
    rows = sorted(runs, key=lambda run: run.result.workflow_latency)
    best = rows[0].result
    proposed = next(run.result for run in runs if run.result.baseline == "WaferStateFlow")
    failure = next(
        (
            run.result
            for run in rows
            if run.result.baseline != "WaferStateFlow"
            and run.result.workflow_latency <= proposed.workflow_latency
        ),
        None,
    )
    lines = [
        "# WaferStateFlow Report",
        "",
        "## 1. Executive Summary",
        "",
        f"Workflow `{summary['workflow']}` redundancy ratio is {summary['input_redundancy_ratio']:.2f}. "
        f"Best simulated scheduler is `{best.baseline}` with latency {best.workflow_latency:.3f}. "
        f"`WaferStateFlow` latency is {proposed.workflow_latency:.3f}.",
        "",
        "## 2. Problem Characterization",
        "",
        f"- Input redundancy ratio: {summary['input_redundancy_ratio']:.2f}",
        f"- Hotness top-share: {summary['top_state_share']:.2f}",
        f"- Dynamic hot-state count: {summary['dynamic_hot_state_count']}",
        "",
        "## 3. Method",
        "",
        "- State Access Graph keeps task, document, tool, intermediate, and output states explicit.",
        "- Hotness combines token size, future accesses, access cost, and criticality.",
        "- State policy chooses inline/cache/pin/replicate/shard/evict/recompute using expected saved cost.",
        "- State-centric wave scheduling forms ready-operator waves around hot input states.",
        "- Wafer placement uses a mesh byte-hop model with region memory pressure.",
        "",
        "## 4. Baselines",
        "",
        "| baseline | modeled behavior |",
        "| --- | --- |",
    ]
    for run in runs:
        lines.append(f"| `{run.result.baseline}` | {run.result.notes} |")
    lines.extend(
        [
            "",
            "## 5. Results",
            "",
            "| baseline | latency | materialization bytes | byte-hop | max link load | p95 link load | hotspot | max link util | memory pressure | crit wait | avg wave |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for run in sorted(runs, key=lambda item: item.result.baseline):
        result = run.result
        lines.append(
            f"| `{result.baseline}` | {result.workflow_latency:.3f} | "
            f"{result.state_materialization_bytes} | {result.state_movement_byte_hop} | "
            f"{result.max_link_load} | {result.p95_link_load:.1f} | {result.hotspot_region or '-'} | "
            f"{result.max_link_utilization:.3f} | {result.region_memory_pressure:.3f} | "
            f"{result.critical_path_wait:.3f} | {result.average_wave_batch_size:.2f} |"
        )
    lines.extend(["", "## 6. Ablations", ""])
    replicate = next((run.result for run in runs if run.result.baseline == "replicate_all_hot_states"), None)
    pin = next((run.result for run in runs if run.result.baseline == "single_pin_hot_state"), None)
    if replicate and pin:
        lines.append(
            f"`replicate_all_hot_states` gives latency {replicate.workflow_latency:.3f} with memory pressure "
            f"{replicate.region_memory_pressure:.3f}; `single_pin_hot_state` gives latency {pin.workflow_latency:.3f} "
            f"and byte-hop {pin.state_movement_byte_hop}."
        )
    lines.extend(["", "## 7. Failure Cases", ""])
    if failure:
        lines.append(
            f"In this run `{failure.baseline}` is no worse than WaferStateFlow. This is a required negative case: "
            "when hot states are small enough for simpler cache-aware baselines, wafer placement is not necessary."
        )
    else:
        lines.append(
            "No baseline beat WaferStateFlow in this run, but this does not prove the platform claim. "
            "Run low-fanout or memory-rich sweeps to find negative cases."
        )
    lines.extend(
        [
            "",
            "## 8. What This Means for the Paper",
            "",
            f"- H1: {'supported' if summary['h1_supported'] else 'unsupported'} for this workflow.",
            f"- H2: {'supported' if summary['h2_supported'] else 'unsupported'} for this workflow.",
            "- H3: requires dynamic-hotness sweep; this single run is not enough.",
            "- H4: supported only if WaferStateFlow beats request/operator-centric baselines outside cache-rich regimes.",
            "- Next experiments: dynamic-hotness probability sweep, fanout sweep, and memory-capacity sensitivity.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_characterization_figures(out_dir: Path, analysis: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    rows = analysis["state_hotness"][:20]
    if rows:
        plt.figure(figsize=(7, 4))
        plt.plot([row["rank"] for row in rows], [row["cumulative_hotness_fraction"] for row in rows], marker="o")
        plt.xlabel("State rank")
        plt.ylabel("Cumulative hotness share")
        plt.tight_layout()
        plt.savefig(out_dir / "hotness_skew.png")
        plt.close()
    fanout = analysis["state_fanout"][:12]
    if fanout:
        plt.figure(figsize=(8, 4))
        plt.bar([row["state_id"] for row in fanout], [row["consumer_count"] for row in fanout])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Consumers")
        plt.tight_layout()
        plt.savefig(out_dir / "state_fanout.png")
        plt.close()


def _write_scheduler_figures(out_dir: Path, analysis: dict[str, Any], runs: list[SimulationRun]) -> None:
    _write_characterization_figures(out_dir, analysis)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    labels = [run.result.baseline for run in runs]
    latencies = [run.result.workflow_latency for run in runs]
    plt.figure(figsize=(9, 4))
    plt.bar(labels, latencies)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Workflow latency")
    plt.tight_layout()
    plt.savefig(out_dir / "scheduler_latency.png")
    plt.close()


def _cell(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, sort_keys=True)
    return value
