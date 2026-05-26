"""Report/export helpers for FlowMorph characterization."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


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
        writer = csv.DictWriter(f, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _cell(row.get(name, "")) for name in names})


def write_flowmorph_report(
    out_dir: str | Path,
    summaries: list[dict[str, Any]],
    timeline_rows: list[dict[str, Any]],
    operator_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "metadata.json", {"experiment": "flowmorph_characterization"})
    write_json(out / "config.json", config)
    write_csv(out / "flowmorph_summary.csv", summaries)
    write_csv(out / "frontier_timeline.csv", timeline_rows)
    write_csv(out / "phase_operators.csv", operator_rows)
    _write_markdown_report(out / "report.md", summaries)
    _write_figures(out / "figures", summaries)


def write_flowmorph_scheduler_report(
    out_dir: str | Path,
    summaries: list[dict[str, Any]],
    schedule_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "metadata.json", {"experiment": "flowmorph_v1_scheduler_comparison"})
    write_json(out / "config.json", config)
    write_csv(out / "scheduler_summary.csv", summaries)
    write_csv(out / "scheduler_trace.csv", schedule_rows)
    write_csv(out / "workflow_selection.csv", selection_rows)
    _write_scheduler_markdown_report(out / "report.md", summaries, selection_rows)


def _write_markdown_report(path: Path, summaries: list[dict[str, Any]]) -> None:
    frontier_candidates = [
        row for row in summaries
        if row.get("opportunity_taxonomy") in {"frontier_only", "frontier_and_phase"}
    ]
    combined_candidates = [
        row for row in summaries if row.get("opportunity_taxonomy") == "frontier_and_phase"
    ]
    v1_decision = (
        "continue FlowMorph-v1"
        if len(frontier_candidates) >= 2
        else "do not continue FlowMorph-v1 from this synthetic evidence alone"
    )
    v2_decision = (
        "continue FlowMorph-v2"
        if len(combined_candidates) >= 2
        else "do not continue FlowMorph-v2 from this synthetic evidence alone"
    )
    lines = [
        "# FlowMorph Problem Characterization",
        "",
        "## Executive Summary",
        "",
        "FlowMorph measures frontier-aware phase/resource irregularity in agent workflows using a PhaseDAG. "
        "This report does not make WaferStateFlow claims and does not implement wafer scheduling.",
        "",
        f"FlowMorph-v1 gate: **{v1_decision}** "
        f"({len(frontier_candidates)} workflows are frontier_only or frontier_and_phase).",
        f"FlowMorph-v2 gate: **{v2_decision}** "
        f"({len(combined_candidates)} workflows are frontier_and_phase).",
        "",
        "## Workflow Metrics",
        "",
        "| workflow | taxonomy | frontier_morphing_opportunity | phase_morphing_opportunity | combined_opportunity | frontier CV | max frontier | median frontier | width drop | wide work | narrow critical | serial frac | parallel slack | phase variation | idle | partition imbalance |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(summaries, key=lambda item: str(item["workflow"])):
        lines.append(
            f"| `{row['workflow']}` | {row['opportunity_taxonomy']} | "
            f"{row['frontier_morphing_opportunity']} | "
            f"{row['phase_morphing_opportunity']} | "
            f"{row['combined_opportunity']} | "
            f"{float(row['frontier_width_cv']):.2f} | "
            f"{int(row['max_frontier_width'])} | "
            f"{float(row['median_frontier_width']):.2f} | "
            f"{float(row['width_drop_ratio']):.2f} | "
            f"{float(row['wide_stage_work_fraction']):.2f} | "
            f"{float(row['narrow_critical_stage_fraction']):.2f} | "
            f"{float(row['critical_path_serial_fraction']):.2f} | "
            f"{float(row['parallel_slack']):.2f} | "
            f"{float(row['phase_mix_variation']):.2f} | "
            f"{float(row['fixed_worker_idle_fraction']):.2f} | "
            f"{float(row['fixed_prefill_decode_partition_imbalance']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Decision Gate",
            "",
            "- frontier_only: frontier width varies strongly or parallel slack is high, while phase mix is stable.",
            "- phase_only: frontier variation is low, while phase mix varies strongly.",
            "- frontier_and_phase: both frontier opportunity and phase opportunity are present.",
            "- weak: neither frontier nor phase opportunity crosses the thresholds.",
            "- Continue FlowMorph-v1 only if multiple workflows are frontier_only or frontier_and_phase.",
            "- Continue FlowMorph-v2 only if multiple workflows are frontier_and_phase.",
            "- This run is characterization only; no wafer scheduling or placement is implemented.",
            "- Workflow generators were not tuned to force success; negative results remain in the table.",
            "",
            "## Interpretation",
            "",
        ]
    )
    negative = [row for row in summaries if row["opportunity_taxonomy"] in {"phase_only", "weak"}]
    if negative:
        lines.append(
            "At least one workflow does not support frontier-aware morphing. Treat FlowMorph as conditional, "
            "not as an assumed win across all agent workflows."
        )
    else:
        lines.append(
            "All workflows expose frontier-aware morphing opportunity in this synthetic run. The next step "
            "would still be a non-wafer FlowMorph scheduler prototype and real trace validation."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_scheduler_markdown_report(
    path: Path,
    summaries: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
) -> None:
    selected = [row for row in selection_rows if row["selected"] == "yes"]
    positive = [
        row for row in selected
        if row["selection_reason"] == "frontier_positive"
    ]
    control = [
        row for row in selected
        if row["selection_reason"] == "negative_control"
    ]
    lines = [
        "# FlowMorph-v1 Frontier Scheduler Prototype",
        "",
        "## Executive Summary",
        "",
        "This experiment uses an abstract worker-resource model over PhaseDAG inputs. "
        "It does not implement wafer-specific placement and makes no wafer performance claims.",
        "",
        f"Selected frontier-positive workflows: {len(positive)}. Negative controls: {len(control)}.",
        "",
        "## Workflow Selection",
        "",
        "| workflow | taxonomy | selected | reason |",
        "| --- | --- | --- | --- |",
    ]
    for row in sorted(selection_rows, key=lambda item: str(item["workflow"])):
        lines.append(
            f"| `{row['workflow']}` | {row['opportunity_taxonomy']} | "
            f"{row['selected']} | {row['selection_reason']} |"
        )
    lines.extend(
        [
            "",
            "## Scheduler Metrics",
            "",
            "| workflow | scheduler | policy | taxonomy | latency | idle | critical path delay | mode switches | wide utilization | narrow latency |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(summaries, key=lambda item: (str(item["workflow"]), str(item["scheduler"]))):
        lines.append(
            f"| `{row['workflow']}` | {row['scheduler']} | {row['policy']} | "
            f"{row['opportunity_taxonomy']} | "
            f"{float(row['workflow_latency']):.3f} | "
            f"{float(row['worker_idle_fraction']):.2f} | "
            f"{float(row['critical_path_delay']):.3f} | "
            f"{int(row['mode_switch_count'])} | "
            f"{float(row['wide_stage_utilization']):.2f} | "
            f"{float(row['narrow_stage_latency']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- `frontier_aware_morphing` uses parallel mode on wide frontiers.",
            "- It uses consolidated fast-lane mode for narrow ready sets with critical operators.",
            "- Weak-frontier workflows fall back to `fixed_worker_pool`; `iterative` is kept as a negative control.",
            "- Baselines include `fixed_worker_pool`, `static_full_resource`, `static_split_resource`, `always_parallel`, and `always_consolidated`.",
            "- These results are scheduler-prototype evidence only, not wafer placement or wafer speedup evidence.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_figures(out_dir: Path, summaries: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not summaries:
        return
    rows = sorted(summaries, key=lambda item: str(item["workflow"]))
    labels = [str(row["workflow"]) for row in rows]
    plt.figure(figsize=(8, 4))
    plt.bar(labels, [float(row["frontier_width_cv"]) for row in rows], label="frontier CV")
    plt.plot(labels, [float(row["phase_mix_variation"]) for row in rows], marker="o", label="phase variation")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Irregularity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "flowmorph_irregularity.png")
    plt.close()


def _cell(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, sort_keys=True)
    return value
