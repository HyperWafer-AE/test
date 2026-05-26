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
        writer = csv.DictWriter(f, fieldnames=names)
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


def _write_markdown_report(path: Path, summaries: list[dict[str, Any]]) -> None:
    decisions = {str(row["decision"]) for row in summaries}
    if "continue_to_flowmorph_scheduling" in decisions:
        overall = "continue to FlowMorph scheduling for workflows with strong irregularity"
    else:
        overall = "weak direction under these synthetic workflow settings"
    lines = [
        "# FlowMorph Problem Characterization",
        "",
        "## Executive Summary",
        "",
        "FlowMorph measures phase/resource irregularity in agent workflows using a PhaseDAG. "
        "This report does not make WaferStateFlow claims and does not implement wafer scheduling.",
        "",
        f"Overall gate: **{overall}**.",
        "",
        "## Workflow Metrics",
        "",
        "| workflow | max frontier | frontier CV | phase entropy | phase variation | critical path | work/CP | idle | partition imbalance | decision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in sorted(summaries, key=lambda item: str(item["workflow"])):
        lines.append(
            f"| `{row['workflow']}` | {int(row['max_frontier_width'])} | "
            f"{float(row['frontier_width_cv']):.2f} | "
            f"{float(row['mean_phase_mix_entropy']):.2f} | "
            f"{float(row['phase_mix_variation']):.2f} | "
            f"{float(row['critical_path_length']):.3f} | "
            f"{float(row['total_work_to_critical_path_ratio']):.2f} | "
            f"{float(row['fixed_worker_idle_fraction']):.2f} | "
            f"{float(row['fixed_prefill_decode_partition_imbalance']):.2f} | "
            f"{row['decision']} |"
        )
    lines.extend(
        [
            "",
            "## Decision Gate",
            "",
            "- If frontier width is mostly constant and phase mix is stable, this direction is weak.",
            "- If frontier width and phase mix both vary strongly, continue to FlowMorph scheduling.",
            "- This run is characterization only; no wafer scheduling or placement is implemented.",
            "",
            "## Interpretation",
            "",
        ]
    )
    weak = [row for row in summaries if row["decision"] == "weak_direction"]
    if weak:
        lines.append(
            "At least one workflow fails the irregularity gate. Treat FlowMorph scheduling as conditional, "
            "not as an assumed win across all agent workflows."
        )
    else:
        lines.append(
            "All workflows pass the irregularity gate in this synthetic run. The next step would be a "
            "non-wafer FlowMorph scheduler prototype and real trace validation."
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
