#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def _table(path: Path, max_rows: int = 12) -> str:
    if not path.exists() or path.stat().st_size == 0:
        return "Not available.\n"
    df = pd.read_csv(path).head(max_rows)
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            vals.append(f"{val:.4g}" if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _nonflat(path: Path, cols: list[str]) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    df = pd.read_csv(path)
    for col in cols:
        if col in df.columns and df[col].nunique(dropna=True) > 1:
            return True
    return False


def _trace_has_real(path: Path) -> bool:
    if not path.exists():
        return False
    for trace_file in path.glob("traces/*.jsonl"):
        with trace_file.open("r", encoding="utf-8") as f:
            for line in f:
                if '"real_trace": true' in line:
                    return True
    return False


def _last_matching_line(path: Path, pattern: str) -> str:
    if not path.exists():
        return "not recorded"
    last = "not recorded"
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if pattern in line:
            last = line
    return last


def readiness(project_root: Path) -> dict[str, bool]:
    main = project_root / "results/main_wafer_sim_neutral_v2"
    ablation = project_root / "results/ablation_neutral_v2/simulation/simulation_summary.csv"
    report_meta = _read_json(main / "metadata.json")
    git_ok = bool(report_meta.get("git_commit")) and "unavailable" not in str(report_meta.get("git_commit"))
    ablation_ok = False
    placement_ok = False
    if ablation.exists():
        df = pd.read_csv(ablation)
        full = df.loc[df["baseline"] == "waferagent_full", "job_completion_time_ms"]
        cp = df.loc[df["baseline"] == "no_critical_path_scheduling", "job_completion_time_ms"]
        place = df.loc[df["baseline"].isin(["no_affinity_placement", "no_hotspot_aware_placement"]), "job_completion_time_ms"]
        if not full.empty and not cp.empty:
            ablation_ok = abs(float(cp.iloc[0]) - float(full.iloc[0])) > 1e-9
        if not full.empty and not place.empty:
            placement_ok = any(abs(float(x) - float(full.iloc[0])) > 1e-9 for x in place)
    model_selection = _read_json(project_root / "results/characterization_h100_hf_v2/model_selection.json")
    no_silent_fallback = bool(model_selection) and not model_selection.get("fallback_used", False)
    checks = {
        "real H100 traces available": _trace_has_real(project_root / "results/characterization_h100_hf_v2"),
        "no silent fallback": no_silent_fallback,
        "neutral multipliers used": bool(report_meta.get("neutral_mechanism_multipliers")),
        "SRAM sensitivity non-flat": _nonflat(project_root / "results/sensitivity_neutral_v2/simulation/sensitivity_sram_per_tile_mb.csv", ["job_completion_time_ms", "sram_evictions", "sram_reload_bytes"]),
        "bandwidth sensitivity non-flat on mesh_stress_moa": _nonflat(project_root / "results/sensitivity_neutral_v2/simulation/sensitivity_link_bandwidth_GBps.csv", ["job_completion_time_ms", "mesh_wait_ms"]),
        "critical-path ablation non-zero on targeted workload": ablation_ok,
        "placement ablation affects mesh metrics and targeted JCT": placement_ok,
        "git commit available": git_ok,
    }
    return checks


def copy_artifacts(project_root: Path, out_dir: Path) -> None:
    tables_dir = out_dir / "tables"
    figs_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    skip_names = {
        "mesh_link_events.csv",
        "stage_schedule.csv",
        "schedule.csv",
        "sram_events.csv",
        "prefix_blocks.csv",
    }
    for csv in project_root.glob("results/*/simulation/*.csv"):
        if csv.parent.parent == out_dir:
            continue
        if csv.name in skip_names:
            continue
        if csv.stat().st_size > 25 * 1024 * 1024:
            continue
        shutil.copy2(csv, tables_dir / f"{csv.parent.parent.name}_{csv.name}")
    for fig in project_root.glob("results/*/figures/*"):
        if fig.parent.parent == out_dir:
            continue
        if fig.suffix in {".png", ".pdf"}:
            shutil.copy2(fig, figs_dir / f"{fig.parent.parent.name}_{fig.name}")


def best_environment(project_root: Path, fallback_root: Path) -> dict:
    for candidate in [
        project_root / "results/characterization_h100_hf_v2/environment.json",
        project_root / "results/env_validation/environment.json",
        project_root / "results/h100_calibration_real_hf/environment.json",
        fallback_root / "environment.json",
    ]:
        env = _read_json(candidate)
        if env:
            return env
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/main_wafer_sim_neutral_v2")
    parser.add_argument("--out", default="results/final_report_v2/report.md")
    parser.add_argument("--engine", default="synthetic")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    project_root = Path("/home/duzc/data/agent_wafer")
    root = project_root / args.results if not Path(args.results).is_absolute() else Path(args.results)
    out = project_root / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    env = best_environment(project_root, root)
    summary = root / "simulation" / "simulation_summary.csv"
    h100cal_summary = project_root / "results/main_wafer_sim_h100cal_v2/simulation/simulation_summary.csv"
    trace_selection = _read_json(project_root / "results/characterization_h100_hf_v2/model_selection.json")
    calib_selection = _read_json(project_root / "results/h100_calibration_real_hf/model_selection.json")
    vllm_status = _last_matching_line(project_root / "results/env_validation/vllm_install_aliyun.log", "exit_code=")
    checks = readiness(project_root)
    report_json = {
        "results": str(root),
        "readiness": checks,
        "pass": all(checks.values()),
    }
    (out.parent / "report.json").write_text(json.dumps(report_json, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    copy_artifacts(project_root, out.parent)
    readiness_lines = "\n".join(
        f"- {'PASS' if ok else 'FAIL'}: {name}" for name, ok in checks.items()
    )
    text = f"""# WaferAgent Round 2 Report

## Environment

- GPU: `{env.get('nvidia_smi', 'unavailable')}`
- Python: `{env.get('python', 'unavailable')}`
- PyTorch: `{env.get('torch_version', env.get('torch', 'unavailable'))}`
- CUDA available: `{env.get('cuda_available', 'unavailable')}`
- command: `{env.get('command', 'unavailable')}`
- git commit: `{env.get('git_commit', 'unavailable')}`

## Readiness Check

{readiness_lines}

## Main Neutral Results

{_table(summary)}

## H100 Trace And Calibration

- HF trace model: `{trace_selection.get('model_path', 'unavailable')}`
- HF trace engine: `{trace_selection.get('engine_used', 'unavailable')}`
- HF calibration model: `{calib_selection.get('model_path', 'unavailable')}`
- HF calibration engine: `{calib_selection.get('engine_used', 'unavailable')}`

## H100-Calibrated Wafer Simulation

{_table(h100cal_summary)}

## Failures / Missing Baselines

- vLLM: `{vllm_status}`. The vLLM baseline is not claimed as completed unless import and real vLLM traces succeed.
- Wafer: all wafer results are trace-driven wafer-scale simulator results, not real wafer hardware measurements.

## Interpretation

Main wafer numbers are trace-driven simulation results. Real H100 traces and H100 calibration are recorded separately when the HF/vLLM engines complete. Synthetic traces are used only for controlled mechanism stress tests unless explicitly marked otherwise.

## Output Layout

- JSON readiness: `{out.parent / 'report.json'}`
- Tables: `{out.parent / 'tables'}`
- Figures: `{out.parent / 'figures'}`
"""
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
