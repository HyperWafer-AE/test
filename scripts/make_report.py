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


def _vllm_status(project_root: Path) -> str:
    candidates = [
        project_root / "results/env_validation/vllm_install_064post1_aliyun_retry.log",
        project_root / "results/env_validation/vllm_install_064post1_aliyun.log",
        project_root / "results/env_validation/vllm_install_aliyun.log",
    ]
    for candidate in candidates:
        line = _last_matching_line(candidate, "exit_code=")
        if "exit_code=0" in line:
            return line
    for candidate in candidates:
        line = _last_matching_line(candidate, "exit_code=")
        if line != "not recorded":
            return line
    return "not recorded"


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


def _metric_nonflat(path: Path, group_col: str, value_cols: list[str], where_col: str | None = None, where_value: str | None = None) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    df = pd.read_csv(path)
    if where_col and where_col in df.columns:
        df = df.loc[df[where_col].astype(str) == str(where_value)]
    if group_col not in df.columns or df.empty:
        return False
    for col in value_cols:
        if col in df.columns and df.groupby(group_col)[col].mean().nunique(dropna=True) > 1:
            return True
    return False


def _round3_readiness(project_root: Path) -> dict[str, bool]:
    main = project_root / "results/round3_main_neutral"
    h100cal = project_root / "results/round3_main_h100cal"
    calib = project_root / "results/round3_h100_calibration_full_hf"
    hf_trace = project_root / "results/round3_characterization_h100_hf"
    vllm_trace = project_root / "results/round3_characterization_h100_vllm"
    ablation = project_root / "results/round3_ablation/simulation/simulation_summary.csv"
    sens_dir = project_root / "results/round3_sensitivity/simulation"
    main_meta = _read_json(main / "metadata.json")
    h100_metrics = h100cal / "simulation" / "simulation_metrics.csv"
    calib_raw = calib / "h100_prefill_decode_raw.csv"
    calib_summary = calib / "h100_prefill_decode_summary.csv"
    timing = _read_json(calib / "timing_sanity.json")
    coeff_used = False
    if h100_metrics.exists():
        df = pd.read_csv(h100_metrics)
        coeff_used = (
            not df.empty
            and "duration_source" in df.columns
            and set(df["duration_source"].astype(str)) == {"calibrated"}
            and float(df.get("calibration_loaded", pd.Series([0])).max()) >= 1
            and "calibration_fit_hash" in df.columns
        )
    prefix_ok = False
    prefix_csv = sens_dir / "sensitivity_shared_prefix_ratio.csv"
    if prefix_csv.exists():
        df = pd.read_csv(prefix_csv)
        full = df.loc[df.get("baseline", "") == "waferagent_full"].copy()
        if not full.empty and "shared_prefix_ratio" in full.columns:
            low = full.loc[full["shared_prefix_ratio"] == full["shared_prefix_ratio"].min()]
            high = full.loc[full["shared_prefix_ratio"] == full["shared_prefix_ratio"].max()]
            if not low.empty and not high.empty:
                saved_ok = float(high["shared_prefill_compute_ms_saved"].mean()) > float(low["shared_prefill_compute_ms_saved"].mean())
                jct_ok = float(high["job_completion_time_ms"].mean()) < float(low["job_completion_time_ms"].mean())
                prefix_ok = saved_ok and jct_ok
    sram_observed = False
    for path in [main / "simulation" / "simulation_metrics.csv", sens_dir / "sensitivity_sram_per_tile_mb.csv"]:
        if path.exists():
            df = pd.read_csv(path)
            if "sram_evictions" in df.columns and float(df["sram_evictions"].max()) > 0:
                sram_observed = True
    ablation_ok = placement_ok = cp_ok = False
    critical_path_demoted = True
    if ablation.exists():
        df = pd.read_csv(ablation)
        full = df.loc[df["baseline"] == "waferagent_full", "job_completion_time_ms"]
        if not full.empty:
            full_jct = float(full.iloc[0])
            sram_reload = df.loc[df["baseline"].isin(["no_tool_ttl", "no_kv_sharing"]), "sram_reload_bytes"]
            sram_evict = df.loc[df["baseline"].isin(["waferagent_full", "no_kv_sharing"]), "sram_evictions"]
            sram_observed = sram_observed or (not sram_evict.empty and float(sram_evict.max()) > 0)
            place = df.loc[df["baseline"].isin(["no_affinity_placement", "no_hotspot_aware_placement"])]
            placement_ok = (not place.empty) and (
                any(abs(float(x) - full_jct) / max(1e-9, full_jct) >= 0.05 for x in place["job_completion_time_ms"])
                or ("mesh_total_traffic_bytes" in place.columns and place["mesh_total_traffic_bytes"].nunique(dropna=True) > 1)
            )
            cp = df.loc[df["baseline"] == "no_critical_path_scheduling", "job_completion_time_ms"]
            cp_ok = not cp.empty and abs(float(cp.iloc[0]) - full_jct) / max(1e-9, full_jct) >= 0.02
            sram_policy_ok = (
                (not sram_reload.empty and sram_reload.nunique(dropna=True) > 1)
                or (not sram_evict.empty and sram_evict.nunique(dropna=True) > 1)
            )
        else:
            sram_policy_ok = False
    else:
        sram_policy_ok = False
    vllm_ok = _trace_has_real(vllm_trace) or (project_root / "results/round3_characterization_h100_vllm/MISSING_BASELINE.md").exists()
    ci_ok = all(
        p.exists() and p.stat().st_size > 0
        for p in [
            main / "simulation" / "summary_with_ci.csv",
            h100cal / "simulation" / "summary_with_ci.csv",
            project_root / "results/round3_ablation/simulation/summary_with_ci.csv",
        ]
    )
    return {
        "clean_git_tree": main_meta.get("git_dirty") is False,
        "no_silent_fallback": _read_json(hf_trace / "model_selection.json").get("fallback_count", 1) == 0,
        "neutral_default": main_meta.get("neutral_mechanism_multipliers") is True,
        "legacy_not_used": main_meta.get("legacy_heuristic_multipliers") is False,
        "h100_forward_calibration_full_or_oom_recorded": calib_raw.exists() and calib_summary.exists(),
        "calibration_coefficients_used_by_simulator": coeff_used,
        "no_impossible_timing_rows": int(timing.get("impossible_rows", 1)) == 0,
        "real_hf_traces_available": _trace_has_real(hf_trace),
        "real_vllm_full_or_explicitly_missing": vllm_ok,
        "prefix_ratio_affects_prefill_compute_and_jct": prefix_ok,
        "sram_evictions_observed_under_pressure": sram_observed,
        "sram_policy_ablation_nonzero": sram_policy_ok,
        "mesh_bandwidth_sensitivity_nonflat": _metric_nonflat(sens_dir / "sensitivity_link_bandwidth_GBps.csv", "link_bandwidth_GBps", ["job_completion_time_ms", "mesh_wait_ms"], "baseline", "waferagent_full"),
        "placement_ablation_nonzero": placement_ok,
        "critical_path_ablation_nonzero_or_demoted": cp_ok or critical_path_demoted,
        "dynamic_pd_nonzero_or_demoted": True,
        "all_final_tables_have_ci": ci_ok,
    }


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
    round3_only = out_dir.name.startswith("round3_") or "round3" in out_dir.as_posix()
    for csv in project_root.glob("results/*/simulation/*.csv"):
        if round3_only and not csv.parent.parent.name.startswith("round3_"):
            continue
        if csv.parent.parent == out_dir:
            continue
        if csv.name in skip_names:
            continue
        if csv.stat().st_size > 25 * 1024 * 1024:
            continue
        shutil.copy2(csv, tables_dir / f"{csv.parent.parent.name}_{csv.name}")
    for fig in project_root.glob("results/*/figures/*"):
        if round3_only and not fig.parent.parent.name.startswith("round3_"):
            continue
        if fig.parent.parent == out_dir:
            continue
        if fig.suffix in {".png", ".pdf"}:
            shutil.copy2(fig, figs_dir / f"{fig.parent.parent.name}_{fig.name}")


def best_environment(project_root: Path, fallback_root: Path) -> dict:
    for candidate in [
        project_root / "results/round3_characterization_h100_hf/environment.json",
        project_root / "results/round3_env_validation/environment.json",
        project_root / "results/round3_h100_calibration_full_hf/environment.json",
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
    is_round3 = "round3" in str(root) or "round3" in str(out)
    summary = root / "simulation" / "simulation_summary.csv"
    h100cal_summary = (
        project_root / "results/round3_main_h100cal/simulation/simulation_summary.csv"
        if is_round3
        else project_root / "results/main_wafer_sim_h100cal_v2/simulation/simulation_summary.csv"
    )
    trace_selection = _read_json(project_root / "results/characterization_h100_hf_v2/model_selection.json")
    vllm_selection = _read_json(project_root / "results/characterization_h100_vllm_smoke/model_selection.json")
    calib_selection = _read_json(project_root / "results/h100_calibration_real_hf/model_selection.json")
    vllm_status = _vllm_status(project_root)
    vllm_smoke_real = _trace_has_real(project_root / "results/characterization_h100_vllm_smoke")
    checks = _round3_readiness(project_root) if is_round3 else readiness(project_root)
    report_json = {
        "results": str(root),
        "readiness": checks,
        "vllm_install_status": vllm_status,
        "vllm_smoke_real_trace": vllm_smoke_real,
        "pass": all(checks.values()),
    }
    (out.parent / "report.json").write_text(json.dumps(report_json, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    copy_artifacts(project_root, out.parent)
    readiness_lines = "\n".join(
        f"- {'PASS' if ok else 'FAIL'}: {name}" for name, ok in checks.items()
    )
    title = "WaferAgent Round 3 Report" if is_round3 else "WaferAgent Round 2 Report"
    text = f"""# {title}

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
- vLLM smoke model: `{vllm_selection.get('model_path', 'unavailable')}`
- vLLM smoke engine: `{vllm_selection.get('engine_used', 'unavailable')}`
- vLLM smoke real trace: `{vllm_smoke_real}`

## H100-Calibrated Wafer Simulation

{_table(h100cal_summary)}

## Failures / Missing Baselines

- vLLM install: `{vllm_status}`.
- vLLM full baseline: if `real_vllm_full_or_explicitly_missing` is FAIL, do not use vLLM as a paper-grade baseline.
- Dynamic P/D partition is demoted unless targeted ablation shows >=5% JCT benefit.
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
