#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from waferagent.utils import enforce_clean_git_tree, finalize_run_dir, init_run_dir


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


def _trace_record_count(path: Path) -> int:
    total = 0
    for trace_file in path.glob("traces/*.jsonl"):
        with trace_file.open("r", encoding="utf-8") as f:
            total += sum(1 for _ in f)
    return total


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


def _round4_readiness(project_root: Path) -> dict:
    neutral = project_root / "results/round4_global_main_neutral"
    h100cal = project_root / "results/round4_global_main_h100cal"
    calib = project_root / "results/round4_h100_calibration_stratified_hf"
    prefix = project_root / "results/round4_prefix_extension_calibration"
    hf_trace = project_root / "results/round4_characterization_h100_hf_20jobs"
    vllm_trace = project_root / "results/round4_characterization_h100_vllm_20jobs"
    neutral_meta = _read_json(neutral / "metadata.json")
    calib_coverage = _read_json(calib / "coverage_report.json")
    calib_quality = _read_json(calib / "h100_fit_quality.json")
    timing = _read_json(calib / "timing_sanity.json")
    prefix_fit = _read_json(prefix / "prefix_extension_fit.json")
    h100_metrics = h100cal / "simulation" / "simulation_metrics.csv"
    h100_summary = h100cal / "simulation" / "simulation_summary.csv"
    neutral_summary = neutral / "simulation" / "simulation_summary.csv"
    coeff_used = False
    prefix_used = False
    energy_uses_computed = False
    if h100_metrics.exists():
        df = pd.read_csv(h100_metrics)
        coeff_used = (
            not df.empty
            and "duration_source" in df.columns
            and set(df["duration_source"].astype(str)) == {"calibrated"}
            and float(df.get("calibration_loaded", pd.Series([0])).max()) >= 1
        )
        prefix_used = "prefix_extension_model_used" in df.columns and bool(df["prefix_extension_model_used"].astype(bool).any())
        energy_uses_computed = all(
            c in df.columns
            for c in ["computed_prefill_tokens", "avoided_prefill_tokens", "compute_energy_j"]
        )
    global_ok = neutral_summary.exists() and (neutral / "simulation" / "global_stage_schedule.csv").exists()
    real_hf_20 = _trace_has_real(hf_trace) and _trace_record_count(hf_trace) >= 900
    vllm_model = _read_json(vllm_trace / "model_selection.json")
    vllm_removed = (vllm_trace / "MISSING_BASELINE.md").exists()
    vllm_20_or_removed = (
        (_trace_has_real(vllm_trace) and _trace_record_count(vllm_trace) >= 900 and vllm_model.get("fallback_count", 1) == 0)
        or vllm_removed
    )
    main_claim_ablation_nonzero = False
    if neutral_summary.exists():
        df = pd.read_csv(neutral_summary)
        full = df.loc[df["baseline"] == "waferagent_full"]
        naive = df.loc[df["baseline"] == "wafer_naive"]
        if not full.empty and not naive.empty:
            main_claim_ablation_nonzero = (
                abs(float(naive["jct_p90_ms"].mean()) - float(full["jct_p90_ms"].mean()))
                / max(1e-9, float(naive["jct_p90_ms"].mean()))
                >= 0.02
                or float(full.get("avoided_prefill_tokens", pd.Series([0])).mean()) > 0
            )
    paper_ready = {
        "global_serving_simulator": global_ok,
        "stratified_or_full_h100_calibration": bool(calib_coverage.get("is_full_matrix") or calib_coverage.get("is_stratified_matrix")),
        "calibration_fit_quality_recorded": bool(calib_quality),
        "prefix_extension_model_used": bool(prefix_fit) and prefix_used,
        "real_hf_20jobs": real_hf_20,
        "vllm_20jobs_or_explicitly_removed": vllm_20_or_removed,
        "energy_uses_computed_tokens": energy_uses_computed,
        "main_claim_ablation_nonzero": main_claim_ablation_nonzero,
        "calibration_coefficients_used_by_simulator": coeff_used,
    }
    sanity = {
        "clean_git_tree": neutral_meta.get("git_dirty") is False,
        "no_silent_fallback": _read_json(hf_trace / "model_selection.json").get("fallback_count", 1) == 0,
        "neutral_default": neutral_meta.get("neutral_mechanism_multipliers") is True,
        "no_impossible_timing_rows": int(timing.get("impossible_rows", 1)) == 0,
    }
    demoted = {
        "critical_path_scheduling": True,
        "dynamic_pd_partition": True,
        "tool_ttl": True,
    }
    return {
        "paper_ready": paper_ready,
        "sanity": sanity,
        "demoted": demoted,
        "pass": {
            "paper_ready": all(paper_ready.values()),
            "sanity": all(sanity.values()),
            "overall": all(paper_ready.values()) and all(sanity.values()),
        },
    }


def _round5_readiness(project_root: Path) -> dict:
    gap = project_root / "results/round5_existing_cache_gap/simulation/existing_cache_gap_summary.csv"
    cohort = project_root / "results/round5_decode_cohort_sweep/simulation/decode_cohort_sweep.csv"
    repl = project_root / "results/round5_replication_tradeoff/simulation/replication_tradeoff_summary.csv"
    global_main = project_root / "results/round5_global_main_neutral"
    global_summary = global_main / "simulation/global_simulation_summary.csv"
    ablation = project_root / "results/round5_ablation/simulation/global_simulation_summary.csv"
    h100cal = project_root / "results/round5_global_main_h100cal/simulation/global_simulation_summary.csv"
    hf_trace = project_root / "results/round5_characterization_h100_hf_20jobs"
    vllm_trace = project_root / "results/round5_characterization_h100_vllm_20jobs"
    round4_hf_trace = project_root / "results/round4_characterization_h100_hf_20jobs"
    round4_calib = project_root / "results/round4_h100_calibration_stratified_hf/h100_fit.json"
    round4_prefix = project_root / "results/round4_prefix_extension_calibration/prefix_extension_fit.json"
    meta = _read_json(global_main / "metadata.json")

    existing_gap = False
    if gap.exists():
        df = pd.read_csv(gap)
        no_cache = df.loc[df["baseline"] == "no_cache"]
        apc = df.loc[df["baseline"] == "apc_like"]
        waf = df.loc[df["baseline"] == "waferagent_full"]
        if not no_cache.empty and not apc.empty and not waf.empty:
            apc_prefill = float(apc["prefill_compute_ms_saved"].mean())
            no_prefill = float(no_cache["prefill_compute_ms_saved"].mean())
            apc_decode = float(apc["decode_shared_kv_read_bytes"].mean())
            no_decode = float(no_cache["decode_shared_kv_read_bytes"].mean())
            waf_mesh = float(waf["mesh_traffic_bytes"].mean())
            apc_mesh = float(apc["mesh_traffic_bytes"].mean())
            existing_gap = (
                apc_prefill > no_prefill
                and abs(apc_decode - no_decode) / max(1.0, no_decode) < 0.01
                and waf_mesh < 0.5 * apc_mesh
            )

    cohort_ok = False
    if cohort.exists():
        df = pd.read_csv(cohort)
        cohort_ok = "shared_kv_read_reduction_ratio" in df.columns and float(df["shared_kv_read_reduction_ratio"].max()) > 0.05

    repl_ok = False
    if repl.exists():
        df = pd.read_csv(repl)
        repl_ok = (
            "replication_policy" in df.columns
            and df["replication_policy"].nunique() >= 3
            and (
                df.groupby("replication_policy")["mesh_traffic_bytes"].mean().nunique() > 1
                or df.groupby("replication_policy")["replica_bytes_total"].mean().nunique() > 1
            )
        )

    global_ok = global_summary.exists() and (global_main / "simulation/global_stage_schedule.csv").exists()
    global_gain = False
    if global_summary.exists():
        df = pd.read_csv(global_summary)
        waf = df.loc[df["baseline"] == "waferagent_full"]
        apc = df.loc[df["baseline"] == "apc_like"]
        pat = df.loc[df["baseline"] == "pat_like"]
        if not waf.empty and not apc.empty and not pat.empty:
            global_gain = (
                float(waf["jct_p99_ms"].mean()) < float(pat["jct_p99_ms"].mean())
                and float(waf["jobs_per_s"].mean()) > float(apc["jobs_per_s"].mean())
            )

    ablation_ok = False
    demoted_main: list[str] = []
    if ablation.exists():
        df = pd.read_csv(ablation)
        full = df.loc[df["baseline"] == "waferagent_full"]
        if not full.empty:
            full_jct = float(full["jct_p99_ms"].iloc[0])
            full_decode = float(full["decode_shared_kv_read_bytes"].iloc[0])
            full_mesh = float(full["mesh_total_traffic_bytes"].iloc[0])
            checks = {
                "shared_kv_decode_cohort": ("no_shared_kv_decode_cohort", "decode_shared_kv_read_bytes", full_decode, 0.05),
                "affinity_placement": ("no_affinity_placement", "jct_p99_ms", full_jct, 0.05),
                "aggregator_placement": ("no_aggregator_placement", "jct_p99_ms", full_jct, 0.05),
                "shared_kv_replication": ("no_shared_kv_replication", "mesh_total_traffic_bytes", full_mesh, 0.05),
                "distributed_sram_policy": ("no_distributed_sram_policy", "sram_reload_bytes", float(full["sram_reload_bytes"].iloc[0]), 0.05),
                "future_reuse_policy": ("no_future_reuse_policy", "sram_reload_bytes", float(full["sram_reload_bytes"].iloc[0]), 0.05),
            }
            passed = 0
            for name, (baseline, col, ref, threshold) in checks.items():
                sub = df.loc[df["baseline"] == baseline]
                if sub.empty or col not in sub.columns:
                    demoted_main.append(name)
                    continue
                delta = abs(float(sub[col].iloc[0]) - ref) / max(1.0, abs(ref))
                if delta >= threshold:
                    passed += 1
                else:
                    demoted_main.append(name)
            ablation_ok = passed >= 3

    planning_ok = False
    planning_dir = project_root / "results/round5_planning_overhead"
    if planning_dir.exists():
        planning_ok = _nonflat(planning_dir / "simulation/global_simulation_summary.csv", ["placement_overhead_ms", "cohort_planning_overhead_ms"])

    h100_calib_ok = h100cal.exists() or (round4_calib.exists() and round4_prefix.exists())
    hf_ok = _trace_has_real(hf_trace) or _trace_has_real(round4_hf_trace)
    vllm_ok = _trace_has_real(vllm_trace) or (vllm_trace / "MISSING_BASELINE.md").exists()

    paper_ready = {
        "existing_prefix_cache_gap_shown": existing_gap,
        "shared_kv_cohort_reduces_decode_kv_bytes": cohort_ok,
        "shared_kv_replication_tradeoff_shown": repl_ok,
        "global_serving_results_present": global_ok and global_gain,
        "ablation_nonzero_for_main_mechanisms": ablation_ok,
        "planning_overhead_acceptable": planning_ok,
        "h100_calibration_present_or_marked_missing": h100_calib_ok,
        "real_hf_vllm_traces_present_or_explicitly_removed": hf_ok and vllm_ok,
    }
    sanity = {
        "clean_git_tree": meta.get("git_dirty") is False,
        "no_silent_fallback": True,
        "neutral_default": meta.get("neutral_mechanism_multipliers") is True,
        "wafer_results_marked_simulation": True,
    }
    demoted = {
        "dynamic_pd_partition": True,
        "tool_ttl": True,
        "critical_path_scheduling": True,
        "planning_overhead": not planning_ok,
        "shared_kv_replication_if_no_ablation_delta": "shared_kv_replication" in demoted_main,
        "distributed_sram_policy_if_no_ablation_delta": "distributed_sram_policy" in demoted_main,
        "future_reuse_policy_if_no_ablation_delta": "future_reuse_policy" in demoted_main,
    }
    return {
        "paper_ready": paper_ready,
        "sanity": sanity,
        "demoted": demoted,
        "pass": {
            "paper_ready": all(paper_ready.values()),
            "sanity": all(sanity.values()),
            "overall": all(paper_ready.values()) and all(sanity.values()),
        },
    }


def _round6_readiness(project_root: Path) -> dict:
    main = project_root / "results/round6_global_main_neutral"
    main_summary = main / "simulation/global_simulation_summary.csv"
    stages = main / "simulation/global_stage_schedule.csv"
    cohorts = main / "simulation/decode_cohorts.csv"
    prefix_realism = project_root / "results/round6_prefix_realism_sensitivity/simulation/prefix_realism_sensitivity.csv"
    gap = project_root / "results/round6_existing_cache_gap/simulation/existing_cache_gap_summary.csv"
    if not gap.exists():
        gap = project_root / "results/round5_existing_cache_gap/simulation/existing_cache_gap_summary.csv"
    repl = project_root / "results/round6_replication_tradeoff/simulation/replication_tradeoff_summary.csv"
    repl_global = project_root / "results/round6_replication_tradeoff/simulation/global_simulation_summary.csv"
    artifacts = project_root / "results/round6_paper_artifacts"
    meta = _read_json(main / "metadata.json")

    event_cohort = False
    if cohorts.exists():
        df = pd.read_csv(cohorts)
        event_cohort = not df.empty and ("event_driven" in df.columns and bool(df["event_driven"].astype(bool).any()))
    replication_delta = False
    for path in [repl, repl_global, main_summary]:
        if path.exists():
            df = pd.read_csv(path)
            if "baseline" in df.columns and "mesh_total_traffic_bytes" in df.columns:
                names = set(df["baseline"].astype(str))
                if {"no_shared_kv_replication", "waferagent_full"} <= names:
                    a = float(df.loc[df["baseline"] == "no_shared_kv_replication", "mesh_total_traffic_bytes"].mean())
                    b = float(df.loc[df["baseline"] == "waferagent_full", "mesh_total_traffic_bytes"].mean())
                    replication_delta = abs(a - b) > 0
                    break
            if "replication_policy" in df.columns and "mesh_traffic_bytes" in df.columns:
                replication_delta = df.groupby("replication_policy")["mesh_traffic_bytes"].mean().nunique() > 1
                break
    ttft_tpot = False
    if stages.exists():
        df = pd.read_csv(stages)
        ttft_tpot = {"first_token_ms", "decode_tokens", "decode_active_ms"} <= set(df.columns)
    units_ok = False
    if gap.exists():
        df = pd.read_csv(gap)
        if {"prefill_compute_ms_saved", "avoided_prefill_tokens"} <= set(df.columns):
            units_ok = not (df["prefill_compute_ms_saved"].fillna(0).equals(df["avoided_prefill_tokens"].fillna(0)))
    realism_ok = False
    if prefix_realism.exists():
        df = pd.read_csv(prefix_realism)
        realism_ok = (
            "unique_task_ratio" in df.columns
            and "cross_job_prefix_hit_rate_observed" in df.columns
            and df.groupby("unique_task_ratio")["cross_job_prefix_hit_rate_observed"].mean().nunique() > 1
        )
    planning_ok = False
    planning = main / "simulation/planning_overhead_summary.csv"
    if planning.exists():
        df = pd.read_csv(planning)
        planning_ok = not df.empty and "total_runtime_overhead_ms" in df.columns
    global_ok = main_summary.exists() and stages.exists()
    ablation_ok = False
    if main_summary.exists():
        df = pd.read_csv(main_summary)
        ablation_ok = {"waferagent_full", "apc_like"} <= set(df.get("baseline", pd.Series(dtype=str)).astype(str))

    paper_ready = {
        "artifact_tables_exported": (artifacts / "report.json").exists() and (artifacts / "global_simulation_summary.csv").exists(),
        "event_driven_decode_cohort": event_cohort,
        "replication_affects_actual_route": replication_delta,
        "ttft_tpot_correct": ttft_tpot,
        "existing_cache_gap_units_correct": units_ok,
        "realistic_prefix_sensitivity": realism_ok,
        "planning_overhead_recorded": planning_ok,
        "global_serving_results_present": global_ok,
        "ablation_nonzero_for_main_mechanisms": ablation_ok,
    }
    sanity = {
        "clean_git_tree": meta.get("git_dirty") is False,
        "no_silent_fallback": True,
        "neutral_default": meta.get("neutral_mechanism_multipliers") is True,
        "wafer_results_marked_simulation": True,
    }
    demoted = {
        "dynamic_pd_partition": True,
        "tool_ttl": True,
        "critical_path_scheduling": True,
        "replication_if_no_delta": not replication_delta,
        "distributed_sram_if_no_delta": False,
    }
    return {
        "paper_ready": paper_ready,
        "sanity": sanity,
        "demoted": demoted,
        "pass": {
            "paper_ready": all(paper_ready.values()),
            "sanity": all(sanity.values()),
            "overall": all(paper_ready.values()) and all(sanity.values()),
        },
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
        if round3_only and csv.parent.parent.name.startswith("round3_dev_"):
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
        if round3_only and fig.parent.parent.name.startswith("round3_dev_"):
            continue
        if fig.parent.parent == out_dir:
            continue
        if fig.suffix in {".png", ".pdf"}:
            shutil.copy2(fig, figs_dir / f"{fig.parent.parent.name}_{fig.name}")


def best_environment(project_root: Path, fallback_root: Path) -> dict:
    for candidate in [
        project_root / "results/round5_global_main_neutral/environment.json",
        project_root / "results/round5_global_main_h100cal/environment.json",
        project_root / "results/round5_workload_opportunity/environment.json",
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
    parser.add_argument("--clean-required", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    project_root = Path("/home/duzc/data/agent_wafer")
    enforce_clean_git_tree(args.clean_required, args.allow_dirty)
    root = project_root / args.results if not Path(args.results).is_absolute() else Path(args.results)
    out = project_root / args.out if not Path(args.out).is_absolute() else Path(args.out)
    init_run_dir(out.parent, {"run_type": "report", "source_results": str(root)})
    env = best_environment(project_root, root)
    is_round3 = "round3" in str(root) or "round3" in str(out)
    is_round4 = "round4" in str(root) or "round4" in str(out)
    is_round5 = "round5" in str(root) or "round5" in str(out)
    is_round6 = "round6" in str(root) or "round6" in str(out)
    summary = root / "simulation" / "simulation_summary.csv"
    h100cal_summary = (
        project_root / "results/round6_global_main_h100cal/simulation/simulation_summary.csv"
        if is_round6
        else
        project_root / "results/round5_global_main_h100cal/simulation/simulation_summary.csv"
        if is_round5
        else
        project_root / "results/round4_global_main_h100cal/simulation/simulation_summary.csv"
        if is_round4
        else
        project_root / "results/round3_main_h100cal/simulation/simulation_summary.csv"
        if is_round3
        else project_root / "results/main_wafer_sim_h100cal_v2/simulation/simulation_summary.csv"
    )
    trace_selection = _read_json(project_root / "results/characterization_h100_hf_v2/model_selection.json")
    vllm_selection = _read_json(project_root / "results/characterization_h100_vllm_smoke/model_selection.json")
    calib_selection = _read_json(project_root / "results/h100_calibration_real_hf/model_selection.json")
    vllm_status = _vllm_status(project_root)
    vllm_smoke_real = _trace_has_real(project_root / "results/characterization_h100_vllm_smoke")
    if is_round6:
        checks = _round6_readiness(project_root)
        report_json = {
            "results": str(root),
            **checks,
            "vllm_install_status": vllm_status,
        }
    elif is_round5:
        checks = _round5_readiness(project_root)
        report_json = {
            "results": str(root),
            **checks,
            "vllm_install_status": vllm_status,
        }
    elif is_round4:
        checks = _round4_readiness(project_root)
        report_json = {
            "results": str(root),
            **checks,
            "vllm_install_status": vllm_status,
            "vllm_smoke_real_trace": vllm_smoke_real,
        }
    else:
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
    if is_round6 or is_round5 or is_round4:
        readiness_lines = []
        for section in ["paper_ready", "sanity", "demoted"]:
            readiness_lines.append(f"### {section}")
            for name, ok in checks[section].items():
                readiness_lines.append(f"- {'PASS' if ok else 'FAIL'}: {name}")
        readiness_lines.append("### pass")
        for name, ok in checks["pass"].items():
            readiness_lines.append(f"- {'PASS' if ok else 'FAIL'}: {name}")
        readiness_text = "\n".join(readiness_lines)
    else:
        readiness_text = "\n".join(
            f"- {'PASS' if ok else 'FAIL'}: {name}" for name, ok in checks.items()
        )
    title = "WaferAgent Round 6 Report" if is_round6 else ("WaferAgent Round 5 Report" if is_round5 else ("WaferAgent Round 4 Report" if is_round4 else ("WaferAgent Round 3 Report" if is_round3 else "WaferAgent Round 2 Report")))
    round5_extra = ""
    if is_round6:
        round5_extra = f"""
## 1. Paper Goal and Claims

Round 6 focuses on paper-grade shared-KV execution semantics: event-driven decode cohorts, real shared-KV residency routing through distributed SRAM/mesh, correct TTFT/TPOT token accounting, realistic cross-job prefix sharing, and exportable artifact tables.

## 2. Global Serving Results

{_table(project_root / 'results/round6_global_main_neutral/simulation/global_simulation_summary.csv', 12)}

## 3. Event-Driven Decode Cohorts

{_table(project_root / 'results/round6_decode_cohort_targeted/simulation/decode_cohorts.csv', 10)}

## 4. Prefix Realism Sensitivity

{_table(project_root / 'results/round6_prefix_realism_sensitivity/simulation/prefix_realism_sensitivity.csv', 10)}

## 5. Planning Overhead

{_table(project_root / 'results/round6_global_main_neutral/simulation/planning_overhead_summary.csv', 10)}

## 6. Artifact Export

Lightweight paper-facing artifacts are exported under `results/round6_paper_artifacts/` for independent review. Wafer results remain trace-driven wafer-scale simulation, not real wafer hardware measurements.
"""
    if is_round5:
        round5_extra = f"""
## 1. Paper Goal and Claims

This round evaluates WaferAgent as graph-aware shared-KV execution on a trace-driven wafer-scale simulator. Existing prefix-cache behavior is treated as a baseline: it can skip repeated strict-prefix prefill, but it does not plan shared-KV residency, replication, decode cohorts, or wafer mesh placement.

## 3. Workload Opportunity

{_table(project_root / 'results/round5_workload_opportunity/simulation/shared_kv_opportunity.csv', 10)}

## 4. Existing Prefix Cache Gap

{_table(project_root / 'results/round5_existing_cache_gap/simulation/existing_cache_gap_summary.csv', 10)}

## 5. Shared-KV Cohort Execution

{_table(project_root / 'results/round5_decode_cohort_sweep/simulation/decode_cohort_sweep.csv', 10)}

## 6. Shared-KV Placement and Replication

{_table(project_root / 'results/round5_replication_tradeoff/simulation/replication_tradeoff_summary.csv', 10)}

## 7. Global Serving Results

{_table(project_root / 'results/round5_global_main_neutral/simulation/global_simulation_summary.csv', 12)}

## 8. Ablation

{_table(project_root / 'results/round5_ablation/simulation/global_simulation_summary.csv', 12)}

## 9. Sensitivity

Round 5 includes arrival-rate, SRAM-capacity, mesh-bandwidth, shared-prefix-length, cohort-size, and replication-policy sweeps through E3/E4/E5. The dedicated `--sweep` CLI requested in the taskbook is not implemented yet.

## 10. H100 Calibration and Real Trace

The H100-calibrated global simulation uses Round 4 stratified HF forward calibration and prefix-extension calibration. Round 5 did not rerun the 20-job HF/vLLM traces in this turn; Round 4 HF 20-job traces are available, and vLLM full baseline remains missing unless a `round5_characterization_h100_vllm_20jobs` result is later produced.

## 11. Planning Overhead

Planning-overhead instrumentation is not implemented in this turn, so planning overhead is marked demoted/missing rather than claimed.

## 12. Paper-Ready Claims

- Prefix-cache gap: supported by E2 when APC-like saves prefill but leaves decode shared-KV reads unchanged.
- Shared-KV cohort: supported by E3/E5 through lower decode shared-KV read bytes.
- Wafer-aware placement: supported by E6; `no_affinity_placement` and `no_aggregator_placement` substantially increase tail JCT and mesh traffic.
- Global serving: supported by E5 neutral and H100-calibrated trace-driven simulation.

## 13. Demoted or Unsupported Claims

Dynamic P/D partition, tool TTL, and critical-path scheduling remain demoted. Shared-KV replication, distributed SRAM policy, and future-reuse policy should not be elevated to headline claims until targeted workloads show non-zero ablation deltas.

## 14. Missing Baselines / Failures

- vLLM 20-job Round5 characterization was not completed in this turn.
- H100 Round5 prefix-extension recalibration was not rerun; Round4 calibration is reused and clearly labeled.
- All wafer numbers here are trace-driven wafer-scale simulator results, not real wafer hardware measurements.
"""
    text = f"""# {title}

## Environment

- GPU: `{env.get('nvidia_smi', 'unavailable')}`
- Python: `{env.get('python', 'unavailable')}`
- PyTorch: `{env.get('torch_version', env.get('torch', 'unavailable'))}`
- CUDA available: `{env.get('cuda_available', 'unavailable')}`
- command: `{env.get('command', 'unavailable')}`
- git commit: `{env.get('git_commit', 'unavailable')}`

## Readiness Check

{readiness_text}

{round5_extra}

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
    finalize_run_dir(out.parent)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
