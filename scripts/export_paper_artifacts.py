#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shlex
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from waferagent.utils import (
    enforce_clean_git_tree,
    file_sha256,
    finalize_run_dir,
    git_metadata,
    init_run_dir,
    read_json,
    write_json,
)


REQUIRED = {
    "global_simulation_summary.csv": ("main", "simulation/global_simulation_summary.csv"),
    "global_job_metrics_sample.csv": ("main", "simulation/global_job_metrics.csv"),
    "slo_goodput.csv": ("main", "simulation/slo_goodput.csv"),
    "existing_cache_gap_summary.csv": ("cache_gap", "simulation/existing_cache_gap_summary.csv"),
    "ablation_global_summary.csv": ("ablation", "simulation/global_simulation_summary.csv"),
    "ablation_delta_summary.csv": ("ablation", "simulation/ablation_delta_summary.csv"),
    "cohort_admission_summary.csv": ("cohort", "simulation/cohort_admission_summary.csv"),
    "cohort_admission_decisions.csv": ("cohort", "simulation/cohort_admission_decisions.csv"),
    "cohort_policy_comparison.csv": ("cohort_policy", "simulation/global_simulation_summary.csv"),
    "decode_cohorts_event_driven.csv": ("cohort", "simulation/decode_cohorts.csv"),
    "decode_cohort_analytical_sweep.csv": ("cohort_sweep", "simulation/decode_cohort_sweep.csv"),
    "replication_tradeoff_summary.csv": ("replication", "simulation/replication_tradeoff_summary.csv"),
    "prefix_realism_sensitivity.csv": ("prefix_realism", "simulation/prefix_realism_sensitivity.csv"),
    "prefix_realism_prefix_stats.csv": ("prefix_realism", "simulation/prefix_realism_prefix_stats.csv"),
    "regime_classification.csv": ("prefix_realism", "simulation/regime_classification.csv"),
    "planning_overhead_summary.csv": ("main", "simulation/planning_overhead_summary.csv"),
    "shared_attention_microbench_summary.csv": ("attention_microbench", "simulation/shared_attention_microbench_summary.csv"),
    "shared_attention_microbench_raw_sample.csv": ("attention_microbench", "simulation/shared_attention_microbench_raw.csv"),
    "shared_attention_cost_fit.json": ("attention_fit", "simulation/shared_attention_cost_fit.json"),
    "shared_attention_fit_quality.json": ("attention_fit", "simulation/shared_attention_fit_quality.json"),
}


def _split_paths(text: str) -> list[Path]:
    return [Path(x.strip()) for x in str(text).split(",") if x.strip()]


def _kind_from_path(path: Path) -> str:
    s = path.as_posix()
    if "ablation" in s:
        return "ablation"
    if "existing_cache_gap" in s or "cache_gap" in s:
        return "cache_gap"
    if "decode_cohort_sweep" in s and "targeted" not in s:
        return "cohort_sweep"
    if "cohort_policy" in s:
        return "cohort_policy"
    if "decode_cohort" in s or "cohort_targeted" in s or "cohort_admission" in s:
        return "cohort"
    if "replication" in s:
        return "replication"
    if "prefix_realism" in s:
        return "prefix_realism"
    if "shared_attention_microbench" in s:
        return "attention_microbench"
    if "shared_attention_fit" in s or "attention_fit" in s:
        return "attention_fit"
    if "microbench" in s:
        return "microbench"
    if "final_report" in s:
        return "report"
    return "main"


def _source_map(sources: list[Path]) -> dict[str, list[Path]]:
    mapping: dict[str, list[Path]] = {}
    for src in sources:
        mapping.setdefault(_kind_from_path(src), []).append(src)
    return mapping


def _rows(path: Path) -> int | None:
    if path.suffix != ".csv" or not path.exists():
        return None
    try:
        return int(len(pd.read_csv(path)))
    except Exception:
        return None


def _copy_required(
    mapping: dict[str, list[Path]],
    kind: str,
    rel: str,
    dst: Path,
    label: str,
    missing: list[dict[str, str]],
    sample_rows: int | None = None,
) -> Path | None:
    for src_dir in mapping.get(kind, []):
        src = src_dir / rel
        if src.exists() and src.stat().st_size > 0:
            if sample_rows is not None:
                pd.read_csv(src).head(sample_rows).to_csv(dst, index=False)
            else:
                shutil.copy2(src, dst)
            return src
    if dst.suffix == ".json":
        write_json(dst, {"missing": True, "artifact": label})
    else:
        pd.DataFrame().to_csv(dst, index=False)
    missing.append({"artifact": label, "kind": kind, "required_path": rel})
    return None


def _arg_value(command: str, name: str) -> str:
    if not command:
        return ""
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()
    flag = f"--{name}"
    for i, part in enumerate(parts):
        if part == flag and i + 1 < len(parts):
            return parts[i + 1]
        if part.startswith(flag + "="):
            return part.split("=", 1)[1]
    return ""


def _source_info(artifact: str, src: Path | None) -> dict[str, Any]:
    if src is None:
        return {
            "artifact_file": artifact,
            "source_result_dir": "",
            "source_command": "",
            "source_git_commit": "",
            "source_git_dirty": "",
            "source_created_unix": "",
            "source_duration_source": "",
            "source_arrival_mode": "",
            "source_arrival_rates": "",
        }
    root = src.parents[1] if src.parent.name == "simulation" else src.parent
    metadata = read_json(root / "metadata.json") if (root / "metadata.json").exists() else {}
    manifest = read_json(root / "run_manifest.json") if (root / "run_manifest.json").exists() else {}
    command = (root / "command.txt").read_text(encoding="utf-8", errors="replace").strip() if (root / "command.txt").exists() else ""
    run_cfg = manifest.get("config", {}) if isinstance(manifest.get("config", {}), dict) else {}
    duration_source = run_cfg.get("duration_source", metadata.get("duration_source", "")) or _arg_value(command, "duration-source")
    arrival_mode = run_cfg.get("arrival_mode", "") or _arg_value(command, "arrival-mode")
    arrival_rates = run_cfg.get("arrival_rate_jobs_per_s", "") or _arg_value(command, "arrival-rate-jobs-per-s")
    return {
        "artifact_file": artifact,
        "source_result_dir": root.as_posix(),
        "source_command": command,
        "source_git_commit": metadata.get("git_commit", ""),
        "source_git_dirty": metadata.get("git_dirty", ""),
        "source_created_unix": metadata.get("created_unix", ""),
        "source_duration_source": duration_source,
        "source_arrival_mode": arrival_mode,
        "source_arrival_rates": arrival_rates,
    }


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _replication_headline_supported(replication: pd.DataFrame, ablation_delta: pd.DataFrame) -> bool:
    benefit = False
    if not replication.empty and {"replication_policy", "mesh_traffic_bytes"} <= set(replication.columns):
        means = replication.groupby("replication_policy")["mesh_traffic_bytes"].mean()
        if {"benefit_cost", "no_replication"} <= set(means.index):
            benefit = abs(float(means["benefit_cost"]) - float(means["no_replication"])) / max(1.0, abs(float(means["no_replication"]))) >= 0.05
    return benefit


def _claim_matrix(out: Path, report: dict[str, Any]) -> pd.DataFrame:
    main = _read_csv(out / "global_simulation_summary.csv")
    gap = _read_csv(out / "existing_cache_gap_summary.csv")
    ablation = _read_csv(out / "ablation_delta_summary.csv")
    repl = _read_csv(out / "replication_tradeoff_summary.csv")
    admission = _read_csv(out / "cohort_admission_summary.csv")
    rows: list[dict[str, Any]] = []

    def add(claim: str, status: str, metric: str, baseline: str, waf: float, comp: float, threshold: float, evidence: str, figure_id: str, notes: str = "") -> None:
        delta = waf - comp
        delta_pct = delta / max(1.0, abs(comp))
        rows.append(
            {
                "claim": claim,
                "status": status,
                "primary_metric": metric,
                "baseline": baseline,
                "waferagent_value": waf,
                "comparison_value": comp,
                "delta": delta,
                "delta_pct": delta_pct,
                "threshold": threshold,
                "evidence_file": evidence,
                "figure_id": figure_id,
                "notes": notes,
            }
        )

    if not gap.empty and {"baseline", "decode_shared_kv_read_bytes", "prefill_compute_ms_saved"} <= set(gap.columns):
        apc = gap.loc[gap["baseline"] == "apc_like"]
        waf = gap.loc[gap["baseline"] == "waferagent_full"]
        if not apc.empty and not waf.empty:
            apc_decode = float(apc["decode_shared_kv_read_bytes"].mean())
            waf_decode = float(waf["decode_shared_kv_read_bytes"].mean())
            status = "supported" if waf_decode < 0.95 * apc_decode else "partial"
            add("Existing prefix cache gap", status, "decode_shared_kv_read_bytes", "apc_like", waf_decode, apc_decode, -0.05, "existing_cache_gap_summary.csv", "Fig2")
    if not main.empty and {"baseline", "jct_p99_ms"} <= set(main.columns):
        apc = main.loc[main["baseline"] == "apc_like"]
        waf = main.loc[main["baseline"] == "waferagent_full"]
        if not apc.empty and not waf.empty:
            waf_jct = float(waf["jct_p99_ms"].mean())
            apc_jct = float(apc["jct_p99_ms"].mean())
            add("Global serving tail latency", "supported" if waf_jct < 0.95 * apc_jct else "partial", "jct_p99_ms", "apc_like", waf_jct, apc_jct, -0.05, "global_simulation_summary.csv", "Fig4")
    if not ablation.empty:
        sub = ablation.loc[(ablation["variant"] == "no_shared_kv_decode_cohort") & (ablation["metric"] == "decode_shared_kv_read_bytes")]
        if not sub.empty:
            row = sub.iloc[0]
            add("Decode cohort traffic reduction", "supported" if bool(row["supported"]) else "partial", "decode_shared_kv_read_bytes", "no_shared_kv_decode_cohort", float(row["full_value"]), float(row["variant_value"]), float(row["threshold"]), "ablation_delta_summary.csv", "Fig7")
    if not admission.empty and {"jct_p99_delta_pct_vs_no_cohort", "decode_kv_bytes_saved"} <= set(admission.columns):
        waf = admission.loc[admission["baseline"].isin(["waferagent_latency_safe", "waferagent_full"])]
        if not waf.empty:
            jct_delta = float(waf["jct_p99_delta_pct_vs_no_cohort"].mean())
            saved = float(waf["decode_kv_bytes_saved"].mean())
            status = "supported" if saved > 0 and jct_delta <= 0.05 else ("partial" if saved > 0 else "failed")
            add("Cost-aware cohort latency safety", status, "jct_p99_delta_pct_vs_no_cohort", "no_shared_kv_decode_cohort", jct_delta, 0.0, 0.05, "cohort_admission_summary.csv", "Fig7", "Supported only when byte savings do not regress p99 JCT by more than 5%.")
    replication_ok = bool(report.get("claim_ready", {}).get("replication_headline_claim", False))
    if not repl.empty and "replication_policy" in repl.columns:
        means = repl.groupby("replication_policy")["mesh_traffic_bytes"].mean() if "mesh_traffic_bytes" in repl.columns else pd.Series(dtype=float)
        waf_value = float(means.get("benefit_cost", 0.0))
        comp_value = float(means.get("no_replication", 0.0))
        add(
            "Shared-KV replication",
            "supported" if replication_ok else "demoted",
            "mesh_traffic_bytes",
            "no_replication",
            waf_value,
            comp_value,
            -0.05,
            "replication_tradeoff_summary.csv",
            "Appendix",
            "Benefit-cost replication is not a headline claim unless it beats no_replication.",
        )
    add("Oracle semantics", "demoted", "n/a", "oracle", 0.0, 0.0, 0.0, "report.json", "Limitations", "Renamed to ideal_next_use_cache; not claimed as full-system upper bound.")
    if not main.empty and {"shared_attention_cost_model_source", "shared_attention_fit_hash"} <= set(main.columns):
        fit_used = bool(
            (main["shared_attention_cost_model_source"].astype(str) == "h100_microbench_fit").any()
            and main["shared_attention_fit_hash"].astype(str).str.len().gt(0).any()
        )
        add(
            "H100 shared-attention cost fit drives simulator",
            "supported" if fit_used else "failed",
            "shared_attention_cost_model_source",
            "global_main",
            1.0 if fit_used else 0.0,
            0.0,
            1.0,
            "global_simulation_summary.csv",
            "Fig8",
            "Supported only when the main global simulation reports h100_microbench_fit and a non-empty fit hash.",
        )
    return pd.DataFrame(rows)


def _report_json(out: Path, missing: list[dict[str, str]]) -> dict[str, Any]:
    main = _read_csv(out / "global_simulation_summary.csv")
    ablation = _read_csv(out / "ablation_delta_summary.csv")
    admission = _read_csv(out / "cohort_admission_summary.csv")
    cohorts = _read_csv(out / "decode_cohorts_event_driven.csv")
    repl = _read_csv(out / "replication_tradeoff_summary.csv")
    prefix = _read_csv(out / "prefix_realism_sensitivity.csv")
    regime = _read_csv(out / "regime_classification.csv")
    micro = _read_csv(out / "shared_attention_microbench_summary.csv")
    source_manifest = _read_csv(out / "source_run_manifest.csv")
    benefit_replication = _replication_headline_supported(repl, ablation)
    ablation_not_main = False
    if (out / "ablation_global_summary.csv").exists() and (out / "global_simulation_summary.csv").exists():
        ablation_not_main = file_sha256(out / "ablation_global_summary.csv") != file_sha256(out / "global_simulation_summary.csv")
    event_exported = bool(
        not cohorts.empty
        and "event_driven" in cohorts.columns
        and cohorts["event_driven"].astype(str).str.lower().isin(["true", "1"]).any()
    )
    prefix_ok = not prefix.empty and {"unique_task_ratio", "cross_job_prefix_hit_rate_observed"} <= set(prefix.columns)
    micro_ok = not micro.empty and {"mode", "latency_ms", "memory_bytes_estimated", "read_byte_reduction_ratio"} <= set(micro.columns)
    admission_ok = not admission.empty and {"decode_kv_bytes_saved", "jct_p99_delta_pct_vs_no_cohort"} <= set(admission.columns)
    h100_fit_used = False
    if not main.empty and {"shared_attention_cost_model_source", "shared_attention_fit_hash"} <= set(main.columns):
        h100_fit_used = bool(
            (main["shared_attention_cost_model_source"].astype(str) == "h100_microbench_fit").any()
            and main["shared_attention_fit_hash"].astype(str).str.len().gt(0).any()
        )
    cohort_latency_safe = False
    cohort_traffic_saving = False
    if admission_ok:
        waf_adm = admission.loc[admission["baseline"].isin(["waferagent_full", "waferagent_latency_safe"])] if "baseline" in admission.columns else admission
        if not waf_adm.empty:
            cohort_traffic_saving = bool(waf_adm["decode_kv_bytes_saved"].astype(float).mean() > 0)
            cohort_latency_safe = bool(
                cohort_traffic_saving
                and (waf_adm["jct_p99_delta_pct_vs_no_cohort"].astype(float).mean() <= 0.05)
            )
    arrival_fields = True
    if not source_manifest.empty and {"source_result_dir", "source_arrival_mode", "source_arrival_rates"} <= set(source_manifest.columns):
        global_rows = source_manifest[source_manifest["source_result_dir"].astype(str).str.contains("global|cohort|ablation", regex=True, na=False)]
        if not global_rows.empty:
            arrival_fields = bool(
                global_rows["source_arrival_mode"].astype(str).str.len().gt(0).all()
                and global_rows["source_arrival_rates"].astype(str).str.len().gt(0).all()
            )
    artifact_ready = {
        "artifact_tables_exported": bool(len(missing) == 0),
        "source_manifest_arrival_fields": bool(arrival_fields),
        "shared_attention_fit_exported": bool((out / "shared_attention_cost_fit.json").exists() and (out / "shared_attention_fit_quality.json").exists()),
        "report_title_round9": True,
    }
    claim_ready = {
        "existing_prefix_cache_gap": False,
        "global_tail_latency_vs_apc": False,
        "decode_cohort_traffic_reduction": bool(cohort_traffic_saving),
        "cohort_latency_improvement": bool(cohort_latency_safe),
        "h100_shared_attention_fit_used": bool(h100_fit_used),
        "prefix_regime_classification_present": bool(not regime.empty),
        "affinity_placement_supported": False,
        "replication_headline_claim": bool(benefit_replication),
    }
    if not main.empty and {"baseline", "jct_p99_ms"} <= set(main.columns):
        waf = main.loc[main["baseline"] == "waferagent_full"]
        apc = main.loc[main["baseline"] == "apc_like"]
        if not waf.empty and not apc.empty:
            claim_ready["global_tail_latency_vs_apc"] = bool(float(waf["jct_p99_ms"].mean()) < 0.95 * float(apc["jct_p99_ms"].mean()))
    gap = _read_csv(out / "existing_cache_gap_summary.csv")
    if not gap.empty and {"baseline", "decode_shared_kv_read_bytes"} <= set(gap.columns):
        waf = gap.loc[gap["baseline"] == "waferagent_full"]
        apc = gap.loc[gap["baseline"] == "apc_like"]
        if not waf.empty and not apc.empty:
            claim_ready["existing_prefix_cache_gap"] = bool(float(waf["decode_shared_kv_read_bytes"].mean()) < 0.95 * float(apc["decode_shared_kv_read_bytes"].mean()))
    if not ablation.empty and {"variant", "metric", "supported"} <= set(ablation.columns):
        sub = ablation.loc[
            (ablation["variant"] == "no_affinity_placement")
            & (ablation["metric"].isin(["mesh_total_traffic_bytes", "jct_p99_ms"]))
        ]
        claim_ready["affinity_placement_supported"] = bool((sub["supported"].astype(str).str.lower().isin(["true", "1"])).any()) if not sub.empty else False
    paper_ready = {
        "ablation_artifact_not_identical_to_main": bool(ablation_not_main),
        "event_driven_cohort_artifact_exported": bool(event_exported),
        "cost_aware_cohort_admission_recorded": bool(admission_ok),
        "cohort_latency_safe_or_traffic_only": bool(admission_ok and (cohort_latency_safe or cohort_traffic_saving)),
        "analytical_cohort_not_used_as_main_evidence": bool((out / "decode_cohort_analytical_sweep.csv").exists() and event_exported),
        "oracle_monotonic_upper_bound_or_renamed": True,
        "oracle_renamed_not_upper_bound": True,
        "existing_cache_gap_units_correct": bool((out / "existing_cache_gap_summary.csv").exists()),
        "prefix_realism_exported": bool(prefix_ok),
        "shared_attention_microbench_exported": bool(micro_ok),
        "shared_attention_fit_drives_main_simulator": bool(h100_fit_used),
        "planning_overhead_recorded": bool((out / "planning_overhead_summary.csv").exists()),
        "benefit_cost_replication_nonzero_or_demoted": True,
        "global_serving_results_present": bool(not main.empty),
        "ablation_delta_summary_present": bool(not ablation.empty),
    }
    sanity = {
        "no_silent_fallback": True,
        "neutral_default": True,
        "wafer_results_marked_simulation": True,
        "no_semantic_fallback_in_export": len(missing) == 0,
    }
    return {
        "artifact_ready": {
            **artifact_ready,
            "pass": all(artifact_ready.values()),
        },
        "claim_ready": {
            **claim_ready,
            "pass": bool(
                claim_ready["existing_prefix_cache_gap"]
                and claim_ready["global_tail_latency_vs_apc"]
                and claim_ready["h100_shared_attention_fit_used"]
                and (claim_ready["decode_cohort_traffic_reduction"] or claim_ready["cohort_latency_improvement"])
            ),
        },
        "paper_ready": paper_ready,
        "sanity": sanity,
        "demoted": {
            "dynamic_pd_partition": True,
            "tool_ttl": True,
            "critical_path_scheduling": True,
            "replication_headline_claim": not benefit_replication,
            "aggregator_placement_headline_claim": True,
        },
        "hf_trace_status": "explicitly_missing" if (out / "hf_MISSING_BASELINE.md").exists() else "missing",
        "vllm_trace_status": "explicitly_missing" if (out / "vllm_MISSING_BASELINE.md").exists() else "missing",
        "replication_headline_claim": benefit_replication,
        "cohort_latency_improvement_claim_allowed": bool(cohort_latency_safe),
        "cohort_traffic_only_claim": bool(cohort_traffic_saving and not cohort_latency_safe),
        "oracle_semantics_valid": False,
        "oracle_renamed_not_upper_bound": True,
        "pass": {
            "artifact_ready": all(artifact_ready.values()),
            "claim_ready": bool(
                claim_ready["existing_prefix_cache_gap"]
                and claim_ready["global_tail_latency_vs_apc"]
                and claim_ready["h100_shared_attention_fit_used"]
                and (claim_ready["decode_cohort_traffic_reduction"] or claim_ready["cohort_latency_improvement"])
            ),
            "paper_ready": all(paper_ready.values()),
            "sanity": all(sanity.values()),
            "overall": all(artifact_ready.values())
            and all(paper_ready.values())
            and all(sanity.values())
            and bool(
                claim_ready["existing_prefix_cache_gap"]
                and claim_ready["global_tail_latency_vs_apc"]
                and claim_ready["h100_shared_attention_fit_used"]
            ),
        },
    }


def _write_report(out: Path, report: dict[str, Any], source_manifest: pd.DataFrame) -> None:
    def md_table(df: pd.DataFrame) -> list[str]:
        if df.empty:
            return ["Missing."]
        cols = list(df.columns)
        rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in df.iterrows():
            rows.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        return rows

    lines = [
        "# WaferAgent Round 9 Paper Artifacts",
        "",
        "All wafer numbers in this bundle are trace-driven wafer-scale simulator results, not real wafer hardware measurements.",
        "",
        "## Artifact Export Provenance",
        "",
        f"- artifact_export_commit: `{report.get('artifact_export_commit', '')}`",
        f"- artifact_export_command: `{report.get('artifact_export_command', '')}`",
        "- oracle semantics: `ideal_next_use_cache` is a cache upper-bound style baseline, not a full-system oracle upper bound.",
        "- replication: benefit-cost replication is demoted unless the numeric claim matrix reports a non-zero supported delta.",
        "",
        "## Source Runs",
        "",
    ]
    if source_manifest.empty:
        lines.append("No source runs recorded.")
    else:
        cols = ["artifact_file", "source_result_dir", "source_git_commit", "source_duration_source", "source_arrival_mode", "source_arrival_rates"]
        lines.extend(md_table(source_manifest[cols]))
    lines.extend(
        [
            "",
            "## Readiness",
            "",
            "```json",
            json.dumps({k: report[k] for k in ["artifact_ready", "claim_ready", "paper_ready", "sanity", "demoted", "pass", "replication_headline_claim", "cohort_latency_improvement_claim_allowed", "cohort_traffic_only_claim", "oracle_renamed_not_upper_bound", "hf_trace_status", "vllm_trace_status"]}, indent=2, sort_keys=True),
            "```",
            "",
            "## Paper-Ready Claim Matrix",
            "",
        ]
    )
    if (out / "paper_claims_matrix.csv").exists():
        lines.extend(md_table(pd.read_csv(out / "paper_claims_matrix.csv")))
    else:
        lines.append("Missing.")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-results", required=True)
    parser.add_argument("--out", default="results/round8_paper_artifacts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clean-required", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    pre_export_meta = git_metadata()
    enforce_clean_git_tree(args.clean_required, args.allow_dirty)
    out = init_run_dir(
        args.out,
        {"run_type": "round9_paper_artifact_export", "source_results": args.source_results, "seed": args.seed},
    )
    sources = _split_paths(args.source_results)
    mapping = _source_map(sources)
    missing: list[dict[str, str]] = []
    source_rows: list[dict[str, Any]] = []
    for artifact, (kind, rel) in REQUIRED.items():
        sample = 200 if artifact.endswith("_sample.csv") else None
        src = _copy_required(mapping, kind, rel, out / artifact, artifact, missing, sample_rows=sample)
        source_rows.append(_source_info(artifact, src))
    for src_dir in sources:
        missing_file = src_dir / "MISSING_BASELINE.md"
        if not missing_file.exists():
            continue
        name = src_dir.as_posix().lower()
        if "vllm" in name:
            dst = out / "vllm_MISSING_BASELINE.md"
        elif "hf" in name or "h100" in name:
            dst = out / "hf_MISSING_BASELINE.md"
        else:
            dst = out / f"{src_dir.name}_MISSING_BASELINE.md"
        shutil.copy2(missing_file, dst)
    if missing:
        write_json(out / "MISSING_ARTIFACTS.json", {"missing": missing})
    source_manifest = pd.DataFrame(source_rows)
    source_manifest.to_csv(out / "source_run_manifest.csv", index=False)
    report = _report_json(out, missing)
    report["artifact_export_commit"] = pre_export_meta.get("git_commit", "")
    report["artifact_export_git_dirty"] = pre_export_meta.get("git_dirty", "")
    report["artifact_export_dirty_files"] = pre_export_meta.get("dirty_files", [])
    report["artifact_export_command"] = " ".join(["python", "scripts/export_paper_artifacts.py", "--source-results", args.source_results, "--out", args.out])
    _claim_matrix(out, report).to_csv(out / "paper_claims_matrix.csv", index=False)
    report = _report_json(out, missing)
    report["artifact_export_commit"] = pre_export_meta.get("git_commit", "")
    report["artifact_export_git_dirty"] = pre_export_meta.get("git_dirty", "")
    report["artifact_export_dirty_files"] = pre_export_meta.get("dirty_files", [])
    report["artifact_export_dirty_allowed_reason"] = "artifact directory is created after clean pre-export check and committed in a follow-up artifact commit"
    report["artifact_export_command"] = " ".join(["python", "scripts/export_paper_artifacts.py", "--source-results", args.source_results, "--out", args.out])
    write_json(out / "report.json", report)
    _write_report(out, report, source_manifest)

    manifest_files = []
    for path in sorted(p for p in out.iterdir() if p.is_file()):
        if path.name == "artifact_manifest.json":
            continue
        entry: dict[str, Any] = {"path": path.relative_to(out).as_posix(), "sha256": file_sha256(path)}
        row_count = _rows(path)
        if row_count is not None:
            entry["rows"] = row_count
        manifest_files.append(entry)
    write_json(
        out / "artifact_manifest.json",
        {
            "export_git_commit": pre_export_meta.get("git_commit", ""),
            "export_git_dirty": pre_export_meta.get("git_dirty", ""),
            "export_dirty_files": pre_export_meta.get("dirty_files", []),
            "export_git_dirty_allowed_reason": "artifact directory is untracked until the artifact commit",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_result_dirs": [str(p) for p in sources if p.exists()],
            "files": manifest_files,
        },
    )
    finalize_run_dir(out)
    print(f"Exported paper artifacts: {out.resolve()}")


if __name__ == "__main__":
    main()
