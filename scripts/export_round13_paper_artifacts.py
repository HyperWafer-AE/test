#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shlex
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from waferagent.utils import enforce_clean_git_tree, file_sha256, finalize_run_dir, git_metadata, init_run_dir, read_json, write_json


def _split(text: str) -> list[Path]:
    return [Path(x.strip()) for x in str(text).split(",") if x.strip()]


def _find(sources: list[Path], rel: str, contains: str | None = None) -> Path | None:
    for src in sources:
        if contains and contains not in src.as_posix():
            continue
        p = src / rel
        if p.exists() and p.stat().st_size > 0:
            return p
    for src in sources:
        p = src / rel
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def _copy_csv(sources: list[Path], rel: str, dst: Path, contains: str | None = None) -> pd.DataFrame:
    src = _find(sources, rel, contains)
    if src:
        shutil.copy2(src, dst)
        return pd.read_csv(dst)
    pd.DataFrame().to_csv(dst, index=False)
    return pd.DataFrame()


def _copy_json(sources: list[Path], rel: str, dst: Path, contains: str | None = None) -> dict:
    src = _find(sources, rel, contains)
    if src:
        shutil.copy2(src, dst)
        return read_json(dst)
    return {}


def _cmd_value(command: str, flag: str) -> str:
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()
    if flag in parts:
        idx = parts.index(flag)
        if idx + 1 < len(parts):
            return parts[idx + 1]
    prefix = flag + "="
    for part in parts:
        if part.startswith(prefix):
            return part[len(prefix):]
    return ""


def _copy_missing(sources: list[Path], out: Path) -> None:
    for label in ["hf", "vllm"]:
        for src in sources:
            if f"h100_{label}" in src.as_posix():
                missing = src / "MISSING_BASELINE.md"
                if missing.exists():
                    name = f"{label}_10jobs_MISSING_BASELINE.md" if "10jobs" in src.as_posix() else f"{label}_MISSING_BASELINE.md"
                    shutil.copy2(missing, out / name)


def _source_manifest(sources: list[Path]) -> pd.DataFrame:
    rows = []
    for src in sources:
        meta = read_json(src / "metadata.json") if (src / "metadata.json").exists() else {}
        cmd = (src / "command.txt").read_text(encoding="utf-8", errors="replace").strip() if (src / "command.txt").exists() else ""
        rows.append(
            {
                "source_result_dir": src.as_posix(),
                "source_git_commit": meta.get("git_commit", ""),
                "source_git_dirty": meta.get("git_dirty", ""),
                "source_command": cmd,
                "source_arrival_mode": _cmd_value(cmd, "--arrival-mode"),
                "source_arrival_rates": _cmd_value(cmd, "--arrival-rate-jobs-per-s"),
                "source_duration_source": _cmd_value(cmd, "--duration-source"),
                "source_accounting_mode": _cmd_value(cmd, "--shared-attention-accounting"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-results", required=True)
    parser.add_argument("--out", default="results/round13_paper_artifacts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clean-required", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    pre_meta = git_metadata()
    enforce_clean_git_tree(args.clean_required, args.allow_dirty)
    out = init_run_dir(args.out, {"run_type": "round13_paper_artifacts", "source_results": args.source_results, "seed": args.seed})
    sources = _split(args.source_results)

    randomized_summary = _copy_csv(sources, "simulation/randomized_regime_summary.csv", out / "randomized_regime_summary.csv", "randomized_regime")
    randomized_oracle = _copy_csv(sources, "simulation/randomized_policy_oracle_labels.csv", out / "randomized_policy_oracle_labels.csv", "randomized_regime")
    randomized_eval = _copy_csv(sources, "simulation/randomized_policy_prediction_eval.csv", out / "randomized_policy_prediction_eval.csv", "randomized_regime")
    randomized_class = _copy_csv(sources, "simulation/randomized_regime_classification.csv", out / "randomized_regime_classification.csv", "randomized_regime")
    validation = _copy_csv(sources, "simulation/controlled_workload_validation.csv", out / "controlled_workload_validation.csv", "randomized_regime")
    validation_nodes = _copy_csv(sources, "simulation/controlled_workload_validation_nodes.csv", out / "controlled_workload_validation_nodes.csv", "randomized_regime")
    policy_assignments = _copy_csv(sources, "simulation/policy_assignments.csv", out / "policy_assignments.csv", "randomized_regime")
    policy_stage = _copy_csv(sources, "simulation/policy_effective_stage_map.csv", out / "policy_effective_stage_map.csv", "randomized_regime")
    policy_summary = _copy_csv(sources, "simulation/policy_summary.csv", out / "policy_summary.csv", "randomized_regime")
    fair_summary = _copy_csv(sources, "simulation/fair_baseline_summary.csv", out / "fair_baseline_summary.csv", "randomized_regime")
    fair_delta = _copy_csv(sources, "simulation/fair_baseline_delta.csv", out / "fair_baseline_delta.csv", "randomized_regime")
    mechanism_summary = _copy_csv(sources, "simulation/mechanism_attribution_summary.csv", out / "mechanism_attribution_summary.csv", "randomized_regime")
    mechanism_delta = _copy_csv(sources, "simulation/mechanism_attribution_delta.csv", out / "mechanism_attribution_delta.csv", "randomized_regime")
    semantics_detail = _copy_csv(sources, "simulation/adaptive_stage_semantics.csv", out / "adaptive_stage_semantics.csv", "adaptive_semantics")
    semantics_summary = _copy_csv(sources, "simulation/adaptive_semantics_summary.csv", out / "adaptive_semantics_summary.csv", "adaptive_semantics")
    v2_eval = _copy_csv(sources, "simulation/policy_selector_v2_eval.csv", out / "policy_selector_v2_eval.csv", "policy_selector_v2_eval")
    v2_decisions = _copy_csv(sources, "simulation/policy_selector_v2_decisions.csv", out / "policy_selector_v2_decisions.csv", "policy_selector_v2_eval")
    v2_importance = _copy_csv(sources, "simulation/policy_selector_v2_feature_importance.csv", out / "policy_selector_v2_feature_importance.csv", "policy_selector_v2")
    model = _find(sources, "simulation/policy_selector_v2_model.json", "policy_selector_v2")
    if model:
        shutil.copy2(model, out / "policy_selector_v2_model.json")
    # Carry forward paper-facing stable evidence from Round12.
    for filename in [
        "existing_cache_gap_summary.csv",
        "global_simulation_summary.csv",
        "shared_attention_fit_quality.json",
        "shared_attention_fit_validation.csv",
        "hf_real_trace_characterization.csv",
        "vllm_real_trace_characterization.csv",
        "hf_trace_completion_status.json",
        "vllm_trace_completion_status.json",
    ]:
        src = _find(sources, filename, "round12_paper_artifacts")
        if src:
            shutil.copy2(src, out / filename)
    _copy_missing(sources, out)

    controlled_pass = bool(not validation.empty and "pass" in validation.columns and validation["pass"].astype(bool).all())
    semantics_pass = bool(not semantics_summary.empty and "pass" in semantics_summary.columns and semantics_summary["pass"].astype(bool).all())
    randomized_pass = False
    if not randomized_eval.empty and {"split", "pass"}.issubset(randomized_eval.columns):
        test = randomized_eval[randomized_eval["split"] == "test"]
        randomized_pass = bool(not test.empty and test["pass"].astype(bool).all())
    unexplained_ok = False
    if not mechanism_summary.empty and "unexplained_pct" in mechanism_summary.columns:
        beneficial = mechanism_summary[pd.to_numeric(mechanism_summary.get("jct_p99_delta_vs_apc", pd.Series(dtype=float)), errors="coerce") >= 0.05]
        if beneficial.empty:
            unexplained_ok = False
        else:
            unexplained_ok = bool((pd.to_numeric(beneficial["unexplained_pct"], errors="coerce") <= 20.0).mean() >= 0.90)
    real_scope_consistent = bool((out / "hf_trace_completion_status.json").exists() or (out / "vllm_trace_completion_status.json").exists() or (out / "hf_10jobs_MISSING_BASELINE.md").exists() or (out / "vllm_10jobs_MISSING_BASELINE.md").exists())
    existing_gap = bool((out / "existing_cache_gap_summary.csv").exists() and (out / "existing_cache_gap_summary.csv").stat().st_size > 0)
    report = {
        "paper_writing_allowed": bool(controlled_pass and semantics_pass and randomized_pass and unexplained_ok and real_scope_consistent),
        "evaluation_ready": {
            "controlled_validation_nonempty_and_all_pass": controlled_pass,
            "adaptive_semantics_audit_pass": semantics_pass,
            "randomized_heldout_policy_pass": randomized_pass,
            "mechanism_attribution_unexplained_under_20pct": unexplained_ok,
            "real_trace_scope_consistent": real_scope_consistent,
        },
        "claims_allowed": {
            "existing_prefix_cache_gap": existing_gap,
            "adaptive_non_worse": randomized_pass,
            "high_opportunity_speedup": randomized_pass,
            "universal_superiority": False,
            "replication": False,
        },
    }
    claims = pd.DataFrame(
        [
            {
                "claim": "Randomized held-out adaptive policy",
                "status": "supported" if randomized_pass else "failed",
                "primary_metric": "test_fraction_non_worse_than_apc_within_5pct",
                "evidence_file": "randomized_policy_prediction_eval.csv",
            },
            {
                "claim": "Adaptive execution semantics audit",
                "status": "supported" if semantics_pass else "failed",
                "primary_metric": "semantic_violations",
                "evidence_file": "adaptive_semantics_summary.csv",
            },
            {
                "claim": "Mechanism attribution unexplained bound",
                "status": "supported" if unexplained_ok else "failed",
                "primary_metric": "unexplained_pct",
                "evidence_file": "mechanism_attribution_summary.csv",
            },
            {
                "claim": "Real trace scope",
                "status": "mini_sanity" if real_scope_consistent else "missing",
                "primary_metric": "trace_completion_status",
                "evidence_file": "hf/vllm trace status or missing baseline",
            },
        ]
    )
    claims.to_csv(out / "paper_claims_matrix.csv", index=False)
    write_json(out / "report.json", report)
    (out / "report.md").write_text(
        "# WaferAgent Round13 Paper Artifacts\n\n"
        "All wafer performance numbers are trace-driven wafer-scale simulator results, not real wafer hardware measurements.\n\n"
        f"```json\n{json.dumps(report, indent=2, sort_keys=True)}\n```\n",
        encoding="utf-8",
    )
    _source_manifest(sources).to_csv(out / "source_run_manifest.csv", index=False)
    files = []
    for p in sorted(x for x in out.rglob("*") if x.is_file()):
        if p.name == "artifact_manifest.json":
            continue
        files.append({"path": p.relative_to(out).as_posix(), "sha256": file_sha256(p)})
    write_json(out / "artifact_manifest.json", {"created_at": datetime.now(timezone.utc).isoformat(), "export_git_commit": pre_meta.get("git_commit", ""), "files": files})
    finalize_run_dir(out)
    print(f"Exported Round13 paper artifacts: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
