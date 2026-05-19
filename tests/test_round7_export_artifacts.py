from __future__ import annotations

import shutil
import subprocess
import sys
import json
from pathlib import Path

import pandas as pd

from waferagent.baselines import get_baseline
from waferagent.utils import PROJECT_ROOT, file_sha256, write_json


def _write_source(root: Path, kind: str) -> Path:
    src = root / kind
    sim = src / "simulation"
    sim.mkdir(parents=True, exist_ok=True)
    write_json(src / "metadata.json", {"git_commit": "testcommit", "git_dirty": False, "created_unix": 1})
    write_json(
        src / "run_manifest.json",
        {"config": {"duration_source": "synthetic", "arrival_mode": "poisson", "arrival_rate_jobs_per_s": "1"}},
    )
    (src / "command.txt").write_text(f"run {kind}\n", encoding="utf-8")
    return src


def _make_sources(root: Path) -> list[Path]:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    main = _write_source(root, "round7_global_main_neutral")
    pd.DataFrame(
        [
            {"baseline": "apc_like", "jct_p99_ms": 20, "jobs_per_s": 1, "decode_shared_kv_read_bytes": 100, "mesh_total_traffic_bytes": 1000},
            {"baseline": "waferagent_full", "jct_p99_ms": 10, "jobs_per_s": 2, "decode_shared_kv_read_bytes": 50, "mesh_total_traffic_bytes": 100},
        ]
    ).to_csv(main / "simulation/global_simulation_summary.csv", index=False)
    pd.DataFrame([{"baseline": "waferagent_full", "job_id": "j0"}]).to_csv(main / "simulation/global_job_metrics.csv", index=False)
    pd.DataFrame([{"baseline": "waferagent_full", "slo_goodput_jobs_per_s": 1.0}]).to_csv(main / "simulation/slo_goodput.csv", index=False)
    pd.DataFrame([{"baseline": "waferagent_full", "total_runtime_overhead_ms": 1.0}]).to_csv(main / "simulation/planning_overhead_summary.csv", index=False)

    gap = _write_source(root, "round7_existing_cache_gap")
    pd.DataFrame(
        [
            {"baseline": "apc_like", "decode_shared_kv_read_bytes": 100, "prefill_compute_ms_saved": 5, "avoided_prefill_tokens": 10},
            {"baseline": "waferagent_full", "decode_shared_kv_read_bytes": 50, "prefill_compute_ms_saved": 5, "avoided_prefill_tokens": 10},
        ]
    ).to_csv(gap / "simulation/existing_cache_gap_summary.csv", index=False)

    ablation = _write_source(root, "round7_ablation")
    pd.DataFrame(
        [
            {"baseline": "waferagent_full", "jct_p99_ms": 10, "decode_shared_kv_read_bytes": 50, "mesh_total_traffic_bytes": 100},
            {"baseline": "no_shared_kv_decode_cohort", "jct_p99_ms": 12, "decode_shared_kv_read_bytes": 80, "mesh_total_traffic_bytes": 120},
        ]
    ).to_csv(ablation / "simulation/global_simulation_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "variant": "no_shared_kv_decode_cohort",
                "metric": "decode_shared_kv_read_bytes",
                "full_value": 50,
                "variant_value": 80,
                "delta_abs": 30,
                "delta_pct": 0.6,
                "paper_claim": "decode cohort",
                "threshold": 0.05,
                "supported": True,
            }
        ]
    ).to_csv(ablation / "simulation/ablation_delta_summary.csv", index=False)

    cohort = _write_source(root, "round7_decode_cohort_targeted")
    pd.DataFrame(
        [
            {
                "cohort_id": "c0",
                "shared_kv_id": "p",
                "node_ids": "a,b",
                "planned_start_ms": 1,
                "max_wait_ms": 2,
                "cohort_size": 2,
                "event_driven": True,
                "baseline": "waferagent_full",
                "arrival_rate_jobs_per_s": 1,
                "expected_shared_kv_bytes_read": 10,
                "expected_query_transfer_bytes": 1,
                "expected_merge_bytes": 1,
            }
        ]
    ).to_csv(cohort / "simulation/decode_cohorts.csv", index=False)

    sweep = _write_source(root, "round7_decode_cohort_sweep")
    pd.DataFrame([{"cohort_size": 2, "shared_kv_read_reduction_ratio": 0.5}]).to_csv(sweep / "simulation/decode_cohort_sweep.csv", index=False)

    repl = _write_source(root, "round7_replication_tradeoff")
    pd.DataFrame(
        [
            {"replication_policy": "no_replication", "mesh_traffic_bytes": 100},
            {"replication_policy": "benefit_cost", "mesh_traffic_bytes": 100},
        ]
    ).to_csv(repl / "simulation/replication_tradeoff_summary.csv", index=False)

    prefix = _write_source(root, "round7_prefix_realism_sensitivity")
    pd.DataFrame([{"unique_task_ratio": 0, "cross_job_prefix_hit_rate_observed": 0.9}]).to_csv(prefix / "simulation/prefix_realism_sensitivity.csv", index=False)
    pd.DataFrame([{"unique_task_ratio": 0, "cross_job_prefix_hit_rate_observed": 0.9}]).to_csv(prefix / "simulation/prefix_realism_prefix_stats.csv", index=False)

    micro = _write_source(root, "round7_shared_kv_microbench_h100")
    pd.DataFrame(
        [{"shared_prefix_tokens": 512, "num_queries": 4, "naive_latency_ms": 2, "cohort_latency_ms": 1, "speedup": 2, "naive_read_bytes": 4, "cohort_read_bytes": 1, "read_byte_reduction_ratio": 0.75}]
    ).to_csv(micro / "simulation/shared_kv_microbench_summary.csv", index=False)
    pd.DataFrame([{"mode": "cohort", "latency_ms": 1}]).to_csv(micro / "simulation/shared_kv_microbench_raw.csv", index=False)
    return [main, gap, ablation, cohort, sweep, repl, prefix, micro]


def _run_export(root: Path, sources: list[Path], name: str = "out") -> Path:
    out = root / name
    cmd = [
        sys.executable,
        "scripts/export_paper_artifacts.py",
        "--source-results",
        ",".join(str(s) for s in sources),
        "--out",
        str(out),
        "--allow-dirty",
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    return out


def test_export_ablation_not_same_as_main_and_no_semantic_fallback():
    root = PROJECT_ROOT / "tmp" / "round7_export_test"
    sources = _make_sources(root)
    out = _run_export(root, sources)
    assert not (out / "MISSING_ARTIFACTS.json").exists()
    assert file_sha256(out / "ablation_global_summary.csv") != file_sha256(out / "global_simulation_summary.csv")


def test_export_event_driven_cohorts_present_and_prefix_microbench_exported():
    root = PROJECT_ROOT / "tmp" / "round7_export_test_event"
    out = _run_export(root, _make_sources(root))
    cohorts = pd.read_csv(out / "decode_cohorts_event_driven.csv")
    assert cohorts["event_driven"].astype(bool).any()
    assert (out / "decode_cohort_analytical_sweep.csv").exists()
    assert (out / "prefix_realism_sensitivity.csv").exists()
    assert (out / "shared_kv_microbench_summary.csv").exists()


def test_export_missing_required_artifact_fails_in_report():
    root = PROJECT_ROOT / "tmp" / "round7_export_test_missing"
    sources = _make_sources(root)
    shutil.rmtree(sources[2])
    out = _run_export(root, [s for s in sources if s.exists()])
    assert (out / "MISSING_ARTIFACTS.json").exists()
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert not bool(report["paper_ready"]["artifact_tables_exported"])


def test_paper_claims_matrix_has_numeric_evidence_and_oracle_renamed():
    root = PROJECT_ROOT / "tmp" / "round7_export_test_claims"
    out = _run_export(root, _make_sources(root))
    claims = pd.read_csv(out / "paper_claims_matrix.csv")
    assert {"claim", "status", "primary_metric", "waferagent_value", "comparison_value", "delta", "threshold"} <= set(claims.columns)
    assert claims["status"].isin(["supported", "partially_supported", "demoted", "missing"]).all()
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert bool(report["paper_ready"]["oracle_renamed_not_upper_bound"])
    assert get_baseline("ideal_next_use_cache").name == "ideal_next_use_cache"
