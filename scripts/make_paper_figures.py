#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from waferagent.paper_figures import bar_from_csv, line_from_csv, stacked_cache_gap
from waferagent.utils import enforce_clean_git_tree, finalize_run_dir, init_run_dir


def _maybe(fn, *args) -> None:
    try:
        if Path(args[0]).exists():
            fn(*args)
    except Exception as exc:
        print(f"figure skipped for {args[0]}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/round5_final_report")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--engine", default="synthetic")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--gpus", default="")
    parser.add_argument("--clean-required", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    enforce_clean_git_tree(args.clean_required, args.allow_dirty)
    out = init_run_dir(args.out, {"run_type": "paper_figures"})
    fig = out / "figures"
    _maybe(bar_from_csv, "results/round5_workload_opportunity/simulation/shared_kv_opportunity.csv", "workload", "expected_decode_shared_kv_read_bytes", fig / "fig2_shared_kv_opportunity_by_workload", None)
    _maybe(bar_from_csv, "results/round5_workload_opportunity/simulation/shared_kv_opportunity.csv", "workload", "safe_shared_prefix_tokens", fig / "fig3_decode_vs_prefill_reuse_opportunity", None)
    _maybe(stacked_cache_gap, "results/round5_existing_cache_gap/simulation/existing_cache_gap_summary.csv", fig / "fig4_prefix_cache_gap_stacked_bars")
    _maybe(bar_from_csv, "results/round5_existing_cache_gap/simulation/existing_cache_gap_summary.csv", "baseline", "decode_shared_kv_read_bytes", fig / "fig5_decode_shared_kv_bytes_by_baseline")
    _maybe(bar_from_csv, "results/round5_existing_cache_gap/simulation/existing_cache_gap_summary.csv", "baseline", "mesh_traffic_bytes", fig / "fig6_mesh_traffic_after_prefix_cache")
    _maybe(line_from_csv, "results/round5_decode_cohort_sweep/simulation/decode_cohort_sweep.csv", "cohort_size", "shared_kv_read_reduction_ratio", fig / "fig7_cohort_size_vs_shared_kv_read_reduction", "num_agents")
    _maybe(line_from_csv, "results/round5_replication_tradeoff/simulation/replication_tradeoff_summary.csv", "sram_per_tile_mb", "mesh_traffic_bytes", fig / "fig10_replication_pareto_sram_vs_mesh", "replication_policy")
    _maybe(line_from_csv, "results/round5_global_main_neutral/simulation/global_simulation_summary.csv", "arrival_rate_jobs_per_s", "jobs_per_s", fig / "fig13_slo_goodput_vs_arrival_rate", "baseline")
    _maybe(line_from_csv, "results/round5_global_main_neutral/simulation/global_simulation_summary.csv", "arrival_rate_jobs_per_s", "jct_p99_ms", fig / "fig14_jct_p99_vs_arrival_rate", "baseline")
    _maybe(bar_from_csv, "results/round5_ablation/simulation/global_simulation_summary.csv", "baseline", "jct_p99_ms", fig / "fig18_ablation_jct", None)
    finalize_run_dir(out)
    print(f"Paper figures complete: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
