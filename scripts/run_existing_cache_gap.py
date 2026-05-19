#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from waferagent.arrival import ArrivalConfig
from waferagent.global_simulator import simulate_global
from waferagent.mesh import MeshConfig
from waferagent.paper_figures import bar_from_csv, stacked_cache_gap
from waferagent.simulator import load_trace_glob
from waferagent.utils import enforce_clean_git_tree, finalize_run_dir, init_run_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True)
    parser.add_argument("--wafer-config", default="configs/wafer/wse_like.yaml")
    parser.add_argument("--baselines", default="no_cache,apc_like,kvflow_like,pat_like,waferagent_full")
    parser.add_argument("--out", default="results/round5_existing_cache_gap")
    parser.add_argument("--duration-source", default="synthetic", choices=["synthetic", "trace", "calibrated"])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--engine", default="synthetic")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--gpus", default="")
    parser.add_argument("--clean-required", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    enforce_clean_git_tree(args.clean_required, args.allow_dirty)
    out = init_run_dir(
        args.out,
        {
            "run_type": "existing_cache_gap",
            "traces": args.traces,
            "wafer_config": args.wafer_config,
            "baselines": args.baselines,
            "duration_source": args.duration_source,
            "seed": args.seed,
        },
    )
    traces = load_trace_glob(args.traces)
    mesh = MeshConfig.from_yaml(args.wafer_config)
    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]
    result = simulate_global(
        traces,
        mesh,
        baselines,
        ArrivalConfig(mode="closed_loop", seed=args.seed),
        seed=args.seed,
        duration_source=args.duration_source,
    )
    summary = result["global_simulation_summary"].copy()
    metrics = result["global_job_metrics"].copy()
    summary["avoided_prefill_tokens"] = summary.get("avoided_prefill_tokens", 0.0)
    summary["prefill_compute_ms_saved"] = summary.get("shared_prefill_compute_ms_saved", 0.0)
    summary["mesh_traffic_bytes"] = summary.get("mesh_total_traffic_bytes", 0.0)
    summary["jct_ms"] = summary.get("jct_p50_ms", 0.0)
    summary["slo_goodput"] = result["slo_goodput"].groupby("baseline")["slo_goodput_jobs_per_s"].max().reindex(summary["baseline"]).to_numpy()
    summary["energy_j"] = summary.get("energy_per_job_j", 0.0)
    sim = out / "simulation"
    summary.to_csv(sim / "existing_cache_gap_summary.csv", index=False)
    metrics.to_csv(sim / "existing_cache_gap_per_workload.csv", index=False)
    summary.to_csv(sim / "existing_cache_gap_per_baseline.csv", index=False)
    for name, df in result.items():
        df.to_csv(sim / f"{name}.csv", index=False)
    fig = out / "figures"
    stacked_cache_gap(sim / "existing_cache_gap_summary.csv", fig / "fig4_prefix_cache_gap_stacked_bars")
    bar_from_csv(sim / "existing_cache_gap_summary.csv", "baseline", "decode_shared_kv_read_bytes", fig / "fig5_decode_shared_kv_bytes_by_baseline")
    bar_from_csv(sim / "existing_cache_gap_summary.csv", "baseline", "mesh_traffic_bytes", fig / "fig6_mesh_traffic_after_prefix_cache")
    write_json(out / "model_selection.json", {"engine_used": args.engine, "model": args.model, "fallback_count": 0})
    finalize_run_dir(out)
    print(f"Existing cache gap complete: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
