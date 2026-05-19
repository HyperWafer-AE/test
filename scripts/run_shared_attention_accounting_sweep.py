#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from waferagent.arrival import ArrivalConfig
from waferagent.global_simulator import simulate_global
from waferagent.mesh import MeshConfig
from waferagent.paper_figures import line_from_csv
from waferagent.simulator import load_trace_glob
from waferagent.utils import enforce_clean_git_tree, finalize_run_dir, init_run_dir, write_json


def _csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _floats(text: str) -> list[float]:
    return [float(x) for x in _csv(text)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True)
    parser.add_argument("--wafer-config", default="configs/wafer/wse_like.yaml")
    parser.add_argument("--shared-attention-cost-fit", required=True)
    parser.add_argument("--accounting-modes", default="stage_amortized,cohort_stage,per_member")
    parser.add_argument("--baselines", default="apc_like,pat_like_traffic_only,waferagent_latency_safe")
    parser.add_argument("--arrival-mode", default="poisson", choices=["closed_loop", "poisson", "burst", "replay"])
    parser.add_argument("--arrival-rate-jobs-per-s", default="2,4,8,16")
    parser.add_argument("--max-jobs", type=int, default=100)
    parser.add_argument("--out", default="results/round10_attention_accounting_sweep")
    parser.add_argument("--duration-source", default="synthetic", choices=["trace", "calibrated", "synthetic"])
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
            "run_type": "shared_attention_accounting_sweep",
            "traces": args.traces,
            "wafer_config": args.wafer_config,
            "shared_attention_cost_fit": args.shared_attention_cost_fit,
            "accounting_modes": args.accounting_modes,
            "baselines": args.baselines,
            "arrival_mode": args.arrival_mode,
            "arrival_rate_jobs_per_s": args.arrival_rate_jobs_per_s,
            "max_jobs": args.max_jobs,
            "duration_source": args.duration_source,
            "seed": args.seed,
        },
    )
    traces = load_trace_glob(args.traces)
    if args.max_jobs:
        keep = set(sorted({tr.job_id for tr in traces})[: args.max_jobs])
        traces = [tr for tr in traces if tr.job_id in keep]
    mesh = MeshConfig.from_yaml(args.wafer_config)
    rows = []
    for mode in _csv(args.accounting_modes):
        for rate in _floats(args.arrival_rate_jobs_per_s):
            result = simulate_global(
                traces,
                mesh,
                _csv(args.baselines),
                ArrivalConfig(mode=args.arrival_mode, rate_jobs_per_s=rate, seed=args.seed, max_jobs=args.max_jobs),
                seed=args.seed,
                duration_source=args.duration_source,
                shared_attention_cost_fit=args.shared_attention_cost_fit,
                shared_attention_accounting=mode,
            )
            summary = result["global_simulation_summary"].copy()
            summary["accounting_mode"] = mode
            summary["arrival_rate_jobs_per_s"] = rate
            rows.append(summary)
    sim = out / "simulation"
    summary_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    summary_df.to_csv(sim / "accounting_summary.csv", index=False)
    deltas = []
    if not summary_df.empty:
        base = summary_df[summary_df["accounting_mode"] == "cohort_stage"]
        for _, row in summary_df.iterrows():
            ref = base[(base["baseline"] == row["baseline"]) & (base["arrival_rate_jobs_per_s"] == row["arrival_rate_jobs_per_s"])]
            if ref.empty:
                continue
            ref_row = ref.iloc[0]
            for metric in ["jct_p99_ms", "jobs_per_s", "decode_shared_kv_read_bytes"]:
                deltas.append(
                    {
                        "accounting_mode": row["accounting_mode"],
                        "baseline": row["baseline"],
                        "arrival_rate_jobs_per_s": row["arrival_rate_jobs_per_s"],
                        "metric": metric,
                        "value": row[metric],
                        "cohort_stage_value": ref_row[metric],
                        "delta_pct_vs_cohort_stage": (row[metric] - ref_row[metric]) / max(1.0, abs(ref_row[metric])),
                    }
                )
    pd.DataFrame(deltas).to_csv(sim / "accounting_delta.csv", index=False)
    fig = out / "figures"
    if not summary_df.empty:
        line_from_csv(sim / "accounting_summary.csv", "arrival_rate_jobs_per_s", "jct_p99_ms", fig / "fig_accounting_modes_p99", hue="accounting_mode")
    write_json(out / "model_selection.json", {"engine_used": args.engine, "model": args.model, "fallback_count": 0})
    finalize_run_dir(out)
    print(f"Shared-attention accounting sweep complete: {Path(out).resolve()}")


if __name__ == "__main__":
    main()

