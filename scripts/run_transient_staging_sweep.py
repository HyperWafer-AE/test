#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from waferagent.arrival import ArrivalConfig
from waferagent.global_simulator import simulate_global
from waferagent.kv_model import ModelKVConfig
from waferagent.mesh import MeshConfig
from waferagent.paper_figures import bar_from_csv
from waferagent.simulator import load_trace_glob
from waferagent.transient_staging import decide_transient_staging
from waferagent.utils import enforce_clean_git_tree, finalize_run_dir, init_run_dir, write_json


def _csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True)
    parser.add_argument("--wafer-config", default="configs/wafer/wse_like.yaml")
    parser.add_argument("--shared-attention-cost-fit", required=True)
    parser.add_argument("--baselines", default="waferagent_latency_safe_no_staging,waferagent_latency_safe_transient_staging,waferagent_latency_safe_persistent_replication")
    parser.add_argument("--arrival-rate-jobs-per-s", type=float, default=8.0)
    parser.add_argument("--max-jobs", type=int, default=100)
    parser.add_argument("--out", default="results/round10_transient_staging_sweep")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--clean-required", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    enforce_clean_git_tree(args.clean_required, args.allow_dirty)
    out = init_run_dir(args.out, {"run_type": "transient_staging_sweep", **vars(args)})
    traces = load_trace_glob(args.traces)
    if args.max_jobs:
        keep = set(sorted({tr.job_id for tr in traces})[: args.max_jobs])
        traces = [tr for tr in traces if tr.job_id in keep]
    mesh = MeshConfig.from_yaml(args.wafer_config)
    result = simulate_global(
        traces,
        mesh,
        _csv(args.baselines),
        ArrivalConfig(mode="poisson", rate_jobs_per_s=args.arrival_rate_jobs_per_s, seed=args.seed, max_jobs=args.max_jobs),
        seed=args.seed,
        duration_source="synthetic",
        shared_attention_cost_fit=args.shared_attention_cost_fit,
        shared_attention_accounting="cohort_stage",
    )
    sim = out / "simulation"
    for name, df in result.items():
        df.to_csv(sim / f"{name}.csv", index=False)
    cohorts = result["decode_cohorts"]
    events = []
    if not cohorts.empty:
        bytes_per_ms = max(1.0, mesh.link_bandwidth_GBps * 1e9 / 1000.0)
        kv_per_token = ModelKVConfig().kv_bytes_per_token
        for _, row in cohorts.iterrows():
            token_len = int(row.get("expected_shared_kv_bytes_read", 0) / max(1, kv_per_token))
            prefix_bytes = int(max(0, row.get("expected_shared_kv_bytes_read", 0)) / max(1, row.get("cohort_size", 1)))
            dec = decide_transient_staging(
                str(row.get("shared_kv_id", "")),
                prefix_bytes,
                int(row.get("cohort_size", 1)),
                token_len,
                str(row.get("shared_kv_region", "r0c0") or "r0c0"),
                str(row.get("shared_kv_region", "r0c0") or "r0c0"),
                float(row.get("planned_start_ms", 0.0)),
                1.0,
                bytes_per_ms,
                int(mesh.sram_per_tile_mb * 1024 * 1024 * 64),
            )
            events.append(dec.to_dict())
    event_df = pd.DataFrame(events)
    event_df.to_csv(sim / "transient_staging_events.csv", index=False)
    summary = result["global_simulation_summary"].copy()
    summary["staging_events"] = int(len(event_df))
    summary["accepted_staging_events"] = int(event_df["accepted"].sum()) if not event_df.empty else 0
    summary.to_csv(sim / "transient_staging_summary.csv", index=False)
    fig = out / "figures"
    if not summary.empty:
        bar_from_csv(sim / "transient_staging_summary.csv", "baseline", "jct_p99_ms", fig / "fig_transient_staging_vs_replication")
    write_json(out / "model_selection.json", {"engine_used": "synthetic", "fallback_count": 0})
    finalize_run_dir(out)
    print(f"Transient staging sweep complete: {Path(out).resolve()}")


if __name__ == "__main__":
    main()

