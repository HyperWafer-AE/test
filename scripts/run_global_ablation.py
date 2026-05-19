#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from waferagent.arrival import ArrivalConfig
from waferagent.global_simulator import simulate_global
from waferagent.mesh import MeshConfig
from waferagent.simulator import load_trace_glob
from waferagent.statistics import write_summary_with_ci
from waferagent.utils import enforce_clean_git_tree, finalize_run_dir, init_run_dir, write_json


DEFAULT_VARIANTS = (
    "waferagent_full,no_kv_sharing,no_shared_kv_decode_cohort,no_affinity_placement,"
    "no_aggregator_placement,no_shared_kv_replication,no_distributed_sram_policy,no_future_reuse_policy"
)


def _rates(text: str) -> list[float]:
    return [float(x) for x in str(text).split(",") if x.strip()]


def _variants(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _delta_rows(summary: pd.DataFrame) -> list[dict]:
    metrics = [
        ("decode_shared_kv_read_bytes", "decode cohort", 0.05, "higher_variant"),
        ("mesh_total_traffic_bytes", "placement/mesh", 0.05, "higher_variant"),
        ("jct_p99_ms", "tail latency", 0.05, "higher_variant"),
        ("shared_prefill_compute_ms_saved", "prefix compute reuse", 0.95, "lower_variant"),
        ("sram_reload_bytes", "SRAM/future reuse", 0.05, "higher_variant"),
        ("energy_per_job_j", "energy", 0.02, "higher_variant"),
    ]
    rows: list[dict] = []
    for rate, group in summary.groupby("arrival_rate_jobs_per_s", dropna=False):
        full = group.loc[group["baseline"] == "waferagent_full"]
        if full.empty:
            continue
        full_row = full.iloc[0]
        for _, variant in group.iterrows():
            name = str(variant["baseline"])
            if name == "waferagent_full":
                continue
            for metric, claim, threshold, direction in metrics:
                if metric not in group.columns:
                    continue
                full_value = float(full_row.get(metric, 0.0))
                variant_value = float(variant.get(metric, 0.0))
                delta_abs = variant_value - full_value
                denom = max(1.0, abs(full_value))
                delta_pct = delta_abs / denom
                if direction == "higher_variant":
                    supported = delta_pct >= threshold
                else:
                    supported = delta_pct <= -threshold
                if name == "no_kv_sharing" and metric == "shared_prefill_compute_ms_saved":
                    supported = variant_value <= 1e-9 and full_value > 0
                    delta_pct = -1.0 if full_value > 0 else 0.0
                    delta_abs = variant_value - full_value
                rows.append(
                    {
                        "arrival_rate_jobs_per_s": rate,
                        "variant": name,
                        "metric": metric,
                        "full_value": full_value,
                        "variant_value": variant_value,
                        "delta_abs": delta_abs,
                        "delta_pct": delta_pct,
                        "paper_claim": claim,
                        "threshold": threshold,
                        "supported": bool(supported),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True)
    parser.add_argument("--wafer-config", default="configs/wafer/wse_like.yaml")
    parser.add_argument("--arrival-mode", default="poisson", choices=["closed_loop", "poisson", "burst", "replay"])
    parser.add_argument("--arrival-rate-jobs-per-s", default="4,8,16")
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--duration-source", default="synthetic", choices=["trace", "calibrated", "synthetic"])
    parser.add_argument("--calibration", default="")
    parser.add_argument("--prefix-extension-calibration", default="")
    parser.add_argument("--shared-attention-cost-fit", default="")
    parser.add_argument("--shared-attention-accounting", default="cohort_stage", choices=["stage_amortized", "cohort_stage", "per_member"])
    parser.add_argument("--out", default="results/round7_ablation")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--engine", default="synthetic")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--gpus", default="")
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--legacy-heuristic-multipliers", action="store_true")
    parser.add_argument("--clean-required", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    enforce_clean_git_tree(args.clean_required, args.allow_dirty)
    neutral = not bool(args.legacy_heuristic_multipliers)
    out = init_run_dir(
        args.out,
        {
            "run_type": "global_ablation",
            "traces": args.traces,
            "wafer_config": args.wafer_config,
            "arrival_mode": args.arrival_mode,
            "arrival_rate_jobs_per_s": args.arrival_rate_jobs_per_s,
            "variants": args.variants,
            "duration_source": args.duration_source,
            "shared_attention_accounting": args.shared_attention_accounting,
            "neutral_mechanism_multipliers": neutral,
            "seed": args.seed,
            "max_jobs": args.max_jobs,
        },
    )
    traces = load_trace_glob(args.traces)
    if args.max_jobs:
        keep = set(sorted({tr.job_id for tr in traces})[: args.max_jobs])
        traces = [tr for tr in traces if tr.job_id in keep]
    mesh_cfg = MeshConfig.from_yaml(args.wafer_config)
    combined: dict[str, list[pd.DataFrame]] = {}
    variants = _variants(args.variants)
    for rate in _rates(args.arrival_rate_jobs_per_s):
        result = simulate_global(
            traces,
            mesh_cfg,
            variants,
            ArrivalConfig(mode=args.arrival_mode, rate_jobs_per_s=rate, seed=args.seed, max_jobs=args.max_jobs or 0),
            seed=args.seed,
            neutral_multipliers=neutral,
            calibration=args.calibration or None,
            prefix_extension_calibration=args.prefix_extension_calibration or None,
            shared_attention_cost_fit=args.shared_attention_cost_fit or None,
            shared_attention_accounting=args.shared_attention_accounting,
            duration_source=args.duration_source,
        )
        for name, df in result.items():
            tmp = df.copy()
            tmp["arrival_rate_jobs_per_s"] = rate
            combined.setdefault(name, []).append(tmp)

    sim = out / "simulation"
    for name, parts in combined.items():
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        df.to_csv(sim / f"{name}.csv", index=False)
        if name == "global_simulation_summary":
            df.to_csv(sim / "simulation_summary.csv", index=False)
    summary = pd.concat(combined.get("global_simulation_summary", []), ignore_index=True)
    pd.DataFrame(_delta_rows(summary)).to_csv(sim / "ablation_delta_summary.csv", index=False)
    if "global_job_metrics" in combined:
        write_summary_with_ci(pd.concat(combined["global_job_metrics"], ignore_index=True), sim / "summary_with_ci.csv")
    write_json(out / "model_selection.json", {"engine_used": args.engine, "model": args.model, "fallback_count": 0})
    finalize_run_dir(out)
    print(f"Global ablation complete: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
