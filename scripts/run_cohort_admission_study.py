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


def _rates(text: str) -> list[float]:
    return [float(x) for x in str(text).split(",") if x.strip()]


def _baselines(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _annotate_admission(summary: pd.DataFrame, admission: pd.DataFrame, slo: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return admission
    out = admission.copy()
    full = summary.loc[summary["baseline"] == "waferagent_full"]
    no = summary.loc[summary["baseline"] == "no_shared_kv_decode_cohort"]
    if not full.empty and not no.empty:
        full_row = full.iloc[0]
        no_row = no.iloc[0]
        out["jct_p99_delta_vs_no_cohort"] = float(full_row["jct_p99_ms"]) - float(no_row["jct_p99_ms"])
        no_jct = max(1.0, float(no_row["jct_p99_ms"]))
        out["jct_p99_delta_pct_vs_no_cohort"] = out["jct_p99_delta_vs_no_cohort"] / no_jct
        out["decode_kv_bytes_saved"] = float(no_row.get("decode_shared_kv_read_bytes", 0.0)) - float(full_row.get("decode_shared_kv_read_bytes", 0.0))
    if not slo.empty and {"baseline", "slo_ms", "slo_goodput_jobs_per_s"} <= set(slo.columns):
        full_slo = slo.loc[(slo["baseline"] == "waferagent_full") & (slo["slo_type"] == "jct_ms")]
        no_slo = slo.loc[(slo["baseline"] == "no_shared_kv_decode_cohort") & (slo["slo_type"] == "jct_ms")]
        if not full_slo.empty and not no_slo.empty:
            merged = full_slo.merge(no_slo, on="slo_ms", suffixes=("_full", "_no"))
            if not merged.empty:
                out["slo_goodput_delta_vs_no_cohort"] = float(
                    (merged["slo_goodput_jobs_per_s_full"] - merged["slo_goodput_jobs_per_s_no"]).mean()
                )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True)
    parser.add_argument("--wafer-config", default="configs/wafer/wse_like.yaml")
    parser.add_argument("--arrival-mode", default="burst", choices=["closed_loop", "poisson", "burst", "replay"])
    parser.add_argument("--arrival-rate-jobs-per-s", default="16")
    parser.add_argument("--baselines", default="waferagent_full,no_shared_kv_decode_cohort")
    parser.add_argument("--duration-source", default="synthetic", choices=["trace", "calibrated", "synthetic"])
    parser.add_argument("--out", default="results/round8_cohort_admission")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-jobs", type=int, default=0)
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
            "run_type": "cohort_admission_study",
            "traces": args.traces,
            "wafer_config": args.wafer_config,
            "arrival_mode": args.arrival_mode,
            "arrival_rate_jobs_per_s": args.arrival_rate_jobs_per_s,
            "baselines": args.baselines,
            "duration_source": args.duration_source,
            "seed": args.seed,
            "max_jobs": args.max_jobs,
        },
    )
    traces = load_trace_glob(args.traces)
    if args.max_jobs:
        keep = set(sorted({tr.job_id for tr in traces})[: args.max_jobs])
        traces = [tr for tr in traces if tr.job_id in keep]
    mesh = MeshConfig.from_yaml(args.wafer_config)
    combined: dict[str, list[pd.DataFrame]] = {}
    for rate in _rates(args.arrival_rate_jobs_per_s):
        result = simulate_global(
            traces,
            mesh,
            _baselines(args.baselines),
            ArrivalConfig(mode=args.arrival_mode, rate_jobs_per_s=rate, seed=args.seed, max_jobs=args.max_jobs or 0),
            seed=args.seed,
            duration_source=args.duration_source,
        )
        for name, df in result.items():
            tmp = df.copy()
            tmp["arrival_rate_jobs_per_s"] = rate
            combined.setdefault(name, []).append(tmp)

    sim = out / "simulation"
    materialized: dict[str, pd.DataFrame] = {}
    for name, parts in combined.items():
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        materialized[name] = df
        df.to_csv(sim / f"{name}.csv", index=False)
    admission_summary = _annotate_admission(
        materialized.get("global_simulation_summary", pd.DataFrame()),
        materialized.get("cohort_admission_summary", pd.DataFrame()),
        materialized.get("slo_goodput", pd.DataFrame()),
    )
    admission_summary.to_csv(sim / "cohort_admission_summary.csv", index=False)
    if "global_job_metrics" in materialized:
        write_summary_with_ci(materialized["global_job_metrics"], sim / "summary_with_ci.csv")
    write_json(out / "model_selection.json", {"engine_used": args.engine, "model": args.model, "fallback_count": 0})
    finalize_run_dir(out)
    print(f"Cohort admission study complete: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
