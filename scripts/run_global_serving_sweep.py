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


def _parse_floats(text: str) -> list[float]:
    return [float(x) for x in str(text).split(",") if x.strip()]


def _parse_baselines(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True)
    parser.add_argument("--wafer-config", default="configs/wafer/wse_like.yaml")
    parser.add_argument("--arrival-mode", default="poisson", choices=["closed_loop", "poisson", "burst", "replay"])
    parser.add_argument("--arrival-rate-jobs-per-s", default="2.0")
    parser.add_argument("--arrival-replay", default="")
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--baselines", default="wafer_naive,continuum_like,kvflow_like,waferagent_full,oracle")
    parser.add_argument("--duration-source", default="synthetic", choices=["trace", "calibrated", "synthetic"])
    parser.add_argument("--calibration", default="")
    parser.add_argument("--prefix-extension-calibration", default="")
    parser.add_argument("--shared-attention-cost-fit", default="")
    parser.add_argument("--shared-attention-accounting", default="cohort_stage", choices=["stage_amortized", "cohort_stage", "per_member"])
    parser.add_argument("--out", default="results/round4_global_main_neutral")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--engine", default="synthetic")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--gpus", default="")
    parser.add_argument("--legacy-heuristic-multipliers", action="store_true")
    parser.add_argument("--clean-required", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    enforce_clean_git_tree(args.clean_required, args.allow_dirty)
    neutral = not bool(args.legacy_heuristic_multipliers)
    out = init_run_dir(
        args.out,
        {
            "run_type": "global_serving_sweep",
            "traces": args.traces,
            "wafer_config": args.wafer_config,
            "arrival_mode": args.arrival_mode,
            "arrival_rate_jobs_per_s": args.arrival_rate_jobs_per_s,
            "baselines": args.baselines,
            "duration_source": args.duration_source,
            "calibration": args.calibration,
            "prefix_extension_calibration": args.prefix_extension_calibration,
            "shared_attention_cost_fit": args.shared_attention_cost_fit,
            "shared_attention_accounting": args.shared_attention_accounting,
            "neutral_mechanism_multipliers": neutral,
            "legacy_heuristic_multipliers": bool(args.legacy_heuristic_multipliers),
            "seed": args.seed,
            "engine": args.engine,
            "model": args.model,
            "gpus": args.gpus,
            "clean_required": bool(args.clean_required),
        },
    )
    traces = load_trace_glob(args.traces)
    if args.max_jobs:
        keep = set(sorted({tr.job_id for tr in traces})[: args.max_jobs])
        traces = [tr for tr in traces if tr.job_id in keep]
    mesh_cfg = MeshConfig.from_yaml(args.wafer_config)
    baselines = _parse_baselines(args.baselines)
    combined: dict[str, list[pd.DataFrame]] = {}
    for rate in _parse_floats(args.arrival_rate_jobs_per_s):
        result = simulate_global(
            traces,
            mesh_cfg,
            baselines,
            ArrivalConfig(
                mode=args.arrival_mode,
                rate_jobs_per_s=rate,
                seed=args.seed,
                max_jobs=args.max_jobs or 0,
                replay_path=args.arrival_replay or None,
            ),
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

    sim_dir = out / "simulation"
    for name, parts in combined.items():
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        df.to_csv(sim_dir / f"{name}.csv", index=False)
        if name == "global_job_metrics":
            df.to_csv(sim_dir / "simulation_metrics.csv", index=False)
        if name == "global_simulation_summary":
            df.to_csv(sim_dir / "simulation_summary.csv", index=False)
    if "global_job_metrics" in combined:
        metrics = pd.concat(combined["global_job_metrics"], ignore_index=True)
        write_summary_with_ci(metrics, sim_dir / "summary_with_ci.csv")
    write_json(
        out / "model_selection.json",
        {"engine_used": args.engine, "model": args.model, "fallback_count": 0, "global_serving": True},
    )
    finalize_run_dir(out)
    print(f"Global serving sweep complete: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
