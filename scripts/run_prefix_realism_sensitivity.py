#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from waferagent.arrival import ArrivalConfig
from waferagent.global_simulator import simulate_global
from waferagent.llm_runner import RunnerConfig
from waferagent.mesh import MeshConfig
from waferagent.trace_collector import collect_graph_traces
from waferagent.utils import enforce_clean_git_tree, finalize_run_dir, init_run_dir, write_json
from waferagent.workloads import WorkloadParams, generate_workload


def _ints(text: str) -> list[int]:
    return [int(x) for x in str(text).split(",") if x.strip()]


def _floats(text: str) -> list[float]:
    return [float(x) for x in str(text).split(",") if x.strip()]


def _prefix_hit_rate(traces) -> float:
    seen_jobs: dict[str, set[str]] = {}
    hits = 0
    total = 0
    for tr in traces:
        for prefix in tr.shared_prefix_ids:
            total += 1
            jobs = seen_jobs.setdefault(prefix, set())
            if jobs and tr.job_id not in jobs:
                hits += 1
            jobs.add(tr.job_id)
    return hits / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wafer-config", default="configs/wafer/wse_like.yaml")
    parser.add_argument("--cross-job-task-group-size", default="1,2,5,10,20,50")
    parser.add_argument("--unique-task-ratio", default="0,0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--workloads", default="debate,moa,long_context_swe_stress")
    parser.add_argument("--num-jobs", type=int, default=20)
    parser.add_argument("--baselines", default="apc_like,waferagent_full")
    parser.add_argument("--duration-source", default="synthetic", choices=["synthetic", "trace", "calibrated"])
    parser.add_argument("--out", default="results/round6_prefix_realism_sensitivity")
    parser.add_argument("--seed", type=int, default=11)
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
            "run_type": "prefix_realism_sensitivity",
            "wafer_config": args.wafer_config,
            "cross_job_task_group_size": args.cross_job_task_group_size,
            "unique_task_ratio": args.unique_task_ratio,
            "workloads": args.workloads,
            "num_jobs": args.num_jobs,
            "duration_source": args.duration_source,
            "seed": args.seed,
        },
    )
    mesh = MeshConfig.from_yaml(args.wafer_config)
    workloads = [x.strip() for x in args.workloads.split(",") if x.strip()]
    baselines = [x.strip() for x in args.baselines.split(",") if x.strip()]
    rows = []
    all_summaries = []
    for group_size in _ints(args.cross_job_task_group_size):
        for unique_ratio in _floats(args.unique_task_ratio):
            graphs = []
            for workload in workloads:
                for j in range(args.num_jobs):
                    graphs.append(
                        generate_workload(
                            WorkloadParams(
                                workload=workload,
                                job_id=f"{workload}_realism_job_{j}",
                                seed=args.seed + j,
                                cross_job_task_group_size=group_size,
                                unique_task_ratio=unique_ratio,
                                prefix_namespace=f"round6-g{group_size}-u{unique_ratio}",
                            )
                        )
                    )
            traces = collect_graph_traces(
                graphs,
                f"round6_prefix_realism_g{group_size}_u{unique_ratio}",
                RunnerConfig(engine="synthetic", seed=args.seed),
            )
            result = simulate_global(
                traces,
                mesh,
                baselines,
                ArrivalConfig(mode="closed_loop", seed=args.seed),
                seed=args.seed,
                duration_source=args.duration_source,
            )
            summary = result["global_simulation_summary"].copy()
            summary["cross_job_task_group_size"] = group_size
            summary["unique_task_ratio"] = unique_ratio
            summary["cross_job_prefix_hit_rate_observed"] = _prefix_hit_rate(traces)
            all_summaries.append(summary)
            rows.append(
                {
                    "cross_job_task_group_size": group_size,
                    "unique_task_ratio": unique_ratio,
                    "cross_job_prefix_hit_rate_observed": _prefix_hit_rate(traces),
                    "num_trace_records": len(traces),
                    "num_unique_prefixes": len({p for tr in traces for p in tr.shared_prefix_ids}),
                }
            )
    sim = out / "simulation"
    pd.DataFrame(rows).to_csv(sim / "prefix_realism_prefix_stats.csv", index=False)
    combined = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    combined.to_csv(sim / "prefix_realism_sensitivity.csv", index=False)
    regime_rows = []
    if not combined.empty:
        for (group_size, unique_ratio), sub in combined.groupby(["cross_job_task_group_size", "unique_task_ratio"]):
            apc = sub.loc[sub["baseline"] == "apc_like"]
            waf = sub.loc[sub["baseline"] == "waferagent_full"]
            if apc.empty or waf.empty:
                continue
            apc_jct = float(apc["jct_p99_ms"].mean())
            waf_jct = float(waf["jct_p99_ms"].mean())
            hit = float(waf["cross_job_prefix_hit_rate_observed"].mean()) if "cross_job_prefix_hit_rate_observed" in waf.columns else 0.0
            reduction = float(waf["decode_kv_read_reduction_ratio"].mean()) if "decode_kv_read_reduction_ratio" in waf.columns else 0.0
            delta = (waf_jct - apc_jct) / max(1.0, apc_jct)
            if hit >= 0.08 and reduction >= 0.20 and delta <= 0.0:
                label = "high_reuse_high_decode_pressure"
            elif delta > 0.0:
                label = "low_reuse_apc_better"
            else:
                label = "medium_reuse"
            regime_rows.append(
                {
                    "cross_job_task_group_size": group_size,
                    "unique_task_ratio": unique_ratio,
                    "cross_job_prefix_hit_rate_observed": hit,
                    "shared_kv_read_reduction_ratio": reduction,
                    "waferagent_vs_apc_jct_p99_delta_pct": delta,
                    "waferagent_jct_p99_ms": waf_jct,
                    "apc_like_jct_p99_ms": apc_jct,
                    "regime_label": label,
                }
            )
    pd.DataFrame(regime_rows).to_csv(sim / "regime_classification.csv", index=False)
    stats = pd.DataFrame(rows)
    monotonic = {
        "overall_declines_with_unique_task_ratio": False,
        "by_group_size": {},
    }
    for group_size, sub in stats.groupby("cross_job_task_group_size"):
        ordered = sub.sort_values("unique_task_ratio")
        first = float(ordered["cross_job_prefix_hit_rate_observed"].iloc[0]) if not ordered.empty else 0.0
        last = float(ordered["cross_job_prefix_hit_rate_observed"].iloc[-1]) if not ordered.empty else 0.0
        nonincreasing_steps = int(
            sum(
                b <= a + 1e-9
                for a, b in zip(
                    ordered["cross_job_prefix_hit_rate_observed"].tolist(),
                    ordered["cross_job_prefix_hit_rate_observed"].tolist()[1:],
                )
            )
        )
        monotonic["by_group_size"][str(group_size)] = {
            "first_hit_rate": first,
            "last_hit_rate": last,
            "declines": last <= first,
            "nonincreasing_steps": nonincreasing_steps,
            "num_steps": max(0, len(ordered) - 1),
        }
    if monotonic["by_group_size"]:
        monotonic["overall_declines_with_unique_task_ratio"] = all(v["declines"] for v in monotonic["by_group_size"].values())
    write_json(sim / "prefix_realism_monotonicity.json", monotonic)
    write_json(out / "model_selection.json", {"engine_used": args.engine, "model": args.model, "fallback_count": 0})
    finalize_run_dir(out)
    print(f"Prefix realism sensitivity complete: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
