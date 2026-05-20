#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from waferagent.arrival import ArrivalConfig
from waferagent.global_simulator import simulate_global
from waferagent.llm_runner import RunnerConfig
from waferagent.mesh import MeshConfig
from waferagent.paper_figures import line_from_csv
from waferagent.trace_collector import collect_graph_traces
from waferagent.utils import enforce_clean_git_tree, finalize_run_dir, init_run_dir, write_json
from waferagent.workloads_realism import ControlledRegimeConfig, generate_controlled_regime_graphs
from waferagent.controlled_workloads import (
    StrictControlledSharedKVConfig,
    generate_strict_controlled_shared_kv_graphs,
    strict_controlled_validation_rows,
)


def _ints(text: str) -> list[int]:
    return [int(x) for x in str(text).split(",") if x.strip()]


def _floats(text: str) -> list[float]:
    return [float(x) for x in str(text).split(",") if x.strip()]


def _csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _prefix_hit_rate(traces) -> float:
    seen: dict[str, set[str]] = {}
    hit = total = 0
    for tr in traces:
        for pid in tr.shared_prefix_ids:
            total += 1
            jobs = seen.setdefault(pid, set())
            if jobs and tr.job_id not in jobs:
                hit += 1
            jobs.add(tr.job_id)
    return hit / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", default="strict", choices=["strict", "legacy"])
    parser.add_argument("--reuse-group-size", default="1,2,4,8,16,32,64")
    parser.add_argument("--shared-prefix-tokens", default="512,2048,8192,32768")
    parser.add_argument("--private-suffix-tokens", default="512")
    parser.add_argument("--decode-tokens", default="32,128,512")
    parser.add_argument("--num-jobs", type=int, default=100)
    parser.add_argument("--num-agents-per-job", default="8")
    parser.add_argument("--fanin", action="store_true")
    parser.add_argument("--arrival-rate-jobs-per-s", default="2,4,8,16")
    parser.add_argument("--baselines", default="apc_like,pat_like_traffic_only,waferagent_latency_safe,waferagent_adaptive")
    parser.add_argument("--wafer-config", default="configs/wafer/wse_like.yaml")
    parser.add_argument("--shared-attention-cost-fit", required=True)
    parser.add_argument("--shared-attention-accounting", default="cohort_stage", choices=["stage_amortized", "cohort_stage", "per_member"])
    parser.add_argument("--out", default="results/round10_controlled_regime_sweep")
    parser.add_argument("--duration-source", default="synthetic", choices=["trace", "calibrated", "synthetic"])
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--clean-required", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    enforce_clean_git_tree(args.clean_required, args.allow_dirty)
    out = init_run_dir(args.out, {"run_type": "controlled_regime_sweep", **vars(args)})
    mesh = MeshConfig.from_yaml(args.wafer_config)
    rows = []
    opp_rows = []
    validation_rows = []
    policy_rows = []
    policy_summary_rows = []
    for group_size in _ints(args.reuse_group_size):
        for shared_tokens in _ints(args.shared_prefix_tokens):
            for private_tokens in _ints(args.private_suffix_tokens):
                for decode_tokens in _ints(args.decode_tokens):
                    for num_agents in _ints(args.num_agents_per_job):
                        if args.generator == "strict":
                            cfg = StrictControlledSharedKVConfig(
                                num_jobs=args.num_jobs,
                                reuse_group_size=group_size,
                                shared_prefix_tokens=shared_tokens,
                                private_suffix_tokens=private_tokens,
                                decode_tokens=decode_tokens,
                                num_agents_per_job=num_agents,
                                fanin=bool(args.fanin),
                                seed=args.seed,
                            )
                            graphs = generate_strict_controlled_shared_kv_graphs(cfg)
                            validation_rows.extend(strict_controlled_validation_rows(graphs, cfg))
                        else:
                            cfg = ControlledRegimeConfig(
                                num_jobs=args.num_jobs,
                                reuse_group_size=group_size,
                                shared_prefix_tokens=shared_tokens,
                                private_suffix_tokens=private_tokens,
                                decode_tokens=decode_tokens,
                                num_agents_per_job=num_agents,
                                seed=args.seed,
                            )
                            graphs = generate_controlled_regime_graphs(cfg)
                        run_id = f"controlled_g{group_size}_s{shared_tokens}_p{private_tokens}_d{decode_tokens}_a{num_agents}"
                        traces = collect_graph_traces(graphs, run_id, RunnerConfig(engine="synthetic", seed=args.seed))
                        hit_rate = _prefix_hit_rate(traces)
                        for rate in _floats(args.arrival_rate_jobs_per_s):
                            result = simulate_global(
                                traces,
                                mesh,
                                _csv(args.baselines),
                                ArrivalConfig(mode="poisson", rate_jobs_per_s=rate, seed=args.seed, max_jobs=args.num_jobs),
                                seed=args.seed,
                                duration_source=args.duration_source,
                                shared_attention_cost_fit=args.shared_attention_cost_fit,
                                shared_attention_accounting=args.shared_attention_accounting,
                            )
                            summary = result["global_simulation_summary"].copy()
                            summary["reuse_group_size"] = group_size
                            summary["shared_prefix_tokens"] = shared_tokens
                            summary["private_suffix_tokens"] = private_tokens
                            summary["decode_tokens"] = decode_tokens
                            summary["num_agents_per_job"] = num_agents
                            summary["cross_job_prefix_hit_rate_observed"] = hit_rate
                            summary["shared_attention_accounting_mode"] = args.shared_attention_accounting
                            rows.append(summary)
                            for name in ["policy_decisions", "policy_summary"]:
                                df = result.get(name, pd.DataFrame()).copy()
                                if not df.empty:
                                    df["reuse_group_size"] = group_size
                                    df["shared_prefix_tokens"] = shared_tokens
                                    df["private_suffix_tokens"] = private_tokens
                                    df["decode_tokens"] = decode_tokens
                                    df["num_agents_per_job"] = num_agents
                                    df["arrival_rate_jobs_per_s"] = rate
                                    if name == "policy_decisions":
                                        policy_rows.append(df)
                                    else:
                                        policy_summary_rows.append(df)
    combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    sim = out / "simulation"
    combined.to_csv(sim / "controlled_regime_summary.csv", index=False)
    regime_rows = []
    group_cols = ["reuse_group_size", "shared_prefix_tokens", "private_suffix_tokens", "decode_tokens", "num_agents_per_job", "arrival_rate_jobs_per_s"]
    for keys, sub in combined.groupby(group_cols):
        apc = sub[sub["baseline"] == "apc_like"]
        waf = sub[sub["baseline"].isin(["waferagent_adaptive", "waferagent_latency_safe"])]
        if "waferagent_adaptive" in set(waf["baseline"]):
            waf = waf[waf["baseline"] == "waferagent_adaptive"]
        pat = sub[sub["baseline"] == "pat_like_traffic_only"]
        if apc.empty or waf.empty:
            continue
        apc_jct = float(apc["jct_p99_ms"].mean())
        waf_jct = float(waf["jct_p99_ms"].mean())
        delta = (waf_jct - apc_jct) / max(1.0, apc_jct)
        reduction = float(waf["decode_kv_read_reduction_ratio"].mean())
        hit = float(waf["cross_job_prefix_hit_rate_observed"].mean())
        if delta < -0.05:
            label = "waferagent_latency_beneficial"
        elif reduction > 0.05:
            label = "traffic_reduction_only"
        else:
            label = "low_reuse_apc_better"
        regime_rows.append(
            {
                "reuse_group_size": keys[0],
                "shared_prefix_tokens": keys[1],
                "private_suffix_tokens": keys[2],
                "decode_tokens": keys[3],
                "num_agents_per_job": keys[4],
                "arrival_rate_jobs_per_s": keys[5],
                "cross_job_prefix_hit_rate_observed": hit,
                "shared_kv_read_reduction_ratio": reduction,
                "waferagent_vs_apc_jct_p99_delta_pct": delta,
                "apc_like_jct_p99_ms": apc_jct,
                "waferagent_jct_p99_ms": waf_jct,
                "regime_label": label,
            }
        )
        opp_rows.append(
            {
                "reuse_group_size": keys[0],
                "shared_prefix_tokens": keys[1],
                "private_suffix_tokens": keys[2],
                "decode_tokens": keys[3],
                "num_agents_per_job": keys[4],
                "arrival_rate_jobs_per_s": keys[5],
                "opportunity": hit * keys[1] * keys[3] * keys[4],
                "realized_speedup_vs_apc": apc_jct / max(1e-9, waf_jct),
            }
        )
    pd.DataFrame(regime_rows).to_csv(sim / "controlled_regime_classification.csv", index=False)
    pd.DataFrame(opp_rows).to_csv(sim / "opportunity_vs_realized_speedup.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(sim / "controlled_workload_validation.csv", index=False)
    (pd.concat(policy_rows, ignore_index=True) if policy_rows else pd.DataFrame()).to_csv(sim / "policy_decisions.csv", index=False)
    (pd.concat(policy_summary_rows, ignore_index=True) if policy_summary_rows else pd.DataFrame()).to_csv(sim / "policy_summary.csv", index=False)
    fig = out / "figures"
    if not combined.empty:
        line_from_csv(sim / "controlled_regime_summary.csv", "arrival_rate_jobs_per_s", "jct_p99_ms", fig / "fig_regime_map", hue="baseline")
    write_json(out / "model_selection.json", {"engine_used": "synthetic", "fallback_count": 0})
    finalize_run_dir(out)
    print(f"Controlled regime sweep complete: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
