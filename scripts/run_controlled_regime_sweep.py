#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
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
    strict_controlled_validation_summary_rows,
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


def _split_name(keys: tuple[object, ...], seed: int) -> str:
    text = "|".join(map(str, keys)) + f"|seed={seed}"
    value = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % 100
    if value < 60:
        return "train"
    if value < 80:
        return "validation"
    return "test"


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
    parser.add_argument("--arrival-mode", default="poisson", choices=["poisson", "closed_loop", "burst", "replay"])
    parser.add_argument("--arrival-rate-jobs-per-s", default="2,4,8,16")
    parser.add_argument("--baselines", default="apc_like,apc_like_with_affinity_placement,pat_like_traffic_only,pat_like_with_affinity_placement,waferagent_latency_safe,waferagent_no_decode_cohort,waferagent_no_affinity_placement,waferagent_no_kv_sharing,waferagent_no_shared_kv_placement,waferagent_adaptive")
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
    validation_node_rows = []
    policy_rows = []
    policy_assignment_rows = []
    policy_stage_rows = []
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
                            validation_rows.extend(strict_controlled_validation_summary_rows(graphs, cfg))
                            validation_node_rows.extend(strict_controlled_validation_rows(graphs, cfg))
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
                                ArrivalConfig(mode=args.arrival_mode, rate_jobs_per_s=rate, seed=args.seed, max_jobs=args.num_jobs),
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
                            for name in ["policy_decisions", "policy_assignments", "policy_effective_stage_map", "policy_summary"]:
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
                                    elif name == "policy_assignments":
                                        policy_assignment_rows.append(df)
                                    elif name == "policy_effective_stage_map":
                                        policy_stage_rows.append(df)
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
    oracle_rows = []
    attribution_rows = []
    attribution_delta_rows = []
    if not combined.empty:
        for keys, sub in combined.groupby(group_cols):
            vals = {str(row["baseline"]): row for _, row in sub.iterrows()}
            apc = vals.get("apc_like")
            pat = vals.get("pat_like_traffic_only")
            waf = vals.get("waferagent_latency_safe")
            adaptive = vals.get("waferagent_adaptive")
            candidates = {k: v for k, v in {"apc_like": apc, "pat_like_traffic_only": pat, "waferagent_latency_safe": waf}.items() if v is not None}
            if not candidates or apc is None or adaptive is None:
                continue
            oracle_policy, oracle_row = min(candidates.items(), key=lambda kv: float(kv[1]["jct_p99_ms"]))
            split = _split_name(tuple(keys[:5]), args.seed)
            apc_jct = float(apc["jct_p99_ms"])
            adaptive_jct = float(adaptive["jct_p99_ms"])
            oracle_jct = float(oracle_row["jct_p99_ms"])
            adaptive_speedup = (apc_jct - adaptive_jct) / max(1.0, apc_jct)
            oracle_speedup = (apc_jct - oracle_jct) / max(1.0, apc_jct)
            oracle_rows.append(
                {
                    "reuse_group_size": keys[0],
                    "shared_prefix_tokens": keys[1],
                    "private_suffix_tokens": keys[2],
                    "decode_tokens": keys[3],
                    "num_agents_per_job": keys[4],
                    "arrival_rate_jobs_per_s": keys[5],
                    "split": split,
                    "oracle_best_policy": oracle_policy,
                    "oracle_best_jct_p99_ms": oracle_jct,
                    "apc_like_jct_p99_ms": apc_jct,
                    "waferagent_adaptive_jct_p99_ms": adaptive_jct,
                    "adaptive_regret_ms": max(0.0, adaptive_jct - oracle_jct),
                    "adaptive_slowdown_vs_apc": (adaptive_jct - apc_jct) / max(1.0, apc_jct),
                    "oracle_speedup_vs_apc": oracle_speedup,
                    "adaptive_speedup_vs_apc": adaptive_speedup,
                    "oracle_beneficial": oracle_speedup >= 0.05 and oracle_policy != "apc_like",
                    "adaptive_beneficial": adaptive_speedup >= 0.05,
                    "adaptive_non_worse_than_apc_within_5pct": adaptive_jct <= 1.05 * apc_jct,
                    "adaptive_within_5pct_of_oracle": adaptive_jct <= 1.05 * oracle_jct,
                }
            )

            full = waf
            no_decode = vals.get("waferagent_no_decode_cohort")
            no_aff = vals.get("waferagent_no_affinity_placement")
            no_kv = vals.get("waferagent_no_kv_sharing")
            no_place = vals.get("waferagent_no_shared_kv_placement")
            apc_aff = vals.get("apc_like_with_affinity_placement")
            pat_aff = vals.get("pat_like_with_affinity_placement")

            def _delta(base, variant, metric):
                if base is None or variant is None or metric not in base or metric not in variant:
                    return 0.0
                return (float(variant[metric]) - float(base[metric])) / max(1.0, abs(float(base[metric])))

            shared_delta = _delta(full, no_decode, "decode_shared_kv_read_bytes")
            placement_delta = max(_delta(full, no_aff, "mesh_total_traffic_bytes"), _delta(full, no_aff, "jct_p99_ms"))
            prefill_delta = 0.0
            if full is not None and no_kv is not None and "shared_prefill_compute_ms_saved" in full:
                prefill_delta = (float(full["shared_prefill_compute_ms_saved"]) - float(no_kv.get("shared_prefill_compute_ms_saved", 0.0))) / max(1.0, abs(float(full["shared_prefill_compute_ms_saved"])))
            shared_place_delta = max(_delta(full, no_place, "mesh_total_traffic_bytes"), _delta(full, no_place, "jct_p99_ms"))
            apc_affinity_delta = _delta(apc, apc_aff, "jct_p99_ms")
            pat_affinity_delta = _delta(pat, pat_aff, "jct_p99_ms")
            causes = []
            if shared_delta > 0.05:
                causes.append("shared_kv_decode_benefit")
            if placement_delta > 0.05 or shared_place_delta > 0.05 or apc_affinity_delta < -0.05 or pat_affinity_delta < -0.05:
                causes.append("placement_mesh_benefit")
            if prefill_delta > 0.5:
                causes.append("prefill_cache_benefit")
            if len(causes) > 1:
                primary = "mixed_benefit"
            elif causes:
                primary = causes[0]
            else:
                primary = "unexplained_benefit" if adaptive_speedup >= 0.05 else "not_beneficial"
            base_row = {
                "reuse_group_size": keys[0],
                "shared_prefix_tokens": keys[1],
                "private_suffix_tokens": keys[2],
                "decode_tokens": keys[3],
                "num_agents_per_job": keys[4],
                "arrival_rate_jobs_per_s": keys[5],
                "primary_benefit_source": primary,
                "prefill_saving_delta": prefill_delta,
                "shared_kv_read_reduction_delta": shared_delta,
                "placement_mesh_delta": placement_delta,
                "shared_kv_placement_delta": shared_place_delta,
                "apc_affinity_jct_delta": apc_affinity_delta,
                "pat_affinity_jct_delta": pat_affinity_delta,
                "jct_p99_delta_vs_apc": adaptive_speedup,
            }
            attribution_rows.append(base_row)
            for metric, value in {
                "prefill_saving_delta": prefill_delta,
                "shared_kv_read_reduction_delta": shared_delta,
                "placement_mesh_delta": placement_delta,
                "shared_kv_placement_delta": shared_place_delta,
            }.items():
                attribution_delta_rows.append({**base_row, "metric": metric, "delta": value})

    oracle_df = pd.DataFrame(oracle_rows)
    oracle_df.to_csv(sim / "policy_oracle_labels.csv", index=False)
    for split in ["train", "validation", "test"]:
        oracle_df[oracle_df.get("split", pd.Series(dtype=str)) == split].to_csv(sim / f"policy_{split}_regimes.csv", index=False)
    eval_rows = []
    for split in ["all", "train", "validation", "test"]:
        sub = oracle_df if split == "all" else oracle_df[oracle_df["split"] == split]
        if sub.empty:
            continue
        regret = pd.to_numeric(sub["adaptive_regret_ms"], errors="coerce")
        slowdown = pd.to_numeric(sub["adaptive_slowdown_vs_apc"], errors="coerce")
        actual_beneficial = sub["oracle_beneficial"].astype(bool)
        pred_beneficial = sub["adaptive_beneficial"].astype(bool)
        true_positive = int((actual_beneficial & pred_beneficial).sum())
        eval_rows.append(
            {
                "split": split,
                "num_regimes": len(sub),
                "selection_accuracy_vs_oracle_best": float(sub["adaptive_within_5pct_of_oracle"].astype(bool).mean()),
                "mean_regret_ms": float(regret.mean()),
                "p95_regret_ms": float(regret.quantile(0.95)),
                "max_slowdown_vs_apc": float(slowdown.max()),
                "fraction_non_worse_than_apc_within_5pct": float(sub["adaptive_non_worse_than_apc_within_5pct"].astype(bool).mean()),
                "beneficial_regime_recall": true_positive / max(1, int(actual_beneficial.sum())),
                "beneficial_regime_precision": true_positive / max(1, int(pred_beneficial.sum())),
                "heldout_beneficial_regimes": int((pred_beneficial & (sub["adaptive_speedup_vs_apc"] >= 0.05)).sum()),
                "pass": bool(
                    float(sub["adaptive_non_worse_than_apc_within_5pct"].astype(bool).mean()) >= 0.95
                    and float(slowdown.max()) <= 0.10
                    and int((pred_beneficial & (sub["adaptive_speedup_vs_apc"] >= 0.05)).sum()) >= 1
                ),
            }
        )
    pd.DataFrame(eval_rows).to_csv(sim / "policy_prediction_eval.csv", index=False)
    pd.DataFrame(attribution_rows).to_csv(sim / "mechanism_attribution_summary.csv", index=False)
    pd.DataFrame(attribution_delta_rows).to_csv(sim / "mechanism_attribution_delta.csv", index=False)
    validation_df = pd.DataFrame(validation_rows)
    validation_nodes_df = pd.DataFrame(validation_node_rows)
    validation_df.to_csv(sim / "controlled_workload_validation.csv", index=False)
    validation_nodes_df.to_csv(sim / "controlled_workload_validation_nodes.csv", index=False)
    (pd.concat(policy_rows, ignore_index=True) if policy_rows else pd.DataFrame()).to_csv(sim / "policy_decisions.csv", index=False)
    (pd.concat(policy_assignment_rows, ignore_index=True) if policy_assignment_rows else pd.DataFrame()).to_csv(sim / "policy_assignments.csv", index=False)
    (pd.concat(policy_stage_rows, ignore_index=True) if policy_stage_rows else pd.DataFrame()).to_csv(sim / "policy_effective_stage_map.csv", index=False)
    (pd.concat(policy_summary_rows, ignore_index=True) if policy_summary_rows else pd.DataFrame()).to_csv(sim / "policy_summary.csv", index=False)
    fig = out / "figures"
    if not combined.empty:
        line_from_csv(sim / "controlled_regime_summary.csv", "arrival_rate_jobs_per_s", "jct_p99_ms", fig / "fig_regime_map", hue="baseline")
    write_json(out / "model_selection.json", {"engine_used": "synthetic", "fallback_count": 0})
    finalize_run_dir(out)
    print(f"Controlled regime sweep complete: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
