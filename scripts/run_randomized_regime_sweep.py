#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import pandas as pd

from waferagent.arrival import ArrivalConfig
from waferagent.controlled_workloads import (
    StrictControlledSharedKVConfig,
    generate_strict_controlled_shared_kv_graphs,
    strict_controlled_validation_rows,
    strict_controlled_validation_summary_rows,
)
from waferagent.global_simulator import simulate_global
from waferagent.llm_runner import RunnerConfig
from waferagent.mesh import MeshConfig
from waferagent.trace_collector import collect_graph_traces
from waferagent.trace_schema import TraceRecord
from waferagent.utils import enforce_clean_git_tree, finalize_run_dir, init_run_dir, write_json


DEFAULT_BASELINES = (
    "apc_like,apc_like_global_orchestrator,pat_like_traffic_only,pat_like_global_orchestrator,"
    "waferagent_no_kv_sharing,waferagent_no_decode_cohort,waferagent_no_affinity_placement,"
    "waferagent_no_shared_kv_placement,waferagent_latency_safe,waferagent_adaptive"
)


def _csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _log_int(rng: random.Random, lo: int, hi: int) -> int:
    value = int(round(math.exp(rng.uniform(math.log(lo), math.log(hi)))))
    return max(lo, min(hi, value))


def _split_for_index(idx: int, num_configs: int) -> str:
    train_cut = int(round(num_configs * 0.60))
    val_cut = int(round(num_configs * 0.80))
    if idx < train_cut:
        return "train"
    if idx < val_cut:
        return "validation"
    return "test"


def _prefix_reuse_stats(traces: list[TraceRecord]) -> dict[str, float]:
    prefix_jobs: dict[str, set[str]] = {}
    prefix_nodes_by_job: dict[tuple[str, str], int] = {}
    total_uses = 0
    for tr in traces:
        for pid in tr.shared_prefix_ids:
            total_uses += 1
            prefix_jobs.setdefault(pid, set()).add(tr.job_id)
            key = (pid, tr.job_id)
            prefix_nodes_by_job[key] = prefix_nodes_by_job.get(key, 0) + 1
    if total_uses <= 0:
        return {
            "cross_job_shared_prefix_reuse_rate": 0.0,
            "intra_job_shared_prefix_reuse_rate": 0.0,
        }
    cross_uses = sum(1 for tr in traces for pid in tr.shared_prefix_ids if len(prefix_jobs.get(pid, set())) > 1)
    intra_uses = sum(1 for tr in traces for pid in tr.shared_prefix_ids if prefix_nodes_by_job.get((pid, tr.job_id), 0) > 1)
    return {
        "cross_job_shared_prefix_reuse_rate": cross_uses / total_uses,
        "intra_job_shared_prefix_reuse_rate": intra_uses / total_uses,
    }


def _value(vals: dict[str, pd.Series], baseline: str, metric: str) -> float | None:
    row = vals.get(baseline)
    if row is None or metric not in row:
        return None
    try:
        return float(row[metric])
    except Exception:
        return None


def _safe_delta(new: float | None, base: float | None) -> float:
    if new is None or base is None:
        return 0.0
    return (new - base) / max(1.0, abs(base))


def _benefit_pct(component: float, total: float) -> float:
    return 0.0 if total <= 0 else max(0.0, component) / max(1e-9, total) * 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-configs", type=int, default=160)
    parser.add_argument("--num-jobs", type=int, default=20)
    parser.add_argument("--wafer-config", default="configs/wafer/wse_like.yaml")
    parser.add_argument("--baselines", default=DEFAULT_BASELINES)
    parser.add_argument("--shared-attention-cost-fit", required=True)
    parser.add_argument("--shared-attention-accounting", default="cohort_stage", choices=["stage_amortized", "cohort_stage", "per_member"])
    parser.add_argument("--out", default="results/round13_randomized_regime_sweep")
    parser.add_argument("--duration-source", default="synthetic", choices=["trace", "calibrated", "synthetic"])
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--clean-required", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    enforce_clean_git_tree(args.clean_required, args.allow_dirty)
    out = init_run_dir(args.out, {"run_type": "randomized_regime_sweep", **vars(args)})
    mesh = MeshConfig.from_yaml(args.wafer_config)
    rng = random.Random(args.seed)
    configs: list[dict[str, object]] = []
    for config_id in range(args.num_configs):
        configs.append(
            {
                "config_id": config_id,
                "split": _split_for_index(config_id, args.num_configs),
                "reuse_group_size": rng.randint(1, 128),
                "shared_prefix_tokens": _log_int(rng, 256, 65536),
                "private_suffix_tokens": _log_int(rng, 64, 4096),
                "decode_tokens": _log_int(rng, 16, 2048),
                "num_agents_per_job": rng.choice([2, 4, 8, 16, 32]),
                "arrival_rate_jobs_per_s": rng.choice([1.0, 2.0, 4.0, 8.0, 16.0, 32.0]),
                "fanin": rng.choice([False, True]),
            }
        )

    summary_parts: list[pd.DataFrame] = []
    validation_rows: list[dict[str, object]] = []
    validation_node_rows: list[dict[str, object]] = []
    policy_rows: list[pd.DataFrame] = []
    policy_assignment_rows: list[pd.DataFrame] = []
    policy_stage_rows: list[pd.DataFrame] = []
    policy_summary_rows: list[pd.DataFrame] = []
    baselines = _csv(args.baselines)
    for cfg_row in configs:
        cfg = StrictControlledSharedKVConfig(
            num_jobs=args.num_jobs,
            reuse_group_size=int(cfg_row["reuse_group_size"]),
            shared_prefix_tokens=int(cfg_row["shared_prefix_tokens"]),
            private_suffix_tokens=int(cfg_row["private_suffix_tokens"]),
            decode_tokens=int(cfg_row["decode_tokens"]),
            num_agents_per_job=int(cfg_row["num_agents_per_job"]),
            fanin=bool(cfg_row["fanin"]),
            seed=args.seed + int(cfg_row["config_id"]),
        )
        graphs = generate_strict_controlled_shared_kv_graphs(cfg)
        validation_rows.extend({**row, **cfg_row} for row in strict_controlled_validation_summary_rows(graphs, cfg))
        validation_node_rows.extend({**row, **cfg_row} for row in strict_controlled_validation_rows(graphs, cfg))
        run_id = f"randomized_cfg_{cfg_row['config_id']}"
        traces = collect_graph_traces(graphs, run_id, RunnerConfig(engine="synthetic", seed=args.seed))
        reuse = _prefix_reuse_stats(traces)
        result = simulate_global(
            traces,
            mesh,
            baselines,
            ArrivalConfig(
                mode="poisson",
                rate_jobs_per_s=float(cfg_row["arrival_rate_jobs_per_s"]),
                seed=args.seed,
                max_jobs=args.num_jobs,
            ),
            seed=args.seed,
            duration_source=args.duration_source,
            shared_attention_cost_fit=args.shared_attention_cost_fit,
            shared_attention_accounting=args.shared_attention_accounting,
        )
        summary = result["global_simulation_summary"].copy()
        for key, value in {**cfg_row, **reuse}.items():
            summary[key] = value
        summary_parts.append(summary)
        for name, target in [
            ("policy_decisions", policy_rows),
            ("policy_assignments", policy_assignment_rows),
            ("policy_effective_stage_map", policy_stage_rows),
            ("policy_summary", policy_summary_rows),
        ]:
            df = result.get(name, pd.DataFrame()).copy()
            if not df.empty:
                for key, value in cfg_row.items():
                    df[key] = value
                target.append(df)

    sim = out / "simulation"
    combined = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
    combined.to_csv(sim / "randomized_regime_summary.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(sim / "controlled_workload_validation.csv", index=False)
    pd.DataFrame(validation_node_rows).to_csv(sim / "controlled_workload_validation_nodes.csv", index=False)
    (pd.concat(policy_rows, ignore_index=True) if policy_rows else pd.DataFrame()).to_csv(sim / "policy_decisions.csv", index=False)
    (pd.concat(policy_assignment_rows, ignore_index=True) if policy_assignment_rows else pd.DataFrame()).to_csv(sim / "policy_assignments.csv", index=False)
    (pd.concat(policy_stage_rows, ignore_index=True) if policy_stage_rows else pd.DataFrame()).to_csv(sim / "policy_effective_stage_map.csv", index=False)
    (pd.concat(policy_summary_rows, ignore_index=True) if policy_summary_rows else pd.DataFrame()).to_csv(sim / "policy_summary.csv", index=False)

    group_cols = ["config_id", "split"]
    oracle_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    fair_rows: list[dict[str, object]] = []
    fair_delta_rows: list[dict[str, object]] = []
    attr_rows: list[dict[str, object]] = []
    attr_delta_rows: list[dict[str, object]] = []
    for (config_id, split), sub in combined.groupby(group_cols):
        vals = {str(row["baseline"]): row for _, row in sub.iterrows()}
        apc = vals.get("apc_like")
        pat = vals.get("pat_like_traffic_only")
        apc_go = vals.get("apc_like_global_orchestrator")
        pat_go = vals.get("pat_like_global_orchestrator")
        waf = vals.get("waferagent_latency_safe")
        adaptive = vals.get("waferagent_adaptive")
        candidates = {k: v for k, v in {"apc_like": apc, "pat_like_traffic_only": pat, "waferagent_latency_safe": waf}.items() if v is not None}
        if not candidates or apc is None or adaptive is None:
            continue
        oracle_policy, oracle_row = min(candidates.items(), key=lambda kv: float(kv[1]["jct_p99_ms"]))
        apc_jct = float(apc["jct_p99_ms"])
        adaptive_jct = float(adaptive["jct_p99_ms"])
        oracle_jct = float(oracle_row["jct_p99_ms"])
        adaptive_speedup = (apc_jct - adaptive_jct) / max(1.0, apc_jct)
        oracle_speedup = (apc_jct - oracle_jct) / max(1.0, apc_jct)
        base = {k: vals.get("waferagent_adaptive", apc).get(k, "") for k in [
            "reuse_group_size",
            "shared_prefix_tokens",
            "private_suffix_tokens",
            "decode_tokens",
            "num_agents_per_job",
            "arrival_rate_jobs_per_s",
            "fanin",
            "cross_job_shared_prefix_reuse_rate",
            "intra_job_shared_prefix_reuse_rate",
        ]}
        oracle_rows.append(
            {
                "config_id": config_id,
                "split": split,
                **base,
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
        reduction = float(adaptive.get("decode_kv_read_reduction_ratio", 0.0))
        label = "waferagent_latency_beneficial" if adaptive_speedup >= 0.05 else ("traffic_reduction_only" if reduction > 0.05 else "low_reuse_apc_better")
        class_rows.append(
            {
                "config_id": config_id,
                "split": split,
                **base,
                "shared_decode_cohort_reuse_rate": reduction,
                "placement_mesh_reduction_ratio": -_safe_delta(_value(vals, "waferagent_latency_safe", "mesh_total_traffic_bytes"), _value(vals, "waferagent_no_affinity_placement", "mesh_total_traffic_bytes")),
                "scheduler_parallelism_gain": -_safe_delta(_value(vals, "apc_like_global_orchestrator", "jct_p99_ms"), _value(vals, "apc_like", "jct_p99_ms")),
                "waferagent_vs_apc_jct_p99_delta_pct": (adaptive_jct - apc_jct) / max(1.0, apc_jct),
                "regime_label": label,
            }
        )
        for baseline in ["apc_like", "apc_like_global_orchestrator", "pat_like_traffic_only", "pat_like_global_orchestrator", "waferagent_adaptive"]:
            row = vals.get(baseline)
            if row is not None:
                fair_rows.append({"config_id": config_id, "split": split, **base, "baseline": baseline, "jct_p99_ms": row["jct_p99_ms"], "mesh_total_traffic_bytes": row.get("mesh_total_traffic_bytes", 0.0)})
        for name, new, old in [
            ("apc_global_orchestrator_vs_engine", _value(vals, "apc_like_global_orchestrator", "jct_p99_ms"), _value(vals, "apc_like", "jct_p99_ms")),
            ("pat_global_orchestrator_vs_engine", _value(vals, "pat_like_global_orchestrator", "jct_p99_ms"), _value(vals, "pat_like_traffic_only", "jct_p99_ms")),
            ("adaptive_vs_apc_global_orchestrator", _value(vals, "waferagent_adaptive", "jct_p99_ms"), _value(vals, "apc_like_global_orchestrator", "jct_p99_ms")),
        ]:
            fair_delta_rows.append({"config_id": config_id, "split": split, **base, "comparison": name, "jct_delta_pct": _safe_delta(new, old)})

        total_benefit = max(0.0, apc_jct - adaptive_jct)
        components = {
            "benefit_scheduler_orchestrator": max(0.0, apc_jct - (_value(vals, "apc_like_global_orchestrator", "jct_p99_ms") or apc_jct)),
            "benefit_decode_cohort": max(0.0, (_value(vals, "apc_like_global_orchestrator", "jct_p99_ms") or apc_jct) - (_value(vals, "pat_like_global_orchestrator", "jct_p99_ms") or apc_jct)),
            "benefit_affinity_placement": max(0.0, (_value(vals, "waferagent_no_affinity_placement", "jct_p99_ms") or adaptive_jct) - (_value(vals, "waferagent_latency_safe", "jct_p99_ms") or adaptive_jct)),
            "benefit_shared_kv_placement": max(0.0, (_value(vals, "waferagent_no_shared_kv_placement", "jct_p99_ms") or adaptive_jct) - (_value(vals, "waferagent_latency_safe", "jct_p99_ms") or adaptive_jct)),
            "benefit_prefill_cache": max(0.0, (_value(vals, "waferagent_no_kv_sharing", "jct_p99_ms") or adaptive_jct) - (_value(vals, "waferagent_latency_safe", "jct_p99_ms") or adaptive_jct)),
            "benefit_adaptive_selection": max(0.0, (_value(vals, "waferagent_latency_safe", "jct_p99_ms") or adaptive_jct) - adaptive_jct),
        }
        comp_sum = sum(components.values())
        scale = min(1.0, total_benefit / comp_sum) if comp_sum > 0 and total_benefit > 0 else 1.0
        pct_components = {k + "_pct": _benefit_pct(v * scale, total_benefit) for k, v in components.items()}
        unexplained = 0.0 if total_benefit <= 0 else max(0.0, 100.0 - sum(pct_components.values()))
        primary = "not_beneficial"
        if total_benefit > 0:
            primary = max(pct_components, key=pct_components.get).replace("benefit_", "").replace("_pct", "")
            if unexplained > 80.0:
                primary = "unexplained"
        attr = {
            "config_id": config_id,
            "split": split,
            **base,
            "jct_p99_delta_vs_apc": adaptive_speedup,
            **pct_components,
            "unexplained_pct": unexplained,
            "primary_benefit_source": primary,
        }
        attr_rows.append(attr)
        for metric, value in pct_components.items():
            attr_delta_rows.append({**attr, "metric": metric, "value_pct": value})
        attr_delta_rows.append({**attr, "metric": "unexplained_pct", "value_pct": unexplained})

    oracle_df = pd.DataFrame(oracle_rows)
    oracle_df.to_csv(sim / "randomized_policy_oracle_labels.csv", index=False)
    pd.DataFrame(class_rows).to_csv(sim / "randomized_regime_classification.csv", index=False)
    pd.DataFrame(fair_rows).to_csv(sim / "fair_baseline_summary.csv", index=False)
    pd.DataFrame(fair_delta_rows).to_csv(sim / "fair_baseline_delta.csv", index=False)
    pd.DataFrame(attr_rows).to_csv(sim / "mechanism_attribution_summary.csv", index=False)
    pd.DataFrame(attr_delta_rows).to_csv(sim / "mechanism_attribution_delta.csv", index=False)
    for split in ["train", "validation", "test"]:
        oracle_df[oracle_df.get("split", pd.Series(dtype=str)) == split].to_csv(sim / f"randomized_policy_{split}_regimes.csv", index=False)
    eval_rows: list[dict[str, object]] = []
    for split in ["all", "train", "validation", "test"]:
        sub = oracle_df if split == "all" else oracle_df[oracle_df["split"] == split]
        if sub.empty:
            continue
        regret = pd.to_numeric(sub["adaptive_regret_ms"], errors="coerce")
        slowdown = pd.to_numeric(sub["adaptive_slowdown_vs_apc"], errors="coerce")
        actual_beneficial = sub["oracle_beneficial"].astype(bool)
        pred_beneficial = sub["adaptive_beneficial"].astype(bool)
        tp = int((actual_beneficial & pred_beneficial).sum())
        eval_rows.append(
            {
                "split": split,
                "num_regimes": len(sub),
                "selection_accuracy_vs_oracle_best": float(sub["adaptive_within_5pct_of_oracle"].astype(bool).mean()),
                "mean_regret_ms": float(regret.mean()),
                "p95_regret_ms": float(regret.quantile(0.95)),
                "max_slowdown_vs_apc": float(slowdown.max()),
                "fraction_non_worse_than_apc_within_5pct": float(sub["adaptive_non_worse_than_apc_within_5pct"].astype(bool).mean()),
                "beneficial_regime_recall": tp / max(1, int(actual_beneficial.sum())),
                "beneficial_regime_precision": tp / max(1, int(pred_beneficial.sum())),
                "beneficial_test_regimes": int((pred_beneficial & (sub["adaptive_speedup_vs_apc"] >= 0.05)).sum()),
                "pass": bool(
                    float(sub["adaptive_non_worse_than_apc_within_5pct"].astype(bool).mean()) >= 0.95
                    and float(slowdown.max()) <= 0.10
                    and (tp / max(1, int(actual_beneficial.sum()))) >= 0.80
                    and (tp / max(1, int(pred_beneficial.sum()))) >= 0.70
                    and int((pred_beneficial & (sub["adaptive_speedup_vs_apc"] >= 0.05)).sum()) >= (5 if split == "test" else 1)
                ),
            }
        )
    pd.DataFrame(eval_rows).to_csv(sim / "randomized_policy_prediction_eval.csv", index=False)
    write_json(out / "model_selection.json", {"engine_used": "synthetic", "fallback_count": 0})
    finalize_run_dir(out)
    print(f"Randomized regime sweep complete: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
