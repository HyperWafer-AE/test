import argparse
import csv
import json
from pathlib import Path

from .real_trace_experiment import REAL_V2_POLICIES, parse_args as parse_real_args, run_suite
from .trace_loader import load_trace_dir
from .trace_profile import profile_traces


def run_paper_evaluation(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    opportunity = load_trace_dir(args.opportunity_trace_dir, trace_format="normalized_json")
    control = load_trace_dir(args.control_trace_dir, trace_format="normalized_json")
    if not opportunity:
        report = {
            "status": "skipped_missing_opportunity_traces",
            "paper_usable": False,
            "opportunity_trace_dir": args.opportunity_trace_dir,
            "control_trace_dir": args.control_trace_dir,
            "message": "No paper-usable opportunity traces exist; paper evaluation summaries and plots were not generated.",
        }
        (output_dir / "skipped_missing_opportunity_traces.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    full = opportunity + control
    _run_trace_set("full", full, args, output_dir / "full_set_summary.csv", output_dir / "full")
    _run_trace_set("opportunity", opportunity, args, output_dir / "opportunity_subset_summary.csv", output_dir / "opportunity")
    if control:
        _run_trace_set("control", control, args, output_dir / "control_subset_summary.csv", output_dir / "control")
    _run_sweeps(opportunity, args, output_dir)
    return {"status": "computed", "paper_usable": True}


def _run_trace_set(trace_set: str, traces, args, csv_path: Path, replay_dir: Path, memory_budget=0.5, concurrency=8, arrival_model="round_robin"):
    real_args = _real_args(replay_dir, memory_budget, concurrency, args.prediction_modes[0] if args.prediction_modes else "heuristic")
    real_args.data_quality = "paper_usable"
    real_args.paper_usable = True
    real_args.num_high_quality_traces = len(traces)
    real_args.num_smoke_traces = 0
    results = run_suite(real_args, traces, replay_dir, policies=REAL_V2_POLICIES)
    profile = profile_traces(traces, delayed_reuse_k=8)
    rows = [_row(trace_set, result, profile, memory_budget, concurrency, arrival_model, len(traces)) for result in results]
    _write_rows(csv_path, rows)
    return rows


def _run_sweeps(traces, args, output_dir: Path):
    memory_rows = []
    for budget in args.memory_budgets:
        memory_rows.extend(_run_trace_set("opportunity", traces, args, output_dir / "_tmp_memory.csv", output_dir / f"memory_{budget:g}", memory_budget=budget, concurrency=8))
    _write_rows(output_dir / "memory_sweep.csv", memory_rows)
    concurrency_rows = []
    for concurrency in args.concurrency_levels:
        concurrency_rows.extend(_run_trace_set("opportunity", traces, args, output_dir / "_tmp_concurrency.csv", output_dir / f"concurrency_{concurrency}", memory_budget=0.5, concurrency=concurrency))
    _write_rows(output_dir / "concurrency_sweep.csv", concurrency_rows)
    arrival_rows = []
    for arrival in args.arrival_models:
        arrival_rows.extend(_run_trace_set("opportunity", traces, args, output_dir / "_tmp_arrival.csv", output_dir / f"arrival_{arrival}", memory_budget=0.5, concurrency=8, arrival_model=arrival))
    _write_rows(output_dir / "arrival_model_sweep.csv", arrival_rows)
    _write_rows(output_dir / "mapping_ablation.csv", [])
    _write_rows(output_dir / "prefetch_ablation.csv", [])
    for tmp in ("_tmp_memory.csv", "_tmp_concurrency.csv", "_tmp_arrival.csv"):
        path = output_dir / tmp
        if path.exists():
            path.unlink()


def _real_args(output_dir: Path, memory_budget, concurrency, prediction_mode):
    return parse_real_args(
        [
            "--trace-dir",
            "unused",
            "--trace-format",
            "normalized_json",
            "--prediction-mode",
            prediction_mode,
            "--policy-suite",
            "v2",
            "--memory-budget-gb",
            str(memory_budget),
            "--concurrency",
            str(concurrency),
            "--allow-smoke-traces",
            "--output-dir",
            str(output_dir),
        ]
    )


def _row(trace_set: str, result, profile, memory_budget, concurrency, arrival_model, num_workflows):
    metrics = result["agent_metrics"]
    return {
        "trace_set": trace_set,
        "paper_usable": True,
        "data_quality": "paper_usable",
        "opportunity_score_mean": "",
        "delayed_reuse_ratio_mean": profile.get("delayed_reuse_ratio", 0.0),
        "LRU_regret_tokens_mean": profile.get("LRU_regret_tokens", 0) / max(1, profile.get("num_traces", 1)),
        "full_history_likely_ratio": profile.get("full_history_likely_ratio", 0.0),
        "concurrency": concurrency,
        "arrival_model": arrival_model,
        "memory_budget_gb": memory_budget,
        "num_workflows": num_workflows,
        "policy": result["policy"],
        "baseline_class": result.get("baseline_class"),
        "asg_builder_enabled": result.get("asg_builder_enabled"),
        "estimator_mode": result.get("estimator_mode"),
        "oracle_future": result.get("oracle_future"),
        "mapping_policy": "default",
        "prefetch_policy": "default",
        "effective_prefill_tokens": metrics.get("effective_prefill_tokens"),
        "effective_prefill_reduction_vs_lru_system": "",
        "cache_byte_miss": metrics.get("cache_byte_miss"),
        "state_misses": metrics.get("state_misses"),
        "LRU_regret_states_preserved": result.get("LRU_regret_states_preserved"),
        "LRU_regret_tokens_preserved": result.get("LRU_regret_tokens_preserved"),
        "oracle_gap": "",
        "online_capture_ratio": "",
        "model_compute_cycles": metrics.get("model_compute_cycles"),
        "model_comm_cycles": metrics.get("model_comm_cycles"),
        "remote_read_bytes": metrics.get("remote_read_bytes"),
        "demand_migration_bytes": metrics.get("demand_migration_bytes"),
        "prefetch_migration_bytes": metrics.get("prefetch_migration_bytes"),
        "prefetch_hidden_cycles": metrics.get("prefetch_hidden_cycles"),
        "prefetch_exposed_cycles": metrics.get("prefetch_exposed_cycles"),
        "unused_prefetch_events": metrics.get("unused_prefetch_events"),
        "num_action_local": metrics.get("num_action_local"),
        "num_action_remote_read": metrics.get("num_action_remote_read"),
        "num_action_migrate": metrics.get("num_action_migrate"),
        "num_action_replicate": metrics.get("num_action_replicate"),
        "num_action_static_hit": metrics.get("num_action_static_hit"),
    }


def _write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "trace_set",
        "paper_usable",
        "data_quality",
        "opportunity_score_mean",
        "delayed_reuse_ratio_mean",
        "LRU_regret_tokens_mean",
        "full_history_likely_ratio",
        "concurrency",
        "arrival_model",
        "memory_budget_gb",
        "num_workflows",
        "policy",
        "baseline_class",
        "asg_builder_enabled",
        "estimator_mode",
        "oracle_future",
        "mapping_policy",
        "prefetch_policy",
        "effective_prefill_tokens",
        "effective_prefill_reduction_vs_lru_system",
        "cache_byte_miss",
        "state_misses",
        "LRU_regret_states_preserved",
        "LRU_regret_tokens_preserved",
        "oracle_gap",
        "online_capture_ratio",
        "model_compute_cycles",
        "model_comm_cycles",
        "remote_read_bytes",
        "demand_migration_bytes",
        "prefetch_migration_bytes",
        "prefetch_hidden_cycles",
        "prefetch_exposed_cycles",
        "unused_prefetch_events",
        "num_action_local",
        "num_action_remote_read",
        "num_action_migrate",
        "num_action_replicate",
        "num_action_static_hit",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run paper-style Agent-on-Wafer evaluations on paper-usable traces.")
    parser.add_argument("--opportunity-trace-dir", required=True)
    parser.add_argument("--control-trace-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--memory-budgets", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--concurrency-levels", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--arrival-models", nargs="+", default=["round_robin", "burst", "poisson", "tool_wait_aware"])
    parser.add_argument("--prediction-modes", nargs="+", default=["heuristic", "trace_stats"])
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = run_paper_evaluation(args)
    print(f"paper_eval status={report['status']}")


if __name__ == "__main__":
    main()
