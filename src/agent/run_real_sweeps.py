import argparse
import csv
import json
from pathlib import Path

from .real_trace_experiment import (
    _metric_for,
    _oracle_gap,
    load_traces_from_args,
    parse_args as parse_real_args,
    run_suite,
    write_missing_report,
)


MEMORY_BUDGETS_GB = (0.25, 0.5, 1.0, 2.0)
CONCURRENCY_VALUES = (1, 2, 4, 8, 16)


def build_real_args(args, output_dir: Path):
    argv = [
        "--trace-dir",
        args.trace_dir,
        "--trace-format",
        args.trace_format,
        "--prediction-mode",
        args.prediction_mode,
        "--policy-suite",
        "v2",
        "--memory-budget-gb",
        str(args.base_memory_budget_gb),
        "--concurrency",
        str(args.base_concurrency),
        "--output-dir",
        str(output_dir),
    ]
    if args.max_traces is not None:
        argv += ["--max-traces", str(args.max_traces)]
    if args.min_turns:
        argv += ["--min-turns", str(args.min_turns)]
    if args.filter_success != "all":
        argv += ["--filter-success", args.filter_success]
    if args.train_trace_dir:
        argv += ["--train-trace-dir", args.train_trace_dir]
    if args.reject_accumulated_fallback:
        argv.append("--reject-accumulated-fallback")
    if args.enable_common_observation_compression:
        argv.append("--enable-common-observation-compression")
    if args.enable_asg_observation_compression:
        argv.append("--enable-asg-observation-compression")
    return parse_real_args(argv)


def rows_for_results(sweep_type: str, setting, results: list[dict]) -> list[dict]:
    oracle_eff = _metric_for(results, "asg-retention-v2-oracle", "effective_prefill_tokens")
    lru_eff = _metric_for(results, "lru-system", "effective_prefill_tokens")
    rows = []
    for result in results:
        metrics = result["agent_metrics"]
        rows.append(
            {
                "sweep_type": sweep_type,
                "setting": setting,
                "policy": result["policy"],
                "effective_prefill_tokens": metrics["effective_prefill_tokens"],
                "cache_hit_ratio": metrics["cache_hit_ratio"],
                "state_misses": metrics["state_misses"],
                "LRU_regret_states_preserved": result["LRU_regret_states_preserved"],
                "oracle_gap": _oracle_gap(metrics["effective_prefill_tokens"], lru_eff, oracle_eff),
                "model_compute_cycles": metrics["model_compute_cycles"],
                "model_comm_cycles": metrics["model_comm_cycles"],
            }
        )
    return rows


def write_rows(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sweep_type",
        "setting",
        "policy",
        "effective_prefill_tokens",
        "cache_hit_ratio",
        "state_misses",
        "LRU_regret_states_preserved",
        "oracle_gap",
        "model_compute_cycles",
        "model_comm_cycles",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_sweep(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_args = build_real_args(args, output_dir / "_base")
    traces = load_traces_from_args(base_args)
    if not traces:
        report_path = write_missing_report(base_args, output_dir)
        (output_dir / "missing_report.json").write_text(
            json.dumps(
                {
                    "status": "missing_real_trace_data",
                    "message": "No loadable real traces were found. Sweep CSVs were not generated.",
                    "real_trace_missing_report": str(report_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return [], []

    memory_rows = []
    for memory_budget in MEMORY_BUDGETS_GB:
        sweep_args = build_real_args(args, output_dir / f"memory_{memory_budget:g}")
        sweep_args.memory_budget_gb = memory_budget
        results = run_suite(sweep_args, traces, Path(sweep_args.output_dir))
        memory_rows.extend(rows_for_results("memory_budget_gb", memory_budget, results))
    write_rows(output_dir / "memory_sweep.csv", memory_rows)

    concurrency_rows = []
    for concurrency in CONCURRENCY_VALUES:
        sweep_args = build_real_args(args, output_dir / f"concurrency_{concurrency}")
        sweep_args.concurrency = concurrency
        results = run_suite(sweep_args, traces, Path(sweep_args.output_dir))
        concurrency_rows.extend(rows_for_results("concurrency", concurrency, results))
    write_rows(output_dir / "concurrency_sweep.csv", concurrency_rows)
    return memory_rows, concurrency_rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run real-trace memory and concurrency sweeps.")
    parser.add_argument("--trace-dir", default="traces/real")
    parser.add_argument(
        "--trace-format",
        choices=("auto", "normalized_jsonl", "normalized_json", "swe_gym", "codetracer", "agentlens", "generic_react_jsonl"),
        default="auto",
    )
    parser.add_argument("--max-traces", type=int)
    parser.add_argument("--min-turns", type=int, default=0)
    parser.add_argument("--filter-success", choices=("all", "success", "fail"), default="all")
    parser.add_argument("--reject-accumulated-fallback", action="store_true")
    parser.add_argument("--prediction-mode", choices=("heuristic", "trace_stats"), default="heuristic")
    parser.add_argument("--train-trace-dir")
    parser.add_argument("--base-memory-budget-gb", type=float, default=0.5)
    parser.add_argument("--base-concurrency", type=int, default=8)
    parser.add_argument("--enable-common-observation-compression", action="store_true")
    parser.add_argument("--enable-asg-observation-compression", action="store_true")
    parser.add_argument("--output-dir", default="agent_results/real_sweeps")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    memory_rows, concurrency_rows = run_sweep(args)
    print(f"Wrote {len(memory_rows)} memory-sweep rows and {len(concurrency_rows)} concurrency-sweep rows to {args.output_dir}")


if __name__ == "__main__":
    main()
