import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Replay local H100 traces in Agent-on-Wafer simulator.")
    parser.add_argument("--trace-dir", default="traces/local_h100_normalized")
    parser.add_argument("--cfg", default="src/platform/cfgs/wamis_hd_distributed.cfg")
    parser.add_argument("--topology", default="wamis")
    parser.add_argument("--output-dir", default="agent_results/local_h100/wafer_replay")
    parser.add_argument("--memory-budgets", type=float, nargs="+", default=[0.25, 0.5, 1, 2])
    parser.add_argument("--concurrency-levels", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--policies", nargs="+", default=["lru-system", "kvflow-like", "asg-retention-v2-online", "asg-retention-v2-oracle", "asg-placement-v2-online", "asg-prefetch-v2-online"])
    parser.add_argument("--check-invariants", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse existing per-run policy JSON/summary files and only rewrite aggregate CSVs.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.output_dir)
    if out.exists() and not args.reuse_existing:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for mem in args.memory_budgets:
        for conc in args.concurrency_levels:
            run_dir = out / f"m{mem}_c{conc}"
            if not args.reuse_existing or not (run_dir / "summary.csv").exists():
                cmd = [
                    sys.executable,
                    "-m",
                    "src.agent.real_trace_experiment",
                    "--trace-dir",
                    args.trace_dir,
                    "--trace-format",
                    "normalized_json",
                    "--output-dir",
                    str(run_dir),
                    "--cfg",
                    args.cfg,
                    "--topology",
                    args.topology,
                    "--memory-budget-gb",
                    str(mem),
                    "--concurrency",
                    str(conc),
                    "--min-turns",
                    "4",
                ]
                if args.check_invariants:
                    cmd.append("--check-invariants")
                result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "runner.log").write_text(result.stdout, encoding="utf-8")
            all_rows.extend(collect_run_rows(run_dir, args.policies, mem, conc))
    write_csv(out / "main_summary.csv", all_rows)
    write_csv(out / "memory_sweep.csv", [row for row in all_rows if str(row.get("concurrency")) == str(args.concurrency_levels[0])])
    write_csv(out / "concurrency_sweep.csv", [row for row in all_rows if str(row.get("memory_budget")) == str(args.memory_budgets[0])])
    report = {
        "policies": args.policies,
        "internal_kv_metrics": "not available from SGLang/vLLM traces; replay uses normalized state events and analytical KV bytes",
        "check_invariants": bool(args.check_invariants),
    }
    (out / "replay_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote replay summaries to {out}")


def collect_run_rows(run_dir: Path, policies: list, mem: float, conc: int) -> list:
    summary = run_dir / "summary.csv"
    if not summary.exists():
        return []
    rows = []
    for row in csv.DictReader(summary.open(encoding="utf-8")):
        if row["policy"] not in policies:
            continue
        row["memory_budget"] = row.get("memory_budget") or mem
        row["concurrency"] = row.get("concurrency") or conc
        policy_json = run_dir / f"{row['policy']}.json"
        if policy_json.exists():
            row.update(derived_replay_fields(json.loads(policy_json.read_text(encoding="utf-8"))))
        rows.append(row)
    return rows


def derived_replay_fields(result: dict) -> dict:
    metrics = result.get("agent_metrics") or {}
    timing = result.get("timing") or {}
    total_cycles = int(timing.get("total_cycles") or 0)
    num_llm_steps = int(metrics.get("num_llm_steps") or 0)
    return {
        "total_cycles": total_cycles,
        "p95_llm_step_latency": "",
        "p99_llm_step_latency": "",
        "throughput_llm_steps_per_cycle": num_llm_steps / total_cycles if total_cycles else 0.0,
        "llm_step_latency_quantiles_available": False,
    }


def write_csv(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    preferred = [
        "policy",
        "prediction_mode",
        "estimator_mode",
        "concurrency",
        "memory_budget",
        "total_cycles",
        "p95_llm_step_latency",
        "p99_llm_step_latency",
        "throughput_llm_steps_per_cycle",
        "effective_prefill_tokens",
        "cache_byte_miss",
        "state_misses",
        "LRU_regret_states_preserved",
        "LRU_regret_tokens_preserved",
    ]
    fields = [field for field in preferred if any(field in row for row in rows)]
    fields.extend(field for row in rows for field in row.keys() if field not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
