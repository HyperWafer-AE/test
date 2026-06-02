import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from src.agent.trace_loader import load_trace_file


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Select opportunity/control traces from intrinsic state-reuse features.")
    parser.add_argument("--trace-dir", default="traces/local_h100_long_normalized")
    parser.add_argument("--output-dir", default="agent_results/local_h100_long/trace_selection")
    parser.add_argument("--delayed-reuse-k", type=int, default=8)
    parser.add_argument("--memory-budget-gb", type=float, default=0.1)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    trace_dir = Path(args.trace_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    files = sorted(trace_dir.glob("*.json"))
    for idx, path in enumerate(files):
        trace = load_trace_file(path, trace_format="normalized_json")
        rows.append(score_trace(trace, idx, path, args.delayed_reuse_k, args.memory_budget_gb))
    opportunity = [
        row
        for row in rows
        if row["expected_control_or_opportunity"] == "opportunity"
        and (row["delayed_reuse_ratio"] >= 0.05 or row["LRU_regret_tokens"] > 0)
    ]
    opportunity = sorted(opportunity, key=lambda row: (-row["LRU_regret_tokens"], -row["delayed_reuse_ratio"], -row["file_context_mean_tokens"]))[: max(20, min(30, len(opportunity)))]
    controls = [row for row in rows if row["expected_control_or_opportunity"] == "control"]
    controls = sorted(controls, key=lambda row: (row["delayed_reuse_ratio"], row["LRU_regret_tokens"]))[: max(8, min(10, len(controls)))]
    write_scores(output_dir / "all_trace_scores.csv", rows)
    write_manifest(output_dir / "opportunity_manifest.json", opportunity, args)
    write_manifest(output_dir / "control_manifest.json", controls, args)
    copy_subset(opportunity, output_dir / "opportunity_traces")
    copy_subset(controls, output_dir / "control_traces")
    report = selection_report(rows, opportunity, controls, args)
    (output_dir / "selection_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"selected opportunity={len(opportunity)} control={len(controls)} under {output_dir}")


def score_trace(trace: list, trace_idx: int, path: Path, delayed_k: int, memory_budget_gb: float) -> dict:
    state_meta = {}
    accesses = defaultdict(list)
    workflow_id = path.stem
    task_family = "unknown"
    expected = "unknown"
    for event_idx, event in enumerate(trace):
        meta = event.get("metadata") or {}
        workflow_id = meta.get("workflow_id", workflow_id)
        task_family = meta.get("task_family", task_family)
        expected = meta.get("expected_control_or_opportunity", expected)
        if event.get("type") == "state":
            state_meta[event["state_id"]] = {
                "state_type": event.get("state_type", "unknown"),
                "tokens": int(event.get("tokens", 1)),
            }
        elif event.get("type") == "llm":
            for state_id in event.get("input_state_ids", []):
                accesses[state_id].append(event_idx)
            state_meta[event.get("new_state_id")] = {"state_type": event.get("new_state_type", "assistant_delta"), "tokens": int(event.get("output_tokens", 1))}
        elif event.get("type") == "tool":
            state_meta[event.get("new_state_id")] = {"state_type": event.get("new_state_type", "tool_observation"), "tokens": int(event.get("output_tokens", 1))}

    reuse_distances = []
    delayed_tokens = 0
    regret_tokens = 0
    regret_count = 0
    by_type_tokens = defaultdict(list)
    for state_id, meta in state_meta.items():
        by_type_tokens[meta["state_type"]].append(meta["tokens"])
    for state_id, positions in accesses.items():
        token_len = int(state_meta.get(state_id, {}).get("tokens", 1))
        counted_regret = False
        for left, right in zip(positions, positions[1:]):
            distance = right - left
            reuse_distances.append(distance)
            if distance >= delayed_k and token_len >= 128:
                delayed_tokens += token_len
                if not counted_regret:
                    regret_tokens += token_len
                    regret_count += 1
                    counted_regret = True
    llm_count = sum(1 for event in trace if event.get("type") == "llm")
    tool_count = sum(1 for event in trace if event.get("type") == "tool")
    live_tokens = sum(int(meta.get("tokens", 0)) for meta in state_meta.values())
    estimated_kv_bytes = live_tokens * 64 * 5120 * 2 * 2
    pressure = estimated_kv_bytes / (memory_budget_gb * (1024**3))
    return {
        "trace_idx": trace_idx,
        "trace_path": str(path),
        "workflow_id": workflow_id,
        "task_family": task_family,
        "expected_control_or_opportunity": expected,
        "num_llm_events": llm_count,
        "num_tool_events": tool_count,
        "num_steps": llm_count + tool_count,
        "delayed_reuse_ratio": sum(1 for distance in reuse_distances if distance >= delayed_k) / len(reuse_distances) if reuse_distances else 0.0,
        "reuse_distance_p50": percentile(reuse_distances, 50),
        "file_context_reuse_distance_p50": p50_for_type(accesses, state_meta, "file_context"),
        "high_value_delayed_reuse_tokens": delayed_tokens,
        "LRU_regret_candidate_count": regret_count,
        "LRU_regret_tokens": regret_tokens,
        "file_context_mean_tokens": mean(by_type_tokens["file_context"]),
        "test_failure_summary_mean_tokens": mean(by_type_tokens["test_failure_summary"]),
        "edit_diff_mean_tokens": mean(by_type_tokens["edit_diff"]),
        "live_state_tokens": live_tokens,
        "estimated_kv_bytes": estimated_kv_bytes,
        "estimated_memory_pressure_ratio": pressure,
    }


def p50_for_type(accesses, state_meta, state_type):
    distances = []
    for state_id, positions in accesses.items():
        if state_meta.get(state_id, {}).get("state_type") != state_type:
            continue
        distances.extend(right - left for left, right in zip(positions, positions[1:]))
    return percentile(distances, 50)


def percentile(values, pct):
    values = sorted(values)
    if not values:
        return 0.0
    return values[round((pct / 100.0) * (len(values) - 1))]


def mean(values):
    return sum(values) / len(values) if values else 0.0


def write_scores(path: Path, rows: list):
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(path: Path, rows: list, args):
    payload = {
        "selection_rule": "intrinsic delayed reuse, high-value state tokens, and task control/opportunity label only; no ASG replay metrics used",
        "delayed_reuse_k": args.delayed_reuse_k,
        "memory_budget_gb": args.memory_budget_gb,
        "num_traces": len(rows),
        "traces": rows,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def copy_subset(rows: list, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        src = Path(row["trace_path"])
        shutil.copy2(src, out_dir / src.name)


def selection_report(rows, opportunity, controls, args):
    def summarize(items):
        return {
            "num_traces": len(items),
            "delayed_reuse_ratio_mean": mean([row["delayed_reuse_ratio"] for row in items]),
            "LRU_regret_tokens_sum": sum(row["LRU_regret_tokens"] for row in items),
            "file_context_mean_tokens": mean([row["file_context_mean_tokens"] for row in items]),
            "test_failure_summary_mean_tokens": mean([row["test_failure_summary_mean_tokens"] for row in items]),
            "task_family_distribution": dict(Counter(row["task_family"] for row in items)),
        }
    return {
        "all": summarize(rows),
        "opportunity": summarize(opportunity),
        "control": summarize(controls),
        "acceptance": {
            "opportunity_traces_ge_20": len(opportunity) >= 20,
            "control_traces_ge_8": len(controls) >= 8,
            "opportunity_higher_delayed_reuse_than_control": summarize(opportunity)["delayed_reuse_ratio_mean"] > summarize(controls)["delayed_reuse_ratio_mean"],
            "opportunity_higher_lru_regret_tokens_than_control": summarize(opportunity)["LRU_regret_tokens_sum"] > summarize(controls)["LRU_regret_tokens_sum"],
        },
        "notes": "Selection is trace-intrinsic and independent of ASG policy replay.",
    }


if __name__ == "__main__":
    main()
