import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

from src.agent.trace_audit import audit_traces
from src.agent.trace_loader import load_trace_dir
from src.agent.trace_profile import profile_traces, write_profile_outputs
from src.agent.trace_schema import validate_trace


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Normalize local H100 traces and run quality audit/profile.")
    parser.add_argument("--input-dir", default="traces/local_h100")
    parser.add_argument("--output-dir", default="traces/local_h100_normalized")
    parser.add_argument("--output-format", choices=("normalized_json",), default="normalized_json")
    parser.add_argument("--audit-output", default="agent_results/local_h100/trace_audit.json")
    parser.add_argument("--profile-output", default="agent_results/local_h100/trace_profile.json")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    traces = []
    for src in candidate_trace_files(input_dir):
        payload = json.loads(src.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            continue
        trace = validate_trace(payload)
        dst = output_dir / normalized_name(src)
        dst.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
        traces.append(trace)
    audit = audit_traces(traces, min_turns=4, allow_reconstructed_full_history=False, delayed_reuse_k=8)
    profile = profile_traces(traces, delayed_reuse_k=8)
    pressure_rows, pressure_summary = estimate_kv_pressure(traces)
    profile.update(pressure_summary)
    audit.update(aggregate_required_metrics(traces, audit, profile))
    Path(args.audit_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.audit_output).write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    write_profile_outputs(profile, args.profile_output)
    pressure_output = inferred_pressure_output(args.audit_output)
    write_pressure_csv(pressure_output, pressure_rows)
    print(f"normalized {len(traces)} traces to {output_dir}")


def candidate_trace_files(input_dir: Path):
    roots = [input_dir / "workflows", input_dir / "concurrent"]
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.json")):
            if path.name == "manifest.json":
                continue
            yield path


def normalized_name(src: Path) -> str:
    parts = [part for part in src.parts if part not in {"traces", "local_h100", "local_h100_long"}]
    return "__".join(parts)


def aggregate_required_metrics(traces: list, audit: dict, profile: dict) -> dict:
    llm = sum(1 for trace in traces for event in trace if event.get("type") == "llm")
    tool = sum(1 for trace in traces for event in trace if event.get("type") == "tool")
    state_types = Counter()
    full_history = 0
    concurrency_levels = set()
    for trace in traces:
        for event in trace:
            meta = event.get("metadata") or {}
            if event.get("type") == "state":
                state_types[event.get("state_type", "unknown")] += 1
            if event.get("type") == "llm":
                full_history += int(bool(meta.get("full_history_likely")))
            wid = meta.get("workflow_id", "")
            if isinstance(wid, str) and (wid.startswith("c") or wid.startswith("long_c")) and "_wf_" in wid:
                try:
                    head = wid.split("_wf_", 1)[0]
                    level = head.replace("long_c", "").replace("c", "")
                    concurrency_levels.add(int(level))
                except Exception:
                    pass
    reuse = [item["reuse_quality"] for item in audit.get("traces", [])]
    delayed = sum(item.get("delayed_reuse_ratio", 0.0) for item in reuse) / len(reuse) if reuse else 0.0
    regret = sum(item.get("LRU_regret_candidate_count", 0) for item in reuse)
    return {
        "num_workflows": len(traces),
        "num_llm_events": llm,
        "num_tool_events": tool,
        "full_history_likely_ratio": full_history / llm if llm else 0.0,
        "delayed_reuse_ratio": delayed,
        "LRU_regret_candidate_count": regret,
        "high_value_delayed_reuse_count": len(profile.get("lru_regret_candidates", [])),
        "high_value_delayed_reuse_tokens": profile.get("high_value_delayed_reuse_tokens", 0),
        "LRU_regret_tokens": profile.get("LRU_regret_tokens", 0),
        "state_type_distribution": dict(state_types),
        "live_kv_bytes_over_time": profile.get("live_kv_bytes_over_time", {}),
        "estimated_memory_pressure_ratio": profile.get("estimated_memory_pressure_ratio", {}),
        "cross_workflow_interleaving": bool(concurrency_levels),
        "concurrency_levels_covered": sorted(concurrency_levels),
    }


def estimate_kv_pressure(traces: list, memory_budget_gb: float = 0.1, n_layers: int = 64, hidden_size: int = 5120, dtype_bytes: int = 2):
    bytes_per_token = n_layers * hidden_size * 2 * dtype_bytes
    rows = []
    pressure_values = []
    live_token_values = []
    for trace_idx, trace in enumerate(traces):
        live_tokens = 0
        seen_states = set()
        workflow_id = trace_workflow_id(trace, trace_idx)
        for event_idx, event in enumerate(trace):
            if event.get("type") == "state" and event.get("state_id") not in seen_states:
                seen_states.add(event.get("state_id"))
                live_tokens += int(event.get("tokens", 0) or 0)
            elif event.get("type") == "tool" and event.get("new_state_id") not in seen_states:
                seen_states.add(event.get("new_state_id"))
                live_tokens += int(event.get("output_tokens", 0) or 0)
            elif event.get("type") == "llm" and event.get("new_state_id") not in seen_states:
                seen_states.add(event.get("new_state_id"))
                live_tokens += int(event.get("output_tokens", 0) or 0)
            live_bytes = live_tokens * bytes_per_token
            pressure = live_bytes / (memory_budget_gb * (1024**3))
            pressure_values.append(pressure)
            live_token_values.append(live_tokens)
            rows.append(
                {
                    "trace_idx": trace_idx,
                    "workflow_id": workflow_id,
                    "event_idx": event_idx,
                    "event_type": event.get("type"),
                    "timestamp": event.get("timestamp_start") or event.get("timestamp_end") or (event.get("metadata") or {}).get("global_timestamp") or event_idx,
                    "live_state_tokens": live_tokens,
                    "live_state_bytes": live_bytes,
                    "estimated_kv_bytes": live_bytes,
                    "estimated_memory_pressure_ratio": pressure,
                    "memory_budget_gb": memory_budget_gb,
                    "bytes_per_token": bytes_per_token,
                }
            )
    summary = {
        "live_kv_bytes_over_time": {
            "count": len(rows),
            "max": max((int(row["estimated_kv_bytes"]) for row in rows), default=0),
            "mean": sum(int(row["estimated_kv_bytes"]) for row in rows) / len(rows) if rows else 0,
        },
        "estimated_memory_pressure_ratio": {
            "count": len(pressure_values),
            "max": max(pressure_values) if pressure_values else 0.0,
            "mean": sum(pressure_values) / len(pressure_values) if pressure_values else 0.0,
        },
        "live_state_tokens_over_time": {
            "count": len(live_token_values),
            "max": max(live_token_values) if live_token_values else 0,
            "mean": sum(live_token_values) / len(live_token_values) if live_token_values else 0.0,
        },
    }
    return rows, summary


def trace_workflow_id(trace: list, fallback_idx: int) -> str:
    for event in trace:
        workflow_id = (event.get("metadata") or {}).get("workflow_id")
        if workflow_id:
            return str(workflow_id)
    return f"trace_{fallback_idx}"


def inferred_pressure_output(audit_output: str) -> Path:
    path = Path(audit_output)
    return path.parent / "estimated_kv_pressure.csv"


def write_pressure_csv(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "trace_idx",
        "workflow_id",
        "event_idx",
        "event_type",
        "timestamp",
        "live_state_tokens",
        "live_state_bytes",
        "estimated_kv_bytes",
        "estimated_memory_pressure_ratio",
        "memory_budget_gb",
        "bytes_per_token",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
