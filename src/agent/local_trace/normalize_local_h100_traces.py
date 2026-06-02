import argparse
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
    audit.update(aggregate_required_metrics(traces, audit, profile))
    Path(args.audit_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.audit_output).write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    write_profile_outputs(profile, args.profile_output)
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
    parts = [part for part in src.parts if part not in {"traces", "local_h100"}]
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
            if isinstance(wid, str) and wid.startswith("c") and "_wf_" in wid:
                try:
                    concurrency_levels.add(int(wid.split("_", 1)[0][1:]))
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
        "high_value_delayed_reuse_tokens": profile.get("high_value_delayed_reuse_tokens", 0),
        "state_type_distribution": dict(state_types),
        "cross_workflow_interleaving": bool(concurrency_levels),
        "concurrency_levels_covered": sorted(concurrency_levels),
    }


if __name__ == "__main__":
    main()
