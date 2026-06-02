import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

from .trace_loader import load_trace_dir


def compose_serving_workload(trace_dir, trace_format: str, arrival_model: str, concurrency: int, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    traces = load_trace_dir(trace_dir, trace_format=trace_format)
    report_path = Path("agent_results/serving_workload_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if not traces:
        report = {
            "status": "missing_traces",
            "trace_dir": str(trace_dir),
            "num_workflows": 0,
            "concurrency": concurrency,
            "arrival_model": arrival_model,
            "message": "No traces were available to compose into a serving workload.",
        }
        output.write_text("[]", encoding="utf-8")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    arrivals = _arrival_times(len(traces), arrival_model, concurrency)
    workflow_events = []
    for workflow_id, trace in enumerate(traces):
        namespaced = []
        for local_idx, event in enumerate(trace):
            item = dict(event)
            metadata = dict(item.get("metadata") or {})
            metadata.update(
                {
                    "source_trace_id": workflow_id,
                    "workflow_id": f"workflow_{workflow_id}",
                    "arrival_time": arrivals[workflow_id],
                    "active_workflow_id": f"workflow_{workflow_id}",
                    "composition_model": arrival_model,
                    "workflow_local_event_index": local_idx,
                }
            )
            item["metadata"] = metadata
            namespaced.append(item)
        workflow_events.append(namespaced)
    composed = _interleave(workflow_events, arrivals, arrival_model, concurrency)
    output.write_text(json.dumps(composed, indent=2), encoding="utf-8")
    report = _report(composed, len(traces), concurrency, arrival_model)
    report["status"] = "composed"
    report["output"] = str(output)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _arrival_times(num_workflows: int, arrival_model: str, concurrency: int):
    if arrival_model == "burst":
        return [0 for _ in range(num_workflows)]
    if arrival_model == "staggered":
        return [idx * max(1, concurrency // 2) for idx in range(num_workflows)]
    if arrival_model == "poisson":
        rng = random.Random(0)
        current = 0.0
        arrivals = []
        for _ in range(num_workflows):
            current += rng.expovariate(1.0 / max(1, concurrency))
            arrivals.append(int(round(current)))
        return arrivals
    return [idx % max(1, concurrency) for idx in range(num_workflows)]


def _interleave(workflow_events, arrivals, arrival_model: str, concurrency: int):
    ready = sorted(range(len(workflow_events)), key=lambda idx: (arrivals[idx], idx))
    offsets = [0 for _ in workflow_events]
    output = []
    while any(offset < len(events) for offset, events in zip(offsets, workflow_events)):
        progressed = False
        for idx in ready:
            if offsets[idx] >= len(workflow_events[idx]):
                continue
            if arrival_model == "tool_wait_aware" and output:
                last = output[-1]
                if last.get("type") == "tool" and idx == int(str((last.get("metadata") or {}).get("source_trace_id", -1))):
                    continue
            output.append(workflow_events[idx][offsets[idx]])
            offsets[idx] += 1
            progressed = True
            if len(output) % max(1, concurrency) == 0:
                break
        if not progressed:
            for idx in ready:
                if offsets[idx] < len(workflow_events[idx]):
                    output.append(workflow_events[idx][offsets[idx]])
                    offsets[idx] += 1
                    break
    return output


def _report(events, num_workflows: int, concurrency: int, arrival_model: str) -> dict:
    active = Counter()
    live_kv = []
    tool_waits = 0
    llms = 0
    for idx, event in enumerate(events):
        workflow = (event.get("metadata") or {}).get("workflow_id", "unknown")
        active[idx // max(1, concurrency)] += 1
        if event.get("type") == "llm":
            llms += 1
            live_kv.append(len(event.get("input_state_ids", [])))
        elif event.get("type") == "tool":
            tool_waits += 1
    return {
        "num_workflows": num_workflows,
        "concurrency": concurrency,
        "arrival_model": arrival_model,
        "active_workflows_over_time": dict(active),
        "estimated_live_kv_bytes_over_time": [count * 32 * 4096 * 2 * 2 for count in live_kv],
        "memory_pressure_ratio": (max(live_kv) * 32 * 4096 * 2 * 2 / (0.5 * 1024**3)) if live_kv else 0.0,
        "tool_wait_overlap_opportunity": tool_waits / max(1, llms + tool_waits),
        "interleaving_statistics": {
            "num_events": len(events),
            "num_llm_events": llms,
            "num_tool_events": tool_waits,
            "events_per_workflow_mean": len(events) / max(1, num_workflows),
        },
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compose single-workflow traces into a concurrent serving workload.")
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--trace-format", default="normalized_json")
    parser.add_argument("--arrival-model", choices=("round_robin", "burst", "poisson", "staggered", "tool_wait_aware"), default="round_robin")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = compose_serving_workload(args.trace_dir, args.trace_format, args.arrival_model, args.concurrency, args.output)
    print(f"serving_workload status={report['status']} workflows={report.get('num_workflows', 0)}")


if __name__ == "__main__":
    main()
