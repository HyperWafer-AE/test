import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Iterable, List

from .trace_loader import load_trace_dir, load_trace_file


def profile_traces(traces: Iterable[List[dict]], delayed_reuse_k: int = 8) -> dict:
    traces = list(traces)
    state_rows = []
    reuse_distances = []
    accumulated_context_lengths = []
    append_tokens = []
    tool_latencies = []
    observation_sizes = []
    state_type_counter = Counter()
    phase_tool_transitions = Counter()
    top_saved_by_type = Counter()
    cross_agent_reused = 0
    reused_states = 0
    lru_regret_candidates = []

    for trace_idx, trace in enumerate(traces):
        state_meta = _collect_state_meta(trace)
        accesses = defaultdict(list)
        agents = defaultdict(set)
        last_phase_tool = None
        for step_idx, event in enumerate(trace):
            event_type = event.get("type")
            if event_type == "state":
                state_type_counter[event.get("state_type", "unknown")] += 1
            elif event_type == "llm":
                total_context = 0
                for state_id in event.get("input_state_ids", []):
                    meta = state_meta.get(state_id, {})
                    total_context += int(meta.get("tokens", 1))
                    accesses[state_id].append(step_idx)
                    agents[state_id].add(event.get("agent", "agent_0"))
                accumulated_context_lengths.append(total_context)
                append_tokens.append(int(event.get("append_tokens", 0)))
                current = (event.get("phase") or "unknown", "llm")
                if last_phase_tool is not None:
                    phase_tool_transitions[(last_phase_tool, current)] += 1
                last_phase_tool = current
            elif event_type == "tool":
                state_type_counter[event.get("new_state_type", "tool_observation")] += 1
                tool_latencies.append(int(event.get("latency", 0)))
                observation_sizes.append(int(event.get("output_tokens", 0)))
                current = (event.get("phase") or "unknown", event.get("tool", "tool"))
                if last_phase_tool is not None:
                    phase_tool_transitions[(last_phase_tool, current)] += 1
                last_phase_tool = current

        for state_id, positions in accesses.items():
            meta = state_meta.get(state_id, {})
            state_type = meta.get("state_type", "unknown")
            token_len = int(meta.get("tokens", 1))
            if len(positions) > 1:
                reused_states += 1
                if len(agents[state_id]) > 1:
                    cross_agent_reused += 1
            for left, right in zip(positions, positions[1:]):
                distance = right - left
                reuse_distances.append(distance)
                top_saved_by_type[state_type] += token_len
                if distance > delayed_reuse_k and token_len >= 128:
                    lru_regret_candidates.append(
                        {
                            "trace_idx": trace_idx,
                            "state_id": state_id,
                            "state_type": state_type,
                            "token_len": token_len,
                            "reuse_distance": distance,
                        }
                    )
            state_rows.append(
                {
                    "trace_idx": trace_idx,
                    "state_id": state_id,
                    "state_type": state_type,
                    "tokens": token_len,
                    "access_count": len(positions),
                    "first_access": positions[0] if positions else "",
                    "last_access": positions[-1] if positions else "",
                    "cross_agent": len(agents[state_id]) > 1,
                }
            )

    num_llm = sum(1 for trace in traces for event in trace if event.get("type") == "llm")
    num_tool = sum(1 for trace in traces for event in trace if event.get("type") == "tool")
    turn_counts = [sum(1 for event in trace if event.get("type") == "llm") for trace in traces]
    state_counts = [sum(1 for event in trace if event.get("type") == "state") for trace in traces]
    delayed_ratio = (
        sum(1 for distance in reuse_distances if distance > delayed_reuse_k) / len(reuse_distances)
        if reuse_distances
        else 0.0
    )
    cross_agent_ratio = cross_agent_reused / reused_states if reused_states else 0.0

    return {
        "num_traces": len(traces),
        "num_llm_events": num_llm,
        "num_tool_events": num_tool,
        "turn_count_distribution": _dist(turn_counts),
        "state_count_distribution": _dist(state_counts),
        "state_type_distribution": dict(state_type_counter),
        "token_distribution_per_state_type": _token_dist_by_type(state_rows),
        "accumulated_context_length_distribution": _dist(accumulated_context_lengths),
        "append_tokens_distribution": _dist(append_tokens),
        "state_reuse_distance_distribution": _dist(reuse_distances),
        "delayed_reuse_ratio": delayed_ratio,
        "delayed_reuse_k": delayed_reuse_k,
        "cross_agent_reuse_ratio": cross_agent_ratio,
        "lru_regret_candidates": sorted(lru_regret_candidates, key=lambda item: (-item["token_len"], -item["reuse_distance"]))[:100],
        "num_lru_regret_candidates": len(lru_regret_candidates),
        "top_state_types_by_future_saved_prefill_tokens": dict(top_saved_by_type.most_common()),
        "phase_tool_transition_counts": {f"{a}->{b}": count for (a, b), count in phase_tool_transitions.items()},
        "tool_latency_distribution": _dist(tool_latencies),
        "observation_size_distribution": _dist(observation_sizes),
        "state_rows": state_rows,
        "delayed_reuse_by_trace": delayed_reuse_by_trace(traces, delayed_reuse_k),
    }


def delayed_reuse_by_trace(traces: Iterable[List[dict]], delayed_reuse_k: int = 8) -> List[dict]:
    rows = []
    for trace_idx, trace in enumerate(traces):
        accesses = defaultdict(list)
        for step_idx, event in enumerate(trace):
            if event.get("type") != "llm":
                continue
            for state_id in event.get("input_state_ids", []):
                accesses[state_id].append(step_idx)
        distances = [
            right - left
            for positions in accesses.values()
            for left, right in zip(positions, positions[1:])
        ]
        ratio = sum(1 for value in distances if value > delayed_reuse_k) / len(distances) if distances else 0.0
        rows.append({"trace_idx": trace_idx, "delayed_reuse_ratio": ratio, "num_reuses": len(distances)})
    return rows


def write_profile_outputs(profile: dict, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_rows = profile.pop("state_rows", [])
    output_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    csv_path = output_path.with_name(f"{output_path.stem}_state_stats.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["trace_idx", "state_id", "state_type", "tokens", "access_count", "first_access", "last_access", "cross_agent"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(state_rows)
    profile["state_rows"] = state_rows
    return csv_path


def _collect_state_meta(trace):
    meta = {}
    for idx, event in enumerate(trace):
        if event.get("type") == "state":
            meta[event["state_id"]] = {
                "state_type": event.get("state_type", "unknown"),
                "tokens": int(event.get("tokens", 1)),
                "birth_step": idx,
            }
        elif event.get("type") == "tool":
            meta[event["new_state_id"]] = {
                "state_type": event.get("new_state_type", "tool_observation"),
                "tokens": int(event.get("output_tokens", 1)),
                "birth_step": idx,
            }
        elif event.get("type") == "llm":
            meta[event["new_state_id"]] = {
                "state_type": event.get("new_state_type", "assistant_delta"),
                "tokens": int(event.get("output_tokens", 1)),
                "birth_step": idx,
            }
    return meta


def _dist(values):
    values = [int(value) for value in values]
    if not values:
        return {"count": 0, "min": 0, "p50": 0, "p90": 0, "max": 0, "mean": 0.0}
    values_sorted = sorted(values)
    return {
        "count": len(values_sorted),
        "min": values_sorted[0],
        "p50": _percentile(values_sorted, 50),
        "p90": _percentile(values_sorted, 90),
        "max": values_sorted[-1],
        "mean": sum(values_sorted) / len(values_sorted),
    }


def _percentile(values, pct):
    if not values:
        return 0
    idx = int(round((pct / 100.0) * (len(values) - 1)))
    return values[max(0, min(len(values) - 1, idx))]


def _token_dist_by_type(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["state_type"]].append(row["tokens"])
    return {state_type: _dist(tokens) for state_type, tokens in grouped.items()}


def _load_from_args(args):
    if args.trace_dir:
        return load_trace_dir(
            args.trace_dir,
            trace_format=args.trace_format,
            max_traces=args.max_traces,
            min_turns=args.min_turns,
            filter_success=args.filter_success,
        )
    if args.trace_file:
        return [load_trace_file(args.trace_file, trace_format=args.trace_format)]
    raise ValueError("Provide --trace-dir or --trace-file")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Profile normalized or public code-agent traces.")
    parser.add_argument("--trace-dir")
    parser.add_argument("--trace-file")
    parser.add_argument("--trace-format", choices=("auto", "normalized_jsonl", "normalized_json", "swe_gym", "codetracer", "agentlens", "generic_react_jsonl"), default="auto")
    parser.add_argument("--max-traces", type=int)
    parser.add_argument("--min-turns", type=int, default=0)
    parser.add_argument("--filter-success", choices=("all", "success", "fail"), default="all")
    parser.add_argument("--delayed-reuse-k", type=int, default=8)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    traces = _load_from_args(args)
    profile = profile_traces(traces, delayed_reuse_k=args.delayed_reuse_k)
    csv_path = write_profile_outputs(profile, args.output)
    print(f"Wrote trace profile to {args.output}")
    print(f"Wrote state stats CSV to {csv_path}")


if __name__ == "__main__":
    main()
