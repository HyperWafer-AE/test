import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List

from .trace_loader import load_trace_dir, load_trace_file


HIGH_VALUE_TOKEN_THRESHOLD = 128


def profile_traces(traces: Iterable[List[dict]], delayed_reuse_k: int = 8) -> dict:
    traces = list(traces)
    state_rows = []
    reuse_rows = []
    reuse_distances = []
    accumulated_context_lengths = []
    append_tokens = []
    tool_latencies = []
    observation_sizes = []
    state_type_counter = Counter()
    phase_transition_matrix = Counter()
    phase_distribution = Counter()
    tool_type_distribution = Counter()
    prompt_reconstruction_distribution = Counter()
    top_saved_by_type = Counter()
    high_value_delayed_by_type = Counter()
    lru_regret_by_type = Counter()
    reuse_distance_by_type = defaultdict(list)
    access_count_by_type = defaultdict(list)
    lifetime_by_type = defaultdict(list)
    cross_agent_reused = 0
    reused_states = 0
    lru_regret_candidates = []
    high_value_delayed_reuse_tokens = 0
    lru_regret_tokens = 0
    failure_tools = 0
    num_tool = 0
    full_history_llms = 0
    num_llm = 0

    for trace_idx, trace in enumerate(traces):
        state_meta = _collect_state_meta(trace)
        accesses = defaultdict(list)
        agents = defaultdict(set)
        last_phase = None
        for step_idx, event in enumerate(trace):
            event_type = event.get("type")
            if event_type == "state":
                state_type_counter[event.get("state_type", "unknown")] += 1
            elif event_type == "llm":
                num_llm += 1
                metadata = event.get("metadata") or {}
                prompt_reconstruction_distribution[metadata.get("prompt_reconstruction", "unknown")] += 1
                if metadata.get("full_history_likely"):
                    full_history_llms += 1
                total_context = 0
                for state_id in event.get("input_state_ids", []):
                    meta = state_meta.get(state_id, {})
                    total_context += int(meta.get("tokens", 1))
                    accesses[state_id].append(step_idx)
                    agents[state_id].add(event.get("agent", "agent_0"))
                accumulated_context_lengths.append(total_context)
                append_tokens.append(int(event.get("append_tokens", 0)))
                phase = event.get("phase") or "unknown"
                phase_distribution[phase] += 1
                if last_phase is not None:
                    phase_transition_matrix[(last_phase, phase)] += 1
                last_phase = phase
            elif event_type == "tool":
                num_tool += 1
                state_type_counter[event.get("new_state_type", "tool_observation")] += 1
                tool = event.get("tool", "tool")
                tool_type_distribution[tool] += 1
                status = str(event.get("status", "ok")).lower()
                if status in {"fail", "failed", "error"}:
                    failure_tools += 1
                tool_latencies.append(int(event.get("latency", 0)))
                observation_sizes.append(int(event.get("output_tokens", 0)))
                phase = event.get("phase") or "unknown"
                phase_distribution[phase] += 1
                if last_phase is not None:
                    phase_transition_matrix[(last_phase, phase)] += 1
                last_phase = phase

        for state_id, meta in state_meta.items():
            positions = accesses.get(state_id, [])
            state_type = meta.get("state_type", "unknown")
            token_len = int(meta.get("tokens", 1))
            cross_agent = len(agents[state_id]) > 1
            if len(positions) > 1:
                reused_states += 1
                if cross_agent:
                    cross_agent_reused += 1
            access_count_by_type[state_type].append(len(positions))
            if positions:
                lifetime_by_type[state_type].append(max(positions) - int(meta.get("birth_step", positions[0])))
            state_rows.append(
                {
                    "trace_idx": trace_idx,
                    "state_id": state_id,
                    "state_type": state_type,
                    "tokens": token_len,
                    "access_count": len(positions),
                    "first_access": positions[0] if positions else "",
                    "last_access": positions[-1] if positions else "",
                    "cross_agent": cross_agent,
                    "has_exact_token_hash": bool(meta.get("exact_token_hash")),
                    "has_semantic_key": bool(meta.get("semantic_key")),
                }
            )
            seen_regret_for_state = False
            for left, right in zip(positions, positions[1:]):
                distance = right - left
                delayed = distance > delayed_reuse_k
                high_value = token_len >= HIGH_VALUE_TOKEN_THRESHOLD
                reuse_distances.append(distance)
                reuse_distance_by_type[state_type].append(distance)
                top_saved_by_type[state_type] += token_len
                if delayed and high_value:
                    high_value_delayed_reuse_tokens += token_len
                    high_value_delayed_by_type[state_type] += token_len
                    if not seen_regret_for_state:
                        lru_regret_tokens += token_len
                        lru_regret_by_type[state_type] += token_len
                        lru_regret_candidates.append(
                            {
                                "trace_idx": trace_idx,
                                "state_id": state_id,
                                "state_type": state_type,
                                "token_len": token_len,
                                "reuse_distance": distance,
                            }
                        )
                        seen_regret_for_state = True
                reuse_rows.append(
                    {
                        "trace_idx": trace_idx,
                        "state_id": state_id,
                        "state_type": state_type,
                        "tokens": token_len,
                        "left_step": left,
                        "right_step": right,
                        "reuse_distance": distance,
                        "delayed": delayed,
                        "high_value": high_value,
                        "cross_agent": cross_agent,
                    }
                )

    turn_counts = [sum(1 for event in trace if event.get("type") == "llm") for trace in traces]
    state_counts = [sum(1 for event in trace if event.get("type") == "state") for trace in traces]
    delayed_ratio = (
        sum(1 for distance in reuse_distances if distance > delayed_reuse_k) / len(reuse_distances)
        if reuse_distances
        else 0.0
    )
    cross_agent_ratio = cross_agent_reused / reused_states if reused_states else 0.0
    full_history_likely_ratio = full_history_llms / num_llm if num_llm else 0.0
    warning = ""
    if delayed_ratio < 0.05:
        warning = "Delayed reuse is near zero; LRU is expected to be a strong baseline."

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
        "reuse_distance_by_state_type": _dist_by_type(reuse_distance_by_type),
        "access_count_by_state_type": _dist_by_type(access_count_by_type),
        "state_lifetime_by_state_type": _dist_by_type(lifetime_by_type),
        "delayed_reuse_ratio": delayed_ratio,
        "delayed_reuse_k": delayed_reuse_k,
        "cross_agent_reuse_ratio": cross_agent_ratio,
        "prompt_reconstruction_distribution": dict(prompt_reconstruction_distribution),
        "full_history_likely_ratio": full_history_likely_ratio,
        "high_value_delayed_reuse_tokens": high_value_delayed_reuse_tokens,
        "high_value_delayed_reuse_by_state_type": dict(high_value_delayed_by_type.most_common()),
        "LRU_regret_tokens": lru_regret_tokens,
        "LRU_regret_by_state_type": dict(lru_regret_by_type.most_common()),
        "LRU_regret_candidate_count": len(lru_regret_candidates),
        "lru_regret_candidates": sorted(
            lru_regret_candidates,
            key=lambda item: (-item["token_len"], -item["reuse_distance"]),
        )[:100],
        "num_lru_regret_candidates": len(lru_regret_candidates),
        "top_state_types_by_future_saved_prefill_tokens": dict(top_saved_by_type.most_common()),
        "tool_type_distribution": dict(tool_type_distribution),
        "failure_tool_ratio": failure_tools / num_tool if num_tool else 0.0,
        "phase_distribution": dict(phase_distribution),
        "phase_transition_matrix": {f"{src}->{dst}": count for (src, dst), count in phase_transition_matrix.items()},
        "tool_latency_distribution": _dist(tool_latencies),
        "observation_size_distribution": _dist(observation_sizes),
        "warning": warning,
        "state_rows": state_rows,
        "reuse_rows": reuse_rows,
        "delayed_reuse_by_trace": delayed_reuse_by_trace(traces, delayed_reuse_k),
    }


def delayed_reuse_by_trace(traces: Iterable[List[dict]], delayed_reuse_k: int = 8) -> List[dict]:
    rows = []
    for trace_idx, trace in enumerate(traces):
        accesses = defaultdict(list)
        state_meta = _collect_state_meta(trace)
        delayed_tokens = 0
        lru_regret_candidates = 0
        for step_idx, event in enumerate(trace):
            if event.get("type") != "llm":
                continue
            for state_id in event.get("input_state_ids", []):
                accesses[state_id].append(step_idx)
        distances = []
        for state_id, positions in accesses.items():
            token_len = int(state_meta.get(state_id, {}).get("tokens", 1))
            state_has_regret = False
            for left, right in zip(positions, positions[1:]):
                distance = right - left
                distances.append(distance)
                if distance > delayed_reuse_k and token_len >= HIGH_VALUE_TOKEN_THRESHOLD:
                    delayed_tokens += token_len
                    if not state_has_regret:
                        lru_regret_candidates += 1
                        state_has_regret = True
        ratio = sum(1 for value in distances if value > delayed_reuse_k) / len(distances) if distances else 0.0
        rows.append(
            {
                "trace_idx": trace_idx,
                "delayed_reuse_ratio": ratio,
                "num_reuses": len(distances),
                "high_value_delayed_reuse_tokens": delayed_tokens,
                "LRU_regret_candidate_count": lru_regret_candidates,
            }
        )
    return rows


def write_profile_outputs(profile: dict, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_rows = profile.pop("state_rows", [])
    reuse_rows = profile.pop("reuse_rows", [])
    output_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    state_csv_path = output_path.with_name(f"{output_path.stem}_state_stats.csv")
    with state_csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "trace_idx",
            "state_id",
            "state_type",
            "tokens",
            "access_count",
            "first_access",
            "last_access",
            "cross_agent",
            "has_exact_token_hash",
            "has_semantic_key",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(state_rows)

    reuse_csv_path = output_path.with_name(f"{output_path.stem}_reuse_stats.csv")
    with reuse_csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "trace_idx",
            "state_id",
            "state_type",
            "tokens",
            "left_step",
            "right_step",
            "reuse_distance",
            "delayed",
            "high_value",
            "cross_agent",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(reuse_rows)

    profile["state_rows"] = state_rows
    profile["reuse_rows"] = reuse_rows
    return state_csv_path, reuse_csv_path


def _collect_state_meta(trace):
    meta = {}
    for idx, event in enumerate(trace):
        if event.get("type") == "state":
            metadata = event.get("metadata") or {}
            meta[event["state_id"]] = {
                "state_type": event.get("state_type", "unknown"),
                "tokens": int(event.get("tokens", 1)),
                "birth_step": idx,
                "semantic_key": event.get("semantic_key") or metadata.get("semantic_key"),
                "exact_token_hash": event.get("exact_token_hash") or metadata.get("exact_token_hash"),
            }
        elif event.get("type") == "tool":
            meta[event["new_state_id"]] = {
                "state_type": event.get("new_state_type", "tool_observation"),
                "tokens": int(event.get("output_tokens", 1)),
                "birth_step": idx,
                "semantic_key": None,
                "exact_token_hash": None,
            }
        elif event.get("type") == "llm":
            meta[event["new_state_id"]] = {
                "state_type": event.get("new_state_type", "assistant_delta"),
                "tokens": int(event.get("output_tokens", 1)),
                "birth_step": idx,
                "semantic_key": None,
                "exact_token_hash": None,
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


def _dist_by_type(grouped_values):
    return {state_type: _dist(values) for state_type, values in grouped_values.items()}


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
    parser.add_argument(
        "--trace-format",
        choices=("auto", "normalized_jsonl", "normalized_json", "swe_gym", "codetracer", "agentlens", "generic_react_jsonl"),
        default="auto",
    )
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
    state_csv_path, reuse_csv_path = write_profile_outputs(profile, args.output)
    print(f"Wrote trace profile to {args.output}")
    print(f"Wrote state stats CSV to {state_csv_path}")
    print(f"Wrote reuse stats CSV to {reuse_csv_path}")


if __name__ == "__main__":
    main()
