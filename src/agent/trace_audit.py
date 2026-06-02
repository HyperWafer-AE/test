import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List

from .trace_loader import load_trace_dir, load_trace_file
from .trace_profile import HIGH_VALUE_TOKEN_THRESHOLD, delayed_reuse_by_trace


def audit_traces(
    traces: Iterable[List[dict]],
    min_turns: int = 4,
    allow_reconstructed_full_history: bool = False,
    delayed_reuse_k: int = 8,
) -> dict:
    traces = list(traces)
    trace_reports = [
        audit_single_trace(
            trace,
            trace_idx=idx,
            min_turns=min_turns,
            allow_reconstructed_full_history=allow_reconstructed_full_history,
            delayed_reuse_k=delayed_reuse_k,
        )
        for idx, trace in enumerate(traces)
    ]
    high_quality = [item for item in trace_reports if item["quality_label"] == "high"]
    low_quality = [item for item in trace_reports if item["quality_label"] == "low"]
    reason_counter = Counter(reason for item in low_quality for reason in item["exclusion_reasons"])
    return {
        "num_loaded_traces": len(traces),
        "num_high_quality_traces": len(high_quality),
        "num_low_quality_traces": len(low_quality),
        "min_turns": min_turns,
        "allow_reconstructed_full_history": allow_reconstructed_full_history,
        "delayed_reuse_k": delayed_reuse_k,
        "exclusion_reason_counts": dict(reason_counter),
        "real_trace_data_available": len(traces) > 0,
        "high_quality_real_trace_available": len(high_quality) > 0,
        "warning": _dataset_warning(traces, high_quality),
        "traces": trace_reports,
    }


def audit_single_trace(
    trace: List[dict],
    trace_idx: int,
    min_turns: int,
    allow_reconstructed_full_history: bool,
    delayed_reuse_k: int,
) -> dict:
    state_meta = _state_meta(trace)
    llms = [event for event in trace if event.get("type") == "llm"]
    tools = [event for event in trace if event.get("type") == "tool"]
    states = [event for event in trace if event.get("type") == "state"]
    agents = {event.get("agent") for event in trace if event.get("agent")}
    prompt_modes = Counter((event.get("metadata") or {}).get("prompt_reconstruction", "unknown") for event in llms)
    full_history_flags = [(event.get("metadata") or {}).get("full_history_likely", False) for event in llms]
    input_counts = [len(event.get("input_state_ids", [])) for event in llms]
    context_tokens = [
        sum(state_meta.get(state_id, {}).get("tokens", 1) for state_id in event.get("input_state_ids", []))
        for event in llms
    ]
    accesses = defaultdict(list)
    access_agents = defaultdict(set)
    for step_idx, event in enumerate(trace):
        if event.get("type") != "llm":
            continue
        for state_id in event.get("input_state_ids", []):
            accesses[state_id].append(step_idx)
            access_agents[state_id].add(event.get("agent", "agent_0"))

    reuse_distances = []
    high_value_delayed = 0
    regret_candidates = 0
    reused_states = 0
    cross_agent_reused = 0
    for state_id, positions in accesses.items():
        token_len = int(state_meta.get(state_id, {}).get("tokens", 1))
        if len(positions) > 1:
            reused_states += 1
            if len(access_agents[state_id]) > 1:
                cross_agent_reused += 1
        state_has_regret = False
        for left, right in zip(positions, positions[1:]):
            distance = right - left
            reuse_distances.append(distance)
            if distance > delayed_reuse_k and token_len >= HIGH_VALUE_TOKEN_THRESHOLD:
                high_value_delayed += 1
                if not state_has_regret:
                    regret_candidates += 1
                    state_has_regret = True

    delayed_ratio = (
        sum(1 for distance in reuse_distances if distance > delayed_reuse_k) / len(reuse_distances)
        if reuse_distances
        else 0.0
    )
    reconstructed_full_history = bool(full_history_flags) and sum(bool(flag) for flag in full_history_flags) / len(full_history_flags) > 0.5
    exclusion_reasons = []
    if len(llms) < min_turns:
        exclusion_reasons.append("too_few_llm_events")
    if not tools:
        exclusion_reasons.append("no_tool_events")
    if not reuse_distances:
        exclusion_reasons.append("no_nontrivial_state_reuse")
    if reconstructed_full_history and not allow_reconstructed_full_history:
        exclusion_reasons.append("reconstructed_full_history_likely")
    quality_label = "low" if exclusion_reasons else "high"

    return {
        "trace_idx": trace_idx,
        "quality_label": quality_label,
        "exclusion_reasons": exclusion_reasons,
        "basic": {
            "num_events": len(trace),
            "num_state_events": len(states),
            "num_llm_events": len(llms),
            "num_tool_events": len(tools),
            "num_agents": len(agents),
            "success": _success_label(trace),
        },
        "prompt_reconstruction_quality": {
            "has_explicit_input_state_ids": any(bool(event.get("input_state_ids")) for event in llms),
            "has_explicit_prompt_segments": any(bool(event.get("input_segments")) for event in llms),
            "reconstructed_from_messages": prompt_modes.get("message_history", 0) > 0,
            "reconstructed_full_history_likely": reconstructed_full_history,
            "prompt_reconstruction_distribution": dict(prompt_modes),
            "average_input_state_count": _mean(input_counts),
            "average_accumulated_context_tokens": _mean(context_tokens),
            "monotonic_context_growth_score": _monotonic_growth_score(input_counts),
        },
        "state_quality": {
            "fraction_states_with_exact_token_hash": _fraction(
                states,
                lambda event: bool(event.get("exact_token_hash") or (event.get("metadata") or {}).get("exact_token_hash")),
            ),
            "fraction_states_with_semantic_key": _fraction(
                states,
                lambda event: bool(event.get("semantic_key") or (event.get("metadata") or {}).get("semantic_key")),
            ),
            "fraction_tool_states": _state_fraction(states, {"tool_observation", "web_result", "subagent_output"}),
            "fraction_file_context_states": _state_fraction(states, {"file_context"}),
            "fraction_failure_states": _state_fraction(states, {"test_failure_summary", "failure_summary", "raw_error_log"}),
            "fraction_unknown_or_dialogue_delta": _state_fraction(states, {"unknown", "dialogue_delta"}),
        },
        "reuse_quality": {
            "reuse_distance_distribution": _dist(reuse_distances),
            "delayed_reuse_ratio": delayed_ratio,
            "cross_agent_reuse_ratio": cross_agent_reused / reused_states if reused_states else 0.0,
            "high_value_delayed_reuse_count": high_value_delayed,
            "LRU_regret_candidate_count": regret_candidates,
        },
    }


def _state_meta(trace: List[dict]) -> dict:
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


def _success_label(trace: List[dict]):
    values = []
    for event in trace:
        metadata = event.get("metadata") or {}
        if "trace_success" in metadata:
            values.append(metadata["trace_success"])
    if not values:
        return None
    return any(bool(value) for value in values)


def _dataset_warning(traces, high_quality):
    if not traces:
        return "No trace files were loaded; do not claim real-trace validation."
    if not high_quality:
        return "No high-quality traces under current criteria; report sample/low-quality results only."
    return ""


def _fraction(items, predicate) -> float:
    if not items:
        return 0.0
    return sum(1 for item in items if predicate(item)) / len(items)


def _state_fraction(states, state_types: set) -> float:
    return _fraction(states, lambda event: event.get("state_type", "unknown") in state_types)


def _mean(values) -> float:
    return sum(values) / len(values) if values else 0.0


def _dist(values):
    values = [int(value) for value in values]
    if not values:
        return {"count": 0, "min": 0, "p50": 0, "p90": 0, "max": 0, "mean": 0.0}
    values = sorted(values)
    return {
        "count": len(values),
        "min": values[0],
        "p50": _percentile(values, 50),
        "p90": _percentile(values, 90),
        "max": values[-1],
        "mean": sum(values) / len(values),
    }


def _percentile(values, pct):
    if not values:
        return 0
    idx = int(round((pct / 100.0) * (len(values) - 1)))
    return values[max(0, min(len(values) - 1, idx))]


def _monotonic_growth_score(values) -> float:
    if len(values) <= 1:
        return 0.0
    comparisons = len(values) - 1
    non_decreasing = sum(1 for left, right in zip(values, values[1:]) if right >= left)
    return non_decreasing / comparisons if comparisons else 0.0


def _load_from_args(args):
    if args.trace_dir:
        return load_trace_dir(
            args.trace_dir,
            trace_format=args.trace_format,
            max_traces=args.max_traces,
            min_turns=0,
            filter_success=args.filter_success,
        )
    if args.trace_file:
        return [load_trace_file(args.trace_file, trace_format=args.trace_format)]
    raise ValueError("Provide --trace-dir or --trace-file")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Audit normalized/public real-trace quality.")
    parser.add_argument("--trace-dir")
    parser.add_argument("--trace-file")
    parser.add_argument(
        "--trace-format",
        choices=("auto", "normalized_jsonl", "normalized_json", "swe_gym", "codetracer", "agentlens", "generic_react_jsonl"),
        default="auto",
    )
    parser.add_argument("--max-traces", type=int)
    parser.add_argument("--min-turns", type=int, default=4)
    parser.add_argument("--filter-success", choices=("all", "success", "fail"), default="all")
    parser.add_argument("--allow-reconstructed-full-history", action="store_true")
    parser.add_argument("--delayed-reuse-k", type=int, default=8)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    traces = _load_from_args(args)
    report = audit_traces(
        traces,
        min_turns=args.min_turns,
        allow_reconstructed_full_history=args.allow_reconstructed_full_history,
        delayed_reuse_k=args.delayed_reuse_k,
    )
    report["delayed_reuse_by_trace"] = delayed_reuse_by_trace(traces, delayed_reuse_k=args.delayed_reuse_k)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote trace audit to {output}")
    print(
        "loaded={loaded} high_quality={high} low_quality={low}".format(
            loaded=report["num_loaded_traces"],
            high=report["num_high_quality_traces"],
            low=report["num_low_quality_traces"],
        )
    )


if __name__ == "__main__":
    main()
