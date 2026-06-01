import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional


@dataclass
class DemandPrediction:
    reuse_probability: float
    expected_future_accesses: float
    next_use_distance: float
    expected_saved_cycles: float

    def to_dict(self) -> dict:
        return asdict(self)


class PredictionMode:
    ORACLE = "oracle"
    HEURISTIC = "heuristic"
    TRACE_STATS = "trace_stats"


class DemandPredictor:
    def __init__(
        self,
        mode: str = PredictionMode.HEURISTIC,
        horizon: int = 16,
        prefill_cycles_per_token: int = 1000,
        stats: Optional[dict] = None,
    ):
        self.mode = mode
        self.horizon = int(horizon)
        self.prefill_cycles_per_token = int(prefill_cycles_per_token)
        self.stats = stats or {}

    def predict(self, state, graph, phase_score: dict, current_step: int, agent: str = None) -> DemandPrediction:
        if self.mode == PredictionMode.TRACE_STATS and self.stats:
            return self._predict_trace_stats(state, graph, phase_score, current_step, agent)
        return self._predict_heuristic(state, graph, phase_score, current_step, agent)

    def _predict_heuristic(self, state, graph, phase_score: dict, current_step: int, agent: str = None) -> DemandPrediction:
        age = max(0, current_step - state.birth_step)
        recency = max(0, current_step - state.last_access)
        current_phase = max(phase_score.items(), key=lambda item: (item[1], item[0]))[0] if phase_score else "unknown"
        base = {
            "system_prefix": 0.98,
            "task_prefix": 0.95,
            "agent_role": 0.92,
            "shared_prefix": 0.88,
            "file_context": 0.62,
            "edit_diff": 0.68,
            "test_failure_summary": 0.78,
            "failure_summary": 0.62,
            "summary_state": 0.70,
            "subagent_output": 0.76,
            "assistant_delta": 0.45,
            "dialogue_delta": 0.42,
            "tool_observation": 0.25,
            "web_result": 0.35,
            "raw_error_log": 0.08,
            "speculative_state": 0.05,
        }.get(state.state_type, 0.35)
        if state.state_type in {"file_context", "edit_diff"} and current_phase in {"execute", "verify"}:
            base += 0.18
        if state.state_type in {"test_failure_summary", "failure_summary"} and current_phase in {"verify", "failure"}:
            base += 0.20
        if state.state_type == "edit_diff" and current_phase == "verify":
            base += 0.12
        if state.state_type == "tool_observation" and state.access_count <= 1 and state.token_len > 2048:
            base -= 0.18
        if state.state_type == "subagent_output":
            base += 0.12 if recency <= 4 else -0.10
        if agent is not None and state.owner == agent:
            base += 0.08
        if state.access_count >= 2:
            base += min(0.22, 0.06 * state.access_count)
        base -= min(0.40, 0.025 * age)
        base += min(0.15, 0.03 * max(0, 6 - recency))
        reuse_probability = max(0.0, min(1.0, base))
        expected_future_accesses = reuse_probability * (1.0 + min(4.0, state.access_count / 2.0))
        if state.state_type in {"system_prefix", "task_prefix", "agent_role"}:
            expected_future_accesses = max(expected_future_accesses, 2.0)
        next_use_distance = 1.0 / max(0.05, reuse_probability)
        expected_saved_cycles = (
            expected_future_accesses
            * reuse_probability
            * state.token_len
            * self.prefill_cycles_per_token
        )
        return DemandPrediction(
            reuse_probability=reuse_probability,
            expected_future_accesses=expected_future_accesses,
            next_use_distance=next_use_distance,
            expected_saved_cycles=expected_saved_cycles,
        )

    def _predict_trace_stats(self, state, graph, phase_score: dict, current_step: int, agent: str = None) -> DemandPrediction:
        current_phase = max(phase_score.items(), key=lambda item: (item[1], item[0]))[0] if phase_score else "unknown"
        key = _stats_key(
            state_type=state.state_type,
            phase=current_phase,
            producer_tool=(state.metadata or {}).get("tool", "none"),
            age=max(0, current_step - state.birth_step),
            access_count=state.access_count,
        )
        bucket = self.stats.get("buckets", {}).get("|".join(key)) or self.stats.get("global", {})
        reuse_probability = float(bucket.get("reuse_probability", 0.25))
        expected_future_accesses = float(bucket.get("expected_future_accesses", reuse_probability))
        next_use_distance = float(bucket.get("next_use_distance", self.horizon))
        saved_tokens = float(bucket.get("expected_saved_tokens", state.token_len * expected_future_accesses))
        return DemandPrediction(
            reuse_probability=reuse_probability,
            expected_future_accesses=expected_future_accesses,
            next_use_distance=next_use_distance,
            expected_saved_cycles=saved_tokens * self.prefill_cycles_per_token,
        )


def fit_trace_stats(traces: Iterable[list[dict]], horizon: int = 16) -> dict:
    buckets = defaultdict(lambda: Counter(count=0, reused=0, future_accesses=0, saved_tokens=0, next_use_total=0))
    global_counter = Counter(count=0, reused=0, future_accesses=0, saved_tokens=0, next_use_total=0)
    for trace in traces:
        state_meta = _state_meta(trace)
        llm_positions = [(idx, event) for idx, event in enumerate(trace) if event.get("type") == "llm"]
        prior_accesses = Counter()
        last_access = {}
        for llm_idx, (step_idx, event) in enumerate(llm_positions):
            phase = event.get("phase") or "unknown"
            future_window = llm_positions[llm_idx + 1 : llm_idx + 1 + horizon]
            future_inputs = [sid for _, future in future_window for sid in future.get("input_state_ids", [])]
            future_counts = Counter(future_inputs)
            for state_id in event.get("input_state_ids", []):
                meta = state_meta.get(state_id, {})
                age = step_idx - int(meta.get("birth_step", 0))
                key = _stats_key(
                    meta.get("state_type", "unknown"),
                    phase,
                    meta.get("producer_tool", "none"),
                    age,
                    prior_accesses[state_id],
                )
                counter = buckets["|".join(key)]
                future_count = future_counts[state_id]
                reused = int(future_count > 0)
                next_use = _next_use_distance(future_window, state_id, step_idx, horizon)
                token_len = int(meta.get("tokens", 1))
                _accumulate(counter, reused, future_count, token_len * future_count, next_use)
                _accumulate(global_counter, reused, future_count, token_len * future_count, next_use)
                prior_accesses[state_id] += 1
                last_access[state_id] = step_idx
    return {
        "horizon": horizon,
        "buckets": {key: _finalize_counter(counter, horizon) for key, counter in buckets.items()},
        "global": _finalize_counter(global_counter, horizon),
    }


def _state_meta(trace):
    meta = {}
    for idx, event in enumerate(trace):
        if event.get("type") == "state":
            meta[event["state_id"]] = {
                "state_type": event.get("state_type", "unknown"),
                "tokens": int(event.get("tokens", 1)),
                "birth_step": idx,
                "producer_tool": (event.get("metadata") or {}).get("tool", "none"),
            }
        elif event.get("type") == "tool":
            meta[event["new_state_id"]] = {
                "state_type": event.get("new_state_type", "tool_observation"),
                "tokens": int(event.get("output_tokens", 1)),
                "birth_step": idx,
                "producer_tool": event.get("tool", "tool"),
            }
        elif event.get("type") == "llm":
            meta[event["new_state_id"]] = {
                "state_type": event.get("new_state_type", "assistant_delta"),
                "tokens": int(event.get("output_tokens", 1)),
                "birth_step": idx,
                "producer_tool": "llm",
            }
    return meta


def _stats_key(state_type: str, phase: str, producer_tool: str, age: int, access_count: int):
    return (
        str(state_type),
        str(phase or "unknown"),
        str(producer_tool or "none"),
        _age_bucket(age),
        _access_bucket(access_count),
    )


def _age_bucket(age: int) -> str:
    if age <= 2:
        return "age_0_2"
    if age <= 8:
        return "age_3_8"
    if age <= 32:
        return "age_9_32"
    return "age_33p"


def _access_bucket(access_count: int) -> str:
    if access_count <= 0:
        return "access_0"
    if access_count == 1:
        return "access_1"
    if access_count <= 3:
        return "access_2_3"
    return "access_4p"


def _next_use_distance(future_window, state_id: str, current_step: int, horizon: int) -> int:
    for step_idx, event in future_window:
        if state_id in event.get("input_state_ids", []):
            return max(1, step_idx - current_step)
    return horizon


def _accumulate(counter, reused: int, future_count: int, saved_tokens: int, next_use: int):
    counter["count"] += 1
    counter["reused"] += reused
    counter["future_accesses"] += future_count
    counter["saved_tokens"] += saved_tokens
    counter["next_use_total"] += next_use


def _finalize_counter(counter, horizon: int) -> dict:
    count = max(1, counter["count"])
    return {
        "count": int(counter["count"]),
        "reuse_probability": counter["reused"] / count,
        "expected_future_accesses": counter["future_accesses"] / count,
        "expected_saved_tokens": counter["saved_tokens"] / count,
        "next_use_distance": counter["next_use_total"] / count if counter["count"] else horizon,
    }


def main(argv=None):
    from .trace_loader import load_trace_dir

    parser = argparse.ArgumentParser(description="Fit simple trace-stat demand predictor buckets.")
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--trace-format", default="auto")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    stats = fit_trace_stats(load_trace_dir(args.trace_dir, args.trace_format), horizon=args.horizon)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Wrote trace-stat predictor to {args.output}")


if __name__ == "__main__":
    main()
