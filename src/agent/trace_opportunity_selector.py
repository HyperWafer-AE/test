import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from .trace_audit import audit_single_trace
from .trace_loader import load_trace_file


TRACE_SUFFIXES = (".traj.json", ".trajectory.json", ".jsonl", ".json")
NON_DIALOGUE_TYPES = {
    "file_context",
    "edit_diff",
    "test_failure_summary",
    "failure_summary",
    "raw_error_log",
    "subagent_output",
    "summary_state",
    "web_result",
    "tool_observation",
}


def select_traces(
    trace_dirs,
    trace_format: str = "auto",
    output_dir="agent_results/trace_selection",
    max_traces: int = 200,
    min_turns: int = 4,
    delayed_reuse_k: int = 8,
    memory_budget_gb: float = 0.5,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = _candidate_files([Path(path) for path in trace_dirs], max_traces=max_traces)
    rows = []
    load_failures = []
    for idx, path in enumerate(candidates):
        try:
            trace = load_trace_file(path, trace_format=trace_format)
            audit = audit_single_trace(
                trace,
                trace_idx=idx,
                min_turns=min_turns,
                allow_reconstructed_full_history=False,
                delayed_reuse_k=delayed_reuse_k,
            )
            metrics = trace_intrinsic_metrics(trace, delayed_reuse_k=delayed_reuse_k, memory_budget_gb=memory_budget_gb)
            row = {
                "trace_idx": idx,
                "source_file": str(path),
                "trace_id": _trace_id(path),
                "load_status": "loaded",
                "quality_label": audit["quality_label"],
                "paper_usable": audit["quality_label"] == "high",
                "exclusion_reasons": ";".join(audit["exclusion_reasons"]),
                **metrics,
            }
        except Exception as exc:
            load_failures.append({"source_file": str(path), "reason": str(exc)})
            row = {
                "trace_idx": idx,
                "source_file": str(path),
                "trace_id": _trace_id(path),
                "load_status": "load_failed",
                "quality_label": "unusable",
                "paper_usable": False,
                "exclusion_reasons": f"load_failed:{exc}",
            }
        rows.append(row)

    _assign_opportunity_scores(rows)
    _classify_rows(rows)
    _write_outputs(output_dir, rows, load_failures, trace_dirs, trace_format, max_traces, min_turns, delayed_reuse_k, memory_budget_gb)
    return {
        "num_candidates": len(candidates),
        "num_loaded": sum(1 for row in rows if row.get("load_status") == "loaded"),
        "num_paper_usable": sum(1 for row in rows if row.get("paper_usable")),
        "num_opportunity_rich": sum(1 for row in rows if row.get("selection_label") == "opportunity_rich"),
        "num_matched_control": sum(1 for row in rows if row.get("selection_label") == "matched_control"),
        "output_dir": str(output_dir),
    }


def trace_intrinsic_metrics(trace, delayed_reuse_k: int = 8, memory_budget_gb: float = 0.5) -> dict:
    state_meta = _state_meta(trace)
    llms = [event for event in trace if event.get("type") == "llm"]
    tools = [event for event in trace if event.get("type") == "tool"]
    agents = {event.get("agent") for event in trace if event.get("agent")}
    prompt_modes = Counter((event.get("metadata") or {}).get("prompt_reconstruction", "unknown") for event in llms)
    full_history_flags = [(event.get("metadata") or {}).get("full_history_likely", False) for event in llms]
    growth_scores = [
        float((event.get("metadata") or {}).get("monotonic_context_growth_score", 0.0) or 0.0)
        for event in llms
    ]
    state_types = Counter(meta.get("state_type", "unknown") for meta in state_meta.values())
    accesses = defaultdict(list)
    access_phases = defaultdict(list)
    access_agents = defaultdict(set)
    for step_idx, event in enumerate(trace):
        if event.get("type") != "llm":
            continue
        phase = event.get("phase") or "unknown"
        for state_id in event.get("input_state_ids", []):
            accesses[state_id].append(step_idx)
            access_phases[state_id].append(phase)
            access_agents[state_id].add(event.get("agent", "agent_0"))

    delayed_reuses = 0
    total_reuses = 0
    high_value_delayed_reuse_count = 0
    high_value_delayed_reuse_tokens = 0
    lru_regret_candidate_count = 0
    lru_regret_tokens = 0
    cross_agent_reused = 0
    reused_states = 0
    phase_dependent_states = 0
    phase_dependent_tokens = 0
    reuse_tokens_by_type = Counter()
    large_state_reuse_tokens = 0
    for state_id, positions in accesses.items():
        meta = state_meta.get(state_id, {})
        tokens = int(meta.get("tokens", 1))
        state_type = meta.get("state_type", "unknown")
        if len(positions) > 1:
            reused_states += 1
            if len(access_agents[state_id]) > 1:
                cross_agent_reused += 1
            if len(set(access_phases[state_id])) > 1:
                phase_dependent_states += 1
                phase_dependent_tokens += tokens
            reuse_tokens_by_type[state_type] += tokens * (len(positions) - 1)
            if tokens >= 1024:
                large_state_reuse_tokens += tokens * (len(positions) - 1)
        state_has_regret = False
        for left, right in zip(positions, positions[1:]):
            distance = right - left
            total_reuses += 1
            if distance > delayed_reuse_k:
                delayed_reuses += 1
                if tokens >= 128:
                    high_value_delayed_reuse_count += 1
                    high_value_delayed_reuse_tokens += tokens
                    if not state_has_regret:
                        lru_regret_candidate_count += 1
                        lru_regret_tokens += tokens
                        state_has_regret = True

    memory_budget_bytes = max(1, int(float(memory_budget_gb) * (1024**3)))
    state_bytes = {state_id: _kv_bytes(meta.get("tokens", 1)) for state_id, meta in state_meta.items()}
    live_bytes = [
        sum(state_bytes.get(state_id, _kv_bytes(1)) for state_id in event.get("input_state_ids", []))
        for event in llms
    ]
    total_unique_bytes = sum(state_bytes.values())
    estimated_live_kv_pressure = max(live_bytes) / memory_budget_bytes if live_bytes else 0.0
    memory_pressure_ratio = total_unique_bytes / memory_budget_bytes
    delayed_reuse_ratio = delayed_reuses / total_reuses if total_reuses else 0.0
    cross_agent_reuse_ratio = cross_agent_reused / reused_states if reused_states else 0.0
    phase_dependent_reuse_score = phase_dependent_tokens / max(1, sum(meta.get("tokens", 1) for meta in state_meta.values()))
    unknown_or_dialogue = sum(count for state_type, count in state_types.items() if state_type in {"unknown", "dialogue_delta"})
    state_count = sum(state_types.values())
    return {
        "num_llm_events": len(llms),
        "num_tool_events": len(tools),
        "num_agents": len(agents),
        "prompt_reconstruction_distribution": json.dumps(dict(prompt_modes), sort_keys=True),
        "full_history_likely": bool(full_history_flags) and sum(bool(flag) for flag in full_history_flags) / len(full_history_flags) > 0.5,
        "full_history_likely_ratio": sum(bool(flag) for flag in full_history_flags) / len(full_history_flags) if full_history_flags else 0.0,
        "monotonic_context_growth_score": sum(growth_scores) / len(growth_scores) if growth_scores else 0.0,
        "fraction_states_with_exact_token_hash": _fraction(state_meta.values(), lambda meta: bool(meta.get("exact_token_hash"))),
        "fraction_states_with_semantic_key": _fraction(state_meta.values(), lambda meta: bool(meta.get("semantic_key"))),
        "state_type_distribution": json.dumps(dict(state_types), sort_keys=True),
        "fraction_unknown_or_dialogue_delta": unknown_or_dialogue / state_count if state_count else 1.0,
        "delayed_reuse_ratio": delayed_reuse_ratio,
        "high_value_delayed_reuse_count": high_value_delayed_reuse_count,
        "high_value_delayed_reuse_tokens": high_value_delayed_reuse_tokens,
        "LRU_regret_candidate_count": lru_regret_candidate_count,
        "LRU_regret_tokens": lru_regret_tokens,
        "cross_agent_reuse_ratio": cross_agent_reuse_ratio,
        "phase_dependent_reuse_score": phase_dependent_reuse_score,
        "failure_state_reuse_tokens": reuse_tokens_by_type["failure_summary"] + reuse_tokens_by_type["raw_error_log"],
        "file_context_reuse_tokens": reuse_tokens_by_type["file_context"],
        "test_failure_reuse_tokens": reuse_tokens_by_type["test_failure_summary"],
        "edit_diff_reuse_tokens": reuse_tokens_by_type["edit_diff"],
        "large_state_reuse_tokens": large_state_reuse_tokens,
        "estimated_live_kv_pressure": estimated_live_kv_pressure,
        "estimated_memory_pressure_ratio": memory_pressure_ratio,
        "estimated_lru_eviction_risk": min(1.0, memory_pressure_ratio) * max(delayed_reuse_ratio, 1.0 if lru_regret_candidate_count else 0.0),
        "total_state_tokens": sum(int(meta.get("tokens", 1)) for meta in state_meta.values()),
    }


def _assign_opportunity_scores(rows):
    loaded = [row for row in rows if row.get("load_status") == "loaded"]
    specs = [
        ("LRU_regret_tokens", 0.30),
        ("high_value_delayed_reuse_tokens", 0.20),
        ("phase_dependent_reuse_score", 0.15),
        ("cross_agent_reuse_ratio", 0.10),
        ("failure_state_reuse_tokens", 0.10),
        ("combined_code_state_reuse_tokens", 0.10),
        ("estimated_memory_pressure_ratio", 0.05),
    ]
    for row in loaded:
        row["combined_code_state_reuse_tokens"] = (
            float(row.get("file_context_reuse_tokens", 0) or 0)
            + float(row.get("test_failure_reuse_tokens", 0) or 0)
            + float(row.get("edit_diff_reuse_tokens", 0) or 0)
        )
    maxima = {
        key: max((float(row.get(key, 0) or 0) for row in loaded), default=0.0)
        for key, _ in specs
    }
    for row in rows:
        if row.get("load_status") != "loaded":
            row["asg_opportunity_score"] = 0.0
            continue
        score = 0.0
        for key, weight in specs:
            denom = maxima.get(key, 0.0)
            value = float(row.get(key, 0) or 0)
            normalized = min(1.0, value / denom) if denom > 0 else 0.0
            score += weight * normalized
        row["asg_opportunity_score"] = score


def _classify_rows(rows):
    paper = [row for row in rows if _paper_gate(row)]
    scores = sorted(float(row.get("asg_opportunity_score", 0.0) or 0.0) for row in paper)
    threshold = scores[max(0, int(math.floor(0.75 * (len(scores) - 1))))] if scores else 0.0
    for row in rows:
        if row.get("load_status") != "loaded":
            row["selection_label"] = "unusable"
        elif not _paper_gate(row):
            row["selection_label"] = "smoke_only" if row.get("quality_label") == "medium" else "unusable"
        elif float(row.get("asg_opportunity_score", 0.0) or 0.0) >= max(0.10, threshold):
            row["selection_label"] = "opportunity_rich"
        else:
            row["selection_label"] = "matched_control"


def _paper_gate(row) -> bool:
    prompt_modes = _json_dict(row.get("prompt_reconstruction_distribution"))
    mostly_accumulated = prompt_modes.get("accumulated_fallback", 0) > max(0, int(row.get("num_llm_events", 0))) / 2
    state_types = _json_dict(row.get("state_type_distribution"))
    has_non_dialogue = any(state_types.get(state_type, 0) > 0 for state_type in NON_DIALOGUE_TYPES)
    has_opportunity = (
        float(row.get("delayed_reuse_ratio", 0) or 0) > 0
        or int(row.get("LRU_regret_candidate_count", 0) or 0) > 0
        or float(row.get("cross_agent_reuse_ratio", 0) or 0) > 0
    )
    return (
        row.get("load_status") == "loaded"
        and not bool(row.get("full_history_likely"))
        and not mostly_accumulated
        and int(row.get("num_tool_events", 0) or 0) > 0
        and has_non_dialogue
        and has_opportunity
    )


def _write_outputs(output_dir, rows, load_failures, trace_dirs, trace_format, max_traces, min_turns, delayed_reuse_k, memory_budget_gb):
    fields = sorted({key for row in rows for key in row})
    with (output_dir / "all_trace_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifests = {
        "opportunity_traces_manifest.json": [row for row in rows if row.get("selection_label") == "opportunity_rich"],
        "matched_control_manifest.json": [row for row in rows if row.get("selection_label") == "matched_control"],
        "smoke_only_manifest.json": [row for row in rows if row.get("selection_label") == "smoke_only"],
        "unusable_manifest.json": [row for row in rows if row.get("selection_label") == "unusable"],
    }
    for name, items in manifests.items():
        (output_dir / name).write_text(json.dumps({"traces": items}, indent=2), encoding="utf-8")
    config = {
        "trace_dirs": [str(path) for path in trace_dirs],
        "trace_format": trace_format,
        "max_traces": max_traces,
        "min_turns": min_turns,
        "delayed_reuse_k": delayed_reuse_k,
        "memory_budget_gb": memory_budget_gb,
    }
    selected_trace_ids = [row["trace_id"] for row in rows if row.get("selection_label") in {"opportunity_rich", "matched_control"}]
    excluded_trace_ids = [row["trace_id"] for row in rows if row.get("selection_label") in {"smoke_only", "unusable"}]
    signature = hashlib.sha256(json.dumps({"config": config, "selected": selected_trace_ids, "excluded": excluded_trace_ids}, sort_keys=True).encode("utf-8")).hexdigest()
    report = {
        "selection_config": config,
        "num_candidates": len(rows),
        "num_loaded": sum(1 for row in rows if row.get("load_status") == "loaded"),
        "num_paper_usable": sum(1 for row in rows if _paper_gate(row)),
        "num_opportunity_rich": len(manifests["opportunity_traces_manifest.json"]),
        "num_matched_control": len(manifests["matched_control_manifest.json"]),
        "num_smoke_only": len(manifests["smoke_only_manifest.json"]),
        "num_unusable": len(manifests["unusable_manifest.json"]),
        "load_failures": load_failures,
        "selected_trace_ids": selected_trace_ids,
        "excluded_trace_ids": excluded_trace_ids,
        "selection_signature_sha256": signature,
        "selection_timestamp": _timestamp(),
    }
    (output_dir / "selection_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_protocol(output_dir / "selection_protocol.md", report)


def _write_protocol(path: Path, report: dict):
    path.write_text(
        "\n".join(
            [
                "# Trace Selection Protocol",
                "",
                "- Traces are selected by workload-intrinsic metrics only.",
                "- ASG/LRU/KVFlow performance outputs are not read by this selector.",
                "- The full candidate set, ASG-opportunity subset, matched control subset, smoke-only set, and unusable set are all reported.",
                "- Negative/control results must not be hidden.",
                "- Smoke-only/full-history traces are not used for paper claims.",
                "",
                f"selection_timestamp: {report['selection_timestamp']}",
                f"selection_signature_sha256: {report['selection_signature_sha256']}",
                f"num_candidates: {report['num_candidates']}",
                f"num_opportunity_rich: {report['num_opportunity_rich']}",
                f"num_matched_control: {report['num_matched_control']}",
                f"num_smoke_only: {report['num_smoke_only']}",
                f"num_unusable: {report['num_unusable']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _state_meta(trace):
    meta = {}
    for idx, event in enumerate(trace):
        if event.get("type") == "state":
            metadata = event.get("metadata") or {}
            meta[event["state_id"]] = {
                "state_type": event.get("state_type", "unknown"),
                "tokens": int(event.get("tokens", 1)),
                "birth_step": idx,
                "exact_token_hash": event.get("exact_token_hash") or metadata.get("exact_token_hash"),
                "semantic_key": event.get("semantic_key") or metadata.get("semantic_key"),
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


def _candidate_files(trace_dirs, max_traces: int):
    files = []
    seen = set()
    for root in trace_dirs:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            name = path.name.lower()
            if name in {"report.json", "summary.json", "profile.json"}:
                continue
            if not any(name.endswith(suffix) for suffix in TRACE_SUFFIXES):
                continue
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(path)
            if len(files) >= max_traces:
                return sorted(files)
    return sorted(files)


def _trace_id(path: Path) -> str:
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]


def _kv_bytes(tokens, n_layers: int = 32, hidden_size: int = 4096, dtype_bytes: int = 2) -> int:
    return int(tokens) * n_layers * hidden_size * dtype_bytes * 2


def _fraction(items, predicate) -> float:
    items = list(items)
    return sum(1 for item in items if predicate(item)) / len(items) if items else 0.0


def _json_dict(value) -> dict:
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value or "{}")
    except Exception:
        return {}


def _timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Pre-register trace opportunity subsets from trace-intrinsic features.")
    parser.add_argument("--trace-dirs", nargs="+", required=True)
    parser.add_argument("--trace-format", default="auto")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-traces", type=int, default=200)
    parser.add_argument("--min-turns", type=int, default=4)
    parser.add_argument("--delayed-reuse-k", type=int, default=8)
    parser.add_argument("--memory-budget-gb", type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = select_traces(
        args.trace_dirs,
        trace_format=args.trace_format,
        output_dir=args.output_dir,
        max_traces=args.max_traces,
        min_turns=args.min_turns,
        delayed_reuse_k=args.delayed_reuse_k,
        memory_budget_gb=args.memory_budget_gb,
    )
    print(
        "candidates={num_candidates} paper_usable={num_paper_usable} opportunity={num_opportunity_rich} control={num_matched_control}".format(
            **result
        )
    )


if __name__ == "__main__":
    main()
