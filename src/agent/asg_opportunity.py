import argparse
import json
from pathlib import Path

from .real_trace_experiment import _metric_for, parse_args as parse_real_args, run_suite
from .trace_audit import audit_traces
from .trace_loader import load_trace_dir
from .trace_profile import profile_traces


def analyze_opportunity(trace_dir, trace_format: str, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traces = load_trace_dir(trace_dir, trace_format=trace_format)
    audit = audit_traces(traces, min_turns=4, allow_reconstructed_full_history=False, delayed_reuse_k=8)
    high_indices = [item["trace_idx"] for item in audit.get("traces", []) if item.get("quality_label") == "high"]
    if not high_indices:
        report = {
            "status": "missing_high_quality_traces",
            "trace_dir": str(trace_dir),
            "trace_format": trace_format,
            "paper_usable": False,
            "audit_summary": {
                "num_loaded_traces": audit.get("num_loaded_traces", 0),
                "paper_usable_trace_count": audit.get("paper_usable_trace_count", 0),
                "smoke_test_trace_count": audit.get("smoke_test_trace_count", 0),
                "unusable_trace_count": audit.get("unusable_trace_count", 0),
                "exclusion_reason_counts": audit.get("exclusion_reason_counts", {}),
            },
            "manual_steps_required": [
                "Build traces/real_high_quality with source data that exposes per-step state context or exact prompt states.",
                "Rerun python -m src.agent.build_real_trace_set before opportunity analysis.",
            ],
        }
        missing_path = output_path.with_name("asg_opportunity_missing_report.json")
        missing_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    selected = [traces[idx] for idx in high_indices]
    profile = profile_traces(selected, delayed_reuse_k=8)
    replay_dir = output_path.parent / "asg_opportunity_replay"
    real_args = parse_real_args(
        [
            "--trace-dir",
            str(trace_dir),
            "--trace-format",
            trace_format,
            "--prediction-mode",
            "heuristic",
            "--policy-suite",
            "v2",
            "--memory-budget-gb",
            "0.5",
            "--concurrency",
            "8",
            "--allow-smoke-traces",
            "--output-dir",
            str(replay_dir),
        ]
    )
    real_args.data_quality = "paper_usable"
    real_args.paper_usable = True
    real_args.num_high_quality_traces = len(selected)
    real_args.num_smoke_traces = 0
    results = run_suite(
        real_args,
        selected,
        replay_dir,
        policies=(
            "lru-system",
            "asg-retention-v2-graph-only",
            "asg-retention-v2-online",
            "asg-retention-v2-oracle",
        ),
    )
    lru_eff = _metric_for(results, "lru-system", "effective_prefill_tokens")
    online_eff = _metric_for(results, "asg-retention-v2-online", "effective_prefill_tokens")
    oracle_eff = _metric_for(results, "asg-retention-v2-oracle", "effective_prefill_tokens")
    oracle_gain = (lru_eff - oracle_eff) if lru_eff is not None and oracle_eff is not None else 0
    online_capture_ratio = (
        (lru_eff - online_eff) / oracle_gain
        if oracle_gain and online_eff is not None
        else 0.0
    )
    report = {
        "status": "computed",
        "paper_usable": True,
        "num_high_quality_traces": len(selected),
        "LRU_regret_candidate_count": profile.get("LRU_regret_candidate_count", 0),
        "LRU_regret_tokens": profile.get("LRU_regret_tokens", 0),
        "delayed_reuse_ratio": profile.get("delayed_reuse_ratio", 0.0),
        "high_value_delayed_reuse_tokens": profile.get("high_value_delayed_reuse_tokens", 0),
        "cross_agent_reuse_ratio": profile.get("cross_agent_reuse_ratio", 0.0),
        "oracle_gain_over_lru": oracle_gain,
        "online_capture_ratio": online_capture_ratio,
        "opportunity_class": _classify(
            oracle_gain,
            profile.get("LRU_regret_candidate_count", 0),
            online_capture_ratio,
        ),
        "replay_output_dir": str(replay_dir),
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _classify(oracle_gain, regret_count, capture_ratio) -> str:
    if oracle_gain <= 0 or regret_count <= 0:
        return "none"
    if capture_ratio < 0.25:
        return "weak"
    if capture_ratio <= 0.60:
        return "moderate"
    return "strong"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Classify whether a trace set offers ASG retention opportunity.")
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--trace-format", default="normalized_json")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = analyze_opportunity(args.trace_dir, args.trace_format, args.output)
    status = report.get("status")
    print(f"ASG opportunity status={status}")


if __name__ == "__main__":
    main()
