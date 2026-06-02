import argparse
import csv
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _maybe_reexec_in_venv():
    repo_root = _repo_root()
    venv_python = repo_root / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        return
    if Path(sys.executable).resolve() == venv_python.resolve():
        return
    try:
        import numpy  # noqa: F401
    except ModuleNotFoundError:
        os.execv(str(venv_python), [str(venv_python), "-m", "src.agent.real_trace_experiment"] + sys.argv[1:])


_maybe_reexec_in_venv()


def _setup_paths():
    repo_root = _repo_root()
    paths = [
        repo_root,
        repo_root / "src",
        repo_root / "utils",
        repo_root / "src" / "scheduling",
        repo_root / "src" / "scheduling" / "communication" / "topology",
        repo_root / "src" / "backend" / "analytical",
    ]
    for path in paths:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_setup_paths()

from event_driver import collective_event_driver
import numpy as np
from read_cfg import cfg_to_dict
from tlm import tlm2d
from WAMIS_HD import wamis_hdc

from .agent_event_builder import AgentEventBuilder
from .demand_predictor import DemandPredictor, PredictionMode, fit_trace_stats
from .metrics import postprocess_agent_events
from .model_profile import ModelProfile
from .state_manager import (
    ASGPlacementV2StateManager,
    ASGPrefetchV2StateManager,
    ASGRetentionV2StateManager,
    KVFlowLikeStateManager,
    LRUStateManager,
    NoCacheStateManager,
)
from .trace_audit import audit_traces
from .trace_loader import load_trace_dir, load_trace_file
from .trace_profile import delayed_reuse_by_trace, profile_traces


REAL_V2_POLICIES = (
    "nocache",
    "lru-basic",
    "lru-system",
    "kvflow-like",
    "asg-retention-v2-graph-only",
    "asg-retention-v2-online",
    "asg-retention-v2-trace-stats",
    "asg-retention-v2-oracle",
    "asg-placement-v2-online",
    "asg-prefetch-v2-online",
)


def build_hardware(cfg_path: str, topology: str):
    cfg = cfg_to_dict(cfg_path)
    if topology == "wamis":
        return wamis_hdc(cfg)
    if topology == "tlm":
        return tlm2d(cfg)
    raise ValueError(f"Unsupported topology: {topology}")


def build_model(args) -> ModelProfile:
    return ModelProfile(
        n_layers=args.n_layers,
        hidden_size=args.hidden_size,
        dtype_bytes=args.dtype_bytes,
        prefill_cycles_per_token=args.prefill_cycles_per_token,
        decode_cycles_per_token=args.decode_cycles_per_token,
    )


def load_traces_from_args(args):
    if args.trace_dir:
        return load_trace_dir(
            args.trace_dir,
            trace_format=args.trace_format,
            max_traces=args.max_traces,
            min_turns=args.min_turns,
            filter_success=args.filter_success,
            reject_accumulated_fallback=args.reject_accumulated_fallback,
        )
    if args.trace_file:
        return [
            load_trace_file(
                args.trace_file,
                trace_format=args.trace_format,
                reject_accumulated_fallback=args.reject_accumulated_fallback,
            )
        ]
    raise ValueError("Provide --trace-dir or --trace-file")


def namespace_trace(trace, trace_idx: int):
    state_map = {}
    agent_map = {}

    def map_state(state_id):
        if state_id not in state_map:
            state_map[state_id] = f"trace{trace_idx}:{state_id}"
        return state_map[state_id]

    def map_agent(agent):
        if agent not in agent_map:
            agent_map[agent] = f"trace{trace_idx}_{agent}"
        return agent_map[agent]

    namespaced = []
    for event in trace:
        item = dict(event)
        if item.get("type") == "state":
            item["state_id"] = map_state(item["state_id"])
            if item.get("owner") not in {"shared", None}:
                item["owner"] = map_agent(item["owner"])
        elif item.get("type") == "llm":
            item["agent"] = map_agent(item.get("agent", "agent_0"))
            item["input_state_ids"] = [map_state(state_id) for state_id in item.get("input_state_ids", [])]
            item["new_state_id"] = map_state(item["new_state_id"])
            item["event_id"] = f"trace{trace_idx}:{item.get('event_id', len(namespaced))}"
        elif item.get("type") == "tool":
            item["agent"] = map_agent(item.get("agent", "agent_0"))
            item["new_state_id"] = map_state(item["new_state_id"])
            item["event_id"] = f"trace{trace_idx}:{item.get('event_id', len(namespaced))}"
        metadata = dict(item.get("metadata") or {})
        metadata["trace_idx"] = trace_idx
        item["metadata"] = metadata
        namespaced.append(item)
    return namespaced


def interleave_traces(traces, concurrency: int):
    concurrency = max(1, int(concurrency))
    output = []
    for batch_start in range(0, len(traces), concurrency):
        batch = [namespace_trace(trace, idx) for idx, trace in enumerate(traces[batch_start : batch_start + concurrency], batch_start)]
        offsets = [0 for _ in batch]
        while any(offset < len(trace) for offset, trace in zip(offsets, batch)):
            for idx, trace in enumerate(batch):
                if offsets[idx] >= len(trace):
                    continue
                output.append(trace[offsets[idx]])
                offsets[idx] += 1
    return output


def make_state_manager(policy: str, memory_budget_bytes: int, model: ModelProfile, args):
    if policy == "nocache":
        return NoCacheStateManager(memory_budget_bytes)
    if policy in {"lru-basic", "lru-system"}:
        return LRUStateManager(memory_budget_bytes)
    if policy == "kvflow-like":
        return KVFlowLikeStateManager(memory_budget_bytes)
    manager_cls = {
        "asg-retention-v2-graph-only": ASGRetentionV2StateManager,
        "asg-retention-v2-online": ASGRetentionV2StateManager,
        "asg-retention-v2-trace-stats": ASGRetentionV2StateManager,
        "asg-retention-v2-oracle": ASGRetentionV2StateManager,
        "asg-placement-v2-online": ASGPlacementV2StateManager,
        "asg-prefetch-v2-online": ASGPrefetchV2StateManager,
    }[policy]
    return manager_cls(
        memory_budget_bytes,
        prefill_cycles_per_token=model.prefill_cycles_per_token,
        knapsack_granularity_bytes=int(args.knapsack_granularity_mb * 1024 * 1024),
        max_future_access_cap=args.max_future_access_cap,
        storage_penalty=args.storage_penalty,
        knapsack_max_candidates=args.knapsack_max_candidates,
        current_prompt_bonus=args.asg_current_prompt_bonus,
        recency_bonus=args.asg_recency_bonus,
    )


def policy_backend_name(policy: str) -> str:
    return {
        "nocache": "nocache",
        "lru-basic": "lru",
        "lru-system": "lru",
        "kvflow-like": "kvflow",
        "asg-retention-v2-graph-only": "asg-retention-v2",
        "asg-retention-v2-online": "asg-retention-v2",
        "asg-retention-v2-trace-stats": "asg-retention-v2",
        "asg-retention-v2-oracle": "asg-oracle-retention",
        "asg-placement-v2-online": "asg-placement-v2",
        "asg-prefetch-v2-online": "asg-prefetch-v2",
    }[policy]


def _estimator_mode_for(policy: str, args) -> str:
    if policy == "asg-retention-v2-oracle":
        return "oracle"
    if policy == "asg-retention-v2-trace-stats":
        return PredictionMode.TRACE_STATS
    if policy in {
        "asg-retention-v2-online",
        "asg-placement-v2-online",
        "asg-prefetch-v2-online",
    }:
        return PredictionMode.HEURISTIC
    return "none"


def _baseline_class_for(policy: str) -> str:
    if policy in {"nocache", "lru-basic"}:
        return "basic"
    if policy in {"lru-system", "kvflow-like"}:
        return "system"
    if policy == "asg-retention-v2-oracle":
        return "oracle"
    if policy.startswith("asg-"):
        return "asg"
    return "unknown"


def prompt_quality_summary(trace) -> dict:
    prompt_modes = Counter()
    full_history_count = 0
    growth_scores = []
    num_llm = 0
    for event in trace:
        if event.get("type") != "llm":
            continue
        num_llm += 1
        metadata = event.get("metadata") or {}
        prompt_modes[metadata.get("prompt_reconstruction", "unknown")] += 1
        if metadata.get("full_history_likely"):
            full_history_count += 1
        if "monotonic_context_growth_score" in metadata:
            growth_scores.append(float(metadata.get("monotonic_context_growth_score") or 0.0))
    return {
        "num_llm_events": num_llm,
        "prompt_reconstruction_distribution": dict(prompt_modes),
        "full_history_likely_ratio": full_history_count / num_llm if num_llm else 0.0,
        "mean_monotonic_context_growth_score": (
            sum(growth_scores) / len(growth_scores) if growth_scores else 0.0
        ),
    }


def _prompt_quality_label(summary: dict) -> str:
    num_llm = int(summary.get("num_llm_events", 0))
    if num_llm <= 0:
        return "no_llm_events"
    prompt_modes = summary.get("prompt_reconstruction_distribution") or {}
    dominant_mode, dominant_count = max(prompt_modes.items(), key=lambda item: item[1]) if prompt_modes else ("unknown", 0)
    full_history_ratio = float(summary.get("full_history_likely_ratio", 0.0))
    if prompt_modes.get("accumulated_fallback", 0) / num_llm > 0.5:
        return "accumulated_fallback"
    if full_history_ratio >= 0.5:
        return f"{dominant_mode}_full_history_risk"
    if dominant_count / num_llm >= 0.9:
        return dominant_mode
    return "mixed"


def run_real_policy(
    policy: str,
    args,
    trace,
    predictors,
    num_traces: int,
    regret_state_tokens: dict,
    prompt_quality: dict,
    selected_trace_count=None,
    delayed_reuse_ratio=None,
):
    model = build_model(args)
    hardware = build_hardware(args.cfg, args.topology)
    if hasattr(hardware, "frequency"):
        hardware.frequency = args.hardware_frequency_ghz
    random.seed(args.scheduler_seed)
    np.random.seed(args.scheduler_seed)
    memory_budget_bytes = int(args.memory_budget_gb * (1024 ** 3))
    state_manager = make_state_manager(policy, memory_budget_bytes, model, args)
    state_manager.policy_name = policy_backend_name(policy)
    active_nodes = max(1, len(getattr(hardware, "nodes_set", [])))
    per_node_memory_mb = args.per_node_memory_mb
    if per_node_memory_mb is None:
        per_node_memory_mb = max(16, int(args.memory_budget_gb * 1024 / active_nodes))

    topology_enabled = policy in {"asg-placement-v2-online", "asg-prefetch-v2-online"}
    prefetch_enabled = policy == "asg-prefetch-v2-online"
    oracle_future = policy == "asg-retention-v2-oracle"
    is_asg_policy = policy.startswith("asg-")
    estimator_mode = _estimator_mode_for(policy, args)
    active_predictor = predictors.get(estimator_mode) if estimator_mode in {PredictionMode.HEURISTIC, PredictionMode.TRACE_STATS} else None
    static_replica = policy != "lru-basic" and not args.disable_static_replica
    common_compression = args.enable_common_observation_compression and not args.disable_observation_compression
    asg_specific_compression = (
        is_asg_policy
        and args.enable_asg_observation_compression
        and not args.disable_observation_compression
    )
    compression = common_compression or asg_specific_compression
    compression_threshold = (
        args.asg_large_observation_token_threshold
        if asg_specific_compression
        else args.large_observation_token_threshold
    )
    compression_ratio = (
        args.asg_observation_compression_ratio
        if asg_specific_compression
        else args.observation_compression_ratio
    )

    builder = AgentEventBuilder(
        hardware_platform=hardware,
        model_profile=model,
        state_manager=state_manager,
        enable_prefetch=prefetch_enabled,
        enable_topology_placement=topology_enabled,
        agent_placement=args.agent_placement,
        per_node_budget_bytes=int(per_node_memory_mb * 1024 * 1024),
        max_prefetch_states=args.max_prefetch_states,
        tool_latency_scale=args.tool_latency_scale,
        effective_bandwidth_bytes_per_cycle=args.effective_bandwidth_bytes_per_cycle,
        prefetch_reuse_threshold=args.prefetch_reuse_threshold,
        prefetch_next_use_threshold=args.prefetch_next_use_threshold,
        max_prefetch_bytes=args.max_prefetch_bytes,
        prefetch_wait_fraction=args.prefetch_wait_fraction,
        enable_observation_compression=compression,
        large_observation_token_threshold=compression_threshold,
        observation_compression_ratio=compression_ratio,
        future_horizon=args.horizon,
        oracle_future=oracle_future,
        comm_cost_model=args.comm_cost_model,
        demand_predictor=active_predictor,
        enable_static_replica=static_replica,
    )
    events_dict, graph, metrics = builder.build(trace)
    total_cycles, pure_comp_cycles, pure_comm_cycles = collective_event_driver(events_dict, hardware)
    postprocess_agent_events(events_dict, metrics, total_cycles, pure_comp_cycles, pure_comm_cycles)
    retained_types = Counter(state.state_type for state in graph.states.values() if state.resident)
    regret_preserved = sum(1 for state_id in regret_state_tokens if state_id in graph.states and graph.states[state_id].resident)
    regret_tokens_preserved = sum(
        int(tokens)
        for state_id, tokens in regret_state_tokens.items()
        if state_id in graph.states and graph.states[state_id].resident
    )
    return {
        "policy": policy,
        "prediction_mode": estimator_mode,
        "estimator_mode": estimator_mode,
        "asg_builder_enabled": is_asg_policy,
        "oracle_future": oracle_future,
        "baseline_class": _baseline_class_for(policy),
        "prompt_reconstruction_quality": prompt_quality,
        "prompt_reconstruction_quality_label": _prompt_quality_label(prompt_quality),
        "observation_compression_class": (
            "asg_specific" if asg_specific_compression else "common" if common_compression else "none"
        ),
        "data_quality": getattr(args, "data_quality", "unknown"),
        "paper_usable": bool(getattr(args, "paper_usable", False)),
        "num_high_quality_traces": int(getattr(args, "num_high_quality_traces", 0)),
        "num_smoke_traces": int(getattr(args, "num_smoke_traces", 0)),
        "num_traces": num_traces,
        "selected_trace_count": selected_trace_count,
        "delayed_reuse_ratio": delayed_reuse_ratio,
        "concurrency": args.concurrency,
        "memory_budget": args.memory_budget_gb,
        "timing": {
            "total_cycles": int(total_cycles),
            "pure_comp_cycles": int(pure_comp_cycles),
            "pure_comm_cycles": int(pure_comm_cycles),
        },
        "agent_metrics": metrics.to_dict(),
        "retained_state_type_distribution": dict(retained_types),
        "LRU_regret_states_preserved": regret_preserved,
        "LRU_regret_tokens_preserved": regret_tokens_preserved,
        "event_stats": {
            "busybarn_events": len(events_dict),
            "asg_states": len(graph.states),
            "asg_execs": len(graph.execs),
        },
    }


def regret_candidate_tokens(trace, horizon: int = 16, min_tokens: int = 128) -> dict:
    state_tokens = {}
    for event in trace:
        if event.get("type") == "state":
            state_tokens[event["state_id"]] = int(event.get("tokens", 1))
        elif event.get("type") == "tool":
            state_tokens[event["new_state_id"]] = int(event.get("output_tokens", 1))
        elif event.get("type") == "llm":
            state_tokens[event["new_state_id"]] = int(event.get("output_tokens", 1))
    llm_steps = [(idx, event) for idx, event in enumerate(trace) if event.get("type") == "llm"]
    candidates = {}
    for pos, (step_idx, event) in enumerate(llm_steps):
        for state_id in event.get("input_state_ids", []):
            future = llm_steps[pos + 1 : pos + horizon + 1]
            reused = any(state_id in future_event.get("input_state_ids", []) for _, future_event in future)
            if reused and state_tokens.get(state_id, 0) >= min_tokens:
                candidates[state_id] = int(state_tokens.get(state_id, 0))
    return candidates


def regret_candidate_ids(trace, horizon: int = 16, min_tokens: int = 128) -> set:
    return set(regret_candidate_tokens(trace, horizon=horizon, min_tokens=min_tokens))


def write_summary_csv(path, results):
    oracle_eff = _metric_for(results, "asg-retention-v2-oracle", "effective_prefill_tokens")
    lru_eff = _metric_for(results, "lru-system", "effective_prefill_tokens")
    fields = [
        "policy",
        "prediction_mode",
        "estimator_mode",
        "asg_builder_enabled",
        "oracle_future",
        "baseline_class",
        "prompt_reconstruction_quality",
        "observation_compression_class",
        "data_quality",
        "paper_usable",
        "num_high_quality_traces",
        "num_smoke_traces",
        "num_traces",
        "selected_trace_count",
        "delayed_reuse_ratio",
        "concurrency",
        "memory_budget",
        "effective_prefill_tokens",
        "effective_prefill_reduction",
        "cache_hit_ratio",
        "cache_byte_miss",
        "model_compute_cycles",
        "model_comm_cycles",
        "remote_read_bytes",
        "demand_migration_bytes",
        "prefetch_migration_bytes",
        "prefetch_hidden_cycles",
        "prefetch_exposed_cycles",
        "unused_prefetch_events",
        "state_misses",
        "local_state_hits",
        "remote_state_hits",
        "retained_state_type_distribution",
        "LRU_regret_states_preserved",
        "LRU_regret_tokens_preserved",
        "oracle_gap",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            metrics = result["agent_metrics"]
            writer.writerow(
                {
                    "policy": result["policy"],
                    "prediction_mode": result["prediction_mode"],
                    "estimator_mode": result["estimator_mode"],
                    "asg_builder_enabled": result["asg_builder_enabled"],
                    "oracle_future": result["oracle_future"],
                    "baseline_class": result["baseline_class"],
                    "prompt_reconstruction_quality": result["prompt_reconstruction_quality_label"],
                    "observation_compression_class": result["observation_compression_class"],
                    "data_quality": result["data_quality"],
                    "paper_usable": result["paper_usable"],
                    "num_high_quality_traces": result["num_high_quality_traces"],
                    "num_smoke_traces": result["num_smoke_traces"],
                    "num_traces": result["num_traces"],
                    "selected_trace_count": result.get("selected_trace_count", ""),
                    "delayed_reuse_ratio": result.get("delayed_reuse_ratio", ""),
                    "concurrency": result["concurrency"],
                    "memory_budget": result["memory_budget"],
                    "effective_prefill_tokens": metrics["effective_prefill_tokens"],
                    "effective_prefill_reduction": metrics["effective_prefill_reduction"],
                    "cache_hit_ratio": metrics["cache_hit_ratio"],
                    "cache_byte_miss": metrics["cache_byte_miss"],
                    "model_compute_cycles": metrics["model_compute_cycles"],
                    "model_comm_cycles": metrics["model_comm_cycles"],
                    "remote_read_bytes": metrics["remote_read_bytes"],
                    "demand_migration_bytes": metrics["demand_migration_bytes"],
                    "prefetch_migration_bytes": metrics["prefetch_migration_bytes"],
                    "prefetch_hidden_cycles": metrics["prefetch_hidden_cycles"],
                    "prefetch_exposed_cycles": metrics["prefetch_exposed_cycles"],
                    "unused_prefetch_events": metrics["unused_prefetch_events"],
                    "state_misses": metrics["state_misses"],
                    "local_state_hits": metrics["local_state_hits"],
                    "remote_state_hits": metrics["remote_state_hits"],
                    "retained_state_type_distribution": json.dumps(result["retained_state_type_distribution"], sort_keys=True),
                    "LRU_regret_states_preserved": result["LRU_regret_states_preserved"],
                    "LRU_regret_tokens_preserved": result["LRU_regret_tokens_preserved"],
                    "oracle_gap": _oracle_gap(metrics["effective_prefill_tokens"], lru_eff, oracle_eff),
                }
            )


def _metric_for(results, policy: str, metric: str):
    for result in results:
        if result["policy"] == policy:
            return result["agent_metrics"][metric]
    return None


def _oracle_gap(value, lru_value, oracle_value):
    if lru_value is None or oracle_value is None or lru_value == oracle_value:
        return ""
    return (value - oracle_value) / (lru_value - oracle_value)


def write_missing_report(args, output_dir: Path) -> Path:
    report = {
        "status": "missing_real_trace_data",
        "trace_dir": args.trace_dir,
        "trace_file": args.trace_file,
        "trace_format": args.trace_format,
        "message": "No loadable real trace files were found. No policy replay results were generated.",
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    local_report = output_dir / "real_trace_missing_report.json"
    local_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    canonical_report = _repo_root() / "agent_results" / "real_trace_missing_report.json"
    canonical_report.parent.mkdir(parents=True, exist_ok=True)
    if canonical_report.resolve() != local_report.resolve():
        canonical_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return local_report


def write_missing_high_quality_report(args, output_dir: Path, audit_report: dict) -> Path:
    report = {
        "status": "skipped_missing_high_quality_traces",
        "trace_dir": args.trace_dir,
        "trace_file": args.trace_file,
        "trace_format": args.trace_format,
        "require_high_quality": bool(args.require_high_quality),
        "allow_smoke_traces": bool(args.allow_smoke_traces),
        "paper_usable": False,
        "message": "No paper-usable high-quality traces were available; no policy replay summary was generated.",
        "audit_summary": {
            "num_loaded_traces": audit_report.get("num_loaded_traces", 0),
            "paper_usable_trace_count": audit_report.get("paper_usable_trace_count", 0),
            "smoke_test_trace_count": audit_report.get("smoke_test_trace_count", 0),
            "unusable_trace_count": audit_report.get("unusable_trace_count", 0),
            "exclusion_reason_counts": audit_report.get("exclusion_reason_counts", {}),
        },
        "manual_steps_required": [
            "Extract raw archives with python -m src.agent.extract_trace_archives.",
            "Fetch or manually download public traces that expose per-step state context or exact prompt state ids.",
            "Run python -m src.agent.build_real_trace_set to populate traces/real_high_quality.",
            "Use --allow-smoke-traces only for developer smoke tests, not paper claims.",
        ],
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "skipped_missing_high_quality_traces.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def select_traces_for_replay(args, traces, output_dir: Path):
    audit_report = audit_traces(
        traces,
        min_turns=args.min_turns if args.min_turns else 4,
        allow_reconstructed_full_history=False,
        delayed_reuse_k=max(1, args.horizon // 2),
    )
    (Path(output_dir) / "input_trace_quality_audit.json").write_text(
        json.dumps(audit_report, indent=2),
        encoding="utf-8",
    )
    high_indices = [
        item["trace_idx"]
        for item in audit_report.get("traces", [])
        if item.get("quality_label") == "high"
    ]
    smoke_indices = [
        item["trace_idx"]
        for item in audit_report.get("traces", [])
        if item.get("quality_label") == "medium"
    ]
    args.num_high_quality_traces = len(high_indices)
    args.num_smoke_traces = len(smoke_indices)
    if args.allow_smoke_traces:
        args.data_quality = "smoke_only" if not high_indices or smoke_indices else "mixed_with_smoke"
        args.paper_usable = False
        return traces, audit_report

    selected = [traces[idx] for idx in high_indices]
    args.data_quality = "paper_usable" if selected else "missing_high_quality"
    args.paper_usable = bool(selected)
    if selected:
        return selected, audit_report
    write_missing_high_quality_report(args, output_dir, audit_report)
    return [], audit_report


def run_suite(args, traces, output_dir: Path, policies=REAL_V2_POLICIES):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_trace = interleave_traces(traces, args.concurrency)
    if not combined_trace:
        raise ValueError("No trace events to replay.")
    train_traces = traces
    if args.train_trace_dir:
        train_traces = load_trace_dir(
            args.train_trace_dir,
            trace_format=args.trace_format,
            max_traces=args.max_traces,
            min_turns=args.min_turns,
            filter_success=args.filter_success,
            reject_accumulated_fallback=args.reject_accumulated_fallback,
        )
    need_trace_stats = any(_estimator_mode_for(policy, args) == PredictionMode.TRACE_STATS for policy in policies)
    stats = fit_trace_stats(train_traces, horizon=args.horizon) if need_trace_stats else None
    predictors = {
        PredictionMode.HEURISTIC: DemandPredictor(
            mode=PredictionMode.HEURISTIC,
            horizon=args.horizon,
            prefill_cycles_per_token=args.prefill_cycles_per_token,
        ),
    }
    if need_trace_stats:
        predictors[PredictionMode.TRACE_STATS] = DemandPredictor(
            mode=PredictionMode.TRACE_STATS,
            horizon=args.horizon,
            prefill_cycles_per_token=args.prefill_cycles_per_token,
            stats=stats,
        )
    diagnostics_predictor = predictors.get(
        args.prediction_mode,
        predictors.get(PredictionMode.HEURISTIC),
    )
    predictor = diagnostics_predictor or DemandPredictor(
        mode=PredictionMode.HEURISTIC,
        horizon=args.horizon,
        prefill_cycles_per_token=args.prefill_cycles_per_token,
    )
    (output_dir / "predictor_errors.json").write_text(
        json.dumps(prediction_diagnostics(combined_trace, predictor, args.horizon), indent=2),
        encoding="utf-8",
    )
    prompt_quality = prompt_quality_summary(combined_trace)
    dataset_profile = profile_traces(traces, delayed_reuse_k=max(1, args.horizon // 2))
    regret_tokens = regret_candidate_tokens(combined_trace, horizon=args.horizon)
    results = []
    for policy in policies:
        result = run_real_policy(
            policy,
            args,
            combined_trace,
            predictors,
            len(traces),
            regret_tokens,
            prompt_quality,
            selected_trace_count=getattr(args, "selected_trace_count", None),
            delayed_reuse_ratio=dataset_profile.get("delayed_reuse_ratio"),
        )
        results.append(result)
        (output_dir / f"{policy}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_summary_csv(output_dir / "summary.csv", results)
    return results


def prediction_diagnostics(trace, predictor: DemandPredictor, horizon: int = 16) -> dict:
    state_meta = {}
    last_access = {}
    access_count = Counter()
    errors = []
    llm_indices = [idx for idx, event in enumerate(trace) if event.get("type") == "llm"]
    for idx, event in enumerate(trace):
        if event.get("type") == "state":
            state_meta[event["state_id"]] = {
                "state_type": event.get("state_type", "unknown"),
                "token_len": int(event.get("tokens", 1)),
                "birth_step": idx,
                "owner": event.get("owner", "shared"),
                "metadata": event.get("metadata") or {},
            }
        elif event.get("type") == "tool":
            state_meta[event["new_state_id"]] = {
                "state_type": event.get("new_state_type", "tool_observation"),
                "token_len": int(event.get("output_tokens", 1)),
                "birth_step": idx,
                "owner": event.get("agent", "agent_0"),
                "metadata": {"tool": event.get("tool", "tool")},
            }
        elif event.get("type") == "llm":
            phase = event.get("phase") or "unknown"
            phase_score = {phase: 1.0}
            future_llms = [trace[fidx] for fidx in llm_indices if idx < fidx <= idx + horizon]
            for state_id in event.get("input_state_ids", []):
                meta = state_meta.get(state_id)
                if meta is None:
                    continue
                state = SimpleNamespace(
                    state_id=state_id,
                    state_type=meta["state_type"],
                    token_len=meta["token_len"],
                    birth_step=meta["birth_step"],
                    last_access=last_access.get(state_id, meta["birth_step"]),
                    access_count=access_count[state_id],
                    owner=meta["owner"],
                    metadata=meta["metadata"],
                )
                pred = predictor.predict(state, None, phase_score, idx, agent=event.get("agent"))
                actual = sum(1 for future in future_llms if state_id in future.get("input_state_ids", []))
                errors.append(pred.expected_future_accesses - actual)
                access_count[state_id] += 1
                last_access[state_id] = idx
            state_meta[event["new_state_id"]] = {
                "state_type": event.get("new_state_type", "assistant_delta"),
                "token_len": int(event.get("output_tokens", 1)),
                "birth_step": idx,
                "owner": event.get("agent", "agent_0"),
                "metadata": {"tool": "llm"},
            }
    if not errors:
        return {"samples": 0, "mae": 0.0, "bias": 0.0}
    return {
        "samples": len(errors),
        "mae": sum(abs(value) for value in errors) / len(errors),
        "bias": sum(errors) / len(errors),
    }


def write_delayed_reuse_subset(args, traces, output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = delayed_reuse_by_trace(traces, delayed_reuse_k=max(1, args.horizon // 2))
    ratios = sorted(row["delayed_reuse_ratio"] for row in rows)
    median_ratio = ratios[len(ratios) // 2] if ratios else 0.0
    selected = [traces[row["trace_idx"]] for row in rows if row["delayed_reuse_ratio"] > median_ratio]
    metadata = {
        "median_delayed_reuse_ratio": median_ratio,
        "selected_trace_count": len(selected),
        "total_trace_count": len(traces),
        "rows": rows,
        "warning": "" if selected else "No trace exceeded the median delayed-reuse ratio; subset replay was skipped.",
    }
    (output_dir / "delayed_reuse_subset.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    subset_output = output_dir / "delayed_reuse_subset"
    subset_output.mkdir(parents=True, exist_ok=True)
    if not selected:
        write_summary_csv(subset_output / "summary.csv", [])
        return
    subset_args = SimpleNamespace(**vars(args))
    subset_args.output_dir = str(subset_output)
    subset_args.selected_trace_count = len(selected)
    run_suite(
        subset_args,
        selected,
        subset_output,
        policies=(
            "lru-system",
            "asg-retention-v2-graph-only",
            "asg-retention-v2-online",
            "asg-retention-v2-oracle",
        ),
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Replay real code-agent traces through Agent-on-Wafer policies.")
    parser.add_argument("--trace-dir")
    parser.add_argument("--trace-file")
    parser.add_argument("--trace-format", choices=("auto", "normalized_jsonl", "normalized_json", "swe_gym", "codetracer", "agentlens", "generic_react_jsonl"), default="auto")
    parser.add_argument("--max-traces", type=int)
    parser.add_argument("--min-turns", type=int, default=0)
    parser.add_argument("--filter-success", choices=("all", "success", "fail"), default="all")
    parser.add_argument("--reject-accumulated-fallback", action="store_true")
    parser.add_argument("--require-high-quality", action="store_true")
    parser.add_argument("--allow-smoke-traces", action="store_true")
    parser.add_argument("--train-trace-dir")
    parser.add_argument("--prediction-mode", choices=(PredictionMode.ORACLE, PredictionMode.HEURISTIC, PredictionMode.TRACE_STATS), default=PredictionMode.HEURISTIC)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--policy-suite", choices=("v2",), default="v2")
    parser.add_argument("--memory-budget-gb", type=float, default=0.5)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cfg", default="src/platform/cfgs/wamis_hd_distributed.cfg")
    parser.add_argument("--topology", choices=("wamis", "tlm"), default="wamis")
    parser.add_argument("--scheduler-seed", type=int, default=0)
    parser.add_argument("--agent-placement", choices=("round_robin", "compact"), default="round_robin")
    parser.add_argument("--per-node-memory-mb", type=float)
    parser.add_argument("--tool-latency-scale", type=int, default=1_000_000)
    parser.add_argument("--hardware-frequency-ghz", type=float, default=1.0)
    parser.add_argument("--effective-bandwidth-bytes-per-cycle", type=float, default=64.0)
    parser.add_argument("--comm-cost-model", choices=("heuristic", "backend"), default="backend")
    parser.add_argument("--knapsack-granularity-mb", type=float, default=16)
    parser.add_argument("--max-future-access-cap", type=int, default=8)
    parser.add_argument("--storage-penalty", type=float, default=0.0)
    parser.add_argument("--knapsack-max-candidates", type=int, default=2048)
    parser.add_argument("--asg-current-prompt-bonus", type=float, default=100.0)
    parser.add_argument("--asg-recency-bonus", type=float, default=1.0)
    parser.add_argument("--max-prefetch-states", type=int, default=2)
    parser.add_argument("--prefetch-reuse-threshold", type=float, default=0.6)
    parser.add_argument("--prefetch-next-use-threshold", type=float, default=2.0)
    parser.add_argument("--max-prefetch-bytes", type=int, default=536870912)
    parser.add_argument("--prefetch-wait-fraction", type=float, default=0.8)
    parser.add_argument("--disable-static-replica", action="store_true")
    parser.add_argument("--enable-common-observation-compression", action="store_true")
    parser.add_argument("--enable-asg-observation-compression", action="store_true")
    parser.add_argument("--disable-observation-compression", action="store_true")
    parser.add_argument("--large-observation-token-threshold", type=int, default=2048)
    parser.add_argument("--observation-compression-ratio", type=float, default=0.25)
    parser.add_argument("--asg-large-observation-token-threshold", type=int, default=512)
    parser.add_argument("--asg-observation-compression-ratio", type=float, default=0.25)
    parser.add_argument("--n-layers", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--dtype-bytes", type=int, default=2)
    parser.add_argument("--prefill-cycles-per-token", type=int, default=1000)
    parser.add_argument("--decode-cycles-per-token", type=int, default=5000)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    traces = load_traces_from_args(args)
    if not traces:
        if args.require_high_quality or not args.allow_smoke_traces:
            missing_report = write_missing_high_quality_report(
                args,
                output_dir,
                {
                    "num_loaded_traces": 0,
                    "paper_usable_trace_count": 0,
                    "smoke_test_trace_count": 0,
                    "unusable_trace_count": 0,
                    "exclusion_reason_counts": {"no_trace_files_loaded": 1},
                },
            )
        else:
            missing_report = write_missing_report(args, output_dir)
        print(f"No loadable real traces found. Wrote missing-data report to {missing_report}")
        return []
    selected_traces, audit_report = select_traces_for_replay(args, traces, output_dir)
    _ = audit_report
    if not selected_traces:
        print(f"No high-quality traces available. Wrote skipped report to {output_dir / 'skipped_missing_high_quality_traces.json'}")
        return []
    results = run_suite(args, selected_traces, output_dir)
    write_delayed_reuse_subset(args, selected_traces, output_dir)
    print(f"Wrote {len(results)} real-trace policy results to {output_dir}")
    print("policy,effective_prefill_tokens,cache_hit_ratio,oracle_gap")
    oracle_eff = _metric_for(results, "asg-retention-v2-oracle", "effective_prefill_tokens")
    lru_eff = _metric_for(results, "lru-system", "effective_prefill_tokens")
    for result in results:
        metrics = result["agent_metrics"]
        print(
            f"{result['policy']},{metrics['effective_prefill_tokens']},{metrics['cache_hit_ratio']},{_oracle_gap(metrics['effective_prefill_tokens'], lru_eff, oracle_eff)}"
        )


if __name__ == "__main__":
    main()
