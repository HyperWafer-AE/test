import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path


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
        os.execv(str(venv_python), [str(venv_python), "-m", "src.agent.experiment"] + sys.argv[1:])


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
from .metrics import postprocess_agent_events
from .model_profile import ModelProfile
from .state_manager import (
    ASGPlacementStateManager,
    ASGPrefetchStateManager,
    ASGRetentionStateManager,
    ASGStateManager,
    KVFlowLikeStateManager,
    LRUStateManager,
    NoCacheStateManager,
)
from .workload import generate_synthetic_trace, load_trace_json, save_trace_json


POLICIES = (
    "nocache",
    "lru",
    "kvflow",
    "asg-retention",
    "asg-placement",
    "asg-prefetch",
    "asg",
)
RUN_ALL_POLICIES = ("nocache", "lru", "kvflow", "asg-retention", "asg-placement", "asg-prefetch")


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


def build_state_manager(policy: str, memory_budget_bytes: int, model: ModelProfile):
    if policy == "nocache":
        return NoCacheStateManager(memory_budget_bytes)
    if policy == "lru":
        return LRUStateManager(memory_budget_bytes)
    if policy == "kvflow":
        return KVFlowLikeStateManager(memory_budget_bytes)
    manager_cls = {
        "asg-retention": ASGRetentionStateManager,
        "asg-placement": ASGPlacementStateManager,
        "asg-prefetch": ASGPrefetchStateManager,
        "asg": ASGPrefetchStateManager,
    }.get(policy)
    if manager_cls is not None:
        manager = manager_cls(
            memory_budget_bytes,
            prefill_cycles_per_token=model.prefill_cycles_per_token,
        )
        if policy == "asg":
            manager.policy_name = "asg"
        return manager
    raise ValueError(f"Unsupported policy: {policy}")


def workload_stats(trace):
    return {
        "num_events": len(trace),
        "num_state_events": sum(1 for event in trace if event.get("type") == "state"),
        "num_llm_events": sum(1 for event in trace if event.get("type") == "llm"),
        "num_tool_events": sum(1 for event in trace if event.get("type") == "tool"),
        "total_append_tokens": sum(int(event.get("append_tokens", 0)) for event in trace),
        "llm_output_tokens": sum(int(event.get("output_tokens", 0)) for event in trace if event.get("type") == "llm"),
        "tool_output_tokens": sum(int(event.get("output_tokens", 0)) for event in trace if event.get("type") == "tool"),
    }


def run_policy(policy: str, args, trace):
    model = build_model(args)
    hardware = build_hardware(args.cfg, args.topology)
    random.seed(args.scheduler_seed)
    np.random.seed(args.scheduler_seed)
    memory_budget_bytes = int(args.memory_budget_gb * (1024 ** 3))
    state_manager = build_state_manager(policy, memory_budget_bytes, model)
    active_nodes = max(1, len(getattr(hardware, "nodes_set", [])))
    if args.per_node_memory_mb is None:
        per_node_memory_mb = max(16, int(args.memory_budget_gb * 1024 / active_nodes))
    else:
        per_node_memory_mb = args.per_node_memory_mb
    topology_enabled = not args.disable_topology_placement and policy in {"asg-placement", "asg-prefetch", "asg"}
    prefetch_enabled = not args.disable_prefetch and policy in {"asg-prefetch", "asg"}
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
        enable_observation_compression=args.enable_observation_compression,
        large_observation_token_threshold=args.large_observation_token_threshold,
        observation_compression_ratio=args.observation_compression_ratio,
    )
    events_dict, graph, metrics = builder.build(trace)
    total_cycles, pure_comp_cycles, pure_comm_cycles = collective_event_driver(events_dict, hardware)
    postprocess_agent_events(events_dict, metrics)
    return {
        "policy": policy,
        "args": vars(args),
        "timing": {
            "total_cycles": int(total_cycles),
            "pure_comp_cycles": int(pure_comp_cycles),
            "pure_comm_cycles": int(pure_comm_cycles),
        },
        "agent_metrics": metrics.to_dict(),
        "workload_stats": workload_stats(trace),
        "event_stats": {
            "busybarn_events": len(events_dict),
            "asg_states": len(graph.states),
            "asg_execs": len(graph.execs),
        },
    }


def write_json(path, data):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_summary_csv(path, results):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "policy",
        "total_cycles",
        "pure_comp_cycles",
        "pure_comm_cycles",
        "effective_prefill_tokens",
        "total_input_tokens",
        "append_tokens",
        "decode_tokens",
        "llm_output_tokens",
        "tool_output_tokens",
        "cache_hit_ratio",
        "effective_prefill_reduction",
        "kv_migration_bytes",
        "demand_migration_bytes",
        "prefetch_migration_bytes",
        "num_kv_migrations",
        "num_prefetch_events",
        "local_state_hits",
        "remote_state_hits",
        "state_misses",
        "tool_wait_cycles",
        "tool_wait_overlap_ratio",
        "remote_read_bytes",
        "num_remote_reads",
        "remote_read_cycles",
        "migration_skipped_by_cost",
        "static_replica_bytes",
        "num_static_replicas",
        "unused_prefetch_events",
        "migration_cost_estimate_cycles",
        "remote_read_cost_estimate_cycles",
        "busybarn_events",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            metrics = result["agent_metrics"]
            timing = result["timing"]
            writer.writerow(
                {
                    "policy": result["policy"],
                    "total_cycles": timing["total_cycles"],
                    "pure_comp_cycles": timing["pure_comp_cycles"],
                    "pure_comm_cycles": timing["pure_comm_cycles"],
                    "effective_prefill_tokens": metrics["effective_prefill_tokens"],
                    "total_input_tokens": metrics["total_input_tokens"],
                    "append_tokens": metrics["append_tokens"],
                    "decode_tokens": metrics["decode_tokens"],
                    "llm_output_tokens": metrics["llm_output_tokens"],
                    "tool_output_tokens": metrics["tool_output_tokens"],
                    "cache_hit_ratio": metrics["cache_hit_ratio"],
                    "effective_prefill_reduction": metrics["effective_prefill_reduction"],
                    "kv_migration_bytes": metrics["kv_migration_bytes"],
                    "demand_migration_bytes": metrics["demand_migration_bytes"],
                    "prefetch_migration_bytes": metrics["prefetch_migration_bytes"],
                    "num_kv_migrations": metrics["num_kv_migrations"],
                    "num_prefetch_events": metrics["num_prefetch_events"],
                    "local_state_hits": metrics["local_state_hits"],
                    "remote_state_hits": metrics["remote_state_hits"],
                    "state_misses": metrics["state_misses"],
                    "tool_wait_cycles": metrics["tool_wait_cycles"],
                    "tool_wait_overlap_ratio": metrics["tool_wait_overlap_ratio"],
                    "remote_read_bytes": metrics["remote_read_bytes"],
                    "num_remote_reads": metrics["num_remote_reads"],
                    "remote_read_cycles": metrics["remote_read_cycles"],
                    "migration_skipped_by_cost": metrics["migration_skipped_by_cost"],
                    "static_replica_bytes": metrics["static_replica_bytes"],
                    "num_static_replicas": metrics["num_static_replicas"],
                    "unused_prefetch_events": metrics["unused_prefetch_events"],
                    "migration_cost_estimate_cycles": metrics["migration_cost_estimate_cycles"],
                    "remote_read_cost_estimate_cycles": metrics["remote_read_cost_estimate_cycles"],
                    "busybarn_events": result["event_stats"]["busybarn_events"],
                }
            )


def build_trace(args):
    if args.trace_json:
        return load_trace_json(args.trace_json)
    trace = generate_synthetic_trace(
        num_agents=args.num_agents,
        turns_per_agent=args.turns,
        shared_prefix_tokens=args.shared_prefix_tokens,
        role_tokens=args.role_tokens,
        append_tokens_mean=args.append_tokens_mean,
        output_tokens_mean=args.output_tokens_mean,
        tool_probability=args.tool_probability,
        tool_latency_mean=args.tool_latency_mean,
        tool_output_tokens_mean=args.tool_output_tokens_mean,
        failure_probability=args.failure_probability,
        seed=args.seed,
        shared_state_probability=args.shared_state_probability,
        cross_agent_share_probability=args.cross_agent_share_probability,
        tool_output_shared_probability=args.tool_output_shared_probability,
        agent_handoff_probability=args.agent_handoff_probability,
        large_observation_probability=args.large_observation_probability,
    )
    if args.save_trace:
        save_trace_json(trace, args.save_trace)
    return trace


def _flag_present(argv, flag):
    return any(item == flag or item.startswith(f"{flag}=") for item in argv)


def _apply_stress_defaults(args, argv):
    if not args.stress_placement:
        return args
    if not _flag_present(argv, "--num-agents"):
        args.num_agents = max(args.num_agents, 8)
    if not _flag_present(argv, "--cross-agent-share-probability"):
        args.cross_agent_share_probability = 0.35
    if not _flag_present(argv, "--agent-handoff-probability"):
        args.agent_handoff_probability = 0.25
    if not _flag_present(argv, "--large-observation-probability"):
        args.large_observation_probability = 0.25
    if not _flag_present(argv, "--tool-probability"):
        args.tool_probability = 0.7
    if not _flag_present(argv, "--memory-budget-gb"):
        args.memory_budget_gb = min(args.memory_budget_gb, 0.5)
    return args


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Run Agent-on-Wafer MVP experiments on BusyBarn.")
    parser.add_argument("--policy", choices=POLICIES, default="asg")
    parser.add_argument("--run-all-policies", action="store_true")
    parser.add_argument("--cfg", default="src/platform/cfgs/wamis_hd_distributed.cfg")
    parser.add_argument("--topology", choices=("wamis", "tlm"), default="wamis")
    parser.add_argument("--num-agents", type=int, default=8)
    parser.add_argument("--turns", type=int, default=32)
    parser.add_argument("--memory-budget-gb", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--scheduler-seed", type=int, default=0)
    parser.add_argument("--output", default="agent_results/asg.json")
    parser.add_argument("--output-dir", default="agent_results")
    parser.add_argument("--trace-json")
    parser.add_argument("--save-trace")
    parser.add_argument("--disable-prefetch", action="store_true")
    parser.add_argument("--disable-topology-placement", action="store_true")
    parser.add_argument("--agent-placement", choices=("round_robin", "compact"), default="round_robin")
    parser.add_argument("--per-node-memory-mb", type=float)
    parser.add_argument("--max-prefetch-states", type=int, default=2)
    parser.add_argument("--stress-placement", action="store_true")
    parser.add_argument("--tool-latency-scale", type=int, default=1_000_000)
    parser.add_argument("--hardware-frequency-ghz", type=float, default=1.0)
    parser.add_argument("--effective-bandwidth-bytes-per-cycle", type=float, default=64.0)
    parser.add_argument("--prefetch-reuse-threshold", type=float, default=0.6)
    parser.add_argument("--prefetch-next-use-threshold", type=float, default=2.0)
    parser.add_argument("--max-prefetch-bytes", type=int, default=536870912)
    parser.add_argument("--prefetch-wait-fraction", type=float, default=0.8)
    parser.add_argument("--enable-observation-compression", action="store_true")
    parser.add_argument("--large-observation-token-threshold", type=int, default=2048)
    parser.add_argument("--observation-compression-ratio", type=float, default=0.25)

    parser.add_argument("--shared-prefix-tokens", type=int, default=1024)
    parser.add_argument("--role-tokens", type=int, default=128)
    parser.add_argument("--append-tokens-mean", type=int, default=128)
    parser.add_argument("--output-tokens-mean", type=int, default=96)
    parser.add_argument("--tool-probability", type=float, default=0.55)
    parser.add_argument("--tool-latency-mean", type=int, default=1000)
    parser.add_argument("--tool-output-tokens-mean", type=int, default=256)
    parser.add_argument("--failure-probability", type=float, default=0.12)
    parser.add_argument("--shared-state-probability", type=float, default=0.15)
    parser.add_argument("--cross-agent-share-probability", type=float, default=0.10)
    parser.add_argument("--tool-output-shared-probability", type=float, default=0.20)
    parser.add_argument("--agent-handoff-probability", type=float, default=0.08)
    parser.add_argument("--large-observation-probability", type=float, default=0.10)

    parser.add_argument("--n-layers", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--dtype-bytes", type=int, default=2)
    parser.add_argument("--prefill-cycles-per-token", type=int, default=1000)
    parser.add_argument("--decode-cycles-per-token", type=int, default=5000)
    return _apply_stress_defaults(parser.parse_args(argv), argv)


def print_summary(results):
    print("policy,total_cycles,effective_prefill_tokens,kv_migration_bytes,prefetch_kv_bytes,pure_comm_cycles")
    for result in results:
        metrics = result["agent_metrics"]
        timing = result["timing"]
        print(
            "{policy},{total},{prefill},{migration},{prefetch},{comm}".format(
                policy=result["policy"],
                total=timing["total_cycles"],
                prefill=metrics["effective_prefill_tokens"],
                migration=metrics["kv_migration_bytes"],
                prefetch=metrics["prefetch_kv_bytes"],
                comm=timing["pure_comm_cycles"],
            )
        )


def main(argv=None):
    args = parse_args(argv)
    trace = build_trace(args)
    if args.run_all_policies:
        output_dir = Path(args.output_dir)
        results = []
        for policy in RUN_ALL_POLICIES:
            result = run_policy(policy, args, trace)
            results.append(result)
            write_json(output_dir / f"{policy}.json", result)
        write_summary_csv(output_dir / "summary.csv", results)
        print(f"Wrote {len(results)} policy results to {output_dir}")
        print_summary(results)
    else:
        result = run_policy(args.policy, args, trace)
        write_json(args.output, result)
        print(f"Wrote {args.policy} result to {args.output}")
        print(
            "total_cycles={total_cycles} effective_prefill_tokens={prefill}".format(
                total_cycles=result["timing"]["total_cycles"],
                prefill=result["agent_metrics"]["effective_prefill_tokens"],
            )
        )
        print_summary([result])


if __name__ == "__main__":
    main()
