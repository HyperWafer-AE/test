#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from waferagent.arrival import ArrivalConfig
from waferagent.controlled_workloads import (
    StrictControlledSharedKVConfig,
    generate_strict_controlled_shared_kv_graphs,
    strict_controlled_validation_rows,
    strict_controlled_validation_summary_rows,
)
from waferagent.global_simulator import simulate_global
from waferagent.graph_ir import AgentEdge, AgentGraph
from waferagent.llm_runner import RunnerConfig
from waferagent.mesh import MeshConfig
from waferagent.trace_collector import collect_graph_traces
from waferagent.utils import enforce_clean_git_tree, finalize_run_dir, init_run_dir, write_json


def _clone_graph_with_prefix(graph: AgentGraph, tag: str, job_offset: int) -> AgentGraph:
    """Clone a controlled graph with disjoint job/node IDs while preserving prefix IDs."""

    new_job_id = f"{tag}_job_{job_offset}"
    node_map = {node_id: f"{tag}_{node_id}" for node_id in graph.nodes}
    cloned = AgentGraph(new_job_id, f"{graph.workload}_{tag}", graph.seed)
    for old_id, node in graph.nodes.items():
        new_id = node_map[old_id]
        cloned.add_node(
            replace(
                node,
                node_id=new_id,
                job_id=new_job_id,
                agent_id=f"{tag}_{node.agent_id}",
                deps=[node_map[d] for d in node.deps],
                prompt_hash=f"{tag}:{node.prompt_hash}",
                private_prefix_ids=[f"{tag}:{pid}" for pid in node.private_prefix_ids],
                metadata={**node.metadata, "adaptive_mixed_regime": tag},
            )
        )
    for edge in graph.edges:
        cloned.add_edge(
            AgentEdge(
                src=node_map[edge.src],
                dst=node_map[edge.dst],
                edge_type=edge.edge_type,
                message_token_len=edge.message_token_len,
                shared_kv_bytes=edge.shared_kv_bytes,
                metadata={**edge.metadata, "adaptive_mixed_regime": tag},
            )
        )
    cloned.critical_path_lengths()
    return cloned


def _build_mixed_graphs(num_jobs_per_regime: int, seed: int) -> tuple[list[AgentGraph], list[dict[str, object]], list[dict[str, object]]]:
    high_opportunity = StrictControlledSharedKVConfig(
        num_jobs=num_jobs_per_regime,
        reuse_group_size=max(2, min(8, num_jobs_per_regime)),
        shared_prefix_tokens=8192,
        private_suffix_tokens=128,
        decode_tokens=512,
        num_agents_per_job=2,
        fanin=False,
        seed=seed,
    )
    queue_risk_fallback = StrictControlledSharedKVConfig(
        num_jobs=num_jobs_per_regime,
        reuse_group_size=max(2, min(8, num_jobs_per_regime)),
        shared_prefix_tokens=32768,
        private_suffix_tokens=128,
        decode_tokens=512,
        num_agents_per_job=16,
        fanin=False,
        seed=seed + 1,
    )
    graphs: list[AgentGraph] = []
    validation_summary: list[dict[str, object]] = []
    validation_nodes: list[dict[str, object]] = []
    offset = 0
    for tag, cfg in [("highop", high_opportunity), ("fallback", queue_risk_fallback)]:
        raw_graphs = generate_strict_controlled_shared_kv_graphs(cfg)
        validation_summary.extend(
            {**row, "adaptive_mixed_regime": tag}
            for row in strict_controlled_validation_summary_rows(raw_graphs, cfg)
        )
        validation_nodes.extend(
            {**row, "adaptive_mixed_regime": tag}
            for row in strict_controlled_validation_rows(raw_graphs, cfg)
        )
        for idx, graph in enumerate(raw_graphs):
            graphs.append(_clone_graph_with_prefix(graph, tag, offset + idx))
        offset += len(raw_graphs)
    return graphs, validation_summary, validation_nodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wafer-config", default="configs/wafer/wse_like.yaml")
    parser.add_argument("--arrival-mode", default="poisson", choices=["closed_loop", "poisson", "burst", "replay"])
    parser.add_argument("--arrival-rate-jobs-per-s", type=float, default=4.0)
    parser.add_argument("--num-jobs-per-regime", type=int, default=8)
    parser.add_argument("--baselines", default="apc_like,waferagent_latency_safe,waferagent_adaptive")
    parser.add_argument("--duration-source", default="synthetic", choices=["trace", "calibrated", "synthetic"])
    parser.add_argument("--shared-attention-cost-fit", required=True)
    parser.add_argument("--shared-attention-accounting", default="cohort_stage", choices=["stage_amortized", "cohort_stage", "per_member"])
    parser.add_argument("--out", default="results/round12_adaptive_policy_sweep")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--clean-required", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    enforce_clean_git_tree(args.clean_required, args.allow_dirty)
    out = init_run_dir(args.out, {"run_type": "adaptive_policy_sweep", **vars(args)})
    mesh = MeshConfig.from_yaml(args.wafer_config)
    graphs, validation_summary, validation_nodes = _build_mixed_graphs(args.num_jobs_per_regime, args.seed)
    traces_path = out / "traces" / "traces.jsonl"
    traces = collect_graph_traces(
        graphs,
        "round12_adaptive_policy_sweep",
        RunnerConfig(engine="synthetic", seed=args.seed),
        out_jsonl=traces_path,
    )
    result = simulate_global(
        traces,
        mesh,
        [x.strip() for x in args.baselines.split(",") if x.strip()],
        ArrivalConfig(
            mode=args.arrival_mode,
            rate_jobs_per_s=args.arrival_rate_jobs_per_s,
            seed=args.seed,
            max_jobs=len({tr.job_id for tr in traces}),
        ),
        seed=args.seed,
        duration_source=args.duration_source,
        shared_attention_cost_fit=args.shared_attention_cost_fit,
        shared_attention_accounting=args.shared_attention_accounting,
    )
    sim = out / "simulation"
    for name, df in result.items():
        df.to_csv(sim / f"{name}.csv", index=False)
    pd.DataFrame(validation_summary).to_csv(sim / "controlled_workload_validation.csv", index=False)
    pd.DataFrame(validation_nodes).to_csv(sim / "controlled_workload_validation_nodes.csv", index=False)
    write_json(
        out / "model_selection.json",
        {"engine_used": "synthetic", "fallback_count": 0, "adaptive_policy_sweep": True},
    )
    finalize_run_dir(out)
    print(f"Adaptive policy sweep complete: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
