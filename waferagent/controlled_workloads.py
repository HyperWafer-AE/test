from __future__ import annotations

from dataclasses import dataclass

from waferagent.graph_ir import AgentEdge, AgentGraph, AgentNode, EdgeType, NodeType
from waferagent.utils import sha256_text


@dataclass(frozen=True)
class StrictControlledSharedKVConfig:
    num_jobs: int
    reuse_group_size: int
    shared_prefix_tokens: int
    private_suffix_tokens: int
    decode_tokens: int
    num_agents_per_job: int
    fanin: bool = False
    seed: int = 0


def generate_strict_controlled_shared_kv_graphs(cfg: StrictControlledSharedKVConfig) -> list[AgentGraph]:
    """Generate a controlled workload without min/max overrides.

    Every agent node has exactly the requested shared/private/decode token
    counts. Prefix IDs are shared only within reuse groups.
    """
    if cfg.num_jobs <= 0 or cfg.reuse_group_size <= 0 or cfg.num_agents_per_job <= 0:
        raise ValueError("num_jobs, reuse_group_size, and num_agents_per_job must be positive")
    if cfg.shared_prefix_tokens < 0 or cfg.private_suffix_tokens < 0 or cfg.decode_tokens < 0:
        raise ValueError("token counts must be non-negative")

    graphs: list[AgentGraph] = []
    input_tokens = cfg.shared_prefix_tokens + cfg.private_suffix_tokens
    for job_idx in range(cfg.num_jobs):
        job_id = f"controlled_job_{job_idx}"
        group_id = job_idx // cfg.reuse_group_size
        shared_prefix_id = sha256_text(
            f"strict-controlled|seed={cfg.seed}|group={group_id}|shared={cfg.shared_prefix_tokens}"
        )
        graph = AgentGraph(job_id, "controlled_shared_kv_reuse", cfg.seed)
        worker_ids: list[str] = []
        for agent_idx in range(cfg.num_agents_per_job):
            node_id = f"{job_id}_agent_{agent_idx}"
            prompt_key = f"{job_id}|{node_id}|{shared_prefix_id}"
            graph.add_node(
                AgentNode(
                    node_id=node_id,
                    job_id=job_id,
                    agent_id=f"agent_{agent_idx}",
                    round_id=0,
                    node_type=NodeType.LLM_CALL,
                    role="controlled_worker",
                    input_token_len=input_tokens,
                    expected_output_token_len=cfg.decode_tokens,
                    actual_output_token_len=cfg.decode_tokens,
                    shared_prefix_ids=[shared_prefix_id] if cfg.shared_prefix_tokens else [],
                    private_prefix_ids=[sha256_text(prompt_key + "|private")],
                    shared_prefix_token_len=cfg.shared_prefix_tokens,
                    private_prefix_token_len=cfg.private_suffix_tokens,
                    prompt_hash=sha256_text(prompt_key),
                    metadata={
                        "controlled_shared_prefix_tokens": cfg.shared_prefix_tokens,
                        "controlled_private_suffix_tokens": cfg.private_suffix_tokens,
                        "controlled_decode_tokens": cfg.decode_tokens,
                        "controlled_reuse_group_id": group_id,
                        "controlled_reuse_group_size": cfg.reuse_group_size,
                    },
                )
            )
            worker_ids.append(node_id)
        if cfg.fanin and worker_ids:
            agg_id = f"{job_id}_aggregator"
            prompt_key = f"{job_id}|{agg_id}|{shared_prefix_id}"
            graph.add_node(
                AgentNode(
                    node_id=agg_id,
                    job_id=job_id,
                    agent_id="aggregator",
                    round_id=1,
                    node_type=NodeType.AGGREGATE,
                    role="controlled_aggregator",
                    input_token_len=input_tokens + cfg.num_agents_per_job * cfg.decode_tokens,
                    expected_output_token_len=cfg.decode_tokens,
                    actual_output_token_len=cfg.decode_tokens,
                    shared_prefix_ids=[shared_prefix_id] if cfg.shared_prefix_tokens else [],
                    private_prefix_ids=[sha256_text(prompt_key + "|private")],
                    shared_prefix_token_len=cfg.shared_prefix_tokens,
                    private_prefix_token_len=cfg.private_suffix_tokens + cfg.num_agents_per_job * cfg.decode_tokens,
                    prompt_hash=sha256_text(prompt_key),
                    metadata={"controlled_reuse_group_id": group_id},
                )
            )
            for src in worker_ids:
                graph.add_edge(AgentEdge(src=src, dst=agg_id, edge_type=EdgeType.GENERATED_MESSAGE, message_token_len=cfg.decode_tokens))
        graph.critical_path_lengths()
        graphs.append(graph)
    return graphs


def strict_controlled_validation_rows(
    graphs: list[AgentGraph],
    cfg: StrictControlledSharedKVConfig,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    expected_unique_prefixes = (cfg.num_jobs + cfg.reuse_group_size - 1) // cfg.reuse_group_size
    prefixes = {
        pid
        for graph in graphs
        for node in graph.nodes.values()
        for pid in node.shared_prefix_ids
    }
    for graph in graphs:
        for node in graph.nodes.values():
            if node.role == "controlled_aggregator":
                continue
            rows.append(
                {
                    "job_id": graph.graph_id,
                    "node_id": node.node_id,
                    "requested_shared_prefix_tokens": cfg.shared_prefix_tokens,
                    "actual_shared_prefix_tokens": node.shared_prefix_token_len,
                    "requested_private_suffix_tokens": cfg.private_suffix_tokens,
                    "actual_private_suffix_tokens": node.private_prefix_token_len,
                    "requested_decode_tokens": cfg.decode_tokens,
                    "actual_decode_tokens": node.actual_output_token_len,
                    "requested_reuse_group_size": cfg.reuse_group_size,
                    "unique_prefixes_observed": len(prefixes),
                    "expected_unique_prefixes": expected_unique_prefixes,
                    "pass": (
                        node.shared_prefix_token_len == cfg.shared_prefix_tokens
                        and node.private_prefix_token_len == cfg.private_suffix_tokens
                        and node.actual_output_token_len == cfg.decode_tokens
                        and len(prefixes) == expected_unique_prefixes
                    ),
                }
            )
    return rows
