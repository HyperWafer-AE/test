from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

from waferagent.graph_ir import AgentEdge, AgentGraph, AgentNode, EdgeType, NodeType
from waferagent.utils import sha256_text, stable_rng_seed


@dataclass(frozen=True)
class WorkloadParams:
    workload: str
    job_id: str
    seed: int = 0
    num_agents: int = 4
    num_rounds: int = 1
    shared_prefix_ratio: float = 0.5
    input_len: int = 2048
    output_len: int = 128
    num_layers: int = 3
    width: int = 4
    fan_in_policy: str = "all_to_all"
    num_workers: int = 4
    num_tools_per_worker: int = 1
    tool_latency_distribution: str = "fixed"
    mean_tool_latency_ms: float = 1000.0
    message_token_len: int | None = None


def _node(
    params: WorkloadParams,
    graph: AgentGraph,
    node_id: str,
    agent_id: str,
    round_id: int,
    node_type: NodeType,
    role: str,
    shared_prefix_id: str,
    input_len: int | None = None,
    output_len: int | None = None,
    tool_latency_ms: float = 0.0,
    deps: list[str] | None = None,
) -> AgentNode:
    in_len = int(input_len if input_len is not None else params.input_len)
    out_len = int(output_len if output_len is not None else params.output_len)
    shared_tokens = max(0, min(in_len, int(round(in_len * params.shared_prefix_ratio))))
    private_tokens = max(0, in_len - shared_tokens)
    prompt_key = f"{params.workload}|{params.job_id}|{node_id}|{shared_prefix_id}"
    n = AgentNode(
        node_id=node_id,
        job_id=params.job_id,
        agent_id=agent_id,
        round_id=round_id,
        node_type=node_type,
        role=role,
        input_token_len=in_len,
        expected_output_token_len=out_len,
        actual_output_token_len=out_len,
        shared_prefix_ids=[shared_prefix_id] if shared_tokens else [],
        private_prefix_ids=[sha256_text(prompt_key + "|private")],
        deps=list(deps or []),
        tool_latency_ms=tool_latency_ms,
        shared_prefix_token_len=shared_tokens,
        private_prefix_token_len=private_tokens,
        prompt_hash=sha256_text(prompt_key),
    )
    graph.add_node(n)
    return n


def _edge(
    graph: AgentGraph,
    src: str,
    dst: str,
    edge_type: EdgeType = EdgeType.GENERATED_MESSAGE,
    message_token_len: int = 0,
) -> None:
    graph.add_edge(AgentEdge(src=src, dst=dst, edge_type=edge_type, message_token_len=message_token_len))


def _tool_latency(params: WorkloadParams, rng: random.Random) -> float:
    mean = max(0.0, params.mean_tool_latency_ms)
    if params.tool_latency_distribution == "lognormal":
        sigma = 0.8
        mu = math.log(max(1.0, mean)) - 0.5 * sigma * sigma
        return float(rng.lognormvariate(mu, sigma))
    if params.tool_latency_distribution == "pareto":
        alpha = 2.0
        xm = mean * (alpha - 1) / alpha if mean else 0.0
        return float(xm / (rng.random() ** (1.0 / alpha))) if xm else 0.0
    return mean


def debate(params: WorkloadParams) -> AgentGraph:
    rng = random.Random(stable_rng_seed(params.seed, params.job_id, "debate"))
    graph = AgentGraph(params.job_id, "debate", params.seed)
    shared = sha256_text(f"debate|{params.job_id}|shared-task")
    previous_layer: list[str] = []
    for rnd in range(params.num_rounds):
        proposers: list[str] = []
        critics: list[str] = []
        for i in range(params.num_agents):
            node_id = f"{params.job_id}_r{rnd}_proposer_{i}"
            _node(params, graph, node_id, f"proposer_{i}", rnd, NodeType.LLM_CALL, "proposer", shared)
            for dep in previous_layer:
                _edge(graph, dep, node_id, EdgeType.GENERATED_MESSAGE, params.output_len)
            proposers.append(node_id)
        for i in range(params.num_agents):
            node_id = f"{params.job_id}_r{rnd}_critic_{i}"
            _node(
                params,
                graph,
                node_id,
                f"critic_{i}",
                rnd,
                NodeType.LLM_CALL,
                "critic",
                shared,
                input_len=params.input_len + params.output_len,
            )
            _edge(graph, proposers[i % len(proposers)], node_id, EdgeType.GENERATED_MESSAGE, params.output_len)
            critics.append(node_id)
        agg_id = f"{params.job_id}_r{rnd}_judge"
        _node(
            params,
            graph,
            agg_id,
            "judge",
            rnd,
            NodeType.AGGREGATE,
            "judge",
            shared,
            input_len=params.input_len + params.num_agents * params.output_len,
            output_len=max(32, params.output_len // 2),
        )
        for dep in critics:
            _edge(graph, dep, agg_id, EdgeType.GENERATED_MESSAGE, params.output_len)
        previous_layer = [agg_id]
    graph.critical_path_lengths()
    return graph


def moa(params: WorkloadParams) -> AgentGraph:
    graph = AgentGraph(params.job_id, "moa", params.seed)
    shared = sha256_text(f"moa|{params.job_id}|shared-task")
    previous: list[str] = []
    width = params.width or params.num_agents
    for layer in range(params.num_layers):
        current: list[str] = []
        layer_width = 1 if layer == params.num_layers - 1 else width
        for i in range(layer_width):
            role = "synthesizer" if layer == params.num_layers - 1 else f"layer_{layer}"
            node_id = f"{params.job_id}_l{layer}_agent_{i}"
            dep_tokens = len(previous) * params.output_len
            _node(
                params,
                graph,
                node_id,
                f"{role}_{i}",
                layer,
                NodeType.SUMMARIZE if role == "synthesizer" else NodeType.LLM_CALL,
                role,
                shared,
                input_len=params.input_len + dep_tokens,
                output_len=max(32, params.output_len // 2) if role == "synthesizer" else params.output_len,
            )
            deps = previous
            if params.fan_in_policy == "top_k":
                deps = previous[: max(1, min(2, len(previous)))]
            elif params.fan_in_policy == "tree" and previous:
                deps = [previous[i % len(previous)]]
            for dep in deps:
                _edge(graph, dep, node_id, EdgeType.GENERATED_MESSAGE, params.output_len)
            current.append(node_id)
        previous = current
    graph.critical_path_lengths()
    return graph


def planner_worker_tool(params: WorkloadParams) -> AgentGraph:
    rng = random.Random(stable_rng_seed(params.seed, params.job_id, "planner_worker_tool"))
    graph = AgentGraph(params.job_id, "planner_worker_tool", params.seed)
    shared = sha256_text(f"planner|{params.job_id}|shared-task")
    planner = f"{params.job_id}_planner"
    _node(params, graph, planner, "planner", 0, NodeType.LLM_CALL, "planner", shared)
    workers_done: list[str] = []
    for w in range(params.num_workers):
        worker = f"{params.job_id}_worker_{w}_start"
        _node(params, graph, worker, f"worker_{w}", 0, NodeType.LLM_CALL, "worker", shared)
        _edge(graph, planner, worker, EdgeType.GENERATED_MESSAGE, params.output_len)
        last = worker
        for t in range(params.num_tools_per_worker):
            tool = f"{params.job_id}_worker_{w}_tool_{t}"
            _node(
                params,
                graph,
                tool,
                f"worker_{w}_tool",
                t,
                NodeType.TOOL_CALL,
                "tool",
                shared,
                input_len=32,
                output_len=16,
                tool_latency_ms=_tool_latency(params, rng),
            )
            _edge(graph, last, tool, EdgeType.CONTROL_DEPENDENCY, 16)
            cont = f"{params.job_id}_worker_{w}_after_tool_{t}"
            _node(
                params,
                graph,
                cont,
                f"worker_{w}",
                t + 1,
                NodeType.LLM_CALL,
                "worker_continue",
                shared,
                input_len=params.input_len + 64,
            )
            _edge(graph, tool, cont, EdgeType.TOOL_RESULT, 16)
            last = cont
        workers_done.append(last)
    agg = f"{params.job_id}_planner_aggregate"
    _node(
        params,
        graph,
        agg,
        "planner",
        1,
        NodeType.AGGREGATE,
        "planner_aggregate",
        shared,
        input_len=params.input_len + params.num_workers * params.output_len,
        output_len=params.output_len,
    )
    for dep in workers_done:
        _edge(graph, dep, agg, EdgeType.GENERATED_MESSAGE, params.output_len)
    final = f"{params.job_id}_final"
    _node(params, graph, final, "final", 2, NodeType.SUMMARIZE, "final", shared, output_len=64)
    _edge(graph, agg, final, EdgeType.GENERATED_MESSAGE, params.output_len)
    graph.critical_path_lengths()
    return graph


def swe_like(params: WorkloadParams) -> AgentGraph:
    graph = AgentGraph(params.job_id, "swe_like", params.seed)
    shared = sha256_text(f"swe|{params.job_id}|repo-context")
    p = WorkloadParams(**{**params.__dict__, "shared_prefix_ratio": params.shared_prefix_ratio})
    planner = f"{p.job_id}_planner"
    _node(p, graph, planner, "planner", 0, NodeType.LLM_CALL, "planner", shared, input_len=p.input_len)
    readers = []
    for i in range(max(2, p.num_workers)):
        node_id = f"{p.job_id}_reader_{i}"
        _node(p, graph, node_id, f"reader_{i}", 0, NodeType.LLM_CALL, "reader", shared, input_len=p.input_len)
        _edge(graph, planner, node_id, EdgeType.CONTROL_DEPENDENCY, 32)
        readers.append(node_id)
    patcher = f"{p.job_id}_patch_proposer"
    _node(
        p,
        graph,
        patcher,
        "patcher",
        1,
        NodeType.LLM_CALL,
        "patch_proposer",
        shared,
        input_len=p.input_len + len(readers) * p.output_len,
        output_len=p.output_len * 2,
    )
    for r in readers:
        _edge(graph, r, patcher, EdgeType.GENERATED_MESSAGE, p.output_len)
    reviewer = f"{p.job_id}_reviewer"
    _node(p, graph, reviewer, "reviewer", 2, NodeType.VERIFY, "reviewer", shared, output_len=64)
    _edge(graph, patcher, reviewer, EdgeType.GENERATED_MESSAGE, p.output_len)
    final = f"{p.job_id}_final_patch_summary"
    _node(p, graph, final, "summarizer", 3, NodeType.SUMMARIZE, "final_patch_summarizer", shared, output_len=64)
    _edge(graph, reviewer, final, EdgeType.QUALITY_DEPENDENCY, 64)
    graph.critical_path_lengths()
    return graph


def rag_like(params: WorkloadParams) -> AgentGraph:
    graph = AgentGraph(params.job_id, "rag_like", params.seed)
    shared = sha256_text(f"rag|{params.job_id}|evidence")
    retriever = f"{params.job_id}_retriever"
    _node(
        params,
        graph,
        retriever,
        "retriever",
        0,
        NodeType.TOOL_CALL,
        "retriever",
        shared,
        input_len=64,
        output_len=max(64, params.output_len),
        tool_latency_ms=params.mean_tool_latency_ms,
    )
    domain_agents = []
    for i in range(params.num_agents):
        node_id = f"{params.job_id}_domain_{i}"
        _node(
            params,
            graph,
            node_id,
            f"domain_{i}",
            1,
            NodeType.LLM_CALL,
            "domain_agent",
            shared,
            input_len=params.input_len + params.output_len,
        )
        _edge(graph, retriever, node_id, EdgeType.TOOL_RESULT, params.output_len)
        domain_agents.append(node_id)
    critic = f"{params.job_id}_critic"
    _node(
        params,
        graph,
        critic,
        "critic",
        2,
        NodeType.VERIFY,
        "critic",
        shared,
        input_len=params.input_len + len(domain_agents) * params.output_len,
    )
    for d in domain_agents:
        _edge(graph, d, critic, EdgeType.GENERATED_MESSAGE, params.output_len)
    final = f"{params.job_id}_final"
    _node(params, graph, final, "synthesizer", 3, NodeType.SUMMARIZE, "final_synthesizer", shared)
    _edge(graph, critic, final, EdgeType.QUALITY_DEPENDENCY, params.output_len)
    graph.critical_path_lengths()
    return graph


def long_context_swe_stress(params: WorkloadParams) -> AgentGraph:
    p = WorkloadParams(
        **{
            **params.__dict__,
            "workload": "long_context_swe_stress",
            "input_len": max(params.input_len, 8192),
            "shared_prefix_ratio": params.shared_prefix_ratio,
            "num_workers": max(params.num_workers, params.num_agents),
        }
    )
    graph = swe_like(p)
    graph.workload = "long_context_swe_stress"
    return graph


def mesh_stress_moa(params: WorkloadParams) -> AgentGraph:
    p = WorkloadParams(
        **{
            **params.__dict__,
            "workload": "mesh_stress_moa",
            "num_layers": max(params.num_layers, 4),
            "width": max(params.width, params.num_agents, 16),
            "fan_in_policy": "all_to_all",
            "output_len": params.message_token_len or max(params.output_len, 512),
        }
    )
    graph = moa(p)
    graph.workload = "mesh_stress_moa"
    for edge in graph.edges:
        edge.message_token_len = p.message_token_len or p.output_len
    return graph


def sram_pressure_debate(params: WorkloadParams) -> AgentGraph:
    p = WorkloadParams(
        **{
            **params.__dict__,
            "workload": "sram_pressure_debate",
            "num_agents": max(params.num_agents, 8),
            "num_rounds": max(params.num_rounds, 2),
            "input_len": max(params.input_len, 8192),
            "shared_prefix_ratio": max(params.shared_prefix_ratio, 0.5),
        }
    )
    graph = debate(p)
    graph.workload = "sram_pressure_debate"
    return graph


def tool_pause_resume_loop(params: WorkloadParams) -> AgentGraph:
    p = WorkloadParams(
        **{
            **params.__dict__,
            "workload": "tool_pause_resume_loop",
            "num_workers": max(params.num_workers, params.num_agents, 4),
            "num_tools_per_worker": max(params.num_tools_per_worker, 4),
        }
    )
    graph = planner_worker_tool(p)
    graph.workload = "tool_pause_resume_loop"
    return graph


WORKLOAD_BUILDERS: dict[str, Callable[[WorkloadParams], AgentGraph]] = {
    "debate": debate,
    "moa": moa,
    "planner_worker_tool": planner_worker_tool,
    "swe_like": swe_like,
    "rag_like": rag_like,
    "long_context_swe_stress": long_context_swe_stress,
    "mesh_stress_moa": mesh_stress_moa,
    "sram_pressure_debate": sram_pressure_debate,
    "tool_pause_resume_loop": tool_pause_resume_loop,
}


def generate_workload(params: WorkloadParams) -> AgentGraph:
    try:
        return WORKLOAD_BUILDERS[params.workload](params)
    except KeyError as exc:
        raise ValueError(f"Unknown workload {params.workload}") from exc


def generate_workload_set(
    workloads: list[str],
    num_jobs: int,
    seed: int = 0,
    **overrides,
) -> list[AgentGraph]:
    graphs: list[AgentGraph] = []
    for workload in workloads:
        for j in range(num_jobs):
            params = WorkloadParams(
                workload=workload,
                job_id=f"{workload}_job_{j}",
                seed=seed + j,
                **overrides,
            )
            graphs.append(generate_workload(params))
    return graphs
