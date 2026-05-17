from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import networkx as nx


class NodeType(str, Enum):
    LLM_PREFILL = "llm_prefill"
    LLM_DECODE = "llm_decode"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    AGGREGATE = "aggregate"
    VERIFY = "verify"
    SUMMARIZE = "summarize"
    EARLY_EXIT = "early_exit"


class EdgeType(str, Enum):
    PROMPT_PREFIX = "prompt_prefix"
    GENERATED_MESSAGE = "generated_message"
    TOOL_RESULT = "tool_result"
    SHARED_CONTEXT = "shared_context"
    KV_DEPENDENCY = "kv_dependency"
    QUALITY_DEPENDENCY = "quality_dependency"
    CONTROL_DEPENDENCY = "control_dependency"


@dataclass
class AgentNode:
    node_id: str
    job_id: str
    agent_id: str
    round_id: int
    node_type: NodeType
    role: str
    model_id: str = "synthetic"
    input_token_len: int = 0
    expected_output_token_len: int = 0
    actual_output_token_len: int = 0
    shared_prefix_ids: list[str] = field(default_factory=list)
    private_prefix_ids: list[str] = field(default_factory=list)
    deps: list[str] = field(default_factory=list)
    fan_in: int = 0
    fan_out: int = 0
    criticality: float = 0.0
    slack: float = 0.0
    quality_weight: float = 1.0
    tool_latency_ms: float = 0.0
    kv_bytes_estimated: int = 0
    placement_region: str | None = None
    scheduler_tag: str | None = None
    shared_prefix_token_len: int = 0
    private_prefix_token_len: int = 0
    prompt_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["node_type"] = self.node_type.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentNode":
        data = dict(data)
        data["node_type"] = NodeType(data["node_type"])
        return cls(**data)

    @property
    def runtime_weight(self) -> float:
        tokens = self.input_token_len + max(self.expected_output_token_len, self.actual_output_token_len)
        return max(1.0, float(tokens) + self.tool_latency_ms / 10.0)


@dataclass
class AgentEdge:
    src: str
    dst: str
    edge_type: EdgeType = EdgeType.GENERATED_MESSAGE
    message_token_len: int = 0
    shared_kv_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["edge_type"] = self.edge_type.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentEdge":
        data = dict(data)
        data["edge_type"] = EdgeType(data["edge_type"])
        return cls(**data)


class AgentGraph:
    def __init__(self, graph_id: str, workload: str, seed: int = 0):
        self.graph_id = graph_id
        self.workload = workload
        self.seed = seed
        self.nodes: dict[str, AgentNode] = {}
        self.edges: list[AgentEdge] = []

    def add_node(self, node: AgentNode) -> None:
        if node.node_id in self.nodes:
            raise ValueError(f"Duplicate node_id {node.node_id}")
        self.nodes[node.node_id] = node

    def add_edge(self, edge: AgentEdge) -> None:
        if edge.src not in self.nodes or edge.dst not in self.nodes:
            raise KeyError(f"Edge references missing nodes: {edge.src}->{edge.dst}")
        self.edges.append(edge)
        if edge.src not in self.nodes[edge.dst].deps:
            self.nodes[edge.dst].deps.append(edge.src)
        self._refresh_degrees()

    def _refresh_degrees(self) -> None:
        fan_in = {node_id: 0 for node_id in self.nodes}
        fan_out = {node_id: 0 for node_id in self.nodes}
        for edge in self.edges:
            fan_out[edge.src] += 1
            fan_in[edge.dst] += 1
        for node_id, node in self.nodes.items():
            node.fan_in = fan_in[node_id]
            node.fan_out = fan_out[node_id]

    def to_networkx(self) -> nx.DiGraph:
        g = nx.DiGraph(graph_id=self.graph_id, workload=self.workload, seed=self.seed)
        for node_id, node in self.nodes.items():
            g.add_node(node_id, **node.to_dict())
        for edge in self.edges:
            g.add_edge(edge.src, edge.dst, **edge.to_dict())
        return g

    def validate_acyclic(self) -> bool:
        g = self.to_networkx()
        if not nx.is_directed_acyclic_graph(g):
            cycle = nx.find_cycle(g)
            raise ValueError(f"AgentGraph contains a cycle: {cycle}")
        return True

    def topological_order(self) -> list[str]:
        self.validate_acyclic()
        return list(nx.topological_sort(self.to_networkx()))

    def critical_path_lengths(self) -> dict[str, float]:
        order = self.topological_order()
        downstream: dict[str, float] = {}
        for node_id in reversed(order):
            succ = [edge.dst for edge in self.edges if edge.src == node_id]
            best_child = max((downstream[s] for s in succ), default=0.0)
            downstream[node_id] = self.nodes[node_id].runtime_weight + best_child
        max_len = max(downstream.values(), default=1.0)

        earliest: dict[str, float] = {}
        for node_id in order:
            deps = self.nodes[node_id].deps
            earliest[node_id] = max(
                (earliest[d] + self.nodes[d].runtime_weight for d in deps), default=0.0
            )
        for node_id in order:
            node = self.nodes[node_id]
            node.criticality = downstream[node_id] / max_len
            node.slack = max(0.0, max_len - earliest[node_id] - downstream[node_id])
        return downstream

    def fan_in_out_stats(self) -> dict[str, float]:
        self._refresh_degrees()
        if not self.nodes:
            return {"nodes": 0, "edges": 0}
        fan_in = [n.fan_in for n in self.nodes.values()]
        fan_out = [n.fan_out for n in self.nodes.values()]
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "max_fan_in": max(fan_in),
            "max_fan_out": max(fan_out),
            "avg_fan_in": sum(fan_in) / len(fan_in),
            "avg_fan_out": sum(fan_out) / len(fan_out),
        }

    def shared_prefix_stats(self) -> dict[str, float]:
        uses: dict[str, int] = {}
        tokens: dict[str, int] = {}
        total_input = 0
        shared_uses = 0
        for node in self.nodes.values():
            total_input += node.input_token_len
            for prefix_id in node.shared_prefix_ids:
                uses[prefix_id] = uses.get(prefix_id, 0) + 1
                tokens[prefix_id] = max(tokens.get(prefix_id, 0), node.shared_prefix_token_len)
                if uses[prefix_id] > 1:
                    shared_uses += 1
        unique_shared_tokens = sum(tokens.values())
        logical_shared_tokens = sum(
            node.shared_prefix_token_len for node in self.nodes.values() if node.shared_prefix_ids
        )
        return {
            "unique_shared_prefixes": len(uses),
            "shared_prefix_reuse_events": shared_uses,
            "unique_shared_tokens": unique_shared_tokens,
            "logical_shared_tokens": logical_shared_tokens,
            "shared_prefix_ratio": logical_shared_tokens / total_input if total_input else 0.0,
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "workload": self.workload,
            "seed": self.seed,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "AgentGraph":
        graph = cls(data["graph_id"], data["workload"], int(data.get("seed", 0)))
        for node_data in data.get("nodes", []):
            graph.add_node(AgentNode.from_dict(node_data))
        for edge_data in data.get("edges", []):
            graph.add_edge(AgentEdge.from_dict(edge_data))
        graph._refresh_degrees()
        return graph
