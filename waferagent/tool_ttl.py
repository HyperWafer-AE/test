from __future__ import annotations

from waferagent.graph_ir import AgentGraph, NodeType


def tool_resume_probability(graph: AgentGraph, node_id: str) -> float:
    node = graph.nodes[node_id]
    for dep in node.deps:
        if graph.nodes[dep].node_type == NodeType.TOOL_CALL:
            return 0.9
    return 0.0


def is_tool_resume_node(graph: AgentGraph, node_id: str) -> bool:
    return tool_resume_probability(graph, node_id) > 0.0
