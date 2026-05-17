from __future__ import annotations

import pytest

from waferagent.graph_ir import AgentEdge, AgentGraph, AgentNode, NodeType


def test_graph_ir_dag_and_critical_path():
    g = AgentGraph("g", "unit")
    g.add_node(AgentNode("a", "g", "a", 0, NodeType.LLM_CALL, "a", input_token_len=10))
    g.add_node(AgentNode("b", "g", "b", 1, NodeType.LLM_CALL, "b", input_token_len=20))
    g.add_node(AgentNode("c", "g", "c", 1, NodeType.LLM_CALL, "c", input_token_len=5))
    g.add_edge(AgentEdge("a", "b"))
    g.add_edge(AgentEdge("a", "c"))
    assert g.topological_order()[0] == "a"
    cp = g.critical_path_lengths()
    assert cp["a"] > cp["c"]
    assert g.fan_in_out_stats()["max_fan_out"] == 2


def test_graph_ir_cycle_detection():
    g = AgentGraph("g", "unit")
    g.add_node(AgentNode("a", "g", "a", 0, NodeType.LLM_CALL, "a"))
    g.add_node(AgentNode("b", "g", "b", 0, NodeType.LLM_CALL, "b"))
    g.add_edge(AgentEdge("a", "b"))
    g.add_edge(AgentEdge("b", "a"))
    with pytest.raises(ValueError):
        g.validate_acyclic()


def test_graph_json_roundtrip():
    g = AgentGraph("g", "unit")
    g.add_node(AgentNode("a", "g", "a", 0, NodeType.LLM_CALL, "a"))
    restored = AgentGraph.from_json(g.to_json())
    assert restored.nodes["a"].node_type == NodeType.LLM_CALL
