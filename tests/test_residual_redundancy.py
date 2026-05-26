from waferstateflow.ir import OperatorNode, StateAccessGraph, StateNode
from waferstateflow.residual_redundancy_analyzer import (
    ResidualAnalysisConfig,
    analyze_residual_redundancy,
)


def test_residual_analysis_decomposes_prefix_operator_kvflow_and_residual():
    graph = StateAccessGraph("residual_toy")
    graph.add_state(
        StateNode(
            "S_prefix",
            "task",
            token_size=100,
            prefix_compatible=True,
            kv_cacheable=True,
            prompt_position=0,
        )
    )
    graph.add_state(StateNode("S_residual", "output", token_size=50, metadata={"dynamic_hot_candidate": True}))
    graph.add_state(StateNode("S_det", "summary", token_size=40, deterministic=True))
    graph.add_operator(OperatorNode("O_det_producer", "aggregate", "builder", deterministic=True))
    graph.connect_operator_to_state("O_det_producer", "S_det")

    for i in range(3):
        graph.add_operator(OperatorNode(f"O_prefix_{i}", "llm", "reader"))
        graph.connect_state_to_operator("S_prefix", f"O_prefix_{i}")
        graph.add_operator(OperatorNode(f"O_residual_{i}", "llm", "dynamic_reader"))
        graph.connect_state_to_operator("S_residual", f"O_residual_{i}")
        graph.add_operator(OperatorNode(f"O_det_{i}", "llm", "det_reader"))
        graph.connect_state_to_operator("S_det", f"O_det_{i}")

    result = analyze_residual_redundancy(graph, ResidualAnalysisConfig(kvflow_capacity_bytes=0))
    rows = {row["state_id"]: row for row in result["state_rows"]}

    assert rows["S_prefix"]["prefix_covered"]
    assert rows["S_prefix"]["prefix_covered_tokens"] == 200
    assert rows["S_det"]["operator_cache_covered"]
    assert rows["S_det"]["operator_cache_covered_tokens"] == 80
    assert rows["S_residual"]["residual_candidate"]
    assert rows["S_residual"]["residual_token_weighted_fanout"] == 100
    assert result["summary"]["residual_token_weighted_fanout"] == 100
    assert result["summary"]["dynamic_hot_residual_fraction"] == 1.0


def test_kvflow_capacity_can_remove_remaining_residual():
    graph = StateAccessGraph("kvflow_capacity")
    graph.add_state(StateNode("S_nonprefix", "doc", token_size=100, kv_size_bytes=100))
    for i in range(3):
        graph.add_operator(OperatorNode(f"O_{i}", "llm", "reader"))
        graph.connect_state_to_operator("S_nonprefix", f"O_{i}")

    covered = analyze_residual_redundancy(graph, ResidualAnalysisConfig(kvflow_capacity_bytes=1000))
    row = covered["state_rows"][0]
    assert row["kvflow_cache_covered"]
    assert row["residual_token_weighted_fanout"] == 0
    assert covered["summary"]["residual_redundancy_ratio"] == 1.0

    uncovered = analyze_residual_redundancy(graph, ResidualAnalysisConfig(kvflow_capacity_bytes=0))
    assert uncovered["state_rows"][0]["residual_candidate"]
    assert uncovered["summary"]["residual_redundancy_ratio"] > 1.0
