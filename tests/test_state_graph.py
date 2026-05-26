from waferstateflow.ir import OperatorNode, StateAccessGraph, StateNode


def test_toy_state_access_graph_topology_and_exports(tmp_path):
    graph = StateAccessGraph("toy")
    graph.add_state(StateNode("S_task", kind="task", token_size=100))
    graph.add_operator(OperatorNode("Analyst", kind="llm", role="analyst"))
    graph.add_state(StateNode("S_summary", kind="output", token_size=40))
    graph.add_operator(OperatorNode("Reviewer", kind="llm", role="reviewer"))

    graph.connect_state_to_operator("S_task", "Analyst")
    graph.connect_operator_to_state("Analyst", "S_summary")
    graph.connect_state_to_operator("S_summary", "Reviewer")
    graph.update_operator_input_tokens()
    graph.compute_lifetimes()

    assert graph.operator_topological_order() == ["Analyst", "Reviewer"]
    assert graph.state_consumers("S_task") == ["Analyst"]
    assert graph.operator_inputs("Reviewer") == ["S_summary"]
    assert graph.producer("S_summary") == "Analyst"
    assert graph.state_fanout("S_task") == 1
    assert graph.operators["Analyst"].estimated_input_tokens == 100
    assert graph.states["S_summary"].lifetime_start == 0
    assert graph.states["S_summary"].lifetime_end == 1

    graph.export_csv(tmp_path)
    graph.export_json(tmp_path / "graph.json")

    assert (tmp_path / "state_nodes.csv").exists()
    assert (tmp_path / "operator_nodes.csv").exists()
    assert (tmp_path / "access_edges.csv").exists()
    assert (tmp_path / "graph.json").read_text(encoding="utf-8").startswith("{")
