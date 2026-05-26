from waferstateflow.hotness import HotnessConfig, HotnessTracker
from waferstateflow.ir import OperatorNode, StateAccessGraph, StateNode


def _graph_for_hotness():
    graph = StateAccessGraph("hotness")
    graph.add_state(StateNode("S_shared", "doc", token_size=1000))
    for i in range(4):
        op = OperatorNode(f"O_{i}", "llm", f"role_{i}", criticality=1.5)
        graph.add_operator(op)
        graph.connect_state_to_operator("S_shared", op.op_id)
    graph.update_operator_input_tokens()
    return graph


def test_static_hot_state_reflects_fanout_and_size():
    graph = _graph_for_hotness()
    tracker = HotnessTracker(graph, HotnessConfig(promote_threshold=2000, demote_threshold=500))
    assert tracker.static_hotness(graph.states["S_shared"]) >= 3000


def test_dynamic_hotness_promotes_then_demotes_with_hysteresis():
    graph = _graph_for_hotness()
    tracker = HotnessTracker(graph, HotnessConfig(alpha=0.2, promote_threshold=900, demote_threshold=250))
    assert tracker.classify("S_shared") == "hot"

    graph.add_state(StateNode("S_dynamic", "output", token_size=800))
    tracker = HotnessTracker(graph, HotnessConfig(alpha=0.5, promote_threshold=500, demote_threshold=200))
    assert tracker.classify("S_dynamic") == "cold"
    tracker.observe_access("S_dynamic", access_cost=2.0)
    assert tracker.classify("S_dynamic") == "hot"

    tracker.cool("S_dynamic")
    assert tracker.classify("S_dynamic") == "hot"
    tracker.cool("S_dynamic")
    tracker.cool("S_dynamic")
    assert tracker.classify("S_dynamic") == "cold"
