from waferstateflow.ir import OperatorNode, StateAccessGraph, StateNode
from waferstateflow.schedulers import StateCentricWaveScheduler


def _fanout_graph(token_size=1000, consumers=4, critical=False):
    graph = StateAccessGraph("fanout")
    graph.add_state(StateNode("S_hot", "doc", token_size=token_size))
    for i in range(consumers):
        op = OperatorNode(
            f"O_{i}",
            "llm",
            f"role_{i}",
            criticality=2.5 if critical and i == 0 else 1.0,
            ready_time=0,
        )
        graph.add_operator(op)
        graph.connect_state_to_operator("S_hot", op.op_id)
    graph.update_operator_input_tokens()
    graph.compute_lifetimes()
    return graph


def test_high_fanout_state_triggers_wave():
    graph = _fanout_graph()
    scheduler = StateCentricWaveScheduler(min_wave_hotness=1000)
    wave = scheduler.next_wave(graph, set(), current_time=0)
    assert wave.seed_state_id == "S_hot"
    assert wave.batch_size > 1


def test_critical_path_operator_is_not_over_waited():
    graph = _fanout_graph(token_size=10, consumers=3, critical=True)
    scheduler = StateCentricWaveScheduler(min_wave_hotness=1000, critical_wait_threshold=2)
    wave = scheduler.next_wave(graph, set(), current_time=5)
    assert wave.operator_ids == ("O_0",)
    assert wave.seed_state_id is None


def test_low_hotness_state_does_not_force_wave():
    graph = _fanout_graph(token_size=10, consumers=3)
    scheduler = StateCentricWaveScheduler(min_wave_hotness=1000)
    wave = scheduler.next_wave(graph, set(), current_time=0)
    assert wave.batch_size == 1
    assert wave.seed_state_id is None
