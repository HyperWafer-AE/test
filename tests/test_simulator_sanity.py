from waferstateflow.ir import StateNode
from waferstateflow.simulator import BackendProfile, SimulationConfig, simulate_workflow
from waferstateflow.simulator import _operator_time
from waferstateflow.ir import OperatorNode, StateAccessGraph
from waferstateflow.state_policy import PolicyConfig, decide_state_policy
from waferstateflow.wafer_topology import WaferTopology
from waferstateflow.workflow_generators import generate_workflow


def test_near_placement_has_lower_byte_hop_than_far_placement():
    topology = WaferTopology(mesh_x=4, mesh_y=4)
    state = StateNode("S", "doc", token_size=1000)
    near = topology.byte_hop(state.materialized_size_bytes, "R_0_0", "R_0_1")
    far = topology.byte_hop(state.materialized_size_bytes, "R_0_0", "R_3_3")
    assert near < far


def test_replicating_small_state_reduces_remote_access_cost():
    topology = WaferTopology(mesh_x=4, mesh_y=4)
    state = StateNode("S", "doc", token_size=100)
    consumers = ["R_0_0", "R_3_3"]
    replicated = topology.place_state(state, "replicate", consumers)
    pinned = topology.place_state(state, "pin", consumers)
    assert replicated.placement_byte_hop <= pinned.placement_byte_hop


def test_large_state_under_memory_pressure_does_not_blindly_replicate():
    state = StateNode("S", "doc", token_size=20000)
    state.consumers = ["O_0", "O_1", "O_2", "O_3"]
    state.static_hotness = 100000
    decision = decide_state_policy(
        state,
        memory_pressure=0.96,
        config=PolicyConfig(memory_pressure_critical=0.9),
    )
    assert decision.policy == "shard"


def test_replicate_pin_and_shard_memory_accounting():
    state = StateNode("S", "doc", token_size=1000, kv_size_bytes=0)
    size = state.materialized_size_bytes

    replicate_topology = WaferTopology(mesh_x=4, mesh_y=4, region_memory_capacity=size * 10)
    replicated = replicate_topology.place_state(state, "replicate", ["R_0_0", "R_3_3"], commit=True)
    assert replicate_topology.region_memory_used["R_0_0"] == size
    assert replicate_topology.region_memory_used["R_3_3"] == size
    assert replicated.memory_bytes == size * 2

    pin_topology = WaferTopology(mesh_x=4, mesh_y=4, region_memory_capacity=size * 10)
    pinned = pin_topology.place_state(state, "pin", ["R_0_0", "R_3_3"], commit=True)
    assert len(pinned.regions) == 1
    assert pin_topology.region_memory_used[pinned.regions[0]] == size

    shard_topology = WaferTopology(mesh_x=4, mesh_y=4, region_memory_capacity=2000)
    sharded = shard_topology.place_state(state, "shard", ["R_0_0", "R_3_3"], commit=True)
    expected_per_region = (size + len(sharded.regions) - 1) // len(sharded.regions)
    assert all(shard_topology.region_memory_used[region] == expected_per_region for region in sharded.regions)


def test_xy_routing_accumulates_directed_link_loads_and_hotspot():
    topology = WaferTopology(mesh_x=3, mesh_y=3)
    loads = {}
    byte_hop = topology.add_link_transfer(loads, 100, "R_0_0", "R_2_1")
    assert byte_hop == 300
    assert loads[("R_0_0", "R_1_0")] == 100
    assert loads[("R_1_0", "R_2_0")] == 100
    assert loads[("R_2_0", "R_2_1")] == 100
    summary = topology.link_load_summary(loads)
    assert summary["max_link_load"] == 100
    assert summary["p95_link_load"] == 100.0
    assert summary["hotspot_region"]


def test_operator_time_does_not_depend_on_scheduler_name():
    op = OperatorNode("O", "llm", "role", estimated_input_tokens=1000, estimated_output_tokens=50)
    cfg = SimulationConfig(backend_profile=BackendProfile(batch_prefill_multiplier=0.9))
    assert _operator_time(op, "WaferStateFlow", 4, cfg) == _operator_time(
        op, "helium_like_operator_schedule", 4, cfg
    )


def test_simulator_records_state_access_events_from_actual_accesses():
    graph = StateAccessGraph("events")
    graph.add_state(
        StateNode(
            "S_dynamic",
            "output",
            token_size=500,
            metadata={"dynamic_hot_candidate": True},
        )
    )
    for i in range(2):
        op = OperatorNode(f"O_{i}", "llm", f"role_{i}")
        graph.add_operator(op)
        graph.connect_state_to_operator("S_dynamic", op.op_id)
    graph.update_operator_input_tokens()
    run = simulate_workflow(
        graph,
        "WaferStateFlow",
        topology=WaferTopology(mesh_x=2, mesh_y=2),
        config=SimulationConfig(worker_count=2),
        state_policy="dynamic",
    )
    dynamic_events = [event for event in run.state_access_events if event["state_id"] == "S_dynamic"]
    assert len(dynamic_events) == 2
    assert dynamic_events[-1]["dynamic_hotness_after"] > dynamic_events[0]["dynamic_hotness_before"]


def test_helium_like_and_kvflow_like_are_not_identical_by_construction():
    graph = generate_workflow(
        "mapreduce",
        num_agents=4,
        shared_state_size=1000,
        unique_state_size=100,
        output_token_mean=50,
        seed=11,
    )
    cfg = SimulationConfig(worker_count=4, global_cache_bytes=1024)
    helium = simulate_workflow(graph, "helium_like_operator_schedule", config=cfg)
    kvflow = simulate_workflow(graph, "kvflow_like_future_eviction", config=cfg)
    assert (
        helium.result.state_materialization_bytes != kvflow.result.state_materialization_bytes
        or helium.result.wave_count != kvflow.result.wave_count
        or helium.wave_schedule != kvflow.wave_schedule
    )
