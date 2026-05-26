from waferstateflow.ir import StateNode
from waferstateflow.state_policy import PolicyConfig, decide_state_policy
from waferstateflow.wafer_topology import WaferTopology


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
