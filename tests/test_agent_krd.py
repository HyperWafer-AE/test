import os
import runpy
import sys
from copy import deepcopy


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "utils"))
sys.path.append(os.path.join(ROOT, "src/partition"))
sys.path.append(os.path.join(ROOT, "src/scheduling"))
sys.path.append(os.path.join(ROOT, "src/backend/analytical"))
sys.path.append(os.path.join(ROOT, "src/scheduling/communication/topology"))


from read_cfg import cfg_to_dict
from WAMIS_HD import wamis_hdc
from add_communication import build_event
from event_driver import event_driver
from src.agent.krd import build_krds
from src.agent.kv_benchmark import build_kv_read_behaviors
from src.agent.kv_tensor import materialize_kv_tensors
from src.agent.metrics import collect_kv_metrics
from src.agent.placement import apply_placement, place_states
from src.agent.trace import make_coding_trace


CFG_PATH = os.path.join(
    ROOT,
    "models",
    "agentic_krd_static",
    "cfg",
    "config_ch2x2_bw256_co4x4_bw256_t128x64_failpattern0.cfg",
)


def _ensure_cfg():
    if not os.path.exists(CFG_PATH):
        runpy.run_path(os.path.join(ROOT, "models", "agentic_krd_static", "cfg.py"), run_name="__main__")


def _hardware():
    _ensure_cfg()
    return wamis_hdc(cfg_to_dict(CFG_PATH))


def _trace(block_elems=1024):
    return make_coding_trace(
        num_workflows=2,
        agents_per_workflow=3,
        repo_blocks=4,
        issue_blocks=2,
        private_blocks=2,
        block_elems=block_elems,
        seed=123,
    )


def _run_policy(policy):
    hardware_platform = _hardware()
    data_dict = {}
    beha_dict = {}
    event_dict = {}
    agents, states = _trace()
    states = materialize_kv_tensors(data_dict, states)
    krds = build_krds(agents, states, mode="workflow")
    plan = place_states(
        agents,
        states,
        krds,
        hardware_platform,
        policy=policy,
        dijkstra=True,
        region_size=8,
    )
    apply_placement(data_dict, states, plan)
    build_kv_read_behaviors(beha_dict, data_dict, agents, states, plan)
    hops, comm_dist, comm_loads, _tc_loads, _vu_loads = build_event(
        beha_dict,
        data_dict,
        hardware_platform,
        event_dict,
        dijkstra_routing=True,
    )
    metrics = collect_kv_metrics(event_dict, data_dict, states, comm_loads)
    total_cycles, pure_comp_cycles, pure_comm_cycles = event_driver(deepcopy(event_dict), deepcopy(hardware_platform))
    metrics.update(
        {
            "policy": policy,
            "total_cycles": total_cycles,
            "pure_comp_cycles": pure_comp_cycles,
            "pure_comm_cycles": pure_comm_cycles,
            "total_hop_bytes": hops,
            "communication_distances": comm_dist,
            "replica_bytes": plan.replica_bytes,
            "num_krds": len(plan.krds),
        }
    )
    return metrics, data_dict, states, plan


def test_tensor_materialization():
    data_dict = {}
    _agents, states = _trace()
    states = materialize_kv_tensors(data_dict, states)
    for state in states:
        assert state.data_tag[0] == 2
        tensor = data_dict[state.data_tag]
        assert len(tensor.generated_splitted_tag_dict) == state.num_blocks
        for split in tensor.generated_splitted_tag_dict:
            assert split in tensor.generated_split_location
            assert tensor.generated_split_location[split] == []


def test_placement_non_empty():
    hardware_platform = _hardware()
    data_dict = {}
    agents, states = _trace()
    states = materialize_kv_tensors(data_dict, states)
    krds = build_krds(agents, states, mode="workflow")
    assert len(krds) == 2
    plan = place_states(agents, states, krds, hardware_platform, policy="central", dijkstra=True)
    apply_placement(data_dict, states, plan)
    for state in states:
        tensor = data_dict[state.data_tag]
        for split in tensor.generated_splitted_tag_dict:
            assert tensor.generated_split_location[split]
        if state.owner_agent_id is not None:
            assert plan.state_locations[state.state_id] == [plan.agent_decode_nodes[state.owner_agent_id]]


def test_event_build_completes():
    metrics, _data_dict, _states, _plan = _run_policy("central")
    assert metrics["total_cycles"] >= 0
    assert metrics["kv_comm_bytes"] > 0


def test_replication_improves_or_matches_hop_bytes():
    central, _data_dict, _states, _plan = _run_policy("central")
    full, _data_dict, _states, _plan = _run_policy("full_replication")
    assert full["kv_hop_bytes"] <= central["kv_hop_bytes"]


def test_selective_replication_is_bounded():
    full, _data_dict, _states, _plan = _run_policy("full_replication")
    selective, _data_dict, _states, _plan = _run_policy("krd_selective")
    assert selective["replica_bytes"] <= full["replica_bytes"]


if __name__ == "__main__":
    test_tensor_materialization()
    test_placement_non_empty()
    test_event_build_completes()
    test_replication_improves_or_matches_hop_bytes()
    test_selective_replication_is_bounded()
    print("agent KRD smoke tests passed")

