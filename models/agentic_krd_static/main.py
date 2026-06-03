import argparse
import csv
import json
import os
import sys
from copy import deepcopy
from dataclasses import replace


file_path = os.path.dirname(os.path.realpath(__file__))
repo_root = os.path.abspath(os.path.join(file_path, "../.."))
sys.path.append(repo_root)
sys.path.append(os.path.join(repo_root, "utils"))
sys.path.append(os.path.join(repo_root, "src/partition"))
sys.path.append(os.path.join(repo_root, "src/scheduling"))
sys.path.append(os.path.join(repo_root, "src/backend/analytical"))
sys.path.append(os.path.join(repo_root, "src/scheduling/communication/topology"))


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


DEFAULT_CFG = os.path.join(
    file_path,
    "cfg",
    "config_ch2x2_bw256_co4x4_bw256_t128x64_failpattern0.cfg",
)


def attach_plan_to_agents(agents, plan):
    id_to_krd = {}
    for krd in plan.krds:
        for agent_id in krd.agent_ids:
            id_to_krd[agent_id] = krd.krd_id
    return [
        replace(
            agent,
            decode_node=plan.agent_decode_nodes[agent.agent_id],
            krd_id=id_to_krd[agent.agent_id],
        )
        for agent in agents
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="WaferAgent static KRD placement microbenchmark")
    parser.add_argument("--cfg", default=DEFAULT_CFG)
    parser.add_argument("--policy", choices=["central", "full_replication", "krd_selective"], required=True)
    parser.add_argument("--num-workflows", type=int, default=2)
    parser.add_argument("--agents-per-workflow", type=int, default=3)
    parser.add_argument("--repo-blocks", type=int, default=8)
    parser.add_argument("--issue-blocks", type=int, default=2)
    parser.add_argument("--private-blocks", type=int, default=4)
    parser.add_argument("--block-elems", type=int, default=65536)
    parser.add_argument("--region-size", type=int, default=8)
    parser.add_argument("--krd-mode", choices=["workflow", "affinity"], default="workflow")
    parser.add_argument("--dijkstra-routing", action="store_true")
    parser.add_argument("--gain-threshold", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", default=os.path.join(file_path, "results", "metrics.json"))
    parser.add_argument("--csv", default=None)
    return parser.parse_args()


def write_outputs(metrics, out_path, csv_path=None):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")

    csv_path = csv_path or os.path.join(os.path.dirname(os.path.abspath(out_path)), "summary.csv")
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    fieldnames = [
        "policy",
        "total_cycles",
        "pure_comp_cycles",
        "pure_comm_cycles",
        "total_hop_bytes",
        "communication_distances",
        "kv_hop_bytes",
        "kv_comm_bytes",
        "kv_comm_events",
        "total_comm_events",
        "max_link_load",
        "avg_link_load",
        "replica_bytes",
        "num_krds",
        "num_agents",
        "num_states",
        "krd_mode",
    ]
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({name: metrics.get(name, "") for name in fieldnames})


def run(args):
    hardware_cfg = cfg_to_dict(args.cfg)
    hardware_platform = wamis_hdc(hardware_cfg)

    data_dict = {}
    beha_dict = {}
    event_dict = {}

    agents, states = make_coding_trace(
        num_workflows=args.num_workflows,
        agents_per_workflow=args.agents_per_workflow,
        repo_blocks=args.repo_blocks,
        issue_blocks=args.issue_blocks,
        private_blocks=args.private_blocks,
        block_elems=args.block_elems,
        seed=args.seed,
    )
    states = materialize_kv_tensors(data_dict, states)
    krds = build_krds(agents, states, mode=args.krd_mode)
    plan = place_states(
        agents=agents,
        states=states,
        krds=krds,
        hardware_platform=hardware_platform,
        policy=args.policy,
        dijkstra=args.dijkstra_routing,
        gain_threshold=args.gain_threshold,
        region_size=args.region_size,
    )
    apply_placement(data_dict, states, plan)
    agents = attach_plan_to_agents(agents, plan)
    build_kv_read_behaviors(beha_dict, data_dict, agents, states, plan)

    hops, comm_dist, comm_loads, _tc_loads, _vu_loads = build_event(
        beha_dict=beha_dict,
        data_dict=data_dict,
        hardware_platform=hardware_platform,
        event_dict=event_dict,
        dijkstra_routing=args.dijkstra_routing,
        alpha=100,
        beta=1,
        gamma=100,
    )

    metrics = collect_kv_metrics(event_dict, data_dict, states, comm_loads)
    sim_hw = deepcopy(hardware_platform)
    sim_events = deepcopy(event_dict)
    total_cycles, pure_comp_cycles, pure_comm_cycles = event_driver(sim_events, sim_hw)

    metrics.update(
        {
            "policy": args.policy,
            "total_cycles": float(total_cycles),
            "pure_comp_cycles": float(pure_comp_cycles),
            "pure_comm_cycles": float(pure_comm_cycles),
            "total_hop_bytes": int(hops),
            "communication_distances": float(comm_dist),
            "replica_bytes": int(plan.replica_bytes),
            "num_krds": int(len(plan.krds)),
            "num_agents": int(len(agents)),
            "num_states": int(len(states)),
            "krd_mode": args.krd_mode,
            "dijkstra_routing": bool(args.dijkstra_routing),
            "gain_threshold": float(args.gain_threshold),
        }
    )
    write_outputs(metrics, args.out, args.csv)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


if __name__ == "__main__":
    run(parse_args())

