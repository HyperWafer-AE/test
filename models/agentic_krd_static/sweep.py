import argparse
import os
from types import SimpleNamespace


from cfg import main as write_cfg
from main import DEFAULT_CFG, run


file_path = os.path.dirname(os.path.realpath(__file__))
RESULTS_DIR = os.path.join(file_path, "results")


def _base_args(policy, label, csv_path, overrides):
    params = {
        "cfg": DEFAULT_CFG,
        "policy": policy,
        "num_workflows": 2,
        "agents_per_workflow": 3,
        "repo_blocks": 8,
        "issue_blocks": 2,
        "private_blocks": 4,
        "block_elems": 16384,
        "region_size": 8,
        "krd_mode": "workflow",
        "max_private_blocks_per_krd": None,
        "dijkstra_routing": True,
        "gain_threshold": 0.0,
        "pressure_penalty_scale": 1.0,
        "seed": 123,
        "out": os.path.join(RESULTS_DIR, f"sweep_{label}_{policy}.json"),
        "csv": csv_path,
    }
    params.update(overrides)
    return SimpleNamespace(**params)


def _quick_cases():
    return [
        ("quick", {}),
    ]


def _full_cases():
    cases = []
    for krd_mode in ["workflow", "affinity"]:
        for region_size in [4, 8]:
            for num_workflows, agents_per_workflow in [(2, 3), (4, 4)]:
                label = f"full_{krd_mode}_r{region_size}_w{num_workflows}_a{agents_per_workflow}"
                cases.append(
                    (
                        label,
                        {
                            "krd_mode": krd_mode,
                            "region_size": region_size,
                            "num_workflows": num_workflows,
                            "agents_per_workflow": agents_per_workflow,
                            "block_elems": 65536,
                            "max_private_blocks_per_krd": 16 if krd_mode == "affinity" else None,
                        },
                    )
                )
    return cases


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep WaferAgent Stage-1 static KRD placement policies")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true", help="Run a small three-policy acceptance sweep")
    group.add_argument("--full", action="store_true", help="Run a broader deterministic sweep")
    return parser.parse_args()


def main():
    args = parse_args()
    write_cfg()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "sweep_summary.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)

    cases = _full_cases() if args.full else _quick_cases()
    for label, overrides in cases:
        for policy in ["central", "full_replication", "krd_selective"]:
            metrics = run(_base_args(policy, label, csv_path, overrides))
            print(
                label,
                policy,
                "cycles=", metrics["total_cycles"],
                "kv_hop_bytes=", metrics["kv_hop_bytes"],
                "resident_bytes=", metrics["resident_bytes"],
                "extra_replica_bytes=", metrics["extra_replica_bytes"],
            )


if __name__ == "__main__":
    main()

