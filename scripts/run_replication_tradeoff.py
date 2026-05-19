#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from waferagent.kv_model import ModelKVConfig
from waferagent.mesh import MeshConfig
from waferagent.paper_figures import line_from_csv
from waferagent.shared_kv import extract_shared_kv_objects, plan_shared_kv_replication
from waferagent.utils import enforce_clean_git_tree, finalize_run_dir, init_run_dir
from waferagent.workloads import WorkloadParams, generate_workload


def _floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", default="moa_decode_cohort_stress")
    parser.add_argument("--sram-per-tile-mb", default="2,4,8,16,32")
    parser.add_argument("--link-bandwidth-GBps", default="10,25,50,100,200")
    parser.add_argument("--replication-policies", default="no_replication,replicate_all,benefit_cost,oracle")
    parser.add_argument("--out", default="results/round5_replication_tradeoff")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--engine", default="synthetic")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--gpus", default="")
    parser.add_argument("--clean-required", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    enforce_clean_git_tree(args.clean_required, args.allow_dirty)
    out = init_run_dir(args.out, {"run_type": "replication_tradeoff", "seed": args.seed})
    policies = [p.strip() for p in args.replication_policies.split(",") if p.strip()]
    graph = generate_workload(WorkloadParams(workload=args.workload, job_id="replication", seed=args.seed, num_agents=16, input_len=8192, output_len=256))
    objects, _ = extract_shared_kv_objects(graph, ModelKVConfig())
    for obj in objects:
        obj.candidate_regions = ["r0c0", "r0c1", "r1c0", "r1c1", "r2c2", "r3c3"]
    rows = []
    per_obj = []
    for sram in _floats(args.sram_per_tile_mb):
        for bw in _floats(args.link_bandwidth_GBps):
            cfg = MeshConfig(64, 64, sram, 0.5, 0.25, bw, 0.2, True, 0.5, 0.2)
            for policy in policies:
                planned, stats = plan_shared_kv_replication([o for o in objects], policy, cfg)
                mesh_bytes = max(0.0, sum(o.expected_decode_kv_read_bytes_without_cohort for o in planned) - stats["saved_mesh_traffic_bytes"])
                row = {
                    "workload": args.workload,
                    "sram_per_tile_mb": sram,
                    "link_bandwidth_GBps": bw,
                    "replication_policy": policy,
                    "mesh_traffic_bytes": mesh_bytes,
                    **stats,
                }
                rows.append(row)
                for obj in planned:
                    per_obj.append({**obj.to_dict(), "replication_policy": policy, "sram_per_tile_mb": sram, "link_bandwidth_GBps": bw})
    sim = out / "simulation"
    df = pd.DataFrame(rows)
    df.to_csv(sim / "replication_tradeoff_summary.csv", index=False)
    pd.DataFrame(per_obj).to_csv(sim / "replication_policy_per_shared_kv.csv", index=False)
    df[["replication_policy", "sram_per_tile_mb", "link_bandwidth_GBps", "mesh_traffic_bytes", "replica_bytes_total"]].to_csv(sim / "sram_mesh_pareto.csv", index=False)
    fig = out / "figures"
    line_from_csv(sim / "replication_tradeoff_summary.csv", "sram_per_tile_mb", "mesh_traffic_bytes", fig / "fig10_replication_pareto_sram_vs_mesh", hue="replication_policy")
    line_from_csv(sim / "replication_tradeoff_summary.csv", "sram_per_tile_mb", "replica_bytes_total", fig / "fig11_sram_capacity_knee", hue="replication_policy")
    line_from_csv(sim / "replication_tradeoff_summary.csv", "link_bandwidth_GBps", "mesh_traffic_bytes", fig / "fig12_mesh_bandwidth_sensitivity_replication", hue="replication_policy")
    finalize_run_dir(out)
    print(f"Replication tradeoff complete: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
