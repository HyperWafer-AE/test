#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from waferagent.llm_runner import RunnerConfig
from waferagent.mesh import MeshConfig
from waferagent.metrics import write_characterization_tables
from waferagent.plotting import plot_smoke_latency
from waferagent.simulator import write_simulation_outputs
from waferagent.trace_collector import collect_graph_traces
from waferagent.trace_schema import write_traces
from waferagent.utils import enforce_clean_git_tree, finalize_run_dir, init_run_dir
from waferagent.workloads import WorkloadParams, generate_workload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="synthetic")
    parser.add_argument("--wafer-config", default="configs/wafer/toy_smoke.yaml")
    parser.add_argument("--out", default="results/smoke")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", default="auto")
    parser.add_argument("--gpus", default="")
    parser.add_argument("--neutral-mechanism-multipliers", action="store_true")
    parser.add_argument("--clean-required", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    enforce_clean_git_tree(args.clean_required, args.allow_dirty)
    out = init_run_dir(args.out, {"run_type": "smoke", "engine": args.engine, "model": args.model, "gpus": args.gpus, "seed": args.seed, "neutral_mechanism_multipliers": bool(args.neutral_mechanism_multipliers), "clean_required": bool(args.clean_required)})
    graphs = [
        generate_workload(
            WorkloadParams(
                workload="debate",
                job_id=f"smoke_debate_{i}",
                seed=args.seed + i,
                num_agents=2,
                num_rounds=1,
                input_len=256,
                output_len=32,
                shared_prefix_ratio=0.5,
            )
        )
        for i in range(2)
    ]
    traces = collect_graph_traces(graphs, "smoke", RunnerConfig(engine=args.engine), out / "traces" / "smoke.jsonl")
    # Preserve the requested exact file name even if collector already wrote it.
    write_traces(out / "traces" / "smoke.jsonl", traces)
    write_characterization_tables(graphs, traces, out / "simulation")
    mesh_cfg = MeshConfig.from_yaml(args.wafer_config)
    metrics, _ = write_simulation_outputs(
        traces,
        mesh_cfg,
        ["wafer_naive", "kvflow_like", "continuum_like", "waferagent_full"],
        out / "simulation",
        seed=args.seed,
    )
    metrics.to_csv(out / "simulation" / "smoke_metrics.csv", index=False)
    plot_smoke_latency(out / "simulation" / "smoke_metrics.csv", out / "figures" / "smoke_latency_breakdown")
    finalize_run_dir(out)
    print(f"Smoke complete: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
