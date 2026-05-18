#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from waferagent.mesh import MeshConfig
from waferagent.plotting import plot_sensitivity
from waferagent.simulator import simulate
from waferagent.trace_collector import collect_graph_traces
from waferagent.llm_runner import RunnerConfig
from waferagent.utils import init_run_dir
from waferagent.workloads import WorkloadParams, generate_workload


def _run_case(out, mesh_cfg, param_name: str, value, seed: int, neutral: bool) -> pd.DataFrame:
    workload = "debate"
    if param_name == "tool_latency_mean_ms":
        workload = "tool_pause_resume_loop"
    elif param_name == "link_bandwidth_GBps":
        workload = "mesh_stress_moa"
    elif param_name == "sram_per_tile_mb":
        workload = "sram_pressure_debate"
    params = {
        "workload": workload,
        "job_id": f"sensitivity_{param_name}_{value}",
        "seed": seed,
        "num_agents": 16 if workload in {"mesh_stress_moa", "sram_pressure_debate"} else 4,
        "num_rounds": 2,
        "shared_prefix_ratio": 0.5,
        "input_len": 8192 if workload == "sram_pressure_debate" else 2048,
        "output_len": 128,
        "mean_tool_latency_ms": 1000.0,
    }
    if param_name == "num_agents":
        params["num_agents"] = int(value)
    elif param_name == "num_rounds":
        params["num_rounds"] = int(value)
    elif param_name == "shared_prefix_ratio":
        params["shared_prefix_ratio"] = float(value)
    elif param_name == "tool_latency_mean_ms":
        params["mean_tool_latency_ms"] = float(value)
        params["num_tools_per_worker"] = 4
    elif param_name == "input_len":
        params["input_len"] = int(value)
    elif param_name == "output_len":
        params["output_len"] = int(value)
    graph = generate_workload(WorkloadParams(**params))
    traces = collect_graph_traces([graph], out.name, RunnerConfig(engine="synthetic"))
    metrics, _, _, _, _ = simulate(
        traces,
        mesh_cfg,
        ["wafer_naive", "waferagent_full"],
        seed=seed,
        neutral_multipliers=neutral,
    )
    metrics[param_name] = value
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="synthetic")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--gpus", default="")
    parser.add_argument("--wafer-config", default="configs/wafer/wse_like.yaml")
    parser.add_argument("--out", default="results/sensitivity")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--neutral-mechanism-multipliers", action="store_true")
    args = parser.parse_args()

    out = init_run_dir(args.out, {"run_type": "sensitivity", "engine": args.engine, "wafer_config": args.wafer_config})
    base_cfg = MeshConfig.from_yaml(args.wafer_config)
    sweeps = {
        "num_agents": [2, 4, 8, 16, 32],
        "shared_prefix_ratio": [0.0, 0.25, 0.5, 0.75, 0.9],
        "tool_latency_mean_ms": [0, 100, 1000, 5000, 20000],
        "input_len": [512, 2048, 8192, 32768],
        "link_bandwidth_GBps": [10, 25, 50, 100, 200],
        "sram_per_tile_mb": [2, 4, 8, 16, 32],
    }
    fig_names = {
        "num_agents": "fig_sensitivity_agents",
        "shared_prefix_ratio": "fig_sensitivity_prefix_ratio",
        "tool_latency_mean_ms": "fig_sensitivity_tool_latency",
        "input_len": "fig_sensitivity_context_len",
        "link_bandwidth_GBps": "fig_sensitivity_mesh_bandwidth",
        "sram_per_tile_mb": "fig_sensitivity_sram_capacity",
    }
    all_rows = []
    for param, values in sweeps.items():
        rows = []
        for value in values:
            cfg = base_cfg
            if param == "link_bandwidth_GBps":
                cfg = MeshConfig(**{**base_cfg.__dict__, "link_bandwidth_GBps": float(value)})
            elif param == "sram_per_tile_mb":
                cfg = MeshConfig(**{**base_cfg.__dict__, "tile_sram_mb": float(value)})
            metrics = _run_case(out, cfg, param, value, args.seed, args.neutral_mechanism_multipliers)
            rows.append(metrics)
        df = pd.concat(rows, ignore_index=True)
        df.to_csv(out / "simulation" / f"sensitivity_{param}.csv", index=False)
        plot_sensitivity(out / "simulation" / f"sensitivity_{param}.csv", param, out / "figures" / fig_names[param])
        all_rows.append(df)
    pd.concat(all_rows, ignore_index=True).to_csv(out / "simulation" / "sensitivity_all.csv", index=False)
    print(f"Sensitivity complete: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
