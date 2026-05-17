#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from waferagent.mesh import MeshConfig
from waferagent.plotting import (
    plot_energy,
    plot_jct_distribution,
    plot_kv_memory,
    plot_main_speedup,
    plot_mesh_hotspot,
)
from waferagent.simulator import load_trace_glob, write_simulation_outputs
from waferagent.utils import init_run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True)
    parser.add_argument("--wafer-config", default="configs/wafer/wse_like.yaml")
    parser.add_argument("--baselines", default="wafer_naive,kvflow_like,continuum_like,waferagent_full")
    parser.add_argument("--out", default="results/main_wafer_sim")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    out = init_run_dir(args.out, {"run_type": "simulation_sweep", "traces": args.traces, "wafer_config": args.wafer_config})
    traces = load_trace_glob(args.traces)
    mesh_cfg = MeshConfig.from_yaml(args.wafer_config)
    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]
    write_simulation_outputs(traces, mesh_cfg, baselines, out / "simulation", seed=args.seed)
    summary = out / "simulation" / "simulation_summary.csv"
    metrics = out / "simulation" / "simulation_metrics.csv"
    fig = out / "figures"
    plot_main_speedup(summary, fig / "fig_main_speedup")
    plot_jct_distribution(metrics, fig / "fig_main_jct_distribution")
    plot_kv_memory(summary, fig / "fig_kv_memory_saving")
    plot_mesh_hotspot(summary, fig / "fig_mesh_traffic_hotspot")
    plot_energy(summary, fig / "fig_energy_per_job")
    print(f"Simulation sweep complete: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
