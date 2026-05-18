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
from waferagent.utils import enforce_clean_git_tree, finalize_run_dir, init_run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True)
    parser.add_argument("--wafer-config", default="configs/wafer/wse_like.yaml")
    parser.add_argument("--baselines", default="wafer_naive,kvflow_like,continuum_like,waferagent_full")
    parser.add_argument("--out", default="results/main_wafer_sim")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--engine", default="synthetic")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--gpus", default="")
    parser.add_argument("--calibration", default="")
    parser.add_argument("--duration-source", default="synthetic", choices=["trace", "calibrated", "synthetic"])
    parser.add_argument("--allow-mesh-compute-overlap", action="store_true")
    parser.add_argument("--legacy-heuristic-multipliers", action="store_true")
    parser.add_argument("--neutral-mechanism-multipliers", action="store_true", help="Deprecated no-op; neutral is the default.")
    parser.add_argument("--clean-required", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    enforce_clean_git_tree(args.clean_required, args.allow_dirty)
    neutral = not bool(args.legacy_heuristic_multipliers)
    out = init_run_dir(args.out, {"run_type": "simulation_sweep", "traces": args.traces, "wafer_config": args.wafer_config, "neutral_mechanism_multipliers": neutral, "legacy_heuristic_multipliers": bool(args.legacy_heuristic_multipliers), "duration_source": args.duration_source, "engine": args.engine, "model": args.model, "gpus": args.gpus, "calibration": args.calibration, "clean_required": bool(args.clean_required)})
    traces = load_trace_glob(args.traces)
    mesh_cfg = MeshConfig.from_yaml(args.wafer_config)
    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]
    write_simulation_outputs(
        traces,
        mesh_cfg,
        baselines,
        out / "simulation",
        seed=args.seed,
        neutral_multipliers=neutral,
        calibration=args.calibration or None,
        duration_source=args.duration_source,
        allow_mesh_compute_overlap=bool(args.allow_mesh_compute_overlap),
    )
    summary = out / "simulation" / "simulation_summary.csv"
    metrics = out / "simulation" / "simulation_metrics.csv"
    fig = out / "figures"
    plot_main_speedup(summary, fig / "fig_main_speedup")
    plot_jct_distribution(metrics, fig / "fig_main_jct_distribution")
    plot_kv_memory(summary, fig / "fig_kv_memory_saving")
    plot_mesh_hotspot(summary, fig / "fig_mesh_traffic_hotspot")
    plot_energy(summary, fig / "fig_energy_per_job")
    finalize_run_dir(out)
    print(f"Simulation sweep complete: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
