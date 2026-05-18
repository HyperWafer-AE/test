#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from waferagent.baselines import ablations
from waferagent.mesh import MeshConfig
from waferagent.plotting import plot_ablation, plot_mesh_hotspot
from waferagent.simulator import load_trace_glob, write_simulation_outputs
from waferagent.utils import enforce_clean_git_tree, finalize_run_dir, init_run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True)
    parser.add_argument("--wafer-config", default="configs/wafer/wse_like.yaml")
    parser.add_argument("--out", default="results/ablation")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--engine", default="synthetic")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--gpus", default="")
    parser.add_argument("--calibration", default="")
    parser.add_argument("--duration-source", default="synthetic", choices=["trace", "calibrated", "synthetic"])
    parser.add_argument("--legacy-heuristic-multipliers", action="store_true")
    parser.add_argument("--neutral-mechanism-multipliers", action="store_true", help="Deprecated no-op; neutral is the default.")
    parser.add_argument("--clean-required", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    enforce_clean_git_tree(args.clean_required, args.allow_dirty)
    neutral = not bool(args.legacy_heuristic_multipliers)
    out = init_run_dir(args.out, {"run_type": "ablation", "traces": args.traces, "wafer_config": args.wafer_config, "neutral_mechanism_multipliers": neutral, "legacy_heuristic_multipliers": bool(args.legacy_heuristic_multipliers), "duration_source": args.duration_source, "engine": args.engine, "model": args.model, "gpus": args.gpus, "calibration": args.calibration, "clean_required": bool(args.clean_required)})
    traces = load_trace_glob(args.traces)
    mesh_cfg = MeshConfig.from_yaml(args.wafer_config)
    variants = list(ablations(neutral=neutral).keys())
    write_simulation_outputs(
        traces,
        mesh_cfg,
        variants,
        out / "simulation",
        seed=args.seed,
        neutral_multipliers=neutral,
        calibration=args.calibration or None,
        duration_source=args.duration_source,
    )
    summary = out / "simulation" / "simulation_summary.csv"
    fig = out / "figures"
    plot_ablation(summary, fig / "fig_ablation_speedup")
    plot_mesh_hotspot(summary, fig / "fig_ablation_kv_mesh")
    finalize_run_dir(out)
    print(f"Ablation complete: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
