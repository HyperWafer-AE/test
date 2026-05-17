#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from waferagent.baselines import ABLATIONS
from waferagent.mesh import MeshConfig
from waferagent.plotting import plot_ablation, plot_mesh_hotspot
from waferagent.simulator import load_trace_glob, write_simulation_outputs
from waferagent.utils import init_run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True)
    parser.add_argument("--wafer-config", default="configs/wafer/wse_like.yaml")
    parser.add_argument("--out", default="results/ablation")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    out = init_run_dir(args.out, {"run_type": "ablation", "traces": args.traces, "wafer_config": args.wafer_config})
    traces = load_trace_glob(args.traces)
    mesh_cfg = MeshConfig.from_yaml(args.wafer_config)
    variants = list(ABLATIONS.keys())
    write_simulation_outputs(traces, mesh_cfg, variants, out / "simulation", seed=args.seed)
    summary = out / "simulation" / "simulation_summary.csv"
    fig = out / "figures"
    plot_ablation(summary, fig / "fig_ablation_speedup")
    plot_mesh_hotspot(summary, fig / "fig_ablation_kv_mesh")
    print(f"Ablation complete: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
