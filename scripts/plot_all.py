#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from waferagent.plotting import (
    plot_energy,
    plot_jct_distribution,
    plot_kv_memory,
    plot_main_speedup,
    plot_mesh_hotspot,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    root = Path(args.results)
    summary = root / "simulation" / "simulation_summary.csv"
    metrics = root / "simulation" / "simulation_metrics.csv"
    fig = root / "figures"
    plot_main_speedup(summary, fig / "fig_main_speedup")
    plot_jct_distribution(metrics, fig / "fig_main_jct_distribution")
    plot_kv_memory(summary, fig / "fig_kv_memory_saving")
    plot_mesh_hotspot(summary, fig / "fig_mesh_traffic_hotspot")
    plot_energy(summary, fig / "fig_energy_per_job")


if __name__ == "__main__":
    main()
