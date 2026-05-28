"""Approximate object reuse and locality analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def reuse_distances(objects_df: pd.DataFrame) -> pd.DataFrame:
    if objects_df.empty:
        return pd.DataFrame(columns=["trace_id", "object_id", "object_type", "reuse_distance"])

    tmp = objects_df.copy()
    tmp["step_id"] = pd.to_numeric(tmp["step_id"], errors="coerce").fillna(0).astype(int)
    rows = []
    for (trace_id, object_id), group in tmp.sort_values(["trace_id", "object_id", "step_id"]).groupby(
        ["trace_id", "object_id"]
    ):
        steps = group["step_id"].to_numpy()
        if len(steps) < 2:
            continue
        object_type = group["object_type"].iloc[0]
        for prev, cur in zip(steps, steps[1:]):
            rows.append(
                {
                    "trace_id": trace_id,
                    "object_id": object_id,
                    "object_type": object_type,
                    "reuse_distance": int(cur - prev),
                }
            )
    return pd.DataFrame(rows)


def _plot_reuse_cdf(distances: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if distances.empty:
        ax.text(0.5, 0.5, "No repeated object accesses", ha="center", va="center", transform=ax.transAxes)
    else:
        for object_type, group in distances.groupby("object_type"):
            vals = group["reuse_distance"].clip(lower=0).sort_values().to_numpy()
            y = np.arange(1, len(vals) + 1) / len(vals)
            ax.step(vals, y, where="post", label=object_type)
        ax.set_xscale("symlog", linthresh=1)
        ax.grid(True, alpha=0.3)
        ax.legend()
    ax.set_title("Object Reuse Distance CDF")
    ax.set_xlabel("step distance between repeated accesses")
    ax.set_ylabel("CDF")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_object_locality_analysis(
    objects_df: pd.DataFrame,
    figures_dir: Path,
) -> dict[str, Any]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    distances = reuse_distances(objects_df)
    _plot_reuse_cdf(distances, figures_dir / "object_reuse_distance_cdf.png")

    if distances.empty:
        return {
            "reuse_distances": distances,
            "reuse_events": 0,
            "median_reuse_distance": np.nan,
            "short_reuse_fraction": np.nan,
        }
    return {
        "reuse_distances": distances,
        "reuse_events": int(len(distances)),
        "median_reuse_distance": float(distances["reuse_distance"].median()),
        "short_reuse_fraction": float((distances["reuse_distance"] <= 3).mean()),
    }
