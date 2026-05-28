"""Stable-object and synthetic-bucket locality analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SYNTHETIC_TYPES = {"large_observation_bucket", "observation", "test_log"}


def _stable_mask(objects_df: pd.DataFrame) -> pd.Series:
    if objects_df.empty:
        return pd.Series(dtype=bool)
    if "stable_object" in objects_df.columns:
        values = objects_df["stable_object"]
        if values.dtype == object:
            return values.astype(str).str.lower().isin({"true", "1", "yes"})
        return values.astype(bool)
    return objects_df["object_type"].isin({"file", "browser_page", "test_case"})


def reuse_distances(objects_df: pd.DataFrame) -> pd.DataFrame:
    if objects_df.empty:
        return pd.DataFrame(
            columns=["trace_id", "object_id", "object_type", "reuse_distance", "reuse_class"]
        )

    tmp = objects_df.copy()
    tmp["step_id"] = pd.to_numeric(tmp["step_id"], errors="coerce").fillna(0).astype(int)
    tmp["reuse_class"] = np.where(_stable_mask(tmp), "stable", "synthetic")
    rows = []
    for (trace_id, object_id), group in tmp.sort_values(["trace_id", "object_id", "step_id"]).groupby(
        ["trace_id", "object_id"], dropna=False
    ):
        steps = group["step_id"].to_numpy()
        if len(steps) < 2:
            continue
        object_type = group["object_type"].iloc[0]
        reuse_class = group["reuse_class"].iloc[0]
        for prev, cur in zip(steps, steps[1:]):
            rows.append(
                {
                    "trace_id": trace_id,
                    "object_id": object_id,
                    "object_type": object_type,
                    "reuse_distance": int(cur - prev),
                    "reuse_class": reuse_class,
                }
            )
    return pd.DataFrame(rows)


def _plot_reuse_cdf(distances: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if distances.empty:
        ax.text(0.5, 0.5, "No repeated accesses", ha="center", va="center", transform=ax.transAxes)
    else:
        for object_type, group in distances.groupby("object_type"):
            vals = group["reuse_distance"].clip(lower=0).sort_values().to_numpy()
            y = np.arange(1, len(vals) + 1) / len(vals)
            ax.step(vals, y, where="post", label=object_type)
        ax.set_xscale("symlog", linthresh=1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("step distance between repeated accesses")
    ax.set_ylabel("CDF")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _summary(objects_df: pd.DataFrame, distances: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if objects_df.empty:
        return pd.DataFrame(
            columns=[
                "reuse_class",
                "object_accesses",
                "unique_objects",
                "reuse_events",
                "median_reuse_distance",
                "short_reuse_fraction_le3",
            ]
        )
    tmp = objects_df.copy()
    tmp["reuse_class"] = np.where(_stable_mask(tmp), "stable", "synthetic")
    for cls in ("stable", "synthetic"):
        obj_g = tmp[tmp["reuse_class"] == cls]
        dist_g = distances[distances["reuse_class"] == cls] if not distances.empty else pd.DataFrame()
        rows.append(
            {
                "reuse_class": cls,
                "object_accesses": int(len(obj_g)),
                "unique_objects": int(obj_g["object_id"].nunique()) if not obj_g.empty else 0,
                "reuse_events": int(len(dist_g)),
                "median_reuse_distance": float(dist_g["reuse_distance"].median()) if not dist_g.empty else np.nan,
                "short_reuse_fraction_le3": float((dist_g["reuse_distance"] <= 3).mean()) if not dist_g.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_object_locality_analysis(
    objects_df: pd.DataFrame,
    figures_dir: Path,
    tables_dir: Path | None = None,
) -> dict[str, Any]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    if tables_dir is not None:
        tables_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    distances = reuse_distances(objects_df)
    stable_distances = distances[distances["reuse_class"] == "stable"] if not distances.empty else distances
    synthetic_distances = distances[distances["reuse_class"] == "synthetic"] if not distances.empty else distances

    _plot_reuse_cdf(
        stable_distances,
        figures_dir / "stable_object_reuse_distance_cdf.png",
        "Stable Object Reuse Distance CDF",
    )
    _plot_reuse_cdf(
        synthetic_distances,
        figures_dir / "synthetic_bucket_reuse_distance_cdf.png",
        "Synthetic Bucket Reuse Distance CDF",
    )
    # Backward-compatible plot contains all reuse distances but should not be
    # used for H3 evidence.
    _plot_reuse_cdf(distances, figures_dir / "object_reuse_distance_cdf.png", "All Object Reuse Distance CDF")

    summary = _summary(objects_df, distances)
    if tables_dir is not None:
        summary.to_csv(tables_dir / "object_reuse_summary.csv", index=False)

    stable_row = summary[summary["reuse_class"] == "stable"]
    synthetic_row = summary[summary["reuse_class"] == "synthetic"]
    return {
        "reuse_distances": distances,
        "stable_reuse_distances": stable_distances,
        "synthetic_reuse_distances": synthetic_distances,
        "object_reuse_summary": summary,
        "stable_reuse_events": int(stable_row["reuse_events"].iloc[0]) if not stable_row.empty else 0,
        "median_stable_reuse_distance": float(stable_row["median_reuse_distance"].iloc[0])
        if not stable_row.empty
        else np.nan,
        "synthetic_reuse_events": int(synthetic_row["reuse_events"].iloc[0]) if not synthetic_row.empty else 0,
        "median_synthetic_reuse_distance": float(synthetic_row["median_reuse_distance"].iloc[0])
        if not synthetic_row.empty
        else np.nan,
        # Compatibility keys.
        "reuse_events": int(len(stable_distances)),
        "median_reuse_distance": float(stable_distances["reuse_distance"].median())
        if not stable_distances.empty
        else np.nan,
        "short_reuse_fraction": float((stable_distances["reuse_distance"] <= 3).mean())
        if not stable_distances.empty
        else np.nan,
    }
