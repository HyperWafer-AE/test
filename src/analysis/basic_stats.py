"""Basic descriptive statistics and CDF figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _prepare_dirs(figures_dir: Path, tables_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)


def plot_cdf(values: pd.Series | list[float], path: Path, title: str, xlabel: str, log_x: bool = False) -> None:
    arr = pd.Series(values).dropna().astype(float)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if arr.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        arr = arr.sort_values()
        y = np.arange(1, len(arr) + 1) / len(arr)
        x = arr.to_numpy()
        if log_x:
            x = x + 1.0
        ax.plot(x, y, linewidth=2)
        ax.grid(True, alpha=0.3)
        if log_x:
            ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def dataset_summary_table(traces_df: pd.DataFrame, steps_df: pd.DataFrame) -> pd.DataFrame:
    if traces_df.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "num_traces",
                "num_steps",
                "success_rate",
                "median_steps",
                "mean_steps",
                "median_obs_chars_per_trace",
                "mean_obs_chars_per_trace",
            ]
        )

    obs_by_trace = (
        steps_df.groupby("trace_id")["observation_len_chars"].sum()
        if not steps_df.empty
        else pd.Series(dtype=float)
    )
    tmp = traces_df.copy()
    tmp["obs_chars"] = tmp["trace_id"].map(obs_by_trace).fillna(0)

    rows = []
    for dataset, group in tmp.groupby("dataset", dropna=False):
        trace_ids = set(group["trace_id"])
        steps_count = int(steps_df[steps_df["trace_id"].isin(trace_ids)].shape[0]) if not steps_df.empty else 0
        success = pd.to_numeric(group["success"], errors="coerce")
        rows.append(
            {
                "dataset": dataset,
                "num_traces": int(group.shape[0]),
                "num_steps": steps_count,
                "success_rate": float(success.mean()) if success.notna().any() else np.nan,
                "median_steps": float(pd.to_numeric(group["total_steps"], errors="coerce").median()),
                "mean_steps": float(pd.to_numeric(group["total_steps"], errors="coerce").mean()),
                "median_obs_chars_per_trace": float(group["obs_chars"].median()),
                "mean_obs_chars_per_trace": float(group["obs_chars"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("dataset")


def top_tools_table(steps_df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    if steps_df.empty:
        return pd.DataFrame(columns=["semantic_tool", "tool_wrapper", "phase", "count", "trace_count", "pct_steps"])
    tmp = steps_df.copy()
    tmp["semantic_tool"] = tmp.get("semantic_tool", tmp.get("tool_name")).fillna("unknown").astype(str)
    tmp["tool_wrapper"] = tmp.get("tool_wrapper", tmp["semantic_tool"]).fillna("unknown").astype(str)
    total = max(len(tmp), 1)
    grouped = (
        tmp.groupby(["semantic_tool", "tool_wrapper", "phase"], dropna=False)
        .agg(count=("step_id", "size"), trace_count=("trace_id", "nunique"))
        .reset_index()
    )
    grouped["pct_steps"] = grouped["count"] / total
    return grouped.sort_values(["count", "semantic_tool"], ascending=[False, True]).head(top_n)


def run_basic_stats(
    traces_df: pd.DataFrame,
    steps_df: pd.DataFrame,
    figures_dir: Path,
    tables_dir: Path,
) -> dict[str, Any]:
    _prepare_dirs(figures_dir, tables_dir)
    sns.set_theme(style="whitegrid")

    summary = dataset_summary_table(traces_df, steps_df)
    summary.to_csv(tables_dir / "dataset_summary.csv", index=False)

    top_tools = top_tools_table(steps_df)
    top_tools.to_csv(tables_dir / "top_tools.csv", index=False)

    plot_cdf(
        pd.to_numeric(traces_df.get("total_steps", pd.Series(dtype=float)), errors="coerce"),
        figures_dir / "trajectory_length_cdf.png",
        "Trajectory Length CDF",
        "steps per trace",
    )

    if steps_df.empty:
        tool_calls = pd.Series(dtype=float)
    else:
        tool_calls = steps_df[steps_df["tool_name"].notna()].groupby("trace_id").size()
        tool_calls = traces_df["trace_id"].map(tool_calls).fillna(0)
    plot_cdf(
        tool_calls,
        figures_dir / "tool_calls_per_trace_cdf.png",
        "Tool Calls Per Trace CDF",
        "tool calls per trace",
    )

    obs_sizes = (
        pd.to_numeric(steps_df.get("observation_len_chars", pd.Series(dtype=float)), errors="coerce")
        if not steps_df.empty
        else pd.Series(dtype=float)
    )
    plot_cdf(
        obs_sizes,
        figures_dir / "observation_size_cdf.png",
        "Observation Size CDF",
        "observation chars + 1",
        log_x=True,
    )

    obs_nonzero = obs_sizes[obs_sizes > 0]
    return {
        "dataset_summary": summary,
        "top_tools": top_tools,
        "num_traces": int(traces_df.shape[0]),
        "num_steps": int(steps_df.shape[0]),
        "median_steps": float(pd.to_numeric(traces_df.get("total_steps"), errors="coerce").median())
        if not traces_df.empty
        else np.nan,
        "obs_p95": float(obs_nonzero.quantile(0.95)) if not obs_nonzero.empty else 0.0,
        "obs_p50": float(obs_nonzero.quantile(0.50)) if not obs_nonzero.empty else 0.0,
    }
