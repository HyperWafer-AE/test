"""Success vs failure trajectory comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _entropy(values: pd.Series) -> float:
    counts = values.dropna().value_counts()
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    return float(-(p * np.log2(p)).sum())


def per_trace_metrics(traces_df: pd.DataFrame, steps_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    steps_sorted = steps_df.sort_values(["trace_id", "step_id"]) if not steps_df.empty else steps_df
    grouped_steps = dict(tuple(steps_sorted.groupby("trace_id"))) if not steps_sorted.empty else {}

    for _, trace in traces_df.iterrows():
        trace_id = trace["trace_id"]
        group = grouped_steps.get(trace_id, pd.DataFrame(columns=steps_df.columns))
        tools = group["tool_name"].fillna("none/unknown") if not group.empty else pd.Series(dtype=str)
        phases = group["phase"].fillna("unknown") if not group.empty else pd.Series(dtype=str)
        if len(tools) > 1:
            repeated = float((tools.iloc[1:].to_numpy() == tools.iloc[:-1].to_numpy()).mean())
        else:
            repeated = 0.0
        success = trace.get("success")
        if pd.isna(success):
            status = "unknown"
        else:
            status = "success" if bool(success) else "failure"
        rows.append(
            {
                "trace_id": trace_id,
                "dataset": trace.get("dataset"),
                "status": status,
                "success": bool(success) if not pd.isna(success) else np.nan,
                "total_steps": float(trace.get("total_steps") or 0),
                "tool_entropy": _entropy(tools),
                "phase_entropy": _entropy(phases),
                "observation_bytes": float(group["observation_len_chars"].sum()) if not group.empty else 0.0,
                "repeated_action_ratio": repeated,
                "error_rate": float(group["error_flag"].astype(bool).mean()) if not group.empty else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _mannwhitney_p(a: pd.Series, b: pd.Series) -> float:
    if len(a.dropna()) < 2 or len(b.dropna()) < 2:
        return np.nan
    try:
        from scipy.stats import mannwhitneyu

        return float(mannwhitneyu(a.dropna(), b.dropna(), alternative="two-sided").pvalue)
    except Exception:
        return np.nan


def success_failure_table(trace_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "total_steps",
        "tool_entropy",
        "phase_entropy",
        "observation_bytes",
        "repeated_action_ratio",
        "error_rate",
    ]
    rows = []
    success = trace_metrics[trace_metrics["status"] == "success"]
    failure = trace_metrics[trace_metrics["status"] == "failure"]
    for metric in metric_cols:
        s = pd.to_numeric(success.get(metric, pd.Series(dtype=float)), errors="coerce")
        f = pd.to_numeric(failure.get(metric, pd.Series(dtype=float)), errors="coerce")
        rows.append(
            {
                "metric": metric,
                "success_mean": float(s.mean()) if not s.empty else np.nan,
                "failure_mean": float(f.mean()) if not f.empty else np.nan,
                "success_median": float(s.median()) if not s.empty else np.nan,
                "failure_median": float(f.median()) if not f.empty else np.nan,
                "delta_failure_minus_success": float(f.mean() - s.mean()) if not s.empty and not f.empty else np.nan,
                "mannwhitney_p": _mannwhitney_p(s, f),
            }
        )
    return pd.DataFrame(rows)


def _plot_bars(trace_metrics: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(14, 4), constrained_layout=True)
    metrics = [
        ("total_steps", "steps"),
        ("tool_entropy", "tool entropy"),
        ("observation_bytes", "obs bytes"),
        ("repeated_action_ratio", "repeat ratio"),
        ("error_rate", "error rate"),
    ]
    if trace_metrics.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        plot_df = trace_metrics[trace_metrics["status"].isin(["success", "failure"])].copy()
        for ax, (metric, title) in zip(axes, metrics):
            sns.barplot(
                data=plot_df,
                x="status",
                y=metric,
                hue="status",
                errorbar=("ci", 68),
                ax=ax,
                palette="Set2",
                legend=False,
            )
            ax.set_title(title)
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=25)
            ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Success vs Failure Trajectory Metrics", y=1.04)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_success_failure_analysis(
    traces_df: pd.DataFrame,
    steps_df: pd.DataFrame,
    figures_dir: Path,
    tables_dir: Path,
) -> dict[str, Any]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    trace_metrics = per_trace_metrics(traces_df, steps_df)
    table = success_failure_table(trace_metrics)
    table.to_csv(tables_dir / "success_failure_metrics.csv", index=False)
    _plot_bars(trace_metrics, figures_dir / "success_vs_failure_bars.png")
    return {"trace_metrics": trace_metrics, "success_failure_metrics": table}
