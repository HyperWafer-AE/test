"""Tool/phase transition locality analysis."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PHASE_ORDER = [
    "explore/read",
    "edit/write",
    "execute/test",
    "retrieve/browser",
    "verify/final",
    "unknown",
]


def build_transitions(steps_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if steps_df.empty:
        return pd.DataFrame(columns=["trace_id", "cur_tool", "next_tool", "cur_phase", "next_phase"])

    tmp = steps_df.copy()
    tmp["tool_name"] = tmp["tool_name"].fillna("none/unknown")
    tmp["phase"] = tmp["phase"].fillna("unknown")
    for trace_id, group in tmp.sort_values(["trace_id", "step_id"]).groupby("trace_id"):
        records = group[["tool_name", "phase"]].to_dict("records")
        for cur, nxt in zip(records, records[1:]):
            rows.append(
                {
                    "trace_id": trace_id,
                    "cur_tool": cur["tool_name"],
                    "next_tool": nxt["tool_name"],
                    "cur_phase": cur["phase"],
                    "next_phase": nxt["phase"],
                }
            )
    return pd.DataFrame(rows)


def _row_normalized_matrix(df: pd.DataFrame, row_col: str, col_col: str, labels: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(0.0, index=labels, columns=labels)
    counts = pd.crosstab(df[row_col], df[col_col]).reindex(index=labels, columns=labels, fill_value=0)
    denom = counts.sum(axis=1).replace(0, np.nan)
    return counts.div(denom, axis=0).fillna(0.0)


def _plot_heatmap(matrix: pd.DataFrame, path: Path, title: str, fmt: str = ".2f") -> None:
    fig_w = max(7, 0.38 * len(matrix.columns) + 3)
    fig_h = max(5, 0.32 * len(matrix.index) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    if matrix.empty:
        ax.text(0.5, 0.5, "No transitions", ha="center", va="center", transform=ax.transAxes)
    else:
        sns.heatmap(matrix, cmap="viridis", annot=len(matrix) <= 12, fmt=fmt, ax=ax, cbar_kws={"label": "P(next | current)"})
    ax.set_title(title)
    ax.set_xlabel("next")
    ax.set_ylabel("current")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _train_test_trace_split(transitions: pd.DataFrame, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    if transitions.empty:
        return transitions, transitions
    rng = np.random.default_rng(seed)
    trace_ids = np.array(sorted(transitions["trace_id"].unique()))
    if len(trace_ids) < 3:
        return transitions, transitions
    rng.shuffle(trace_ids)
    cut = max(1, int(0.7 * len(trace_ids)))
    train_ids = set(trace_ids[:cut])
    train = transitions[transitions["trace_id"].isin(train_ids)]
    test = transitions[~transitions["trace_id"].isin(train_ids)]
    if train.empty or test.empty:
        return transitions, transitions
    return train, test


def next_tool_recall(transitions: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    if transitions.empty:
        return pd.DataFrame(
            [
                {"k": 1, "conditional_recall": np.nan, "global_recall": np.nan},
                {"k": 3, "conditional_recall": np.nan, "global_recall": np.nan},
                {"k": 5, "conditional_recall": np.nan, "global_recall": np.nan},
            ]
        )

    train, test = _train_test_trace_split(transitions, seed)
    conditional: dict[str, list[str]] = {}
    for cur, group in train.groupby("cur_tool"):
        conditional[cur] = group["next_tool"].value_counts().index.tolist()
    global_rank = train["next_tool"].value_counts().index.tolist()

    rows = []
    for k in (1, 3, 5):
        cond_hits = 0
        global_hits = 0
        total = 0
        for _, row in test.iterrows():
            pred = conditional.get(row["cur_tool"], global_rank)[:k]
            global_pred = global_rank[:k]
            cond_hits += int(row["next_tool"] in pred)
            global_hits += int(row["next_tool"] in global_pred)
            total += 1
        rows.append(
            {
                "k": k,
                "conditional_recall": cond_hits / total if total else np.nan,
                "global_recall": global_hits / total if total else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _plot_recall(recall_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    if recall_df.empty or recall_df["conditional_recall"].isna().all():
        ax.text(0.5, 0.5, "No transition test data", ha="center", va="center", transform=ax.transAxes)
    else:
        width = 0.36
        x = np.arange(len(recall_df))
        ax.bar(x - width / 2, recall_df["conditional_recall"], width, label="P(next | current tool)")
        ax.bar(x + width / 2, recall_df["global_recall"], width, label="global top-k")
        ax.set_xticks(x, [f"top-{int(k)}" for k in recall_df["k"]])
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
    ax.set_title("Next-Tool Top-k Recall")
    ax.set_ylabel("recall")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_transition_analysis(
    steps_df: pd.DataFrame,
    figures_dir: Path,
    tables_dir: Path,
    seed: int = 0,
) -> dict[str, Any]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    transitions = build_transitions(steps_df)

    if transitions.empty:
        top_pairs = pd.DataFrame(columns=["cur_tool", "next_tool", "count", "probability"])
    else:
        pair_counts = transitions.groupby(["cur_tool", "next_tool"]).size().reset_index(name="count")
        cur_counts = transitions.groupby("cur_tool").size().rename("cur_count")
        top_pairs = pair_counts.merge(cur_counts, on="cur_tool")
        top_pairs["probability"] = top_pairs["count"] / top_pairs["cur_count"]
        top_pairs = top_pairs.drop(columns=["cur_count"]).sort_values(
            ["count", "probability"], ascending=[False, False]
        )
    top_pairs.head(100).to_csv(tables_dir / "tool_transition_top_pairs.csv", index=False)

    if transitions.empty:
        top_tools: list[str] = []
    else:
        tool_counts = Counter(transitions["cur_tool"]) + Counter(transitions["next_tool"])
        top_tools = [tool for tool, _ in tool_counts.most_common(20)]
    tool_matrix = _row_normalized_matrix(transitions, "cur_tool", "next_tool", top_tools)
    _plot_heatmap(tool_matrix, figures_dir / "tool_transition_heatmap.png", "Tool Transition Heatmap")

    phase_matrix = _row_normalized_matrix(transitions, "cur_phase", "next_phase", PHASE_ORDER)
    phase_matrix.to_csv(tables_dir / "phase_transition_matrix.csv")
    _plot_heatmap(phase_matrix, figures_dir / "phase_transition_heatmap.png", "Phase Transition Heatmap")

    recall_df = next_tool_recall(transitions, seed=seed)
    _plot_recall(recall_df, figures_dir / "topk_next_tool_recall.png")

    return {
        "transitions": transitions,
        "top_pairs": top_pairs,
        "phase_matrix": phase_matrix,
        "next_tool_recall": recall_df,
        "top1_recall": float(recall_df.loc[recall_df["k"] == 1, "conditional_recall"].iloc[0])
        if not recall_df.empty
        else np.nan,
        "top3_recall": float(recall_df.loc[recall_df["k"] == 3, "conditional_recall"].iloc[0])
        if not recall_df.empty
        else np.nan,
    }
