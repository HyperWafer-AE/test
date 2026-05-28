"""Layered transition locality analysis.

The old pipeline measured only wrapper-level ``tool_name`` transitions, which
overstates locality when a harness repeatedly emits ``bash_command`` or
``execute_bash``.  This module reports wrapper, semantic, phase, and collapsed
semantic transitions separately.
"""

from __future__ import annotations

from collections import Counter
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


def _value_series(df: pd.DataFrame, col: str, fallback: str = "unknown") -> pd.Series:
    if col in df.columns:
        return df[col].fillna(fallback).astype(str).replace({"": fallback})
    return pd.Series([fallback] * len(df), index=df.index)


def _transition_rows(
    steps_df: pd.DataFrame,
    current_col: str,
    next_col: str | None = None,
    view: str = "semantic_tool",
    collapse_wrapper_self_loops: bool = False,
) -> pd.DataFrame:
    next_col = next_col or current_col
    rows: list[dict[str, Any]] = []
    if steps_df.empty:
        return pd.DataFrame(columns=["view", "trace_id", "current", "next"])

    tmp = steps_df.copy()
    tmp[current_col] = _value_series(tmp, current_col)
    tmp[next_col] = _value_series(tmp, next_col)
    tmp["tool_wrapper"] = _value_series(tmp, "tool_wrapper")
    tmp["semantic_tool"] = _value_series(tmp, "semantic_tool")
    tmp["phase"] = _value_series(tmp, "phase")
    for trace_id, group in tmp.sort_values(["trace_id", "step_id"]).groupby("trace_id"):
        if collapse_wrapper_self_loops:
            keep = []
            prev_wrapper = None
            for _, step in group.iterrows():
                wrapper = step["tool_wrapper"]
                if wrapper == prev_wrapper:
                    continue
                keep.append(step)
                prev_wrapper = wrapper
            records = keep
        else:
            records = [row for _, row in group.iterrows()]
        for cur, nxt in zip(records, records[1:]):
            rows.append(
                {
                    "view": view,
                    "trace_id": trace_id,
                    "current": cur[current_col],
                    "next": nxt[next_col],
                }
            )
    return pd.DataFrame(rows)


def _top_pairs(transitions: pd.DataFrame) -> pd.DataFrame:
    if transitions.empty:
        return pd.DataFrame(columns=["current", "next", "count", "probability"])
    pair_counts = transitions.groupby(["current", "next"]).size().reset_index(name="count")
    cur_counts = transitions.groupby("current").size().rename("cur_count")
    top_pairs = pair_counts.merge(cur_counts, on="current")
    top_pairs["probability"] = top_pairs["count"] / top_pairs["cur_count"]
    return top_pairs.drop(columns=["cur_count"]).sort_values(
        ["count", "probability"], ascending=[False, False]
    )


def _row_normalized_matrix(transitions: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    if not labels:
        return pd.DataFrame()
    if transitions.empty:
        return pd.DataFrame(0.0, index=labels, columns=labels)
    counts = pd.crosstab(transitions["current"], transitions["next"]).reindex(
        index=labels, columns=labels, fill_value=0
    )
    denom = counts.sum(axis=1).replace(0, np.nan)
    return counts.div(denom, axis=0).fillna(0.0)


def _top_labels(transitions: pd.DataFrame, limit: int = 20) -> list[str]:
    if transitions.empty:
        return []
    counts = Counter(transitions["current"]) + Counter(transitions["next"])
    return [label for label, _ in counts.most_common(limit)]


def _plot_heatmap(matrix: pd.DataFrame, path: Path, title: str) -> None:
    fig_w = max(7, 0.38 * max(len(matrix.columns), 1) + 3)
    fig_h = max(5, 0.32 * max(len(matrix.index), 1) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    if matrix.empty:
        ax.text(0.5, 0.5, "No transitions", ha="center", va="center", transform=ax.transAxes)
    else:
        sns.heatmap(
            matrix,
            cmap="viridis",
            annot=len(matrix) <= 12,
            fmt=".2f",
            ax=ax,
            cbar_kws={"label": "P(next | current)"},
        )
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


def transition_recall(transitions: pd.DataFrame, view: str, seed: int = 0) -> pd.DataFrame:
    rows = []
    if transitions.empty:
        for k in (1, 3, 5):
            rows.append(
                {
                    "view": view,
                    "k": k,
                    "conditional_recall": np.nan,
                    "global_recall": np.nan,
                    "delta_vs_global": np.nan,
                    "n_test": 0,
                }
            )
        return pd.DataFrame(rows)

    train, test = _train_test_trace_split(transitions, seed)
    conditional = {
        cur: group["next"].value_counts().index.tolist() for cur, group in train.groupby("current")
    }
    global_rank = train["next"].value_counts().index.tolist()
    for k in (1, 3, 5):
        cond_hits = 0
        global_hits = 0
        total = 0
        for _, row in test.iterrows():
            pred = conditional.get(row["current"], global_rank)[:k]
            global_pred = global_rank[:k]
            cond_hits += int(row["next"] in pred)
            global_hits += int(row["next"] in global_pred)
            total += 1
        cond = cond_hits / total if total else np.nan
        glob = global_hits / total if total else np.nan
        rows.append(
            {
                "view": view,
                "k": k,
                "conditional_recall": cond,
                "global_recall": glob,
                "delta_vs_global": cond - glob if not np.isnan(cond) and not np.isnan(glob) else np.nan,
                "n_test": total,
            }
        )
    return pd.DataFrame(rows)


def _plot_recall_by_view(recall_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    if recall_df.empty or recall_df["conditional_recall"].isna().all():
        ax.text(0.5, 0.5, "No transition recall data", ha="center", va="center", transform=ax.transAxes)
    else:
        plot_df = recall_df.copy()
        plot_df["topk"] = "top-" + plot_df["k"].astype(str)
        sns.barplot(
            data=plot_df,
            x="topk",
            y="conditional_recall",
            hue="view",
            ax=ax,
            palette="Set2",
        )
        for k, group in plot_df.groupby("topk"):
            if not group.empty:
                ax.axhline(group["global_recall"].max(), color="gray", linewidth=0.6, alpha=0.25)
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.3)
    ax.set_title("Next-State Top-k Recall by View")
    ax.set_ylabel("conditional recall")
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

    views = {
        "wrapper_tool": _transition_rows(steps_df, "tool_wrapper", view="wrapper_tool"),
        "semantic_tool": _transition_rows(steps_df, "semantic_tool", view="semantic_tool"),
        "phase": _transition_rows(steps_df, "phase", view="phase"),
        "collapsed_semantic_tool": _transition_rows(
            steps_df,
            "semantic_tool",
            view="collapsed_semantic_tool",
            collapse_wrapper_self_loops=True,
        ),
        "collapsed_phase": _transition_rows(
            steps_df,
            "phase",
            view="collapsed_phase",
            collapse_wrapper_self_loops=True,
        ),
    }

    top_pair_tables: dict[str, pd.DataFrame] = {}
    heatmap_files = {
        "wrapper_tool": "wrapper_tool_transition_heatmap.png",
        "semantic_tool": "semantic_tool_transition_heatmap.png",
        "collapsed_semantic_tool": "collapsed_semantic_tool_transition_heatmap.png",
        "phase": "phase_transition_heatmap.png",
    }
    table_files = {
        "wrapper_tool": "wrapper_tool_transition_top_pairs.csv",
        "semantic_tool": "semantic_tool_transition_top_pairs.csv",
        "collapsed_semantic_tool": "collapsed_semantic_tool_transition_top_pairs.csv",
    }

    for view, transitions in views.items():
        top_pairs = _top_pairs(transitions)
        top_pair_tables[view] = top_pairs
        if view in table_files:
            top_pairs.head(100).to_csv(tables_dir / table_files[view], index=False)
        if view in heatmap_files:
            labels = PHASE_ORDER if view == "phase" else _top_labels(transitions, 20)
            matrix = _row_normalized_matrix(transitions, labels)
            _plot_heatmap(matrix, figures_dir / heatmap_files[view], f"{view.replace('_', ' ').title()} Transition Heatmap")
            if view == "phase":
                matrix.to_csv(tables_dir / "phase_transition_matrix.csv")

    # Backward-compatible files now point to semantic-tool results.
    top_pair_tables["semantic_tool"].head(100).to_csv(tables_dir / "tool_transition_top_pairs.csv", index=False)
    sem_matrix = _row_normalized_matrix(views["semantic_tool"], _top_labels(views["semantic_tool"], 20))
    _plot_heatmap(sem_matrix, figures_dir / "tool_transition_heatmap.png", "Semantic Tool Transition Heatmap")

    recall_df = pd.concat(
        [
            transition_recall(views["wrapper_tool"], "wrapper_tool", seed=seed),
            transition_recall(views["semantic_tool"], "semantic_tool", seed=seed),
            transition_recall(views["collapsed_semantic_tool"], "collapsed_semantic_tool", seed=seed),
            transition_recall(views["phase"], "phase", seed=seed),
            transition_recall(views["collapsed_phase"], "collapsed_phase", seed=seed),
        ],
        ignore_index=True,
    )
    recall_df.to_csv(tables_dir / "transition_recall_by_view.csv", index=False)
    _plot_recall_by_view(recall_df, figures_dir / "topk_next_tool_recall_by_view.png")
    # Backward-compatible figure.
    _plot_recall_by_view(
        recall_df[recall_df["view"].isin(["semantic_tool", "phase"])],
        figures_dir / "topk_next_tool_recall.png",
    )

    return {
        "transitions_by_view": views,
        "top_pairs_by_view": top_pair_tables,
        "top_pairs": top_pair_tables["semantic_tool"],
        "transition_recall_by_view": recall_df,
        "next_tool_recall": recall_df[recall_df["view"] == "semantic_tool"].copy(),
        "top1_recall": float(
            recall_df[(recall_df["view"] == "semantic_tool") & (recall_df["k"] == 1)][
                "conditional_recall"
            ].iloc[0]
        ),
        "top3_recall": float(
            recall_df[(recall_df["view"] == "semantic_tool") & (recall_df["k"] == 3)][
                "conditional_recall"
            ].iloc[0]
        ),
    }
