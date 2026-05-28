"""Early-fingerprint predictability with baselines and bootstrap CIs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _cosine_rows(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(pred, axis=1) * np.linalg.norm(truth, axis=1)
    out = np.full(len(pred), np.nan, dtype=float)
    mask = denom > 0
    out[mask] = (pred[mask] * truth[mask]).sum(axis=1) / denom[mask]
    return out


def _safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
        return np.nan
    try:
        from scipy.stats import spearmanr

        corr = spearmanr(y_true, y_pred).correlation
        return float(corr) if corr is not None else np.nan
    except Exception:
        return np.nan


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return np.nan


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return np.nan
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    if denom <= 0:
        return np.nan
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def _evaluate_predictions(
    y_tool: np.ndarray,
    pred_tool: np.ndarray,
    y_steps: np.ndarray,
    pred_steps: np.ndarray,
    y_obs: np.ndarray,
    pred_obs: np.ndarray,
    y_long: np.ndarray,
    pred_long_score: np.ndarray,
) -> dict[str, float]:
    pred_tool = np.clip(np.asarray(pred_tool, dtype=float), 0, None)
    row_sums = pred_tool.sum(axis=1, keepdims=True)
    pred_tool = np.divide(pred_tool, row_sums, out=np.zeros_like(pred_tool), where=row_sums > 0)
    cosines = _cosine_rows(pred_tool, y_tool)
    top_hits = []
    for pred, truth in zip(pred_tool, y_tool):
        if truth.sum() <= 0:
            continue
        top_hits.append(int(int(np.argmax(truth)) in set(np.argsort(pred)[-3:])))
    pred_long = (pred_long_score >= 0.5).astype(int)
    return {
        "future_tool_cosine": float(np.nanmean(cosines)) if np.isfinite(cosines).any() else np.nan,
        "future_tool_top3_recall": float(np.mean(top_hits)) if top_hits else np.nan,
        "remaining_steps_mae": float(np.mean(np.abs(y_steps - pred_steps))),
        "remaining_steps_r2": _r2(y_steps, pred_steps),
        "future_obs_spearman": _safe_spearman(y_obs, pred_obs),
        "future_obs_mae": float(np.mean(np.abs(y_obs - pred_obs))),
        "long_auc": _safe_auc(y_long, pred_long_score),
        "long_accuracy": float(np.mean(y_long == pred_long)),
    }


def _bootstrap_ci(
    y_tool: np.ndarray,
    pred_tool: np.ndarray,
    y_steps: np.ndarray,
    pred_steps: np.ndarray,
    y_obs: np.ndarray,
    pred_obs: np.ndarray,
    y_long: np.ndarray,
    pred_long_score: np.ndarray,
    seed: int,
    n_bootstrap: int,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    n = len(y_steps)
    if n == 0 or n_bootstrap <= 0:
        return {}
    values: dict[str, list[float]] = {}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        metrics = _evaluate_predictions(
            y_tool[idx],
            pred_tool[idx],
            y_steps[idx],
            pred_steps[idx],
            y_obs[idx],
            pred_obs[idx],
            y_long[idx],
            pred_long_score[idx],
        )
        for key, value in metrics.items():
            if not np.isnan(value):
                values.setdefault(key, []).append(value)
    return {
        key: (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
        for key, vals in values.items()
        if vals
    }


def _split_indices(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    if n < 6:
        return idx, idx
    cut = max(2, int(0.7 * n))
    return idx[:cut], idx[cut:]


def _trace_examples(
    traces_df: pd.DataFrame,
    steps_df: pd.DataFrame,
    k: int,
    top_tools: list[str],
    long_threshold: float,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    if steps_df.empty:
        return examples
    trace_meta = traces_df.set_index("trace_id").to_dict("index") if not traces_df.empty else {}
    tmp = steps_df.copy()
    tmp["semantic_tool"] = tmp.get("semantic_tool", tmp.get("tool_name")).fillna("unknown").astype(str)
    tmp["tool_wrapper"] = tmp.get("tool_wrapper", tmp["semantic_tool"]).fillna("unknown").astype(str)
    tmp["phase"] = tmp["phase"].fillna("unknown").astype(str)
    for trace_id, group in tmp.sort_values(["trace_id", "step_id"]).groupby("trace_id"):
        total_steps = len(group)
        if total_steps == 0:
            continue
        first = group.head(k)
        future = group.iloc[k:]
        meta = trace_meta.get(trace_id, {})
        features: dict[str, float] = {
            "obs_bytes_first_k": float(first["observation_len_chars"].sum()),
            "msg_tokens_first_k": float(first["message_tokens_est"].sum()),
            "tool_args_len_first_k": float(first["tool_args_len"].sum()),
            "error_seen_first_k": float(first["error_flag"].astype(bool).any()),
            "test_seen_first_k": float((first["phase"] == "execute/test").any()),
            "first_step_obs": float(first["observation_len_chars"].iloc[0]) if len(first) else 0.0,
            "first_step_msg_tokens": float(first["message_tokens_est"].iloc[0]) if len(first) else 0.0,
        }
        for tool, count in first["semantic_tool"].value_counts().items():
            features[f"semantic_tool:{tool}"] = float(count)
        for wrapper, count in first["tool_wrapper"].value_counts().items():
            features[f"wrapper:{wrapper}"] = float(count)
        for phase, count in first["phase"].value_counts().items():
            features[f"phase:{phase}"] = float(count)

        future_counts = future["semantic_tool"].value_counts()
        future_vec = np.array([float(future_counts.get(tool, 0.0)) for tool in top_tools], dtype=float)
        if future_vec.sum() > 0:
            future_vec = future_vec / future_vec.sum()
        examples.append(
            {
                "trace_id": trace_id,
                "task_id": str(meta.get("task_id") or "unknown"),
                "agent_or_harness": str(meta.get("agent_or_harness") or "unknown"),
                "task_harness": f"{meta.get('task_id') or 'unknown'}::{meta.get('agent_or_harness') or 'unknown'}",
                "features": features,
                "future_tool_hist": future_vec,
                "remaining_steps": float(max(total_steps - k, 0)),
                "future_obs_bytes": float(future["observation_len_chars"].sum()),
                "long_flag": int(total_steps >= long_threshold),
            }
        )
    return examples


def _group_baseline_predictions(
    examples: list[dict[str, Any]],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    group_key: str | None,
    y_tool: np.ndarray,
    y_steps: np.ndarray,
    y_obs: np.ndarray,
    y_long: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    global_tool = y_tool[train_idx].mean(axis=0)
    global_steps = float(y_steps[train_idx].mean())
    global_obs = float(y_obs[train_idx].mean())
    global_long = float(y_long[train_idx].mean())
    if group_key is None:
        n = len(test_idx)
        return (
            np.repeat(global_tool.reshape(1, -1), n, axis=0),
            np.repeat(global_steps, n),
            np.repeat(global_obs, n),
            np.repeat(global_long, n),
        )
    groups: dict[str, dict[str, Any]] = {}
    for idx in train_idx:
        key = examples[idx][group_key]
        slot = groups.setdefault(key, {"idx": []})
        slot["idx"].append(idx)
    stats = {}
    for key, slot in groups.items():
        ids = np.array(slot["idx"], dtype=int)
        stats[key] = (
            y_tool[ids].mean(axis=0),
            float(y_steps[ids].mean()),
            float(y_obs[ids].mean()),
            float(y_long[ids].mean()),
        )
    pred_tool = []
    pred_steps = []
    pred_obs = []
    pred_long = []
    for idx in test_idx:
        key = examples[idx][group_key]
        tool, steps, obs, long = stats.get(key, (global_tool, global_steps, global_obs, global_long))
        pred_tool.append(tool)
        pred_steps.append(steps)
        pred_obs.append(obs)
        pred_long.append(long)
    return np.vstack(pred_tool), np.asarray(pred_steps), np.asarray(pred_obs), np.asarray(pred_long)


def _early_model_predictions(
    examples: list[dict[str, Any]],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    y_tool: np.ndarray,
    y_steps: np.ndarray,
    y_obs: np.ndarray,
    y_long: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        from sklearn.dummy import DummyClassifier, DummyRegressor
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.feature_extraction import DictVectorizer
    except Exception:
        return _group_baseline_predictions(examples, train_idx, test_idx, None, y_tool, y_steps, y_obs, y_long)

    vectorizer = DictVectorizer(sparse=True)
    x_all = vectorizer.fit_transform([ex["features"] for ex in examples])
    x_train = x_all[train_idx]
    x_test = x_all[test_idx]

    if len(train_idx) >= 8 and y_tool[train_idx].sum() > 0:
        tool_model = RandomForestRegressor(
            n_estimators=80, max_depth=10, min_samples_leaf=2, random_state=seed, n_jobs=-1
        )
        tool_model.fit(x_train, y_tool[train_idx])
        pred_tool = np.asarray(tool_model.predict(x_test), dtype=float)
    else:
        pred_tool = _group_baseline_predictions(examples, train_idx, test_idx, None, y_tool, y_steps, y_obs, y_long)[0]

    step_model = (
        RandomForestRegressor(n_estimators=80, max_depth=10, min_samples_leaf=2, random_state=seed, n_jobs=-1)
        if len(train_idx) >= 8 and np.unique(y_steps[train_idx]).size > 1
        else DummyRegressor(strategy="mean")
    )
    step_model.fit(x_train, y_steps[train_idx])
    pred_steps = np.asarray(step_model.predict(x_test), dtype=float)

    obs_model = (
        RandomForestRegressor(n_estimators=80, max_depth=10, min_samples_leaf=2, random_state=seed, n_jobs=-1)
        if len(train_idx) >= 8 and np.unique(y_obs[train_idx]).size > 1
        else DummyRegressor(strategy="mean")
    )
    obs_model.fit(x_train, y_obs[train_idx])
    pred_obs = np.asarray(obs_model.predict(x_test), dtype=float)

    if len(train_idx) >= 8 and np.unique(y_long[train_idx]).size == 2:
        clf = RandomForestClassifier(
            n_estimators=120, max_depth=10, min_samples_leaf=2, random_state=seed, n_jobs=-1
        )
        clf.fit(x_train, y_long[train_idx])
        pred_long = np.asarray(clf.predict_proba(x_test)[:, 1], dtype=float)
    else:
        clf = DummyClassifier(strategy="prior")
        clf.fit(x_train, y_long[train_idx])
        pred_long = np.repeat(float(y_long[train_idx].mean()), len(test_idx))
    return pred_tool, pred_steps, pred_obs, pred_long


def _evaluate_k(
    traces_df: pd.DataFrame,
    steps_df: pd.DataFrame,
    k: int,
    top_tools: list[str],
    long_threshold: float,
    seed: int,
    n_bootstrap: int,
) -> pd.DataFrame:
    examples = _trace_examples(traces_df, steps_df, k, top_tools, long_threshold)
    if not examples:
        return pd.DataFrame()
    train_idx, test_idx = _split_indices(len(examples), seed=seed)
    y_tool = np.vstack([ex["future_tool_hist"] for ex in examples])
    y_steps = np.array([ex["remaining_steps"] for ex in examples], dtype=float)
    y_obs = np.array([ex["future_obs_bytes"] for ex in examples], dtype=float)
    y_long = np.array([ex["long_flag"] for ex in examples], dtype=int)
    y_tool_test = y_tool[test_idx]
    y_steps_test = y_steps[test_idx]
    y_obs_test = y_obs[test_idx]
    y_long_test = y_long[test_idx]

    baselines = {
        "global_mean_baseline": (None, None),
        "task_baseline": ("task_id", None),
        "harness_baseline": ("agent_or_harness", None),
        "task_harness_baseline": ("task_harness", None),
        "early_fingerprint_model": (None, "early"),
    }
    rows = []
    global_metrics: dict[str, float] | None = None
    for baseline_name, (group_key, model_kind) in baselines.items():
        if model_kind == "early":
            pred_tool, pred_steps, pred_obs, pred_long_score = _early_model_predictions(
                examples, train_idx, test_idx, y_tool, y_steps, y_obs, y_long, seed=seed
            )
        else:
            pred_tool, pred_steps, pred_obs, pred_long_score = _group_baseline_predictions(
                examples, train_idx, test_idx, group_key, y_tool, y_steps, y_obs, y_long
            )
        metrics = _evaluate_predictions(
            y_tool_test, pred_tool, y_steps_test, pred_steps, y_obs_test, pred_obs, y_long_test, pred_long_score
        )
        ci = _bootstrap_ci(
            y_tool_test,
            pred_tool,
            y_steps_test,
            pred_steps,
            y_obs_test,
            pred_obs,
            y_long_test,
            pred_long_score,
            seed=seed + 17,
            n_bootstrap=n_bootstrap,
        )
        if baseline_name == "global_mean_baseline":
            global_metrics = metrics
        row: dict[str, Any] = {
            "K": k,
            "baseline": baseline_name,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
        }
        for key, value in metrics.items():
            row[key] = value
            if key in ci:
                row[f"{key}_ci_low"] = ci[key][0]
                row[f"{key}_ci_high"] = ci[key][1]
        if global_metrics is not None:
            row["future_tool_cosine_delta_global"] = metrics["future_tool_cosine"] - global_metrics["future_tool_cosine"]
            row["future_tool_top3_recall_delta_global"] = (
                metrics["future_tool_top3_recall"] - global_metrics["future_tool_top3_recall"]
            )
            row["remaining_steps_mae_delta_global"] = global_metrics["remaining_steps_mae"] - metrics["remaining_steps_mae"]
            row["future_obs_mae_delta_global"] = global_metrics["future_obs_mae"] - metrics["future_obs_mae"]
            row["long_auc_delta_global"] = metrics["long_auc"] - global_metrics["long_auc"] if not np.isnan(metrics["long_auc"]) and not np.isnan(global_metrics["long_auc"]) else np.nan
            row["long_accuracy_delta_global"] = metrics["long_accuracy"] - global_metrics["long_accuracy"]
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_baselines(metrics: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    if metrics.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No fingerprint data", ha="center", va="center", transform=ax.transAxes)
    else:
        sns.lineplot(
            data=metrics,
            x="K",
            y="future_tool_cosine",
            hue="baseline",
            marker="o",
            ax=axes[0],
        )
        axes[0].set_title("Future Semantic-Tool Cosine")
        axes[0].grid(True, alpha=0.3)
        sns.lineplot(
            data=metrics,
            x="K",
            y="long_auc",
            hue="baseline",
            marker="o",
            ax=axes[1],
            legend=False,
        )
        axes[1].set_title("Long-Trajectory AUC")
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, alpha=0.3)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_fingerprint_analysis(
    traces_df: pd.DataFrame,
    steps_df: pd.DataFrame,
    figures_dir: Path,
    tables_dir: Path,
    seed: int = 0,
    ks: tuple[int, ...] = (1, 2, 3, 5),
    n_bootstrap: int = 200,
) -> dict[str, Any]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    if steps_df.empty or traces_df.empty:
        metrics = pd.DataFrame()
    else:
        tool_col = "semantic_tool" if "semantic_tool" in steps_df.columns else "tool_name"
        tool_counts = steps_df[tool_col].fillna("unknown").astype(str).value_counts()
        top_tools = tool_counts.head(50).index.tolist()
        long_threshold = float(pd.to_numeric(traces_df["total_steps"], errors="coerce").quantile(0.75))
        parts = [
            _evaluate_k(
                traces_df,
                steps_df,
                k,
                top_tools,
                long_threshold,
                seed=seed + k,
                n_bootstrap=n_bootstrap,
            )
            for k in ks
        ]
        metrics = pd.concat([p for p in parts if not p.empty], ignore_index=True) if parts else pd.DataFrame()

    metrics.to_csv(tables_dir / "fingerprint_metrics_by_baseline.csv", index=False)
    # Backward-compatible table keeps early model rows.
    early = metrics[metrics["baseline"] == "early_fingerprint_model"].copy() if not metrics.empty else metrics
    early.to_csv(tables_dir / "fingerprint_metrics.csv", index=False)
    _plot_baselines(metrics, figures_dir / "early_fingerprint_vs_baselines.png")
    _plot_baselines(early, figures_dir / "early_fingerprint_predictability.png")
    return {"fingerprint_metrics_by_baseline": metrics, "fingerprint_metrics": early}
