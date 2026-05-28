"""Early-fingerprint predictability analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return np.nan
    return float(np.dot(a, b) / denom)


def _safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
        return np.nan
    try:
        from scipy.stats import spearmanr

        corr = spearmanr(y_true, y_pred).correlation
        return float(corr) if corr is not None else np.nan
    except Exception:
        return np.nan


def _trace_examples(steps_df: pd.DataFrame, k: int, top_tools: list[str], long_threshold: float) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    if steps_df.empty:
        return examples

    tmp = steps_df.copy()
    tmp["tool_name"] = tmp["tool_name"].fillna("none/unknown")
    tmp["phase"] = tmp["phase"].fillna("unknown")
    for trace_id, group in tmp.sort_values(["trace_id", "step_id"]).groupby("trace_id"):
        total_steps = len(group)
        if total_steps == 0:
            continue
        first = group.head(k)
        future = group.iloc[k:]
        features: dict[str, float] = {
            "obs_bytes_first_k": float(first["observation_len_chars"].sum()),
            "msg_tokens_first_k": float(first["message_tokens_est"].sum()),
            "tool_args_len_first_k": float(first["tool_args_len"].sum()),
            "error_seen_first_k": float(first["error_flag"].astype(bool).any()),
            "test_seen_first_k": float((first["phase"] == "execute/test").any()),
            "first_step_obs": float(first["observation_len_chars"].iloc[0]) if len(first) else 0.0,
            "first_step_msg_tokens": float(first["message_tokens_est"].iloc[0]) if len(first) else 0.0,
        }
        for tool, count in first["tool_name"].value_counts().items():
            features[f"tool:{tool}"] = float(count)
        for phase, count in first["phase"].value_counts().items():
            features[f"phase:{phase}"] = float(count)

        future_counts = future["tool_name"].value_counts()
        future_vec = np.array([float(future_counts.get(tool, 0.0)) for tool in top_tools], dtype=float)
        if future_vec.sum() > 0:
            future_vec = future_vec / future_vec.sum()

        examples.append(
            {
                "trace_id": trace_id,
                "features": features,
                "future_tool_hist": future_vec,
                "remaining_steps": float(max(total_steps - k, 0)),
                "future_obs_bytes": float(future["observation_len_chars"].sum()),
                "long_flag": int(total_steps >= long_threshold),
            }
        )
    return examples


def _split_indices(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    if n < 6:
        return idx, idx
    cut = max(2, int(0.7 * n))
    return idx[:cut], idx[cut:]


def _evaluate_k(steps_df: pd.DataFrame, k: int, top_tools: list[str], long_threshold: float, seed: int) -> dict[str, Any]:
    examples = _trace_examples(steps_df, k, top_tools, long_threshold)
    if not examples:
        return {
            "K": k,
            "n_train": 0,
            "n_test": 0,
            "future_tool_cosine": np.nan,
            "future_tool_top3_recall": np.nan,
            "remaining_steps_mae": np.nan,
            "remaining_steps_r2": np.nan,
            "future_obs_spearman": np.nan,
            "future_obs_mae": np.nan,
            "long_auc": np.nan,
            "long_accuracy": np.nan,
        }

    try:
        from sklearn.dummy import DummyClassifier, DummyRegressor
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score
    except Exception:
        DictVectorizer = None  # type: ignore
        DummyClassifier = DummyRegressor = None  # type: ignore
        RandomForestClassifier = RandomForestRegressor = None  # type: ignore
        LinearRegression = LogisticRegression = None  # type: ignore

        def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:  # type: ignore
            return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

        def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:  # type: ignore
            y_true_arr = np.asarray(y_true)
            denom = float(np.sum((y_true_arr - y_true_arr.mean()) ** 2))
            if denom <= 0:
                return np.nan
            return float(1.0 - np.sum((y_true_arr - np.asarray(y_pred)) ** 2) / denom)

        def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:  # type: ignore
            return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))

        def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:  # type: ignore
            raise ValueError("sklearn unavailable")

    train_idx, test_idx = _split_indices(len(examples), seed=seed)
    y_tool = np.vstack([ex["future_tool_hist"] for ex in examples])
    y_steps = np.array([ex["remaining_steps"] for ex in examples], dtype=float)
    y_obs = np.array([ex["future_obs_bytes"] for ex in examples], dtype=float)
    y_long = np.array([ex["long_flag"] for ex in examples], dtype=int)

    if DictVectorizer is None:
        pred_tool = np.repeat(y_tool[train_idx].mean(axis=0, keepdims=True), len(test_idx), axis=0)
        pred_steps = np.repeat(y_steps[train_idx].mean(), len(test_idx))
        pred_obs = np.repeat(y_obs[train_idx].mean(), len(test_idx))
        pred_long = np.repeat(round(float(y_long[train_idx].mean())), len(test_idx))
        pred_long_score = np.repeat(float(y_long[train_idx].mean()), len(test_idx))
    else:
        vectorizer = DictVectorizer(sparse=True)
        x_all = vectorizer.fit_transform([ex["features"] for ex in examples])
        x_train = x_all[train_idx]
        x_test = x_all[test_idx]

        if len(train_idx) >= 4 and y_tool[train_idx].sum() > 0:
            tool_model = RandomForestRegressor(
                n_estimators=80,
                max_depth=8,
                min_samples_leaf=1,
                random_state=seed,
                n_jobs=-1,
            )
            tool_model.fit(x_train, y_tool[train_idx])
            pred_tool = np.asarray(tool_model.predict(x_test), dtype=float)
        else:
            pred_tool = np.repeat(y_tool[train_idx].mean(axis=0, keepdims=True), len(test_idx), axis=0)
        pred_tool = np.clip(pred_tool, 0, None)

        if len(train_idx) >= 4 and np.unique(y_steps[train_idx]).size > 1:
            step_model = LinearRegression()
            step_model.fit(x_train, y_steps[train_idx])
            pred_steps = np.asarray(step_model.predict(x_test), dtype=float)
        else:
            step_model = DummyRegressor(strategy="mean")
            step_model.fit(x_train, y_steps[train_idx])
            pred_steps = np.asarray(step_model.predict(x_test), dtype=float)

        if len(train_idx) >= 4 and np.unique(y_obs[train_idx]).size > 1:
            obs_model = RandomForestRegressor(
                n_estimators=80,
                max_depth=8,
                random_state=seed,
                n_jobs=-1,
            )
            obs_model.fit(x_train, y_obs[train_idx])
            pred_obs = np.asarray(obs_model.predict(x_test), dtype=float)
        else:
            obs_model = DummyRegressor(strategy="mean")
            obs_model.fit(x_train, y_obs[train_idx])
            pred_obs = np.asarray(obs_model.predict(x_test), dtype=float)

        if len(train_idx) >= 6 and np.unique(y_long[train_idx]).size == 2:
            try:
                clf = LogisticRegression(max_iter=1000, class_weight="balanced")
                clf.fit(x_train, y_long[train_idx])
                pred_long = clf.predict(x_test)
                pred_long_score = clf.predict_proba(x_test)[:, 1]
            except Exception:
                clf = RandomForestClassifier(n_estimators=80, random_state=seed, n_jobs=-1)
                clf.fit(x_train, y_long[train_idx])
                pred_long = clf.predict(x_test)
                pred_long_score = clf.predict_proba(x_test)[:, 1]
        else:
            clf = DummyClassifier(strategy="most_frequent")
            clf.fit(x_train, y_long[train_idx])
            pred_long = clf.predict(x_test)
            pred_long_score = np.repeat(float(y_long[train_idx].mean()), len(test_idx))

    y_tool_test = y_tool[test_idx]
    cosines = [_cosine(p, t) for p, t in zip(pred_tool, y_tool_test)]
    valid_cosines = [c for c in cosines if not np.isnan(c)]
    top_hits = []
    for pred, truth in zip(pred_tool, y_tool_test):
        if truth.sum() <= 0:
            continue
        actual_top = int(np.argmax(truth))
        pred_top3 = set(np.argsort(pred)[-3:])
        top_hits.append(int(actual_top in pred_top3))

    y_steps_test = y_steps[test_idx]
    y_obs_test = y_obs[test_idx]
    y_long_test = y_long[test_idx]
    try:
        remaining_r2 = float(r2_score(y_steps_test, pred_steps)) if len(y_steps_test) > 1 else np.nan
    except Exception:
        remaining_r2 = np.nan
    try:
        long_auc = (
            float(roc_auc_score(y_long_test, pred_long_score))
            if len(np.unique(y_long_test)) == 2
            else np.nan
        )
    except Exception:
        long_auc = np.nan

    return {
        "K": k,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "future_tool_cosine": float(np.mean(valid_cosines)) if valid_cosines else np.nan,
        "future_tool_top3_recall": float(np.mean(top_hits)) if top_hits else np.nan,
        "remaining_steps_mae": float(mean_absolute_error(y_steps_test, pred_steps)),
        "remaining_steps_r2": remaining_r2,
        "future_obs_spearman": _safe_spearman(y_obs_test, pred_obs),
        "future_obs_mae": float(mean_absolute_error(y_obs_test, pred_obs)),
        "long_auc": long_auc,
        "long_accuracy": float(accuracy_score(y_long_test, pred_long)),
    }


def _plot_metrics(metrics: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if metrics.empty or metrics["future_tool_cosine"].isna().all():
        ax.text(0.5, 0.5, "No fingerprint data", ha="center", va="center", transform=ax.transAxes)
    else:
        plot_df = metrics[["K", "future_tool_cosine", "long_auc", "long_accuracy", "remaining_steps_r2"]].copy()
        plot_df["long_auc_or_acc"] = plot_df["long_auc"].fillna(plot_df["long_accuracy"])
        plot_df = plot_df.drop(columns=["long_auc", "long_accuracy"])
        plot_df = plot_df.melt(id_vars="K", var_name="metric", value_name="value")
        plot_df["value"] = plot_df["value"].clip(lower=-1, upper=1)
        sns.lineplot(data=plot_df, x="K", y="value", hue="metric", marker="o", ax=ax)
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
        ax.set_ylim(-1, 1)
        ax.grid(True, alpha=0.3)
    ax.set_title("Early Fingerprint Predictability")
    ax.set_ylabel("score")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_fingerprint_analysis(
    traces_df: pd.DataFrame,
    steps_df: pd.DataFrame,
    figures_dir: Path,
    tables_dir: Path,
    seed: int = 0,
    ks: tuple[int, ...] = (1, 2, 3, 5),
) -> dict[str, Any]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    if steps_df.empty or traces_df.empty:
        metrics = pd.DataFrame([_evaluate_k(steps_df, k, [], 0, seed) for k in ks])
    else:
        tool_counts = steps_df["tool_name"].fillna("none/unknown").value_counts()
        top_tools = tool_counts.head(40).index.tolist()
        long_threshold = float(pd.to_numeric(traces_df["total_steps"], errors="coerce").quantile(0.75))
        metrics = pd.DataFrame(
            [_evaluate_k(steps_df, k, top_tools, long_threshold, seed=seed + k) for k in ks]
        )

    metrics.to_csv(tables_dir / "fingerprint_metrics.csv", index=False)
    _plot_metrics(metrics, figures_dir / "early_fingerprint_predictability.png")
    return {"fingerprint_metrics": metrics}
