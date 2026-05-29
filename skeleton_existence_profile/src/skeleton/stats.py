from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd


def entropy(values: Iterable[str]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    out = 0.0
    for c in counts.values():
        p = c / total
        out -= p * math.log2(p)
    return out


def bootstrap_ci_by_trace(
    rows: pd.DataFrame,
    value_fn,
    trace_col: str = "trace_id",
    n_boot: int = 50,
    seed: int = 0,
) -> tuple[float, float]:
    if rows.empty or trace_col not in rows.columns:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    tids = np.array(sorted(rows[trace_col].astype(str).unique()))
    if len(tids) == 0:
        return (np.nan, np.nan)
    vals = []
    groups = {tid: g for tid, g in rows.groupby(rows[trace_col].astype(str))}
    for _ in range(n_boot):
        sample_ids = rng.choice(tids, size=len(tids), replace=True)
        sample = pd.concat([groups[str(t)] for t in sample_ids], ignore_index=True)
        vals.append(value_fn(sample))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mann_whitney(success_values: pd.Series, failure_values: pd.Series) -> tuple[float, float]:
    try:
        from scipy.stats import mannwhitneyu

        if len(success_values) == 0 or len(failure_values) == 0:
            return np.nan, np.nan
        stat, p = mannwhitneyu(failure_values, success_values, alternative="two-sided")
        return float(stat), float(p)
    except Exception:
        return np.nan, np.nan


def auc_score(y_true: list[int], scores: list[float]) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        if len(set(y_true)) < 2:
            return np.nan
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return np.nan


def cosine(a: Counter, b: Counter) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def topk_recall(pred: list[str], truth: Iterable[str], k: int) -> float:
    truth_set = set(truth)
    if not truth_set:
        return np.nan
    return len(set(pred[:k]) & truth_set) / len(truth_set)


def train_test_split_trace(trace_ids: Iterable[str], seed: int = 0, train_frac: float = 0.7) -> tuple[set[str], set[str]]:
    ids = np.array(sorted(set(map(str, trace_ids))))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    cut = max(1, int(len(ids) * train_frac)) if len(ids) > 1 else len(ids)
    return set(ids[:cut]), set(ids[cut:])
