from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


def summary_with_ci(metrics: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    group_cols = group_cols or ["baseline"]
    numeric = [
        c
        for c in metrics.columns
        if pd.api.types.is_numeric_dtype(metrics[c]) and c not in {"seed"}
    ]
    rows = []
    for keys, sub in metrics.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n"] = len(sub)
        for col in numeric:
            mean = float(sub[col].mean())
            std = float(sub[col].std(ddof=1)) if len(sub) > 1 else 0.0
            ci95 = 1.96 * std / math.sqrt(len(sub)) if len(sub) > 1 else 0.0
            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std
            row[f"{col}_ci95"] = ci95
        rows.append(row)
    return pd.DataFrame(rows)


def write_summary_with_ci(metrics: pd.DataFrame, out_path: str | Path, group_cols: list[str] | None = None) -> pd.DataFrame:
    df = summary_with_ci(metrics, group_cols)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df
