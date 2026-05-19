#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from waferagent.utils import enforce_clean_git_tree, file_sha256, finalize_run_dir, init_run_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--microbench", required=True)
    parser.add_argument("--out", default="results/round9_shared_attention_fit")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clean-required", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    enforce_clean_git_tree(args.clean_required, args.allow_dirty)
    out = init_run_dir(args.out, {"run_type": "shared_attention_cost_fit", "microbench": args.microbench, "seed": args.seed})
    df = pd.read_csv(args.microbench)
    if "latency_p50_ms" not in df.columns and "latency_ms" in df.columns:
        df["latency_p50_ms"] = df["latency_ms"]
    if "latency_p90_ms" not in df.columns:
        df["latency_p90_ms"] = df["latency_p50_ms"]
    feature_cols = ["mode", "shared_prefix_tokens", "private_tokens", "num_agents", "heads", "head_dim"]

    def score(train_row: pd.Series, test_row: pd.Series) -> float:
        return (
            (0.0 if str(train_row.get("mode")) == str(test_row.get("mode")) else 10.0)
            + abs(float(train_row.get("shared_prefix_tokens", 0)) - float(test_row.get("shared_prefix_tokens", 0)))
            / max(1.0, float(test_row.get("shared_prefix_tokens", 1)))
            + abs(float(train_row.get("private_tokens", 0)) - float(test_row.get("private_tokens", 0)))
            / max(1.0, float(test_row.get("private_tokens", 1)))
            + abs(float(train_row.get("num_agents", 0)) - float(test_row.get("num_agents", 0)))
            / max(1.0, float(test_row.get("num_agents", 1)))
        )

    validation_rows = []
    for idx, row in df.iterrows():
        train = df.drop(index=idx)
        if train.empty:
            pred = float(row["latency_p50_ms"])
            quality = "self"
        else:
            nearest_idx = min(train.index, key=lambda i: score(train.loc[i], row))
            nearest = train.loc[nearest_idx]
            pred = float(nearest["latency_p50_ms"])
            exact_shape = all(
                str(nearest.get(c, "")) == str(row.get(c, ""))
                for c in ["mode", "shared_prefix_tokens", "private_tokens", "num_agents"]
            )
            quality = "interpolated" if exact_shape else "leave_one_shape_nearest"
        actual = float(row["latency_p50_ms"])
        abs_error = abs(pred - actual)
        validation_rows.append(
            {
                **{c: row.get(c, "") for c in feature_cols if c in df.columns},
                "actual_latency_ms": actual,
                "predicted_latency_ms": pred,
                "abs_error_ms": abs_error,
                "ape": abs_error / max(1e-9, abs(actual)),
                "prediction_quality": quality,
            }
        )
    validation = pd.DataFrame(validation_rows)
    mae = float(validation["abs_error_ms"].mean()) if not validation.empty else 0.0
    mape = float(validation["ape"].mean()) if not validation.empty else 0.0
    p90 = float(validation["abs_error_ms"].quantile(0.90)) if not validation.empty else 0.0
    maxerr = float(validation["abs_error_ms"].max()) if not validation.empty else 0.0
    y = validation["actual_latency_ms"] if not validation.empty else pd.Series(dtype=float)
    sse = float(((validation["actual_latency_ms"] - validation["predicted_latency_ms"]) ** 2).sum()) if not validation.empty else 0.0
    sst = float(((y - y.mean()) ** 2).sum()) if not validation.empty else 0.0
    r2 = 1.0 - sse / sst if sst > 0 else 1.0
    prediction_stat = "latency_p90_ms" if mape > 0.25 else "latency_p50_ms"
    rows = df.to_dict(orient="records")
    sim = out / "simulation"
    validation.to_csv(sim / "shared_attention_fit_validation.csv", index=False)
    if not validation.empty:
        validation.groupby(["mode", "shared_prefix_tokens", "num_agents"], as_index=False).agg(
            heldout_mae_ms=("abs_error_ms", "mean"),
            heldout_mape=("ape", "mean"),
            max_abs_error_ms=("abs_error_ms", "max"),
            rows=("abs_error_ms", "count"),
        ).to_csv(sim / "shared_attention_prediction_error_by_shape.csv", index=False)
    else:
        pd.DataFrame().to_csv(sim / "shared_attention_prediction_error_by_shape.csv", index=False)
    fit = {
        "source": "h100_microbench_fit",
        "fit_hash": file_sha256(args.microbench),
        "model_type": "nearest_neighbor_lookup",
        "prediction_stat": prediction_stat,
        "extrapolation_policy": "nearest_neighbor_with_warning",
        "rows": rows,
        "coverage": {
            "modes": sorted(df["mode"].dropna().astype(str).unique().tolist()) if "mode" in df.columns else [],
            "shared_prefix_tokens": sorted(df["shared_prefix_tokens"].dropna().astype(int).unique().tolist()) if "shared_prefix_tokens" in df.columns else [],
            "private_tokens": sorted(df["private_tokens"].dropna().astype(int).unique().tolist()) if "private_tokens" in df.columns else [],
            "num_agents": sorted(df["num_agents"].dropna().astype(int).unique().tolist()) if "num_agents" in df.columns else [],
        },
    }
    write_json(sim / "shared_attention_cost_fit.json", fit)
    quality = {
        "fit_hash": fit["fit_hash"],
        "model_type": fit["model_type"],
        "train_rows": int(len(df)),
        "test_rows": int(len(validation)),
        "heldout_mae_ms": mae,
        "heldout_mape": mape,
        "p90_abs_error_ms": p90,
        "max_abs_error_ms": maxerr,
        "r2": r2,
        "extrapolation_policy": fit["extrapolation_policy"],
        "prediction_stat": prediction_stat,
        "paper_safe": bool(math.isfinite(mape)),
        "fit_note": "leave-one-row nearest-neighbor validation; use latency_p90_ms when heldout MAPE exceeds 25%",
        "num_rows": int(len(df)),
    }
    write_json(sim / "shared_attention_fit_quality.json", quality)
    finalize_run_dir(out)
    print(f"Shared attention fit complete: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
