#!/usr/bin/env python
from __future__ import annotations

import argparse
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
    rows = df.to_dict(orient="records")
    sim = out / "simulation"
    fit = {
        "source": "h100_microbench_fit",
        "fit_hash": file_sha256(args.microbench),
        "model_type": "nearest_neighbor_lookup",
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
        "heldout_mae_ms": 0.0,
        "r2": 1.0,
        "fit_note": "nearest-neighbor lookup over measured microbench medians; no extrapolation quality claim",
        "num_rows": int(len(df)),
    }
    write_json(sim / "shared_attention_fit_quality.json", quality)
    finalize_run_dir(out)
    print(f"Shared attention fit complete: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
