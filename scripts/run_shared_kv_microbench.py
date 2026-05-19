#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from waferagent.paper_figures import line_from_csv
from waferagent.utils import enforce_clean_git_tree, finalize_run_dir, init_run_dir


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--shared-prefix-tokens", default="512,2048,8192,32768")
    parser.add_argument("--num-queries", default="1,2,4,8,16")
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--out", default="results/round5_shared_kv_microbench_h100")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--engine", default="synthetic")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--gpus", default="")
    parser.add_argument("--clean-required", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    enforce_clean_git_tree(args.clean_required, args.allow_dirty)
    out = init_run_dir(args.out, {"run_type": "shared_kv_microbench", "device": args.device})
    import torch

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rows = []
    for tokens in _ints(args.shared_prefix_tokens):
        for queries in _ints(args.num_queries):
            for rep in range(args.reps):
                for mode in ["naive_per_agent_shared_kv_read", "cohort_shared_kv_read"]:
                    try:
                        k = torch.randn(args.heads, tokens, args.head_dim, device=device, dtype=torch.float16)
                        q = torch.randn(queries, args.heads, args.head_dim, device=device, dtype=torch.float16)
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        start = time.perf_counter()
                        if mode == "naive_per_agent_shared_kv_read":
                            total = 0
                            for i in range(queries):
                                total = total + torch.einsum("hd,htd->ht", q[i], k).sum()
                        else:
                            total = torch.einsum("qhd,htd->qht", q, k).sum()
                        total.item()
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        ms = (time.perf_counter() - start) * 1000.0
                        oom = False
                    except RuntimeError as exc:
                        ms = 0.0
                        oom = "out of memory" in str(exc).lower()
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                    elem_bytes = 2
                    shared_kv_bytes = args.heads * tokens * args.head_dim * elem_bytes * 2
                    read_multiplier = queries if mode == "naive_per_agent_shared_kv_read" else 1
                    rows.append(
                        {
                            "shared_prefix_tokens": tokens,
                            "num_queries": queries,
                            "rep": rep,
                            "mode": mode,
                            "latency_ms": ms,
                            "dram_bytes_estimated": shared_kv_bytes * read_multiplier,
                            "shared_kv_read_bytes": shared_kv_bytes * read_multiplier,
                            "oom": oom,
                        }
                    )
    raw = pd.DataFrame(rows)
    sim = out / "simulation"
    raw.to_csv(sim / "shared_kv_microbench_raw.csv", index=False)
    valid = raw.loc[~raw["oom"].astype(bool)]
    summary_long = valid.groupby(["shared_prefix_tokens", "num_queries", "mode"], as_index=False).agg(
        latency_ms=("latency_ms", "median"),
        dram_bytes_estimated=("dram_bytes_estimated", "median"),
        shared_kv_read_bytes=("shared_kv_read_bytes", "median"),
    )
    pivot = summary_long.pivot_table(
        index=["shared_prefix_tokens", "num_queries"],
        columns="mode",
        values=["latency_ms", "shared_kv_read_bytes"],
        aggfunc="first",
    )
    if not pivot.empty:
        pivot.columns = ["_".join(col).strip() for col in pivot.columns.to_flat_index()]
        pivot = pivot.reset_index()
        summary = pd.DataFrame(
            {
                "shared_prefix_tokens": pivot["shared_prefix_tokens"],
                "num_queries": pivot["num_queries"],
                "naive_latency_ms": pivot.get("latency_ms_naive_per_agent_shared_kv_read", 0),
                "cohort_latency_ms": pivot.get("latency_ms_cohort_shared_kv_read", 0),
                "naive_read_bytes": pivot.get("shared_kv_read_bytes_naive_per_agent_shared_kv_read", 0),
                "cohort_read_bytes": pivot.get("shared_kv_read_bytes_cohort_shared_kv_read", 0),
            }
        )
        summary["speedup"] = summary["naive_latency_ms"] / summary["cohort_latency_ms"].clip(lower=1e-9)
        summary["read_byte_reduction_ratio"] = 1.0 - summary["cohort_read_bytes"] / summary["naive_read_bytes"].clip(lower=1)
        summary["device"] = str(device)
        summary["dtype"] = "float16"
        summary["reps"] = args.reps
    else:
        summary = pd.DataFrame(
            columns=[
                "shared_prefix_tokens",
                "num_queries",
                "naive_latency_ms",
                "cohort_latency_ms",
                "speedup",
                "naive_read_bytes",
                "cohort_read_bytes",
                "read_byte_reduction_ratio",
                "device",
                "dtype",
                "reps",
            ]
        )
    summary_long.to_csv(sim / "shared_kv_microbench_long.csv", index=False)
    summary.to_csv(sim / "shared_kv_microbench_summary.csv", index=False)
    plot_long = summary_long.rename(columns={"latency_ms": "latency_ms"})
    plot_long.to_csv(sim / "shared_kv_microbench_plot.csv", index=False)
    line_from_csv(sim / "shared_kv_microbench_plot.csv", "shared_prefix_tokens", "latency_ms", out / "figures" / "fig9_h100_shared_kv_microbench", hue="mode")
    finalize_run_dir(out)
    print(f"Shared KV microbench complete: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
