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
                try:
                    k = torch.randn(args.heads, tokens, args.head_dim, device=device, dtype=torch.float16)
                    q = torch.randn(queries, args.heads, args.head_dim, device=device, dtype=torch.float16)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    _ = torch.einsum("qhd,htd->qht", q, k).sum()
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    ms = (time.perf_counter() - start) * 1000.0
                    oom = False
                except RuntimeError as exc:
                    ms = 0.0
                    oom = "out of memory" in str(exc).lower()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                rows.append({"shared_prefix_tokens": tokens, "num_queries": queries, "rep": rep, "latency_ms": ms, "oom": oom})
    raw = pd.DataFrame(rows)
    sim = out / "simulation"
    raw.to_csv(sim / "shared_kv_microbench_raw.csv", index=False)
    summary = raw.loc[~raw["oom"].astype(bool)].groupby(["shared_prefix_tokens", "num_queries"], as_index=False).agg(
        latency_ms=("latency_ms", "median")
    )
    summary.to_csv(sim / "shared_kv_microbench_summary.csv", index=False)
    line_from_csv(sim / "shared_kv_microbench_summary.csv", "shared_prefix_tokens", "latency_ms", out / "figures" / "fig9_h100_shared_kv_microbench", hue="num_queries")
    finalize_run_dir(out)
    print(f"Shared KV microbench complete: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
