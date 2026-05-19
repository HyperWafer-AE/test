#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from waferagent.shared_attention_microbench import SharedAttentionCase, run_shared_attention_case
from waferagent.utils import enforce_clean_git_tree, file_sha256, finalize_run_dir, init_run_dir, write_json


def _ints(text: str) -> list[int]:
    return [int(x) for x in str(text).split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shared-prefix-tokens", default="512,2048,8192,16384")
    parser.add_argument("--private-tokens", default="64,256,1024")
    parser.add_argument("--num-agents", default="1,2,4,8,16")
    parser.add_argument("--heads", type=int, default=28)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--out", default="results/round8_shared_attention_microbench_h100")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--engine", default="synthetic")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--gpus", default="")
    parser.add_argument("--clean-required", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    enforce_clean_git_tree(args.clean_required, args.allow_dirty)
    out = init_run_dir(
        args.out,
        {
            "run_type": "shared_attention_microbench",
            "device": args.device,
            "shared_prefix_tokens": args.shared_prefix_tokens,
            "private_tokens": args.private_tokens,
            "num_agents": args.num_agents,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "reps": args.reps,
            "seed": args.seed,
        },
    )
    rows = []
    modes = ["independent_attention", "cohort_attention", "split_shared_private_merge"]
    for shared in _ints(args.shared_prefix_tokens):
        for private in _ints(args.private_tokens):
            for agents in _ints(args.num_agents):
                for mode in modes:
                    rows.extend(
                        run_shared_attention_case(
                            SharedAttentionCase(
                                mode=mode,
                                shared_prefix_tokens=shared,
                                private_tokens=private,
                                num_agents=agents,
                                heads=args.heads,
                                head_dim=args.head_dim,
                                device=args.device,
                                dtype=args.dtype,
                            ),
                            reps=args.reps,
                            seed=args.seed + shared + private + agents,
                        )
                    )
    sim = out / "simulation"
    raw = pd.DataFrame(rows)
    raw.to_csv(sim / "shared_attention_microbench_raw.csv", index=False)
    valid = raw.loc[~raw["oom"].astype(bool)].copy()
    summary = valid.groupby(
        ["mode", "shared_prefix_tokens", "private_tokens", "num_agents", "heads", "head_dim", "device", "dtype", "exact_merge"],
        as_index=False,
    ).agg(
        latency_ms=("latency_ms", "median"),
        memory_bytes_estimated=("memory_bytes_estimated", "median"),
        oom=("oom", "sum"),
    )
    if not summary.empty:
        base = summary.loc[summary["mode"] == "independent_attention"][
            ["shared_prefix_tokens", "private_tokens", "num_agents", "latency_ms", "memory_bytes_estimated"]
        ].rename(columns={"latency_ms": "independent_latency_ms", "memory_bytes_estimated": "independent_memory_bytes"})
        summary = summary.merge(base, on=["shared_prefix_tokens", "private_tokens", "num_agents"], how="left")
        summary["latency_speedup"] = summary["independent_latency_ms"] / summary["latency_ms"].clip(lower=1e-9)
        summary["read_byte_reduction_ratio"] = 1.0 - summary["memory_bytes_estimated"] / summary["independent_memory_bytes"].clip(lower=1)
        summary["reps"] = args.reps
    summary.to_csv(sim / "shared_attention_microbench_summary.csv", index=False)
    fit = {
        "source": "shared_attention_microbench",
        "fit_hash": file_sha256(sim / "shared_attention_microbench_summary.csv") if (sim / "shared_attention_microbench_summary.csv").exists() else "",
        "latency_model": "median_lookup",
        "coverage": {
            "shared_prefix_tokens": _ints(args.shared_prefix_tokens),
            "private_tokens": _ints(args.private_tokens),
            "num_agents": _ints(args.num_agents),
            "modes": modes,
        },
    }
    write_json(sim / "shared_attention_cost_fit.json", fit)
    write_json(out / "model_selection.json", {"engine_used": args.engine, "model": args.model, "fallback_count": 0})
    finalize_run_dir(out)
    print(f"Shared attention microbench complete: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
