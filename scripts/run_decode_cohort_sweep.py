#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from waferagent.paper_figures import line_from_csv
from waferagent.utils import enforce_clean_git_tree, finalize_run_dir, init_run_dir


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", default="decode_heavy_shared_prefix")
    parser.add_argument("--num-agents", default="2,4,8,16,32")
    parser.add_argument("--shared-prefix-tokens", default="512,2048,8192,32768")
    parser.add_argument("--output-tokens", default="32,128,512")
    parser.add_argument("--cohort-size", default="1,2,4,8,16")
    parser.add_argument("--out", default="results/round5_decode_cohort_sweep")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--engine", default="synthetic")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--gpus", default="")
    parser.add_argument("--clean-required", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    enforce_clean_git_tree(args.clean_required, args.allow_dirty)
    out = init_run_dir(args.out, {"run_type": "decode_cohort_sweep", "seed": args.seed})
    rows = []
    kv_bytes_per_token = 2 * 28 * 4 * 128 * 2
    for agents in _ints(args.num_agents):
        for prefix in _ints(args.shared_prefix_tokens):
            for output in _ints(args.output_tokens):
                for cohort in _ints(args.cohort_size):
                    shared_kv = prefix * kv_bytes_per_token
                    without = agents * output * shared_kv
                    waves = (agents + cohort - 1) // max(1, cohort)
                    with_cohort = waves * output * shared_kv * 1.15
                    reduction = max(0.0, 1.0 - with_cohort / max(1.0, without))
                    rows.append(
                        {
                            "workload": args.workload,
                            "num_agents": agents,
                            "shared_prefix_tokens": prefix,
                            "output_tokens": output,
                            "cohort_size": cohort,
                            "shared_kv_read_without_cohort": without,
                            "shared_kv_read_with_cohort": with_cohort,
                            "shared_kv_read_reduction_ratio": reduction,
                            "decode_attention_latency_ms": with_cohort / (50e9 / 1000.0),
                        }
                    )
    df = pd.DataFrame(rows)
    sim = out / "simulation"
    df.to_csv(sim / "decode_cohort_sweep.csv", index=False)
    fig = out / "figures"
    line_from_csv(sim / "decode_cohort_sweep.csv", "cohort_size", "shared_kv_read_reduction_ratio", fig / "fig7_cohort_size_vs_shared_kv_read_reduction", hue="num_agents")
    line_from_csv(sim / "decode_cohort_sweep.csv", "shared_prefix_tokens", "decode_attention_latency_ms", fig / "fig8_shared_prefix_len_vs_decode_latency", hue="cohort_size")
    finalize_run_dir(out)
    print(f"Decode cohort sweep complete: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
