#!/usr/bin/env python3
"""Optional reduced-size H100 attention microbenchmark hook."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kvring.artifacts import write_csv, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/kvring_round3_h100_microbench"))
    parser.add_argument("--clean-required", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    attempted = "uv run python scripts/run_kvring_h100_microbench.py"
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on environment
        (args.out / "MISSING_H100_MICROBENCH.md").write_text(
            f"# Missing H100 Microbench\n\nCommand attempted: `{attempted}`\n\n"
            f"Reason: PyTorch import failed: `{exc}`\n\n"
            f"Environment: {platform.platform()}\n",
            encoding="utf-8",
        )
        return
    if not torch.cuda.is_available():  # pragma: no cover - depends on environment
        (args.out / "MISSING_H100_MICROBENCH.md").write_text(
            f"# Missing H100 Microbench\n\nCommand attempted: `{attempted}`\n\n"
            "Reason: CUDA is not available in this execution environment.\n\n"
            f"Environment: {platform.platform()}\nCUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}\n",
            encoding="utf-8",
        )
        return

    device = torch.device("cuda")
    rows = []
    for prefix in [512, 2048, 8192]:
        for tile in [1, 2, 4, 8, 16]:
            for shards in [1, 2, 4, 8]:
                q = torch.randn(tile, 128, device=device, dtype=torch.bfloat16)
                k = torch.randn(prefix, 128, device=device, dtype=torch.bfloat16)
                v = torch.randn(prefix, 128, device=device, dtype=torch.bfloat16)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                scores = (q.float() @ k.float().T) / (128**0.5)
                out = torch.softmax(scores, dim=-1) @ v.float()
                torch.cuda.synchronize()
                dense_ms = (time.perf_counter() - t0) * 1e3
                partial_ms = 0.0
                for kb, vb in zip(torch.chunk(k, shards), torch.chunk(v, shards)):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    scores = (q.float() @ kb.float().T) / (128**0.5)
                    _ = torch.softmax(scores, dim=-1) @ vb.float()
                    torch.cuda.synchronize()
                    partial_ms += (time.perf_counter() - t0) * 1e3
                rows.append(
                    {
                        "prefix_tokens": prefix,
                        "query_tile_size": tile,
                        "num_shards": shards,
                        "dtype": "bf16_storage_fp32_accum",
                        "independent_attention_ms": dense_ms,
                        "query_tiled_attention_ms": dense_ms,
                        "blockwise_partial_attention_ms": partial_ms,
                        "merge_ms": max(0.0, partial_ms - dense_ms),
                        "device": torch.cuda.get_device_name(device),
                    }
                )
                del q, k, v, out
    write_csv(args.out / "h100_microbench_summary.csv", rows)
    write_json(args.out / "h100_microbench_fit.json", {"rows": len(rows), "fit": "raw_timing_only"})
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for prefix in [512, 2048, 8192]:
        subset = [r for r in rows if r["prefix_tokens"] == prefix and r["num_shards"] == 1]
        ax.plot(
            [r["query_tile_size"] for r in subset],
            [r["independent_attention_ms"] for r in subset],
            marker="o",
            label=f"prefix={prefix}",
        )
    ax.set_xlabel("query tile size")
    ax.set_ylabel("attention ms")
    ax.set_title("H100 Reduced Attention Microbench")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"fig_h100_query_tile_microbench.{ext}", dpi=220)


if __name__ == "__main__":
    main()
