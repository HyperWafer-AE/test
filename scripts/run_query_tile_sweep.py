#!/usr/bin/env python3
"""Run query tile size sweep."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kvring.artifacts import plot_query_tile_sweep, result_row, run_mode, write_csv  # noqa: E402
from kvring.config import HardwareConfig, ModelConfig, WorkloadConfig  # noqa: E402


def int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--query-tile-size", type=int_list, required=True)
    p.add_argument("--shared-prefix-tokens", type=int, default=32768)
    p.add_argument("--agents", type=int, default=16)
    p.add_argument("--decode-tokens-per-agent", type=int, default=256)
    p.add_argument("--ring-shards", type=int_list, default=[8, 16])
    p.add_argument("--clean-required", action="store_true")
    args = p.parse_args()
    if args.clean_required and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    hardware = HardwareConfig()
    for r in args.query_tile_size:
        model = ModelConfig(query_tile_size=r)
        workload = WorkloadConfig(args.shared_prefix_tokens, args.agents, args.decode_tokens_per_agent)
        for shards in args.ring_shards:
            for mode in ["kvring_v2_query_tiled_parallel_ring_reduce", "kvring_v2_query_tiled_parallel_tree_reduce"]:
                result = run_mode(mode, model, workload, hardware, query_tile_size=r, num_shards=shards)
                row = result_row(
                    result,
                    shared_prefix_tokens=args.shared_prefix_tokens,
                    shared_kv_bytes=workload.shared_kv_bytes(model),
                    agents=args.agents,
                    decode_tokens=args.decode_tokens_per_agent,
                    query_tile_size=r,
                    num_shards=shards,
                )
                row["tile_formation_risk"] = max(0, r - args.agents)
                rows.append(row)
    write_csv(args.out / "query_tile_summary.csv", rows)
    write_csv(args.out / "query_tile_sweep.csv", rows)
    plot_query_tile_sweep(rows, args.out / "fig_query_tile_sweep")
    print(f"wrote query tile sweep to {args.out}")


if __name__ == "__main__":
    main()
