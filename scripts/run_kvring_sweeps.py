#!/usr/bin/env python3
"""Run the required KVRing Round 2 sweeps."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kvring.artifacts import (  # noqa: E402
    generate_agent_count_sweep,
    generate_query_tile_sweep,
    generate_reduction_topology_sweep,
    generate_shard_count_sweep,
    generate_shared_prefix_sweep,
)


def int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def str_list(s: str) -> list[str]:
    return [x for x in s.split(",") if x]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modes",
        type=str_list,
        default=[
            "replicate_all",
            "pull_kv_independent",
            "central_kv_stationary",
            "kvring_v1",
            "kvring_v2_ring",
            "kvring_v2_tree",
        ],
    )
    parser.add_argument("--query-tile-sizes", type=int_list, default=[1, 2, 4, 8, 16])
    parser.add_argument("--shared-prefix-tokens", type=int_list, default=[2048, 8192, 32768, 65536])
    parser.add_argument("--agents", type=int_list, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--decode-tokens", type=int_list, default=[64, 256, 512])
    parser.add_argument("--out", type=Path, default=Path("results/kvring_round2_sweeps"))
    parser.add_argument("--clean-required", action="store_true")
    args = parser.parse_args()
    if args.clean_required and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    query_rows = generate_query_tile_sweep(
        args.out / "query_tile_sweep.csv",
        args.modes,
        args.query_tile_sizes,
        args.shared_prefix_tokens,
        args.agents,
        args.decode_tokens,
    )
    generate_shared_prefix_sweep(args.out / "shared_prefix_sweep.csv")
    generate_agent_count_sweep(args.out / "agent_count_sweep.csv")
    generate_shard_count_sweep(args.out / "shard_count_sweep.csv")
    generate_reduction_topology_sweep(args.out / "reduction_topology_sweep.csv")
    print(f"wrote required sweeps under {args.out}; query_tile rows={len(query_rows)}")


if __name__ == "__main__":
    main()
