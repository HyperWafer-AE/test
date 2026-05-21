#!/usr/bin/env python3
"""Run placement/reduction topology sweep."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kvring.artifacts import plot_simple_bar, result_row, run_mode, write_csv  # noqa: E402
from kvring.config import HardwareConfig, ModelConfig, WorkloadConfig  # noqa: E402


def int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def str_list(s: str) -> list[str]:
    return [x for x in s.split(",") if x]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--placement", type=str_list, required=True)
    p.add_argument("--reduction-topology", type=str_list, required=True)
    p.add_argument("--shared-prefix-tokens", type=int, default=32768)
    p.add_argument("--agents", type=int, default=8)
    p.add_argument("--decode-tokens-per-agent", type=int, default=256)
    p.add_argument("--ring-shards", type=int_list, default=[8, 16])
    p.add_argument("--clean-required", action="store_true")
    args = p.parse_args()
    if args.clean_required and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    model = ModelConfig(query_tile_size=8)
    workload = WorkloadConfig(args.shared_prefix_tokens, args.agents, args.decode_tokens_per_agent)
    hardware = HardwareConfig()
    rows = []
    mode_by_topo = {
        "selected_ring": "kvring_v2_query_tiled_parallel_ring_reduce",
        "selected_ring_reduce": "kvring_v2_query_tiled_parallel_ring_reduce",
        "binary_tree": "kvring_v2_query_tiled_parallel_tree_reduce",
        "binary_tree_reduce": "kvring_v2_query_tiled_parallel_tree_reduce",
    }
    for placement in args.placement:
        for topo in args.reduction_topology:
            for shards in args.ring_shards:
                mode = mode_by_topo.get(topo, "kvring_v2_query_tiled_parallel_ring_reduce")
                result = run_mode(mode, model, workload, hardware, query_tile_size=8, num_shards=shards, placement=placement)
                rows.append(
                    result_row(
                        result,
                        shared_prefix_tokens=args.shared_prefix_tokens,
                        shared_kv_bytes=workload.shared_kv_bytes(model),
                        agents=args.agents,
                        decode_tokens=args.decode_tokens_per_agent,
                        num_shards=shards,
                        placement=placement,
                    )
                )
    write_csv(args.out / "topology_summary.csv", rows)
    plot_simple_bar(rows, "max_directed_link_load_bytes", "placement", args.out / "fig_topology_hotspot", "Topology Hotspot")
    plot_simple_bar(rows, "attention_stage_proxy_latency_s", "placement", args.out / "fig_topology_latency", "Topology Attention Proxy Latency")
    print(f"wrote topology sweep to {args.out}")


if __name__ == "__main__":
    main()
