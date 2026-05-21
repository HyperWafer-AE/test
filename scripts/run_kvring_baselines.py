#!/usr/bin/env python3
"""Run the required KVRing Round 2 baseline comparison."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kvring.artifacts import plot_simple_bar, result_row, run_mode, write_csv, write_json  # noqa: E402
from kvring.config import HardwareConfig, ModelConfig, WorkloadConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shared-prefix-tokens", type=int, default=32768)
    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--decode-tokens-per-agent", type=int, default=256)
    parser.add_argument("--mesh-rows", type=int, default=16)
    parser.add_argument("--mesh-cols", type=int, default=16)
    parser.add_argument("--ring-shards", type=int, default=8)
    parser.add_argument("--query-tile-size", type=int, default=8)
    parser.add_argument("--link-bandwidth-gbps", type=float, default=100.0)
    parser.add_argument("--region-sram-bandwidth-tibps", type=float, default=20.0)
    parser.add_argument("--clean-required", action="store_true")
    args = parser.parse_args()
    if args.clean_required and args.out.exists():
        shutil.rmtree(args.out)
    sim_dir = args.out / "simulation"
    fig_dir = args.out / "figures"
    sim_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    model = ModelConfig(query_tile_size=args.query_tile_size)
    workload = WorkloadConfig(
        shared_prefix_tokens=args.shared_prefix_tokens,
        concurrent_agents=args.agents,
        decode_tokens_per_agent=args.decode_tokens_per_agent,
    )
    hardware = HardwareConfig(
        mesh_rows=args.mesh_rows,
        mesh_cols=args.mesh_cols,
        link_bandwidth_gbps=args.link_bandwidth_gbps,
        region_sram_bandwidth_tibps=args.region_sram_bandwidth_tibps,
    )
    modes = [
        "replicate_all",
        "pull_kv_independent",
        "central_kv_stationary",
        "kvring_v1",
        "kvring_v2_query_tiled_parallel_ring_reduce",
        "kvring_v2_query_tiled_parallel_tree_reduce",
    ]
    results = [
        run_mode(
            m,
            model,
            workload,
            hardware,
            query_tile_size=args.query_tile_size,
            num_shards=args.ring_shards,
        )
        for m in modes
    ]
    rows = [
        result_row(
            r,
            shared_prefix_tokens=args.shared_prefix_tokens,
            shared_kv_bytes=workload.shared_kv_bytes(model),
            agents=args.agents,
            decode_tokens=args.decode_tokens_per_agent,
            query_tile_size=args.query_tile_size,
            num_shards=args.ring_shards,
        )
        for r in results
    ]
    write_csv(sim_dir / "baseline_summary.csv", rows)
    write_csv(sim_dir / "sram_summary.csv", rows)
    link_rows = []
    for r in results:
        for item in r.top_links(8):
            link_rows.append({"mode": r.mode, "reduction_topology": r.extra.get("reduction_topology", ""), **item})
    write_csv(sim_dir / "link_load_summary.csv", link_rows)
    write_json(sim_dir / "baseline_results.json", {"results": [r.to_full_dict() for r in results]})

    plot_simple_bar(rows, "sram_peak_region_bytes", "mode", fig_dir / "fig_baseline_sram", "Baseline Peak Region SRAM")
    plot_simple_bar(rows, "total_wire_bytes", "mode", fig_dir / "fig_baseline_wire_traffic", "Baseline Directed Wire Traffic")
    plot_simple_bar(rows, "max_directed_link_load_bytes", "mode", fig_dir / "fig_baseline_hotspot", "Baseline Hottest Directed Link")
    plot_simple_bar(rows, "attention_stage_proxy_latency_s", "mode", fig_dir / "fig_baseline_attention_stage_latency", "Shared-Prefix Attention Stage Proxy")
    print(f"wrote baseline artifacts to {args.out}")


if __name__ == "__main__":
    main()
