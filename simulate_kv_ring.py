#!/usr/bin/env python3
"""Compatibility wrapper for the KVRing Round 2 simulator package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kvring.artifacts import write_default_artifacts
from kvring.units import fmt_bytes, fmt_seconds


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KVRing Round 2 default baselines")
    parser.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--round2-full", action="store_true", help="include Central-KV and KVRing-v2 modes")
    args = parser.parse_args()
    results = write_default_artifacts(args.outdir, legacy_only=not args.round2_full)

    print("\n=== KVRing Round 3 Wafer Mesh Simulator ===")
    print("NoC accounting: directed bidirectional channels; VC model is not modeled for performance.")
    print("Scope: attention-only shared-prefix plus local suffix. Non-attention LLM layers are not included.\n")
    header = (
        f"{'Mode':<34} {'Peak SRAM':>12} {'Wire':>12} "
        f"{'Max dir link':>14} {'Attn proxy':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        latency = float(
            r.extra.get(
                "attention_stage_proxy_latency_s",
                r.extra.get("throughput_bound_latency_s", r.estimated_latency_seconds),
            )
        )
        print(
            f"{r.mode:<34} {fmt_bytes(r.peak_region_sram_bytes):>12} "
            f"{fmt_bytes(r.total_wire_bytes):>12} {fmt_bytes(r.max_link_load_bytes):>14} "
            f"{fmt_seconds(latency):>12}"
        )
    print("\nSaved:")
    for name in [
        "kv_ring_results.json",
        "kv_ring_report.txt",
        "kv_ring_comparison.png",
        "kv_ring_comparison.pdf",
        "kv_ring_link_loads.png",
    ]:
        print(f"  {args.outdir / name}")
    print("\nStructured summary:")
    print(json.dumps([r.to_summary_dict() for r in results], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
