#!/usr/bin/env python3
"""Run KVRing query-tile sweep."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kvring.artifacts import generate_query_tile_sweep


def int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def str_list(s: str) -> list[str]:
    return [x for x in s.split(",") if x]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", type=str_list, required=True)
    parser.add_argument("--query-tile-sizes", type=int_list, required=True)
    parser.add_argument("--shared-prefix-tokens", type=int_list, required=True)
    parser.add_argument("--agents", type=int_list, required=True)
    parser.add_argument("--decode-tokens", type=int_list, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clean-required", action="store_true")
    args = parser.parse_args()
    if args.clean_required and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = generate_query_tile_sweep(
        args.out / "query_tile_sweep.csv",
        args.modes,
        args.query_tile_sizes,
        args.shared_prefix_tokens,
        args.agents,
        args.decode_tokens,
    )
    print(f"wrote {len(rows)} rows to {args.out / 'query_tile_sweep.csv'}")


if __name__ == "__main__":
    main()
