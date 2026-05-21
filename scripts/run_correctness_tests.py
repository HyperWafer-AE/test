#!/usr/bin/env python3
"""Run online-softmax numerical correctness experiment."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kvring.artifacts import generate_online_softmax_correctness, write_csv  # noqa: E402


def int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def str_list(s: str) -> list[str]:
    return [x for x in s.split(",") if x]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shards", type=int_list, default=[2, 4, 8, 16])
    p.add_argument("--dtype", type=str_list, default=["fp32", "bf16"])
    p.add_argument("--clean-required", action="store_true")
    args = p.parse_args()
    if args.clean_required and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = generate_online_softmax_correctness(args.out / "correctness_summary.csv")
    rows = [r for r in rows if int(r["num_shards"]) in args.shards and r["kv_storage_precision"] in args.dtype]
    write_csv(args.out / "correctness_summary.csv", rows)
    write_csv(args.out / "numerical_error_summary.csv", rows)
    print(f"wrote correctness rows={len(rows)} to {args.out}")


if __name__ == "__main__":
    main()
