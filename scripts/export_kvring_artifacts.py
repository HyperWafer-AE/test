#!/usr/bin/env python3
"""Export Round 2 paper artifact bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kvring.artifacts import export_round2_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/kvring_round2_paper_artifacts"))
    parser.add_argument("--no-clean", action="store_true")
    args = parser.parse_args()
    export_round2_artifacts(args.out, clean=not args.no_clean)
    print(f"exported KVRing Round 2 artifacts to {args.out}")


if __name__ == "__main__":
    main()
