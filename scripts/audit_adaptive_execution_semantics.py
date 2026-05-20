#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from waferagent.adaptive_semantics import audit_policy_stage_map
from waferagent.utils import enforce_clean_git_tree, finalize_run_dir, init_run_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", default="results/round13_adaptive_semantics_audit")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clean-required", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    enforce_clean_git_tree(args.clean_required, args.allow_dirty)
    out = init_run_dir(args.out, {"run_type": "adaptive_semantics_audit", **vars(args)})
    source = Path(args.results) / "simulation" / "policy_effective_stage_map.csv"
    if not source.exists():
        raise FileNotFoundError(f"Missing policy stage map: {source}")
    stage_map = pd.read_csv(source)
    detail, summary = audit_policy_stage_map(stage_map)
    sim = out / "simulation"
    detail.to_csv(sim / "adaptive_stage_semantics.csv", index=False)
    summary.to_csv(sim / "adaptive_semantics_summary.csv", index=False)
    write_json(
        out / "model_selection.json",
        {
            "engine_used": "synthetic",
            "fallback_count": 0,
            "semantic_audit_pass": bool(not summary.empty and summary["pass"].astype(bool).all()),
        },
    )
    finalize_run_dir(out)
    print(f"Adaptive semantics audit complete: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
