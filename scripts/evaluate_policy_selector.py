#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from waferagent.policy_selector_v2 import ThresholdTablePolicySelector, evaluate_selector
from waferagent.utils import enforce_clean_git_tree, finalize_run_dir, init_run_dir, write_json


def _find_oracle(regimes_path: Path) -> Path:
    candidate = regimes_path.with_name("randomized_policy_oracle_labels.csv")
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not infer oracle labels next to {regimes_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--regimes", required=True)
    parser.add_argument("--oracle-labels", default="")
    parser.add_argument("--out", default="results/round13_policy_selector_v2_eval")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clean-required", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    enforce_clean_git_tree(args.clean_required, args.allow_dirty)
    out = init_run_dir(args.out, {"run_type": "evaluate_policy_selector_v2", **vars(args)})
    selector = ThresholdTablePolicySelector.from_json(args.model)
    regimes_path = Path(args.regimes)
    oracle_path = Path(args.oracle_labels) if args.oracle_labels else _find_oracle(regimes_path)
    regimes = pd.read_csv(regimes_path).drop_duplicates("config_id")
    oracle = pd.read_csv(oracle_path).drop_duplicates("config_id")
    decisions, eval_df = evaluate_selector(selector, regimes, oracle)
    sim = out / "simulation"
    decisions.to_csv(sim / "policy_selector_v2_decisions.csv", index=False)
    eval_df.to_csv(sim / "policy_selector_v2_eval.csv", index=False)
    write_json(out / "model_selection.json", {"engine_used": "synthetic", "fallback_count": 0, "model_type": selector.model_type})
    finalize_run_dir(out)
    print(f"Policy selector v2 evaluated: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
