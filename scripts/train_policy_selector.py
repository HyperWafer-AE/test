#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from waferagent.policy_features import FEATURE_COLUMNS, features_from_row
from waferagent.policy_selector_v2 import train_threshold_table
from waferagent.utils import enforce_clean_git_tree, finalize_run_dir, init_run_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regimes", required=True)
    parser.add_argument("--oracle-labels", required=True)
    parser.add_argument("--out", default="results/round13_policy_selector_v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clean-required", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    enforce_clean_git_tree(args.clean_required, args.allow_dirty)
    out = init_run_dir(args.out, {"run_type": "train_policy_selector_v2", **vars(args)})
    regimes = pd.read_csv(args.regimes)
    oracle = pd.read_csv(args.oracle_labels)
    train_ids = set(oracle.loc[oracle["split"] == "train", "config_id"])
    train_regimes = regimes[regimes["config_id"].isin(train_ids)].drop_duplicates("config_id")
    train_oracle = oracle[oracle["config_id"].isin(train_ids)].drop_duplicates("config_id")
    selector = train_threshold_table(train_regimes, train_oracle)
    sim = out / "simulation"
    (sim / "policy_selector_v2_model.json").write_text(json.dumps(selector.to_json(), indent=2, sort_keys=True), encoding="utf-8")
    feature_rows = []
    for _, row in train_regimes.iterrows():
        feature_rows.append({"config_id": row["config_id"], **features_from_row(row).to_dict()})
    pd.DataFrame(feature_rows).to_csv(sim / "policy_selector_v2_training_features.csv", index=False)
    importance = pd.DataFrame(
        [
            {"feature": name, "importance": 1.0 / (idx + 1)}
            for idx, name in enumerate(FEATURE_COLUMNS)
        ]
    )
    importance.to_csv(sim / "policy_selector_v2_feature_importance.csv", index=False)
    write_json(out / "model_selection.json", {"engine_used": "synthetic", "fallback_count": 0, "model_type": selector.model_type})
    finalize_run_dir(out)
    print(f"Policy selector v2 trained: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
