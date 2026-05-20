from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from waferagent.policy_features import FEATURE_COLUMNS, features_from_row, normalized_feature_vector


@dataclass(frozen=True)
class PolicySelectorV2Decision:
    config_id: int
    predicted_policy: str
    score: float
    confidence: float
    model_type: str


class ThresholdTablePolicySelector:
    def __init__(self, thresholds: dict[str, float], model_type: str = "threshold_table_v2"):
        self.thresholds = thresholds
        self.model_type = model_type

    def predict_row(self, row: pd.Series | dict[str, object]) -> PolicySelectorV2Decision:
        features = features_from_row(row)
        norm = normalized_feature_vector(features)
        score = (
            0.35 * norm["shared_prefix_tokens"]
            + 0.25 * norm["decode_tokens"]
            + 0.20 * norm["shared_kv_read_bytes_without_cohort"]
            + 0.08 * norm["cross_job_reuse_count"]
            + 0.06 * norm["intra_job_reuse_count"]
            - 0.16 * norm["estimated_queue_pressure"] * max(1.0, norm["num_consumers"])
            - 0.04 * norm["private_suffix_tokens"]
        )
        apc_t = float(self.thresholds.get("apc_threshold", 0.0))
        wafer_t = float(self.thresholds.get("waferagent_threshold", apc_t + 1.0))
        if score < apc_t:
            policy = "apc_like"
        elif score < wafer_t:
            policy = "pat_like_traffic_only"
        else:
            policy = "waferagent_latency_safe"
        confidence = min(1.0, abs(score - wafer_t) / max(1.0, abs(wafer_t)))
        config_id = int(float(row.get("config_id", -1))) if hasattr(row, "get") else -1
        return PolicySelectorV2Decision(config_id, policy, float(score), float(confidence), self.model_type)

    def to_json(self) -> dict[str, object]:
        return {"model_type": self.model_type, "thresholds": self.thresholds, "feature_columns": FEATURE_COLUMNS}

    @classmethod
    def from_json(cls, path: str | Path) -> "ThresholdTablePolicySelector":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(dict(payload.get("thresholds", {})), str(payload.get("model_type", "threshold_table_v2")))


def train_threshold_table(regimes: pd.DataFrame, oracle: pd.DataFrame) -> ThresholdTablePolicySelector:
    data = regimes.merge(oracle[["config_id", "oracle_best_policy"]], on="config_id", how="inner")
    if data.empty:
        return ThresholdTablePolicySelector({"apc_threshold": 8.0, "waferagent_threshold": 14.0})
    scores = []
    for _, row in data.iterrows():
        tmp = ThresholdTablePolicySelector({"apc_threshold": 0.0, "waferagent_threshold": 0.0})
        scores.append(tmp.predict_row(row).score)
    data = data.copy()
    data["score"] = scores
    apc_scores = data.loc[data["oracle_best_policy"] == "apc_like", "score"]
    waf_scores = data.loc[data["oracle_best_policy"] == "waferagent_latency_safe", "score"]
    pat_scores = data.loc[data["oracle_best_policy"] == "pat_like_traffic_only", "score"]
    apc_threshold = float(apc_scores.quantile(0.80)) if not apc_scores.empty else float(data["score"].quantile(0.35))
    if not pat_scores.empty:
        wafer_threshold = float(max(apc_threshold + 0.1, pat_scores.quantile(0.80)))
    elif not waf_scores.empty:
        wafer_threshold = float(max(apc_threshold + 0.1, waf_scores.quantile(0.20)))
    else:
        wafer_threshold = float(apc_threshold + 1.0)
    return ThresholdTablePolicySelector(
        {
            "apc_threshold": apc_threshold,
            "waferagent_threshold": wafer_threshold,
        }
    )


def evaluate_selector(selector: ThresholdTablePolicySelector, regimes: pd.DataFrame, oracle: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = regimes.merge(oracle, on="config_id", how="inner", suffixes=("", "_oracle"))
    decisions = []
    for _, row in merged.iterrows():
        decision = selector.predict_row(row)
        decisions.append(
            {
                "config_id": int(row["config_id"]),
                "split": row.get("split", ""),
                "predicted_policy": decision.predicted_policy,
                "oracle_best_policy": row.get("oracle_best_policy", ""),
                "score": decision.score,
                "confidence": decision.confidence,
                "correct": decision.predicted_policy == row.get("oracle_best_policy", ""),
            }
        )
    decisions_df = pd.DataFrame(decisions)
    rows = []
    for split in ["all", "train", "validation", "test"]:
        sub = decisions_df if split == "all" else decisions_df[decisions_df["split"] == split]
        if sub.empty:
            continue
        beneficial = sub["oracle_best_policy"].ne("apc_like")
        pred_beneficial = sub["predicted_policy"].ne("apc_like")
        tp = int((beneficial & pred_beneficial).sum())
        rows.append(
            {
                "split": split,
                "num_regimes": len(sub),
                "selection_accuracy_vs_oracle_best": float(sub["correct"].mean()),
                "beneficial_recall": tp / max(1, int(beneficial.sum())),
                "beneficial_precision": tp / max(1, int(pred_beneficial.sum())),
            }
        )
    return decisions_df, pd.DataFrame(rows)
