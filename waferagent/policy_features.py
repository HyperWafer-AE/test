from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


FEATURE_COLUMNS = [
    "shared_prefix_tokens",
    "private_suffix_tokens",
    "decode_tokens",
    "num_consumers",
    "intra_job_reuse_count",
    "cross_job_reuse_count",
    "reuse_group_size",
    "fan_in_degree",
    "estimated_queue_pressure",
    "shared_kv_read_bytes_without_cohort",
    "expected_cohort_size",
]


@dataclass(frozen=True)
class PolicyFeatures:
    shared_prefix_tokens: float
    private_suffix_tokens: float
    decode_tokens: float
    num_consumers: float
    intra_job_reuse_count: float
    cross_job_reuse_count: float
    reuse_group_size: float
    fan_in_degree: float
    estimated_queue_pressure: float
    shared_kv_read_bytes_without_cohort: float
    expected_cohort_size: float

    def to_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


def features_from_row(row: pd.Series | dict[str, object]) -> PolicyFeatures:
    def f(name: str, default: float = 0.0) -> float:
        value = row.get(name, default) if isinstance(row, dict) else row.get(name, default)
        try:
            if pd.isna(value):
                return default
        except Exception:
            pass
        try:
            return float(value)
        except Exception:
            return default

    shared = f("shared_prefix_tokens")
    decode = f("decode_tokens")
    agents = f("num_agents_per_job", f("num_consumers", 1.0))
    reuse_group = f("reuse_group_size", 1.0)
    arrival = f("arrival_rate_jobs_per_s", 1.0)
    fanin = 1.0 if bool(row.get("fanin", False)) else 0.0
    return PolicyFeatures(
        shared_prefix_tokens=shared,
        private_suffix_tokens=f("private_suffix_tokens"),
        decode_tokens=decode,
        num_consumers=agents,
        intra_job_reuse_count=max(0.0, agents - 1.0),
        cross_job_reuse_count=max(0.0, min(reuse_group, f("num_jobs", reuse_group)) - 1.0),
        reuse_group_size=reuse_group,
        fan_in_degree=fanin * agents,
        estimated_queue_pressure=arrival / 32.0,
        shared_kv_read_bytes_without_cohort=shared * decode * agents,
        expected_cohort_size=max(1.0, min(agents, 16.0)),
    )


def normalized_feature_vector(features: PolicyFeatures) -> dict[str, float]:
    raw = features.to_dict()
    out: dict[str, float] = {}
    for key, value in raw.items():
        if key.endswith("tokens") or key.endswith("bytes_without_cohort"):
            out[key] = math.log1p(max(0.0, value))
        else:
            out[key] = float(value)
    return out
