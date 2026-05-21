"""Admission and objective utilities for KVRing-v3 adaptive selection."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .accounting import ModeResult
from .validation import result_capacity_valid


@dataclass(frozen=True)
class AdmissionWeights:
    lambda_hotspot: float = 0.0
    invalid_capacity_penalty: float = math.inf


def attention_latency(result: ModeResult) -> float:
    return float(
        result.extra.get(
            "attention_stage_proxy_latency_s",
            result.extra.get("throughput_bound_latency_s", result.estimated_latency_seconds),
        )
    )


def normalized_max_link_load(result: ModeResult, normalizer: float | None = None) -> float:
    denom = max(float(normalizer or result.max_link_load_bytes or 1.0), 1.0)
    return float(result.max_link_load_bytes) / denom


def candidate_objective(
    result: ModeResult,
    *,
    weights: AdmissionWeights = AdmissionWeights(),
    max_link_normalizer: float | None = None,
) -> float:
    objective = attention_latency(result)
    objective += weights.lambda_hotspot * normalized_max_link_load(result, max_link_normalizer)
    if not result_capacity_valid(result):
        objective += weights.invalid_capacity_penalty
    return objective


def select_best_candidate(
    candidates: list[ModeResult],
    *,
    weights: AdmissionWeights = AdmissionWeights(),
) -> tuple[ModeResult, list[tuple[ModeResult, float]]]:
    if not candidates:
        raise ValueError("at least one candidate is required")
    normalizer = max((float(c.max_link_load_bytes) for c in candidates), default=1.0)
    scored = [
        (candidate, candidate_objective(candidate, weights=weights, max_link_normalizer=normalizer))
        for candidate in candidates
    ]
    finite = [(candidate, score) for candidate, score in scored if math.isfinite(score)]
    if not finite:
        finite = scored
    best = min(finite, key=lambda item: item[1])[0]
    return best, scored
