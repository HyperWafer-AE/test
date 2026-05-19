from __future__ import annotations

from dataclasses import dataclass


VALID_ACCOUNTING_MODES = {"stage_amortized", "cohort_stage", "per_member"}


@dataclass(frozen=True)
class SharedAttentionAccountingResult:
    member_latency_ms: float
    cohort_stage_latency_ms: float
    charged_once: bool
    mode: str


def normalize_accounting_mode(mode: str | None) -> str:
    normalized = (mode or "cohort_stage").strip().lower().replace("-", "_")
    aliases = {
        "stage_amortized_current": "stage_amortized",
        "amortized": "stage_amortized",
        "cohort_single_resource_stage": "cohort_stage",
        "single_stage": "cohort_stage",
        "no_extra_amortization": "per_member",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in VALID_ACCOUNTING_MODES:
        raise ValueError(f"Unknown shared-attention accounting mode: {mode}")
    return normalized


def account_shared_attention_latency(
    predicted_ms: float,
    num_agents: int,
    member_index: int,
    mode: str | None,
) -> SharedAttentionAccountingResult:
    mode = normalize_accounting_mode(mode)
    predicted_ms = max(0.0, float(predicted_ms))
    agents = max(1, int(num_agents))
    if mode == "stage_amortized":
        return SharedAttentionAccountingResult(predicted_ms / agents, predicted_ms, False, mode)
    if mode == "cohort_stage":
        return SharedAttentionAccountingResult(predicted_ms if member_index == 0 else 0.0, predicted_ms, True, mode)
    return SharedAttentionAccountingResult(predicted_ms, predicted_ms, False, mode)

