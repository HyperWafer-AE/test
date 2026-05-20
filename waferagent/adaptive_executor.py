from __future__ import annotations

from dataclasses import dataclass

from waferagent.policy_assignment import PolicyAssignment
from waferagent.stage_ir import Stage


APC_POLICY = "apc_like"
PAT_POLICY = "pat_like_traffic_only"
WAFER_POLICY = "waferagent_latency_safe"


@dataclass(frozen=True)
class EffectivePolicy:
    selected_policy: str
    kv_sharing: bool
    decode_cohort: bool
    shared_kv_placement: bool
    latency_safe_admission: bool
    traffic_only_admission: bool


def policy_for_stage(stage: Stage, assignments: dict[str, PolicyAssignment]) -> str:
    if stage.shared_prefix_ids:
        return assignments.get(stage.shared_prefix_ids[0], PolicyAssignment(
            prefix_id=stage.shared_prefix_ids[0],
            scope="shared_kv_object",
            selected_policy=APC_POLICY,
            confidence=0.0,
            opportunity_score=0.0,
            predicted_delta_ms_vs_apc=0.0,
            predicted_delta_ms_vs_pat=0.0,
            reason="missing_assignment_default_apc",
        )).selected_policy
    return APC_POLICY


def effective_policy(selected_policy: str) -> EffectivePolicy:
    if selected_policy == WAFER_POLICY or selected_policy.startswith("waferagent"):
        return EffectivePolicy(
            selected_policy=selected_policy,
            kv_sharing=True,
            decode_cohort=True,
            shared_kv_placement=True,
            latency_safe_admission=True,
            traffic_only_admission=False,
        )
    if selected_policy == PAT_POLICY or selected_policy == "pat_like":
        return EffectivePolicy(
            selected_policy=selected_policy,
            kv_sharing=True,
            decode_cohort=True,
            shared_kv_placement=False,
            latency_safe_admission=False,
            traffic_only_admission=True,
        )
    return EffectivePolicy(
        selected_policy=APC_POLICY,
        kv_sharing=True,
        decode_cohort=False,
        shared_kv_placement=False,
        latency_safe_admission=False,
        traffic_only_admission=False,
    )
