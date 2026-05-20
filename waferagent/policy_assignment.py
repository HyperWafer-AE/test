from __future__ import annotations

from dataclasses import dataclass

from waferagent.policy_selector import PolicyDecision, choose_shared_kv_policy
from waferagent.shared_kv import SharedKVObject


@dataclass
class PolicyAssignment:
    prefix_id: str
    scope: str
    selected_policy: str
    confidence: float
    opportunity_score: float
    predicted_delta_ms_vs_apc: float
    predicted_delta_ms_vs_pat: float
    reason: str

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def assignment_from_decision(decision: PolicyDecision) -> PolicyAssignment:
    apc = float(decision.predicted_apc_cost_proxy)
    pat = float(decision.predicted_pat_cost_proxy)
    waf = float(decision.predicted_waferagent_cost_proxy)
    best_gap = max(apc, pat, waf) - min(apc, pat, waf)
    denom = max(1.0, abs(apc) + abs(pat) + abs(waf))
    return PolicyAssignment(
        prefix_id=decision.scope_id,
        scope="shared_kv_object",
        selected_policy=decision.chosen_policy,
        confidence=max(0.0, min(1.0, best_gap / denom)),
        opportunity_score=float(decision.opportunity_score),
        predicted_delta_ms_vs_apc=waf - apc,
        predicted_delta_ms_vs_pat=waf - pat,
        reason=decision.reason,
    )


def build_policy_assignments(
    shared_kv_objects: list[SharedKVObject],
    resource_state: dict[str, float] | None = None,
) -> dict[str, PolicyAssignment]:
    assignments: dict[str, PolicyAssignment] = {}
    for obj in shared_kv_objects:
        decision = choose_shared_kv_policy(
            obj,
            graph_context={"mesh_distance_to_consumers": max(1.0, len(obj.candidate_regions))},
            resource_state=resource_state or {},
        )
        assignments[obj.prefix_id] = assignment_from_decision(decision)
    return assignments

