from __future__ import annotations

from dataclasses import dataclass

from waferagent.shared_kv import SharedKVObject
from waferagent.stage_ir import Stage


@dataclass(frozen=True)
class CohortAdmissionConfig:
    min_bytes_saved: int = 1_048_576
    max_allowed_jct_regression_ms: float = 0.0
    max_critical_wait_ms: float = 0.05
    slo_risk_threshold: float = 0.05
    query_capacity_per_shared_tile: int = 4
    reuse_efficiency_penalty: float = 1.15
    mesh_overhead_fraction: float = 0.05
    resource_delay_fraction: float = 0.10


@dataclass
class CohortAdmissionDecision:
    accepted: bool
    reason: str
    predicted_shared_kv_bytes_saved: float
    predicted_wait_cost_ms: float
    predicted_mesh_cost_ms: float
    predicted_resource_delay_ms: float
    predicted_jct_delta_ms: float
    predicted_slo_risk: float

    def to_dict(self) -> dict:
        return {
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "predicted_shared_kv_bytes_saved": float(self.predicted_shared_kv_bytes_saved),
            "predicted_wait_cost_ms": float(self.predicted_wait_cost_ms),
            "predicted_mesh_cost_ms": float(self.predicted_mesh_cost_ms),
            "predicted_resource_delay_ms": float(self.predicted_resource_delay_ms),
            "predicted_jct_delta_ms": float(self.predicted_jct_delta_ms),
            "predicted_slo_risk": float(self.predicted_slo_risk),
        }


def evaluate_cohort_candidate(
    obj: SharedKVObject,
    batch: list[Stage],
    ready_times: list[float],
    criticalities: list[float],
    bytes_per_ms: float,
    cfg: CohortAdmissionConfig | None = None,
) -> CohortAdmissionDecision:
    cfg = cfg or CohortAdmissionConfig()
    if len(batch) < 2:
        return CohortAdmissionDecision(False, "too_small", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    no_cohort = sum(max(1, s.output_tokens) * obj.kv_bytes for s in batch)
    max_output = max(max(1, s.output_tokens) for s in batch)
    waves = (len(batch) + cfg.query_capacity_per_shared_tile - 1) // cfg.query_capacity_per_shared_tile
    with_cohort = waves * max_output * obj.kv_bytes * cfg.reuse_efficiency_penalty
    saved = max(0.0, float(no_cohort - with_cohort))
    if saved < cfg.min_bytes_saved:
        return CohortAdmissionDecision(False, "low_saving", saved, 0.0, 0.0, 0.0, 0.0, 0.0)

    start = max(ready_times) if ready_times else 0.0
    waits = [max(0.0, start - rt) for rt in ready_times]
    max_wait = max(waits) if waits else 0.0
    critical_wait = max((w for w, c in zip(waits, criticalities) if c > 0.9), default=0.0)
    if critical_wait > cfg.max_critical_wait_ms:
        return CohortAdmissionDecision(
            False,
            "critical_path_wait",
            saved,
            max_wait,
            0.0,
            0.0,
            max_wait,
            1.0,
        )

    saved_latency_ms = saved / max(1.0, bytes_per_ms)
    mesh_cost_ms = (with_cohort * cfg.mesh_overhead_fraction) / max(1.0, bytes_per_ms)
    resource_delay_ms = max_wait * cfg.resource_delay_fraction
    predicted_jct_delta = max_wait + mesh_cost_ms + resource_delay_ms - saved_latency_ms
    predicted_slo_risk = max(0.0, predicted_jct_delta) / max(1.0, max_wait + saved_latency_ms)
    if predicted_slo_risk > cfg.slo_risk_threshold:
        reason = "slo_risk"
        accepted = False
    elif predicted_jct_delta > cfg.max_allowed_jct_regression_ms:
        reason = "jct_regression"
        accepted = False
    else:
        reason = "accepted"
        accepted = True
    return CohortAdmissionDecision(
        accepted,
        reason,
        saved,
        max_wait,
        mesh_cost_ms,
        resource_delay_ms,
        predicted_jct_delta,
        predicted_slo_risk,
    )
