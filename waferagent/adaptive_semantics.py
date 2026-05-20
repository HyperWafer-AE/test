from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class StageSemantics:
    stage_id: str
    prefix_id: str
    selected_policy: str
    placement_policy_applied: str
    scheduling_policy_applied: str
    kv_sharing_applied: bool
    decode_cohort_applied: bool
    shared_kv_placement_applied: bool
    mesh_congestion_policy_applied: str
    resource_partition_policy_applied: str
    latency_safe_admission: bool
    is_semantically_faithful: bool
    violation_reason: str


def infer_stage_semantics(row: dict[str, object]) -> StageSemantics:
    selected = str(row.get("selected_policy", "apc_like") or "apc_like")
    stage_type = str(row.get("stage_type", ""))
    prefix_id = str(row.get("prefix_id", ""))
    stage_id = str(row.get("global_stage_id", row.get("stage_id", "")))
    kv_sharing = bool(prefix_id)
    decode_stage = stage_type == "decode"

    if selected == "pat_like_traffic_only" or selected == "pat_like":
        decode_cohort = decode_stage and bool(prefix_id)
        shared_kv_placement = False
        latency_safe = False
    elif selected == "waferagent_latency_safe" or selected.startswith("waferagent"):
        decode_cohort = decode_stage and bool(prefix_id)
        shared_kv_placement = bool(prefix_id)
        latency_safe = True
    else:
        decode_cohort = False
        shared_kv_placement = False
        latency_safe = False

    violations: list[str] = []
    if selected == "apc_like":
        if decode_cohort:
            violations.append("apc_decode_cohort_enabled")
        if shared_kv_placement:
            violations.append("apc_shared_kv_placement_enabled")
        if not kv_sharing and prefix_id:
            violations.append("apc_kv_sharing_disabled")
    elif selected == "pat_like_traffic_only":
        if decode_stage and prefix_id and not decode_cohort:
            violations.append("pat_decode_cohort_disabled")
        if shared_kv_placement:
            violations.append("pat_shared_kv_placement_enabled")
    elif selected == "waferagent_latency_safe":
        if decode_stage and prefix_id and not decode_cohort:
            violations.append("waferagent_decode_cohort_disabled")
        if prefix_id and not shared_kv_placement:
            violations.append("waferagent_shared_kv_placement_disabled")
        if not latency_safe:
            violations.append("waferagent_latency_safe_admission_disabled")

    return StageSemantics(
        stage_id=stage_id,
        prefix_id=prefix_id,
        selected_policy=selected,
        placement_policy_applied="global_orchestrator",
        scheduling_policy_applied="global_orchestrator",
        kv_sharing_applied=kv_sharing,
        decode_cohort_applied=decode_cohort,
        shared_kv_placement_applied=shared_kv_placement,
        mesh_congestion_policy_applied="global_orchestrator",
        resource_partition_policy_applied="global_orchestrator",
        latency_safe_admission=latency_safe,
        is_semantically_faithful=not violations,
        violation_reason=";".join(violations),
    )


def audit_policy_stage_map(stage_map: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [infer_stage_semantics(row).__dict__ for row in stage_map.to_dict(orient="records")]
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame(
            columns=[
                "selected_policy",
                "stages",
                "faithful_stages",
                "violations",
                "pass",
                "semantic_note",
            ]
        )
    summary = (
        detail.groupby("selected_policy", as_index=False)
        .agg(
            stages=("stage_id", "count"),
            faithful_stages=("is_semantically_faithful", "sum"),
            violations=("is_semantically_faithful", lambda x: int((~x.astype(bool)).sum())),
        )
    )
    summary["pass"] = summary["violations"].astype(int).eq(0)
    summary["semantic_note"] = (
        "policy-specific execution under WaferAgent global_orchestrator; "
        "placement/scheduling are orchestrator-level semantics, not pure engine-level APC/PAT"
    )
    return detail, summary
