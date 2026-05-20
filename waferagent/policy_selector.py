from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from waferagent.kv_model import ModelKVConfig
from waferagent.shared_kv import SharedKVObject
from waferagent.trace_schema import TraceRecord


@dataclass
class PolicyDecision:
    scope_id: str
    chosen_policy: str
    predicted_apc_jct_ms: float
    predicted_pat_jct_ms: float
    predicted_waferagent_jct_ms: float
    opportunity_score: float
    reason: str

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class PolicySelectorConfig:
    bytes_per_ms: float = 50e9 / 1000.0
    min_waferagent_score_ms: float = 1.0
    min_pat_score_ms: float = 0.1


def choose_shared_kv_policy(
    shared_kv_object: SharedKVObject,
    graph_context: dict[str, float] | None = None,
    resource_state: dict[str, float] | None = None,
    cost_model: object | None = None,
    cfg: PolicySelectorConfig | None = None,
) -> PolicyDecision:
    cfg = cfg or PolicySelectorConfig()
    ctx = graph_context or {}
    state = resource_state or {}
    token_len = float(shared_kv_object.token_len)
    decode_tokens = float(sum(shared_kv_object.expected_decode_tokens.values()))
    consumers = float(max(1, len(set(shared_kv_object.decode_users))))
    reuse_count = float(max(0, consumers - 1))
    mesh_distance = float(ctx.get("mesh_distance_to_consumers", max(1.0, len(shared_kv_object.candidate_regions))))
    sram_pressure = float(state.get("sram_pressure", 0.0))
    queue_pressure = float(state.get("queue_pressure", 0.0))
    slack = float(ctx.get("critical_path_slack", 0.0))
    kv_bytes = float(shared_kv_object.kv_bytes)

    shared_kv_read_saved_ms = kv_bytes * max(0.0, decode_tokens) * reuse_count / max(1.0, cfg.bytes_per_ms * max(1.0, consumers))
    mesh_saved_ms = shared_kv_read_saved_ms * min(4.0, mesh_distance) * 0.15
    sram_reload_saved_ms = shared_kv_read_saved_ms * min(1.0, sram_pressure) * 0.05
    cohort_wait_ms = 0.0 if slack > 0 else min(2.0, consumers * 0.05)
    placement_overhead_ms = 0.02 * consumers
    queue_risk_ms = queue_pressure * (0.5 + 0.1 * consumers)
    cost = cohort_wait_ms + placement_overhead_ms + queue_risk_ms
    benefit = shared_kv_read_saved_ms + mesh_saved_ms + sram_reload_saved_ms
    score = benefit - cost

    predicted_apc = max(0.0, benefit)
    predicted_pat = max(0.0, benefit - shared_kv_read_saved_ms * 0.45)
    predicted_wafer = max(0.0, predicted_apc - score)
    # Large shared objects with many simultaneous consumers are precisely where
    # the current simulator shows queue/SRAM amplification. The adaptive policy
    # must be allowed to fall back instead of blindly enabling WaferAgent.
    high_queue_risk = token_len >= 8192 and consumers >= 16
    if high_queue_risk:
        chosen = "apc_like"
        reason = "high_queue_and_sram_risk"
    elif score <= 0:
        chosen = "apc_like"
        reason = "non_positive_opportunity"
    elif score < cfg.min_waferagent_score_ms:
        chosen = "pat_like_traffic_only"
        reason = "traffic_saving_without_latency_confidence"
    else:
        chosen = "waferagent_latency_safe"
        reason = "positive_latency_safe_opportunity"
    return PolicyDecision(
        scope_id=shared_kv_object.prefix_id,
        chosen_policy=chosen,
        predicted_apc_jct_ms=predicted_apc,
        predicted_pat_jct_ms=predicted_pat,
        predicted_waferagent_jct_ms=predicted_wafer,
        opportunity_score=score,
        reason=reason,
    )


def decisions_from_traces(
    traces: Iterable[TraceRecord],
    model_cfg: ModelKVConfig | None = None,
    cfg: PolicySelectorConfig | None = None,
) -> list[PolicyDecision]:
    model_cfg = model_cfg or ModelKVConfig()
    grouped: dict[str, dict[str, object]] = {}
    for tr in traces:
        if not tr.shared_prefix_ids:
            continue
        pid = tr.shared_prefix_ids[0]
        item = grouped.setdefault(pid, {"jobs": set(), "nodes": set(), "decode_tokens": {}, "token_len": 0})
        item["jobs"].add(tr.job_id)  # type: ignore[union-attr]
        item["nodes"].add(tr.node_id)  # type: ignore[union-attr]
        item["decode_tokens"][tr.node_id] = int(tr.output_tokens)  # type: ignore[index]
        item["token_len"] = max(int(item["token_len"]), int(tr.shared_prefix_token_len))
    decisions: list[PolicyDecision] = []
    for pid, item in grouped.items():
        token_len = int(item["token_len"])
        node_ids = sorted(item["nodes"])  # type: ignore[arg-type]
        obj = SharedKVObject(
            prefix_id=pid,
            token_len=token_len,
            kv_bytes=int(token_len * model_cfg.kv_bytes_per_token),
            logical_users=node_ids,
            decode_users=node_ids,
            producer_node=None,
            first_use_step=0,
            last_use_step=max(0, len(node_ids) - 1),
            expected_decode_tokens=dict(item["decode_tokens"]),  # type: ignore[arg-type]
            expected_decode_steps=max(1, max(dict(item["decode_tokens"]).values(), default=1)),  # type: ignore[arg-type]
            candidate_regions=[],
        )
        jobs = item["jobs"]  # type: ignore[assignment]
        decisions.append(
            choose_shared_kv_policy(
                obj,
                graph_context={"mesh_distance_to_consumers": max(1.0, len(jobs))},
                resource_state={"queue_pressure": 0.0},
                cfg=cfg,
            )
        )
    return decisions


def choose_run_policy_from_traces(traces: Iterable[TraceRecord], model_cfg: ModelKVConfig | None = None) -> tuple[str, list[PolicyDecision]]:
    decisions = decisions_from_traces(traces, model_cfg=model_cfg)
    if not decisions:
        return "apc_like", []
    counts: dict[str, int] = {}
    score_by_policy: dict[str, float] = {}
    for d in decisions:
        counts[d.chosen_policy] = counts.get(d.chosen_policy, 0) + 1
        score_by_policy[d.chosen_policy] = score_by_policy.get(d.chosen_policy, 0.0) + d.opportunity_score
    # Prefer the highest aggregate opportunity, with APC as the conservative
    # default when no positive WaferAgent/PAT opportunity is found.
    positive = {k: v for k, v in score_by_policy.items() if v > 0 and k != "apc_like"}
    if positive:
        return max(positive, key=positive.get), decisions
    return "apc_like", decisions
