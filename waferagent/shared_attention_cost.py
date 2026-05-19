from __future__ import annotations

from dataclasses import dataclass

from waferagent.cohort_scheduler import DecodeCohort
from waferagent.shared_kv import SharedKVObject


@dataclass(frozen=True)
class SharedAttentionParams:
    query_capacity_per_shared_tile: int = 4
    reuse_efficiency_penalty: float = 1.15
    merge_overhead_factor: float = 0.03
    query_transfer_factor: float = 1.0


def estimate_shared_attention_cost(
    objects: list[SharedKVObject],
    cohorts: list[DecodeCohort] | None = None,
    params: SharedAttentionParams | None = None,
    bytes_per_ms: float = 50e9 / 1000.0,
    private_tokens_by_node: dict[str, int] | None = None,
    output_tokens_by_node: dict[str, int] | None = None,
    kv_bytes_per_token: int | None = None,
) -> dict[str, float]:
    params = params or SharedAttentionParams()
    cohorts = cohorts or []
    no_cohort_shared = float(sum(o.expected_decode_kv_read_bytes_without_cohort for o in objects))
    cohort_prefixes = {c.shared_kv_id for c in cohorts}
    with_cohort = 0.0
    query_transfer = 0.0
    merge_bytes = 0.0
    for obj in objects:
        if obj.prefix_id not in cohort_prefixes:
            with_cohort += obj.expected_decode_kv_read_bytes_without_cohort
            continue
        obj_cohorts = [c for c in cohorts if c.shared_kv_id == obj.prefix_id]
        for cohort in obj_cohorts:
            agents = max(1, len(cohort.node_ids))
            waves = (agents + max(1, params.query_capacity_per_shared_tile) - 1) // max(1, params.query_capacity_per_shared_tile)
            steps = max(1, obj.expected_decode_steps)
            with_cohort += waves * steps * obj.kv_bytes * params.reuse_efficiency_penalty
            query_transfer += cohort.expected_query_transfer_bytes * params.query_transfer_factor
            merge_bytes += cohort.expected_merge_bytes * params.merge_overhead_factor
    private = 0.0
    private_unavailable = 1.0
    if private_tokens_by_node is not None and kv_bytes_per_token is not None:
        private_unavailable = 0.0
        for obj in objects:
            for node_id, decode_tokens in obj.expected_decode_tokens.items():
                private_tokens = max(0, int(private_tokens_by_node.get(node_id, 0)))
                out_tokens = max(0, int(output_tokens_by_node.get(node_id, decode_tokens))) if output_tokens_by_node else max(0, int(decode_tokens))
                private += float(private_tokens * int(kv_bytes_per_token) * out_tokens)
    total_with = with_cohort + query_transfer + merge_bytes
    saved = max(0.0, no_cohort_shared - with_cohort)
    return {
        "shared_kv_bytes_read_without_cohort": no_cohort_shared,
        "shared_kv_bytes_read_with_cohort": with_cohort,
        "decode_shared_kv_read_bytes": with_cohort,
        "decode_private_kv_read_bytes": private,
        "decode_private_kv_read_bytes_unavailable": private_unavailable,
        "decode_query_transfer_bytes": query_transfer,
        "decode_merge_bytes": merge_bytes,
        "decode_kv_read_reduction_ratio": saved / no_cohort_shared if no_cohort_shared else 0.0,
        "shared_kv_read_reduction_ratio": saved / no_cohort_shared if no_cohort_shared else 0.0,
        "decode_attention_latency_ms": total_with / max(1.0, bytes_per_ms),
    }
