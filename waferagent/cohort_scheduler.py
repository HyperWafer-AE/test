from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from waferagent.shared_kv import SharedKVObject
from waferagent.stage_ir import Stage


@dataclass
class DecodeCohort:
    cohort_id: str
    shared_kv_id: str
    node_ids: list[str]
    planned_start_ms: float
    max_wait_ms: float
    shared_kv_region: str | None
    query_source_regions: dict[str, str]
    private_kv_regions: dict[str, str]
    expected_shared_kv_bytes_read: int
    expected_private_kv_bytes_read: int
    expected_query_transfer_bytes: int
    expected_merge_bytes: int

    def to_dict(self) -> dict:
        return {
            "cohort_id": self.cohort_id,
            "shared_kv_id": self.shared_kv_id,
            "node_ids": ",".join(self.node_ids),
            "planned_start_ms": self.planned_start_ms,
            "max_wait_ms": self.max_wait_ms,
            "shared_kv_region": self.shared_kv_region or "",
            "expected_shared_kv_bytes_read": self.expected_shared_kv_bytes_read,
            "expected_private_kv_bytes_read": self.expected_private_kv_bytes_read,
            "expected_query_transfer_bytes": self.expected_query_transfer_bytes,
            "expected_merge_bytes": self.expected_merge_bytes,
            "cohort_size": len(self.node_ids),
        }


@dataclass(frozen=True)
class CohortConfig:
    enabled: bool = True
    min_group_size: int = 2
    max_group_size: int = 16
    max_wait_ms: float = 2.0
    max_critical_wait_ms: float = 0.2
    min_shared_prefix_tokens: int = 1024
    min_expected_saved_kv_bytes: int = 1_048_576


def form_decode_cohorts(
    stages: dict[str, Stage],
    shared_objects: list[SharedKVObject],
    ready_times: dict[str, float] | None = None,
    criticality: dict[str, float] | None = None,
    node_regions: dict[str, str] | None = None,
    cfg: CohortConfig | None = None,
) -> tuple[list[DecodeCohort], dict[str, float]]:
    cfg = cfg or CohortConfig()
    if not cfg.enabled:
        return [], _stats([])
    ready_times = ready_times or {}
    criticality = criticality or {}
    node_regions = node_regions or {}
    obj_by_prefix = {o.prefix_id: o for o in shared_objects}
    decode_by_prefix: dict[str, list[Stage]] = {}
    for stage in stages.values():
        if stage.stage_type != "decode" or not stage.shared_prefix_ids:
            continue
        prefix = stage.shared_prefix_ids[0]
        obj = obj_by_prefix.get(prefix)
        if obj is None or obj.token_len < cfg.min_shared_prefix_tokens:
            continue
        decode_by_prefix.setdefault(prefix, []).append(stage)

    cohorts: list[DecodeCohort] = []
    for prefix, group in sorted(decode_by_prefix.items()):
        obj = obj_by_prefix[prefix]
        group = sorted(group, key=lambda s: (ready_times.get(s.stage_id, 0.0), s.stage_id))
        batch: list[Stage] = []
        batch_start = 0.0
        for stage in group:
            ready = ready_times.get(stage.stage_id, 0.0)
            if not batch:
                batch = [stage]
                batch_start = ready
                continue
            max_wait = cfg.max_critical_wait_ms if criticality.get(stage.stage_id, 0.0) > 0.9 else cfg.max_wait_ms
            if len(batch) < cfg.max_group_size and ready - batch_start <= max_wait:
                batch.append(stage)
            else:
                _append_if_valid(cohorts, obj, batch, batch_start, cfg, node_regions)
                batch = [stage]
                batch_start = ready
        _append_if_valid(cohorts, obj, batch, batch_start, cfg, node_regions)
    return cohorts, _stats(cohorts)


def _append_if_valid(
    cohorts: list[DecodeCohort],
    obj: SharedKVObject,
    batch: list[Stage],
    planned_start: float,
    cfg: CohortConfig,
    node_regions: dict[str, str],
) -> None:
    cohort = build_decode_cohort(obj, batch, planned_start, cfg, node_regions, f"cohort_{len(cohorts)}")
    if cohort is not None:
        cohorts.append(cohort)


def build_decode_cohort(
    obj: SharedKVObject,
    batch: list[Stage],
    planned_start: float,
    cfg: CohortConfig,
    node_regions: dict[str, str],
    cohort_id: str,
) -> DecodeCohort | None:
    if len(batch) < cfg.min_group_size:
        return None
    no_cohort = sum(max(1, s.output_tokens) * obj.kv_bytes for s in batch)
    with_cohort = max(s.output_tokens for s in batch) * obj.kv_bytes
    if no_cohort - with_cohort < cfg.min_expected_saved_kv_bytes:
        return None
    node_ids = [s.parent_node_id for s in batch]
    q_regions = {n: node_regions.get(n, "") for n in node_ids}
    return DecodeCohort(
        cohort_id=cohort_id,
        shared_kv_id=obj.prefix_id,
        node_ids=node_ids,
        planned_start_ms=planned_start,
        max_wait_ms=cfg.max_wait_ms,
        shared_kv_region=obj.home_region,
        query_source_regions=q_regions,
        private_kv_regions=q_regions.copy(),
        expected_shared_kv_bytes_read=int(with_cohort),
        expected_private_kv_bytes_read=sum(max(0, s.input_tokens - s.shared_prefix_token_len) * max(1, s.output_tokens) for s in batch),
        expected_query_transfer_bytes=sum(max(1, s.output_tokens) * 256 for s in batch),
        expected_merge_bytes=sum(max(1, s.output_tokens) * 64 for s in batch),
    )


def _stats(cohorts: list[DecodeCohort]) -> dict[str, float]:
    sizes = [len(c.node_ids) for c in cohorts]
    total_nodes = sum(sizes)
    return {
        "num_decode_cohorts": float(len(cohorts)),
        "avg_cohort_size": float(sum(sizes) / len(sizes)) if sizes else 0.0,
        "p50_cohort_size": float(median(sizes)) if sizes else 0.0,
        "p90_cohort_size": float(sorted(sizes)[max(0, int(0.9 * len(sizes)) - 1)]) if sizes else 0.0,
        "cohort_wait_ms": float(sum(c.max_wait_ms for c in cohorts)),
        "cohort_wait_on_critical_path_ms": 0.0,
        "decode_nodes_cohorted_ratio": float(total_nodes / max(1, total_nodes)) if total_nodes else 0.0,
    }
