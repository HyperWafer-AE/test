from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Any

from waferagent.graph_ir import AgentGraph, NodeType
from waferagent.kv_model import ModelKVConfig
from waferagent.mesh import MeshConfig
from waferagent.placement import Placement


@dataclass
class SharedKVObject:
    prefix_id: str
    token_len: int
    kv_bytes: int
    logical_users: list[str]
    decode_users: list[str]
    producer_node: str | None
    first_use_step: int
    last_use_step: int
    expected_decode_tokens: dict[str, int]
    expected_decode_steps: int
    candidate_regions: list[str]
    home_region: str | None = None
    replica_regions: list[str] = field(default_factory=list)
    residency_policy: str = "unplanned"
    reuse_distance: int = 0
    criticality_score: float = 0.0

    @property
    def expected_decode_kv_read_bytes_without_cohort(self) -> int:
        return int(self.kv_bytes * sum(max(0, t) for t in self.expected_decode_tokens.values()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "prefix_id": self.prefix_id,
            "token_len": self.token_len,
            "kv_bytes": self.kv_bytes,
            "logical_users": ",".join(self.logical_users),
            "decode_users": ",".join(self.decode_users),
            "producer_node": self.producer_node or "",
            "first_use_step": self.first_use_step,
            "last_use_step": self.last_use_step,
            "expected_decode_steps": self.expected_decode_steps,
            "candidate_regions": ",".join(self.candidate_regions),
            "home_region": self.home_region or "",
            "replica_regions": ",".join(self.replica_regions),
            "residency_policy": self.residency_policy,
            "reuse_distance": self.reuse_distance,
            "criticality_score": self.criticality_score,
            "num_decode_users": len(self.decode_users),
            "expected_decode_kv_read_bytes_without_cohort": self.expected_decode_kv_read_bytes_without_cohort,
        }


@dataclass
class SharedKVExtractionStats:
    num_shared_kv_objects: int
    logical_shared_prefix_tokens: int
    safe_shared_prefix_tokens: int
    unsafe_shared_text_tokens: int
    shared_text_not_prefix_tokens: int
    unsafe_reuse_skipped_tokens: int
    shared_kv_bytes: int
    avg_decode_users_per_object: float
    expected_decode_shared_kv_read_bytes: int

    def to_dict(self) -> dict[str, float | int]:
        return self.__dict__.copy()


def region_id_for_tile(cfg: MeshConfig, tile: tuple[int, int]) -> str:
    rr = max(0, min(cfg.rows - 1, int(tile[0]))) // max(1, int(cfg.sram_region_rows))
    cc = max(0, min(cfg.cols - 1, int(tile[1]))) // max(1, int(cfg.sram_region_cols))
    return f"r{rr}c{cc}"


def _is_safe_strict_prefix(graph: AgentGraph, node_id: str, prefix_id: str) -> bool:
    node = graph.nodes[node_id]
    if not prefix_id or node.shared_prefix_token_len <= 0:
        return False
    # The workload generator only marks strict shared prefixes in shared_prefix_ids.
    # Role/private/dependency messages are stored in private_prefix_ids or deps, not here.
    return bool(node.shared_prefix_ids and node.shared_prefix_ids[0] == prefix_id)


def extract_shared_kv_objects(
    graph: AgentGraph,
    model_cfg: ModelKVConfig | None = None,
    placements: dict[str, Placement] | None = None,
    mesh_cfg: MeshConfig | None = None,
) -> tuple[list[SharedKVObject], SharedKVExtractionStats]:
    model_cfg = model_cfg or ModelKVConfig()
    order = graph.topological_order()
    step = {node_id: i for i, node_id in enumerate(order)}
    groups: dict[str, list[str]] = {}
    unsafe = 0
    for node_id in order:
        node = graph.nodes[node_id]
        if not node.shared_prefix_ids or node.shared_prefix_token_len <= 0:
            continue
        prefix_id = node.shared_prefix_ids[0]
        if _is_safe_strict_prefix(graph, node_id, prefix_id):
            groups.setdefault(prefix_id, []).append(node_id)
        else:
            unsafe += int(node.shared_prefix_token_len)

    objects: list[SharedKVObject] = []
    for prefix_id, users in sorted(groups.items()):
        if len(users) < 2:
            continue
        token_len = min(int(graph.nodes[u].shared_prefix_token_len) for u in users)
        decode_users = [
            u
            for u in users
            if graph.nodes[u].node_type
            in {NodeType.LLM_CALL, NodeType.LLM_PREFILL, NodeType.LLM_DECODE, NodeType.AGGREGATE, NodeType.VERIFY, NodeType.SUMMARIZE}
        ]
        expected_decode = {u: int(graph.nodes[u].actual_output_token_len) for u in decode_users}
        candidate_regions: list[str] = []
        if placements and mesh_cfg:
            candidate_regions = sorted({region_id_for_tile(mesh_cfg, placements[u].tile) for u in users if u in placements})
        obj = SharedKVObject(
            prefix_id=prefix_id,
            token_len=token_len,
            kv_bytes=int(token_len * model_cfg.kv_bytes_per_token),
            logical_users=list(users),
            decode_users=decode_users,
            producer_node=None,
            first_use_step=min(step[u] for u in users),
            last_use_step=max(step[u] for u in users),
            expected_decode_tokens=expected_decode,
            expected_decode_steps=max(expected_decode.values()) if expected_decode else 0,
            candidate_regions=candidate_regions,
            reuse_distance=max(step[u] for u in users) - min(step[u] for u in users),
            criticality_score=max(float(graph.nodes[u].criticality) for u in users),
        )
        objects.append(obj)

    logical_tokens = sum(int(graph.nodes[u].shared_prefix_token_len) for users in groups.values() for u in users)
    safe_tokens = sum(o.token_len for o in objects)
    shared_bytes = sum(o.kv_bytes for o in objects)
    decode_counts = [len(o.decode_users) for o in objects]
    expected_decode_bytes = sum(o.expected_decode_kv_read_bytes_without_cohort for o in objects)
    stats = SharedKVExtractionStats(
        num_shared_kv_objects=len(objects),
        logical_shared_prefix_tokens=logical_tokens,
        safe_shared_prefix_tokens=safe_tokens,
        unsafe_shared_text_tokens=unsafe,
        shared_text_not_prefix_tokens=unsafe,
        unsafe_reuse_skipped_tokens=unsafe,
        shared_kv_bytes=shared_bytes,
        avg_decode_users_per_object=float(sum(decode_counts) / len(decode_counts)) if decode_counts else 0.0,
        expected_decode_shared_kv_read_bytes=expected_decode_bytes,
    )
    return objects, stats


def plan_shared_kv_replication(
    objects: list[SharedKVObject],
    policy: str,
    mesh_cfg: MeshConfig,
    sram_capacity_bytes_by_region: dict[str, int] | None = None,
) -> tuple[list[SharedKVObject], dict[str, float]]:
    sram_capacity_bytes_by_region = dict(sram_capacity_bytes_by_region or {})
    used: dict[str, int] = {}
    replica_bytes = 0
    saved_mesh = 0
    replication_transfer = 0
    eviction_risk = 0
    for obj in objects:
        candidates = obj.candidate_regions or ["r0c0"]
        home = candidates[0]
        if len(candidates) > 1:
            # Median-ish deterministic home: closest to the candidate list middle.
            home = candidates[len(candidates) // 2]
        obj.home_region = home
        obj.residency_policy = policy
        replicas: list[str] = []
        if policy in {"replicate_all", "oracle"}:
            replicas = [r for r in candidates if r != home]
        elif policy in {"benefit_cost", "waferagent_benefit_cost"}:
            for region in candidates:
                if region == home:
                    continue
                cap = sram_capacity_bytes_by_region.get(region, mesh_cfg.sram_region_rows * mesh_cfg.sram_region_cols * mesh_cfg.tile_sram_bytes)
                future_reads = obj.kv_bytes * max(1, len(obj.decode_users))
                copy_cost = obj.kv_bytes
                benefit = future_reads - 1.25 * copy_cost
                if benefit > 0 and used.get(region, 0) + obj.kv_bytes <= cap:
                    replicas.append(region)
                    used[region] = used.get(region, 0) + obj.kv_bytes
        elif policy == "no_replication":
            replicas = []
        obj.replica_regions = sorted(set(replicas))
        replica_bytes += obj.kv_bytes * len(obj.replica_regions)
        replication_transfer += obj.kv_bytes * len(obj.replica_regions)
        saved_mesh += obj.kv_bytes * len(obj.replica_regions) * max(1, len(obj.decode_users))
        for region in obj.replica_regions:
            cap = sram_capacity_bytes_by_region.get(region, mesh_cfg.sram_region_rows * mesh_cfg.sram_region_cols * mesh_cfg.tile_sram_bytes)
            if used.get(region, 0) > cap:
                eviction_risk += used[region] - cap
    return objects, {
        "replica_bytes_total": float(replica_bytes),
        "saved_mesh_traffic_bytes": float(saved_mesh),
        "replication_transfer_bytes": float(replication_transfer),
        "sram_pressure_bytes": float(replica_bytes),
        "eviction_count_due_to_replication": float(eviction_risk > 0),
        "reload_bytes_due_to_under_replication": float(
            sum(o.kv_bytes * max(0, len(o.candidate_regions) - 1 - len(o.replica_regions)) for o in objects)
        ),
        "num_replicas_per_shared_kv": float(median([len(o.replica_regions) for o in objects])) if objects else 0.0,
    }
