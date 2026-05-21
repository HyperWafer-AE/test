"""KVRing-v3 adaptive admission controller."""

from __future__ import annotations

from dataclasses import replace
from typing import List

from .accounting import ModeResult
from .admission import AdmissionWeights, attention_latency, select_best_candidate
from .baselines import simulate_central_kv_stationary, simulate_pull_kv_independent
from .config import Agent, HardwareConfig, ModelConfig, WorkloadConfig
from .kvring_v2 import simulate_kvring_v2
from .mesh import WaferMesh, default_agents
from .validation import result_capacity_valid


def _candidate_results(
    model: ModelConfig,
    workload: WorkloadConfig,
    hardware: HardwareConfig,
    *,
    query_tile_size: int,
    num_shards: int,
    placement: str,
    mesh: WaferMesh | None = None,
    agents: List[Agent] | None = None,
) -> list[ModeResult]:
    mesh = mesh or WaferMesh(hardware.mesh_rows, hardware.mesh_cols)
    agents = agents or default_agents(workload.concurrent_agents, hardware.mesh_rows, hardware.mesh_cols)
    return [
        simulate_pull_kv_independent(model, workload, hardware, mesh=mesh, agents=agents),
        simulate_central_kv_stationary(
            model, workload, hardware, mesh=mesh, agents=agents, query_tile_size=query_tile_size
        ),
        simulate_kvring_v2(
            model,
            workload,
            hardware,
            query_tile_size=query_tile_size,
            num_shards=num_shards,
            reduction="selected_ring",
            placement=placement,
            mesh=mesh,
            agents=agents,
        ),
        simulate_kvring_v2(
            model,
            workload,
            hardware,
            query_tile_size=query_tile_size,
            num_shards=num_shards,
            reduction="binary_tree",
            placement=placement,
            mesh=mesh,
            agents=agents,
        ),
        simulate_kvring_v2(
            model,
            workload,
            hardware,
            query_tile_size=query_tile_size,
            num_shards=num_shards,
            reduction="region_split_ring",
            placement=placement,
            mesh=mesh,
            agents=agents,
        ),
    ]


def simulate_kvring_v3_adaptive(
    model: ModelConfig,
    workload: WorkloadConfig,
    hardware: HardwareConfig,
    *,
    query_tile_size: int = 8,
    num_shards: int = 8,
    placement: str = "serpentine",
    lambda_hotspot: float = 0.0,
    mesh: WaferMesh | None = None,
    agents: List[Agent] | None = None,
) -> ModeResult:
    candidates = _candidate_results(
        model,
        workload,
        hardware,
        query_tile_size=query_tile_size,
        num_shards=num_shards,
        placement=placement,
        mesh=mesh,
        agents=agents,
    )
    weights = AdmissionWeights(lambda_hotspot=lambda_hotspot)
    selected, scored = select_best_candidate(candidates, weights=weights)
    valid_candidates = [candidate for candidate in candidates if result_capacity_valid(candidate)]
    candidate_rows = [
        {
            "mode": candidate.mode,
            "valid_capacity": result_capacity_valid(candidate),
            "attention_stage_proxy_latency_s": attention_latency(candidate),
            "max_directed_link_load_bytes": candidate.max_link_load_bytes,
            "objective": score,
            "reduction_topology": candidate.extra.get("reduction_topology", ""),
        }
        for candidate, score in scored
    ]
    reason = (
        f"selected lowest objective among {len(valid_candidates)} valid candidates"
        if result_capacity_valid(selected)
        else "all candidates invalid; selected least-bad analytical candidate"
    )
    extra = dict(selected.extra)
    extra.update(
        {
            "selected_mode": selected.mode,
            "selected_reduction_topology": selected.extra.get("reduction_topology", ""),
            "selected_query_tile_size": query_tile_size,
            "selected_num_shards": num_shards,
            "selection_reason": reason,
            "candidate_count": len(candidates),
            "valid_candidate_count": len(valid_candidates),
            "candidate_objectives": candidate_rows,
            "lambda_hotspot": lambda_hotspot,
            "latency_bound_used": selected.extra.get("latency_bound_used", "throughput_bound"),
        }
    )
    return replace(
        selected,
        mode="KVRing-v3-adaptive",
        description="Adaptive valid-capacity admission controller over pull, central, and KVRing-v2 collectives.",
        extra=extra,
    )
