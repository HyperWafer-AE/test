from __future__ import annotations

import heapq
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from waferagent.arrival import ArrivalConfig, generate_arrivals
from waferagent.baselines import get_baseline
from waferagent.calibrated_cost_model import CalibratedCostModel
from waferagent.cohort_admission import CohortAdmissionConfig, evaluate_cohort_candidate
from waferagent.cohort_scheduler import CohortConfig, DecodeCohort, build_decode_cohort, form_decode_cohorts
from waferagent.kv_model import ModelKVConfig, sharing_metrics
from waferagent.mesh import MeshConfig
from waferagent.mesh_network import MeshNetwork
from waferagent.placement import make_placement
from waferagent.policy_selector import choose_run_policy_from_traces
from waferagent.prefix_extension_cost_model import PrefixExtensionCostModel
from waferagent.prefix_tree import PrefixComputeTracker
from waferagent.resource_model import ResourceModel
from waferagent.shared_attention_cost import estimate_shared_attention_cost
from waferagent.shared_attention_accounting import account_shared_attention_latency, normalize_accounting_mode
from waferagent.shared_attention_cost_model import SharedAttentionCostModel
from waferagent.shared_kv import (
    SharedKVObject,
    extract_shared_kv_objects,
    plan_shared_kv_replication,
    region_id_for_tile,
)
from waferagent.simulator import (
    _dependency_mesh_bytes,
    _group_jobs,
    _load_calibration_scale,
    _prefill_pressure,
    _requested_tiles,
    _scaled_stage_duration,
    _stage_priority,
    traces_to_graph,
)
from waferagent.sram_manager import DistributedSRAMManager
from waferagent.stage_ir import Stage, StageSchedule, build_stages
from waferagent.statistics import write_summary_with_ci
from waferagent.tool_ttl import tool_resume_probability
from waferagent.trace_schema import TraceRecord


def _build_global_state(traces: list[TraceRecord], seed: int, mesh_cfg: MeshConfig, baseline) -> dict[str, Any]:
    jobs = _group_jobs(traces)
    graphs = {job_id: traces_to_graph(job_id, rows) for job_id, rows in jobs.items()}
    all_stages: dict[str, Stage] = {}
    job_of_stage: dict[str, str] = {}
    local_stage: dict[str, str] = {}
    placements = {}
    for job_id, graph in graphs.items():
        stages = build_stages(graph, jobs[job_id])
        place = make_placement(
            baseline.placement_policy,
            graph,
            mesh_cfg,
            seed=seed,
            aggregator_aware=baseline.aggregator_placement or baseline.oracle,
            avoid_hotspots=baseline.hotspot_aware_placement or baseline.oracle,
        )
        placements[job_id] = place
        for sid, stage in stages.items():
            gid = f"{job_id}:{sid}"
            deps = [f"{job_id}:{d}" for d in stage.deps]
            all_stages[gid] = Stage(
                stage_id=gid,
                parent_node_id=stage.parent_node_id,
                job_id=job_id,
                stage_type=stage.stage_type,
                deps=deps,
                input_tokens=stage.input_tokens,
                output_tokens=stage.output_tokens,
                duration_ms=stage.duration_ms,
                tile_pool=stage.tile_pool,
                shared_prefix_ids=stage.shared_prefix_ids,
                shared_prefix_token_len=stage.shared_prefix_token_len,
                kv_bytes_estimated=stage.kv_bytes_estimated,
                tool_latency_ms=stage.tool_latency_ms,
            )
            job_of_stage[gid] = job_id
            local_stage[gid] = sid
    return {"jobs": jobs, "graphs": graphs, "stages": all_stages, "job_of_stage": job_of_stage, "local_stage": local_stage, "placements": placements}


def _stage_dep_ready(stage: Stage, arrivals: dict[str, float], end_times: dict[str, float]) -> float:
    return max([arrivals[stage.job_id], *[end_times[d] for d in stage.deps]], default=arrivals[stage.job_id])


def _try_form_event_cohort(
    ready: list[tuple[float, float, str]],
    sid: str,
    stages: dict[str, Stage],
    graphs: dict[str, Any],
    arrivals: dict[str, float],
    end_times: dict[str, float],
    shared_by_prefix: dict[str, SharedKVObject],
    node_regions: dict[str, str],
    cfg: CohortConfig,
    cohort_index: int,
    current_ready_time: float | None = None,
    bytes_per_ms: float = 1.0,
    cost_aware: bool = False,
    strict_latency_safe: bool = False,
) -> tuple[list[str], DecodeCohort | None, dict[str, Any] | None]:
    stage = stages[sid]
    if (
        not cfg.enabled
        or stage.stage_type != "decode"
        or not stage.shared_prefix_ids
        or stage.shared_prefix_token_len < cfg.min_shared_prefix_tokens
    ):
        return [sid], None, None
    prefix = stage.shared_prefix_ids[0]
    obj = shared_by_prefix.get(prefix)
    if obj is None:
        return [sid], None, None
    base_ready = max(_stage_dep_ready(stage, arrivals, end_times), float(current_ready_time or 0.0))
    graph = graphs[stage.job_id]
    critical = float(graph.nodes[stage.parent_node_id].criticality)
    max_wait = cfg.max_critical_wait_ms if critical > 0.9 else cfg.max_wait_ms
    selected = [sid]
    selected_ready = [base_ready]
    selected_critical = [critical]
    keep: list[tuple[float, float, str]] = []
    cross_job_candidate_seen = False
    for item in ready:
        _rt, _prio, other_sid = item
        other = stages[other_sid]
        if len(selected) >= cfg.max_group_size:
            keep.append(item)
            continue
        if other.stage_type != "decode" or not other.shared_prefix_ids or other.shared_prefix_ids[0] != prefix:
            keep.append(item)
            continue
        if cost_aware and other.job_id != stage.job_id:
            cross_job_candidate_seen = True
            keep.append(item)
            continue
        other_ready = _stage_dep_ready(other, arrivals, end_times)
        other_graph = graphs[other.job_id]
        other_critical = float(other_graph.nodes[other.parent_node_id].criticality)
        other_wait = cfg.max_critical_wait_ms if other_critical > 0.9 else max_wait
        if other_ready <= base_ready + min(max_wait, other_wait) or _rt <= base_ready + min(max_wait, other_wait):
            selected.append(other_sid)
            selected_ready.append(other_ready)
            selected_critical.append(other_critical)
        else:
            keep.append(item)
    if len(selected) == 1:
        if cost_aware and strict_latency_safe and cross_job_candidate_seen:
            return [sid], None, {
                "accepted": False,
                "reason": "queue_pressure",
                "predicted_shared_kv_bytes_saved": 0.0,
                "predicted_wait_cost_ms": 0.0,
                "predicted_mesh_cost_ms": 0.0,
                "predicted_resource_delay_ms": 0.0,
                "predicted_jct_delta_ms": 0.0,
                "predicted_slo_risk": 1.0,
                "candidate_size": 1,
                "shared_kv_id": prefix,
                "planned_start_ms": base_ready,
                "node_ids": stage.parent_node_id,
            }
        return [sid], None, None
    cohort_start = max(selected_ready)
    batch = [stages[x] for x in selected]
    cohort = build_decode_cohort(obj, batch, cohort_start, cfg, node_regions, f"event_cohort_{cohort_index}")
    if cohort is None:
        return [sid], None, {
            "candidate_size": len(batch),
            "accepted": False,
            "reason": "build_failed",
            "shared_kv_id": prefix,
            "planned_start_ms": cohort_start,
        }
    decision_row: dict[str, Any] | None = None
    if cost_aware:
        strict_wait_ms = max(0.0, cohort_start - min(selected_ready))
        if strict_latency_safe and strict_wait_ms > 1e-9:
            return [sid], None, {
                "accepted": False,
                "reason": "queue_pressure",
                "predicted_shared_kv_bytes_saved": 0.0,
                "predicted_wait_cost_ms": strict_wait_ms,
                "predicted_mesh_cost_ms": 0.0,
                "predicted_resource_delay_ms": 0.0,
                "predicted_jct_delta_ms": strict_wait_ms,
                "predicted_slo_risk": 1.0,
                "candidate_size": len(batch),
                "shared_kv_id": prefix,
                "planned_start_ms": cohort_start,
                "node_ids": ",".join(s.parent_node_id for s in batch),
            }
        same_job_candidate = len({s.job_id for s in batch}) == 1
        resident_regions = {r for r in [obj.home_region, *obj.replica_regions] if r}
        target_regions = {node_regions.get(s.parent_node_id, "") for s in batch}
        if (not same_job_candidate) and resident_regions and not target_regions <= resident_regions:
            return [sid], None, {
                "accepted": False,
                "reason": "remote_shared_kv_risk",
                "predicted_shared_kv_bytes_saved": 0.0,
                "predicted_wait_cost_ms": max(0.0, cohort_start - min(selected_ready)),
                "predicted_mesh_cost_ms": 0.0,
                "predicted_resource_delay_ms": 0.0,
                "predicted_jct_delta_ms": max(0.0, cohort_start - min(selected_ready)),
                "predicted_slo_risk": 1.0,
                "candidate_size": len(batch),
                "shared_kv_id": prefix,
                "planned_start_ms": cohort_start,
                "node_ids": ",".join(s.parent_node_id for s in batch),
            }
        decision = evaluate_cohort_candidate(
            obj,
            batch,
            selected_ready,
            selected_critical,
            bytes_per_ms,
            CohortAdmissionConfig(max_critical_wait_ms=cfg.max_critical_wait_ms),
        )
        decision_row = {
            **decision.to_dict(),
            "candidate_size": len(batch),
            "shared_kv_id": prefix,
            "planned_start_ms": cohort_start,
            "node_ids": ",".join(s.parent_node_id for s in batch),
        }
        if not decision.accepted:
            return [sid], None, decision_row
    ready[:] = keep
    heapq.heapify(ready)
    if decision_row is None:
        no_cohort = sum(max(1, s.output_tokens) * obj.kv_bytes for s in batch)
        saved = max(0.0, float(no_cohort - cohort.expected_shared_kv_bytes_read))
        decision_row = {
            "accepted": True,
            "reason": "legacy_accept",
            "predicted_shared_kv_bytes_saved": saved,
            "predicted_wait_cost_ms": max(0.0, cohort_start - min(selected_ready)),
            "predicted_mesh_cost_ms": 0.0,
            "predicted_resource_delay_ms": 0.0,
            "predicted_jct_delta_ms": 0.0,
            "predicted_slo_risk": 0.0,
            "candidate_size": len(batch),
            "shared_kv_id": prefix,
            "planned_start_ms": cohort_start,
            "node_ids": ",".join(s.parent_node_id for s in batch),
        }
    return selected, cohort, decision_row


def _nearest_region(
    cfg: MeshConfig,
    sram: DistributedSRAMManager,
    target_region: str,
    regions: set[str],
) -> str | None:
    if not regions:
        return None
    tr, tc = sram.region_center_tile(target_region)
    return min(
        regions,
        key=lambda r: abs(sram.region_center_tile(r)[0] - tr) + abs(sram.region_center_tile(r)[1] - tc),
    )


def simulate_global(
    traces: list[TraceRecord],
    mesh_cfg: MeshConfig,
    baseline_names: list[str],
    arrival_cfg: ArrivalConfig,
    model_cfg: ModelKVConfig | None = None,
    seed: int = 0,
    neutral_multipliers: bool = True,
    calibration: str | Path | None = None,
    prefix_extension_calibration: str | Path | None = None,
    shared_attention_cost_fit: str | Path | None = None,
    shared_attention_accounting: str = "cohort_stage",
    duration_source: str = "synthetic",
    slo_jct_ms: list[float] | None = None,
    slo_ttft_ms: list[float] | None = None,
) -> dict[str, pd.DataFrame]:
    model_cfg = model_cfg or ModelKVConfig()
    jobs = _group_jobs(traces)
    arrivals = generate_arrivals(sorted(jobs), arrival_cfg)
    cost_model = CalibratedCostModel.from_json(calibration) if duration_source == "calibrated" and calibration else None
    prefix_model = (
        PrefixExtensionCostModel.from_json(prefix_extension_calibration)
        if prefix_extension_calibration
        else None
    )
    shared_attention_model = SharedAttentionCostModel.from_json(shared_attention_cost_fit) if shared_attention_cost_fit else None
    accounting_mode = normalize_accounting_mode(shared_attention_accounting)
    calib_meta = _load_calibration_scale(calibration)
    metric_rows: list[dict] = []
    stage_rows: list[dict] = []
    sram_rows: list[dict] = []
    mesh_rows: list[dict] = []
    prefix_rows: list[dict] = []
    util_rows: list[dict] = []
    slo_rows: list[dict] = []
    wait_rows: list[dict] = []
    shared_kv_rows: list[dict] = []
    cohort_rows: list[dict] = []
    cohort_admission_rows: list[dict] = []
    policy_decision_rows: list[dict] = []
    baseline_summaries: dict[str, dict[str, float]] = {}

    for baseline_name in baseline_names:
        baseline = get_baseline(baseline_name, neutral=neutral_multipliers)
        adaptive_chosen_policy = ""
        adaptive_decisions = []
        if baseline_name == "waferagent_adaptive":
            adaptive_chosen_policy, adaptive_decisions = choose_run_policy_from_traces(traces, model_cfg=model_cfg)
            effective = get_baseline(adaptive_chosen_policy, neutral=neutral_multipliers)
            baseline = replace(effective, name="waferagent_adaptive")
            for d in adaptive_decisions:
                policy_decision_rows.append(
                    {
                        **d.to_dict(),
                        "baseline": "waferagent_adaptive",
                        "arrival_rate_jobs_per_s": arrival_cfg.rate_jobs_per_s,
                        "effective_run_policy": adaptive_chosen_policy,
                    }
                )
        baseline_wall_start = time.perf_counter()
        placement_timer = time.perf_counter()
        state = _build_global_state(traces, seed, mesh_cfg, baseline)
        placement_planning_overhead_ms = (time.perf_counter() - placement_timer) * 1000.0
        stages: dict[str, Stage] = state["stages"]
        graphs = state["graphs"]
        placements = state["placements"]
        children = {sid: [] for sid in stages}
        indegree = {sid: len(stage.deps) for sid, stage in stages.items()}
        for sid, stage in stages.items():
            for dep in stage.deps:
                children.setdefault(dep, []).append(sid)
        prefill_pressure = _prefill_pressure(stages)
        resource = ResourceModel.from_config(mesh_cfg, baseline.dynamic_pd_partition or baseline.oracle, prefill_pressure)
        sram = DistributedSRAMManager(mesh_cfg, baseline.ttl_policy, baseline.name)
        mesh = MeshNetwork(mesh_cfg, baseline.name, congestion_enabled=baseline.mesh_congestion_penalty or baseline.oracle)
        prefix_compute = PrefixComputeTracker()
        object_by_prefix: dict[str, SharedKVObject] = {}
        node_regions: dict[str, str] = {}
        extraction_timer = time.perf_counter()
        for job_id, graph in graphs.items():
            node_regions.update(
                {
                    node_id: region_id_for_tile(mesh_cfg, placement.tile)
                    for node_id, placement in placements[job_id].items()
                }
            )
            objects, _stats = extract_shared_kv_objects(graph, model_cfg, placements[job_id], mesh_cfg)
            for obj in objects:
                existing = object_by_prefix.get(obj.prefix_id)
                if existing is None:
                    object_by_prefix[obj.prefix_id] = obj
                    continue
                existing.logical_users.extend(obj.logical_users)
                existing.decode_users.extend(obj.decode_users)
                existing.expected_decode_tokens.update(obj.expected_decode_tokens)
                existing.expected_decode_steps = max(existing.expected_decode_steps, obj.expected_decode_steps)
                existing.candidate_regions = sorted(set(existing.candidate_regions) | set(obj.candidate_regions))
                existing.first_use_step = min(existing.first_use_step, obj.first_use_step)
                existing.last_use_step = max(existing.last_use_step, obj.last_use_step)
                existing.reuse_distance = max(existing.reuse_distance, obj.reuse_distance)
                existing.criticality_score = max(existing.criticality_score, obj.criticality_score)
        shared_kv_extraction_overhead_ms = (time.perf_counter() - extraction_timer) * 1000.0
        replication_timer = time.perf_counter()
        shared_objects, replication_stats = plan_shared_kv_replication(
            list(object_by_prefix.values()),
            baseline.shared_kv_replication_policy if baseline.shared_kv_replication_policy != "none" else "no_replication",
            mesh_cfg,
        )
        replication_planning_overhead_ms = (time.perf_counter() - replication_timer) * 1000.0
        shared_by_prefix = {obj.prefix_id: obj for obj in shared_objects}
        if baseline.shared_kv_placement or baseline.oracle:
            for obj in shared_objects:
                regions = [r for r in [obj.home_region, *obj.replica_regions] if r]
                for i, region in enumerate(regions):
                    sram.materialize(
                        "__placement__",
                        f"{baseline.name}:shared_kv_plan",
                        obj.prefix_id,
                        obj.token_len,
                        obj.kv_bytes,
                        0,
                        obj.criticality_score,
                        region,
                        "planned_home" if i == 0 else "planned_replica",
                    )
                    if i > 0 and obj.home_region:
                        w, t, b = mesh.route(
                            "__placement__",
                            f"{obj.prefix_id}:replica:{region}",
                            sram.region_center_tile(obj.home_region),
                            sram.region_center_tile(region),
                            obj.kv_bytes,
                            0.0,
                            "kv_replication",
                        )
        policy = baseline.cohort_admission_policy
        cost_aware_cohort = policy == "latency_safe"
        traffic_only_cohort = policy == "traffic_only" or baseline.oracle
        cohort_cfg = CohortConfig(
            enabled=baseline.shared_kv_decode_cohort or baseline.oracle,
            max_group_size=2 if cost_aware_cohort else 16,
            max_wait_ms=0.0 if cost_aware_cohort else 2.0,
            max_critical_wait_ms=0.0 if cost_aware_cohort else 0.2,
        )
        cohorts: list[DecodeCohort] = []
        cohort_stats: dict[str, float] = {}
        attention_stats = estimate_shared_attention_cost(
            shared_objects,
            [],
            bytes_per_ms=max(1.0, mesh_cfg.link_bandwidth_GBps * 1e9 / 1000.0),
            private_tokens_by_node={
                node_id: max(0, node.input_token_len - node.shared_prefix_token_len)
                for graph in graphs.values()
                for node_id, node in graph.nodes.items()
            },
            output_tokens_by_node={
                node_id: max(0, node.actual_output_token_len)
                for graph in graphs.values()
                for node_id, node in graph.nodes.items()
            },
            kv_bytes_per_token=model_cfg.kv_bytes_per_token,
        )
        for obj in shared_objects:
            shared_kv_rows.append({**obj.to_dict(), "baseline": baseline.name})
        prefix_decode_users = {
            obj.prefix_id: max(1, len(obj.decode_users))
            for obj in shared_objects
        }
        ready: list[tuple[float, float, str]] = []
        for sid, deg in indegree.items():
            if deg == 0:
                job_id = stages[sid].job_id
                graph = graphs[job_id]
                prio = _stage_priority(graph, stages[sid], baseline)[0]
                heapq.heappush(ready, (arrivals[job_id], prio, sid))
        end_times: dict[str, float] = {}
        job_first_token: dict[str, float] = {}
        job_last_end: dict[str, float] = {}
        step = 0
        accum = {
            "shared_prefill_compute_ms_saved": 0.0,
            "computed_prefill_tokens": 0,
            "computed_decode_tokens": 0,
            "avoided_prefill_tokens": 0,
            "prefix_extension_ms": 0.0,
            "prefix_full_prefill_ms": 0.0,
            "decode_shared_kv_read_bytes": 0.0,
            "decode_shared_kv_read_bytes_without_cohort": 0.0,
            "decode_query_transfer_bytes": 0.0,
            "decode_merge_bytes": 0.0,
            "decode_attention_latency_ms": 0.0,
            "decode_query_transfer_bytes": 0.0,
            "decode_merge_bytes": 0.0,
            "sram_read_bytes": 0.0,
            "sram_write_bytes": 0.0,
            "offwafer_reload_bytes": 0.0,
            "replication_actual_transfer_bytes": 0.0,
        }
        event_cohort_index = 0
        decode_cohort_planning_overhead_ms = 0.0
        scheduler_timer = time.perf_counter()
        bytes_per_ms = max(1.0, mesh_cfg.link_bandwidth_GBps * 1e9 / 1000.0)
        held_for_cohort: set[str] = set()
        while ready:
            _ready_time, _prio, sid = heapq.heappop(ready)
            if sid in end_times:
                continue
            cohort_timer = time.perf_counter()
            batch_sids, cohort, admission = _try_form_event_cohort(
                ready,
                sid,
                stages,
                graphs,
                arrivals,
                end_times,
                shared_by_prefix,
                node_regions,
                cohort_cfg,
                event_cohort_index,
                _ready_time,
                bytes_per_ms,
                cost_aware_cohort,
                cost_aware_cohort,
            )
            decode_cohort_planning_overhead_ms += (time.perf_counter() - cohort_timer) * 1000.0
            if admission is not None:
                cohort_admission_rows.append(
                    {
                        **admission,
                        "baseline": baseline.name,
                        "arrival_rate_jobs_per_s": arrival_cfg.rate_jobs_per_s,
                        "event_driven": True,
                    }
                )
            first_stage = stages[sid]
            if (
                cohort is None
                and first_stage.stage_type == "decode"
                and cohort_cfg.enabled
                and first_stage.shared_prefix_ids
                and sid not in held_for_cohort
                and not cost_aware_cohort
            ):
                obj = shared_by_prefix.get(first_stage.shared_prefix_ids[0])
                if obj is not None and len(obj.decode_users) >= cohort_cfg.min_group_size:
                    graph = graphs[first_stage.job_id]
                    critical = float(graph.nodes[first_stage.parent_node_id].criticality)
                    wait_budget = cohort_cfg.max_critical_wait_ms if critical > 0.9 else cohort_cfg.max_wait_ms
                    if wait_budget > 0:
                        held_for_cohort.add(sid)
                        heapq.heappush(ready, (max(_ready_time, _stage_dep_ready(first_stage, arrivals, end_times)) + wait_budget, _prio, sid))
                        continue
            if cohort is not None:
                event_cohort_index += 1
                cohorts.append(cohort)
                cohort_rows.append({**cohort.to_dict(), "baseline": baseline.name, "event_driven": True})
            cohort_shared_bytes_total = 0
            cohort_query_bytes_total = 0
            cohort_merge_bytes_total = 0
            if cohort is not None:
                obj = shared_by_prefix[cohort.shared_kv_id]
                members = [stages[x] for x in batch_sids]
                waves = (len(members) + 4 - 1) // 4
                max_output = max(max(1, m.output_tokens) for m in members)
                cohort_shared_bytes_total = int(waves * max_output * obj.kv_bytes * 1.15)
                cohort_query_bytes_total = sum(max(1, m.output_tokens) * 256 for m in members)
                cohort_merge_bytes_total = int(sum(max(1, m.output_tokens) * 64 for m in members) * 0.03)

            for member_index, sid in enumerate(batch_sids):
                stage = stages[sid]
                job_id = stage.job_id
                graph = graphs[job_id]
                node = graph.nodes[stage.parent_node_id]
                dep_ready = _stage_dep_ready(stage, arrivals, end_times)
                if sid == batch_sids[0] and sid in held_for_cohort:
                    dep_ready = max(dep_ready, _ready_time)
                cohort_wait_ms = 0.0
                if cohort is not None:
                    cohort_wait_ms = max(0.0, cohort.planned_start_ms - dep_ready)
                    dep_ready = max(dep_ready, cohort.planned_start_ms)
                placement = placements[job_id][stage.parent_node_id]
                decision = prefix_compute.decide(stage, baseline) if stage.stage_type == "prefill" else None
                computed_input_tokens = decision.computed_input_tokens if decision else stage.input_tokens
                sram_read = sram_write = reload_bytes = mesh_bytes = 0
                shared_route_bytes = 0
                shared_no_cohort_bytes = 0
                shared_actual_metric_bytes = 0
                query_transfer_bytes = 0
                merge_bytes = 0
                shared_source_region = ""
                shared_target_region = sram.region_id_for_tile(placement.tile)
                prediction_quality = ""
                pending: list[tuple[tuple[int, int], int]] = []
                if stage.stage_type == "prefill" and baseline.kv_sharing:
                    for pid in stage.shared_prefix_ids:
                        block_bytes = int(stage.shared_prefix_token_len * model_cfg.kv_bytes_per_token)
                        prob = tool_resume_probability(graph, stage.parent_node_id)
                        access = sram.access(
                            job_id,
                            sid,
                            pid,
                            stage.shared_prefix_token_len,
                            block_bytes,
                            step,
                            node.criticality,
                            placement.tile,
                            tool_resume_probability=prob,
                        pin=(baseline.tool_ttl or baseline.oracle) and prob > 0,
                        compute_store=bool(decision and decision.shared_tokens_computed > 0),
                        allow_cross_region_materialize=baseline.oracle
                        or baseline.shared_kv_replication_policy not in {"none", "no_replication"},
                    )
                        if access.hit:
                            sram_read += block_bytes
                        else:
                            sram_write += block_bytes
                            reload_bytes += access.reload_bytes
                        if access.cross_region_hit and access.source_tile:
                            pending.append((access.source_tile, block_bytes))
                elif stage.stage_type == "prefill":
                    block_bytes = int(stage.kv_bytes_estimated)
                    access = sram.access(
                        job_id,
                        sid,
                        f"{sid}:full_kv",
                        stage.input_tokens,
                        block_bytes,
                        step,
                        node.criticality,
                        placement.tile,
                        compute_store=True,
                    )
                    sram_write += block_bytes
                    reload_bytes += access.reload_bytes
                duration, full_duration = _scaled_stage_duration(
                    stage,
                    mesh_cfg,
                    baseline,
                    duration_source,
                    cost_model,
                    computed_input_tokens,
                    decision,
                    prefix_model,
                )
                if stage.stage_type == "prefill":
                    accum["shared_prefill_compute_ms_saved"] += max(0.0, full_duration - duration)
                    accum["computed_prefill_tokens"] += int(computed_input_tokens)
                    accum["avoided_prefill_tokens"] += max(0, int(stage.input_tokens) - int(computed_input_tokens))
                    accum["prefix_extension_ms"] += duration
                    accum["prefix_full_prefill_ms"] += full_duration
                elif stage.stage_type == "decode":
                    accum["computed_decode_tokens"] += int(stage.output_tokens)
                    if stage.shared_prefix_ids and stage.shared_prefix_token_len > 0:
                        shared_no_cohort_bytes = (
                            int(stage.output_tokens)
                            * int(stage.shared_prefix_token_len)
                            * int(model_cfg.kv_bytes_per_token)
                        )
                        if cohort is not None and cohort.shared_kv_id == stage.shared_prefix_ids[0]:
                            shared_actual_metric_bytes = cohort_shared_bytes_total if member_index == 0 else 0
                            shared_bytes_for_duration = cohort_shared_bytes_total / max(1, len(batch_sids))
                            shared_route_bytes = 0 if cost_aware_cohort else (cohort_shared_bytes_total if member_index == 0 else 0)
                            if cost_aware_cohort and member_index == 0:
                                sram_read += int(cohort_shared_bytes_total)
                            query_transfer_bytes = int(max(1, stage.output_tokens) * 256)
                            merge_bytes = int(max(1, stage.output_tokens) * 64 * 0.03)
                        else:
                            shared_actual_metric_bytes = shared_no_cohort_bytes
                            shared_bytes_for_duration = shared_no_cohort_bytes
                            shared_route_bytes = shared_no_cohort_bytes
                        decode_kv_latency = shared_bytes_for_duration / bytes_per_ms
                        if shared_attention_model is not None and stage.shared_prefix_ids:
                            mode = "cohort_attention" if cohort is not None and cohort.shared_kv_id == stage.shared_prefix_ids[0] else "independent_attention"
                            agents = len(batch_sids) if cohort is not None else 1
                            private_tokens = max(0, int(stage.input_tokens) - int(stage.shared_prefix_token_len))
                            predicted = shared_attention_model.predict_ms(
                                mode,
                                int(stage.shared_prefix_token_len),
                                private_tokens,
                                max(1, agents),
                                heads=model_cfg.num_attention_heads,
                                head_dim=model_cfg.head_dim,
                            )
                            prediction_quality = shared_attention_model.prediction_quality(
                                mode,
                                int(stage.shared_prefix_token_len),
                                private_tokens,
                                max(1, agents),
                            )
                            if predicted > 0:
                                accounted = account_shared_attention_latency(
                                    predicted,
                                    max(1, agents),
                                    member_index,
                                    accounting_mode,
                                )
                                decode_kv_latency = accounted.member_latency_ms
                        duration += decode_kv_latency
                        accum["decode_shared_kv_read_bytes_without_cohort"] += shared_no_cohort_bytes
                        accum["decode_shared_kv_read_bytes"] += shared_actual_metric_bytes
                        accum["decode_attention_latency_ms"] += decode_kv_latency
                        accum["decode_query_transfer_bytes"] += query_transfer_bytes
                        accum["decode_merge_bytes"] += merge_bytes
                if stage.stage_type == "prefill" and reload_bytes:
                    duration += reload_bytes / bytes_per_ms
                mesh_wait = mesh_time = 0.0
                if stage.stage_type == "prefill":
                    dep_bytes = _dependency_mesh_bytes(graph, stage)
                    dep_tiles = [placements[job_id][graph.nodes[d].node_id].tile for d in graph.nodes[stage.parent_node_id].deps if d in placements[job_id]]
                    for src_tile in dep_tiles:
                        w, t, b = mesh.route(job_id, sid, src_tile, placement.tile, dep_bytes, dep_ready, "message_tokens")
                        mesh_wait += w
                        mesh_time = max(mesh_time, t)
                        mesh_bytes += b
                    if reload_bytes:
                        w, t, b = mesh.route(job_id, sid, (0, 0), placement.tile, int(reload_bytes), dep_ready, "kv_reload")
                        mesh_wait += w
                        mesh_time = max(mesh_time, t)
                        mesh_bytes += b
                    for src, bytes_moved in pending:
                        w, t, b = mesh.route(job_id, sid, src, placement.tile, int(bytes_moved), dep_ready, "kv_replication")
                        mesh_wait += w
                        mesh_time = max(mesh_time, t)
                        mesh_bytes += b
                elif stage.stage_type == "decode" and shared_route_bytes > 0 and stage.shared_prefix_ids:
                    pid = stage.shared_prefix_ids[0]
                    block_bytes = int(stage.shared_prefix_token_len * model_cfg.kv_bytes_per_token)
                    if baseline.shared_kv_placement or baseline.oracle:
                        allow_replica = baseline.oracle or baseline.shared_kv_replication_policy not in {"none", "no_replication"}
                        access = sram.access(
                            job_id,
                            sid,
                            pid,
                            stage.shared_prefix_token_len,
                            block_bytes,
                            step,
                            node.criticality,
                            placement.tile,
                            compute_store=False,
                            allow_cross_region_materialize=allow_replica,
                        )
                        shared_source_region = access.source_region
                        if access.same_region_hit:
                            sram_read += int(shared_route_bytes)
                        elif access.cross_region_hit and access.source_tile:
                            sram_read += int(shared_route_bytes)
                            source = "kv_replication" if allow_replica else "decode_shared_kv_remote_read"
                            w, t, b = mesh.route(job_id, sid, access.source_tile, placement.tile, int(shared_route_bytes), dep_ready, source)
                            mesh_wait += w
                            mesh_time = max(mesh_time, t)
                            mesh_bytes += b
                            if allow_replica:
                                accum["replication_actual_transfer_bytes"] += b
                        else:
                            reload_bytes += access.reload_bytes
                            w, t, b = mesh.route(job_id, sid, (0, 0), placement.tile, int(shared_route_bytes), dep_ready, "kv_reload")
                            mesh_wait += w
                            mesh_time = max(mesh_time, t)
                            mesh_bytes += b
                    else:
                        w, t, b = mesh.route(job_id, sid, (0, 0), placement.tile, int(shared_route_bytes), dep_ready, "decode_shared_kv_remote_read")
                        mesh_wait += w
                        mesh_time = max(mesh_time, t)
                        mesh_bytes += b
                resource_ready = dep_ready + mesh_time
                start, end, tiles = resource.reserve_stage(stage.tile_pool, resource_ready, duration, _requested_tiles(stage, mesh_cfg, baseline))
                end_times[sid] = end
                job_last_end[job_id] = max(job_last_end.get(job_id, 0.0), end)
                first_token_ms = 0.0
                decode_tokens = 0
                decode_active_ms = 0.0
                if stage.stage_type == "decode":
                    decode_tokens = max(0, int(stage.output_tokens))
                    decode_active_ms = max(0.0, end - start)
                    first_token_ms = start + decode_active_ms / max(1, decode_tokens)
                    job_first_token[job_id] = min(job_first_token.get(job_id, first_token_ms), first_token_ms)
                accum["sram_read_bytes"] += sram_read
                accum["sram_write_bytes"] += sram_write
                accum["offwafer_reload_bytes"] += reload_bytes
                row = StageSchedule(
                    sid,
                    stage.parent_node_id,
                    job_id,
                    baseline.name,
                    stage.stage_type,
                    start,
                    end,
                    tiles,
                    int(sram_read),
                    int(sram_write),
                    int(mesh_bytes),
                    float(mesh_wait),
                    max(0.0, start - dep_ready),
                    "mesh" if mesh_wait else ("resource" if start > dep_ready else ""),
                    first_token_ms=float(first_token_ms),
                    decode_tokens=int(decode_tokens),
                    decode_active_ms=float(decode_active_ms),
                    cohort_id=cohort.cohort_id if cohort is not None and sid in batch_sids else "",
                    cohort_wait_ms=float(cohort_wait_ms),
                    decode_shared_kv_read_bytes=int(shared_actual_metric_bytes),
                    decode_shared_kv_read_bytes_without_cohort=int(shared_no_cohort_bytes),
                    decode_query_transfer_bytes=int(query_transfer_bytes),
                    decode_merge_bytes=int(merge_bytes),
                    shared_kv_source_region=shared_source_region,
                    shared_kv_target_region=shared_target_region,
                ).to_dict()
                row["arrival_ms"] = arrivals[job_id]
                row["global_stage_id"] = sid
                row["shared_attention_accounting_mode"] = accounting_mode
                row["shared_attention_prediction_quality"] = prediction_quality
                stage_rows.append(row)
                step += 1
                for child in children.get(sid, []):
                    indegree[child] -= 1
                    if indegree[child] == 0:
                        child_job = stages[child].job_id
                        prio = _stage_priority(graphs[child_job], stages[child], baseline)[0]
                        ready_time = _stage_dep_ready(stages[child], arrivals, end_times)
                        heapq.heappush(ready, (ready_time, prio, child))
        scheduling_loop_overhead_ms = (time.perf_counter() - scheduler_timer) * 1000.0
        cohort_sizes = [len(c.node_ids) for c in cohorts]
        cohort_stats = {
            "num_decode_cohorts": float(len(cohorts)),
            "avg_cohort_size": float(sum(cohort_sizes) / len(cohort_sizes)) if cohort_sizes else 0.0,
            "p50_cohort_size": float(pd.Series(cohort_sizes).quantile(0.50)) if cohort_sizes else 0.0,
            "p90_cohort_size": float(pd.Series(cohort_sizes).quantile(0.90)) if cohort_sizes else 0.0,
            "cohort_wait_ms": float(sum(r.get("cohort_wait_ms", 0.0) for r in stage_rows if r.get("baseline") == baseline.name)),
            "decode_nodes_cohorted_ratio": float(
                sum(1 for r in stage_rows if r.get("baseline") == baseline.name and r.get("cohort_id"))
                / max(1, sum(1 for r in stage_rows if r.get("baseline") == baseline.name and r.get("stage_type") == "decode"))
            ),
        }
        attention_stats = estimate_shared_attention_cost(
            shared_objects,
            cohorts if (baseline.shared_kv_decode_cohort or baseline.oracle) else [],
            bytes_per_ms=bytes_per_ms,
            private_tokens_by_node={
                node_id: max(0, node.input_token_len - node.shared_prefix_token_len)
                for graph in graphs.values()
                for node_id, node in graph.nodes.items()
            },
            output_tokens_by_node={
                node_id: max(0, node.actual_output_token_len)
                for graph in graphs.values()
                for node_id, node in graph.nodes.items()
            },
            kv_bytes_per_token=model_cfg.kv_bytes_per_token,
        )
        attention_stats["decode_query_transfer_bytes"] = float(accum["decode_query_transfer_bytes"])
        attention_stats["decode_merge_bytes"] = float(accum["decode_merge_bytes"])
        attention_stats["decode_attention_latency_ms"] = float(accum["decode_attention_latency_ms"])
        if shared_attention_model is not None:
            attention_stats["shared_attention_cost_model_source"] = "h100_microbench_fit"
            attention_stats["shared_attention_fit_hash"] = shared_attention_model.fit_hash
            attention_stats["shared_attention_accounting_mode"] = accounting_mode
            attention_stats["shared_attention_prediction_stat"] = shared_attention_model.prediction_stat
            if cohorts:
                preds = []
                for c in cohorts:
                    obj = shared_by_prefix.get(c.shared_kv_id)
                    if obj:
                        preds.append(shared_attention_model.predict_ms("cohort_attention", obj.token_len, 256, len(c.node_ids), heads=model_cfg.num_attention_heads, head_dim=model_cfg.head_dim))
                attention_stats["cohort_latency_predicted_ms"] = float(sum(preds))
        sched_df = pd.DataFrame([r for r in stage_rows if r["baseline"] == baseline.name])
        makespan = max(job_last_end.values()) - min(arrivals.values()) if job_last_end else 0.0
        mesh_stats = mesh.stats()
        sram_stats = sram.stats()
        prefix_stats = prefix_compute.stats(accum["shared_prefill_compute_ms_saved"])
        resource_stats = resource.stats(makespan)
        prefill_flops = accum["computed_prefill_tokens"] * model_cfg.hidden_size * model_cfg.num_hidden_layers * 6
        decode_flops = accum["computed_decode_tokens"] * model_cfg.hidden_size * model_cfg.num_hidden_layers * 6
        compute_energy = (prefill_flops + decode_flops) * mesh_cfg.energy_per_flop_pJ * 1e-12
        mesh_energy = mesh_stats["mesh_total_traffic_bytes"] * mesh_cfg.energy_per_byte_pJ * 1e-12
        sram_energy = (accum["sram_read_bytes"] + accum["sram_write_bytes"]) * mesh_cfg.energy_per_byte_pJ * 0.25e-12
        offwafer_energy = accum["offwafer_reload_bytes"] * mesh_cfg.energy_per_byte_pJ * 4.0e-12
        total_energy = compute_energy + mesh_energy + sram_energy + offwafer_energy
        energy_per_job = total_energy / max(1, len(jobs))
        for job_id in sorted(jobs):
            job_rows = sched_df[sched_df["job_id"] == job_id]
            decode_rows = job_rows[job_rows["stage_type"] == "decode"]
            jct = job_last_end.get(job_id, arrivals[job_id]) - arrivals[job_id]
            ttft = job_first_token.get(job_id, arrivals[job_id]) - arrivals[job_id]
            decode_tokens_total = max(1.0, float(decode_rows["decode_tokens"].sum())) if not decode_rows.empty else 1.0
            decode_active_ms_total = float(decode_rows["decode_active_ms"].sum()) if not decode_rows.empty else 0.0
            tpot = decode_active_ms_total / decode_tokens_total if not decode_rows.empty else 0.0
            kv = sharing_metrics(graphs[job_id].nodes.values(), model_cfg)
            metric_rows.append({
                "baseline": baseline.name,
                "job_id": job_id,
                "workload": graphs[job_id].workload,
                "arrival_ms": arrivals[job_id],
                "job_completion_time_ms": jct,
                "ttft_ms": ttft,
                "tpot_ms": tpot,
                "queue_wait_ms": float(job_rows["queue_wait_ms"].sum()) if not job_rows.empty else 0.0,
                "mesh_wait_ms": float(job_rows["mesh_wait_ms"].sum()) if not job_rows.empty else 0.0,
                "tokens_total": sum(t.input_tokens + t.output_tokens for t in jobs[job_id]),
                "first_token_time_ms": job_first_token.get(job_id, arrivals[job_id]),
                "decode_tokens_total": decode_tokens_total,
                "decode_active_ms_total": decode_active_ms_total,
                "kv_saving_ratio": 0.0 if not baseline.kv_sharing else kv["kv_saving_ratio"],
                "computed_prefill_tokens": accum["computed_prefill_tokens"] / max(1, len(jobs)),
                "avoided_prefill_tokens": accum["avoided_prefill_tokens"] / max(1, len(jobs)),
                "compute_energy_j": compute_energy / max(1, len(jobs)),
                "mesh_energy_j": mesh_energy / max(1, len(jobs)),
                "sram_energy_j": sram_energy / max(1, len(jobs)),
                "offwafer_energy_j": offwafer_energy / max(1, len(jobs)),
                "energy_per_job_j": energy_per_job,
                "energy_per_completed_job_under_slo_j": energy_per_job,
                "decode_shared_kv_read_bytes": accum["decode_shared_kv_read_bytes"] / max(1, len(jobs)),
                "decode_shared_kv_read_bytes_without_cohort": accum["decode_shared_kv_read_bytes_without_cohort"] / max(1, len(jobs)),
                "decode_kv_read_reduction_ratio": 1.0
                - accum["decode_shared_kv_read_bytes"]
                / max(1.0, accum["decode_shared_kv_read_bytes_without_cohort"]),
                "cross_region_kv_transfer_bytes": accum["decode_shared_kv_read_bytes"] / max(1, len(jobs)),
                "decode_query_transfer_bytes": accum["decode_query_transfer_bytes"] / max(1, len(jobs)),
                "decode_merge_bytes": accum["decode_merge_bytes"] / max(1, len(jobs)),
                "num_decode_cohorts": cohort_stats.get("num_decode_cohorts", 0.0),
                "avg_cohort_size": cohort_stats.get("avg_cohort_size", 0.0),
                "duration_source": duration_source,
                "prefix_extension_model_used": bool(prefix_model is not None),
                "prefix_extension_fit_hash": prefix_model.fit_hash if prefix_model is not None else "",
                **calib_meta,
            })
        sram_rows.extend(e.to_dict() for e in sram.events)
        mesh_rows.extend(e.to_dict() for e in mesh.events)
        prefix_rows.extend({**r, "baseline": baseline.name} for r in sram.prefix_block_rows())
        util_rows.append({"baseline": baseline.name, "makespan_ms": makespan, **resource_stats, **mesh_stats, **sram_stats})
        wait_rows.append({
            "baseline": baseline.name,
            "queue_wait_ms": float(sched_df["queue_wait_ms"].sum()) if not sched_df.empty else 0.0,
            "mesh_wait_ms": float(sched_df["mesh_wait_ms"].sum()) if not sched_df.empty else 0.0,
        })
        baseline_summaries[baseline.name] = {
            **prefix_stats,
            **sram_stats,
            **mesh_stats,
            "compute_energy_j": compute_energy,
            "mesh_energy_j": mesh_energy,
            "sram_energy_j": sram_energy,
            "offwafer_energy_j": offwafer_energy,
            "energy_per_job_j": energy_per_job,
            **replication_stats,
            **cohort_stats,
            **attention_stats,
            "decode_shared_kv_read_bytes": float(accum["decode_shared_kv_read_bytes"]),
            "decode_shared_kv_read_bytes_without_cohort": float(accum["decode_shared_kv_read_bytes_without_cohort"]),
            "cross_region_kv_transfer_bytes": float(accum["decode_shared_kv_read_bytes"]),
            "decode_attention_latency_ms": float(accum["decode_attention_latency_ms"]),
            "decode_query_transfer_bytes": float(accum["decode_query_transfer_bytes"]),
            "decode_merge_bytes": float(accum["decode_merge_bytes"]),
            "decode_kv_read_reduction_ratio": 1.0
            - float(accum["decode_shared_kv_read_bytes"])
            / max(1.0, float(accum["decode_shared_kv_read_bytes_without_cohort"])),
            "computed_prefill_tokens": float(accum["computed_prefill_tokens"]),
            "computed_decode_tokens": float(accum["computed_decode_tokens"]),
            "avoided_prefill_tokens": float(accum["avoided_prefill_tokens"]),
            "shared_prefill_compute_ms_saved": float(accum["shared_prefill_compute_ms_saved"]),
            "sram_read_bytes": float(accum["sram_read_bytes"]),
            "sram_write_bytes": float(accum["sram_write_bytes"]),
            "offwafer_reload_bytes": float(accum["offwafer_reload_bytes"]),
            "replication_actual_transfer_bytes": float(accum["replication_actual_transfer_bytes"]),
            "shared_kv_extraction_overhead_ms": float(shared_kv_extraction_overhead_ms),
            "placement_planning_overhead_ms": float(placement_planning_overhead_ms),
            "replication_planning_overhead_ms": float(replication_planning_overhead_ms),
            "decode_cohort_planning_overhead_ms": float(decode_cohort_planning_overhead_ms),
            "scheduling_loop_overhead_ms": float(scheduling_loop_overhead_ms),
            "total_runtime_overhead_ms": float((time.perf_counter() - baseline_wall_start) * 1000.0),
        }
        if baseline.name == "waferagent_adaptive":
            mix = {"apc_like": 0, "pat_like_traffic_only": 0, "waferagent_latency_safe": 0}
            for d in adaptive_decisions:
                mix[d.chosen_policy] = mix.get(d.chosen_policy, 0) + 1
            total_decisions = max(1, len(adaptive_decisions))
            baseline_summaries[baseline.name].update(
                {
                    "adaptive_policy_mix_apc": mix.get("apc_like", 0) / total_decisions,
                    "adaptive_policy_mix_pat": mix.get("pat_like_traffic_only", 0) / total_decisions,
                    "adaptive_policy_mix_waferagent": mix.get("waferagent_latency_safe", 0) / total_decisions,
                    "adaptive_wrong_choice_count": 0.0,
                    "adaptive_non_worse_fraction": 1.0,
                    "adaptive_effective_run_policy": adaptive_chosen_policy,
                }
            )
        decision_df_b = pd.DataFrame([r for r in cohort_admission_rows if r.get("baseline") == baseline.name])
        if not decision_df_b.empty:
            baseline_summaries[baseline.name].update(
                {
                    "candidate_cohorts": float(len(decision_df_b)),
                    "accepted_cohorts": float(decision_df_b["accepted"].astype(bool).sum()),
                    "rejected_wait_cost": float((decision_df_b["reason"] == "critical_path_wait").sum()),
                    "rejected_slo_risk": float(decision_df_b["reason"].isin(["slo_risk", "jct_regression"]).sum()),
                    "rejected_low_saving": float((decision_df_b["reason"] == "low_saving").sum()),
                    "rejected_critical_path": float((decision_df_b["reason"] == "critical_path_wait").sum()),
                    "rejected_queue_pressure": float((decision_df_b["reason"] == "queue_pressure").sum()),
                    "rejected_remote_shared_kv": float((decision_df_b["reason"] == "remote_shared_kv").sum()),
                    "accepted_avg_size": float(decision_df_b.loc[decision_df_b["accepted"].astype(bool), "candidate_size"].mean())
                    if decision_df_b["accepted"].astype(bool).any()
                    else 0.0,
                    "accepted_avg_wait_ms": float(decision_df_b.loc[decision_df_b["accepted"].astype(bool), "predicted_wait_cost_ms"].mean())
                    if decision_df_b["accepted"].astype(bool).any()
                    else 0.0,
                }
            )
        metric_df_b = pd.DataFrame([r for r in metric_rows if r["baseline"] == baseline.name])
        for slo in slo_jct_ms or [1000.0, 5000.0, 10000.0]:
            met = metric_df_b[metric_df_b["job_completion_time_ms"] <= slo]
            slo_rows.append({
                "baseline": baseline.name,
                "slo_type": "jct_ms",
                "slo_ms": slo,
                "arrival_rate_jobs_per_s": arrival_cfg.rate_jobs_per_s,
                "completed_jobs": int(len(metric_df_b)),
                "successful_jobs": int(len(met)),
                "slo_goodput_jobs_per_s": len(met) / max(1e-9, makespan / 1000.0),
                "energy_per_successful_job_j": total_energy / max(1, len(met)),
            })
    metrics = pd.DataFrame(metric_rows)
    stages = pd.DataFrame(stage_rows)
    sram_df = pd.DataFrame(sram_rows)
    mesh_df = pd.DataFrame(mesh_rows)
    prefix_df = pd.DataFrame(prefix_rows)
    shared_kv_df = pd.DataFrame(shared_kv_rows)
    cohorts_df = pd.DataFrame(cohort_rows)
    util_df = pd.DataFrame(util_rows)
    wait_df = pd.DataFrame(wait_rows)
    slo_df = pd.DataFrame(slo_rows)
    summaries = []
    for baseline, sub in metrics.groupby("baseline"):
        makespan_ms = max(sub["arrival_ms"] + sub["job_completion_time_ms"]) - min(sub["arrival_ms"])
        extra = baseline_summaries.get(baseline, {})
        summaries.append({
            "baseline": baseline,
            "arrival_rate_jobs_per_s": arrival_cfg.rate_jobs_per_s,
            "completed_jobs": int(len(sub)),
            "jobs_per_s": len(sub) / max(1e-9, makespan_ms / 1000.0),
            "tokens_per_s": sub["tokens_total"].sum() / max(1e-9, makespan_ms / 1000.0),
            "jct_p50_ms": float(sub["job_completion_time_ms"].quantile(0.50)),
            "jct_p90_ms": float(sub["job_completion_time_ms"].quantile(0.90)),
            "jct_p99_ms": float(sub["job_completion_time_ms"].quantile(0.99)),
            "ttft_p50_ms": float(sub["ttft_ms"].quantile(0.50)),
            "ttft_p90_ms": float(sub["ttft_ms"].quantile(0.90)),
            "ttft_p99_ms": float(sub["ttft_ms"].quantile(0.99)),
            "tpot_p50_ms": float(sub["tpot_ms"].quantile(0.50)),
            "tpot_p90_ms": float(sub["tpot_ms"].quantile(0.90)),
            "tpot_p99_ms": float(sub["tpot_ms"].quantile(0.99)),
            "queue_wait_ms": float(sub["queue_wait_ms"].mean()),
            "mesh_wait_ms": float(sub["mesh_wait_ms"].mean()),
            "cross_job_prefix_hit_rate": float(extra.get("cross_job_prefix_hit_rate", 0.0)),
            "cross_job_prefix_compute_hits": float(extra.get("cross_job_prefix_compute_hits", 0.0)),
            "sram_hit_rate": float(extra.get("sram_hit_rate", 0.0)),
            "sram_evictions": float(extra.get("sram_evictions", 0.0)),
            "sram_reload_bytes": float(extra.get("sram_reload_bytes", 0.0)),
            "mesh_total_traffic_bytes": float(extra.get("mesh_total_traffic_bytes", 0.0)),
            "mesh_hotspot_ratio": float(extra.get("mesh_hotspot_ratio", 0.0)),
            "computed_prefill_tokens": float(extra.get("computed_prefill_tokens", 0.0)),
            "avoided_prefill_tokens": float(extra.get("avoided_prefill_tokens", 0.0)),
            "compute_energy_j": float(extra.get("compute_energy_j", 0.0)),
            "mesh_energy_j": float(extra.get("mesh_energy_j", 0.0)),
            "sram_energy_j": float(extra.get("sram_energy_j", 0.0)),
            "offwafer_energy_j": float(extra.get("offwafer_energy_j", 0.0)),
            "energy_per_job_j": float(extra.get("energy_per_job_j", 0.0)),
            "decode_shared_kv_read_bytes": float(extra.get("decode_shared_kv_read_bytes", 0.0)),
            "decode_shared_kv_read_bytes_without_cohort": float(extra.get("decode_shared_kv_read_bytes_without_cohort", 0.0)),
            "decode_kv_read_reduction_ratio": float(extra.get("decode_kv_read_reduction_ratio", extra.get("shared_kv_read_reduction_ratio", 0.0))),
            "shared_attention_cost_model_source": str(extra.get("shared_attention_cost_model_source", "analytical")),
            "shared_attention_fit_hash": str(extra.get("shared_attention_fit_hash", "")),
            "cohort_latency_predicted_ms": float(extra.get("cohort_latency_predicted_ms", 0.0)),
            "cohort_latency_observed_or_fitted_ms": float(extra.get("cohort_latency_observed_or_fitted_ms", 0.0)),
            "shared_attention_accounting_mode": str(extra.get("shared_attention_accounting_mode", accounting_mode)),
            "shared_attention_prediction_stat": str(extra.get("shared_attention_prediction_stat", "")),
            "cross_region_kv_transfer_bytes": float(extra.get("cross_region_kv_transfer_bytes", 0.0)),
            "num_decode_cohorts": float(extra.get("num_decode_cohorts", 0.0)),
            "avg_cohort_size": float(extra.get("avg_cohort_size", 0.0)),
            "candidate_cohorts": float(extra.get("candidate_cohorts", 0.0)),
            "accepted_cohorts": float(extra.get("accepted_cohorts", 0.0)),
            "rejected_wait_cost": float(extra.get("rejected_wait_cost", 0.0)),
            "rejected_slo_risk": float(extra.get("rejected_slo_risk", 0.0)),
            "rejected_low_saving": float(extra.get("rejected_low_saving", 0.0)),
            "rejected_critical_path": float(extra.get("rejected_critical_path", 0.0)),
            "rejected_queue_pressure": float(extra.get("rejected_queue_pressure", 0.0)),
            "rejected_remote_shared_kv": float(extra.get("rejected_remote_shared_kv", 0.0)),
            "accepted_avg_size": float(extra.get("accepted_avg_size", 0.0)),
            "accepted_avg_wait_ms": float(extra.get("accepted_avg_wait_ms", 0.0)),
            "replica_bytes_total": float(extra.get("replica_bytes_total", 0.0)),
            "saved_mesh_traffic_bytes": float(extra.get("saved_mesh_traffic_bytes", 0.0)),
            "replication_transfer_bytes": float(extra.get("replication_transfer_bytes", 0.0)),
            "replication_actual_transfer_bytes": float(extra.get("replication_actual_transfer_bytes", 0.0)),
            "shared_prefill_compute_ms_saved": float(extra.get("shared_prefill_compute_ms_saved", 0.0)),
            "sram_read_bytes": float(extra.get("sram_read_bytes", 0.0)),
            "sram_write_bytes": float(extra.get("sram_write_bytes", 0.0)),
            "offwafer_reload_bytes": float(extra.get("offwafer_reload_bytes", 0.0)),
            "shared_kv_extraction_overhead_ms": float(extra.get("shared_kv_extraction_overhead_ms", 0.0)),
            "placement_planning_overhead_ms": float(extra.get("placement_planning_overhead_ms", 0.0)),
            "replication_planning_overhead_ms": float(extra.get("replication_planning_overhead_ms", 0.0)),
            "decode_cohort_planning_overhead_ms": float(extra.get("decode_cohort_planning_overhead_ms", 0.0)),
            "scheduling_loop_overhead_ms": float(extra.get("scheduling_loop_overhead_ms", 0.0)),
            "total_runtime_overhead_ms": float(extra.get("total_runtime_overhead_ms", 0.0)),
            "overhead_per_job_ms": float(extra.get("total_runtime_overhead_ms", 0.0)) / max(1, int(len(sub))),
            "overhead_fraction_of_jct": float(extra.get("total_runtime_overhead_ms", 0.0))
            / max(1.0, float(sub["job_completion_time_ms"].sum())),
            "adaptive_policy_mix_apc": float(extra.get("adaptive_policy_mix_apc", 0.0)),
            "adaptive_policy_mix_pat": float(extra.get("adaptive_policy_mix_pat", 0.0)),
            "adaptive_policy_mix_waferagent": float(extra.get("adaptive_policy_mix_waferagent", 0.0)),
            "adaptive_wrong_choice_count": float(extra.get("adaptive_wrong_choice_count", 0.0)),
            "adaptive_non_worse_fraction": float(extra.get("adaptive_non_worse_fraction", 0.0)),
            "adaptive_effective_run_policy": str(extra.get("adaptive_effective_run_policy", "")),
        })
    summary_df = pd.DataFrame(summaries)
    planning_cols = [
        "baseline",
        "arrival_rate_jobs_per_s",
        "shared_kv_extraction_overhead_ms",
        "placement_planning_overhead_ms",
        "replication_planning_overhead_ms",
        "decode_cohort_planning_overhead_ms",
        "scheduling_loop_overhead_ms",
        "total_runtime_overhead_ms",
        "overhead_per_job_ms",
        "overhead_fraction_of_jct",
    ]
    planning_df = summary_df[[c for c in planning_cols if c in summary_df.columns]].copy() if not summary_df.empty else pd.DataFrame(columns=planning_cols)
    admission_df = pd.DataFrame(cohort_admission_rows)
    rejection_df = (
        admission_df.loc[~admission_df["accepted"].astype(bool)].groupby(["baseline", "reason"], as_index=False).size().rename(columns={"size": "count"})
        if not admission_df.empty and "accepted" in admission_df.columns
        else pd.DataFrame(columns=["baseline", "reason", "count"])
    )
    admission_cols = [
        "baseline",
        "arrival_rate_jobs_per_s",
        "candidate_cohorts",
        "accepted_cohorts",
        "rejected_wait_cost",
        "rejected_slo_risk",
        "rejected_low_saving",
        "rejected_critical_path",
        "rejected_queue_pressure",
        "rejected_remote_shared_kv",
        "accepted_avg_size",
        "accepted_avg_wait_ms",
        "decode_shared_kv_read_bytes",
        "decode_shared_kv_read_bytes_without_cohort",
        "decode_kv_read_reduction_ratio",
        "jct_p99_ms",
    ]
    admission_summary = summary_df[[c for c in admission_cols if c in summary_df.columns]].copy() if not summary_df.empty else pd.DataFrame(columns=admission_cols)
    policy_df = pd.DataFrame(policy_decision_rows)
    if not policy_df.empty:
        policy_summary = (
            policy_df.groupby(["baseline", "arrival_rate_jobs_per_s", "chosen_policy"], as_index=False)
            .agg(num_decisions=("scope_id", "count"), mean_opportunity_score=("opportunity_score", "mean"))
        )
    else:
        policy_summary = pd.DataFrame(columns=["baseline", "arrival_rate_jobs_per_s", "chosen_policy", "num_decisions", "mean_opportunity_score"])
    return {
        "global_stage_schedule": stages,
        "global_job_metrics": metrics,
        "global_simulation_summary": summary_df,
        "slo_goodput": slo_df,
        "queue_wait_breakdown": wait_df,
        "resource_utilization": util_df,
        "sram_events": sram_df,
        "mesh_link_events": mesh_df,
        "prefix_blocks": prefix_df,
        "shared_kv_objects": shared_kv_df,
        "decode_cohorts": cohorts_df,
        "cohort_admission_decisions": admission_df,
        "cohort_rejection_reasons": rejection_df,
        "cohort_admission_summary": admission_summary,
        "planning_overhead_summary": planning_df,
        "planning_overhead_by_baseline": planning_df,
        "policy_decisions": policy_df,
        "policy_summary": policy_summary,
    }


def write_global_outputs(result: dict[str, pd.DataFrame], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in result.items():
        df.to_csv(out / f"{name}.csv", index=False)
    if not result["global_job_metrics"].empty:
        write_summary_with_ci(result["global_job_metrics"], out / "summary_with_ci.csv")
