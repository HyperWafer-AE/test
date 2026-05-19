from __future__ import annotations

import heapq
from pathlib import Path
from typing import Any

import pandas as pd

from waferagent.arrival import ArrivalConfig, generate_arrivals
from waferagent.baselines import get_baseline
from waferagent.calibrated_cost_model import CalibratedCostModel
from waferagent.cohort_scheduler import CohortConfig, form_decode_cohorts
from waferagent.kv_model import ModelKVConfig, sharing_metrics
from waferagent.mesh import MeshConfig
from waferagent.mesh_network import MeshNetwork
from waferagent.placement import make_placement
from waferagent.prefix_extension_cost_model import PrefixExtensionCostModel
from waferagent.prefix_tree import PrefixComputeTracker
from waferagent.resource_model import ResourceModel
from waferagent.shared_attention_cost import estimate_shared_attention_cost
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
    baseline_summaries: dict[str, dict[str, float]] = {}

    for baseline_name in baseline_names:
        baseline = get_baseline(baseline_name, neutral=neutral_multipliers)
        state = _build_global_state(traces, seed, mesh_cfg, baseline)
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
        shared_objects, replication_stats = plan_shared_kv_replication(
            list(object_by_prefix.values()),
            baseline.shared_kv_replication_policy if baseline.shared_kv_replication_policy != "none" else "no_replication",
            mesh_cfg,
        )
        cohorts, cohort_stats = form_decode_cohorts(
            stages,
            shared_objects,
            node_regions=node_regions,
            cfg=CohortConfig(enabled=baseline.shared_kv_decode_cohort or baseline.oracle),
        )
        attention_stats = estimate_shared_attention_cost(
            shared_objects,
            cohorts if (baseline.shared_kv_decode_cohort or baseline.oracle) else [],
            bytes_per_ms=max(1.0, mesh_cfg.link_bandwidth_GBps * 1e9 / 1000.0),
        )
        for obj in shared_objects:
            shared_kv_rows.append({**obj.to_dict(), "baseline": baseline.name})
        for cohort in cohorts:
            cohort_rows.append({**cohort.to_dict(), "baseline": baseline.name})
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
        }
        while ready:
            _ready_time, _prio, sid = heapq.heappop(ready)
            stage = stages[sid]
            job_id = stage.job_id
            graph = graphs[job_id]
            node = graph.nodes[stage.parent_node_id]
            dep_ready = max([arrivals[job_id], *[end_times[d] for d in stage.deps]], default=arrivals[job_id])
            placement = placements[job_id][stage.parent_node_id]
            decision = prefix_compute.decide(stage, baseline) if stage.stage_type == "prefill" else None
            computed_input_tokens = decision.computed_input_tokens if decision else stage.input_tokens
            sram_read = sram_write = reload_bytes = mesh_bytes = 0
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
                    shared_bytes_no = (
                        int(stage.output_tokens)
                        * int(stage.shared_prefix_token_len)
                        * int(model_cfg.kv_bytes_per_token)
                    )
                    prefix = stage.shared_prefix_ids[0]
                    if baseline.shared_kv_decode_cohort or baseline.oracle:
                        users = prefix_decode_users.get(prefix, 1)
                        waves = (users + 4 - 1) // 4
                        factor = min(1.0, (waves * 1.15) / max(1, users))
                    else:
                        factor = 1.0
                    shared_bytes_actual = shared_bytes_no * factor
                    decode_kv_latency = shared_bytes_actual / max(
                        1.0, mesh_cfg.link_bandwidth_GBps * 1e9 / 1000.0
                    )
                    duration += decode_kv_latency
                    accum["decode_shared_kv_read_bytes_without_cohort"] += shared_bytes_no
                    accum["decode_shared_kv_read_bytes"] += shared_bytes_actual
                    accum["decode_attention_latency_ms"] += decode_kv_latency
            if stage.stage_type == "prefill" and reload_bytes:
                duration += reload_bytes / max(1.0, mesh_cfg.link_bandwidth_GBps * 1e9 / 1000.0)
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
            elif stage.stage_type == "decode" and stage.shared_prefix_ids and stage.shared_prefix_token_len > 0:
                shared_bytes = int(
                    int(stage.output_tokens)
                    * int(stage.shared_prefix_token_len)
                    * int(model_cfg.kv_bytes_per_token)
                )
                if baseline.shared_kv_decode_cohort or baseline.oracle:
                    users = prefix_decode_users.get(stage.shared_prefix_ids[0], 1)
                    waves = (users + 4 - 1) // 4
                    shared_bytes = int(shared_bytes * min(1.0, (waves * 1.15) / max(1, users)))
                src_tile = (0, 0)
                w, t, b = mesh.route(job_id, sid, src_tile, placement.tile, shared_bytes, dep_ready, "decode_shared_kv")
                mesh_wait += w
                mesh_time = max(mesh_time, t)
                mesh_bytes += b
            resource_ready = dep_ready + mesh_time
            start, end, tiles = resource.reserve_stage(stage.tile_pool, resource_ready, duration, _requested_tiles(stage, mesh_cfg, baseline))
            end_times[sid] = end
            job_last_end[job_id] = max(job_last_end.get(job_id, 0.0), end)
            if stage.stage_type == "decode":
                job_first_token.setdefault(job_id, end)
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
            ).to_dict()
            row["arrival_ms"] = arrivals[job_id]
            row["global_stage_id"] = sid
            stage_rows.append(row)
            step += 1
            for child in children.get(sid, []):
                indegree[child] -= 1
                if indegree[child] == 0:
                    child_job = stages[child].job_id
                    prio = _stage_priority(graphs[child_job], stages[child], baseline)[0]
                    ready_time = max([arrivals[child_job], *[end_times[d] for d in stages[child].deps]], default=arrivals[child_job])
                    heapq.heappush(ready, (ready_time, prio, child))
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
        total_energy = compute_energy + mesh_energy
        energy_per_job = total_energy / max(1, len(jobs))
        for job_id in sorted(jobs):
            job_rows = sched_df[sched_df["job_id"] == job_id]
            decode_rows = job_rows[job_rows["stage_type"] == "decode"]
            jct = job_last_end.get(job_id, arrivals[job_id]) - arrivals[job_id]
            ttft = job_first_token.get(job_id, arrivals[job_id]) - arrivals[job_id]
            out_tokens = max(1.0, float(decode_rows["parent_node_id"].count()))
            tpot = float((decode_rows["end_ms"] - decode_rows["start_ms"]).sum()) / out_tokens if not decode_rows.empty else 0.0
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
                "kv_saving_ratio": 0.0 if not baseline.kv_sharing else kv["kv_saving_ratio"],
                "computed_prefill_tokens": accum["computed_prefill_tokens"] / max(1, len(jobs)),
                "avoided_prefill_tokens": accum["avoided_prefill_tokens"] / max(1, len(jobs)),
                "compute_energy_j": compute_energy / max(1, len(jobs)),
                "mesh_energy_j": mesh_energy / max(1, len(jobs)),
                "energy_per_job_j": energy_per_job,
                "energy_per_completed_job_under_slo_j": energy_per_job,
                "decode_shared_kv_read_bytes": accum["decode_shared_kv_read_bytes"] / max(1, len(jobs)),
                "decode_shared_kv_read_bytes_without_cohort": accum["decode_shared_kv_read_bytes_without_cohort"] / max(1, len(jobs)),
                "decode_kv_read_reduction_ratio": 1.0
                - accum["decode_shared_kv_read_bytes"]
                / max(1.0, accum["decode_shared_kv_read_bytes_without_cohort"]),
                "cross_region_kv_transfer_bytes": accum["decode_shared_kv_read_bytes"] / max(1, len(jobs)),
                "decode_query_transfer_bytes": attention_stats.get("decode_query_transfer_bytes", 0.0) / max(1, len(jobs)),
                "decode_merge_bytes": attention_stats.get("decode_merge_bytes", 0.0) / max(1, len(jobs)),
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
            "energy_per_job_j": energy_per_job,
            **replication_stats,
            **cohort_stats,
            **attention_stats,
            "decode_shared_kv_read_bytes": float(accum["decode_shared_kv_read_bytes"]),
            "decode_shared_kv_read_bytes_without_cohort": float(accum["decode_shared_kv_read_bytes_without_cohort"]),
            "cross_region_kv_transfer_bytes": float(accum["decode_shared_kv_read_bytes"]),
            "decode_attention_latency_ms": float(accum["decode_attention_latency_ms"]),
            "computed_prefill_tokens": float(accum["computed_prefill_tokens"]),
            "computed_decode_tokens": float(accum["computed_decode_tokens"]),
            "avoided_prefill_tokens": float(accum["avoided_prefill_tokens"]),
        }
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
            "energy_per_job_j": float(extra.get("energy_per_job_j", 0.0)),
            "decode_shared_kv_read_bytes": float(extra.get("decode_shared_kv_read_bytes", 0.0)),
            "decode_shared_kv_read_bytes_without_cohort": float(extra.get("decode_shared_kv_read_bytes_without_cohort", 0.0)),
            "decode_kv_read_reduction_ratio": float(extra.get("decode_kv_read_reduction_ratio", extra.get("shared_kv_read_reduction_ratio", 0.0))),
            "cross_region_kv_transfer_bytes": float(extra.get("cross_region_kv_transfer_bytes", 0.0)),
            "num_decode_cohorts": float(extra.get("num_decode_cohorts", 0.0)),
            "avg_cohort_size": float(extra.get("avg_cohort_size", 0.0)),
            "replica_bytes_total": float(extra.get("replica_bytes_total", 0.0)),
            "saved_mesh_traffic_bytes": float(extra.get("saved_mesh_traffic_bytes", 0.0)),
            "replication_transfer_bytes": float(extra.get("replication_transfer_bytes", 0.0)),
        })
    return {
        "global_stage_schedule": stages,
        "global_job_metrics": metrics,
        "global_simulation_summary": pd.DataFrame(summaries),
        "slo_goodput": slo_df,
        "queue_wait_breakdown": wait_df,
        "resource_utilization": util_df,
        "sram_events": sram_df,
        "mesh_link_events": mesh_df,
        "prefix_blocks": prefix_df,
        "shared_kv_objects": shared_kv_df,
        "decode_cohorts": cohorts_df,
    }


def write_global_outputs(result: dict[str, pd.DataFrame], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in result.items():
        df.to_csv(out / f"{name}.csv", index=False)
    if not result["global_job_metrics"].empty:
        write_summary_with_ci(result["global_job_metrics"], out / "summary_with_ci.csv")
