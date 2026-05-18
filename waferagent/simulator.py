from __future__ import annotations

import glob
import heapq
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from waferagent.baselines import BaselineConfig, get_baseline
from waferagent.calibrated_cost_model import CalibratedCostModel
from waferagent.graph_ir import AgentEdge, AgentGraph, AgentNode, EdgeType, NodeType
from waferagent.kv_model import ModelKVConfig, sharing_metrics
from waferagent.mesh import MeshConfig
from waferagent.mesh_network import MeshNetwork
from waferagent.placement import Placement, make_placement
from waferagent.prefix_tree import PrefixComputeTracker
from waferagent.resource_model import ResourceModel
from waferagent.sram_manager import DistributedSRAMManager
from waferagent.stage_ir import Stage, StageSchedule, build_stages
from waferagent.statistics import write_summary_with_ci
from waferagent.tool_ttl import tool_resume_probability
from waferagent.trace_schema import TraceRecord, read_traces


def load_trace_glob(patterns: list[str] | str) -> list[TraceRecord]:
    if isinstance(patterns, str):
        patterns = [patterns]
    paths: list[str] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        paths.extend(matches if matches else [pattern])
    traces: list[TraceRecord] = []
    for p in sorted(set(paths)):
        traces.extend(read_traces(p))
    return traces


def traces_to_graph(job_id: str, traces: list[TraceRecord]) -> AgentGraph:
    if not traces:
        raise ValueError("No traces")
    workload = traces[0].workload
    graph = AgentGraph(job_id, workload, seed=0)
    for tr in traces:
        node = AgentNode(
            node_id=tr.node_id,
            job_id=tr.job_id,
            agent_id=tr.agent_id,
            round_id=tr.round_id,
            node_type=NodeType(tr.node_type),
            role=tr.role,
            model_id=tr.model_name,
            input_token_len=tr.input_tokens,
            expected_output_token_len=tr.output_tokens,
            actual_output_token_len=tr.output_tokens,
            shared_prefix_ids=list(tr.shared_prefix_ids),
            private_prefix_ids=list(tr.private_prefix_ids),
            deps=list(tr.deps),
            tool_latency_ms=tr.tool_latency_ms,
            kv_bytes_estimated=tr.kv_bytes_estimated,
            shared_prefix_token_len=tr.shared_prefix_token_len,
            private_prefix_token_len=tr.private_prefix_token_len,
            prompt_hash=tr.prompt_hash,
        )
        graph.add_node(node)
    for tr in traces:
        for dep in tr.deps:
            if dep in graph.nodes and tr.node_id in graph.nodes:
                graph.add_edge(
                    AgentEdge(dep, tr.node_id, EdgeType.GENERATED_MESSAGE, message_token_len=max(1, tr.output_tokens))
                )
    graph.critical_path_lengths()
    return graph


def _group_jobs(traces: Iterable[TraceRecord]) -> dict[str, list[TraceRecord]]:
    jobs: dict[str, list[TraceRecord]] = {}
    for tr in traces:
        jobs.setdefault(tr.job_id, []).append(tr)
    for rows in jobs.values():
        rows.sort(key=lambda r: (r.round_id, r.node_id))
    return jobs


def _prefill_pressure(stages: dict[str, Stage]) -> float:
    prefill = sum(s.duration_ms for s in stages.values() if s.stage_type == "prefill")
    decode = sum(s.duration_ms for s in stages.values() if s.stage_type == "decode")
    total = prefill + decode
    return prefill / total if total else 0.5


def _stage_priority(graph: AgentGraph, stage: Stage, baseline: BaselineConfig) -> tuple[float, str]:
    node = graph.nodes[stage.parent_node_id]
    if baseline.critical_path or baseline.oracle:
        return (-node.criticality, stage.stage_id)
    if baseline.scheduling_policy == "kvflow_like_steps_to_execution":
        return (-len(stage.shared_prefix_ids), stage.stage_id)
    return (float(node.round_id), stage.stage_id)


def _requested_tiles(stage: Stage, mesh_cfg: MeshConfig, baseline: BaselineConfig) -> int:
    if stage.stage_type == "prefill":
        base = max(1, stage.input_tokens // 2048)
    elif stage.stage_type == "decode":
        base = max(1, stage.output_tokens // 64)
    else:
        return 0
    if baseline.oracle:
        base *= 2
    return max(1, min(base, mesh_cfg.total_tiles // 4 or 1))


def _synthetic_prefill_ms(tokens: int) -> float:
    return 0.012 * max(0, tokens) + 1e-6 * max(0, tokens) * max(0, tokens)


def _synthetic_decode_ms(output_tokens: int) -> float:
    return 0.42 * max(1, output_tokens)


def _scaled_stage_duration(
    stage: Stage,
    mesh_cfg: MeshConfig,
    baseline: BaselineConfig,
    duration_source: str = "trace",
    cost_model: CalibratedCostModel | None = None,
    computed_input_tokens: int | None = None,
) -> tuple[float, float]:
    """Return (duration_for_computed_tokens, full_duration_without_prefix_reuse)."""

    tokens = max(0, int(computed_input_tokens if computed_input_tokens is not None else stage.input_tokens))
    full_tokens = max(0, int(stage.input_tokens))
    if stage.stage_type == "prefill":
        scale = mesh_cfg.h100_prefill_to_wafer_scale
        if duration_source == "calibrated" and cost_model is not None:
            computed = cost_model.prefill_ms(tokens) * scale
            full = cost_model.prefill_ms(full_tokens) * scale
        elif duration_source == "synthetic":
            computed = _synthetic_prefill_ms(tokens) * scale
            full = _synthetic_prefill_ms(full_tokens) * scale
        else:
            full = max(0.0, stage.duration_ms * scale)
            ratio = tokens / max(1, full_tokens)
            computed = full * ratio
        return (
            max(0.0, computed * baseline.prefill_time_multiplier),
            max(0.0, full * baseline.prefill_time_multiplier),
        )
    if stage.stage_type == "decode":
        scale = mesh_cfg.h100_decode_to_wafer_scale
        if duration_source == "calibrated" and cost_model is not None:
            full = cost_model.decode_ms(stage.input_tokens, stage.output_tokens) * scale
        elif duration_source == "synthetic":
            full = _synthetic_decode_ms(stage.output_tokens) * scale
        else:
            full = max(0.0, stage.duration_ms * scale)
        return max(0.0, full * baseline.decode_time_multiplier), max(0.0, full * baseline.decode_time_multiplier)
    return max(0.0, stage.duration_ms), max(0.0, stage.duration_ms)


def _dependency_mesh_bytes(graph: AgentGraph, stage: Stage) -> int:
    if not stage.deps:
        return 0
    parent = graph.nodes[stage.parent_node_id]
    dep_tokens = 0
    for dep in parent.deps:
        for edge in graph.edges:
            if edge.src == dep and edge.dst == parent.node_id:
                dep_tokens += max(1, edge.message_token_len)
    if dep_tokens == 0:
        dep_tokens = max(1, stage.input_tokens // 128)
    return dep_tokens * 4


def _simulate_job(
    graph: AgentGraph,
    traces: list[TraceRecord],
    mesh_cfg: MeshConfig,
    baseline: BaselineConfig,
    model_cfg: ModelKVConfig,
    seed: int,
    duration_source: str = "trace",
    cost_model: CalibratedCostModel | None = None,
    allow_mesh_compute_overlap: bool = False,
) -> tuple[dict[str, float | str | int], list[dict], list[dict], list[dict], list[dict], list[dict]]:
    stages = build_stages(graph, traces)
    placements = make_placement(
        baseline.placement_policy,
        graph,
        mesh_cfg,
        seed=seed,
        aggregator_aware=baseline.aggregator_placement or baseline.oracle,
        avoid_hotspots=baseline.hotspot_aware_placement or baseline.oracle,
    )
    resource = ResourceModel.from_config(
        mesh_cfg,
        dynamic_pd_partition=baseline.dynamic_pd_partition or baseline.oracle,
        prefill_pressure=_prefill_pressure(stages),
    )
    sram_capacity = mesh_cfg.total_sram_bytes * (2 if baseline.oracle else 1)
    sram = DistributedSRAMManager(mesh_cfg, baseline.ttl_policy, baseline.name)
    mesh = MeshNetwork(mesh_cfg, baseline.name, congestion_enabled=baseline.mesh_congestion_penalty or baseline.oracle)
    prefix_compute = PrefixComputeTracker()

    children: dict[str, list[str]] = {sid: [] for sid in stages}
    indegree = {sid: len(stage.deps) for sid, stage in stages.items()}
    for sid, stage in stages.items():
        for dep in stage.deps:
            children.setdefault(dep, []).append(sid)

    ready: list[tuple[float, str]] = []
    for sid, deg in indegree.items():
        if deg == 0:
            heapq.heappush(ready, (_stage_priority(graph, stages[sid], baseline)[0], sid))
    end_times: dict[str, float] = {}
    schedule_rows: list[dict] = []
    step = 0
    tool_pause_ms = 0.0
    resume_prefill_ms = 0.0
    resume_reload_bytes = 0
    shared_prefill_compute_ms_saved = 0.0
    cross_region_sram_bytes_total = 0

    while ready:
        _, sid = heapq.heappop(ready)
        stage = stages[sid]
        node = graph.nodes[stage.parent_node_id]
        dep_ready = max((end_times[d] for d in stage.deps), default=0.0)
        placement = placements[stage.parent_node_id]
        sram_read = sram_write = 0
        reload_bytes = 0
        cross_region_bytes = 0
        pending_region_transfers: list[tuple[tuple[int, int], int]] = []
        computed_input_tokens = stage.input_tokens
        if stage.stage_type == "prefill":
            decision = prefix_compute.decide(stage, baseline)
            computed_input_tokens = decision.computed_input_tokens
        else:
            decision = None
        if stage.stage_type == "prefill" and baseline.kv_sharing:
            for pid in stage.shared_prefix_ids:
                block_bytes = int(stage.shared_prefix_token_len * model_cfg.kv_bytes_per_token)
                prob = tool_resume_probability(graph, stage.parent_node_id)
                pin = (baseline.tool_ttl or baseline.oracle) and prob > 0
                access = sram.access(
                    graph.graph_id,
                    stage.stage_id,
                    pid,
                    stage.shared_prefix_token_len,
                    block_bytes,
                    step,
                    node.criticality,
                    placement.tile,
                    tool_resume_probability=prob,
                    pin=pin,
                    compute_store=bool(decision and decision.shared_tokens_computed > 0),
                )
                if access.hit:
                    sram_read += block_bytes
                else:
                    sram_write += block_bytes
                    reload_bytes += access.reload_bytes
                if access.cross_region_hit:
                    cross_region_bytes += block_bytes
                    cross_region_sram_bytes_total += block_bytes
                    if access.source_tile:
                        pending_region_transfers.append((access.source_tile, block_bytes))
            if reload_bytes and tool_resume_probability(graph, stage.parent_node_id) > 0:
                resume_reload_bytes += reload_bytes
        elif stage.stage_type == "prefill" and not baseline.kv_sharing:
            block_id = f"{stage.stage_id}:full_kv"
            block_bytes = int(stage.kv_bytes_estimated)
            prob = tool_resume_probability(graph, stage.parent_node_id)
            access = sram.access(
                graph.graph_id,
                stage.stage_id,
                block_id,
                stage.input_tokens,
                block_bytes,
                step,
                node.criticality,
                placement.tile,
                tool_resume_probability=prob,
                pin=(baseline.tool_ttl or baseline.oracle) and prob > 0,
                compute_store=not (prob > 0 and not (baseline.tool_ttl or baseline.oracle)),
            )
            sram_write += block_bytes
            reload_bytes += access.reload_bytes
            if tool_resume_probability(graph, stage.parent_node_id) > 0:
                resume_reload_bytes += reload_bytes

        base_duration, full_duration = _scaled_stage_duration(
            stage,
            mesh_cfg,
            baseline,
            duration_source=duration_source,
            cost_model=cost_model,
            computed_input_tokens=computed_input_tokens,
        )
        if stage.stage_type == "prefill":
            shared_prefill_compute_ms_saved += max(0.0, full_duration - base_duration)
        if stage.stage_type == "prefill" and reload_bytes:
            reload_penalty_ms = reload_bytes / max(1.0, mesh_cfg.link_bandwidth_GBps * 1e9 / 1000.0)
            if baseline.tool_ttl or baseline.oracle:
                reload_penalty_ms *= 0.25
            base_duration += reload_penalty_ms
            if tool_resume_probability(graph, stage.parent_node_id) > 0:
                resume_prefill_ms += reload_penalty_ms
        if stage.stage_type == "tool":
            tool_pause_ms += base_duration

        mesh_bytes = 0
        mesh_wait = 0.0
        mesh_time = 0.0
        if stage.stage_type == "prefill":
            dep_bytes = _dependency_mesh_bytes(graph, stage)
            dep_tiles = [placements[graph.nodes[d].node_id].tile for d in graph.nodes[stage.parent_node_id].deps if d in placements]
            if dep_tiles:
                for src_tile in dep_tiles:
                    w, t, b = mesh.route(graph.graph_id, sid, src_tile, placement.tile, dep_bytes, dep_ready, "message_tokens")
                    mesh_wait += w
                    mesh_time = max(mesh_time, t)
                    mesh_bytes += b
            if reload_bytes:
                src = (0, 0)
                w, t, b = mesh.route(graph.graph_id, sid, src, placement.tile, int(reload_bytes), dep_ready, "kv_reload")
                mesh_wait += w
                mesh_time = max(mesh_time, t)
                mesh_bytes += b
            for src_tile, bytes_moved in pending_region_transfers:
                w, t, b = mesh.route(graph.graph_id, sid, src_tile, placement.tile, int(bytes_moved), dep_ready, "kv_replication")
                mesh_wait += w
                mesh_time = max(mesh_time, t)
                mesh_bytes += b
        resource_ready = dep_ready if allow_mesh_compute_overlap else dep_ready + mesh_time
        if allow_mesh_compute_overlap:
            base_duration += mesh_time if (baseline.mesh_congestion_penalty or baseline.oracle) else max(0.0, mesh_time - mesh_wait)
        requested = _requested_tiles(stage, mesh_cfg, baseline)
        start, end, assigned_tiles = resource.reserve_stage(stage.tile_pool, resource_ready, base_duration, requested)
        end_times[sid] = end
        row = StageSchedule(
            stage_id=sid,
            parent_node_id=stage.parent_node_id,
            job_id=graph.graph_id,
            baseline=baseline.name,
            stage_type=stage.stage_type,
            start_ms=start,
            end_ms=end,
            assigned_tiles=assigned_tiles,
            sram_read_bytes=int(sram_read),
            sram_write_bytes=int(sram_write),
            mesh_bytes=int(mesh_bytes),
            mesh_wait_ms=float(mesh_wait),
            queue_wait_ms=max(0.0, start - dep_ready),
            stall_reason="mesh" if mesh_wait > 0 else ("resource" if start > dep_ready else ""),
        ).to_dict()
        schedule_rows.append(row)
        step += 1
        for child in children.get(sid, []):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, (_stage_priority(graph, stages[child], baseline)[0], child))

    if len(schedule_rows) != len(stages):
        raise RuntimeError("Stage scheduler failed to schedule all stages")

    sched_df = pd.DataFrame(schedule_rows)
    jct = float(sched_df["end_ms"].max() - sched_df["start_ms"].min()) if len(sched_df) else 0.0
    kv = sharing_metrics(graph.nodes.values(), model_cfg)
    if not baseline.kv_sharing:
        kv["shared_kv_bytes"] = kv["naive_kv_bytes"]
        kv["kv_saving_ratio"] = 0.0
        kv["kv_duplication_ratio"] = 1.0
    mesh_stats = mesh.stats()
    resource_stats = resource.stats(jct)
    sram_stats = sram.stats()
    prefill = float(sched_df.loc[sched_df["stage_type"] == "prefill", "end_ms"].sub(sched_df.loc[sched_df["stage_type"] == "prefill", "start_ms"]).sum())
    decode = float(sched_df.loc[sched_df["stage_type"] == "decode", "end_ms"].sub(sched_df.loc[sched_df["stage_type"] == "decode", "start_ms"]).sum())
    flops = sum(
        (n.input_token_len + n.actual_output_token_len) * model_cfg.hidden_size * model_cfg.num_hidden_layers * 6
        for n in graph.nodes.values()
    )
    energy = flops * mesh_cfg.energy_per_flop_pJ * 1e-12 + mesh_stats["mesh_total_traffic_bytes"] * mesh_cfg.energy_per_byte_pJ * 1e-12
    metrics = {
        "baseline": baseline.name,
        "mechanism_profile": baseline.mechanism_profile,
        "job_id": graph.graph_id,
        "workload": graph.workload,
        "seed": seed,
        "duration_source": duration_source,
        "job_completion_time_ms": jct,
        "p50_latency_ms": float(sched_df["end_ms"].sub(sched_df["start_ms"]).quantile(0.50)) if len(sched_df) else 0.0,
        "p90_latency_ms": float(sched_df["end_ms"].sub(sched_df["start_ms"]).quantile(0.90)) if len(sched_df) else 0.0,
        "p99_latency_ms": float(sched_df["end_ms"].sub(sched_df["start_ms"]).quantile(0.99)) if len(sched_df) else 0.0,
        "goodput_jobs_per_s": 1000.0 / jct if jct else 0.0,
        "naive_kv_bytes": kv["naive_kv_bytes"],
        "kv_bytes_total": kv["shared_kv_bytes"],
        "kv_saving_ratio": kv["kv_saving_ratio"],
        "kv_duplication_ratio": kv["kv_duplication_ratio"],
        "sram_peak_occupancy_bytes": min(kv["shared_kv_bytes"], sram_capacity),
        "sram_overflow_bytes": max(0.0, kv["shared_kv_bytes"] - sram_capacity),
        "prefill_ms_total": prefill,
        "decode_ms_total": decode,
        "queue_wait_ms_total": float(sched_df["queue_wait_ms"].sum()) if len(sched_df) else 0.0,
        "energy_estimated_j": energy,
        "energy_per_job_j": energy,
        "tool_pause_ms": tool_pause_ms,
        "resume_prefill_ms": resume_prefill_ms,
        "resume_reload_bytes": float(resume_reload_bytes),
        "cross_region_sram_bytes": float(cross_region_sram_bytes_total),
        **prefix_compute.stats(shared_prefill_compute_ms_saved),
        **mesh_stats,
        **resource_stats,
        **sram_stats,
    }
    return (
        metrics,
        schedule_rows,
        [e.to_dict() for e in sram.events],
        [e.to_dict() for e in mesh.events],
        sram.prefix_block_rows(),
        [
            {
                "baseline": baseline.name,
                "job_id": graph.graph_id,
                "node_id": node_id,
                "tile_r": p.tile[0],
                "tile_c": p.tile[1],
                "placement_region": p.placement_region,
                "sram_overflow_bytes": p.sram_overflow_bytes,
            }
            for node_id, p in placements.items()
        ],
    )


def _load_calibration_scale(calibration: str | Path | None) -> dict[str, Any]:
    if not calibration:
        return {"calibration_loaded": 0.0}
    try:
        data = json.loads(Path(calibration).read_text(encoding="utf-8"))
        from waferagent.utils import file_sha256

        return {
            "calibration_loaded": 1.0,
            "calibration_coeff_count": float(len(str(data))),
            "calibration_fit_hash": file_sha256(calibration),
        }
    except Exception:
        return {"calibration_loaded": 0.0, "calibration_fit_hash": ""}


def simulate(
    traces: list[TraceRecord],
    mesh_cfg: MeshConfig,
    baseline_names: list[str],
    model_cfg: ModelKVConfig | None = None,
    seed: int = 0,
    neutral_multipliers: bool = True,
    calibration: str | Path | None = None,
    duration_source: str = "trace",
    allow_mesh_compute_overlap: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_cfg = model_cfg or ModelKVConfig()
    jobs = _group_jobs(traces)
    metric_rows: list[dict] = []
    schedule_rows: list[dict] = []
    sram_rows: list[dict] = []
    mesh_rows: list[dict] = []
    prefix_rows: list[dict] = []
    placement_rows: list[dict] = []
    calib_meta = _load_calibration_scale(calibration)
    cost_model = None
    if duration_source == "calibrated":
        if not calibration:
            raise ValueError("--duration-source calibrated requires --calibration")
        cost_model = CalibratedCostModel.from_json(calibration)
    for baseline_name in baseline_names:
        baseline = get_baseline(baseline_name, neutral=neutral_multipliers)
        for job_id, rows in jobs.items():
            graph = traces_to_graph(job_id, rows)
            metrics, sched, sram_events, mesh_events, prefixes, placements = _simulate_job(
                graph,
                rows,
                mesh_cfg,
                baseline,
                model_cfg,
                seed,
                duration_source=duration_source,
                cost_model=cost_model,
                allow_mesh_compute_overlap=allow_mesh_compute_overlap,
            )
            metrics.update(calib_meta)
            metrics["duration_source"] = duration_source
            metric_rows.append(metrics)
            schedule_rows.extend(sched)
            sram_rows.extend(sram_events)
            mesh_rows.extend(mesh_events)
            prefix_rows.extend([{**r, "baseline": baseline.name, "job_id": job_id} for r in prefixes])
            placement_rows.extend(placements)
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(schedule_rows),
        pd.DataFrame(sram_rows),
        pd.DataFrame(mesh_rows),
        pd.DataFrame(prefix_rows),
    )


def write_simulation_outputs(
    traces: list[TraceRecord],
    mesh_cfg: MeshConfig,
    baselines: list[str],
    out_dir: str | Path,
    model_cfg: ModelKVConfig | None = None,
    seed: int = 0,
    neutral_multipliers: bool = True,
    calibration: str | Path | None = None,
    duration_source: str = "trace",
    allow_mesh_compute_overlap: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = Path(out_dir)
    metrics, stages, sram_events, mesh_events, prefix_blocks = simulate(
        traces,
        mesh_cfg,
        baselines,
        model_cfg,
        seed,
        neutral_multipliers,
        calibration,
        duration_source,
        allow_mesh_compute_overlap,
    )
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "simulation_metrics.csv", index=False)
    stages.to_csv(out / "stage_schedule.csv", index=False)
    stages.to_csv(out / "schedule.csv", index=False)
    sram_events.to_csv(out / "sram_events.csv", index=False)
    mesh_events.to_csv(out / "mesh_link_events.csv", index=False)
    if not mesh_events.empty and "traffic_source" in mesh_events.columns:
        breakdown = mesh_events.groupby(["baseline", "traffic_source"], as_index=False).agg(
            mesh_bytes=("bytes", "sum"),
            mesh_wait_ms=("wait_ms", "sum"),
            transfer_ms=("transfer_ms", "sum"),
        )
    else:
        breakdown = pd.DataFrame(columns=["baseline", "traffic_source", "mesh_bytes", "mesh_wait_ms", "transfer_ms"])
    breakdown.to_csv(out / "mesh_traffic_breakdown.csv", index=False)
    prefix_blocks.to_csv(out / "prefix_blocks.csv", index=False)
    summary = metrics.groupby("baseline", as_index=False).agg(
        job_completion_time_ms=("job_completion_time_ms", "mean"),
        p50_latency_ms=("p50_latency_ms", "mean"),
        p90_latency_ms=("p90_latency_ms", "mean"),
        p99_latency_ms=("p99_latency_ms", "mean"),
        goodput_jobs_per_s=("goodput_jobs_per_s", "sum"),
        kv_bytes_total=("kv_bytes_total", "mean"),
        kv_saving_ratio=("kv_saving_ratio", "mean"),
        sram_evictions=("sram_evictions", "mean"),
        sram_reload_bytes=("sram_reload_bytes", "mean"),
        sram_hit_rate=("sram_hit_rate", "mean"),
        shared_prefill_compute_ms_saved=("shared_prefill_compute_ms_saved", "mean"),
        shared_prefill_tokens_saved=("shared_prefill_tokens_saved", "mean"),
        prefix_compute_hit_rate=("prefix_compute_hit_rate", "mean"),
        mesh_total_traffic_bytes=("mesh_total_traffic_bytes", "mean"),
        mesh_wait_ms=("mesh_wait_ms", "mean"),
        mesh_hotspot_ratio=("mesh_hotspot_ratio", "mean"),
        prefill_tile_utilization=("prefill_tile_utilization", "mean"),
        decode_tile_utilization=("decode_tile_utilization", "mean"),
        energy_per_job_j=("energy_per_job_j", "mean"),
    )
    summary.to_csv(out / "simulation_summary.csv", index=False)
    write_summary_with_ci(metrics, out / "summary_with_ci.csv")
    return metrics, stages
