from __future__ import annotations

import glob
from pathlib import Path
from typing import Iterable

import pandas as pd

from waferagent.baselines import BaselineConfig, get_baseline
from waferagent.graph_ir import AgentEdge, AgentGraph, AgentNode, EdgeType, NodeType
from waferagent.kv_model import ModelKVConfig, sharing_metrics
from waferagent.mesh import Mesh, MeshConfig
from waferagent.placement import make_placement
from waferagent.scheduler import ScheduleRecord, schedule_graph
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
                    AgentEdge(dep, tr.node_id, EdgeType.GENERATED_MESSAGE, message_token_len=tr.output_tokens)
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


def _job_metrics(
    graph: AgentGraph,
    schedules: list[ScheduleRecord],
    mesh: Mesh,
    baseline: BaselineConfig,
    model_cfg: ModelKVConfig,
) -> dict[str, float | str]:
    df = pd.DataFrame([s.to_dict() for s in schedules])
    jct = float(df["end_ms"].max() - df["start_ms"].min()) if len(df) else 0.0
    kv = sharing_metrics(graph.nodes.values(), model_cfg)
    if not baseline.kv_sharing:
        kv["shared_kv_bytes"] = kv["naive_kv_bytes"]
        kv["kv_saving_ratio"] = 0.0
        kv["kv_duplication_ratio"] = 1.0
    sram_peak = float(sum(p.sram_need_bytes for p in [])) if False else float(kv["shared_kv_bytes"])
    mesh_stats = mesh.stats()
    prefill = float(df["prefill_ms"].sum()) if len(df) else 0.0
    decode = float(df["decode_ms"].sum()) if len(df) else 0.0
    comm = float(df["comm_ms"].sum()) if len(df) else 0.0
    queue = float(df["queue_wait_ms"].sum()) if len(df) else 0.0
    flops = sum(
        (n.input_token_len + n.actual_output_token_len) * model_cfg.hidden_size * model_cfg.num_hidden_layers * 6
        for n in graph.nodes.values()
    )
    compute_energy_j = flops * mesh.config.energy_per_flop_pJ * 1e-12
    comm_energy_j = mesh_stats["mesh_total_traffic_bytes"] * mesh.config.energy_per_byte_pJ * 1e-12
    cache_hit_rate = float((df["cache_hit"] == "hit").sum() / max(1, (df["cache_hit"].isin(["hit", "miss"])).sum()))
    return {
        "baseline": baseline.name,
        "job_id": graph.graph_id,
        "workload": graph.workload,
        "job_completion_time_ms": jct,
        "p50_latency_ms": float(df["end_ms"].sub(df["start_ms"]).quantile(0.50)) if len(df) else 0.0,
        "p90_latency_ms": float(df["end_ms"].sub(df["start_ms"]).quantile(0.90)) if len(df) else 0.0,
        "p99_latency_ms": float(df["end_ms"].sub(df["start_ms"]).quantile(0.99)) if len(df) else 0.0,
        "goodput_jobs_per_s": 1000.0 / jct if jct else 0.0,
        "naive_kv_bytes": kv["naive_kv_bytes"],
        "kv_bytes_total": kv["shared_kv_bytes"],
        "kv_saving_ratio": kv["kv_saving_ratio"],
        "kv_duplication_ratio": kv["kv_duplication_ratio"],
        "cache_hit_rate": cache_hit_rate,
        "sram_peak_occupancy_bytes": sram_peak,
        "sram_overflow_bytes": max(0.0, sram_peak - mesh.config.total_sram_bytes),
        "prefill_ms_total": prefill,
        "decode_ms_total": decode,
        "communication_time_ms": comm,
        "queue_wait_ms_total": queue,
        "prefill_utilization": prefill / max(jct, prefill) if jct or prefill else 0.0,
        "decode_utilization": decode / max(jct, decode) if jct or decode else 0.0,
        "energy_estimated_j": compute_energy_j + comm_energy_j,
        "energy_per_job_j": compute_energy_j + comm_energy_j,
        **mesh_stats,
    }


def simulate(
    traces: list[TraceRecord],
    mesh_cfg: MeshConfig,
    baseline_names: list[str],
    model_cfg: ModelKVConfig | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_cfg = model_cfg or ModelKVConfig()
    jobs = _group_jobs(traces)
    metric_rows: list[dict[str, float | str]] = []
    schedule_rows: list[dict[str, float | str | int]] = []
    for baseline_name in baseline_names:
        baseline = get_baseline(baseline_name)
        for job_id, rows in jobs.items():
            graph = traces_to_graph(job_id, rows)
            placements = make_placement(
                baseline.placement_policy,
                graph,
                mesh_cfg,
                seed=seed,
                aggregator_aware=baseline.aggregator_placement,
                avoid_hotspots=baseline.mesh_congestion_penalty,
            )
            mesh = Mesh(mesh_cfg)
            schedules = schedule_graph(graph, rows, mesh, placements, baseline)
            metric_rows.append(_job_metrics(graph, schedules, mesh, baseline, model_cfg))
            schedule_rows.extend([s.to_dict() for s in schedules])
    return pd.DataFrame(metric_rows), pd.DataFrame(schedule_rows)


def write_simulation_outputs(
    traces: list[TraceRecord],
    mesh_cfg: MeshConfig,
    baselines: list[str],
    out_dir: str | Path,
    model_cfg: ModelKVConfig | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = Path(out_dir)
    metrics, schedules = simulate(traces, mesh_cfg, baselines, model_cfg, seed)
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "simulation_metrics.csv", index=False)
    schedules.to_csv(out / "schedule.csv", index=False)
    summary = metrics.groupby("baseline", as_index=False).agg(
        job_completion_time_ms=("job_completion_time_ms", "mean"),
        p50_latency_ms=("p50_latency_ms", "mean"),
        p90_latency_ms=("p90_latency_ms", "mean"),
        p99_latency_ms=("p99_latency_ms", "mean"),
        goodput_jobs_per_s=("goodput_jobs_per_s", "sum"),
        kv_bytes_total=("kv_bytes_total", "mean"),
        kv_saving_ratio=("kv_saving_ratio", "mean"),
        mesh_total_traffic_bytes=("mesh_total_traffic_bytes", "mean"),
        mesh_hotspot_ratio=("mesh_hotspot_ratio", "mean"),
        energy_per_job_j=("energy_per_job_j", "mean"),
    )
    summary.to_csv(out / "simulation_summary.csv", index=False)
    return metrics, schedules
