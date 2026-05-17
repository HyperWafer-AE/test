from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from waferagent.graph_ir import AgentGraph
from waferagent.kv_model import ModelKVConfig, sharing_metrics
from waferagent.trace_schema import TraceRecord


def graph_stats(graphs: Iterable[AgentGraph]) -> pd.DataFrame:
    rows = []
    for graph in graphs:
        cp = graph.critical_path_lengths()
        stats = graph.fan_in_out_stats()
        stats.update(graph.shared_prefix_stats())
        stats.update(
            {
                "job_id": graph.graph_id,
                "workload": graph.workload,
                "critical_path_weight": max(cp.values(), default=0.0),
                "total_work_weight": sum(n.runtime_weight for n in graph.nodes.values()),
            }
        )
        rows.append(stats)
    return pd.DataFrame(rows)


def token_stats(traces: Iterable[TraceRecord]) -> pd.DataFrame:
    rows = []
    for tr in traces:
        rows.append(
            {
                "job_id": tr.job_id,
                "workload": tr.workload,
                "node_id": tr.node_id,
                "input_tokens": tr.input_tokens,
                "output_tokens": tr.output_tokens,
                "shared_prefix_token_len": tr.shared_prefix_token_len,
                "private_prefix_token_len": tr.private_prefix_token_len,
                "shared_prefix_ratio": tr.shared_prefix_token_len / tr.input_tokens if tr.input_tokens else 0.0,
            }
        )
    return pd.DataFrame(rows)


def kv_stats(graphs: Iterable[AgentGraph], model_cfg: ModelKVConfig | None = None) -> pd.DataFrame:
    rows = []
    for graph in graphs:
        row = sharing_metrics(graph.nodes.values(), model_cfg)
        row.update({"job_id": graph.graph_id, "workload": graph.workload})
        rows.append(row)
    return pd.DataFrame(rows)


def latency_breakdown(traces: Iterable[TraceRecord]) -> pd.DataFrame:
    rows = []
    for tr in traces:
        rows.append(
            {
                "job_id": tr.job_id,
                "workload": tr.workload,
                "node_id": tr.node_id,
                "ttft_ms": tr.ttft_ms,
                "decode_ms": tr.decode_ms,
                "tool_latency_ms": tr.tool_latency_ms,
                "total_ms": tr.total_ms,
            }
        )
    return pd.DataFrame(rows)


def critical_path_df(graphs: Iterable[AgentGraph]) -> pd.DataFrame:
    rows = []
    for graph in graphs:
        cp = graph.critical_path_lengths()
        total = sum(n.runtime_weight for n in graph.nodes.values())
        rows.append(
            {
                "job_id": graph.graph_id,
                "workload": graph.workload,
                "critical_path_weight": max(cp.values(), default=0.0),
                "total_work_weight": total,
                "critical_path_fraction": max(cp.values(), default=0.0) / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def write_characterization_tables(
    graphs: list[AgentGraph],
    traces: list[TraceRecord],
    out_dir: str | Path,
    model_cfg: ModelKVConfig | None = None,
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "characterization_graph_stats.csv": graph_stats(graphs),
        "characterization_token_stats.csv": token_stats(traces),
        "characterization_kv_stats.csv": kv_stats(graphs, model_cfg),
        "characterization_latency_breakdown.csv": latency_breakdown(traces),
        "characterization_critical_path.csv": critical_path_df(graphs),
    }
    paths: dict[str, Path] = {}
    for name, df in tables.items():
        path = out / name
        df.to_csv(path, index=False)
        paths[name] = path
    return paths
