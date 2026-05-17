from __future__ import annotations

from pathlib import Path
from typing import Iterable

from waferagent.graph_ir import AgentGraph
from waferagent.llm_runner import RunnerConfig, make_runner
from waferagent.trace_schema import TraceRecord, write_traces


def collect_graph_traces(
    graphs: Iterable[AgentGraph],
    run_id: str,
    runner_config: RunnerConfig,
    out_jsonl: str | Path | None = None,
) -> list[TraceRecord]:
    runner = make_runner(runner_config)
    traces: list[TraceRecord] = []
    for graph in graphs:
        for node_id in graph.topological_order():
            traces.append(runner.run_node(run_id, graph.workload, graph.nodes[node_id]))
    if out_jsonl is not None:
        write_traces(out_jsonl, traces)
    return traces


def traces_by_job(traces: Iterable[TraceRecord]) -> dict[str, list[TraceRecord]]:
    jobs: dict[str, list[TraceRecord]] = {}
    for tr in traces:
        jobs.setdefault(tr.job_id, []).append(tr)
    for rows in jobs.values():
        rows.sort(key=lambda r: (r.round_id, r.node_id))
    return jobs
