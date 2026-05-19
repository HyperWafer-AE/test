from __future__ import annotations

from pathlib import Path
import json
import time
from typing import Iterable

from waferagent.graph_ir import AgentGraph
from waferagent.llm_runner import RunnerConfig, make_runner
from waferagent.trace_schema import TraceRecord, read_traces, write_traces


def _topological_layers(graph: AgentGraph) -> list[list[str]]:
    remaining = set(graph.nodes)
    done: set[str] = set()
    layers: list[list[str]] = []
    while remaining:
        ready = sorted(n for n in remaining if all(dep in done for dep in graph.nodes[n].deps))
        if not ready:
            raise ValueError(f"Graph has unsatisfied dependencies or a cycle: {graph.graph_id}")
        layers.append(ready)
        done.update(ready)
        remaining.difference_update(ready)
    return layers


def collect_graph_traces(
    graphs: Iterable[AgentGraph],
    run_id: str,
    runner_config: RunnerConfig,
    out_jsonl: str | Path | None = None,
    resume: bool = False,
    stop_after_minutes: float | None = None,
) -> list[TraceRecord]:
    runner = make_runner(runner_config)
    traces: list[TraceRecord] = []
    out_path = Path(out_jsonl) if out_jsonl is not None else None
    completed_jobs: set[str] = set()
    if resume and out_path is not None and out_path.exists():
        traces = read_traces(out_path)
        completed_jobs = {tr.job_id for tr in traces}
    start = time.time()
    for graph in graphs:
        if graph.graph_id in completed_jobs:
            continue
        graph_rows: list[TraceRecord] = []
        if runner_config.engine == "vllm" and hasattr(runner, "run_nodes"):
            for layer in _topological_layers(graph):
                graph_rows.extend(runner.run_nodes(run_id, graph.workload, [graph.nodes[n] for n in layer]))
        else:
            for node_id in graph.topological_order():
                graph_rows.append(runner.run_node(run_id, graph.workload, graph.nodes[node_id]))
        traces.extend(graph_rows)
        if out_path is not None:
            mode = "a" if out_path.exists() and (resume or completed_jobs or len(traces) > len(graph_rows)) else "w"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open(mode, encoding="utf-8") as f:
                for tr in graph_rows:
                    f.write(json.dumps(tr.to_dict(), sort_keys=True) + "\n")
        if stop_after_minutes and (time.time() - start) / 60.0 >= stop_after_minutes:
            break
    if out_jsonl is not None and out_path is not None and not out_path.exists():
        write_traces(out_jsonl, traces)
    return traces


def traces_by_job(traces: Iterable[TraceRecord]) -> dict[str, list[TraceRecord]]:
    jobs: dict[str, list[TraceRecord]] = {}
    for tr in traces:
        jobs.setdefault(tr.job_id, []).append(tr)
    for rows in jobs.values():
        rows.sort(key=lambda r: (r.round_id, r.node_id))
    return jobs
