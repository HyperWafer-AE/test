from __future__ import annotations

from pathlib import Path
from typing import Iterable

from waferagent.graph_ir import AgentGraph
from waferagent.llm_runner import RunnerConfig, make_runner
from waferagent.trace_schema import TraceRecord, write_traces


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
) -> list[TraceRecord]:
    runner = make_runner(runner_config)
    traces: list[TraceRecord] = []
    for graph in graphs:
        if runner_config.engine == "vllm" and hasattr(runner, "run_nodes"):
            for layer in _topological_layers(graph):
                traces.extend(runner.run_nodes(run_id, graph.workload, [graph.nodes[n] for n in layer]))
        else:
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
