from __future__ import annotations

from waferagent.baselines import get_baseline
from waferagent.llm_runner import RunnerConfig
from waferagent.mesh import Mesh, MeshConfig
from waferagent.placement import make_placement
from waferagent.scheduler import schedule_graph
from waferagent.trace_collector import collect_graph_traces
from waferagent.workloads import WorkloadParams, generate_workload


def test_scheduler_topological_legality():
    graph = generate_workload(
        WorkloadParams(workload="debate", job_id="j", num_agents=2, input_len=64, output_len=8)
    )
    traces = collect_graph_traces([graph], "test", RunnerConfig(engine="synthetic"))
    cfg = MeshConfig(
        rows=4,
        cols=4,
        tile_sram_mb=64,
        tile_prefill_tflops=1,
        tile_decode_tflops=1,
        link_bandwidth_GBps=100,
        link_latency_us=1,
        multicast_supported=True,
        energy_per_flop_pJ=1,
        energy_per_byte_pJ=1,
    )
    placements = make_placement("round_robin", graph, cfg)
    schedule = schedule_graph(graph, traces, Mesh(cfg), placements, get_baseline("wafer_naive"))
    ends = {s.node_id: s.end_ms for s in schedule}
    starts = {s.node_id: s.start_ms for s in schedule}
    for node in graph.nodes.values():
        for dep in node.deps:
            assert starts[node.node_id] >= ends[dep]
