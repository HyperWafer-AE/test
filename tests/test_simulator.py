from __future__ import annotations

from waferagent.llm_runner import RunnerConfig
from waferagent.mesh import MeshConfig
from waferagent.simulator import simulate
from waferagent.trace_collector import collect_graph_traces
from waferagent.workloads import WorkloadParams, generate_workload


def test_simulator_deterministic_toy_graph():
    graph = generate_workload(
        WorkloadParams(workload="planner_worker_tool", job_id="j", num_workers=2, input_len=64, output_len=8)
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
    m1, s1 = simulate(traces, cfg, ["wafer_naive", "waferagent_full"], seed=1)
    m2, s2 = simulate(traces, cfg, ["wafer_naive", "waferagent_full"], seed=1)
    assert m1["job_completion_time_ms"].tolist() == m2["job_completion_time_ms"].tolist()
    assert not s1.empty and not s2.empty
    assert set(m1["baseline"]) == {"wafer_naive", "waferagent_full"}
