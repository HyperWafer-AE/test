from __future__ import annotations

from waferagent.arrival import ArrivalConfig
from waferagent.global_simulator import simulate_global
from waferagent.llm_runner import RunnerConfig
from waferagent.mesh import MeshConfig
from waferagent.trace_collector import collect_graph_traces
from waferagent.workloads import WorkloadParams, generate_workload


def _run(rate: float):
    graphs = [
        generate_workload(
            WorkloadParams(
                workload="moa",
                job_id=f"slo{i}",
                num_layers=3,
                width=4,
                input_len=1536,
                output_len=64,
                shared_prefix_ratio=0.5,
            )
        )
        for i in range(10)
    ]
    traces = collect_graph_traces(graphs, "slo", RunnerConfig(engine="synthetic"))
    cfg = MeshConfig(2, 2, 8, 0.5, 0.25, 25, 1, True, 1, 1, sram_region_rows=1, sram_region_cols=1)
    return simulate_global(
        traces,
        cfg,
        ["wafer_naive"],
        ArrivalConfig(mode="poisson", rate_jobs_per_s=rate, seed=5),
        seed=5,
        duration_source="synthetic",
        slo_jct_ms=[1000.0],
    )["slo_goodput"]["slo_goodput_jobs_per_s"].iloc[0]


def test_slo_goodput_saturates_or_drops_at_high_arrival_rate():
    low = float(_run(0.2))
    mid = float(_run(20.0))
    high = float(_run(1000.0))
    assert mid >= low
    assert high <= mid * 1.5
