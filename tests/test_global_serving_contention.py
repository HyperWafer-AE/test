from __future__ import annotations

from waferagent.arrival import ArrivalConfig
from waferagent.global_simulator import simulate_global
from waferagent.llm_runner import RunnerConfig
from waferagent.mesh import MeshConfig
from waferagent.trace_collector import collect_graph_traces
from waferagent.workloads import WorkloadParams, generate_workload


def _traces(num_jobs: int = 8):
    graphs = [
        generate_workload(
            WorkloadParams(
                workload="debate",
                job_id=f"g{i}",
                num_agents=4,
                input_len=2048,
                output_len=64,
                shared_prefix_ratio=0.5,
            )
        )
        for i in range(num_jobs)
    ]
    return collect_graph_traces(graphs, "global_test", RunnerConfig(engine="synthetic"))


def _mesh() -> MeshConfig:
    return MeshConfig(2, 2, 8, 0.5, 0.25, 25, 1, True, 1, 1, sram_region_rows=1, sram_region_cols=1)


def test_jct_and_queue_wait_increase_with_arrival_rate():
    traces = _traces()
    low = simulate_global(
        traces,
        _mesh(),
        ["wafer_naive"],
        ArrivalConfig(mode="poisson", rate_jobs_per_s=0.2, seed=7),
        seed=7,
        duration_source="synthetic",
    )["global_job_metrics"]
    high = simulate_global(
        traces,
        _mesh(),
        ["wafer_naive"],
        ArrivalConfig(mode="poisson", rate_jobs_per_s=1000.0, seed=7),
        seed=7,
        duration_source="synthetic",
    )["global_job_metrics"]
    assert high["queue_wait_ms"].mean() > low["queue_wait_ms"].mean()
    assert high["job_completion_time_ms"].mean() > low["job_completion_time_ms"].mean()
