from __future__ import annotations

from waferagent.arrival import ArrivalConfig
from waferagent.global_simulator import simulate_global
from waferagent.llm_runner import RunnerConfig
from waferagent.mesh import MeshConfig
from waferagent.trace_collector import collect_graph_traces
from waferagent.workloads import WorkloadParams, generate_workload


def test_cross_job_shared_prefix_compute_hit_is_recorded():
    graphs = [
        generate_workload(
            WorkloadParams(
                workload="debate",
                job_id=f"reuse{i}",
                num_agents=2,
                input_len=512,
                output_len=16,
                shared_prefix_ratio=0.75,
            )
        )
        for i in range(2)
    ]
    traces = collect_graph_traces(graphs, "reuse", RunnerConfig(engine="synthetic"))
    cfg = MeshConfig(4, 4, 16, 1, 1, 50, 1, True, 1, 1, sram_region_rows=2, sram_region_cols=2)
    result = simulate_global(
        traces,
        cfg,
        ["waferagent_full"],
        ArrivalConfig(mode="closed_loop", seed=3),
        seed=3,
        duration_source="synthetic",
    )
    summary = result["global_simulation_summary"]
    assert float(summary["cross_job_prefix_hit_rate"].iloc[0]) > 0.0
    assert float(summary["cross_job_prefix_compute_hits"].iloc[0]) > 0.0
