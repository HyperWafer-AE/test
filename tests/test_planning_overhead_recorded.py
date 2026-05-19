from __future__ import annotations

from waferagent.arrival import ArrivalConfig
from waferagent.global_simulator import simulate_global
from waferagent.llm_runner import RunnerConfig
from waferagent.mesh import MeshConfig
from waferagent.trace_collector import collect_graph_traces
from waferagent.workloads import WorkloadParams, generate_workload


def test_planning_overhead_recorded():
    graph = generate_workload(
        WorkloadParams(workload="moa_decode_cohort_stress", job_id="overhead_job_0", num_agents=4, input_len=4096)
    )
    traces = collect_graph_traces([graph], "overhead", RunnerConfig(engine="synthetic"))
    cfg = MeshConfig(4, 4, 16, 1, 1, 50, 1, True, 1, 1, sram_region_rows=2, sram_region_cols=2)
    result = simulate_global(traces, cfg, ["waferagent_full"], ArrivalConfig(mode="closed_loop"), seed=6)
    overhead = result["planning_overhead_summary"].iloc[0]
    assert overhead["total_runtime_overhead_ms"] > 0
    assert overhead["shared_kv_extraction_overhead_ms"] >= 0
    assert overhead["decode_cohort_planning_overhead_ms"] >= 0
    assert "overhead_fraction_of_jct" in overhead

