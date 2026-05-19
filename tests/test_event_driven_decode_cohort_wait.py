from __future__ import annotations

from waferagent.arrival import ArrivalConfig
from waferagent.global_simulator import simulate_global
from waferagent.llm_runner import RunnerConfig
from waferagent.mesh import MeshConfig
from waferagent.trace_collector import collect_graph_traces
from waferagent.workloads import WorkloadParams, generate_workload


def test_event_driven_decode_cohort_records_wait_and_reduces_reads():
    graph = generate_workload(
        WorkloadParams(workload="decode_heavy_shared_prefix", job_id="cohort_event_job_0", num_agents=8, input_len=8192)
    )
    traces = collect_graph_traces([graph], "cohort_event", RunnerConfig(engine="synthetic"))
    cfg = MeshConfig(16, 16, 16, 10, 10, 50, 1, True, 1, 1, sram_region_rows=2, sram_region_cols=2)
    result = simulate_global(
        traces,
        cfg,
        ["apc_like", "waferagent_full"],
        ArrivalConfig(mode="closed_loop"),
        seed=7,
    )
    cohorts = result["decode_cohorts"]
    assert not cohorts.empty
    assert cohorts["event_driven"].all()
    summary = result["global_simulation_summary"].set_index("baseline")
    assert summary.loc["waferagent_full", "decode_shared_kv_read_bytes"] < summary.loc["apc_like", "decode_shared_kv_read_bytes"]
    assert result["global_stage_schedule"]["cohort_wait_ms"].max() <= 2.0
