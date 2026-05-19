from __future__ import annotations

from waferagent.arrival import ArrivalConfig
from waferagent.global_simulator import simulate_global
from waferagent.llm_runner import RunnerConfig
from waferagent.mesh import MeshConfig
from waferagent.trace_collector import collect_graph_traces
from waferagent.workloads import WorkloadParams, generate_workload


def test_decode_cohort_does_not_reuse_attention_outputs():
    graph = generate_workload(
        WorkloadParams(workload="decode_heavy_shared_prefix", job_id="no_output_reuse_job_0", num_agents=4, input_len=4096)
    )
    traces = collect_graph_traces([graph], "no_output_reuse", RunnerConfig(engine="synthetic"))
    cfg = MeshConfig(16, 16, 16, 10, 10, 50, 1, True, 1, 1, sram_region_rows=2, sram_region_cols=2)
    stages = simulate_global(
        traces,
        cfg,
        ["waferagent_full"],
        ArrivalConfig(mode="closed_loop"),
        seed=10,
    )["global_stage_schedule"]
    cohorted = stages[(stages["stage_type"] == "decode") & (stages["cohort_id"] != "")]
    assert not cohorted.empty
    assert (cohorted["decode_tokens"] > 0).all()
    assert (cohorted["decode_active_ms"] > 0).all()
