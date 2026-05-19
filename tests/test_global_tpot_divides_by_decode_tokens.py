from __future__ import annotations

from waferagent.arrival import ArrivalConfig
from waferagent.global_simulator import simulate_global
from waferagent.llm_runner import RunnerConfig
from waferagent.mesh import MeshConfig
from waferagent.trace_collector import collect_graph_traces
from waferagent.workloads import WorkloadParams, generate_workload


def test_global_tpot_divides_by_decode_tokens():
    graph = generate_workload(
        WorkloadParams(workload="debate", job_id="tpot_job_0", num_agents=2, input_len=1024, output_len=100)
    )
    traces = collect_graph_traces([graph], "tpot", RunnerConfig(engine="synthetic"))
    cfg = MeshConfig(2, 2, 16, 1, 1, 50, 1, True, 1, 1, sram_region_rows=1, sram_region_cols=1)
    result = simulate_global(traces, cfg, ["wafer_naive"], ArrivalConfig(mode="closed_loop"), seed=4)
    metrics = result["global_job_metrics"].iloc[0]
    stages = result["global_stage_schedule"]
    decode = stages[stages["stage_type"] == "decode"]
    expected = decode["decode_active_ms"].sum() / decode["decode_tokens"].sum()
    assert metrics["tpot_ms"] == expected
    assert metrics["tpot_ms"] < decode["decode_active_ms"].sum()

