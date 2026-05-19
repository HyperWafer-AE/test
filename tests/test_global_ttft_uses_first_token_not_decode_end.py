from __future__ import annotations

from waferagent.arrival import ArrivalConfig
from waferagent.global_simulator import simulate_global
from waferagent.llm_runner import RunnerConfig
from waferagent.mesh import MeshConfig
from waferagent.trace_collector import collect_graph_traces
from waferagent.workloads import WorkloadParams, generate_workload


def test_global_ttft_uses_first_token_not_decode_end():
    graph = generate_workload(
        WorkloadParams(workload="debate", job_id="ttft_job_0", num_agents=2, input_len=1024, output_len=100)
    )
    traces = collect_graph_traces([graph], "ttft", RunnerConfig(engine="synthetic"))
    cfg = MeshConfig(2, 2, 16, 1, 1, 50, 1, True, 1, 1, sram_region_rows=1, sram_region_cols=1)
    result = simulate_global(traces, cfg, ["wafer_naive"], ArrivalConfig(mode="closed_loop"), seed=3)
    metrics = result["global_job_metrics"].iloc[0]
    stages = result["global_stage_schedule"]
    first_decode = stages[stages["stage_type"] == "decode"].sort_values("first_token_ms").iloc[0]
    assert metrics["ttft_ms"] == first_decode["first_token_ms"] - metrics["arrival_ms"]
    assert metrics["ttft_ms"] <= first_decode["end_ms"] - metrics["arrival_ms"]

