from __future__ import annotations

from waferagent.arrival import ArrivalConfig
from waferagent.global_simulator import simulate_global
from waferagent.llm_runner import RunnerConfig
from waferagent.mesh import MeshConfig
from waferagent.trace_collector import collect_graph_traces
from waferagent.workloads import WorkloadParams, generate_workload


def test_prefill_ms_saved_is_not_token_count_alias():
    graph = generate_workload(
        WorkloadParams(workload="decode_heavy_shared_prefix", job_id="gap_units_job_0", num_agents=4, input_len=4096)
    )
    traces = collect_graph_traces([graph], "gap_units", RunnerConfig(engine="synthetic"))
    cfg = MeshConfig(4, 4, 16, 1, 1, 50, 1, True, 1, 1, sram_region_rows=2, sram_region_cols=2)
    summary = simulate_global(
        traces,
        cfg,
        ["apc_like"],
        ArrivalConfig(mode="closed_loop"),
        seed=5,
    )["global_simulation_summary"].iloc[0]
    assert summary["shared_prefill_compute_ms_saved"] > 0
    assert summary["avoided_prefill_tokens"] > 0
    assert summary["shared_prefill_compute_ms_saved"] != summary["avoided_prefill_tokens"]

