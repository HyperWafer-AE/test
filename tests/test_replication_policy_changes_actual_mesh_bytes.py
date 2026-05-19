from __future__ import annotations

from waferagent.arrival import ArrivalConfig
from waferagent.global_simulator import simulate_global
from waferagent.llm_runner import RunnerConfig
from waferagent.mesh import MeshConfig
from waferagent.trace_collector import collect_graph_traces
from waferagent.workloads import WorkloadParams, generate_workload


def test_replication_policy_changes_actual_mesh_bytes():
    graph = generate_workload(
        WorkloadParams(workload="moa_decode_cohort_stress", job_id="repl_delta_job_0", num_agents=8, input_len=8192)
    )
    traces = collect_graph_traces([graph], "repl_delta", RunnerConfig(engine="synthetic"))
    cfg = MeshConfig(16, 16, 2048, 10, 10, 25, 1, True, 1, 1, sram_region_rows=1, sram_region_cols=1)
    summary = simulate_global(
        traces,
        cfg,
        ["no_shared_kv_replication", "waferagent_full"],
        ArrivalConfig(mode="closed_loop"),
        seed=9,
    )["global_simulation_summary"].set_index("baseline")
    assert summary.loc["no_shared_kv_replication", "mesh_total_traffic_bytes"] != summary.loc["waferagent_full", "mesh_total_traffic_bytes"]
