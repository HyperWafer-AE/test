from __future__ import annotations

from waferagent.arrival import ArrivalConfig
from waferagent.global_simulator import simulate_global
from waferagent.llm_runner import RunnerConfig
from waferagent.mesh import MeshConfig
from waferagent.trace_collector import collect_graph_traces
from waferagent.workloads import WorkloadParams, generate_workload


def test_shared_kv_residency_routes_remote_when_no_replica():
    graph = generate_workload(
        WorkloadParams(workload="moa_decode_cohort_stress", job_id="remote_kv_job_0", num_agents=8, input_len=8192)
    )
    traces = collect_graph_traces([graph], "remote_kv", RunnerConfig(engine="synthetic"))
    cfg = MeshConfig(16, 16, 1024, 10, 10, 25, 1, True, 1, 1, sram_region_rows=1, sram_region_cols=1)
    result = simulate_global(
        traces,
        cfg,
        ["no_shared_kv_replication", "waferagent_full"],
        ArrivalConfig(mode="closed_loop"),
        seed=8,
    )
    mesh_events = result["mesh_link_events"]
    remote = mesh_events[
        (mesh_events["baseline"] == "no_shared_kv_replication")
        & (mesh_events["traffic_source"] == "decode_shared_kv_remote_read")
    ]
    assert not remote.empty
