from __future__ import annotations

from waferagent.arrival import ArrivalConfig
from waferagent.global_simulator import simulate_global
from waferagent.llm_runner import RunnerConfig
from waferagent.mesh import MeshConfig
from waferagent.trace_collector import collect_graph_traces
from waferagent.workloads import WorkloadParams, generate_workload


def test_apc_saves_prefill_but_not_decode_shared_kv_reads():
    graph = generate_workload(
        WorkloadParams(workload="decode_heavy_shared_prefix", job_id="gap", num_agents=8, input_len=8192)
    )
    traces = collect_graph_traces([graph], "gap", RunnerConfig(engine="synthetic"))
    cfg = MeshConfig(4, 4, 16, 1, 1, 50, 1, True, 1, 1, sram_region_rows=2, sram_region_cols=2)
    out = simulate_global(
        traces,
        cfg,
        ["no_cache", "apc_like", "waferagent_full"],
        ArrivalConfig(mode="closed_loop", seed=1),
        seed=1,
    )["global_simulation_summary"]
    no = out[out["baseline"] == "no_cache"].iloc[0]
    apc = out[out["baseline"] == "apc_like"].iloc[0]
    full = out[out["baseline"] == "waferagent_full"].iloc[0]
    assert apc["avoided_prefill_tokens"] > no["avoided_prefill_tokens"]
    assert apc["decode_shared_kv_read_bytes"] == no["decode_shared_kv_read_bytes"]
    assert full["decode_shared_kv_read_bytes"] < apc["decode_shared_kv_read_bytes"]
