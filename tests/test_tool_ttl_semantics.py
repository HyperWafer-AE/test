from __future__ import annotations

from waferagent.baselines import get_baseline
from waferagent.llm_runner import RunnerConfig
from waferagent.mesh import MeshConfig
from waferagent.simulator import simulate
from waferagent.trace_collector import collect_graph_traces
from waferagent.workloads import WorkloadParams, generate_workload


def test_tool_ttl_does_not_change_tool_latency_but_changes_resume_cost():
    graph = generate_workload(
        WorkloadParams(
            workload="tool_pause_resume_loop",
            job_id="toolttl",
            num_workers=2,
            num_tools_per_worker=2,
            mean_tool_latency_ms=1000,
            input_len=4096,
        )
    )
    traces = collect_graph_traces([graph], "test", RunnerConfig(engine="synthetic"))
    cfg = MeshConfig(4, 4, 64, 1, 1, 50, 1, True, 1, 1)
    metrics, _, _, _, _ = simulate(traces, cfg, ["wafer_naive", "waferagent_full"], neutral_multipliers=True)
    naive = metrics[metrics["baseline"] == "wafer_naive"].iloc[0]
    full = metrics[metrics["baseline"] == "waferagent_full"].iloc[0]
    assert naive["tool_pause_ms"] == full["tool_pause_ms"]
    assert full["resume_reload_bytes"] <= naive["resume_reload_bytes"]
