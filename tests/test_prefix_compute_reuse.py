from __future__ import annotations

from waferagent.llm_runner import RunnerConfig
from waferagent.mesh import MeshConfig
from waferagent.simulator import simulate
from waferagent.trace_collector import collect_graph_traces
from waferagent.workloads import WorkloadParams, generate_workload


def _jct_for_ratio(ratio: float):
    graph = generate_workload(
        WorkloadParams(
            workload="long_context_swe_stress",
            job_id=f"prefix_{ratio}",
            num_workers=8,
            input_len=8192,
            output_len=64,
            shared_prefix_ratio=ratio,
        )
    )
    traces = collect_graph_traces([graph], "test", RunnerConfig(engine="synthetic"))
    cfg = MeshConfig(8, 8, 512, 1, 1, 50, 1, True, 1, 1)
    metrics, *_ = simulate(traces, cfg, ["waferagent_full"], duration_source="synthetic")
    return metrics.iloc[0]


def test_prefix_ratio_reduces_shared_prefill_compute_and_jct():
    low = _jct_for_ratio(0.0)
    high = _jct_for_ratio(0.9)
    assert high["shared_prefill_tokens_saved"] > low["shared_prefill_tokens_saved"]
    assert high["shared_prefill_compute_ms_saved"] > low["shared_prefill_compute_ms_saved"]
    assert high["job_completion_time_ms"] < low["job_completion_time_ms"]
