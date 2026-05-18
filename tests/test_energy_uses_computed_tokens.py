from __future__ import annotations

from waferagent.llm_runner import RunnerConfig
from waferagent.mesh import MeshConfig
from waferagent.simulator import simulate
from waferagent.trace_collector import collect_graph_traces
from waferagent.workloads import WorkloadParams, generate_workload


def _run(shared_prefix_ratio: float):
    graph = generate_workload(
        WorkloadParams(
            workload="long_context_swe_stress",
            job_id=f"energy_{shared_prefix_ratio}",
            num_workers=4,
            input_len=8192,
            output_len=64,
            shared_prefix_ratio=shared_prefix_ratio,
        )
    )
    traces = collect_graph_traces([graph], "energy", RunnerConfig(engine="synthetic"))
    cfg = MeshConfig(4, 4, 16, 1, 1, 50, 1, True, 1, 1, sram_region_rows=2, sram_region_cols=2)
    metrics, *_ = simulate(traces, cfg, ["waferagent_full"], seed=9, duration_source="synthetic")
    return metrics.iloc[0]


def test_energy_tracks_computed_prefill_tokens_not_logical_tokens():
    low = _run(0.0)
    high = _run(0.9)
    assert high["computed_prefill_tokens"] < low["computed_prefill_tokens"]
    assert high["avoided_prefill_tokens"] > low["avoided_prefill_tokens"]
    assert high["compute_energy_j"] < low["compute_energy_j"]
