from __future__ import annotations

from waferagent.llm_runner import RunnerConfig
from waferagent.stage_ir import build_stages
from waferagent.trace_collector import collect_graph_traces
from waferagent.workloads import WorkloadParams, generate_workload


def test_llm_call_splits_prefill_and_decode():
    graph = generate_workload(WorkloadParams(workload="debate", job_id="stage", num_agents=1))
    traces = collect_graph_traces([graph], "test", RunnerConfig(engine="synthetic"))
    stages = build_stages(graph, traces)
    assert any(s.stage_type == "prefill" for s in stages.values())
    assert any(s.stage_type == "decode" for s in stages.values())
    for sid, stage in stages.items():
        if stage.stage_type == "decode":
            assert stage.deps == [sid.replace(".decode", ".prefill")]
