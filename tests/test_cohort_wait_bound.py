from __future__ import annotations

from waferagent.cohort_scheduler import CohortConfig, form_decode_cohorts
from waferagent.kv_model import ModelKVConfig
from waferagent.shared_kv import extract_shared_kv_objects
from waferagent.stage_ir import build_stages
from waferagent.llm_runner import RunnerConfig
from waferagent.trace_collector import collect_graph_traces
from waferagent.workloads import WorkloadParams, generate_workload


def test_cohort_wait_bound_prevents_far_apart_grouping():
    graph = generate_workload(
        WorkloadParams(workload="decode_heavy_shared_prefix", job_id="wait", num_agents=4, input_len=4096)
    )
    traces = collect_graph_traces([graph], "wait", RunnerConfig(engine="synthetic"))
    stages = build_stages(graph, traces)
    objects, _ = extract_shared_kv_objects(graph, ModelKVConfig())
    decode_ids = [sid for sid, st in stages.items() if st.stage_type == "decode"]
    ready = {sid: i * 100.0 for i, sid in enumerate(sorted(decode_ids))}
    cohorts, _ = form_decode_cohorts(
        stages,
        objects,
        ready_times=ready,
        cfg=CohortConfig(max_wait_ms=0.1, min_expected_saved_kv_bytes=1),
    )
    assert not cohorts
