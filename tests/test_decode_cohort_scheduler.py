from __future__ import annotations

from waferagent.cohort_scheduler import CohortConfig, form_decode_cohorts
from waferagent.kv_model import ModelKVConfig
from waferagent.shared_kv import extract_shared_kv_objects
from waferagent.stage_ir import build_stages
from waferagent.llm_runner import RunnerConfig
from waferagent.trace_collector import collect_graph_traces
from waferagent.workloads import WorkloadParams, generate_workload


def test_decode_cohort_scheduler_obeys_wait_and_groups_shared_kv():
    graph = generate_workload(
        WorkloadParams(workload="decode_heavy_shared_prefix", job_id="cohort", num_agents=4, input_len=4096)
    )
    traces = collect_graph_traces([graph], "cohort", RunnerConfig(engine="synthetic"))
    stages = build_stages(graph, traces)
    objects, _ = extract_shared_kv_objects(graph, ModelKVConfig())
    ready = {sid: 0.0 for sid in stages}
    cohorts, stats = form_decode_cohorts(
        stages,
        objects,
        ready_times=ready,
        cfg=CohortConfig(min_shared_prefix_tokens=1024, min_expected_saved_kv_bytes=1),
    )
    assert cohorts
    assert stats["num_decode_cohorts"] > 0
    assert all(len(c.node_ids) >= 2 for c in cohorts)
    assert all(c.max_wait_ms <= 2.0 for c in cohorts)
