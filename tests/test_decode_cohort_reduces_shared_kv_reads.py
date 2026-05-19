from __future__ import annotations

from waferagent.cohort_scheduler import CohortConfig, form_decode_cohorts
from waferagent.kv_model import ModelKVConfig
from waferagent.shared_attention_cost import estimate_shared_attention_cost
from waferagent.shared_kv import extract_shared_kv_objects
from waferagent.stage_ir import build_stages
from waferagent.llm_runner import RunnerConfig
from waferagent.trace_collector import collect_graph_traces
from waferagent.workloads import WorkloadParams, generate_workload


def test_decode_cohort_reduces_shared_kv_reads_without_reusing_outputs():
    graph = generate_workload(
        WorkloadParams(workload="decode_heavy_shared_prefix", job_id="reads", num_agents=8, input_len=8192)
    )
    traces = collect_graph_traces([graph], "reads", RunnerConfig(engine="synthetic"))
    stages = build_stages(graph, traces)
    objects, _ = extract_shared_kv_objects(graph, ModelKVConfig())
    cohorts, _ = form_decode_cohorts(
        stages,
        objects,
        cfg=CohortConfig(min_expected_saved_kv_bytes=1),
    )
    no = estimate_shared_attention_cost(objects, [])
    yes = estimate_shared_attention_cost(objects, cohorts)
    assert yes["decode_shared_kv_read_bytes"] < no["decode_shared_kv_read_bytes"]
    assert yes["decode_query_transfer_bytes"] >= 0
