from waferagent.arrival import ArrivalConfig
from waferagent.cohort_admission import CohortAdmissionConfig, evaluate_cohort_candidate
from waferagent.llm_runner import RunnerConfig
from waferagent.mesh import MeshConfig
from waferagent.shared_kv import SharedKVObject
from waferagent.stage_ir import Stage
from waferagent.trace_collector import collect_graph_traces
from waferagent.global_simulator import simulate_global
from waferagent.workloads import WorkloadParams, generate_workload


def _stage(stage_id: str, output_tokens: int = 128) -> Stage:
    return Stage(
        stage_id=stage_id,
        parent_node_id=stage_id,
        job_id="j",
        stage_type="decode",
        deps=[],
        input_tokens=2048,
        output_tokens=output_tokens,
        duration_ms=1.0,
        tile_pool="decode",
        shared_prefix_ids=["p"],
        shared_prefix_token_len=2048,
        kv_bytes_estimated=1,
    )


def test_cohort_admission_rejects_bad_critical_wait():
    obj = SharedKVObject(
        prefix_id="p",
        token_len=2048,
        kv_bytes=1_000_000,
        logical_users=["a", "b"],
        decode_users=["a", "b"],
        producer_node=None,
        first_use_step=0,
        last_use_step=1,
        expected_decode_tokens={"a": 128, "b": 128},
        expected_decode_steps=128,
        candidate_regions=["r0c0"],
    )
    decision = evaluate_cohort_candidate(
        obj,
        [_stage("a"), _stage("b")],
        ready_times=[0.0, 10.0],
        criticalities=[1.0, 0.0],
        bytes_per_ms=1e9,
        cfg=CohortAdmissionConfig(max_critical_wait_ms=0.1),
    )
    assert not decision.accepted
    assert decision.reason == "critical_path_wait"


def test_cohort_admission_keeps_byte_saving_and_latency_safe():
    graphs = [
        generate_workload(
            WorkloadParams(
                workload="decode_heavy_shared_prefix",
                job_id=f"cohort_admit_job_{i}",
                num_agents=8,
                input_len=8192,
                output_len=512,
                seed=i,
            )
        )
        for i in range(8)
    ]
    traces = collect_graph_traces(graphs, "cohort_admission", RunnerConfig(engine="synthetic"))
    result = simulate_global(
        traces,
        MeshConfig.from_yaml("configs/wafer/wse_like.yaml"),
        ["waferagent_full", "no_shared_kv_decode_cohort"],
        ArrivalConfig(mode="burst", rate_jobs_per_s=16, seed=3),
        seed=3,
    )
    summary = result["global_simulation_summary"].set_index("baseline")
    full = summary.loc["waferagent_full"]
    no = summary.loc["no_shared_kv_decode_cohort"]
    assert full["decode_shared_kv_read_bytes"] < no["decode_shared_kv_read_bytes"]
    assert full["jct_p99_ms"] <= no["jct_p99_ms"] * 1.05
    assert not result["cohort_admission_decisions"].empty
