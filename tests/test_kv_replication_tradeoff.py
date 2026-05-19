from __future__ import annotations

from waferagent.kv_model import ModelKVConfig
from waferagent.mesh import MeshConfig
from waferagent.shared_kv import extract_shared_kv_objects, plan_shared_kv_replication
from waferagent.workloads import WorkloadParams, generate_workload


def test_replication_tradeoff_not_monotone_free():
    graph = generate_workload(
        WorkloadParams(workload="moa_decode_cohort_stress", job_id="rep", num_agents=8, input_len=4096)
    )
    objects, _ = extract_shared_kv_objects(graph, ModelKVConfig())
    for obj in objects:
        obj.candidate_regions = ["r0c0", "r0c1", "r1c0", "r1c1"]
    cfg = MeshConfig(8, 8, 8, 1, 1, 50, 1, True, 1, 1, sram_region_rows=4, sram_region_cols=4)
    _, no = plan_shared_kv_replication([o for o in objects], "no_replication", cfg)
    _, all_rep = plan_shared_kv_replication([o for o in objects], "replicate_all", cfg)
    assert all_rep["replica_bytes_total"] > no["replica_bytes_total"]
    assert all_rep["saved_mesh_traffic_bytes"] > no["saved_mesh_traffic_bytes"]
