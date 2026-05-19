from __future__ import annotations

from waferagent.kv_model import ModelKVConfig
from waferagent.shared_kv import extract_shared_kv_objects
from waferagent.workloads import WorkloadParams, generate_workload


def test_shared_kv_object_extraction_strict_prefix_only():
    graph = generate_workload(
        WorkloadParams(workload="decode_heavy_shared_prefix", job_id="skv", num_agents=4, input_len=2048)
    )
    objects, stats = extract_shared_kv_objects(graph, ModelKVConfig())
    assert objects
    assert stats.num_shared_kv_objects >= 1
    assert stats.safe_shared_prefix_tokens > 0
    assert stats.unsafe_reuse_skipped_tokens == 0
    assert max(len(o.decode_users) for o in objects) >= 2
    assert max(o.expected_decode_kv_read_bytes_without_cohort for o in objects) > 0
