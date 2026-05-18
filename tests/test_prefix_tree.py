from __future__ import annotations

from waferagent.kv_model import ModelKVConfig, build_prefix_blocks
from waferagent.workloads import WorkloadParams, generate_workload


def test_prefix_blocks_record_reuse_count():
    graph = generate_workload(WorkloadParams(workload="debate", job_id="prefix", num_agents=4))
    blocks = build_prefix_blocks(graph.nodes.values(), ModelKVConfig())
    assert blocks
    assert max(b.ref_count for b in blocks.values()) > 1
