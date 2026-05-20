from waferagent.controlled_workloads import (
    StrictControlledSharedKVConfig,
    generate_strict_controlled_shared_kv_graphs,
    strict_controlled_validation_summary_rows,
)
from waferagent.kv_model import ModelKVConfig
from waferagent.policy_assignment import build_policy_assignments
from waferagent.shared_kv import extract_shared_kv_objects


def test_controlled_validation_summary_is_nonempty_and_passes():
    cfg = StrictControlledSharedKVConfig(
        num_jobs=8,
        reuse_group_size=4,
        shared_prefix_tokens=2048,
        private_suffix_tokens=128,
        decode_tokens=64,
        num_agents_per_job=4,
        seed=3,
    )
    graphs = generate_strict_controlled_shared_kv_graphs(cfg)
    rows = strict_controlled_validation_summary_rows(graphs, cfg)
    assert rows
    assert rows[0]["num_nodes_checked"] == 32
    assert rows[0]["unique_prefixes_observed"] == rows[0]["expected_unique_prefixes"]
    assert rows[0]["pass"] is True


def test_policy_assignments_can_mix_apc_and_waferagent():
    cfg_fallback = StrictControlledSharedKVConfig(4, 4, 4096, 128, 32, 16, seed=1)
    cfg_high = StrictControlledSharedKVConfig(4, 4, 8192, 128, 512, 4, seed=2)
    objects = []
    for cfg in [cfg_fallback, cfg_high]:
        for graph in generate_strict_controlled_shared_kv_graphs(cfg):
            extracted, _ = extract_shared_kv_objects(graph, ModelKVConfig(), {}, None)
            objects.extend(extracted)
    assignments = build_policy_assignments(objects)
    chosen = {a.selected_policy for a in assignments.values()}
    assert "apc_like" in chosen
    assert "waferagent_latency_safe" in chosen
