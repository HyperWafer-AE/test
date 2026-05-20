from waferagent.controlled_workloads import (
    StrictControlledSharedKVConfig,
    generate_strict_controlled_shared_kv_graphs,
    strict_controlled_validation_summary_rows,
)
from waferagent.kv_model import ModelKVConfig
from waferagent.policy_assignment import build_policy_assignments
from waferagent.shared_kv import SharedKVObject


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
    model = ModelKVConfig()
    fallback_users = [f"f{i}" for i in range(320)]
    high_users = [f"h{i}" for i in range(16)]
    objects = [
        SharedKVObject(
            prefix_id="fallback",
            token_len=4096,
            kv_bytes=4096 * model.kv_bytes_per_token,
            logical_users=fallback_users,
            decode_users=fallback_users,
            producer_node=None,
            first_use_step=0,
            last_use_step=1,
            expected_decode_tokens={u: 32 for u in fallback_users},
            expected_decode_steps=32,
            candidate_regions=[],
        ),
        SharedKVObject(
            prefix_id="high",
            token_len=8192,
            kv_bytes=8192 * model.kv_bytes_per_token,
            logical_users=high_users,
            decode_users=high_users,
            producer_node=None,
            first_use_step=0,
            last_use_step=1,
            expected_decode_tokens={u: 512 for u in high_users},
            expected_decode_steps=512,
            candidate_regions=[],
        ),
    ]
    assignments = build_policy_assignments(objects)
    chosen = {a.selected_policy for a in assignments.values()}
    assert "apc_like" in chosen
    assert "waferagent_latency_safe" in chosen
