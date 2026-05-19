from __future__ import annotations

from waferagent.shared_attention_cost import estimate_shared_attention_cost
from waferagent.shared_kv import SharedKVObject


def test_shared_attention_cost_private_bytes_use_token_bytes_not_bytes_minus_tokens():
    obj = SharedKVObject(
        prefix_id="p",
        token_len=128,
        kv_bytes=128_000,
        logical_users=["n0"],
        decode_users=["n0"],
        producer_node=None,
        first_use_step=0,
        last_use_step=1,
        expected_decode_tokens={"n0": 10},
        expected_decode_steps=10,
        candidate_regions=["r0c0"],
    )
    stats = estimate_shared_attention_cost(
        [obj],
        private_tokens_by_node={"n0": 7},
        output_tokens_by_node={"n0": 10},
        kv_bytes_per_token=1000,
    )
    assert stats["decode_private_kv_read_bytes"] == 7 * 10 * 1000
    assert stats["decode_private_kv_read_bytes"] != 10 * (obj.kv_bytes - obj.token_len)

