from waferagent.policy_selector import choose_shared_kv_policy
from waferagent.shared_kv import SharedKVObject


def _obj(token_len: int, users: int, decode_tokens: int) -> SharedKVObject:
    return SharedKVObject(
        prefix_id=f"p{token_len}_{users}_{decode_tokens}",
        token_len=token_len,
        kv_bytes=token_len * 4096,
        logical_users=[f"n{i}" for i in range(users)],
        decode_users=[f"n{i}" for i in range(users)],
        producer_node=None,
        first_use_step=0,
        last_use_step=users,
        expected_decode_tokens={f"n{i}": decode_tokens for i in range(users)},
        expected_decode_steps=decode_tokens,
        candidate_regions=[f"r{i}" for i in range(users)],
    )


def test_policy_selector_falls_back_for_low_opportunity():
    decision = choose_shared_kv_policy(_obj(128, 1, 8))
    assert decision.chosen_policy == "apc_like"
    assert decision.opportunity_score <= 0


def test_policy_selector_enables_waferagent_for_high_opportunity():
    decision = choose_shared_kv_policy(_obj(512, 16, 1024))
    assert decision.chosen_policy == "waferagent_latency_safe"
    assert decision.opportunity_score > 0


def test_policy_selector_falls_back_for_high_queue_risk():
    decision = choose_shared_kv_policy(_obj(32768, 16, 1024))
    assert decision.chosen_policy == "apc_like"
    assert decision.reason == "high_queue_and_sram_risk"
