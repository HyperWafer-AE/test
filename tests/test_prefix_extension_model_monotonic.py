from __future__ import annotations

from waferagent.prefix_extension_cost_model import PrefixExtensionCostModel


def test_prefix_extension_cost_is_bounded_and_monotonic():
    model = PrefixExtensionCostModel(
        full_prefill_coef=[0.1, 0.01, 0.0, 0.0, 0.0],
        extend_prefill_coef=[0.2, 0.001, 0.02, 0.0, 0.0, 0.0, 0.0],
        decode_tpot_coef=[0.1, 0.0, 0.0, 0.0, 0.0],
    )
    short = model.extend_prefill_ms(512, 64)
    longer_private = model.extend_prefill_ms(512, 256)
    longer_prefix = model.extend_prefill_ms(2048, 256)
    full = model.full_prefill_ms(512 + 64)
    private_only = model.full_prefill_ms(64)
    assert 0.0 <= short <= full
    assert short > private_only * 0.05
    assert longer_private >= short
    assert longer_prefix >= longer_private
