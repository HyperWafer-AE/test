from __future__ import annotations

from waferagent.prefix_extension_timer import PrefixExtensionTimingResult


def test_prefix_extension_timing_result_records_all_required_fields():
    row = PrefixExtensionTimingResult(
        prefix_len=512,
        private_len=64,
        output_len=16,
        batch_size=2,
        rep=0,
        full_prefill_ms=10.0,
        prefix_prefill_ms=8.0,
        extend_prefill_ms=2.5,
        decode_ms=4.0,
        decode_tpot_ms=0.25,
        peak_gpu_mem_bytes=123,
        oom=False,
        error="",
    ).to_dict()
    assert row["extend_prefill_ms"] <= row["full_prefill_ms"]
    assert row["decode_tpot_ms"] == row["decode_ms"] / row["output_len"]
    assert not row["oom"]
