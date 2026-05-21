from __future__ import annotations

from kvring.config import ModelConfig


def test_llama31_8b_gqa_kv_token_bytes() -> None:
    model = ModelConfig(layers=32, kv_heads=8, head_dim=128, dtype_bytes=2)
    assert model.kv_token_bytes == 2 * 32 * 8 * 128 * 2
    assert model.kv_token_bytes == 128 * 1024
