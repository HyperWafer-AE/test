from __future__ import annotations

import numpy as np

from kvring.attention_math import (
    error_metrics,
    fuse_shared_and_suffix,
    finalize_stats,
    full_attention_reference,
    merge_stats,
    partial_attention_stats,
    quantize_bf16,
    ring_reduce,
    tree_reduce,
)


def test_blockwise_online_softmax_equals_full_softmax() -> None:
    rng = np.random.default_rng(7)
    q = rng.normal(size=(8, 32)).astype(np.float32)
    k = rng.normal(size=(257, 32)).astype(np.float32)
    v = rng.normal(size=(257, 32)).astype(np.float32)
    for query_tile_size in [1, 2, 4, 8]:
        q_tile = q[:query_tile_size]
        for num_shards in [1, 2, 4, 8, 16]:
            for kv_precision in ["fp32", "bf16", "fp16"]:
                if kv_precision == "bf16":
                    k_ref, v_ref = quantize_bf16(k), quantize_bf16(v)
                elif kv_precision == "fp16":
                    k_ref = k.astype(np.float16).astype(np.float32)
                    v_ref = v.astype(np.float16).astype(np.float32)
                else:
                    k_ref, v_ref = k, v
                ref = full_attention_reference(q_tile.astype(np.float64), k_ref, v_ref)
                states = [
                    partial_attention_stats(
                        q_tile,
                        k[block],
                        v[block],
                        state_precision="fp32",
                        kv_storage_precision=kv_precision,
                    )
                    for block in np.array_split(np.arange(k.shape[0]), num_shards)
                ]
                got = finalize_stats(ring_reduce(states, state_precision="fp32"))
                np.testing.assert_allclose(got, ref, rtol=3e-5, atol=3e-5)


def test_shared_prefix_state_plus_private_suffix_fusion_equals_full_attention() -> None:
    rng = np.random.default_rng(9)
    q = rng.normal(size=(3, 24)).astype(np.float32)
    k_shared = rng.normal(size=(128, 24)).astype(np.float32)
    v_shared = rng.normal(size=(128, 24)).astype(np.float32)
    k_suffix = rng.normal(size=(17, 24)).astype(np.float32)
    v_suffix = rng.normal(size=(17, 24)).astype(np.float32)
    shared_state = partial_attention_stats(q, k_shared, v_shared)
    suffix_state = partial_attention_stats(q, k_suffix, v_suffix)
    merged = fuse_shared_and_suffix(shared_state, suffix_state)
    ref = full_attention_reference(q, np.vstack([k_shared, k_suffix]), np.vstack([v_shared, v_suffix]))
    np.testing.assert_allclose(finalize_stats(merged), ref, rtol=2e-6, atol=2e-6)


def test_ring_reduce_and_tree_reduce_are_equivalent() -> None:
    rng = np.random.default_rng(11)
    q = rng.normal(size=(4, 16)).astype(np.float32)
    k = rng.normal(size=(96, 16)).astype(np.float32)
    v = rng.normal(size=(96, 16)).astype(np.float32)
    states = [partial_attention_stats(q, kk, vv) for kk, vv in zip(np.array_split(k, 8), np.array_split(v, 8))]
    direct = merge_stats(states[0], states[1])
    assert direct.m.shape == states[0].m.shape
    ring = finalize_stats(ring_reduce(states))
    tree = finalize_stats(tree_reduce(states))
    ref = full_attention_reference(q, k, v)
    np.testing.assert_allclose(ring, ref, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(tree, ref, rtol=2e-6, atol=2e-6)


def test_bf16_kv_storage_fp32_state_has_bounded_error_and_no_nan() -> None:
    rng = np.random.default_rng(13)
    q = rng.normal(size=(5, 64)).astype(np.float32)
    k = rng.normal(size=(513, 64)).astype(np.float32)
    v = rng.normal(size=(513, 64)).astype(np.float32)
    k_ref = quantize_bf16(k)
    v_ref = quantize_bf16(v)
    ref = full_attention_reference(q, k_ref, v_ref)
    states = [
        partial_attention_stats(
            q,
            k[block],
            v[block],
            state_precision="fp32",
            kv_storage_precision="bf16",
        )
        for block in np.array_split(np.arange(k.shape[0]), 8)
    ]
    got = finalize_stats(tree_reduce(states, state_precision="fp32"))
    metrics = error_metrics(ref, got)
    assert metrics["nan_or_inf_count"] == 0
    assert metrics["max_abs_error"] <= 5e-3


def test_fp16_bf16_state_precision_is_measured_not_headline() -> None:
    rng = np.random.default_rng(17)
    q = rng.normal(size=(2, 32)).astype(np.float32)
    k = rng.normal(size=(128, 32)).astype(np.float32)
    v = rng.normal(size=(128, 32)).astype(np.float32)
    ref = full_attention_reference(q, k, v)
    for precision in ["fp16", "bf16"]:
        states = [
            partial_attention_stats(q, kk, vv, state_precision=precision)
            for kk, vv in zip(np.array_split(k, 4), np.array_split(v, 4))
        ]
        got = finalize_stats(ring_reduce(states, state_precision=precision))
        metrics = error_metrics(ref, got)
        assert metrics["nan_or_inf_count"] == 0
        assert metrics["cosine_similarity"] > 0.99
