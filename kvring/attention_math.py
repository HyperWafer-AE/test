"""Exact online-softmax utilities used by KVRing reductions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


@dataclass
class OnlineSoftmaxState:
    m: np.ndarray
    l: np.ndarray
    z: np.ndarray

    def copy(self) -> "OnlineSoftmaxState":
        return OnlineSoftmaxState(self.m.copy(), self.l.copy(), self.z.copy())


def quantize_bf16(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32)
    bits = y.view(np.uint32)
    rounded = bits + np.uint32(0x00008000)
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def quantize_state(state: OnlineSoftmaxState, precision: str) -> OnlineSoftmaxState:
    if precision == "fp32":
        return state.copy()
    if precision == "fp16":
        return OnlineSoftmaxState(
            state.m.astype(np.float16).astype(np.float32),
            state.l.astype(np.float16).astype(np.float32),
            state.z.astype(np.float16).astype(np.float32),
        )
    if precision == "bf16":
        return OnlineSoftmaxState(
            quantize_bf16(state.m),
            quantize_bf16(state.l),
            quantize_bf16(state.z),
        )
    raise ValueError(f"unsupported precision: {precision}")


def block_state(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    scale: float | None = None,
    state_precision: str = "fp32",
) -> OnlineSoftmaxState:
    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    if scale is None:
        scale = 1.0 / np.sqrt(q.shape[-1])
    scores = (q @ k.T) * np.float32(scale)
    m = np.max(scores, axis=1)
    exp_scores = np.exp(scores - m[:, None], dtype=np.float32)
    l = np.sum(exp_scores, axis=1)
    z = exp_scores @ v
    return quantize_state(OnlineSoftmaxState(m, l, z), state_precision)


def merge_two(
    a: OnlineSoftmaxState,
    b: OnlineSoftmaxState,
    *,
    state_precision: str = "fp32",
) -> OnlineSoftmaxState:
    m = np.maximum(a.m, b.m)
    alpha = np.exp(a.m - m, dtype=np.float32)
    beta = np.exp(b.m - m, dtype=np.float32)
    l = alpha * a.l + beta * b.l
    z = alpha[:, None] * a.z + beta[:, None] * b.z
    return quantize_state(OnlineSoftmaxState(m, l, z), state_precision)


def merge_all(states: Iterable[OnlineSoftmaxState], *, state_precision: str = "fp32") -> OnlineSoftmaxState:
    states = list(states)
    if not states:
        raise ValueError("at least one state is required")
    out = states[0].copy()
    for state in states[1:]:
        out = merge_two(out, state, state_precision=state_precision)
    return out


def ring_reduce(states: List[OnlineSoftmaxState], *, state_precision: str = "fp32") -> OnlineSoftmaxState:
    return merge_all(states, state_precision=state_precision)


def tree_reduce(states: List[OnlineSoftmaxState], *, state_precision: str = "fp32") -> OnlineSoftmaxState:
    level = [s.copy() for s in states]
    while len(level) > 1:
        nxt: List[OnlineSoftmaxState] = []
        for i in range(0, len(level), 2):
            if i + 1 >= len(level):
                nxt.append(level[i])
            else:
                nxt.append(merge_two(level[i], level[i + 1], state_precision=state_precision))
        level = nxt
    return level[0]


def state_output(state: OnlineSoftmaxState) -> np.ndarray:
    return state.z / state.l[:, None]


def full_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, *, scale: float | None = None) -> np.ndarray:
    return state_output(block_state(q, k, v, scale=scale, state_precision="fp32"))


def full_attention_reference(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, *, scale: float | None = None
) -> np.ndarray:
    q64 = np.asarray(q, dtype=np.float64)
    k64 = np.asarray(k, dtype=np.float64)
    v64 = np.asarray(v, dtype=np.float64)
    if scale is None:
        scale = 1.0 / np.sqrt(q64.shape[-1])
    scores = (q64 @ k64.T) * np.float64(scale)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    return (probs @ v64).astype(np.float64)


def partial_attention_stats(
    q: np.ndarray,
    k_block: np.ndarray,
    v_block: np.ndarray,
    *,
    scale: float | None = None,
    state_precision: str = "fp32",
    kv_storage_precision: str = "fp32",
) -> OnlineSoftmaxState:
    if kv_storage_precision == "fp16":
        k_block = np.asarray(k_block, dtype=np.float16).astype(np.float32)
        v_block = np.asarray(v_block, dtype=np.float16).astype(np.float32)
    elif kv_storage_precision == "bf16":
        k_block = quantize_bf16(np.asarray(k_block, dtype=np.float32))
        v_block = quantize_bf16(np.asarray(v_block, dtype=np.float32))
    elif kv_storage_precision != "fp32":
        raise ValueError(f"unsupported KV storage precision: {kv_storage_precision}")
    return block_state(q, k_block, v_block, scale=scale, state_precision=state_precision)


def merge_stats(
    a: OnlineSoftmaxState,
    b: OnlineSoftmaxState,
    *,
    state_precision: str = "fp32",
) -> OnlineSoftmaxState:
    return merge_two(a, b, state_precision=state_precision)


def merge_many_stats(
    stats_list: Iterable[OnlineSoftmaxState], *, state_precision: str = "fp32"
) -> OnlineSoftmaxState:
    return merge_all(stats_list, state_precision=state_precision)


def fuse_shared_and_suffix(
    shared_stats: OnlineSoftmaxState,
    suffix_stats: OnlineSoftmaxState,
    *,
    state_precision: str = "fp32",
) -> OnlineSoftmaxState:
    return merge_two(shared_stats, suffix_stats, state_precision=state_precision)


def local_softmax_stats(
    q: np.ndarray,
    k_shard: np.ndarray,
    v_shard: np.ndarray,
    *,
    scale: float | None = None,
    state_precision: str = "fp32",
    kv_storage_precision: str = "fp32",
) -> OnlineSoftmaxState:
    return partial_attention_stats(
        q,
        k_shard,
        v_shard,
        scale=scale,
        state_precision=state_precision,
        kv_storage_precision=kv_storage_precision,
    )


def finalize_stats(stats: OnlineSoftmaxState) -> np.ndarray:
    return state_output(stats)


def blockwise_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    block_size: int,
    *,
    state_precision: str = "fp32",
) -> np.ndarray:
    states = [
        block_state(q, k[i : i + block_size], v[i : i + block_size], state_precision=state_precision)
        for i in range(0, k.shape[0], block_size)
    ]
    return state_output(merge_all(states, state_precision=state_precision))


def error_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    diff = candidate - reference
    ref_norm = float(np.linalg.norm(reference))
    cand_norm = float(np.linalg.norm(candidate))
    denom = max(ref_norm, 1e-12)
    cosine = float(np.sum(reference * candidate) / max(ref_norm * cand_norm, 1e-12))
    return {
        "max_abs_error": float(np.max(np.abs(diff))),
        "mean_abs_error": float(np.mean(np.abs(diff))),
        "relative_error": float(np.linalg.norm(diff) / denom),
        "cosine_similarity": cosine,
        "nan_or_inf_count": int(np.sum(~np.isfinite(candidate))),
    }
