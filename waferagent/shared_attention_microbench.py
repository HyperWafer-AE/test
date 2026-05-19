from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SharedAttentionCase:
    mode: str
    shared_prefix_tokens: int
    private_tokens: int
    num_agents: int
    heads: int
    head_dim: int
    device: str
    dtype: str = "float16"


def _dtype(torch, dtype: str):
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype in {"fp32", "float32"}:
        return torch.float32
    return torch.float16


def run_shared_attention_case(case: SharedAttentionCase, reps: int = 5, seed: int = 0) -> list[dict[str, Any]]:
    import torch

    device = torch.device(case.device if torch.cuda.is_available() and str(case.device).startswith("cuda") else "cpu")
    dtype = _dtype(torch, case.dtype)
    torch.manual_seed(seed)
    rows: list[dict[str, Any]] = []
    elem_bytes = 2 if dtype in {torch.float16, torch.bfloat16} else 4
    shared_bytes = 2 * case.heads * case.shared_prefix_tokens * case.head_dim * elem_bytes
    private_bytes = 2 * case.heads * case.private_tokens * case.head_dim * elem_bytes

    for rep in range(reps):
        oom = False
        latency_ms = 0.0
        exact_merge = case.mode != "split_shared_private_merge"
        try:
            k_shared = torch.randn(case.heads, case.shared_prefix_tokens, case.head_dim, device=device, dtype=dtype)
            v_shared = torch.randn_like(k_shared)
            q = torch.randn(case.num_agents, case.heads, case.head_dim, device=device, dtype=dtype)
            k_private = torch.randn(case.num_agents, case.heads, case.private_tokens, case.head_dim, device=device, dtype=dtype)
            v_private = torch.randn_like(k_private)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            if case.mode == "independent_attention":
                outs = []
                for agent in range(case.num_agents):
                    logits = torch.einsum("hd,htd->ht", q[agent], k_shared)
                    prob = torch.softmax(logits.float(), dim=-1).to(dtype)
                    outs.append(torch.einsum("ht,htd->hd", prob, v_shared))
                total = torch.stack(outs).sum()
            elif case.mode == "cohort_attention":
                logits = torch.einsum("ahd,htd->aht", q, k_shared)
                prob = torch.softmax(logits.float(), dim=-1).to(dtype)
                total = torch.einsum("aht,htd->ahd", prob, v_shared).sum()
            elif case.mode == "split_shared_private_merge":
                shared_logits = torch.einsum("ahd,htd->aht", q, k_shared)
                private_logits = torch.einsum("ahd,ahtd->aht", q, k_private)
                logits = torch.cat([shared_logits, private_logits], dim=-1)
                prob = torch.softmax(logits.float(), dim=-1).to(dtype)
                ps = prob[:, :, : case.shared_prefix_tokens]
                pp = prob[:, :, case.shared_prefix_tokens :]
                shared_out = torch.einsum("aht,htd->ahd", ps, v_shared)
                private_out = torch.einsum("aht,ahtd->ahd", pp, v_private)
                total = (shared_out + private_out).sum()
                exact_merge = True
            else:
                raise ValueError(f"unknown mode {case.mode}")
            total.item()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latency_ms = (time.perf_counter() - start) * 1000.0
        except RuntimeError as exc:
            oom = "out of memory" in str(exc).lower()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if not oom:
                raise
        if case.mode == "independent_attention":
            memory_bytes = case.num_agents * shared_bytes
        elif case.mode == "cohort_attention":
            memory_bytes = shared_bytes
        else:
            memory_bytes = shared_bytes + case.num_agents * private_bytes
        rows.append(
            {
                "mode": case.mode,
                "shared_prefix_tokens": case.shared_prefix_tokens,
                "private_tokens": case.private_tokens,
                "num_agents": case.num_agents,
                "heads": case.heads,
                "head_dim": case.head_dim,
                "rep": rep,
                "latency_ms": latency_ms,
                "memory_bytes_estimated": int(memory_bytes),
                "shared_read_bytes_estimated": int(memory_bytes if case.mode != "split_shared_private_merge" else shared_bytes),
                "read_byte_reduction_ratio": 0.0,
                "latency_speedup": 1.0,
                "oom": bool(oom),
                "exact_merge": bool(exact_merge),
                "device": str(device),
                "dtype": str(dtype).replace("torch.", ""),
            }
        )
    return rows
