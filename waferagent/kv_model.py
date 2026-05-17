from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class ModelKVConfig:
    num_hidden_layers: int = 32
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int | None = None
    bytes_per_elem: int = 2

    @classmethod
    def from_hf_config(cls, cfg: dict[str, Any]) -> "ModelKVConfig":
        layers = int(cfg.get("num_hidden_layers") or cfg.get("n_layer") or 32)
        hidden = int(cfg.get("hidden_size") or cfg.get("n_embd") or 4096)
        heads = int(cfg.get("num_attention_heads") or cfg.get("n_head") or 32)
        kv_heads = int(cfg.get("num_key_value_heads") or cfg.get("multi_query_group_num") or heads)
        head_dim = int(cfg.get("head_dim") or hidden // heads)
        dtype = str(cfg.get("torch_dtype") or cfg.get("dtype") or "bfloat16").lower()
        bytes_per_elem = 4 if "32" in dtype else 2
        return cls(layers, hidden, heads, kv_heads, head_dim, bytes_per_elem)

    @classmethod
    def from_model_index_item(cls, item: dict[str, Any] | None) -> "ModelKVConfig":
        if not item:
            return cls()
        dtype = str(item.get("dtype_hint") or "bfloat16").lower()
        return cls(
            num_hidden_layers=int(item.get("num_hidden_layers") or 32),
            hidden_size=int(item.get("hidden_size") or 4096),
            num_attention_heads=int(item.get("num_attention_heads") or 32),
            num_key_value_heads=int(item.get("num_key_value_heads") or item.get("num_attention_heads") or 32),
            head_dim=int(item.get("head_dim") or (int(item.get("hidden_size") or 4096) // int(item.get("num_attention_heads") or 32))),
            bytes_per_elem=4 if "32" in dtype else 2,
        )

    @property
    def kv_bytes_per_token(self) -> int:
        head_dim = self.head_dim or self.hidden_size // self.num_attention_heads
        return int(2 * self.num_hidden_layers * self.num_key_value_heads * head_dim * self.bytes_per_elem)


@dataclass
class PrefixBlock:
    prefix_id: str
    token_len: int
    kv_bytes: int
    owner_nodes: list[str] = field(default_factory=list)
    ref_count: int = 0
    first_use_step: int = 0
    last_use_step: int = 0
    reuse_distance: int = 0
    criticality_score: float = 0.0
    tool_resume_probability: float = 0.0


def estimate_kv_bytes(tokens: int, model_cfg: ModelKVConfig | None = None) -> int:
    cfg = model_cfg or ModelKVConfig()
    return int(tokens * cfg.kv_bytes_per_token)


def build_prefix_blocks(nodes: Iterable[Any], model_cfg: ModelKVConfig | None = None) -> dict[str, PrefixBlock]:
    cfg = model_cfg or ModelKVConfig()
    blocks: dict[str, PrefixBlock] = {}
    for step, node in enumerate(nodes):
        node_id = getattr(node, "node_id")
        shared_ids = list(getattr(node, "shared_prefix_ids", []) or [])
        shared_tokens = int(getattr(node, "shared_prefix_token_len", 0) or 0)
        for prefix_id in shared_ids:
            block = blocks.get(prefix_id)
            kv_bytes = shared_tokens * cfg.kv_bytes_per_token
            if block is None:
                block = PrefixBlock(
                    prefix_id=prefix_id,
                    token_len=shared_tokens,
                    kv_bytes=kv_bytes,
                    first_use_step=step,
                )
                blocks[prefix_id] = block
            block.token_len = max(block.token_len, shared_tokens)
            block.kv_bytes = max(block.kv_bytes, kv_bytes)
            block.owner_nodes.append(node_id)
            block.ref_count += 1
            block.last_use_step = step
            block.criticality_score = max(block.criticality_score, float(getattr(node, "criticality", 0.0) or 0.0))
            if float(getattr(node, "tool_latency_ms", 0.0) or 0.0) > 0:
                block.tool_resume_probability = max(block.tool_resume_probability, 0.8)
    for block in blocks.values():
        block.reuse_distance = max(0, block.last_use_step - block.first_use_step)
    return blocks


def sharing_metrics(nodes: Iterable[Any], model_cfg: ModelKVConfig | None = None) -> dict[str, float]:
    cfg = model_cfg or ModelKVConfig()
    node_list = list(nodes)
    naive_tokens = sum(int(getattr(n, "input_token_len", getattr(n, "input_tokens", 0)) or 0) for n in node_list)
    blocks = build_prefix_blocks(node_list, cfg)
    unique_shared_tokens = sum(block.token_len for block in blocks.values())
    private_tokens = 0
    logical_shared_tokens = 0
    for n in node_list:
        input_tokens = int(getattr(n, "input_token_len", getattr(n, "input_tokens", 0)) or 0)
        shared_tokens = int(getattr(n, "shared_prefix_token_len", 0) or 0)
        logical_shared_tokens += shared_tokens
        private_tokens += max(0, input_tokens - shared_tokens)
    naive_kv_bytes = naive_tokens * cfg.kv_bytes_per_token
    shared_kv_bytes = (unique_shared_tokens + private_tokens) * cfg.kv_bytes_per_token
    return {
        "naive_kv_bytes": float(naive_kv_bytes),
        "shared_kv_bytes": float(shared_kv_bytes),
        "unique_shared_tokens": float(unique_shared_tokens),
        "logical_shared_tokens": float(logical_shared_tokens),
        "kv_saving_ratio": 1.0 - shared_kv_bytes / naive_kv_bytes if naive_kv_bytes else 0.0,
        "kv_duplication_ratio": naive_kv_bytes / shared_kv_bytes if shared_kv_bytes else 1.0,
    }


def ttl_priority(block: PrefixBlock, policy: str = "graph_ttl_criticality") -> float:
    if policy == "lru":
        return float(block.last_use_step)
    if policy == "steps_to_execution":
        return float(-block.reuse_distance + block.ref_count)
    return (
        2.0 * block.ref_count
        + 4.0 * block.criticality_score
        - 0.3 * block.reuse_distance
        - 1e-9 * block.kv_bytes
        + 2.0 * block.tool_resume_probability
    )


def evict_blocks(
    blocks: dict[str, PrefixBlock],
    capacity_bytes: int,
    policy: str = "graph_ttl_criticality",
) -> tuple[list[str], int]:
    resident = sum(block.kv_bytes for block in blocks.values())
    if resident <= capacity_bytes:
        return [], 0
    evicted: list[str] = []
    for block in sorted(blocks.values(), key=lambda b: ttl_priority(b, policy)):
        if resident <= capacity_bytes:
            break
        evicted.append(block.prefix_id)
        resident -= block.kv_bytes
    overflow = max(0, resident - capacity_bytes)
    return evicted, overflow
