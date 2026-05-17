from __future__ import annotations

from waferagent.graph_ir import AgentNode, NodeType
from waferagent.kv_model import ModelKVConfig, sharing_metrics


def test_kv_uses_num_key_value_heads_for_gqa():
    cfg = ModelKVConfig(
        num_hidden_layers=2,
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=1,
        bytes_per_elem=2,
    )
    assert cfg.kv_bytes_per_token == 2 * 2 * 1 * 4 * 2


def test_prefix_sharing_reduces_kv_bytes():
    nodes = [
        AgentNode(
            "a",
            "j",
            "a",
            0,
            NodeType.LLM_CALL,
            "a",
            input_token_len=100,
            shared_prefix_ids=["p"],
            shared_prefix_token_len=60,
        ),
        AgentNode(
            "b",
            "j",
            "b",
            0,
            NodeType.LLM_CALL,
            "b",
            input_token_len=100,
            shared_prefix_ids=["p"],
            shared_prefix_token_len=60,
        ),
    ]
    metrics = sharing_metrics(nodes, ModelKVConfig())
    assert metrics["shared_kv_bytes"] < metrics["naive_kv_bytes"]
    assert metrics["kv_duplication_ratio"] > 1.0
