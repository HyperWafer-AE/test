from waferagent.shared_attention_microbench import SharedAttentionCase, run_shared_attention_case


def test_shared_attention_microbench_schema_cpu_small():
    rows = run_shared_attention_case(
        SharedAttentionCase(
            mode="split_shared_private_merge",
            shared_prefix_tokens=8,
            private_tokens=4,
            num_agents=2,
            heads=2,
            head_dim=8,
            device="cpu",
        ),
        reps=1,
        seed=1,
    )
    assert rows
    row = rows[0]
    expected = {
        "mode",
        "shared_prefix_tokens",
        "private_tokens",
        "num_agents",
        "latency_ms",
        "memory_bytes_estimated",
        "read_byte_reduction_ratio",
        "latency_speedup",
        "oom",
        "exact_merge",
    }
    assert expected <= set(row)
    assert row["mode"] == "split_shared_private_merge"
    assert row["exact_merge"] is True
