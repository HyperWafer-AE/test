from waferagent.controlled_workloads import StrictControlledSharedKVConfig, generate_strict_controlled_shared_kv_graphs


def test_strict_controlled_tokens_match_request():
    cfg = StrictControlledSharedKVConfig(
        num_jobs=4,
        reuse_group_size=2,
        shared_prefix_tokens=2048,
        private_suffix_tokens=512,
        decode_tokens=128,
        num_agents_per_job=3,
        seed=7,
    )
    graphs = generate_strict_controlled_shared_kv_graphs(cfg)
    for graph in graphs:
        for node in graph.nodes.values():
            assert node.shared_prefix_token_len == 2048
            assert node.private_prefix_token_len == 512
            assert node.input_token_len == 2560
            assert node.actual_output_token_len == 128


def test_strict_controlled_reuse_group_size_controls_prefix_ids():
    cfg = StrictControlledSharedKVConfig(
        num_jobs=5,
        reuse_group_size=2,
        shared_prefix_tokens=1024,
        private_suffix_tokens=128,
        decode_tokens=32,
        num_agents_per_job=2,
        seed=3,
    )
    graphs = generate_strict_controlled_shared_kv_graphs(cfg)
    job_prefix = {
        graph.graph_id: next(iter(next(iter(graph.nodes.values())).shared_prefix_ids))
        for graph in graphs
    }
    assert job_prefix["controlled_job_0"] == job_prefix["controlled_job_1"]
    assert job_prefix["controlled_job_2"] == job_prefix["controlled_job_3"]
    assert job_prefix["controlled_job_0"] != job_prefix["controlled_job_2"]
    assert len(set(job_prefix.values())) == 3
