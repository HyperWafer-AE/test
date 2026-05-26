from waferstateflow.workflow_generators import WORKFLOW_NAMES, generate_workflow


def test_all_generators_are_reproducible_and_exportable(tmp_path):
    for workflow in WORKFLOW_NAMES:
        graph_a = generate_workflow(
            workflow,
            num_agents=3,
            num_branches=3,
            num_rounds=2,
            shared_state_size=900,
            unique_state_size=90,
            output_token_mean=60,
            seed=7,
        )
        graph_b = generate_workflow(
            workflow,
            num_agents=3,
            num_branches=3,
            num_rounds=2,
            shared_state_size=900,
            unique_state_size=90,
            output_token_mean=60,
            seed=7,
        )
        assert graph_a.to_dict() == graph_b.to_dict()
        assert len(graph_a.states) > 0
        assert len(graph_a.operators) > 0
        assert graph_a.operator_topological_order()

        out = tmp_path / workflow
        graph_a.export_csv(out)
        assert (out / "state_nodes.csv").exists()
        assert (out / "operator_nodes.csv").exists()
        assert (out / "access_edges.csv").exists()
