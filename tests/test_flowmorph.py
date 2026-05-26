from flowmorph.analyzer import FlowMorphConfig, characterize_phase_irregularity
from flowmorph.converters import phase_dag_from_workflow_name
from flowmorph.experiments.run_characterization import main as flowmorph_main
from flowmorph.ir import PhaseDAG, PhaseOperator


def test_phase_dag_topology_and_critical_path():
    dag = PhaseDAG("toy")
    dag.add_operator(PhaseOperator("A", prefill_cost=1.0))
    dag.add_operator(PhaseOperator("B", dependencies=["A"], decode_cost=2.0))
    dag.add_operator(PhaseOperator("C", dependencies=["A"], local_tool_cost=3.0))
    dag.compute_earliest_ready_times()

    assert dag.topological_order() == ["A", "B", "C"]
    assert dag.operators["B"].earliest_ready_time == 1.0
    assert dag.critical_path_length() == 4.0
    assert dag.total_work() == 6.0


def test_converter_from_synthetic_workflow_produces_phase_costs():
    dag = phase_dag_from_workflow_name(
        "mapreduce",
        num_agents=2,
        shared_state_size=300,
        unique_state_size=50,
        output_token_mean=30,
        seed=1,
    )
    assert dag.operators
    assert all(op.estimated_input_tokens >= 0 for op in dag.operators.values())
    assert any(op.prefill_cost > 0 for op in dag.operators.values())
    assert any(op.decode_cost > 0 for op in dag.operators.values())


def test_irregularity_metrics_and_decision_gate_continue_case():
    dag = PhaseDAG("irregular")
    dag.add_operator(PhaseOperator("A", prefill_cost=10.0))
    dag.add_operator(PhaseOperator("B", prefill_cost=1.0))
    dag.add_operator(PhaseOperator("C", decode_cost=10.0, dependencies=["A"]))
    dag.add_operator(PhaseOperator("D", local_tool_cost=8.0, dependencies=["B"]))
    dag.add_operator(PhaseOperator("E", prefill_cost=5.0, dependencies=["C", "D"]))

    result = characterize_phase_irregularity(
        dag,
        FlowMorphConfig(frontier_cv_threshold=0.1, phase_variation_threshold=0.1),
    )
    summary = result["summary"]
    assert summary["frontier_width_cv"] > 0
    assert summary["phase_mix_variation"] > 0
    assert summary["decision"] == "continue_to_flowmorph_scheduling"


def test_irregularity_metrics_and_decision_gate_weak_case():
    dag = PhaseDAG("stable")
    for i in range(4):
        dag.add_operator(PhaseOperator(f"O{i}", prefill_cost=1.0, decode_cost=1.0))
    result = characterize_phase_irregularity(dag)
    assert result["summary"]["decision"] == "weak_direction"


def test_flowmorph_characterization_outputs_all_required_files(tmp_path):
    out = tmp_path / "flowmorph"
    flowmorph_main(
        [
            "--workflows",
            "mapreduce,reflection",
            "--num-agents",
            "3",
            "--shared-state-size",
            "300",
            "--unique-state-size",
            "50",
            "--output-token-mean",
            "30",
            "--seed",
            "2",
            "--out",
            str(out),
        ]
    )
    for filename in [
        "metadata.json",
        "config.json",
        "flowmorph_summary.csv",
        "frontier_timeline.csv",
        "phase_operators.csv",
        "report.md",
        "figures/flowmorph_irregularity.png",
    ]:
        assert (out / filename).exists(), filename
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "does not make WaferStateFlow claims" in report
    assert "no wafer scheduling" in report
