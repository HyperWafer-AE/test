import csv

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
    assert summary["width_drop_ratio"] > 1
    assert summary["parallel_slack"] == summary["total_work_to_critical_path_ratio"]
    assert summary["combined_opportunity"] == "yes"
    assert summary["opportunity_taxonomy"] == "frontier_and_phase"


def test_opportunity_taxonomy_frontier_only_case():
    dag = PhaseDAG("frontier_only")
    for i in range(6):
        dag.add_operator(PhaseOperator(f"W{i}", prefill_cost=1.0, decode_cost=1.0))
    dag.add_operator(
        PhaseOperator(
            "join",
            dependencies=[f"W{i}" for i in range(6)],
            prefill_cost=1.0,
            decode_cost=1.0,
        )
    )

    result = characterize_phase_irregularity(
        dag,
        FlowMorphConfig(frontier_cv_threshold=0.1, phase_variation_threshold=0.1),
    )
    summary = result["summary"]
    assert summary["frontier_morphing_opportunity"] == "yes"
    assert summary["phase_morphing_opportunity"] == "no"
    assert summary["combined_opportunity"] == "no"
    assert summary["opportunity_taxonomy"] == "frontier_only"


def test_opportunity_taxonomy_phase_only_case():
    dag = PhaseDAG("phase_only")
    dag.add_operator(PhaseOperator("prefill", prefill_cost=10.0))
    dag.add_operator(PhaseOperator("decode", dependencies=["prefill"], decode_cost=10.0))
    dag.add_operator(PhaseOperator("tool", dependencies=["decode"], local_tool_cost=10.0))

    result = characterize_phase_irregularity(
        dag,
        FlowMorphConfig(
            frontier_cv_threshold=0.1,
            phase_variation_threshold=0.1,
            parallel_slack_threshold=2.0,
        ),
    )
    summary = result["summary"]
    assert summary["frontier_morphing_opportunity"] == "no"
    assert summary["phase_morphing_opportunity"] == "yes"
    assert summary["combined_opportunity"] == "no"
    assert summary["opportunity_taxonomy"] == "phase_only"


def test_irregularity_metrics_and_decision_gate_weak_case():
    dag = PhaseDAG("stable")
    for i in range(4):
        dag.add_operator(
            PhaseOperator(
                f"O{i}",
                dependencies=[f"O{i - 1}"] if i else [],
                prefill_cost=1.0,
                decode_cost=1.0,
            )
        )
    result = characterize_phase_irregularity(dag)
    assert result["summary"]["opportunity_taxonomy"] == "weak"


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
    assert "FlowMorph-v1 gate" in report
    assert "frontier_only" in report
    with (out / "flowmorph_summary.csv").open(newline="", encoding="utf-8") as f:
        fieldnames = csv.DictReader(f).fieldnames or []
    for column in [
        "frontier_morphing_opportunity",
        "phase_morphing_opportunity",
        "combined_opportunity",
        "opportunity_taxonomy",
        "width_drop_ratio",
        "wide_stage_work_fraction",
        "narrow_critical_stage_fraction",
        "critical_path_serial_fraction",
        "parallel_slack",
    ]:
        assert column in fieldnames
