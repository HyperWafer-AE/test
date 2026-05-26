import csv

from flowmorph.analyzer import FlowMorphConfig, characterize_phase_irregularity
from flowmorph.converters import phase_dag_from_workflow_name
from flowmorph.experiments.run_characterization import main as flowmorph_main
from flowmorph.experiments.run_scheduler_comparison import main as scheduler_main
from flowmorph.experiments.run_scheduler_sensitivity import main as sensitivity_main
from flowmorph.ir import PhaseDAG, PhaseOperator
from flowmorph.schedulers import FrontierSchedulerConfig, run_scheduler


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


def test_frontier_aware_scheduler_uses_parallel_and_fast_lane_modes():
    dag = PhaseDAG("frontier_scheduler")
    for i in range(4):
        dag.add_operator(PhaseOperator(f"W{i}", prefill_cost=4.0, decode_cost=1.0))
    dag.add_operator(
        PhaseOperator(
            "join",
            dependencies=[f"W{i}" for i in range(4)],
            prefill_cost=8.0,
            decode_cost=2.0,
        )
    )

    result = run_scheduler(
        dag,
        "frontier_aware_morphing",
        FrontierSchedulerConfig(
            worker_count=4,
            frontier_cv_threshold=0.1,
            parallel_slack_threshold=1.1,
            phase_variation_threshold=10.0,
        ),
    )
    modes = {row["mode"] for row in result["schedule_rows"]}
    assert "parallel" in modes
    assert "consolidated_fast_lane" in modes
    assert result["summary"]["mode_switch_count"] >= 1
    assert result["summary"]["wide_stage_utilization"] > 0


def test_frontier_aware_scheduler_falls_back_on_weak_frontier():
    dag = PhaseDAG("weak_chain")
    for i in range(4):
        dag.add_operator(
            PhaseOperator(
                f"O{i}",
                dependencies=[f"O{i - 1}"] if i else [],
                prefill_cost=1.0,
                decode_cost=1.0,
            )
        )
    config = FrontierSchedulerConfig(worker_count=4)
    fixed = run_scheduler(dag, "fixed_worker_pool", config)
    morphing = run_scheduler(dag, "frontier_aware_morphing", config)
    assert morphing["summary"]["policy"] == "fallback_fixed_worker_pool"
    assert morphing["summary"]["workflow_latency"] == fixed["summary"]["workflow_latency"]


def test_static_full_resource_alias_is_available():
    dag = PhaseDAG("static_full")
    dag.add_operator(PhaseOperator("A", prefill_cost=2.0))
    dag.add_operator(PhaseOperator("B", prefill_cost=2.0))
    static_full = run_scheduler(dag, "static_full_resource", FrontierSchedulerConfig(worker_count=2))
    always_parallel = run_scheduler(dag, "always_parallel", FrontierSchedulerConfig(worker_count=2))
    assert static_full["summary"]["scheduler"] == "static_full_resource"
    assert static_full["summary"]["workflow_latency"] == always_parallel["summary"]["workflow_latency"]


def test_flowmorph_scheduler_comparison_outputs_required_artifacts(tmp_path):
    out = tmp_path / "scheduler"
    scheduler_main(
        [
            "--workflows",
            "mapreduce,iterative",
            "--schedulers",
            "fixed_worker_pool,always_parallel,always_consolidated,static_split_resource,frontier_aware_morphing",
            "--batch-size",
            "4",
            "--seed",
            "0",
            "--out",
            str(out),
        ]
    )
    for filename in [
        "metadata.json",
        "config.json",
        "scheduler_summary.csv",
        "scheduler_trace.csv",
        "workflow_selection.csv",
        "report.md",
    ]:
        assert (out / filename).exists(), filename
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "does not implement wafer-specific placement" in report
    assert "makes no wafer performance claims" in report
    with (out / "workflow_selection.csv").open(newline="", encoding="utf-8") as f:
        selection = list(csv.DictReader(f))
    reasons = {row["workflow"]: row["selection_reason"] for row in selection}
    assert reasons["mapreduce"] == "frontier_positive"
    assert reasons["iterative"] == "negative_control"
    with (out / "scheduler_summary.csv").open(newline="", encoding="utf-8") as f:
        summary_rows = list(csv.DictReader(f))
    required_metrics = {
        "workflow_latency",
        "worker_idle_fraction",
        "critical_path_delay",
        "mode_switch_count",
        "wide_stage_utilization",
        "narrow_stage_latency",
    }
    assert required_metrics.issubset(summary_rows[0])
    assert {
        "fixed_worker_pool",
        "always_parallel",
        "always_consolidated",
        "frontier_aware_morphing",
    }.issubset({row["scheduler"] for row in summary_rows})


def test_mode_switch_overhead_affects_morphing_and_static_split():
    dag = PhaseDAG("overhead")
    for i in range(4):
        dag.add_operator(PhaseOperator(f"W{i}", prefill_cost=4.0, decode_cost=1.0))
    dag.add_operator(
        PhaseOperator(
            "join",
            dependencies=[f"W{i}" for i in range(4)],
            prefill_cost=8.0,
            decode_cost=2.0,
        )
    )
    base_config = FrontierSchedulerConfig(
        worker_count=4,
        frontier_cv_threshold=0.1,
        parallel_slack_threshold=1.1,
        phase_variation_threshold=10.0,
        mode_switch_overhead=0.0,
    )
    overhead_config = FrontierSchedulerConfig(
        worker_count=4,
        frontier_cv_threshold=0.1,
        parallel_slack_threshold=1.1,
        phase_variation_threshold=10.0,
        mode_switch_overhead=2.0,
    )
    morph_base = run_scheduler(dag, "frontier_aware_morphing", base_config)
    morph_overhead = run_scheduler(dag, "frontier_aware_morphing", overhead_config)
    split_base = run_scheduler(dag, "static_split_resource", base_config)
    split_overhead = run_scheduler(dag, "static_split_resource", overhead_config)
    assert morph_overhead["summary"]["workflow_latency"] > morph_base["summary"]["workflow_latency"]
    assert split_overhead["summary"]["workflow_latency"] > split_base["summary"]["workflow_latency"]
    assert any(row["mode_switch_overhead"] > 0 for row in morph_overhead["schedule_rows"])


def test_scheduler_sensitivity_outputs_oracle_regret_artifacts(tmp_path):
    out = tmp_path / "sensitivity"
    sensitivity_main(
        [
            "--workflows",
            "mapreduce,iterative",
            "--consolidated-speedup-exponents",
            "0.2",
            "--mode-switch-overheads",
            "0,1",
            "--worker-counts",
            "4",
            "--criticality-thresholds",
            "0.8",
            "--batch-size",
            "4",
            "--seed",
            "0",
            "--out",
            str(out),
        ]
    )
    for filename in [
        "metadata.json",
        "config.json",
        "report.md",
        "sensitivity_summary.csv",
        "winner_counts.csv",
        "regret_by_workflow.csv",
    ]:
        assert (out / filename).exists(), filename
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "does not add wafer placement" in report
    assert "makes no wafer performance claims" in report
    assert "best_static_oracle" in report
    with (out / "sensitivity_summary.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    required = {
        "flowmorph_latency",
        "best_static_latency",
        "best_static_oracle_latency",
        "regret",
        "winner",
        "mode_switch_count",
        "frontier_status",
    }
    assert required.issubset(rows[0])
    assert {row["frontier_status"] for row in rows} == {"frontier_positive", "weak"}
    assert any(row["workflow"] == "iterative" and row["frontier_status"] == "weak" for row in rows)
