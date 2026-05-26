from pathlib import Path

from waferstateflow.experiments.run_problem_characterization import main as characterization_main
from waferstateflow.experiments.run_scheduler_comparison import main as scheduler_main
from waferstateflow.experiments.run_workflow_suite import main as suite_main


REQUIRED_EXPERIMENT_FILES = {
    "metadata.json",
    "config.json",
    "state_nodes.csv",
    "operator_nodes.csv",
    "access_edges.csv",
    "state_hotness.csv",
    "policy_decisions.csv",
    "wave_schedule.csv",
    "simulation_summary.csv",
    "report.md",
}


def test_problem_characterization_writes_artifact_contract(tmp_path):
    out = tmp_path / "characterization"
    characterization_main(
        [
            "--workflow",
            "mapreduce",
            "--num-agents",
            "2",
            "--shared-state-size",
            "300",
            "--unique-state-size",
            "50",
            "--output-token-mean",
            "30",
            "--seed",
            "3",
            "--out",
            str(out),
        ]
    )
    _assert_required_files(out)
    assert "Input redundancy ratio" in (out / "report.md").read_text(encoding="utf-8")


def test_scheduler_comparison_writes_artifact_contract(tmp_path):
    out = tmp_path / "scheduler"
    scheduler_main(
        [
            "--workflow",
            "reflection",
            "--num-agents",
            "3",
            "--shared-state-size",
            "300",
            "--unique-state-size",
            "50",
            "--output-token-mean",
            "30",
            "--mesh",
            "4x4",
            "--seed",
            "4",
            "--out",
            str(out),
        ]
    )
    _assert_required_files(out)
    summary = (out / "simulation_summary.csv").read_text(encoding="utf-8")
    assert "WaferStateFlow" in summary
    assert "request_parallel_gpu_like" in summary


def test_workflow_suite_writes_aggregate_and_per_workflow_outputs(tmp_path):
    out = tmp_path / "suite"
    suite_main(
        [
            "--workflows",
            "mapreduce,iterative",
            "--mode",
            "characterization",
            "--batch-size",
            "2",
            "--shared-state-size",
            "300",
            "--unique-state-size",
            "50",
            "--output-token-mean",
            "30",
            "--seed",
            "5",
            "--out",
            str(out),
        ]
    )
    for path in [
        out / "metadata.json",
        out / "config.json",
        out / "characterization_summary.csv",
        out / "scheduler_summary.csv",
        out / "report.md",
        out / "figures" / "suite_redundancy.png",
    ]:
        assert path.exists(), path
    _assert_required_files(out / "characterization" / "mapreduce")
    _assert_required_files(out / "characterization" / "iterative")


def _assert_required_files(out: Path) -> None:
    missing = [filename for filename in REQUIRED_EXPERIMENT_FILES if not (out / filename).exists()]
    assert not missing, missing
