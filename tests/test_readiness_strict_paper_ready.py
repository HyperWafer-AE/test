from __future__ import annotations

import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "make_report", Path("/home/duzc/data/agent_wafer/scripts/make_report.py")
)
make_report = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(make_report)


def test_round4_readiness_schema_is_strict_and_nested():
    report = make_report._round4_readiness(Path("/home/duzc/data/agent_wafer"))
    assert set(report) == {"paper_ready", "sanity", "demoted", "pass"}
    assert "critical_path_ablation_nonzero_or_demoted" not in report["paper_ready"]
    assert "dynamic_pd_nonzero_or_demoted" not in report["paper_ready"]
    assert set(report["pass"]) == {"paper_ready", "sanity", "overall"}
