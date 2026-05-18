from __future__ import annotations

from pathlib import Path
import importlib.util


spec = importlib.util.spec_from_file_location(
    "make_report", "/home/duzc/data/agent_wafer/scripts/make_report.py"
)
make_report = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(make_report)


def test_report_readiness_keys_exist():
    checks = make_report.readiness(Path("/home/duzc/data/agent_wafer"))
    assert "git commit available" in checks
    assert "neutral multipliers used" in checks
