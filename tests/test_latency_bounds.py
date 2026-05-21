from __future__ import annotations

from kvring.artifacts import paper_default_results, result_row


def test_every_default_mode_reports_latency_bounds() -> None:
    rows = [result_row(r) for r in paper_default_results()]
    for row in rows:
        assert float(row["serialized_latency_s"]) >= 0.0
        assert float(row["throughput_bound_latency_s"]) >= 0.0
        assert float(row["critical_path_latency_s"]) >= 0.0
        assert row["latency_bound_used"] in {
            "serialized",
            "throughput_bound",
            "critical_path",
            "conservative_max",
        }
        assert float(row["attention_stage_proxy_latency_s"]) >= 0.0
