from __future__ import annotations

from kvring.artifacts import paper_default_results, result_row
from kvring.validation import filter_invalid_rows, filter_valid_rows


def test_headline_capacity_filter_excludes_invalid_rows() -> None:
    rows = [
        {"mode": "valid", "valid_capacity": True},
        {"mode": "invalid", "valid_capacity": False, "region_capacity_violation": True},
    ]
    assert [r["mode"] for r in filter_valid_rows(rows)] == ["valid"]
    assert [r["mode"] for r in filter_invalid_rows(rows)] == ["invalid"]


def test_paper_default_headline_rows_are_capacity_valid() -> None:
    rows = [result_row(r) for r in paper_default_results()]
    headline = filter_valid_rows(rows)
    assert headline
    assert all(r["valid_capacity"] is True for r in headline)
    assert len(headline) == len(rows)
