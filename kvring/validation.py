"""Validation helpers for capacity and paper-claim gating."""

from __future__ import annotations

from typing import Mapping

from .accounting import ModeResult


def result_capacity_valid(result: ModeResult) -> bool:
    return bool(result.extra.get("valid_capacity", not result.extra.get("region_capacity_violation", False)))


def row_capacity_valid(row: Mapping[str, object]) -> bool:
    value = row.get("valid_capacity", row.get("capacity_valid", True))
    if isinstance(value, bool):
        return value
    return str(value).lower() not in {"false", "0", "no", "invalid"}


def capacity_violation_reason(result: ModeResult) -> str:
    return str(result.extra.get("capacity_violation_reason", ""))


def filter_valid_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in rows if row_capacity_valid(row)]


def filter_invalid_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in rows if not row_capacity_valid(row)]
