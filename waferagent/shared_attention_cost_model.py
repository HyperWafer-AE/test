from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from waferagent.utils import read_json


@dataclass(frozen=True)
class SharedAttentionCostModel:
    fit_hash: str
    rows: list[dict[str, Any]]
    prediction_stat: str = "latency_p50_ms"

    @classmethod
    def from_json(cls, path: str | Path) -> "SharedAttentionCostModel":
        data = read_json(path)
        return cls(
            str(data.get("fit_hash", "")),
            list(data.get("rows", [])),
            str(data.get("prediction_stat", "latency_p50_ms")),
        )

    def predict_ms(
        self,
        mode: str,
        shared_prefix_tokens: int,
        private_tokens: int,
        num_agents: int,
        heads: int | None = 28,
        head_dim: int | None = 128,
    ) -> float:
        heads = int(heads or 28)
        head_dim = int(head_dim or 128)
        candidates = [r for r in self.rows if str(r.get("mode")) == mode]
        if not candidates:
            candidates = self.rows
        if not candidates:
            return 0.0

        def score(row: dict[str, Any]) -> float:
            return (
                abs(float(row.get("shared_prefix_tokens", 0)) - shared_prefix_tokens) / max(1.0, shared_prefix_tokens)
                + abs(float(row.get("private_tokens", 0)) - private_tokens) / max(1.0, private_tokens)
                + abs(float(row.get("num_agents", 0)) - num_agents) / max(1.0, num_agents)
                + 0.1 * abs(float(row.get("heads", heads)) - heads) / max(1.0, heads)
                + 0.1 * abs(float(row.get("head_dim", head_dim)) - head_dim) / max(1.0, head_dim)
            )

        row = min(candidates, key=score)
        preferred = row.get(self.prediction_stat)
        return float(
            preferred
            or row.get("latency_p50_ms")
            or row.get("latency_p90_ms")
            or row.get("latency_ms")
            or row.get("cohort_latency_ms")
            or 0.0
        )

    def prediction_quality(
        self,
        mode: str,
        shared_prefix_tokens: int,
        private_tokens: int,
        num_agents: int,
    ) -> str:
        for row in self.rows:
            if (
                str(row.get("mode")) == mode
                and int(float(row.get("shared_prefix_tokens", -1))) == int(shared_prefix_tokens)
                and int(float(row.get("private_tokens", -1))) == int(private_tokens)
                and int(float(row.get("num_agents", -1))) == int(num_agents)
            ):
                return "interpolated"
        return "nearest_neighbor_or_extrapolated"
