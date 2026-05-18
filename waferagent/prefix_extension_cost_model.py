from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from waferagent.calibrated_cost_model import CalibratedCostModel
from waferagent.utils import file_sha256


def _dot(coef: list[float], features: list[float]) -> float:
    return float(sum(c * x for c, x in zip(coef, features)))


@dataclass
class PrefixExtensionCostModel:
    full_prefill_coef: list[float]
    extend_prefill_coef: list[float]
    decode_tpot_coef: list[float]
    fit_hash: str = ""
    schema_version: str = "round4.prefix_extension.1"

    @classmethod
    def from_json(cls, path: str | Path) -> "PrefixExtensionCostModel":
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(
            full_prefill_coef=[float(x) for x in data["full_prefill_coef"]],
            extend_prefill_coef=[float(x) for x in data["extend_prefill_coef"]],
            decode_tpot_coef=[float(x) for x in data["decode_tpot_coef"]],
            fit_hash=file_sha256(p),
            schema_version=str(data.get("schema_version", "unknown")),
        )

    @classmethod
    def from_calibrated(cls, model: CalibratedCostModel) -> "PrefixExtensionCostModel":
        full = list(model.prefill_coef)
        # Conservative fallback: extension depends on both prefix and private tokens.
        # It is cheaper than full prefill but not equal to private-only prefill.
        ext = [0.0, 0.08, 0.70, 0.0, 2e-7, 1e-6, 0.0]
        return cls(full, ext, list(model.decode_tpot_coef), fit_hash=model.fit_hash)

    def full_prefill_ms(self, input_len: int, batch_size: int = 1) -> float:
        x = float(max(0, input_len))
        b = float(max(1, batch_size))
        return max(0.0, _dot(self.full_prefill_coef, [1.0, x, x * x, b, x * b]))

    def extend_prefill_ms(self, prefix_len: int, private_len: int, batch_size: int = 1) -> float:
        p = float(max(0, prefix_len))
        q = float(max(0, private_len))
        b = float(max(1, batch_size))
        raw = _dot(self.extend_prefill_coef, [1.0, p, q, b, p * q, q * q, p * b])
        full = self.full_prefill_ms(int(p + q), int(b))
        private_floor = 0.05 * self.full_prefill_ms(int(q), int(b))
        return max(private_floor, min(max(0.0, raw), full))

    def decode_ms(self, context_len: int, output_len: int, batch_size: int = 1) -> float:
        x = float(max(0, context_len))
        y = float(max(0, output_len))
        b = float(max(1, batch_size))
        tpot = max(0.0, _dot(self.decode_tpot_coef, [1.0, x, y, b, x * b]))
        return tpot * y
