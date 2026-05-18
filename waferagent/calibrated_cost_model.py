from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from waferagent.utils import file_sha256


def _dot(features: list[float], coef: list[float]) -> float:
    n = min(len(features), len(coef))
    return float(sum(features[i] * coef[i] for i in range(n)))


@dataclass
class CalibratedCostModel:
    prefill_coef: list[float]
    decode_tpot_coef: list[float]
    fit_hash: str = ""
    schema_version: str = "round3.forward.1"

    @classmethod
    def from_json(cls, path: str | Path) -> "CalibratedCostModel":
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        prefill = data.get("prefill_coef") or data.get("prefill_coefficients") or []
        decode = data.get("decode_tpot_coef") or data.get("decode_coefficients") or []
        if not prefill and data.get("prefill_ms_coefficients"):
            old = [float(x) for x in data["prefill_ms_coefficients"]]
            prefill = [old[0], old[1], old[2], 0.0, 0.0]
        if not decode and data.get("per_decode_token_ms_coefficients"):
            old = [float(x) for x in data["per_decode_token_ms_coefficients"]]
            decode = [old[0], old[1], 0.0, old[2], 0.0]
        if not prefill or not decode:
            raise ValueError(f"Calibration file does not contain usable coefficients: {p}")
        return cls(
            prefill_coef=[float(x) for x in prefill],
            decode_tpot_coef=[float(x) for x in decode],
            fit_hash=file_sha256(p),
            schema_version=str(data.get("schema_version", "unknown")),
        )

    def prefill_ms(self, input_tokens: int, batch_size: int = 1) -> float:
        x = float(max(0, input_tokens))
        b = float(max(1, batch_size))
        features = [1.0, x, x * x, b, x * b]
        return max(0.0, _dot(features, self.prefill_coef))

    def decode_ms(self, input_tokens: int, output_tokens: int, batch_size: int = 1) -> float:
        x = float(max(0, input_tokens))
        y = float(max(0, output_tokens))
        b = float(max(1, batch_size))
        features = [1.0, x, y, b, x * b]
        tpot = max(0.0, _dot(features, self.decode_tpot_coef))
        return tpot * max(0.0, y)
