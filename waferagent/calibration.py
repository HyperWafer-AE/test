from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def fit_prefill_decode(csv_path: str | Path, out_json: str | Path) -> dict[str, list[float]]:
    df = pd.read_csv(csv_path)
    x_prefill = np.column_stack(
        [
            np.ones(len(df)),
            df["input_len"].to_numpy(dtype=float),
            df["input_len"].to_numpy(dtype=float) ** 2,
        ]
    )
    y_prefill = df["ttft_ms"].to_numpy(dtype=float)
    a, *_ = np.linalg.lstsq(x_prefill, y_prefill, rcond=None)
    per_decode = df["decode_ms"].to_numpy(dtype=float) / df["output_len"].clip(lower=1).to_numpy(dtype=float)
    x_decode = np.column_stack(
        [
            np.ones(len(df)),
            (df["input_len"] + df["output_len"]).to_numpy(dtype=float),
            df["batch_size"].to_numpy(dtype=float),
        ]
    )
    b, *_ = np.linalg.lstsq(x_decode, per_decode, rcond=None)
    fit = {
        "prefill_ms_coefficients": a.tolist(),
        "per_decode_token_ms_coefficients": b.tolist(),
    }
    Path(out_json).write_text(json.dumps(fit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return fit
