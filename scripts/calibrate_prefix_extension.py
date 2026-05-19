#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from waferagent.model_discovery import load_or_scan, select_model
from waferagent.prefix_extension_timer import PrefixExtensionTimer
from waferagent.utils import enforce_clean_git_tree, file_sha256, finalize_run_dir, init_run_dir, write_json


def _parse_ints(text: str, default: list[int]) -> list[int]:
    if not text:
        return default
    return [int(x) for x in text.split(",") if x.strip()]


def _choose_model(model: str) -> tuple[str, str]:
    index = load_or_scan("/data2/model_zoo", "configs/models.local.json")
    chosen = select_model(index) if model == "auto" else next(
        (m for m in index.get("models", []) if model in m.get("name", "") or model in m.get("path", "")),
        None,
    )
    if not chosen:
        raise FileNotFoundError(f"No local loadable model matched --model {model}")
    return str(chosen["name"]), str(chosen["path"])


def _visible_gpu_id(gpus: str) -> int:
    return 0 if gpus else 0


def _fit_linear(df: pd.DataFrame, target: str, feature_fn) -> tuple[list[float], dict]:
    rows = df.loc[~df["oom"].astype(bool)].copy()
    rows = rows[np.isfinite(rows[target].astype(float))]
    if rows.empty:
        return [], {"r2": 0.0, "mae": 0.0, "heldout_mae": 0.0, "heldout_cases": 0}
    x = np.asarray([feature_fn(r) for _, r in rows.iterrows()], dtype=float)
    y = rows[target].astype(float).to_numpy()
    idx = np.arange(len(rows))
    heldout = idx % 5 == 0 if len(rows) >= 10 else np.zeros(len(rows), dtype=bool)
    train = ~heldout
    if not train.any():
        train = np.ones(len(rows), dtype=bool)
    coef, *_ = np.linalg.lstsq(x[train], y[train], rcond=None)
    pred = x @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    mae = float(np.mean(np.abs(y - pred)))
    heldout_mae = float(np.mean(np.abs(y[heldout] - pred[heldout]))) if heldout.any() else mae
    return [float(v) for v in coef], {
        "r2": 1.0 - ss_res / ss_tot if ss_tot else 1.0,
        "mae": mae,
        "heldout_mae": heldout_mae,
        "heldout_cases": int(heldout.sum()),
    }


def _summarize(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw
    keys = ["prefix_len", "private_len", "output_len", "batch_size"]
    rows = []
    for vals, sub in raw.groupby(keys, dropna=False):
        row = dict(zip(keys, vals))
        row["oom"] = bool(sub["oom"].astype(bool).all())
        for col in [
            "full_prefill_ms",
            "prefix_prefill_ms",
            "extend_prefill_ms",
            "decode_ms",
            "decode_tpot_ms",
            "peak_gpu_mem_bytes",
        ]:
            values = sub.loc[~sub["oom"].astype(bool), col].astype(float)
            row[f"{col}_median"] = float(values.median()) if not values.empty else 0.0
            row[f"{col}_p90"] = float(values.quantile(0.90)) if not values.empty else 0.0
            row[f"{col}_p99"] = float(values.quantile(0.99)) if not values.empty else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _fit(summary: pd.DataFrame, out: Path, model_name: str, model_path: str) -> None:
    fit_df = summary.loc[~summary["oom"].astype(bool)].copy()
    fit_df["full_prefill_ms"] = fit_df["full_prefill_ms_median"]
    fit_df["extend_prefill_ms"] = fit_df["extend_prefill_ms_median"]
    fit_df["decode_tpot_ms"] = fit_df["decode_tpot_ms_median"]
    full_coef, full_q = _fit_linear(
        fit_df,
        "full_prefill_ms",
        lambda r: [
            1.0,
            float(r["prefix_len"]) + float(r["private_len"]),
            (float(r["prefix_len"]) + float(r["private_len"])) ** 2,
            float(r["batch_size"]),
            (float(r["prefix_len"]) + float(r["private_len"])) * float(r["batch_size"]),
        ],
    )
    extend_coef, extend_q = _fit_linear(
        fit_df,
        "extend_prefill_ms",
        lambda r: [
            1.0,
            float(r["prefix_len"]),
            float(r["private_len"]),
            float(r["batch_size"]),
            float(r["prefix_len"]) * float(r["private_len"]),
            float(r["private_len"]) ** 2,
            float(r["prefix_len"]) * float(r["batch_size"]),
        ],
    )
    decode_coef, decode_q = _fit_linear(
        fit_df,
        "decode_tpot_ms",
        lambda r: [
            1.0,
            float(r["prefix_len"]) + float(r["private_len"]),
            float(r["output_len"]),
            float(r["batch_size"]),
            (float(r["prefix_len"]) + float(r["private_len"])) * float(r["batch_size"]),
        ],
    )
    fit = {
        "schema_version": "round4.prefix_extension.1",
        "model_name": model_name,
        "model_path": model_path,
        "full_prefill_features": ["1", "input_tokens", "input_tokens2", "batch_size", "input_tokens_batch"],
        "full_prefill_coef": full_coef,
        "extend_prefill_features": ["1", "prefix_len", "private_len", "batch_size", "prefix_private", "private2", "prefix_batch"],
        "extend_prefill_coef": extend_coef,
        "decode_tpot_features": ["1", "context_tokens", "output_tokens", "batch_size", "context_batch"],
        "decode_tpot_coef": decode_coef,
    }
    write_json(out / "prefix_extension_fit.json", fit)
    write_json(out / "calibration" / "prefix_extension_fit.json", fit)
    quality = {
        "full_prefill_r2": full_q["r2"],
        "extend_prefill_r2": extend_q["r2"],
        "decode_tpot_r2": decode_q["r2"],
        "full_prefill_mae_ms": full_q["mae"],
        "extend_prefill_mae_ms": extend_q["mae"],
        "decode_tpot_mae_ms": decode_q["mae"],
        "heldout_cases": max(full_q["heldout_cases"], extend_q["heldout_cases"], decode_q["heldout_cases"]),
        "fit_hash": file_sha256(out / "prefix_extension_fit.json"),
    }
    write_json(out / "prefix_extension_fit_quality.json", quality)
    write_json(out / "calibration" / "prefix_extension_fit_quality.json", quality)


def _plot(summary: pd.DataFrame, out: Path) -> None:
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    valid = summary.loc[~summary["oom"].astype(bool)].copy()
    if valid.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for private, sub in valid.groupby("private_len"):
        sub.groupby("prefix_len")["extend_prefill_ms_median"].mean().plot(ax=ax, marker="o", label=f"private={private}")
    ax.set_xlabel("prefix tokens")
    ax.set_ylabel("extend prefill ms median")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_prefix_extension_fit.png", dpi=180)
    fig.savefig(fig_dir / "fig_prefix_extension_fit.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="auto")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--out", default="results/round4_prefix_extension_calibration")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--prefix-lens", default="")
    parser.add_argument("--private-lens", default="")
    parser.add_argument("--output-lens", default="")
    parser.add_argument("--batch-sizes", default="")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--clean-required", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    enforce_clean_git_tree(args.clean_required, args.allow_dirty)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    out = init_run_dir(
        args.out,
        {
            "run_type": "prefix_extension_calibration",
            "model": args.model,
            "gpus": args.gpus,
            "seed": args.seed,
            "clean_required": bool(args.clean_required),
        },
    )
    model_name, model_path = _choose_model(args.model)
    write_json(out / "model_selection.json", {"engine_used": "hf_forward_pastkv", "model_name": model_name, "model_path": model_path})
    prefix_lens = _parse_ints(args.prefix_lens, [0, 512, 2048, 8192, 16384, 32768])
    private_lens = _parse_ints(args.private_lens, [1, 64, 256, 1024, 4096])
    output_lens = _parse_ints(args.output_lens, [1, 16, 64, 128])
    batch_sizes = _parse_ints(args.batch_sizes, [1, 2, 4, 8])
    cases = [(p, q, o, b) for p in prefix_lens for q in private_lens for o in output_lens for b in batch_sizes]
    if args.max_cases:
        cases = cases[: args.max_cases]
    gpu_id = _visible_gpu_id(args.gpus)
    timer = PrefixExtensionTimer(model_path=model_path, gpu_id=gpu_id, dtype=args.dtype, seed=args.seed)
    raw_rows: list[dict] = []
    try:
        for prefix_len, private_len, output_len, batch_size in cases:
            for rep in range(args.reps):
                result = timer.run_extension_case(prefix_len, private_len, output_len, batch_size, rep, warmup=args.warmup)
                row = result.to_dict()
                row.update({"model_name": model_name, "model_path": model_path, "gpu_id": gpu_id, "requested_gpus": args.gpus, "seed": args.seed})
                raw_rows.append(row)
                pd.DataFrame(raw_rows).to_csv(out / "prefix_extension_raw.partial.csv", index=False)
                print(
                    f"case prefix={prefix_len} private={private_len} output={output_len} batch={batch_size} rep={rep} "
                    f"oom={row['oom']} extend_ms={row['extend_prefill_ms']:.3f}",
                    flush=True,
                )
    finally:
        timer.close()
    raw = pd.DataFrame(raw_rows)
    summary = _summarize(raw)
    raw.to_csv(out / "prefix_extension_raw.csv", index=False)
    raw.to_csv(out / "calibration" / "prefix_extension_raw.csv", index=False)
    summary.to_csv(out / "prefix_extension_summary.csv", index=False)
    summary.to_csv(out / "calibration" / "prefix_extension_summary.csv", index=False)
    _fit(summary, out, model_name, model_path)
    _plot(summary, out)
    finalize_run_dir(out)
    print(f"Prefix extension calibration complete: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
