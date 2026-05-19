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

from waferagent.h100_forward_timer import H100ForwardTimer, summarize_forward_rows
from waferagent.llm_runner import RunnerConfig, make_runner
from waferagent.model_discovery import load_or_scan, select_model
from waferagent.real_benchmark import BenchmarkCase, run_case_with_runner
from waferagent.utils import (
    append_text,
    enforce_clean_git_tree,
    finalize_run_dir,
    init_run_dir,
    write_json,
)


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
        return [], {"r2": 0.0, "mae": 0.0, "heldout_cases": 0}
    features = np.asarray([feature_fn(r) for _, r in rows.iterrows()], dtype=float)
    y = rows[target].astype(float).to_numpy()
    idx = np.arange(len(rows))
    heldout_mask = idx % 5 == 0 if len(rows) >= 10 else np.zeros(len(rows), dtype=bool)
    train = ~heldout_mask
    if not train.any():
        train = np.ones(len(rows), dtype=bool)
    coef, *_ = np.linalg.lstsq(features[train], y[train], rcond=None)
    pred = features @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    mae = float(np.mean(np.abs(y - pred)))
    heldout_mae = float(np.mean(np.abs(y[heldout_mask] - pred[heldout_mask]))) if heldout_mask.any() else mae
    return [float(x) for x in coef], {
        "r2": 1.0 - ss_res / ss_tot if ss_tot else 1.0,
        "mae": mae,
        "heldout_mae": heldout_mae,
        "heldout_cases": int(heldout_mask.sum()),
    }


def _fit_and_write(summary: pd.DataFrame, out: Path, model_name: str, model_path: str, engine: str) -> dict:
    fit_df = summary.loc[~summary["oom"].astype(bool)].copy()
    fit_df["prefill_ms"] = fit_df["prefill_ms_median"]
    fit_df["tpot_ms"] = fit_df["tpot_ms_median"]

    prefill_coef, prefill_quality = _fit_linear(
        fit_df,
        "prefill_ms",
        lambda r: [
            1.0,
            float(r["input_len"]),
            float(r["input_len"]) ** 2,
            float(r["batch_size"]),
            float(r["input_len"]) * float(r["batch_size"]),
        ],
    )
    decode_coef, decode_quality = _fit_linear(
        fit_df,
        "tpot_ms",
        lambda r: [
            1.0,
            float(r["input_len"]),
            float(r["output_len"]),
            float(r["batch_size"]),
            float(r["input_len"]) * float(r["batch_size"]),
        ],
    )
    fit = {
        "schema_version": "round3.forward.1",
        "engine": engine,
        "model_name": model_name,
        "model_path": model_path,
        "prefill_features": ["1", "input_tokens", "input_tokens2", "batch_size", "input_tokens_batch"],
        "prefill_coef": prefill_coef,
        "decode_features": ["1", "context_tokens", "output_tokens", "batch_size", "context_batch"],
        "decode_tpot_coef": decode_coef,
    }
    quality = {
        "prefill_r2": prefill_quality["r2"],
        "decode_r2": decode_quality["r2"],
        "prefill_mae_ms": prefill_quality["mae"],
        "decode_mae_ms": decode_quality["mae"],
        "prefill_heldout_mae_ms": prefill_quality["heldout_mae"],
        "decode_heldout_mae_ms": decode_quality["heldout_mae"],
        "heldout_cases": max(prefill_quality["heldout_cases"], decode_quality["heldout_cases"]),
    }
    write_json(out / "h100_fit.json", fit)
    write_json(out / "calibration" / "h100_fit.json", fit)
    write_json(out / "h100_fit_quality.json", quality)
    write_json(out / "calibration" / "h100_fit_quality.json", quality)
    return fit


def _plot(summary: pd.DataFrame, out: Path) -> None:
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    valid = summary.loc[~summary["oom"].astype(bool)].copy()
    if valid.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for batch, sub in valid.groupby("batch_size"):
        sub.groupby("input_len")["prefill_ms_median"].mean().plot(ax=ax, marker="o", label=f"b={batch}")
    ax.set_xlabel("input tokens")
    ax.set_ylabel("prefill ms median")
    ax.legend(fontsize=7)
    fig.tight_layout()
    for name in ["fig_prefill_fit", "fig_h100_prefill_decode_calibration"]:
        fig.savefig(fig_dir / f"{name}.png", dpi=180)
        fig.savefig(fig_dir / f"{name}.pdf")
    plt.close(fig)


def _write_coverage_reports(summary: pd.DataFrame, out: Path, args: argparse.Namespace) -> None:
    default_inputs = [128, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    default_outputs = [1, 16, 32, 128, 256]
    default_batches = [1, 2, 4, 8, 16]
    inputs = _parse_ints(args.input_lens, default_inputs)
    outputs = _parse_ints(args.output_lens, default_outputs)
    batches = _parse_ints(args.batch_sizes, default_batches)
    num_cases = len(inputs) * len(outputs) * len(batches)
    if args.max_cases:
        num_cases = min(num_cases, args.max_cases)
    oom_rows = summary.loc[summary["oom"].astype(bool)] if not summary.empty else pd.DataFrame()
    coverage = {
        "input_lens_covered": sorted(int(x) for x in summary.get("input_len", pd.Series(dtype=int)).dropna().unique()),
        "output_lens_covered": sorted(int(x) for x in summary.get("output_len", pd.Series(dtype=int)).dropna().unique()),
        "batch_sizes_covered": sorted(int(x) for x in summary.get("batch_size", pd.Series(dtype=int)).dropna().unique()),
        "num_cases": int(num_cases),
        "num_raw_rows": int(len(summary) * max(1, args.reps)),
        "num_summary_rows": int(len(summary)),
        "num_oom_cases": int(len(oom_rows)),
        "is_full_matrix": inputs == default_inputs
        and outputs == default_outputs
        and batches == default_batches
        and not args.max_cases,
        "is_stratified_matrix": not (
            inputs == default_inputs
            and outputs == default_outputs
            and batches == default_batches
            and not args.max_cases
        ),
    }
    write_json(out / "coverage_report.json", coverage)
    write_json(out / "calibration" / "coverage_report.json", coverage)
    oom_report = {
        "num_oom_cases": int(len(oom_rows)),
        "oom_cases": oom_rows[["input_len", "output_len", "batch_size"]].to_dict("records")
        if not oom_rows.empty
        else [],
    }
    write_json(out / "oom_report.json", oom_report)
    write_json(out / "calibration" / "oom_report.json", oom_report)

    fig, ax = plt.subplots(figsize=(6, 4))
    for batch, sub in valid.groupby("batch_size"):
        sub.groupby("input_len")["tpot_ms_median"].mean().plot(ax=ax, marker="o", label=f"b={batch}")
    ax.set_xlabel("context tokens")
    ax.set_ylabel("decode TPOT ms median")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_decode_tpot.png", dpi=180)
    fig.savefig(fig_dir / "fig_decode_tpot.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for batch, sub in valid.groupby("batch_size"):
        sub.groupby("input_len")["peak_gpu_mem_bytes_median"].mean().plot(ax=ax, marker="o", label=f"b={batch}")
    ax.set_xlabel("input tokens")
    ax.set_ylabel("peak GPU memory bytes")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_memory_scaling.png", dpi=180)
    fig.savefig(fig_dir / "fig_memory_scaling.pdf")
    plt.close(fig)


def _hf_forward(args, out: Path, model_name: str, model_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    gpu_id = _visible_gpu_id(args.gpus)
    timer = H100ForwardTimer(model_path=model_path, gpu_id=gpu_id, dtype=args.dtype, seed=args.seed)
    raw_rows: list[dict] = []
    cases = [
        (i, o, b)
        for i in _parse_ints(args.input_lens, [128, 512, 1024, 2048, 4096, 8192, 16384, 32768])
        for o in _parse_ints(args.output_lens, [1, 16, 32, 128, 256])
        for b in _parse_ints(args.batch_sizes, [1, 2, 4, 8, 16])
    ]
    if args.max_cases:
        cases = cases[: args.max_cases]
    try:
        for input_len, output_len, batch_size in cases:
            for rep in range(args.reps):
                result = timer.run_case(input_len, output_len, batch_size, rep, warmup=args.warmup)
                row = result.to_dict()
                row.update(
                    {
                        "engine": "hf",
                        "model_name": model_name,
                        "model_path": model_path,
                        "gpu_id": gpu_id,
                        "seed": args.seed,
                        "prefill_tokens_per_s": input_len * batch_size / max(1e-9, row["prefill_ms"] / 1000.0)
                        if not row["oom"]
                        else 0.0,
                        "decode_tokens_per_s": output_len * batch_size / max(1e-9, row["decode_ms"] / 1000.0)
                        if not row["oom"]
                        else 0.0,
                    }
                )
                raw_rows.append(row)
                pd.DataFrame(raw_rows).to_csv(out / "h100_prefill_decode_raw.partial.csv", index=False)
                if row["oom"]:
                    append_text(out / "environment.txt", f"OOM case recorded: {row}\n")
                print(
                    "case "
                    f"input={input_len} output={output_len} batch={batch_size} rep={rep} "
                    f"oom={row['oom']} prefill_ms={row['prefill_ms']:.3f} decode_ms={row['decode_ms']:.3f}",
                    flush=True,
                )
    finally:
        timer.close()
    raw = pd.DataFrame(raw_rows)
    summary = pd.DataFrame(summarize_forward_rows(raw_rows))
    for df, name in [(raw, "h100_prefill_decode_raw.csv"), (summary, "h100_prefill_decode_summary.csv")]:
        df.to_csv(out / name, index=False)
        df.to_csv(out / "calibration" / name, index=False)
    summary.to_csv(out / "h100_prefill_decode.csv", index=False)
    summary.to_csv(out / "calibration" / "h100_prefill_decode.csv", index=False)
    return raw, summary


def _fallback_runner_matrix(args, out: Path, model_name: str, model_path: str, engine: str) -> pd.DataFrame:
    runner = make_runner(RunnerConfig(engine=engine, model_name=model_name, model_path=model_path, seed=args.seed))
    cases = [
        BenchmarkCase(i, o, b, args.seed)
        for i in _parse_ints(args.input_lens, [128, 512, 1024])
        for o in _parse_ints(args.output_lens, [1, 16, 32])
        for b in _parse_ints(args.batch_sizes, [1, 2])
    ]
    if args.max_cases:
        cases = cases[: args.max_cases]
    rows = []
    for case in cases:
        traces = run_case_with_runner(out.name, runner, case)
        ttft = max(tr.ttft_ms for tr in traces)
        decode = max(tr.decode_ms for tr in traces)
        rows.append(
            {
                "engine": engine,
                "model_name": model_name,
                "model_path": model_path,
                "gpu_id": int(args.gpus.split(",")[0]) if args.gpus else 0,
                "requested_gpus": args.gpus,
                "input_len": case.input_len,
                "output_len": case.output_len,
                "batch_size": case.batch_size,
                "rep": 0,
                "prefill_ms": ttft,
                "decode_ms": decode,
                "tpot_ms": decode / max(1, case.output_len),
                "total_ms": max(tr.total_ms for tr in traces),
                "peak_gpu_mem_bytes": max((tr.peak_gpu_mem_bytes or 0) for tr in traces),
                "oom": False,
                "dtype": "unavailable",
                "device": "vllm" if engine == "vllm" else "synthetic",
                "seed": args.seed,
            }
        )
    raw = pd.DataFrame(rows)
    summary = pd.DataFrame(summarize_forward_rows(raw.to_dict("records")))
    raw.to_csv(out / "h100_prefill_decode_raw.csv", index=False)
    summary.to_csv(out / "h100_prefill_decode_summary.csv", index=False)
    summary.to_csv(out / "h100_prefill_decode.csv", index=False)
    raw.to_csv(out / "calibration" / "h100_prefill_decode_raw.csv", index=False)
    summary.to_csv(out / "calibration" / "h100_prefill_decode_summary.csv", index=False)
    summary.to_csv(out / "calibration" / "h100_prefill_decode.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="auto")
    parser.add_argument("--engine", default="hf", choices=["hf", "vllm", "synthetic"])
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--out", default="results/h100_calibration_real_hf")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--allow-synthetic-fallback", action="store_true")
    parser.add_argument("--clean-required", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--max-cases", type=int, default=0, help="0 means full matrix")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--input-lens", default="")
    parser.add_argument("--output-lens", default="")
    parser.add_argument("--batch-sizes", default="")
    args = parser.parse_args()

    enforce_clean_git_tree(args.clean_required, args.allow_dirty)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    out = init_run_dir(
        args.out,
        {
            "run_type": "h100_calibration_real",
            "engine": args.engine,
            "model": args.model,
            "gpus": args.gpus,
            "seed": args.seed,
            "forward_based": args.engine == "hf",
            "clean_required": bool(args.clean_required),
        },
    )
    try:
        model_name, model_path = _choose_model(args.model)
    except Exception as exc:
        append_text(out / "environment.txt", f"model selection failed: {exc}\n")
        if not args.allow_synthetic_fallback:
            finalize_run_dir(out)
            raise SystemExit(2)
        model_name, model_path = "synthetic", ""
        args.engine = "synthetic"

    write_json(out / "model_selection.json", {"engine_used": args.engine, "model_name": model_name, "model_path": model_path})
    try:
        if args.engine == "hf":
            raw, summary = _hf_forward(args, out, model_name, model_path)
        else:
            summary = _fallback_runner_matrix(args, out, model_name, model_path, args.engine)
    except Exception as exc:
        append_text(out / "environment.txt", f"calibration failed: {exc}\n")
        if args.allow_synthetic_fallback and args.engine != "synthetic":
            args.engine = "synthetic"
            model_name, model_path = "synthetic", ""
            summary = _fallback_runner_matrix(args, out, model_name, model_path, "synthetic")
            write_json(
                out / "model_selection.json",
                {"engine_used": "synthetic", "model_name": model_name, "model_path": model_path, "fallback_used": True, "failure": str(exc)},
            )
        else:
            finalize_run_dir(out)
            raise

    _fit_and_write(summary, out, model_name, model_path, args.engine)
    _plot(summary, out)
    _write_coverage_reports(summary, out, args)
    impossible = summary.loc[
        (~summary["oom"].astype(bool))
        & (
            (summary["prefill_ms_median"] < 0)
            | (summary["decode_ms_median"] < 0)
            | (summary["total_ms_median"] < summary["prefill_ms_median"])
        )
    ]
    write_json(out / "timing_sanity.json", {"impossible_rows": int(len(impossible))})
    (out / "report.md").write_text(
        "# H100 Forward Calibration Report\n\n"
        f"- engine: `{args.engine}`\n"
        f"- model: `{model_name}`\n"
        f"- summary rows: `{len(summary)}`\n"
        f"- OOM rows: `{int(summary['oom'].astype(bool).sum()) if not summary.empty else 0}`\n"
        f"- impossible timing rows: `{len(impossible)}`\n",
        encoding="utf-8",
    )
    finalize_run_dir(out)
    print(f"Calibration complete: {Path(out).resolve()} summary_rows={len(summary)}")


if __name__ == "__main__":
    main()
