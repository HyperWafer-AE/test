#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from waferagent.calibration import fit_prefill_decode
from waferagent.llm_runner import RunnerConfig
from waferagent.llm_runner import make_runner
from waferagent.model_discovery import load_or_scan, select_model
from waferagent.real_benchmark import BenchmarkCase, run_case_with_runner
from waferagent.utils import append_text, init_run_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="auto")
    parser.add_argument("--engine", default="hf", choices=["hf", "vllm", "synthetic"])
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--out", default="results/h100_calibration_real_hf")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--allow-synthetic-fallback", action="store_true")
    parser.add_argument("--max-cases", type=int, default=0, help="0 means full matrix")
    args = parser.parse_args()

    out = init_run_dir(args.out, {"run_type": "h100_calibration_real", "engine": args.engine, "model": args.model, "gpus": args.gpus, "seed": args.seed})
    index = load_or_scan("/data2/model_zoo", "configs/models.local.json")
    chosen = select_model(index) if args.model == "auto" else next(
        (m for m in index.get("models", []) if args.model in m.get("name", "") or args.model in m.get("path", "")),
        None,
    )
    engine = args.engine
    if engine != "synthetic" and not chosen:
        append_text(out / "environment.txt", f"No loadable local model found for {args.model}\n")
        if not args.allow_synthetic_fallback:
            raise SystemExit(2)
        engine = "synthetic"
    model_name = chosen["name"] if chosen else "synthetic"
    model_path = chosen["path"] if chosen else ""
    runner_cfg = RunnerConfig(engine=engine, model_name=model_name, model_path=model_path, seed=args.seed)
    try:
        runner = make_runner(runner_cfg)
    except Exception as exc:
        append_text(out / "environment.txt", f"runner init failed: {exc}\n")
        if not args.allow_synthetic_fallback:
            raise SystemExit(2)
        engine = "synthetic"
        runner_cfg = RunnerConfig(engine="synthetic", model_name="synthetic", model_path="", seed=args.seed)
        runner = make_runner(runner_cfg)
    input_lens = [128, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    output_lens = [1, 16, 32, 128, 256]
    batch_sizes = [1, 2, 4, 8, 16]
    rows = []
    failures = []
    cases = [BenchmarkCase(i, o, b, args.seed) for i in input_lens for o in output_lens for b in batch_sizes]
    if args.max_cases:
        cases = cases[: args.max_cases]
    for case in cases:
        try:
            traces = run_case_with_runner(out.name, runner, case)
            totals = [tr.total_ms for tr in traces]
            ttft = max(tr.ttft_ms for tr in traces)
            decode = max(tr.decode_ms for tr in traces)
            peak = max((tr.peak_gpu_mem_bytes or 0) for tr in traces)
            rows.append(
                {
                    "engine": engine,
                    "model_name": model_name,
                    "model_path": model_path,
                    "gpu_id": 0,
                    "input_len": case.input_len,
                    "output_len": case.output_len,
                    "batch_size": case.batch_size,
                    "seed": case.seed,
                    "ttft_ms": ttft,
                    "tpot_ms": decode / max(1, case.output_len - 1),
                    "decode_ms": decode,
                    "total_ms": max(totals),
                    "prefill_tokens_per_s": case.input_len * case.batch_size / max(1e-9, ttft / 1000.0),
                    "decode_tokens_per_s": case.output_len * case.batch_size / max(1e-9, decode / 1000.0),
                    "peak_gpu_mem_bytes": peak,
                    "oom": False,
                }
            )
        except RuntimeError as exc:
            msg = str(exc)
            is_oom = "out of memory" in msg.lower() or "cuda" in msg.lower()
            failures.append({"input_len": case.input_len, "output_len": case.output_len, "batch_size": case.batch_size, "error": msg, "oom": is_oom})
            append_text(out / "environment.txt", f"case failed: {failures[-1]}\n")
            if not is_oom and not args.allow_synthetic_fallback:
                raise
    df = pd.DataFrame(rows)
    csv_path = out / "h100_prefill_decode.csv"
    df.to_csv(csv_path, index=False)
    calib_dir_csv = out / "calibration" / "h100_prefill_decode.csv"
    calib_dir_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(calib_dir_csv, index=False)
    fit = fit_prefill_decode(csv_path, out / "h100_fit.json") if not df.empty else {}
    fit2 = out / "calibration" / "h100_fit.json"
    if fit:
        fit2.write_text((out / "h100_fit.json").read_text(encoding="utf-8"), encoding="utf-8")
    write_json(out / "failures.json", failures)
    write_json(out / "model_selection.json", {"engine_used": engine, "model_name": model_name, "model_path": model_path})
    if not df.empty:
        fig_dir = out / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        for batch, sub in df.groupby("batch_size"):
            sub.groupby("input_len")["ttft_ms"].mean().plot(ax=ax, marker="o", label=f"b={batch}")
        ax.set_xlabel("input tokens")
        ax.set_ylabel("TTFT ms")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig_prefill_fit.png", dpi=180)
        fig.savefig(fig_dir / "fig_prefill_fit.pdf")
        fig.savefig(fig_dir / "fig_h100_prefill_decode_calibration.png", dpi=180)
        fig.savefig(fig_dir / "fig_h100_prefill_decode_calibration.pdf")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 4))
        for batch, sub in df.groupby("batch_size"):
            sub.groupby("output_len")["tpot_ms"].mean().plot(ax=ax, marker="o", label=f"b={batch}")
        ax.set_xlabel("output tokens")
        ax.set_ylabel("TPOT ms")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig_decode_tpot.png", dpi=180)
        fig.savefig(fig_dir / "fig_decode_tpot.pdf")
        plt.close(fig)
    (out / "report.md").write_text(
        "# H100 Calibration Report\n\n"
        f"- engine: `{engine}`\n- model: `{model_name}`\n- completed cases: `{len(rows)}`\n- failures: `{len(failures)}`\n",
        encoding="utf-8",
    )
    print(f"Calibration complete: {Path(out).resolve()} cases={len(rows)} failures={len(failures)}")


if __name__ == "__main__":
    main()
