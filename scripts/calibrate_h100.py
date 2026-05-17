#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from waferagent.calibration import fit_prefill_decode
from waferagent.graph_ir import AgentGraph
from waferagent.llm_runner import RunnerConfig, make_runner
from waferagent.model_discovery import load_or_scan, select_model
from waferagent.utils import append_text, init_run_dir, write_json
from waferagent.workloads import WorkloadParams, generate_workload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="auto")
    parser.add_argument("--engine", default="synthetic", choices=["synthetic", "hf", "vllm"])
    parser.add_argument("--gpus", default="")
    parser.add_argument("--out", default="results/h100_calibration")
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    out = init_run_dir(args.out, {"run_type": "h100_calibration", "engine": args.engine, "model": args.model})
    engine = args.engine
    model_name = "synthetic"
    model_path = ""
    if engine != "synthetic":
        index = load_or_scan("/data2/model_zoo", "configs/models.local.json")
        chosen = select_model(index)
        if chosen:
            model_name, model_path = chosen["name"], chosen["path"]
        else:
            append_text(out / "environment.txt", "\nNo local HF model found; falling back to synthetic calibration.\n")
            engine = "synthetic"

    runner_cfg = RunnerConfig(engine=engine, model_name=model_name, model_path=model_path)
    try:
        runner = make_runner(runner_cfg)
    except Exception as exc:
        append_text(out / "environment.txt", f"\n{engine} runner failed: {exc}\nFalling back to synthetic calibration.\n")
        engine = "synthetic"
        runner = make_runner(RunnerConfig(engine="synthetic"))

    rows = []
    input_lens = [128, 512, 1024, 2048, 4096, 8192]
    output_lens = [1, 32, 128, 256]
    batch_sizes = [1, 2, 4, 8]
    for input_len in input_lens:
        for output_len in output_lens:
            for batch_size in batch_sizes:
                params = WorkloadParams(
                    workload="debate",
                    job_id=f"calib_i{input_len}_o{output_len}_b{batch_size}",
                    seed=args.seed,
                    num_agents=1,
                    input_len=input_len,
                    output_len=output_len,
                )
                graph: AgentGraph = generate_workload(params)
                node = graph.nodes[graph.topological_order()[0]]
                tr = runner.run_node(out.name, "calibration", node)
                rows.append(
                    {
                        "engine": engine,
                        "model_name": model_name,
                        "input_len": input_len,
                        "output_len": output_len,
                        "batch_size": batch_size,
                        "ttft_ms": tr.ttft_ms,
                        "decode_ms": tr.decode_ms,
                        "total_ms": tr.total_ms,
                    }
                )
    df = pd.DataFrame(rows)
    csv_path = out / "calibration" / "h100_prefill_decode.csv"
    df.to_csv(csv_path, index=False)
    fit = fit_prefill_decode(csv_path, out / "calibration" / "h100_fit.json")
    write_json(out / "calibration" / "calibration_summary.json", {"fit": fit, "engine_used": engine})
    print(f"Calibration complete: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
