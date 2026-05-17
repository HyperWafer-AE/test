#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from waferagent.kv_model import ModelKVConfig
from waferagent.llm_runner import RunnerConfig
from waferagent.metrics import write_characterization_tables
from waferagent.model_discovery import load_or_scan, select_model
from waferagent.plotting import (
    plot_critical_path,
    plot_dag_examples,
    plot_kv_duplication,
    plot_latency_breakdown,
    plot_shared_prefix_ratio,
)
from waferagent.trace_collector import collect_graph_traces
from waferagent.trace_schema import write_traces
from waferagent.utils import append_text, init_run_dir, write_json
from waferagent.workloads import generate_workload_set


def _resolve_model(engine: str, model: str) -> tuple[str, str, ModelKVConfig, str | None]:
    if engine == "synthetic":
        return "synthetic", "", ModelKVConfig(), None
    index = load_or_scan("/data2/model_zoo", "configs/models.local.json")
    chosen = select_model(index) if model == "auto" else next(
        (m for m in index.get("models", []) if model in m.get("name", "") or model in m.get("path", "")),
        None,
    )
    if not chosen:
        return "synthetic", "", ModelKVConfig(), f"No local model found for --model {model}; using synthetic fallback."
    return chosen["name"], chosen["path"], ModelKVConfig.from_model_index_item(chosen), None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="auto")
    parser.add_argument("--engine", default="synthetic", choices=["synthetic", "hf", "vllm"])
    parser.add_argument("--workloads", default="debate,moa,planner_worker_tool,swe_like,rag_like")
    parser.add_argument("--num-jobs", type=int, default=20)
    parser.add_argument("--gpus", default="")
    parser.add_argument("--concurrency", default="1")
    parser.add_argument("--out", default="results/characterization")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    out = init_run_dir(
        args.out,
        {
            "run_type": "trace_collection",
            "engine": args.engine,
            "model": args.model,
            "gpus": args.gpus,
            "concurrency": args.concurrency,
            "seed": args.seed,
        },
    )
    workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]
    graphs = generate_workload_set(workloads, args.num_jobs, seed=args.seed)
    model_name, model_path, model_cfg, warning = _resolve_model(args.engine, args.model)
    engine = args.engine
    if warning:
        append_text(out / "environment.txt", f"\n{warning}\n")
        engine = "synthetic"
    runner_cfg = RunnerConfig(engine=engine, model_name=model_name, model_path=model_path)
    try:
        traces = collect_graph_traces(graphs, out.name, runner_cfg, out / "traces" / "traces.jsonl")
    except Exception as exc:
        append_text(out / "environment.txt", f"\n{engine} trace collection failed: {exc}\nFalling back to synthetic.\n")
        runner_cfg = RunnerConfig(engine="synthetic", model_name="synthetic", model_path="")
        traces = collect_graph_traces(graphs, out.name, runner_cfg, out / "traces" / "traces.jsonl")
    write_traces(out / "traces" / "traces.jsonl", traces)
    write_json(out / "model_selection.json", {"model_name": model_name, "model_path": model_path, "engine_used": runner_cfg.engine})
    tables = write_characterization_tables(graphs, traces, out / "simulation", model_cfg)
    fig = out / "figures"
    plot_dag_examples(graphs, fig / "fig_workload_dag_examples")
    plot_shared_prefix_ratio(tables["characterization_token_stats.csv"], fig / "fig_shared_prefix_ratio")
    plot_kv_duplication(tables["characterization_kv_stats.csv"], fig / "fig_kv_duplication_ratio")
    plot_latency_breakdown(tables["characterization_latency_breakdown.csv"], fig / "fig_latency_breakdown")
    plot_critical_path(tables["characterization_critical_path.csv"], fig / "fig_critical_path_vs_total_work")
    print(f"Trace collection complete: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
