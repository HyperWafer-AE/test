#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

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
from waferagent.utils import append_text, enforce_clean_git_tree, finalize_run_dir, init_run_dir, write_json
from waferagent.workloads import generate_workload_set
from waferagent.cohort_scheduler import CohortConfig, form_decode_cohorts
from waferagent.shared_kv import extract_shared_kv_objects
from waferagent.stage_ir import build_stages


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
    parser.add_argument("--allow-synthetic-fallback", action="store_true")
    parser.add_argument("--clean-required", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    enforce_clean_git_tree(args.clean_required, args.allow_dirty)
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    out = init_run_dir(
        args.out,
        {
            "run_type": "trace_collection",
            "engine": args.engine,
            "model": args.model,
            "gpus": args.gpus,
            "concurrency": args.concurrency,
            "seed": args.seed,
            "clean_required": bool(args.clean_required),
        },
    )
    workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]
    graphs = generate_workload_set(workloads, args.num_jobs, seed=args.seed)
    model_name, model_path, model_cfg, warning = _resolve_model(args.engine, args.model)
    engine = args.engine
    if warning:
        append_text(out / "environment.txt", f"\n{warning}\n")
        if args.allow_synthetic_fallback:
            engine = "synthetic"
        else:
            finalize_run_dir(out)
            raise SystemExit(2)
    runner_cfg = RunnerConfig(engine=engine, model_name=model_name, model_path=model_path)
    try:
        traces = collect_graph_traces(graphs, out.name, runner_cfg, out / "traces" / "traces.jsonl")
    except Exception as exc:
        append_text(out / "environment.txt", f"\n{engine} trace collection failed: {exc}\n")
        if not args.allow_synthetic_fallback:
            write_json(out / "model_selection.json", {"model_name": model_name, "model_path": model_path, "engine_used": engine, "failure": str(exc), "fallback_used": False})
            finalize_run_dir(out)
            raise SystemExit(2)
        append_text(out / "environment.txt", "Falling back to synthetic because --allow-synthetic-fallback was set.\n")
        runner_cfg = RunnerConfig(engine="synthetic", model_name="synthetic", model_path="")
        traces = collect_graph_traces(graphs, out.name, runner_cfg, out / "traces" / "traces.jsonl")
        for tr in traces:
            tr.fallback_used = True
    write_traces(out / "traces" / "traces.jsonl", traces)
    vllm_rows = []
    for tr in traces:
        meta = tr.metadata or {}
        if tr.engine == "vllm" and meta.get("batch_layer_id"):
            vllm_rows.append(
                {
                    "job_id": tr.job_id,
                    "node_id": tr.node_id,
                    "batch_layer_id": meta.get("batch_layer_id"),
                    "batch_layer_walltime_ms": meta.get("batch_layer_walltime_ms"),
                    "batch_size": meta.get("batch_size"),
                    "num_prompts": meta.get("num_prompts"),
                    "prompt_tokens_total": meta.get("prompt_tokens_total"),
                    "completion_tokens_total": meta.get("completion_tokens_total"),
                    "tokens_per_second": meta.get("tokens_per_second"),
                    "timing_source": tr.timing_source,
                    "timing_scope": tr.timing_scope,
                    "timing_quality": tr.timing_quality,
                }
            )
    if vllm_rows:
        pd.DataFrame(vllm_rows).drop_duplicates(subset=["batch_layer_id"]).to_csv(
            out / "simulation" / "vllm_batch_layer_metrics.csv", index=False
        )
    else:
        pd.DataFrame(
            columns=[
                "job_id",
                "node_id",
                "batch_layer_id",
                "batch_layer_walltime_ms",
                "batch_size",
                "num_prompts",
                "prompt_tokens_total",
                "completion_tokens_total",
                "tokens_per_second",
                "timing_source",
                "timing_scope",
                "timing_quality",
            ]
        ).to_csv(out / "simulation" / "vllm_batch_layer_metrics.csv", index=False)
    write_json(out / "model_selection.json", {"model_name": model_name, "model_path": model_path, "engine_used": runner_cfg.engine, "fallback_count": sum(1 for tr in traces if tr.fallback_used)})
    tables = write_characterization_tables(graphs, traces, out / "simulation", model_cfg)
    shared_rows = []
    cohort_rows = []
    safety_rows = []
    graph_rows = []
    trace_by_job = {}
    for tr in traces:
        trace_by_job.setdefault(tr.job_id, []).append(tr)
    for graph in graphs:
        objects, stats = extract_shared_kv_objects(graph, model_cfg)
        stages = build_stages(graph, trace_by_job[graph.graph_id])
        cohorts, cstats = form_decode_cohorts(
            stages,
            objects,
            cfg=CohortConfig(min_expected_saved_kv_bytes=1),
        )
        graph_rows.append(
            {
                "workload": graph.workload,
                "job_id": graph.graph_id,
                "num_nodes": len(graph.nodes),
                "num_edges": len(graph.edges),
                "fan_in_max": max((n.fan_in for n in graph.nodes.values()), default=0),
                "fan_out_max": max((n.fan_out for n in graph.nodes.values()), default=0),
                "critical_path_fraction": max((n.criticality for n in graph.nodes.values()), default=0.0)
                / max(1e-9, sum(n.criticality for n in graph.nodes.values())),
            }
        )
        shared_rows.append({"workload": graph.workload, "job_id": graph.graph_id, **stats.to_dict()})
        cohort_rows.append({"workload": graph.workload, "job_id": graph.graph_id, **cstats})
        safety_rows.append(
            {
                "workload": graph.workload,
                "job_id": graph.graph_id,
                "safe_shared_prefix_tokens": stats.safe_shared_prefix_tokens,
                "unsafe_shared_text_tokens": stats.unsafe_shared_text_tokens,
                "shared_text_not_prefix_tokens": stats.shared_text_not_prefix_tokens,
                "unsafe_reuse_skipped_tokens": stats.unsafe_reuse_skipped_tokens,
            }
        )
    pd.DataFrame(graph_rows).to_csv(out / "simulation" / "workload_graph_stats.csv", index=False)
    pd.DataFrame(shared_rows).to_csv(out / "simulation" / "shared_kv_opportunity.csv", index=False)
    pd.DataFrame(cohort_rows).to_csv(out / "simulation" / "decode_cohort_opportunity.csv", index=False)
    pd.DataFrame(safety_rows).to_csv(out / "simulation" / "prefix_safety_stats.csv", index=False)
    pd.DataFrame(graph_rows).to_csv(out / "simulation" / "fan_in_out_stats.csv", index=False)
    fig = out / "figures"
    plot_dag_examples(graphs, fig / "fig_workload_dag_examples")
    plot_shared_prefix_ratio(tables["characterization_token_stats.csv"], fig / "fig_shared_prefix_ratio")
    plot_kv_duplication(tables["characterization_kv_stats.csv"], fig / "fig_kv_duplication_ratio")
    plot_latency_breakdown(tables["characterization_latency_breakdown.csv"], fig / "fig_latency_breakdown")
    plot_critical_path(tables["characterization_critical_path.csv"], fig / "fig_critical_path_vs_total_work")
    finalize_run_dir(out)
    print(f"Trace collection complete: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
