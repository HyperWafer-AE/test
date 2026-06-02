import argparse
import asyncio
import csv
import json
import random
import statistics
import subprocess
import time
from pathlib import Path

from src.agent.local_serving.server_metrics_collector import collect_metrics
from src.agent.local_trace.long_benchmark_tasks import ensure_local_long_debug_tasks
from src.agent.local_trace.long_horizon_coding_agent import add_common_args, load_long_tasks, run_long_workflow


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run long-horizon concurrent workflows against one local model server.")
    add_common_args(parser)
    parser.add_argument("--output-dir", default="traces/local_h100_long/concurrent")
    parser.add_argument("--concurrency-levels", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--tasks-per-level", type=int, default=8)
    parser.set_defaults(workspace_root="traces/local_h100_long/workspaces_concurrent")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    ensure_local_long_debug_tasks(args.tasks_dir)
    tasks = load_long_tasks(args.tasks_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    all_summary = []
    timeline = []
    for level in args.concurrency_levels:
        rows, events = asyncio.run(run_level(args, level, tasks))
        all_summary.extend(rows)
        timeline.extend(events)
        print(f"completed concurrency {level} rows={len(rows)}", flush=True)
    write_summary(all_summary, "agent_results/local_h100_long/concurrency_trace_summary.csv")
    timeline_path = Path("agent_results/local_h100_long/concurrency_timeline.json")
    timeline_path.parent.mkdir(parents=True, exist_ok=True)
    timeline_path.write_text(json.dumps({"events": timeline}, indent=2), encoding="utf-8")
    collect_metrics(output="agent_results/local_h100_long/server_metrics_timeseries.csv", duration=12, interval=2, pid=server_pid())
    print(f"wrote long concurrency outputs under {output_root}")


async def run_level(args, level: int, tasks: list):
    selected = [tasks[(args.seed + i) % len(tasks)] for i in range(args.tasks_per_level)]
    out_dir = Path(args.output_dir) / f"c{level}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(level)
    timeline = []
    rows = []
    gpu_samples = []
    stop_sampling = asyncio.Event()
    sampler = asyncio.create_task(sample_gpu_level(level, gpu_samples, stop_sampling))

    async def one(idx, task):
        async with sem:
            workflow_id = f"long_c{level}_wf_{idx:03d}_{task['task_id']}"
            start = time.time()
            timeline.append({"timestamp": start, "event": "workflow_start", "workflow_id": workflow_id, "concurrency": level})
            local_args = argparse.Namespace(**vars(args))
            local_args.output_dir = str(out_dir)
            trace, result = await asyncio.to_thread(run_long_workflow, task, workflow_id, local_args)
            end = time.time()
            out = out_dir / f"{workflow_id}.json"
            out.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
            timeline.append({"timestamp": end, "event": "workflow_end", "workflow_id": workflow_id, "concurrency": level})
            rows.append(
                {
                    "concurrency_level": level,
                    "workflow_id": workflow_id,
                    "success": result["success"],
                    "task_family": result.get("task_family"),
                    "expected_control_or_opportunity": result.get("expected_control_or_opportunity"),
                    "workflow_wall_clock_s": end - start,
                    **request_stats(Path(args.raw_request_dir) / f"{workflow_id}.jsonl"),
                }
            )

    await asyncio.gather(*(one(idx, task) for idx, task in enumerate(selected)))
    stop_sampling.set()
    await sampler
    gpu_summary = summarize_gpu_samples(gpu_samples)
    for row in rows:
        row.update(gpu_summary)
    timeline.extend(inflight_timeline(Path(args.raw_request_dir), prefix=f"long_c{level}_"))
    timeline.extend(gpu_samples)
    return rows, timeline


def request_stats(path: Path) -> dict:
    records = read_jsonl(path)
    lat = [float(r.get("total_latency_ms") or 0.0) for r in records if not r.get("error")]
    ttft = [float(r.get("ttft_ms") or 0.0) for r in records if not r.get("error")]
    return {
        "num_llm_requests": len(records),
        "num_llm_errors": sum(1 for r in records if r.get("error")),
        "ttft_p50_ms": percentile(ttft, 50),
        "ttft_p95_ms": percentile(ttft, 95),
        "latency_p50_ms": percentile(lat, 50),
        "latency_p95_ms": percentile(lat, 95),
        "tokens_per_second_mean": statistics.mean([float(r.get("tokens_per_second") or 0.0) for r in records]) if records else 0.0,
    }


def inflight_timeline(raw_dir: Path, prefix: str):
    points = []
    for path in raw_dir.glob(f"{prefix}*.jsonl"):
        for rec in read_jsonl(path):
            points.append((float(rec.get("timestamp_start") or 0.0), 1, rec.get("workflow_id"), rec.get("request_id")))
            points.append((float(rec.get("timestamp_end") or 0.0), -1, rec.get("workflow_id"), rec.get("request_id")))
    events = []
    inflight = 0
    for ts, delta, workflow_id, request_id in sorted(points):
        inflight += delta
        events.append({"timestamp": ts, "event": "llm_inflight_update", "inflight": inflight, "workflow_id": workflow_id, "request_id": request_id})
    return events


async def sample_gpu_level(level: int, samples: list, stop_event: asyncio.Event):
    while not stop_event.is_set():
        timestamp = time.time()
        for sample in read_gpu_samples():
            samples.append({"timestamp": timestamp, "event": "gpu_sample", "concurrency": level, **sample})
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass


def read_gpu_samples() -> list:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 5:
            rows.append(
                {
                    "gpu_index": int(parts[0]),
                    "memory_used_mib": float(parts[1]),
                    "memory_total_mib": float(parts[2]),
                    "gpu_utilization_pct": float(parts[3]),
                    "power_draw_w": float(parts[4]),
                }
            )
    return rows


def summarize_gpu_samples(samples: list) -> dict:
    memory = [float(item["memory_used_mib"]) for item in samples if "memory_used_mib" in item]
    util = [float(item["gpu_utilization_pct"]) for item in samples if "gpu_utilization_pct" in item]
    power = [float(item["power_draw_w"]) for item in samples if "power_draw_w" in item]
    return {
        "gpu_memory_used_mib_mean": statistics.mean(memory) if memory else 0.0,
        "gpu_memory_used_mib_max": max(memory) if memory else 0.0,
        "gpu_utilization_pct_mean": statistics.mean(util) if util else 0.0,
        "gpu_power_draw_w_mean": statistics.mean(power) if power else 0.0,
    }


def read_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile(values, pct):
    if not values:
        return 0.0
    values = sorted(values)
    return values[round((pct / 100.0) * (len(values) - 1))]


def write_summary(rows: list, path: str):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "concurrency_level",
        "workflow_id",
        "success",
        "task_family",
        "expected_control_or_opportunity",
        "workflow_wall_clock_s",
        "num_llm_requests",
        "num_llm_errors",
        "ttft_p50_ms",
        "ttft_p95_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "tokens_per_second_mean",
        "gpu_memory_used_mib_mean",
        "gpu_memory_used_mib_max",
        "gpu_utilization_pct_mean",
        "gpu_power_draw_w_mean",
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def server_pid() -> int:
    try:
        out = subprocess.check_output(["pgrep", "-f", "sglang.launch_server"], text=True).splitlines()
        return int(out[-1]) if out else 0
    except Exception:
        return 0


if __name__ == "__main__":
    main()
