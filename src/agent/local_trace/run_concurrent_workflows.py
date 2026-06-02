import argparse
import asyncio
import csv
import json
import statistics
import subprocess
import time
from pathlib import Path

from src.agent.local_serving.server_metrics_collector import collect_metrics
from src.agent.local_trace.benchmark_tasks import ensure_local_debug_tasks
from src.agent.local_trace.run_coding_agent_traces import load_tasks, run_workflow


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run concurrent local coding-agent workflows against one model server.")
    parser.add_argument("--server-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", default="local-qwen-32b")
    parser.add_argument("--benchmark", default="local_debug_tasks")
    parser.add_argument("--tasks-dir", default="benchmarks/local_debug_tasks")
    parser.add_argument("--output-dir", default="traces/local_h100/concurrent")
    parser.add_argument("--concurrency-levels", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--tasks-per-level", type=int, default=8)
    parser.add_argument("--framework", default="local_react_coding_agent")
    parser.add_argument("--history-mode", choices=("selective_state", "full_history"), default="selective_state")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--model-path", default="/data1/dg123_data/Qwen-32B")
    parser.add_argument("--raw-request-dir", default="traces/local_h100/raw_requests")
    parser.add_argument("--raw-tool-dir", default="traces/local_h100/raw_tools")
    parser.add_argument("--workspace-root", default="traces/local_h100/workspaces_concurrent")
    parser.add_argument("--stream", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ensure_local_debug_tasks(args.tasks_dir)
    tasks = load_tasks(args.tasks_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    all_summary = []
    timeline = []
    for level in args.concurrency_levels:
        rows, events = asyncio.run(run_level(args, level, tasks))
        all_summary.extend(rows)
        timeline.extend(events)
    write_summary(all_summary, "agent_results/local_h100/concurrency_trace_summary.csv")
    timeline_path = Path("agent_results/local_h100/concurrency_timeline.json")
    timeline_path.parent.mkdir(parents=True, exist_ok=True)
    timeline_path.write_text(json.dumps({"events": timeline}, indent=2), encoding="utf-8")
    pid = _server_pid()
    collect_metrics(duration=10, interval=2, pid=pid)
    print(f"wrote concurrency outputs under {output_root}")


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
            workflow_id = f"c{level}_wf_{idx:03d}_{task['task_id']}"
            start = time.time()
            timeline.append({"timestamp": start, "event": "workflow_start", "workflow_id": workflow_id, "concurrency": level})
            local_args = argparse.Namespace(**vars(args))
            local_args.output_dir = str(out_dir)
            trace, result = await asyncio.to_thread(run_workflow, task, workflow_id, local_args)
            end = time.time()
            out = out_dir / f"{workflow_id}.json"
            out.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
            timeline.append({"timestamp": end, "event": "workflow_end", "workflow_id": workflow_id, "concurrency": level})
            req_stats = request_stats(Path(args.raw_request_dir) / f"{workflow_id}.jsonl")
            rows.append(
                {
                    "concurrency_level": level,
                    "workflow_id": workflow_id,
                    "success": result["success"],
                    "workflow_wall_clock_s": end - start,
                    **req_stats,
                }
            )

    await asyncio.gather(*(one(idx, task) for idx, task in enumerate(selected)))
    stop_sampling.set()
    await sampler
    gpu_summary = summarize_gpu_samples(gpu_samples)
    timeline.extend(inflight_timeline(Path(args.raw_request_dir), prefix=f"c{level}_"))
    timeline.extend(gpu_samples)
    for row in rows:
        row.update(gpu_summary)
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
        if len(parts) < 5:
            continue
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
    if not samples:
        return {
            "gpu_memory_used_mib_mean": 0.0,
            "gpu_memory_used_mib_max": 0.0,
            "gpu_utilization_pct_mean": 0.0,
            "gpu_power_draw_w_mean": 0.0,
        }
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
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def percentile(values, pct):
    if not values:
        return 0.0
    values = sorted(values)
    idx = round((pct / 100.0) * (len(values) - 1))
    return values[int(idx)]


def write_summary(rows: list, path: str):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "concurrency_level",
        "workflow_id",
        "success",
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


def _server_pid() -> int:
    try:
        out = __import__("subprocess").check_output(["pgrep", "-f", "sglang.launch_server"], text=True).splitlines()
        return int(out[-1]) if out else 0
    except Exception:
        return 0


if __name__ == "__main__":
    main()
