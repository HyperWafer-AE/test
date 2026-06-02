import argparse
import csv
import json
import subprocess
import time
from pathlib import Path
from urllib import request as urlrequest


def collect_metrics(
    output: str = "agent_results/local_h100/server_metrics_timeseries.csv",
    server_url: str = "http://127.0.0.1:30000",
    pid: int = 0,
    interval: float = 2.0,
    duration: float = 30.0,
):
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "timestamp",
        "gpu_index",
        "memory_used_mib",
        "memory_total_mib",
        "gpu_utilization_pct",
        "power_draw_w",
        "server_pid",
        "server_rss_kib",
        "server_cpu_pct",
        "metrics_endpoint_available",
        "active_requests",
        "request_throughput",
    ]
    end_time = time.time() + duration
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        while time.time() < end_time:
            rows = gpu_rows()
            proc = process_row(pid)
            endpoint = probe_metrics(server_url)
            for row in rows:
                writer.writerow({**row, **proc, **endpoint, "timestamp": time.time()})
            handle.flush()
            time.sleep(interval)
    return output_path


def gpu_rows():
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True)
    except Exception:
        return [{"gpu_index": "", "memory_used_mib": "", "memory_total_mib": "", "gpu_utilization_pct": "", "power_draw_w": ""}]
    rows = []
    for line in out.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) >= 5:
            rows.append(
                {
                    "gpu_index": parts[0],
                    "memory_used_mib": parts[1],
                    "memory_total_mib": parts[2],
                    "gpu_utilization_pct": parts[3],
                    "power_draw_w": parts[4],
                }
            )
    return rows


def process_row(pid: int):
    if not pid:
        return {"server_pid": "", "server_rss_kib": "", "server_cpu_pct": ""}
    try:
        out = subprocess.check_output(["ps", "-p", str(pid), "-o", "pid=,rss=,pcpu="], text=True).strip()
        parts = out.split()
        return {"server_pid": parts[0], "server_rss_kib": parts[1], "server_cpu_pct": parts[2]}
    except Exception:
        return {"server_pid": pid, "server_rss_kib": "", "server_cpu_pct": ""}


def probe_metrics(server_url: str):
    for suffix in ("/metrics", "/health"):
        try:
            with urlrequest.urlopen(server_url.rstrip("/") + suffix, timeout=1.0) as resp:
                body = resp.read(4096).decode("utf-8", errors="replace")
            if suffix == "/metrics":
                return {
                    "metrics_endpoint_available": True,
                    "active_requests": _extract_metric(body, "sglang:num_running_reqs"),
                    "request_throughput": _extract_metric(body, "sglang:prompt_tokens_total"),
                }
            return {"metrics_endpoint_available": False, "active_requests": "", "request_throughput": ""}
        except Exception:
            continue
    return {"metrics_endpoint_available": False, "active_requests": "", "request_throughput": ""}


def _extract_metric(body: str, name: str):
    for line in body.splitlines():
        if line.startswith(name):
            return line.rsplit(" ", 1)[-1]
    return ""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Collect local SGLang/vLLM server and GPU metrics.")
    parser.add_argument("--output", default="agent_results/local_h100/server_metrics_timeseries.csv")
    parser.add_argument("--server-url", default="http://127.0.0.1:30000")
    parser.add_argument("--pid", type=int, default=0)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=30.0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    path = collect_metrics(args.output, args.server_url, args.pid, args.interval, args.duration)
    report = {
        "output": str(path),
        "internal_kv_metrics": "unavailable unless exported by the serving backend; no KV metrics were fabricated",
    }
    report_path = Path(path).with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

