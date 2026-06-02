import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Plot local H100 trace collection and wafer replay results.")
    parser.add_argument("--trace-profile", default="agent_results/local_h100/trace_profile.json")
    parser.add_argument("--gpu-summary", default="agent_results/local_h100/concurrency_trace_summary.csv")
    parser.add_argument("--wafer-results", default="agent_results/local_h100/wafer_replay")
    parser.add_argument("--output-dir", default="agent_results/local_h100/figures")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.output_dir)
    data_dir = out / "figure_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    profile = json.loads(Path(args.trace_profile).read_text(encoding="utf-8"))
    gpu_rows = list(csv.DictReader(open(args.gpu_summary, encoding="utf-8"))) if Path(args.gpu_summary).exists() else []
    wafer_rows = list(csv.DictReader(open(Path(args.wafer_results) / "main_summary.csv", encoding="utf-8"))) if (Path(args.wafer_results) / "main_summary.csv").exists() else []

    plot_state_characterization(plt, out, data_dir, profile)
    plot_gpu_serving(plt, out, data_dir, gpu_rows)
    plot_wafer_policy(plt, out, data_dir, wafer_rows)
    plot_memory_concurrency(plt, out, data_dir, wafer_rows)
    plot_mapping_prefetch(plt, out, data_dir, wafer_rows)
    captions = [
        "fig_local_trace_workload_characterization: state-token, reuse-distance, and delayed-reuse/LRU-regret workload structure.",
        "fig_local_concurrency_gpu_serving: local GPU serving latency and workflow wall-clock behavior as concurrency changes.",
        "fig_local_wafer_replay_policy: Agent-on-Wafer replay comparison across LRU, KVFlow-like, ASG online, and ASG oracle policies.",
        "fig_local_memory_concurrency_sweep: replay sensitivity to memory budget and workflow concurrency.",
        "fig_local_mapping_prefetch: remote read, demand migration, and prefetch movement diagnostics.",
    ]
    (out / "figure_captions.md").write_text("\n".join(f"- {line}" for line in captions) + "\n", encoding="utf-8")
    print(f"wrote figures to {out}")


def save_all(plt, out: Path, name: str):
    for ext in ("pdf", "svg", "png"):
        plt.savefig(out / f"{name}.{ext}", bbox_inches="tight")
    plt.close()


def plot_state_characterization(plt, out, data_dir, profile):
    token_dist = profile.get("token_distribution_per_state_type", {})
    names = list(token_dist.keys())
    means = [token_dist[n].get("mean", 0) for n in names]
    rows = [{"state_type": n, "mean_tokens": m} for n, m in zip(names, means)]
    write_rows(data_dir / "fig_local_trace_workload_characterization.csv", rows)
    plt.figure(figsize=(10, 4))
    plt.bar(names, means, color="#4C78A8")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Mean tokens")
    plt.title("State Token Distribution")
    save_all(plt, out, "fig_local_trace_workload_characterization")


def plot_gpu_serving(plt, out, data_dir, rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row.get("concurrency_level", 0) or 0)].append(row)
    data = []
    for level, items in sorted(grouped.items()):
        p95 = mean(float(item.get("latency_p95_ms") or 0) for item in items)
        ttft = mean(float(item.get("ttft_p95_ms") or 0) for item in items)
        wall = mean(float(item.get("workflow_wall_clock_s") or 0) for item in items)
        memory = mean(float(item.get("gpu_memory_used_mib_mean") or 0) for item in items)
        data.append(
            {
                "concurrency": level,
                "latency_p95_ms": p95,
                "ttft_p95_ms": ttft,
                "workflow_wall_clock_s": wall,
                "gpu_memory_used_mib_mean": memory,
            }
        )
    write_rows(data_dir / "fig_local_concurrency_gpu_serving.csv", data)
    fig, ax1 = plt.subplots(figsize=(8, 4))
    xs = [d["concurrency"] for d in data]
    ax1.plot(xs, [d["latency_p95_ms"] for d in data], marker="o", label="P95 latency ms")
    ax1.plot(xs, [d["ttft_p95_ms"] for d in data], marker="s", label="P95 TTFT ms")
    ax1.set_xlabel("Concurrency")
    ax1.set_ylabel("Latency (ms)")
    ax2 = ax1.twinx()
    ax2.plot(xs, [d["gpu_memory_used_mib_mean"] for d in data], marker="^", color="#E15759", label="GPU memory MiB")
    ax2.set_ylabel("GPU memory used (MiB)")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    plt.title("Local GPU Serving")
    save_all(plt, out, "fig_local_concurrency_gpu_serving")


def plot_wafer_policy(plt, out, data_dir, rows):
    chosen = [r for r in rows if str(r.get("memory_budget")) in {"0.25", "0.5"} and str(r.get("concurrency")) in {"4", "8"}]
    if not chosen:
        chosen = rows
    by_policy = defaultdict(list)
    for row in chosen:
        by_policy[row.get("policy")].append(float(row.get("effective_prefill_tokens") or 0))
    data = [{"policy": k, "effective_prefill_tokens": mean(v)} for k, v in sorted(by_policy.items())]
    write_rows(data_dir / "fig_local_wafer_replay_policy.csv", data)
    plt.figure(figsize=(9, 4))
    plt.bar([d["policy"] for d in data], [d["effective_prefill_tokens"] for d in data], color="#59A14F")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Effective prefill tokens")
    plt.title("Wafer Replay Policy Comparison")
    save_all(plt, out, "fig_local_wafer_replay_policy")


def plot_memory_concurrency(plt, out, data_dir, rows):
    data = [
        {
            "memory_budget": row.get("memory_budget"),
            "concurrency": row.get("concurrency"),
            "policy": row.get("policy"),
            "effective_prefill_tokens": row.get("effective_prefill_tokens"),
        }
        for row in rows
    ]
    write_rows(data_dir / "fig_local_memory_concurrency_sweep.csv", data)
    plt.figure(figsize=(9, 4))
    for policy in sorted({row.get("policy") for row in rows}):
        subset = [row for row in rows if row.get("policy") == policy and row.get("memory_budget") == "0.25"]
        if not subset:
            continue
        xs = [int(float(row.get("concurrency") or 0)) for row in subset]
        ys = [float(row.get("effective_prefill_tokens") or 0) for row in subset]
        plt.plot(xs, ys, marker="o", label=policy)
    plt.xlabel("Concurrency")
    plt.ylabel("Effective prefill tokens")
    plt.legend(fontsize=7)
    plt.title("Memory/Concurrency Sweep")
    save_all(plt, out, "fig_local_memory_concurrency_sweep")


def plot_mapping_prefetch(plt, out, data_dir, rows):
    metrics = ["remote_read_bytes", "demand_migration_bytes", "prefetch_migration_bytes"]
    by_policy = defaultdict(lambda: Counter())
    for row in rows:
        for metric in metrics:
            by_policy[row.get("policy")][metric] += float(row.get(metric) or 0)
    data = [{"policy": policy, **dict(values)} for policy, values in sorted(by_policy.items())]
    write_rows(data_dir / "fig_local_mapping_prefetch.csv", data)
    plt.figure(figsize=(9, 4))
    xs = range(len(data))
    bottom = [0.0] * len(data)
    for metric in metrics:
        vals = [d.get(metric, 0.0) for d in data]
        plt.bar(xs, vals, bottom=bottom, label=metric)
        bottom = [a + b for a, b in zip(bottom, vals)]
    plt.xticks(list(xs), [d["policy"] for d in data], rotation=35, ha="right")
    plt.ylabel("Bytes")
    plt.legend(fontsize=7)
    plt.title("Mapping and Prefetch Movement")
    save_all(plt, out, "fig_local_mapping_prefetch")


def write_rows(path: Path, rows: list):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
