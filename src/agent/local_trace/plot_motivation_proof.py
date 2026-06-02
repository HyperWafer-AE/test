import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Plot motivation-proof local H100 trace and wafer replay results.")
    parser.add_argument("--trace-profile", default="agent_results/local_h100_long/trace_profile.json")
    parser.add_argument("--selection-report", default="agent_results/local_h100_long/trace_selection/selection_report.json")
    parser.add_argument("--gpu-summary", default="agent_results/local_h100_long/concurrency_trace_summary.csv")
    parser.add_argument("--kv-pressure", default="agent_results/local_h100_long/estimated_kv_pressure.csv")
    parser.add_argument("--wafer-results", default="agent_results/local_h100_long/wafer_replay")
    parser.add_argument("--output-dir", default="agent_results/local_h100_long/motivation_figures")
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
    selection = json.loads(Path(args.selection_report).read_text(encoding="utf-8"))
    gpu_rows = read_csv(args.gpu_summary)
    kv_rows = read_csv(args.kv_pressure)
    wafer_dir = Path(args.wafer_results)
    full = read_csv(wafer_dir / "full_summary.csv")
    opportunity = read_csv(wafer_dir / "opportunity_summary.csv")
    control = read_csv(wafer_dir / "control_summary.csv")

    fig_state_anatomy(plt, out, data_dir, profile)
    fig_delayed_reuse(plt, out, data_dir, profile)
    fig_concurrency(plt, out, data_dir, gpu_rows, kv_rows)
    fig_memory_breaks(plt, out, data_dir, full)
    fig_asg_fix(plt, out, data_dir, opportunity or full)
    fig_opportunity_control(plt, out, data_dir, opportunity, control)
    fig_mapping_prefetch(plt, out, data_dir, full)
    write_captions(out / "figure_captions.md")
    print(f"wrote motivation figures to {out}")


def fig_state_anatomy(plt, out, data_dir, profile):
    state_counts = profile.get("state_type_distribution", {})
    rows = [{"state_type": k, "count": v} for k, v in state_counts.items()]
    write_rows(data_dir / "fig1_state_anatomy.csv", rows)
    plt.figure(figsize=(9, 4))
    plt.bar([r["state_type"] for r in rows], [r["count"] for r in rows], color="#4C78A8")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("State/event count")
    plt.title("Agent Workflow State Anatomy")
    save_all(plt, out, "fig1_agent_workflow_state_anatomy")


def fig_delayed_reuse(plt, out, data_dir, profile):
    delayed = profile.get("high_value_delayed_reuse_by_state_type", {})
    regret = profile.get("LRU_regret_by_state_type", {})
    keys = sorted(set(delayed) | set(regret))
    rows = [{"state_type": k, "high_value_delayed_tokens": delayed.get(k, 0), "lru_regret_tokens": regret.get(k, 0)} for k in keys]
    write_rows(data_dir / "fig2_delayed_reuse.csv", rows)
    plt.figure(figsize=(9, 4))
    xs = range(len(rows))
    plt.bar([x - 0.2 for x in xs], [r["high_value_delayed_tokens"] for r in rows], width=0.4, label="Delayed high-value tokens")
    plt.bar([x + 0.2 for x in xs], [r["lru_regret_tokens"] for r in rows], width=0.4, label="LRU regret tokens")
    plt.xticks(list(xs), [r["state_type"] for r in rows], rotation=30, ha="right")
    plt.ylabel("Tokens")
    plt.legend()
    plt.title("Long-Lived State and Delayed Reuse")
    save_all(plt, out, "fig2_long_lived_state_delayed_reuse")


def fig_concurrency(plt, out, data_dir, gpu_rows, kv_rows):
    grouped = defaultdict(list)
    for row in gpu_rows:
        grouped[int(float(row.get("concurrency_level") or 0))].append(row)
    rows = []
    for level, items in sorted(grouped.items()):
        rows.append(
            {
                "concurrency": level,
                "latency_p95_ms": mean(float(x.get("latency_p95_ms") or 0) for x in items),
                "ttft_p95_ms": mean(float(x.get("ttft_p95_ms") or 0) for x in items),
                "workflow_wall_clock_s": mean(float(x.get("workflow_wall_clock_s") or 0) for x in items),
                "gpu_memory_used_mib_mean": mean(float(x.get("gpu_memory_used_mib_mean") or 0) for x in items),
            }
        )
    if kv_rows:
        max_pressure = max(float(row.get("estimated_memory_pressure_ratio") or 0) for row in kv_rows)
        for row in rows:
            row["estimated_pressure_ratio_max"] = max_pressure
    write_rows(data_dir / "fig3_concurrency_pressure.csv", rows)
    fig, ax1 = plt.subplots(figsize=(8, 4))
    xs = [r["concurrency"] for r in rows]
    ax1.plot(xs, [r["latency_p95_ms"] for r in rows], marker="o", label="P95 latency ms")
    ax1.plot(xs, [r["workflow_wall_clock_s"] for r in rows], marker="s", label="Workflow seconds")
    ax1.set_xlabel("Concurrency")
    ax1.set_ylabel("Latency / wall-clock")
    ax2 = ax1.twinx()
    ax2.plot(xs, [r["gpu_memory_used_mib_mean"] for r in rows], marker="^", color="#E15759", label="GPU memory MiB")
    ax2.set_ylabel("GPU memory MiB")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    plt.title("Concurrency Creates Local Serving Pressure")
    save_all(plt, out, "fig3_concurrency_serving_pressure")


def fig_memory_breaks(plt, out, data_dir, rows):
    lru = [r for r in rows if r.get("policy") == "lru-system" and str(r.get("concurrency")) == "8"]
    lru = sorted(lru, key=lambda r: float(r.get("memory_budget") or 0))
    data = [{"memory_budget": float(r["memory_budget"]), "effective_prefill_tokens": float(r["effective_prefill_tokens"]), "cache_byte_miss": float(r["cache_byte_miss"])} for r in lru]
    write_rows(data_dir / "fig4_memory_breaks_residency.csv", data)
    fig, ax1 = plt.subplots(figsize=(8, 4))
    xs = [r["memory_budget"] for r in data]
    ax1.plot(xs, [r["effective_prefill_tokens"] for r in data], marker="o", label="Effective prefill")
    ax1.set_xlabel("Memory budget GB")
    ax1.set_ylabel("Effective prefill tokens")
    ax2 = ax1.twinx()
    ax2.plot(xs, [r["cache_byte_miss"] for r in data], marker="s", color="#F28E2B", label="Cache byte miss")
    ax2.set_ylabel("Cache byte miss")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    plt.title("Finite Memory Breaks State Residency")
    save_all(plt, out, "fig4_finite_memory_breaks_residency")


def fig_asg_fix(plt, out, data_dir, rows):
    chosen = [r for r in rows if str(r.get("memory_budget")) in {"0.02", "0.05", "0.1"} and str(r.get("concurrency")) in {"8", "16"}]
    if not chosen:
        chosen = rows
    by_policy = defaultdict(list)
    for row in chosen:
        by_policy[row.get("policy")].append(row)
    data = []
    for policy, items in sorted(by_policy.items()):
        data.append(
            {
                "policy": policy,
                "effective_prefill_tokens": mean(float(r.get("effective_prefill_tokens") or 0) for r in items),
                "LRU_regret_tokens_preserved": mean(float(r.get("LRU_regret_tokens_preserved") or 0) for r in items),
                "model_compute_cycles": mean(float(r.get("model_compute_cycles") or 0) for r in items),
            }
        )
    write_rows(data_dir / "fig5_asg_fixes_problem.csv", data)
    plt.figure(figsize=(10, 4))
    plt.bar([r["policy"] for r in data], [r["effective_prefill_tokens"] for r in data], color="#59A14F")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Effective prefill tokens")
    plt.title("ASG Preserves Delayed High-Value States")
    save_all(plt, out, "fig5_asg_fixes_right_problem")


def fig_opportunity_control(plt, out, data_dir, opportunity, control):
    data = []
    for name, rows in (("opportunity", opportunity), ("control", control)):
        if not rows:
            continue
        lru = mean(float(r["effective_prefill_tokens"]) for r in rows if r.get("policy") == "lru-system")
        asg = mean(float(r["effective_prefill_tokens"]) for r in rows if r.get("policy") == "asg-retention-v2-online")
        gain = (lru - asg) / lru if lru else 0.0
        data.append({"subset": name, "lru_effective_prefill": lru, "asg_effective_prefill": asg, "asg_gain": gain})
    write_rows(data_dir / "fig6_opportunity_vs_control.csv", data)
    plt.figure(figsize=(6, 4))
    plt.bar([r["subset"] for r in data], [r["asg_gain"] for r in data], color=["#4C78A8", "#BAB0AC"][: len(data)])
    plt.ylabel("ASG gain vs LRU")
    plt.title("Opportunity vs Recency-Control")
    save_all(plt, out, "fig6_opportunity_vs_control")


def fig_mapping_prefetch(plt, out, data_dir, rows):
    chosen = [r for r in rows if r.get("policy") in {"asg-retention-v2-online", "asg-placement-v2-online", "asg-prefetch-v2-online"}]
    by_policy = defaultdict(list)
    for row in chosen:
        by_policy[row.get("policy")].append(row)
    data = []
    for policy, items in sorted(by_policy.items()):
        data.append(
            {
                "policy": policy,
                "remote_read_bytes": mean(float(r.get("remote_read_bytes") or 0) for r in items),
                "demand_migration_bytes": mean(float(r.get("demand_migration_bytes") or 0) for r in items),
                "prefetch_migration_bytes": mean(float(r.get("prefetch_migration_bytes") or 0) for r in items),
                "prefetch_exposed_cycles": mean(float(r.get("prefetch_exposed_cycles") or 0) for r in items),
            }
        )
    write_rows(data_dir / "fig7_mapping_prefetch.csv", data)
    plt.figure(figsize=(8, 4))
    plt.bar([r["policy"] for r in data], [r["prefetch_migration_bytes"] + r["demand_migration_bytes"] + r["remote_read_bytes"] for r in data], color="#B07AA1")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Bytes")
    plt.title("Wafer Mapping and Prefetch Metrics")
    save_all(plt, out, "fig7_mapping_prefetch")


def read_csv(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    return list(csv.DictReader(path.open(encoding="utf-8")))


def write_rows(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
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


def save_all(plt, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    for suffix in ("pdf", "svg", "png"):
        plt.savefig(out / f"{name}.{suffix}", dpi=200)
    plt.close()


def write_captions(path: Path):
    path.write_text(
        "\n".join(
            [
                "Figure 1: Local H100 traces show agent workflows as explicit state machines with typed prompt segments, not full-history transcripts.",
                "Figure 2: Local H100 traces quantify delayed high-value state reuse and LRU-regret tokens by state type.",
                "Figure 3: Local H100 serving evidence shows concurrency effects on request latency, workflow wall-clock time, and measured GPU memory; no provider-internal KV eviction is claimed.",
                "Figure 4: BusyBarn-backed wafer replay shows finite-memory residency pressure through effective prefill and cache-byte misses under LRU.",
                "Figure 5: BusyBarn-backed wafer replay compares LRU-system, KVFlow-like, ASG-online, and ASG-oracle policies on the opportunity workload.",
                "Figure 6: Opportunity/control replay separates delayed-reuse-heavy traces from recency-control traces.",
                "Figure 7: BusyBarn-backed wafer replay reports mapper/prefetch communication metrics when available.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
