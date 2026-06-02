import argparse
import csv
import json
import sys
from pathlib import Path


sys.path.append(r"C:\Users\123\.codex\skills\isca-figures\scripts")
import isca_style as isca  # noqa: E402


RETENTION_POLICIES = [
    ("nocache", "No cache", "light_gray"),
    ("lru-system", "LRU", "gray"),
    ("kvflow-like", "KVFlow", "orange"),
    ("asg-retention-v2-graph-only", "ASG retention", "blue"),
    ("asg-retention-v2-oracle", "Oracle", "green"),
]

TRAFFIC_POLICIES = [
    ("asg-retention-v2-graph-only", "ASG retention", "blue"),
    ("asg-placement-v2-online", "ASG place", "purple"),
    ("asg-prefetch-v2-online", "ASG prefetch", "red"),
]


def plot_algorithm_comparison(input_csv, output_dir, figure_name="fig_algorithm_bar_synthetic"):
    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    rows = isca.read_csv(input_csv)
    by_policy = {row["policy"]: row for row in rows}
    if "lru-system" not in by_policy:
        raise ValueError("Input CSV must contain lru-system baseline.")
    baseline = isca.num(by_policy["lru-system"], "effective_prefill_tokens")
    if baseline <= 0:
        raise ValueError("lru-system effective_prefill_tokens must be positive.")

    retention_rows = []
    for policy, label, color_key in RETENTION_POLICIES:
        if policy not in by_policy:
            continue
        row = by_policy[policy]
        effective = isca.num(row, "effective_prefill_tokens")
        preserved = isca.num(row, "LRU_regret_tokens_preserved")
        normalized = effective / baseline
        reduction = (baseline - effective) / baseline * 100.0
        retention_rows.append(
            {
                "policy": policy,
                "label": label,
                "color_key": color_key,
                "effective_prefill_tokens": int(effective),
                "normalized_effective_prefill_vs_lru": normalized,
                "reduction_vs_lru_percent": reduction,
                "LRU_regret_tokens_preserved": int(preserved),
                "evidence_class": row.get("evidence_class", "unknown"),
                "can_support_real_paper_claims": row.get("can_support_real_paper_claims", "False"),
            }
        )
    traffic_rows = []
    for policy, label, color_key in TRAFFIC_POLICIES:
        if policy not in by_policy:
            continue
        row = by_policy[policy]
        remote_gb = isca.num(row, "remote_read_bytes") / 1e9
        prefetch_gb = isca.num(row, "prefetch_migration_bytes") / 1e9
        traffic_rows.append(
            {
                "policy": policy,
                "label": label,
                "color_key": color_key,
                "remote_read_gb": remote_gb,
                "prefetch_migration_gb": prefetch_gb,
                "model_comm_cycles": int(isca.num(row, "model_comm_cycles")),
                "prefetch_hidden_cycles": int(isca.num(row, "prefetch_hidden_cycles")),
                "prefetch_exposed_cycles": int(isca.num(row, "prefetch_exposed_cycles")),
            }
        )

    import matplotlib.pyplot as plt

    isca.apply_style(plt)
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35))
    labels = [row["label"] for row in retention_rows]
    xs = list(range(len(retention_rows)))
    colors = [isca.PALETTE[row["color_key"]] for row in retention_rows]

    ax = axes[0]
    vals = [row["normalized_effective_prefill_vs_lru"] for row in retention_rows]
    ax.bar(xs, vals, color=colors, edgecolor="black", linewidth=0.55)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Norm. effective prefill\nvs. LRU")
    ax.set_title("(a) ASG reduces paid prefill")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylim(0, max(vals) * 1.18)
    for x, value, row in zip(xs, vals, retention_rows):
        label = f"{row['reduction_vs_lru_percent']:.0f}%"
        if row["policy"] == "lru-system":
            label = "base"
        elif row["reduction_vs_lru_percent"] < 0:
            label = f"+{-row['reduction_vs_lru_percent']:.0f}%"
        ax.text(x, value + max(vals) * 0.035, label, ha="center", va="bottom", fontsize=6.5)

    ax = axes[1]
    preserved_k = [row["LRU_regret_tokens_preserved"] / 1000.0 for row in retention_rows]
    ax.bar(xs, preserved_k, color=colors, edgecolor="black", linewidth=0.55)
    ax.set_ylabel("Delayed-reuse state kept\n(K tokens)")
    ax.set_title("(b) ASG preserves valuable state")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylim(0, max(preserved_k) * 1.18 if preserved_k else 1.0)
    for x, value in zip(xs, preserved_k):
        if value > 0:
            ax.text(x, value + max(preserved_k) * 0.035, f"{value:.1f}", ha="center", va="bottom", fontsize=6.5)

    ax = axes[2]
    traffic_labels = [row["label"] for row in traffic_rows]
    tx = list(range(len(traffic_rows)))
    remote_vals = [row["remote_read_gb"] for row in traffic_rows]
    prefetch_vals = [row["prefetch_migration_gb"] for row in traffic_rows]
    ax.bar(tx, remote_vals, color=isca.PALETTE["purple"], edgecolor="black", linewidth=0.55, label="Remote read")
    ax.bar(tx, prefetch_vals, bottom=remote_vals, color=isca.PALETTE["red"], edgecolor="black", linewidth=0.55, label="Prefetch move")
    ax.set_ylabel("KV movement (GB)")
    ax.set_title("(c) Placement/prefetch affect traffic")
    ax.set_xticks(tx)
    ax.set_xticklabels(traffic_labels, rotation=28, ha="right")
    ymax = max([a + b for a, b in zip(remote_vals, prefetch_vals)], default=1.0)
    ax.set_ylim(0, ymax * 1.18 if ymax > 0 else 1.0)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")

    for ax in axes:
        ax.grid(axis="x", visible=False)
    isca.finish_axes(axes)
    fig.tight_layout(pad=0.7)
    figures = isca.save_figure(fig, output_dir, figure_name)
    plt.close(fig)

    data_csv = isca.write_csv(output_dir / f"{figure_name}.csv", retention_rows + traffic_rows)
    evidence = {
        "claim": "On the synthetic delayed-reuse suite, ASG retention reduces effective prefill tokens relative to lru-system.",
        "input_csv": str(input_csv),
        "baseline_policy": "lru-system",
        "baseline_effective_prefill_tokens": int(baseline),
        "best_reduction_vs_lru_percent": max(row["reduction_vs_lru_percent"] for row in retention_rows),
        "best_policy": max(retention_rows, key=lambda row: row["reduction_vs_lru_percent"])["policy"],
        "max_remote_read_gb": max((row["remote_read_gb"] for row in traffic_rows), default=0.0),
        "max_prefetch_migration_gb": max((row["prefetch_migration_gb"] for row in traffic_rows), default=0.0),
        "evidence_class": retention_rows[0].get("evidence_class", "unknown") if retention_rows else "unknown",
        "why_previous_bars_matched": "Placement and prefetch policies use the same ASG v2 retention objective for effective_prefill_tokens; their differences appear in traffic and communication metrics.",
    }
    limitations = [
        "Synthetic traces validate controlled algorithm behavior only; they cannot support real-trace paper claims.",
        "Bars use full synthetic suite summary at the default paper-eval setting, not a hardware-measured deployment trace.",
    ]
    report = isca.write_report(output_dir / f"{figure_name}_report.json", figures, [data_csv], evidence, limitations)
    return {"figures": figures, "data_csv": data_csv, "report": report}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Plot ISCA-style algorithm-comparison bars from Agent-on-Wafer summary CSV.")
    parser.add_argument("--input-csv", default="agent_results/paper_eval_synthetic/full_set_summary.csv")
    parser.add_argument("--output-dir", default="agent_results/algorithm_comparison_synthetic")
    parser.add_argument("--figure-name", default="fig_algorithm_bar_synthetic")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = plot_algorithm_comparison(args.input_csv, args.output_dir, args.figure_name)
    print(f"algorithm_comparison figures={len(result['figures'])} report={result['report']}")


if __name__ == "__main__":
    main()
