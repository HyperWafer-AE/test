from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from waferagent.graph_ir import AgentGraph


def _save(fig, out_base: str | Path) -> None:
    out = Path(out_base)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=180)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def plot_smoke_latency(metrics_csv: str | Path, out_base: str | Path) -> None:
    df = pd.read_csv(metrics_csv)
    summary = df.groupby("baseline")[["prefill_ms_total", "decode_ms_total", "communication_time_ms", "queue_wait_ms_total"]].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    summary.plot(kind="bar", stacked=True, ax=ax, color=["#4c78a8", "#f58518", "#54a24b", "#b279a2"])
    ax.set_ylabel("ms")
    ax.set_title("Smoke Latency Breakdown")
    ax.legend(loc="best", fontsize=8)
    _save(fig, out_base)


def plot_dag_examples(graphs: list[AgentGraph], out_base: str | Path) -> None:
    n = min(3, len(graphs))
    fig, axes = plt.subplots(1, max(1, n), figsize=(5 * max(1, n), 4))
    if n == 1:
        axes = [axes]
    for ax, graph in zip(axes, graphs[:n]):
        g = graph.to_networkx()
        layers: dict[int, list[str]] = {}
        for node in g.nodes:
            layers.setdefault(graph.nodes[node].round_id, []).append(node)
        pos = {}
        for x, layer in enumerate(sorted(layers)):
            ids = sorted(layers[layer])
            for y, node in enumerate(ids):
                pos[node] = (x, -y + len(ids) / 2)
        nx.draw_networkx(g, pos=pos, ax=ax, node_size=450, font_size=6, arrows=True, with_labels=False)
        labels = {node: graph.nodes[node].role[:8] for node in g.nodes}
        nx.draw_networkx_labels(g, pos=pos, labels=labels, ax=ax, font_size=6)
        ax.set_title(graph.workload)
        ax.axis("off")
    _save(fig, out_base)


def plot_shared_prefix_ratio(token_csv: str | Path, out_base: str | Path) -> None:
    df = pd.read_csv(token_csv)
    data = df.groupby("workload")["shared_prefix_ratio"].mean().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    data.plot(kind="bar", ax=ax, color="#4c78a8")
    ax.set_ylabel("Shared prefix ratio")
    ax.set_ylim(0, min(1.0, max(0.1, data.max() * 1.25)))
    _save(fig, out_base)


def plot_kv_duplication(kv_csv: str | Path, out_base: str | Path) -> None:
    df = pd.read_csv(kv_csv)
    data = df.groupby("workload")["kv_duplication_ratio"].mean().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    data.plot(kind="bar", ax=ax, color="#f58518")
    ax.set_ylabel("Naive/shared KV ratio")
    _save(fig, out_base)


def plot_latency_breakdown(lat_csv: str | Path, out_base: str | Path) -> None:
    df = pd.read_csv(lat_csv)
    data = df.groupby("workload")[["ttft_ms", "decode_ms", "tool_latency_ms"]].mean().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    data.plot(kind="bar", stacked=True, ax=ax, color=["#4c78a8", "#f58518", "#54a24b"])
    ax.set_ylabel("Mean node ms")
    _save(fig, out_base)


def plot_critical_path(cp_csv: str | Path, out_base: str | Path) -> None:
    df = pd.read_csv(cp_csv)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for workload, sub in df.groupby("workload"):
        ax.scatter(sub["total_work_weight"], sub["critical_path_weight"], label=workload, s=22)
    ax.set_xlabel("Total work weight")
    ax.set_ylabel("Critical path weight")
    ax.legend(fontsize=7)
    _save(fig, out_base)


def plot_main_speedup(summary_csv: str | Path, out_base: str | Path) -> None:
    df = pd.read_csv(summary_csv)
    base = float(df.loc[df["baseline"] == "wafer_naive", "job_completion_time_ms"].mean())
    df["speedup_vs_wafer_naive"] = base / df["job_completion_time_ms"].clip(lower=1e-9)
    fig, ax = plt.subplots(figsize=(6, 4))
    df.set_index("baseline")["speedup_vs_wafer_naive"].plot(kind="bar", ax=ax, color="#4c78a8")
    ax.set_ylabel("Speedup")
    _save(fig, out_base)


def plot_jct_distribution(metrics_csv: str | Path, out_base: str | Path) -> None:
    df = pd.read_csv(metrics_csv)
    fig, ax = plt.subplots(figsize=(7, 4))
    for baseline, sub in df.groupby("baseline"):
        ax.hist(sub["job_completion_time_ms"], bins=20, alpha=0.45, label=baseline)
    ax.set_xlabel("Job completion time (ms)")
    ax.set_ylabel("Jobs")
    ax.legend(fontsize=7)
    _save(fig, out_base)


def plot_kv_memory(summary_csv: str | Path, out_base: str | Path) -> None:
    df = pd.read_csv(summary_csv)
    fig, ax = plt.subplots(figsize=(6, 4))
    df.set_index("baseline")["kv_saving_ratio"].plot(kind="bar", ax=ax, color="#54a24b")
    ax.set_ylabel("KV saving ratio")
    _save(fig, out_base)


def plot_mesh_hotspot(summary_csv: str | Path, out_base: str | Path) -> None:
    df = pd.read_csv(summary_csv)
    fig, ax1 = plt.subplots(figsize=(7, 4))
    x = range(len(df))
    ax1.bar(x, df["mesh_total_traffic_bytes"], color="#4c78a8", label="traffic")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(df["baseline"], rotation=25, ha="right")
    ax1.set_ylabel("Traffic bytes")
    ax2 = ax1.twinx()
    ax2.plot(list(x), df["mesh_hotspot_ratio"], color="#e45756", marker="o", label="hotspot")
    ax2.set_ylabel("Hotspot ratio")
    _save(fig, out_base)


def plot_energy(summary_csv: str | Path, out_base: str | Path) -> None:
    df = pd.read_csv(summary_csv)
    fig, ax = plt.subplots(figsize=(6, 4))
    df.set_index("baseline")["energy_per_job_j"].plot(kind="bar", ax=ax, color="#b279a2")
    ax.set_ylabel("Energy per job (J)")
    _save(fig, out_base)


def plot_ablation(summary_csv: str | Path, out_base: str | Path) -> None:
    plot_main_speedup(summary_csv, out_base)


def plot_sensitivity(csv_path: str | Path, x_col: str, out_base: str | Path) -> None:
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    for baseline, sub in df.groupby("baseline"):
        sub = sub.sort_values(x_col)
        ax.plot(sub[x_col].astype(str), sub["job_completion_time_ms"], marker="o", label=baseline)
    ax.set_xlabel(x_col)
    ax.set_ylabel("JCT ms")
    ax.legend(fontsize=7)
    _save(fig, out_base)
