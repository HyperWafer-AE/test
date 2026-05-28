"""Trace-level proxy simulator for locality-aware Agent Island placement.

This is not a hardware simulator.  It estimates relative object movement on a
5x5 mesh under three simple placement policies:

* Baseline-random: each access executes on a random die, objects have random homes.
* Session-affinity: a trace executes on one die, but object homes remain global.
* Agent-island: a trace gets a 2x2/3x3 region and objects are homed inside it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

Die = tuple[int, int]


def _stable_index(key: str, n: int) -> int:
    if n <= 0:
        return 0
    digest = hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:12], 16) % n


def _mesh(size: int = 5) -> list[Die]:
    return [(x, y) for x in range(size) for y in range(size)]


def _hops(a: Die, b: Die) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _island_origins(mesh_size: int, island_size: int) -> list[Die]:
    return [(x, y) for x in range(mesh_size - island_size + 1) for y in range(mesh_size - island_size + 1)]


def _island_dies(origin: Die, island_size: int) -> list[Die]:
    return [(origin[0] + dx, origin[1] + dy) for dx in range(island_size) for dy in range(island_size)]


def _placement_maps(objects_df: pd.DataFrame, mesh_size: int, island_size: int) -> dict[str, Any]:
    dies = _mesh(mesh_size)
    trace_ids = sorted(objects_df["trace_id"].dropna().unique().tolist()) if not objects_df.empty else []
    origins = _island_origins(mesh_size, island_size)

    trace_die = {
        trace_id: dies[_stable_index(f"trace_die:{trace_id}", len(dies))] for trace_id in trace_ids
    }
    trace_island = {
        trace_id: _island_dies(origins[_stable_index(f"island:{trace_id}", len(origins))], island_size)
        for trace_id in trace_ids
    }
    return {"dies": dies, "trace_die": trace_die, "trace_island": trace_island}


def _exec_die(strategy: str, row: pd.Series, placements: dict[str, Any], row_idx: int) -> Die:
    dies: list[Die] = placements["dies"]
    trace_id = str(row["trace_id"])
    step_id = int(row["step_id"])
    if strategy == "baseline_random":
        return dies[_stable_index(f"exec:{trace_id}:{step_id}:{row_idx}", len(dies))]
    if strategy == "session_affinity":
        return placements["trace_die"].get(trace_id, dies[0])
    island = placements["trace_island"].get(trace_id, dies[:4])
    return island[_stable_index(f"island_exec:{trace_id}:{step_id}", len(island))]


def _home_die(strategy: str, row: pd.Series, placements: dict[str, Any]) -> Die:
    dies: list[Die] = placements["dies"]
    trace_id = str(row["trace_id"])
    object_id = str(row["object_id"])
    key = f"{trace_id}:{object_id}"
    if strategy in {"baseline_random", "session_affinity"}:
        return dies[_stable_index(f"home:{key}", len(dies))]
    island = placements["trace_island"].get(trace_id, dies[:4])
    return island[_stable_index(f"island_home:{key}", len(island))]


def simulate_strategy(
    objects_df: pd.DataFrame,
    strategy: str,
    mesh_size: int = 5,
    island_size: int = 3,
) -> dict[str, float | str]:
    if objects_df.empty:
        return {
            "strategy": strategy,
            "mesh": f"{mesh_size}x{mesh_size}",
            "island_size": island_size if strategy == "agent_island" else 0,
            "object_accesses": 0,
            "avg_hops": 0.0,
            "remote_object_accesses": 0,
            "moved_bytes": 0.0,
            "estimated_latency_units": 0.0,
        }

    placements = _placement_maps(objects_df, mesh_size, island_size)
    total_hops = 0
    remote = 0
    moved_bytes = 0.0
    access_count = 0

    tmp = objects_df.sort_values(["trace_id", "step_id"]).reset_index(drop=True)
    for idx, row in tmp.iterrows():
        size = max(float(pd.to_numeric(row.get("size_chars", 0), errors="coerce") or 0), 1.0)
        exec_die = _exec_die(strategy, row, placements, idx)
        home_die = _home_die(strategy, row, placements)
        hops = _hops(exec_die, home_die)
        total_hops += hops
        remote += int(hops > 0)
        moved_bytes += size * hops
        access_count += 1

    avg_hops = total_hops / access_count if access_count else 0.0
    latency = access_count + 0.35 * total_hops + moved_bytes / 10000.0
    return {
        "strategy": strategy,
        "mesh": f"{mesh_size}x{mesh_size}",
        "island_size": island_size if strategy == "agent_island" else 0,
        "object_accesses": int(access_count),
        "avg_hops": float(avg_hops),
        "remote_object_accesses": int(remote),
        "moved_bytes": float(moved_bytes),
        "estimated_latency_units": float(latency),
    }


def run_wafer_proxy_sim(
    objects_df: pd.DataFrame,
    figures_dir: Path,
    tables_dir: Path,
    mesh_size: int = 5,
    island_size: int = 3,
) -> dict[str, Any]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    strategies = ["baseline_random", "session_affinity", "agent_island"]
    rows = [
        simulate_strategy(objects_df, strategy, mesh_size=mesh_size, island_size=island_size)
        for strategy in strategies
    ]
    results = pd.DataFrame(rows)
    baseline_moved = float(results.loc[results["strategy"] == "baseline_random", "moved_bytes"].iloc[0])
    baseline_latency = float(
        results.loc[results["strategy"] == "baseline_random", "estimated_latency_units"].iloc[0]
    )
    results["movement_reduction_vs_random"] = (
        1.0 - results["moved_bytes"] / baseline_moved if baseline_moved > 0 else 0.0
    )
    results["latency_reduction_vs_random"] = (
        1.0 - results["estimated_latency_units"] / baseline_latency if baseline_latency > 0 else 0.0
    )
    results.to_csv(tables_dir / "wafer_proxy_results.csv", index=False)

    plot_df = results.melt(
        id_vars="strategy",
        value_vars=["movement_reduction_vs_random", "latency_reduction_vs_random"],
        var_name="metric",
        value_name="reduction",
    )
    fig, ax = plt.subplots(figsize=(8, 4.6))
    sns.barplot(data=plot_df, x="strategy", y="reduction", hue="metric", ax=ax, palette="Set2")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_ylim(min(-0.1, float(plot_df["reduction"].min()) - 0.05), 1.0)
    ax.set_title("Wafer Proxy Movement Reduction vs Random")
    ax.set_xlabel("")
    ax.set_ylabel("fraction reduction")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "wafer_proxy_movement_reduction.png", dpi=180)
    plt.close(fig)

    return {"wafer_proxy_results": results}
