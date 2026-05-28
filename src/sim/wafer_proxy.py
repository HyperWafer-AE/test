"""Trace-level proxy simulator for locality-aware Agent Island placement."""

from __future__ import annotations

import hashlib
from collections import defaultdict
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


def _early_hot_sets(
    steps_df: pd.DataFrame,
    objects_df: pd.DataFrame,
    k: int = 3,
    wrong_fraction: float = 0.0,
) -> dict[str, dict[str, set[str]]]:
    hot: dict[str, dict[str, set[str]]] = {}
    if steps_df.empty:
        return hot
    tmp = steps_df.sort_values(["trace_id", "step_id"]).copy()
    tmp["semantic_tool"] = tmp.get("semantic_tool", tmp.get("tool_name")).fillna("unknown").astype(str)
    tmp["phase"] = tmp["phase"].fillna("unknown").astype(str)
    all_tools = sorted(tmp["semantic_tool"].unique().tolist())
    all_phases = sorted(tmp["phase"].unique().tolist())
    all_object_types = sorted(objects_df["object_type"].dropna().astype(str).unique().tolist()) if not objects_df.empty else []
    early_object_types: dict[str, set[str]] = {}
    if not objects_df.empty:
        obj_tmp = objects_df.copy()
        obj_tmp["step_id"] = pd.to_numeric(obj_tmp["step_id"], errors="coerce").fillna(0).astype(int)
        obj_tmp["object_type"] = obj_tmp["object_type"].fillna("unknown").astype(str)
        early_obj = obj_tmp[obj_tmp["step_id"] < k]
        early_object_types = {
            trace_id: set(group["object_type"].tolist())
            for trace_id, group in early_obj.groupby("trace_id")
        }
    for trace_id, group in tmp.groupby("trace_id"):
        first = group.head(k)
        tools = set(first["semantic_tool"].tolist())
        phases = set(first["phase"].tolist())
        if wrong_fraction > 0:
            keep_tools = set()
            for tool in tools:
                if _stable_index(f"wrong-tool:{trace_id}:{tool}", 1000) / 1000.0 < wrong_fraction and all_tools:
                    keep_tools.add(all_tools[_stable_index(f"swap-tool:{trace_id}:{tool}", len(all_tools))])
                else:
                    keep_tools.add(tool)
            keep_phases = set()
            for phase in phases:
                if _stable_index(f"wrong-phase:{trace_id}:{phase}", 1000) / 1000.0 < wrong_fraction and all_phases:
                    keep_phases.add(all_phases[_stable_index(f"swap-phase:{trace_id}:{phase}", len(all_phases))])
                else:
                    keep_phases.add(phase)
            tools, phases = keep_tools, keep_phases
        object_types = set(early_object_types.get(trace_id, set()))
        if wrong_fraction > 0 and all_object_types:
            object_types = {
                all_object_types[_stable_index(f"swap-object-type:{trace_id}:{ot}", len(all_object_types))]
                if _stable_index(f"wrong-object-type:{trace_id}:{ot}", 1000) / 1000.0 < wrong_fraction
                else ot
                for ot in object_types
            }
        hot[trace_id] = {"semantic_tools": tools, "phases": phases, "object_types": object_types}
    return hot


def _row_is_predicted_hot(row: pd.Series, hot_sets: dict[str, dict[str, set[str]]]) -> bool:
    trace_id = str(row["trace_id"])
    hot = hot_sets.get(trace_id)
    if not hot:
        return False
    return (
        str(row.get("semantic_tool", "unknown")) in hot["semantic_tools"]
        or str(row.get("phase", "unknown")) in hot["phases"]
        or str(row.get("object_type", "unknown")) in hot["object_types"]
    )


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


def _global_home(row: pd.Series, placements: dict[str, Any]) -> Die:
    dies: list[Die] = placements["dies"]
    trace_id = str(row["trace_id"])
    object_id = str(row["object_id"])
    return dies[_stable_index(f"home:{trace_id}:{object_id}", len(dies))]


def _island_home(row: pd.Series, placements: dict[str, Any]) -> Die:
    dies: list[Die] = placements["dies"]
    trace_id = str(row["trace_id"])
    object_id = str(row["object_id"])
    island = placements["trace_island"].get(trace_id, dies[:4])
    return island[_stable_index(f"island_home:{trace_id}:{object_id}", len(island))]


def simulate_strategy(
    objects_df: pd.DataFrame,
    strategy: str,
    steps_df: pd.DataFrame | None = None,
    mesh_size: int = 5,
    island_size: int = 3,
    prediction_k: int = 3,
    object_capacity: int = 32,
    wrong_fraction: float = 0.35,
) -> dict[str, float | str]:
    if objects_df.empty:
        return {
            "strategy": strategy,
            "mesh": f"{mesh_size}x{mesh_size}",
            "island_size": island_size if strategy not in {"baseline_random", "session_affinity"} else 0,
            "object_accesses": 0,
            "avg_hops": 0.0,
            "remote_object_accesses": 0,
            "moved_bytes": 0.0,
            "estimated_latency_units": 0.0,
        }

    tmp = objects_df.sort_values(["trace_id", "step_id"]).reset_index(drop=True).copy()
    for col in ("semantic_tool", "phase", "object_type", "object_id"):
        if col not in tmp.columns:
            tmp[col] = "unknown"
        tmp[col] = tmp[col].fillna("unknown").astype(str)
    placements = _placement_maps(tmp, mesh_size, island_size)
    hot_sets = _early_hot_sets(
        steps_df if steps_df is not None else pd.DataFrame(),
        tmp,
        k=prediction_k,
        wrong_fraction=wrong_fraction if strategy == "wrong_prediction_stress" else 0.0,
    )
    capacity_used: dict[str, set[str]] = defaultdict(set)

    total_hops = 0
    remote = 0
    moved_bytes = 0.0
    access_count = 0
    island_homed = 0
    for idx, row in tmp.iterrows():
        size = max(float(pd.to_numeric(row.get("size_chars", 0), errors="coerce") or 0), 1.0)
        exec_die = _exec_die(strategy, row, placements, idx)

        trace_id = str(row["trace_id"])
        object_id = str(row["object_id"])
        if strategy in {"baseline_random", "session_affinity"}:
            home_die = _global_home(row, placements)
        elif strategy == "oracle_island":
            home_die = _island_home(row, placements)
            island_homed += 1
        elif strategy in {"early_fingerprint_island", "wrong_prediction_stress"}:
            if _row_is_predicted_hot(row, hot_sets):
                home_die = _island_home(row, placements)
                island_homed += 1
            else:
                home_die = _global_home(row, placements)
        elif strategy == "capacity_limited_island":
            if _row_is_predicted_hot(row, hot_sets) and (
                object_id in capacity_used[trace_id] or len(capacity_used[trace_id]) < object_capacity
            ):
                capacity_used[trace_id].add(object_id)
                home_die = _island_home(row, placements)
                island_homed += 1
            else:
                home_die = _global_home(row, placements)
        else:
            home_die = _global_home(row, placements)

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
        "island_size": island_size if strategy not in {"baseline_random", "session_affinity"} else 0,
        "object_accesses": int(access_count),
        "island_homed_accesses": int(island_homed),
        "avg_hops": float(avg_hops),
        "remote_object_accesses": int(remote),
        "moved_bytes": float(moved_bytes),
        "estimated_latency_units": float(latency),
    }


def _plot_reductions(results: pd.DataFrame, path: Path) -> None:
    plot_df = results.melt(
        id_vars="strategy",
        value_vars=["movement_reduction_vs_random", "latency_reduction_vs_random"],
        var_name="metric",
        value_name="reduction",
    )
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    sns.barplot(data=plot_df, x="strategy", y="reduction", hue="metric", ax=ax, palette="Set2")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_ylim(min(-0.2, float(plot_df["reduction"].min()) - 0.05), 1.0)
    ax.set_title("Wafer Proxy Movement and Latency Reduction vs Random")
    ax.set_xlabel("")
    ax.set_ylabel("fraction reduction")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_strategy_comparison(results: pd.DataFrame, path: Path) -> None:
    plot_df = results.melt(
        id_vars="strategy",
        value_vars=["avg_hops", "moved_bytes", "estimated_latency_units"],
        var_name="metric",
        value_name="value",
    )
    # Normalize per metric for a readable side-by-side comparison.
    plot_df["normalized_value"] = plot_df.groupby("metric")["value"].transform(
        lambda s: s / s.max() if s.max() > 0 else s
    )
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    sns.barplot(data=plot_df, x="strategy", y="normalized_value", hue="metric", ax=ax, palette="Set3")
    ax.set_title("Wafer Proxy Strategy Comparison")
    ax.set_xlabel("")
    ax.set_ylabel("normalized cost (lower is better)")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_wafer_proxy_sim(
    objects_df: pd.DataFrame,
    figures_dir: Path,
    tables_dir: Path,
    steps_df: pd.DataFrame | None = None,
    mesh_size: int = 5,
    island_size: int = 3,
    prediction_k: int = 3,
    object_capacity: int = 32,
) -> dict[str, Any]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    strategies = [
        "baseline_random",
        "session_affinity",
        "oracle_island",
        "early_fingerprint_island",
        "capacity_limited_island",
        "wrong_prediction_stress",
    ]
    rows = [
        simulate_strategy(
            objects_df,
            strategy,
            steps_df=steps_df,
            mesh_size=mesh_size,
            island_size=island_size,
            prediction_k=prediction_k,
            object_capacity=object_capacity,
        )
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

    _plot_reductions(results, figures_dir / "wafer_proxy_movement_reduction.png")
    _plot_strategy_comparison(results, figures_dir / "wafer_proxy_strategy_comparison.png")
    return {"wafer_proxy_results": results}
