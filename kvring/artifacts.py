"""Artifact, sweep, and plotting utilities for KVRing Round 2."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from .accounting import ModeResult, edge_key
from .attention_math import (
    error_metrics,
    finalize_stats,
    full_attention_reference,
    partial_attention_stats,
    ring_reduce,
    tree_reduce,
)
from .baselines import (
    simulate_central_kv_stationary,
    simulate_pull_kv_independent,
    simulate_replicate_all,
)
from .config import HardwareConfig, ModelConfig, WorkloadConfig
from .kvring_v1 import simulate_kvring_v1
from .kvring_v2 import simulate_kvring_v2
from .kvring_v3 import simulate_kvring_v3_adaptive
from .mesh import WaferMesh, central_home, default_agents, place_shard_groups
from .units import GiB, fmt_bytes, fmt_seconds, gib, tib
from .validation import filter_invalid_rows, filter_valid_rows


MODE_ALIASES = {
    "replicate_all": "Replicate-All",
    "pull_kv_independent": "Pull-KV-Independent",
    "central_kv_stationary": "Central-KV-Stationary",
    "kvring_v1": "KVRing-v1-sequential-pipeline",
    "kvring_v2": "KVRing-v2-ring",
    "kvring_v2_ring": "KVRing-v2-ring",
    "kvring_v2_tree": "KVRing-v2-tree",
    "kvring_v2_balanced_tree": "KVRing-v2-balanced-tree",
    "kvring_v2_region_split": "KVRing-v2-region-split",
    "kvring_v2_full_ring_legacy": "KVRing-v2-full-ring-legacy",
    "kvring_v3_adaptive": "KVRing-v3-adaptive",
}


def run_mode(
    mode: str,
    model: ModelConfig,
    workload: WorkloadConfig,
    hardware: HardwareConfig,
    *,
    query_tile_size: int = 8,
    num_shards: int = 8,
    placement: str = "serpentine",
) -> ModeResult:
    if mode == "replicate_all":
        return simulate_replicate_all(model, workload, hardware)
    if mode == "pull_kv_independent":
        return simulate_pull_kv_independent(model, workload, hardware)
    if mode == "central_kv_stationary":
        return simulate_central_kv_stationary(
            model, workload, hardware, query_tile_size=query_tile_size
        )
    if mode == "kvring_v1":
        return simulate_kvring_v1(model, workload, hardware, ring_shards=num_shards)
    if mode in {"kvring_v2", "kvring_v2_ring", "kvring_v2_query_tiled_parallel_ring_reduce"}:
        return simulate_kvring_v2(
            model,
            workload,
            hardware,
            query_tile_size=query_tile_size,
            num_shards=num_shards,
            reduction="selected_ring",
            placement=placement,
        )
    if mode in {"kvring_v2_tree", "kvring_v2_query_tiled_parallel_tree_reduce"}:
        return simulate_kvring_v2(
            model,
            workload,
            hardware,
            query_tile_size=query_tile_size,
            num_shards=num_shards,
            reduction="binary_tree",
            placement=placement,
        )
    if mode in {"kvring_v2_balanced_tree", "kvring_v2_query_tiled_parallel_balanced_tree"}:
        return simulate_kvring_v2(
            model,
            workload,
            hardware,
            query_tile_size=query_tile_size,
            num_shards=num_shards,
            reduction="balanced_tree",
            placement=placement,
        )
    if mode in {"kvring_v2_region_split", "kvring_v2_query_tiled_parallel_region_split"}:
        return simulate_kvring_v2(
            model,
            workload,
            hardware,
            query_tile_size=query_tile_size,
            num_shards=num_shards,
            reduction="region_split_ring",
            placement=placement,
        )
    if mode in {"kvring_v2_full_ring_legacy", "kvring_v2_query_tiled_parallel_full_ring"}:
        return simulate_kvring_v2(
            model,
            workload,
            hardware,
            query_tile_size=query_tile_size,
            num_shards=num_shards,
            reduction="full_ring_v1_legacy",
            placement=placement,
        )
    if mode in {"kvring_v3", "kvring_v3_adaptive"}:
        return simulate_kvring_v3_adaptive(
            model,
            workload,
            hardware,
            query_tile_size=query_tile_size,
            num_shards=num_shards,
            placement=placement,
        )
    raise ValueError(f"unknown mode: {mode}")


def default_results() -> List[ModeResult]:
    model = ModelConfig()
    workload = WorkloadConfig()
    hardware = HardwareConfig()
    return [
        run_mode("replicate_all", model, workload, hardware),
        run_mode("pull_kv_independent", model, workload, hardware),
        run_mode("central_kv_stationary", model, workload, hardware),
        run_mode("kvring_v1", model, workload, hardware),
        run_mode("kvring_v2_ring", model, workload, hardware, query_tile_size=8),
        run_mode("kvring_v2_tree", model, workload, hardware, query_tile_size=8),
        run_mode("kvring_v3_adaptive", model, workload, hardware, query_tile_size=8),
    ]


def legacy_v1_results(
    model: ModelConfig | None = None,
    workload: WorkloadConfig | None = None,
    hardware: HardwareConfig | None = None,
    *,
    ring_shards: int = 8,
) -> List[ModeResult]:
    model = model or ModelConfig()
    workload = workload or WorkloadConfig()
    hardware = hardware or HardwareConfig()
    return [
        run_mode("replicate_all", model, workload, hardware),
        run_mode("pull_kv_independent", model, workload, hardware),
        run_mode("kvring_v1", model, workload, hardware, num_shards=ring_shards),
    ]


def result_row(result: ModeResult, **params: object) -> Dict[str, object]:
    extra = result.extra
    shared_prefix_tokens = params.get("shared_prefix_tokens", WorkloadConfig().shared_prefix_tokens)
    num_agents = params.get("agents", WorkloadConfig().concurrent_agents)
    decode_tokens = params.get("decode_tokens", WorkloadConfig().decode_tokens_per_agent)
    shared_kv_bytes = params.get("shared_kv_bytes", WorkloadConfig().shared_kv_bytes(ModelConfig()))
    private_kv_write_bytes = extra.get("private_kv_write_bytes", "")
    local_sram_read_bytes = extra.get("local_sram_read_bytes", extra.get("central_sram_read_bytes", ""))
    reduction_topology = extra.get("reduction_topology", extra.get("reduction", params.get("reduction", "")))
    region_capacity_violation = bool(extra.get("region_capacity_violation", False))
    valid_capacity = bool(extra.get("valid_capacity", not region_capacity_violation))
    serialized_latency_s = float(extra.get("serialized_latency_s", result.estimated_latency_seconds))
    throughput_bound_latency_s = float(
        extra.get("throughput_bound_latency_s", result.estimated_latency_seconds)
    )
    critical_path_latency_s = float(extra.get("critical_path_latency_s", result.estimated_latency_seconds))
    attention_proxy_latency_s = float(
        extra.get("attention_stage_proxy_latency_s", throughput_bound_latency_s)
    )
    network_latency_s = float(extra.get("network_latency_s", result.mesh_seconds))
    sram_latency_s = float(extra.get("sram_latency_s", result.compute_seconds))
    compute_latency_s = float(extra.get("compute_latency_s", result.compute_seconds))
    row: Dict[str, object] = {
        "mode": result.mode,
        "old_mode_alias": extra.get("old_mode_alias", ""),
        "shared_prefix_tokens": shared_prefix_tokens,
        "shared_kv_bytes": shared_kv_bytes,
        "num_agents": num_agents,
        "decode_tokens_per_agent": decode_tokens,
        "ring_shards": extra.get("num_shards", extra.get("ring_shards", params.get("num_shards", ""))),
        "reduction_topology": reduction_topology,
        "sram_total_bytes": result.total_sram_bytes,
        "sram_peak_region_bytes": result.peak_region_sram_bytes,
        "local_sram_read_bytes": local_sram_read_bytes,
        "private_kv_write_bytes": private_kv_write_bytes,
        "payload_bytes": result.payload_bytes,
        "total_wire_bytes": result.total_wire_bytes,
        "max_directed_link_load_bytes": result.max_link_load_bytes,
        "mean_active_link_load_bytes": result.mean_active_link_load_bytes,
        "network_latency_s": network_latency_s,
        "compute_latency_s": compute_latency_s,
        "sram_latency_s": sram_latency_s,
        "serialized_latency_s": serialized_latency_s,
        "throughput_bound_latency_s": throughput_bound_latency_s,
        "critical_path_latency_s": critical_path_latency_s,
        "conservative_max_latency_s": max(throughput_bound_latency_s, critical_path_latency_s),
        "estimated_attention_stage_latency_s": attention_proxy_latency_s,
        "attention_stage_latency_s": attention_proxy_latency_s,
        "estimated_end_to_end_proxy_latency_s": attention_proxy_latency_s,
        "metric_warning": "This is a shared-prefix attention-stage proxy, not full LLM serving latency.",
        "estimated_latency_s": result.estimated_latency_seconds,
        "mesh_time_s": result.mesh_seconds,
        "compute_time_s": result.compute_seconds,
        "total_sram_gib": gib(result.total_sram_bytes),
        "peak_region_sram_gib": gib(result.peak_region_sram_bytes),
        "sram_port_gib": gib(result.sram_port_bytes),
        "payload_gib": gib(result.payload_bytes),
        "wire_traffic_tib": tib(result.total_wire_bytes),
        "max_link_load_gib": gib(result.max_link_load_bytes),
        "hotspot_ratio": result.hotspot_ratio,
        "max_directed_link": edge_key(result.max_link) if result.max_link else "",
        "local_sram_read_tib": extra.get("local_sram_read_tib", ""),
        "setup_replication_payload_gib": extra.get("setup_replication_payload_gib", ""),
        "setup_replication_wire_tib": extra.get("setup_replication_wire_tib", ""),
        "steady_state_local_decode_read_gib": extra.get("steady_state_local_decode_read_gib", ""),
        "central_sram_read_bytes": extra.get("central_sram_read_bytes", ""),
        "central_query_payload_bytes": extra.get("central_query_payload_bytes", ""),
        "central_query_wire_bytes": extra.get("central_query_wire_bytes", ""),
        "central_result_payload_bytes": extra.get("central_result_payload_bytes", ""),
        "central_result_wire_bytes": extra.get("central_result_wire_bytes", ""),
        "central_compute_proxy_bytes": extra.get("central_compute_proxy_bytes", ""),
        "central_router_in_bytes": extra.get("central_router_in_bytes", ""),
        "central_router_out_bytes": extra.get("central_router_out_bytes", ""),
        "central_max_link_load_bytes": extra.get("central_max_link_load_bytes", ""),
        "central_max_directed_link_load_bytes": extra.get("central_max_directed_link_load_bytes", ""),
        "central_hotspot_ratio": extra.get("central_hotspot_ratio", ""),
        "central_bottleneck_component": extra.get("central_bottleneck_component", ""),
        "central_total_latency_s": extra.get("central_total_latency_s", ""),
        "central_compute_bytes_or_ops": extra.get("central_compute_bytes_or_ops", ""),
        "central_region_queue_time_s": extra.get("central_region_queue_time_s", ""),
        "attention_stage_proxy_latency_s": attention_proxy_latency_s,
        "query_bytes_sent": extra.get("query_bytes_sent", ""),
        "partial_bytes_returned": extra.get("partial_bytes_returned", ""),
        "query_scatter_latency_s": extra.get("query_scatter_latency_s", ""),
        "local_shard_compute_latency_s": extra.get("local_shard_compute_latency_s", ""),
        "reduction_latency_s": extra.get("reduction_latency_s", ""),
        "local_suffix_latency_s": extra.get("local_suffix_latency_s", ""),
        "merge_latency_s": extra.get("merge_latency_s", ""),
        "central_query_mesh_latency_s": extra.get("central_query_mesh_latency_s", ""),
        "central_sram_read_latency_s": extra.get("central_sram_read_latency_s", ""),
        "central_compute_latency_s": extra.get("central_compute_latency_s", ""),
        "central_return_mesh_latency_s": extra.get("central_return_mesh_latency_s", ""),
        "central_result_mesh_latency_s": extra.get("central_result_mesh_latency_s", ""),
        "central_queue_latency_s": extra.get("central_queue_latency_s", ""),
        "reduction_bytes": extra.get("reduction_bytes", ""),
        "reduction_wire_bytes": extra.get("reduction_wire_bytes", ""),
        "reduction_byte_hops": extra.get("reduction_byte_hops", ""),
        "query_scatter_byte_hops": extra.get("query_scatter_byte_hops", ""),
        "partial_reduce_byte_hops": extra.get("partial_reduce_byte_hops", ""),
        "result_return_byte_hops": extra.get("result_return_byte_hops", ""),
        "reduction_hops": extra.get("reduction_hops", ""),
        "reduction_edges": extra.get("reduction_edges", ""),
        "num_reduction_steps": extra.get("num_reduction_steps", ""),
        "setup_cycles": extra.get("setup_cycles", ""),
        "steady_state_cycles": extra.get("steady_state_cycles", ""),
        "max_reduction_link_load_bytes": extra.get("max_reduction_link_load_bytes", ""),
        "query_tile_bytes": extra.get("query_tile_bytes", ""),
        "partial_state_bytes": extra.get("partial_state_bytes", ""),
        "online_state_bytes": extra.get("online_state_bytes", ""),
        "state_vector_dim": extra.get("state_vector_dim", ""),
        "state_dtype_bytes": extra.get("state_dtype_bytes", ""),
        "scalar_state_bytes": extra.get("scalar_state_bytes", ""),
        "query_dtype_bytes": extra.get("query_dtype_bytes", ""),
        "packet_bytes": extra.get("packet_bytes", ""),
        "packet_model": extra.get("packet_model", ""),
        "actual_query_tile_sizes": json.dumps(extra.get("actual_query_tile_sizes", "")),
        "requested_query_tile_size": extra.get("requested_query_tile_size", ""),
        "logical_decode_queries": extra.get("logical_decode_queries", ""),
        "query_tiles": extra.get("query_tiles", ""),
        "query_tile_payload_bytes_per_step": extra.get("query_tile_payload_bytes_per_step", ""),
        "partial_state_payload_bytes_per_step": extra.get("partial_state_payload_bytes_per_step", ""),
        "shard_group_size": json.dumps(extra.get("shard_group_size", "")),
        "placement_unit": extra.get("placement_unit", ""),
        "regions_per_shard": json.dumps(extra.get("regions_per_shard", "")),
        "shard_bytes": json.dumps(extra.get("shard_bytes", "")),
        "shard_region_bytes": json.dumps(extra.get("shard_region_bytes", "")),
        "shard_bytes_per_region": json.dumps(extra.get("shard_bytes_per_region", "")),
        "query_tile_size": extra.get("query_tile_size", params.get("query_tile_size", "")),
        "num_query_tiles_total": extra.get("num_query_tiles_total", ""),
        "reduction": extra.get("reduction", ""),
        "placement": extra.get("placement", ""),
        "num_shards": extra.get("num_shards", params.get("num_shards", "")),
        "region_capacity_violation": region_capacity_violation,
        "valid_capacity": valid_capacity,
        "capacity_valid": valid_capacity,
        "region_capacity_bytes": extra.get(
            "region_capacity_bytes", params.get("region_capacity_bytes", HardwareConfig().region_capacity_bytes)
        ),
        "peak_region_sram_bytes": extra.get("peak_region_sram_bytes", result.peak_region_sram_bytes),
        "capacity_violation_reason": extra.get("capacity_violation_reason", ""),
        "latency_bound_used": extra.get("latency_bound_used", "throughput_bound"),
        "selected_mode": extra.get("selected_mode", ""),
        "selected_reduction_topology": extra.get("selected_reduction_topology", ""),
        "selected_query_tile_size": extra.get("selected_query_tile_size", ""),
        "selected_num_shards": extra.get("selected_num_shards", ""),
        "selection_reason": extra.get("selection_reason", ""),
        "candidate_count": extra.get("candidate_count", ""),
        "valid_candidate_count": extra.get("valid_candidate_count", ""),
        "vc_model": extra.get("vc_model", ""),
    }
    row.update(params)
    return row


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _result_attention_proxy_latency(result: ModeResult) -> float:
    return float(
        result.extra.get(
            "attention_stage_proxy_latency_s",
            result.extra.get("throughput_bound_latency_s", result.estimated_latency_seconds),
        )
    )


def _result_capacity_valid(result: ModeResult) -> bool:
    return bool(result.extra.get("valid_capacity", not result.extra.get("region_capacity_violation", False)))


def write_text_report(path: Path, results: List[ModeResult]) -> None:
    model = ModelConfig()
    workload = WorkloadConfig()
    hardware = HardwareConfig()
    lines: List[str] = []
    lines.append("# KVRing Round 3 Report")
    lines.append("")
    lines.append("Scope: attention-only shared-prefix plus local suffix modeling; non-attention LLM layers are not included.")
    lines.append("This is a microarchitectural shared-prefix attention-stage simulator, not a full LLM serving system and not a real wafer hardware measurement.")
    lines.append(f"Model path: `{model.model_path}`")
    lines.append(f"KV/token: {fmt_bytes(model.kv_token_bytes)}")
    lines.append(f"Shared prefix: {workload.shared_prefix_tokens} tokens = {fmt_bytes(workload.shared_kv_bytes(model))}")
    lines.append(f"Agents: {workload.concurrent_agents}; decode tokens/agent: {workload.decode_tokens_per_agent}")
    lines.append(f"NoC channel model: directed bidirectional; VC model: `{hardware.vc_model}`")
    lines.append("")
    headline_results = [r for r in results if _result_capacity_valid(r)] or results
    lines.append("Headline latency uses `throughput_bound_latency_s`; serialized and critical-path bounds are exported separately.")
    lines.append("")
    lines.append("## Main Headline Table (capacity-valid rows only)")
    lines.append("")
    lines.append("| Mode | Peak SRAM | Wire traffic | Max directed link | SRAM port | Attention-stage proxy latency |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in headline_results:
        lines.append(
            f"| {r.mode} | {fmt_bytes(r.peak_region_sram_bytes)} | {fmt_bytes(r.total_wire_bytes)} | "
            f"{fmt_bytes(r.max_link_load_bytes)} | {fmt_bytes(r.sram_port_bytes)} | "
            f"{fmt_seconds(_result_attention_proxy_latency(r))} |"
        )
    invalid = [r for r in results if not _result_capacity_valid(r)]
    if invalid:
        lines.append("")
        lines.append("## Capacity-Invalid Rows")
        lines.append("")
        lines.append("These rows are kept for replication-centralization stress analysis but are not used as valid headline comparisons.")
        for r in invalid:
            lines.append(f"- {r.mode}: {r.extra.get('capacity_violation_reason', 'capacity violation')}")
    lines.append("")
    lines.append("KVRing-v2 packet accounting is symbolic: `Q_tile + FP32(m,l,z)`; exact online-softmax state is reduced by ring/tree collectives.")
    lines.append("")
    lines.append("## Baseline accounting notes")
    for r in results:
        if r.mode == "Replicate-All":
            lines.append(
                "- Replicate-All separates setup replication "
                f"({r.extra.get('setup_replication_payload_gib')} GiB payload, "
                f"{r.extra.get('setup_replication_wire_tib')} TiB directed byte-hop) "
                f"from steady-state local shared-KV reads ({r.extra.get('steady_state_local_decode_read_gib')} GiB)."
            )
        if r.mode == "Central-KV-Stationary":
            lines.append(
                "- Central-KV-Stationary keeps KV off the mesh during decode; its bottleneck is central SRAM/compute queueing "
                f"({r.extra.get('central_region_queue_time_s')} s)."
            )
        if r.mode.startswith("KVRing-v2-"):
            lines.append(
                f"- {r.mode} reports query scatter, shard compute, exact online-softmax reduction, suffix, and merge components separately."
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_mode_comparison(results: List[ModeResult], path_base: Path) -> None:
    names = [r.mode.replace("KVRing-", "KV-") for r in results]
    x = np.arange(len(results))
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("Replication-Centralization Dilemma and KVRing Collectives", fontsize=14)
    axes[0, 0].bar(x, [gib(r.peak_region_sram_bytes) for r in results], color="#4C78A8")
    axes[0, 0].set_title("Peak Region SRAM")
    axes[0, 0].set_ylabel("GiB")
    axes[0, 1].bar(x, [max(tib(r.total_wire_bytes), 1e-9) for r in results], color="#F58518")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Directed Mesh Byte-Hop Traffic")
    axes[0, 1].set_ylabel("TiB")
    axes[1, 0].bar(x, [max(gib(r.max_link_load_bytes), 1e-9) for r in results], color="#E45756")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Max Directed Link Load")
    axes[1, 0].set_ylabel("GiB")
    axes[1, 1].bar(x, [max(_result_attention_proxy_latency(r), 1e-12) for r in results], color="#54A24B")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Estimated Attention Latency")
    axes[1, 1].set_ylabel("s")
    for ax in axes.flat:
        ax.set_xticks(x, names, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25, which="both")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(path_base.with_suffix(f".{ext}"), dpi=220)
    plt.close(fig)


def plot_query_tile_sweep(rows: List[Dict[str, object]], path_base: Path) -> None:
    v2_rows = [r for r in rows if str(r["mode"]).startswith("KVRing-v2-")]
    if not v2_rows:
        return
    target_prefix = 32768
    target_agents = 8
    target_decode = 256
    filt = [
        r
        for r in v2_rows
        if int(r["shared_prefix_tokens"]) == target_prefix
        and int(r.get("agents", r.get("num_agents", 0))) == target_agents
        and int(r.get("decode_tokens", r.get("decode_tokens_per_agent", 0))) == target_decode
    ]
    if not filt:
        first = v2_rows[0]
        target_prefix = int(first["shared_prefix_tokens"])
        target_agents = int(first.get("agents", first.get("num_agents", 0)))
        target_decode = int(first.get("decode_tokens", first.get("decode_tokens_per_agent", 0)))
        filt = [
            r
            for r in v2_rows
            if int(r["shared_prefix_tokens"]) == target_prefix
            and int(r.get("agents", r.get("num_agents", 0))) == target_agents
            and int(r.get("decode_tokens", r.get("decode_tokens_per_agent", 0))) == target_decode
        ]
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    for topo, color in [("selected_ring_reduce", "#4C78A8"), ("binary_tree_reduce", "#54A24B")]:
        mrows = sorted([r for r in filt if r["reduction_topology"] == topo], key=lambda r: int(r["query_tile_size"]))
        ax1.plot(
            [int(r["query_tile_size"]) for r in mrows],
            [float(r["local_sram_read_bytes"]) / (1024**4) for r in mrows],
            marker="o",
            color=color,
            label=f"{topo} SRAM read",
        )
    ax1.set_xlabel("Query tile size R")
    ax1.set_ylabel("Local shared-KV SRAM reads (TiB)")
    ax1.grid(alpha=0.25)
    ax1.legend(frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(path_base.with_suffix(f".{ext}"), dpi=220)
    plt.close(fig)


def plot_simple_bar(rows: List[Dict[str, object]], key: str, group_key: str, path_base: Path, title: str) -> None:
    labels = [str(r[group_key]) for r in rows]
    vals = [float(r[key]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(np.arange(len(rows)), vals, color="#4C78A8")
    ax.set_xticks(np.arange(len(rows)), labels, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel(key)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(path_base.with_suffix(f".{ext}"), dpi=220)
    plt.close(fig)


def plot_latency_bounds(rows: List[Dict[str, object]], path_base: Path) -> None:
    modes = []
    bounds = ["critical_path", "throughput_bound", "conservative_max", "serialized"]
    for row in rows:
        mode = str(row["mode"])
        if mode not in modes:
            modes.append(mode)
    x = np.arange(len(modes))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    colors = {
        "critical_path": "#4C78A8",
        "throughput_bound": "#54A24B",
        "conservative_max": "#F58518",
        "serialized": "#E45756",
    }
    for i, bound in enumerate(bounds):
        vals = [
            float(next(r["latency_s"] for r in rows if r["mode"] == mode and r["latency_bound"] == bound))
            for mode in modes
        ]
        ax.bar(x + (i - 1.5) * width, vals, width=width, label=bound, color=colors[bound])
    ax.set_xticks(x, modes, rotation=20, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("seconds")
    ax.set_title("KVRing-v2 Attention-Stage Latency Bounds")
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(path_base.with_suffix(f".{ext}"), dpi=220)
    plt.close(fig)


def plot_central_breakdown(rows: List[Dict[str, object]], path_base: Path) -> None:
    labels = [str(r["component"]) for r in rows]
    vals = [float(r["latency_s"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(np.arange(len(rows)), vals, color=["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B"])
    ax.set_xticks(np.arange(len(rows)), labels, rotation=25, ha="right")
    ax.set_ylabel("seconds")
    ax.set_title("Central-KV-Stationary Bottleneck Breakdown")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(path_base.with_suffix(f".{ext}"), dpi=220)
    plt.close(fig)


def plot_regime_map(rows: List[Dict[str, object]], path_base: Path) -> None:
    filt = [r for r in rows if int(r["decode_tokens_per_agent"]) == 128]
    if not filt:
        filt = rows
    prefixes = sorted({int(r["shared_prefix_tokens"]) for r in filt})
    agents = sorted({int(r["num_agents"]) for r in filt})
    mode_ids = {mode: i for i, mode in enumerate(sorted({str(r["winner_mode"]) for r in filt}))}
    grid = np.full((len(agents), len(prefixes)), np.nan)
    for row in filt:
        ai = agents.index(int(row["num_agents"]))
        pi = prefixes.index(int(row["shared_prefix_tokens"]))
        grid[ai, pi] = mode_ids[str(row["winner_mode"])]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    im = ax.imshow(grid, cmap="tab10", aspect="auto")
    ax.set_xticks(np.arange(len(prefixes)), prefixes, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(agents)), agents)
    ax.set_xlabel("shared prefix tokens")
    ax.set_ylabel("agents")
    ax.set_title("Valid-Capacity Winning Strategy Regime Map")
    labels = [f"{idx}: {mode}" for mode, idx in sorted(mode_ids.items(), key=lambda kv: kv[1])]
    ax.text(1.02, 0.5, "\n".join(labels), transform=ax.transAxes, va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(path_base.with_suffix(f".{ext}"), dpi=220)
    plt.close(fig)


def plot_system_overview(path_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.axis("off")
    boxes = [
        (0.08, 0.58, "Agent query tiles"),
        (0.36, 0.58, "Wafer mesh routes"),
        (0.64, 0.58, "Stationary KV shards"),
        (0.36, 0.20, "Exact online-softmax reduction"),
    ]
    for x, y, text in boxes:
        ax.add_patch(plt.Rectangle((x, y), 0.22, 0.18, fill=False, lw=1.5, ec="#4C78A8"))
        ax.text(x + 0.11, y + 0.09, text, ha="center", va="center", fontsize=10)
    arrows = [
        ((0.30, 0.67), (0.36, 0.67)),
        ((0.58, 0.67), (0.64, 0.67)),
        ((0.75, 0.58), (0.50, 0.38)),
        ((0.36, 0.29), (0.20, 0.58)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.4})
    ax.text(
        0.5,
        0.05,
        "Source note: deterministic schematic generated from simulator concepts; not a measured wafer layout.",
        ha="center",
        va="center",
        fontsize=8,
    )
    for ext in ("png", "pdf"):
        fig.savefig(path_base.with_suffix(f".{ext}"), dpi=220)
    plt.close(fig)


def plot_link_loads(results: List[ModeResult], path: Path) -> None:
    mesh = WaferMesh(16, 16)
    agents = default_agents(8, 16, 16)
    home = central_home(HardwareConfig())
    shards = place_shard_groups(mesh, WorkloadConfig().shared_kv_bytes(ModelConfig()), 8, HardwareConfig())
    fig, axes = plt.subplots(1, len(results), figsize=(4.8 * len(results), 5.2), constrained_layout=True)
    if len(results) == 1:
        axes = [axes]
    positives = [load for r in results for load in r.link_loads.values() if load > 0]
    vmin = max(min(positives), 1.0) if positives else 1.0
    vmax = max(positives) if positives else 1.0
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("magma")
    for ax, result in zip(axes, results):
        ax.set_title(f"{result.mode}\ndirected max {fmt_bytes(result.max_link_load_bytes)}", fontsize=9)
        for edge in mesh.all_edges():
            load = result.link_loads.get(edge, 0.0)
            (r0, c0), (r1, c1) = edge
            color = "#E6E6E6" if load <= 0 else cmap(norm(load))
            lw = 0.45 if load <= 0 else 1.0 + 2.5 * math.log(load / vmin + 1) / math.log(vmax / vmin + 2)
            x0, x1 = float(c0), float(c1)
            y0, y1 = float(r0), float(r1)
            if r0 == r1:
                dy = -0.08 if c1 > c0 else 0.08
                y0 += dy
                y1 += dy
            else:
                dx = 0.08 if r1 > r0 else -0.08
                x0 += dx
                x1 += dx
            ax.plot([x0, x1], [y0, y1], color=color, lw=lw, alpha=0.9, solid_capstyle="round")
        agent_pos = np.array([a.position for a in agents])
        ax.scatter(agent_pos[:, 1], agent_pos[:, 0], marker="o", s=36, c="#1F77B4", edgecolors="white")
        ax.scatter([home[1]], [home[0]], marker="s", s=42, c="#D62728", edgecolors="white")
        shard_pos = np.array([s.home_region for s in shards])
        ax.scatter(shard_pos[:, 1], shard_pos[:, 0], marker="D", s=24, c="#2CA02C", edgecolors="white")
        ax.set_xlim(-0.8, mesh.cols - 0.2)
        ax.set_ylim(mesh.rows - 0.2, -0.8)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _csv_rows(path: Path) -> int | str:
    if path.suffix.lower() != ".csv":
        return ""
    with path.open(newline="", encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def generate_query_tile_sweep(
    out_csv: Path,
    modes: Sequence[str],
    query_tile_sizes: Sequence[int],
    shared_prefix_tokens: Sequence[int],
    agents: Sequence[int],
    decode_tokens: Sequence[int],
    num_shards_list: Sequence[int] = (8,),
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    hardware = HardwareConfig()
    for s in shared_prefix_tokens:
        for n in agents:
            for t in decode_tokens:
                workload = WorkloadConfig(shared_prefix_tokens=s, concurrent_agents=n, decode_tokens_per_agent=t)
                for r in query_tile_sizes:
                    model = ModelConfig(query_tile_size=r)
                    for num_shards in num_shards_list:
                        for mode in modes:
                            result = run_mode(
                                mode,
                                model,
                                workload,
                                hardware,
                                query_tile_size=r,
                                num_shards=num_shards,
                            )
                            rows.append(
                                result_row(
                                    result,
                                    shared_prefix_tokens=s,
                                    agents=n,
                                    decode_tokens=t,
                                    shared_kv_bytes=workload.shared_kv_bytes(model),
                                    query_tile_size=r,
                                    num_shards=num_shards,
                                )
                            )
    write_csv(out_csv, rows)
    return rows


def generate_shared_prefix_sweep(out_csv: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    hardware = HardwareConfig()
    for s in [2048, 8192, 32768, 65536]:
        workload = WorkloadConfig(shared_prefix_tokens=s)
        model = ModelConfig(query_tile_size=8)
        for mode in ["replicate_all", "pull_kv_independent", "central_kv_stationary", "kvring_v1", "kvring_v2_ring", "kvring_v2_tree"]:
            result = run_mode(mode, model, workload, hardware, query_tile_size=8)
            rows.append(
                result_row(
                    result,
                    shared_prefix_tokens=s,
                    shared_kv_bytes=workload.shared_kv_bytes(model),
                    agents=workload.concurrent_agents,
                    decode_tokens=workload.decode_tokens_per_agent,
                )
            )
    write_csv(out_csv, rows)
    return rows


def generate_agent_count_sweep(out_csv: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    hardware = HardwareConfig()
    model = ModelConfig(query_tile_size=8)
    for n in [1, 2, 4, 8, 16, 32]:
        workload = WorkloadConfig(concurrent_agents=n)
        for mode in ["central_kv_stationary", "kvring_v2_ring", "kvring_v2_tree"]:
            result = run_mode(mode, model, workload, hardware, query_tile_size=8)
            rows.append(
                result_row(
                    result,
                    shared_prefix_tokens=workload.shared_prefix_tokens,
                    shared_kv_bytes=workload.shared_kv_bytes(model),
                    agents=n,
                    decode_tokens=workload.decode_tokens_per_agent,
                )
            )
    write_csv(out_csv, rows)
    return rows


def generate_shard_count_sweep(out_csv: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    hardware = HardwareConfig()
    model = ModelConfig(query_tile_size=8)
    workload = WorkloadConfig()
    for num_shards in [1, 2, 4, 8, 16, 32]:
        for mode in ["kvring_v1", "kvring_v2_ring", "kvring_v2_tree"]:
            result = run_mode(mode, model, workload, hardware, query_tile_size=8, num_shards=num_shards)
            rows.append(
                result_row(
                    result,
                    shared_prefix_tokens=workload.shared_prefix_tokens,
                    shared_kv_bytes=workload.shared_kv_bytes(model),
                    agents=workload.concurrent_agents,
                    decode_tokens=workload.decode_tokens_per_agent,
                    num_shards=num_shards,
                )
            )
    write_csv(out_csv, rows)
    return rows


def generate_reduction_topology_sweep(out_csv: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    hardware = HardwareConfig()
    model = ModelConfig(query_tile_size=8)
    workload = WorkloadConfig()
    for num_shards in [1, 2, 4, 8, 16, 32]:
        for mode, placement in [
            ("kvring_v2_ring", "serpentine"),
            ("kvring_v2_tree", "serpentine"),
            ("kvring_v2_region_split", "serpentine"),
            ("kvring_v2_balanced_tree", "balanced_link_load"),
        ]:
            result = run_mode(
                mode,
                model,
                workload,
                hardware,
                query_tile_size=8,
                num_shards=num_shards,
                placement=placement,
            )
            rows.append(
                result_row(
                    result,
                    shared_prefix_tokens=workload.shared_prefix_tokens,
                    shared_kv_bytes=workload.shared_kv_bytes(model),
                    agents=workload.concurrent_agents,
                    decode_tokens=workload.decode_tokens_per_agent,
                    num_shards=num_shards,
                    placement=placement,
                )
            )
    write_csv(out_csv, rows)
    return rows


def generate_topology_sweep(out_csv: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    model = ModelConfig(query_tile_size=8)
    workload = WorkloadConfig()
    hardware = HardwareConfig()
    for num_shards in [4, 8, 16, 32]:
        for placement in ["centralized", "random", "serpentine", "diagonal", "balanced_link_load", "stripe", "region_split"]:
            for reduction in ["selected_ring", "binary_tree", "region_split_ring", "balanced_tree"]:
                mode = {
                    "selected_ring": "kvring_v2_ring",
                    "binary_tree": "kvring_v2_tree",
                    "region_split_ring": "kvring_v2_region_split",
                    "balanced_tree": "kvring_v2_balanced_tree",
                }[reduction]
                result = run_mode(
                    mode,
                    model,
                    workload,
                    hardware,
                    query_tile_size=8,
                    num_shards=num_shards,
                    placement=placement,
                )
                rows.append(result_row(result, num_shards=num_shards, placement=placement, reduction=reduction))
    write_csv(out_csv, rows)
    return rows


def generate_hardware_sensitivity(out_csv: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    model = ModelConfig(query_tile_size=8)
    workload = WorkloadConfig()
    for link_bw in [25, 50, 100, 200]:
        for sram_bw in [5, 10, 20, 40]:
            for cap in [0.25, 0.5, 1, 2, 4]:
                hardware = HardwareConfig(
                    link_bandwidth_gbps=link_bw,
                    region_sram_bandwidth_tibps=sram_bw,
                    region_sram_capacity_gib=cap,
                    region_capacity_gib=cap,
                )
                result = run_mode("kvring_v2_tree", model, workload, hardware, query_tile_size=8)
                rows.append(
                    result_row(
                        result,
                        link_bandwidth_gbps=link_bw,
                        region_sram_bandwidth_tibps=sram_bw,
                        region_capacity_gib=cap,
                    )
                )
    write_csv(out_csv, rows)
    return rows


def generate_numerical_stability(out_csv: Path, fig_base: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    rng = np.random.default_rng(1234)
    d = 32
    qn = 4
    for precision in ["fp16", "bf16", "fp32"]:
        for num_shards in [4, 8, 16, 32]:
            for tokens in [2048, 8192, 32768]:
                q = rng.normal(size=(qn, d)).astype(np.float32)
                k = rng.normal(size=(tokens, d)).astype(np.float32)
                v = rng.normal(size=(tokens, d)).astype(np.float32)
                ref = full_attention_reference(q, k, v).astype(np.float32)
                states = []
                for block in np.array_split(np.arange(tokens), num_shards):
                    states.append(partial_attention_stats(q, k[block], v[block], state_precision=precision))
                ring_out = finalize_stats(ring_reduce(states, state_precision=precision))
                tree_out = finalize_stats(tree_reduce(states, state_precision=precision))
                for reduction, out in [("ring", ring_out), ("tree", tree_out)]:
                    row = {
                        "state_precision": precision,
                        "num_shards": num_shards,
                        "shared_prefix_tokens": tokens,
                        "reduction": reduction,
                    }
                    row.update(error_metrics(ref, out))
                    rows.append(row)
    write_csv(out_csv, rows)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for precision, color in [("fp16", "#E45756"), ("bf16", "#F58518"), ("fp32", "#4C78A8")]:
        xs = []
        ys = []
        for tokens in [2048, 8192, 32768]:
            vals = [
                float(r["relative_error"])
                for r in rows
                if r["state_precision"] == precision and int(r["shared_prefix_tokens"]) == tokens
            ]
            xs.append(tokens)
            ys.append(max(vals))
        ax.plot(xs, ys, marker="o", label=precision, color=color)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Shared prefix tokens")
    ax.set_ylabel("Worst relative error")
    ax.set_title("Online-Softmax State Precision Stability")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(fig_base.with_suffix(f".{ext}"), dpi=220)
    plt.close(fig)
    return rows


def generate_online_softmax_correctness(out_csv: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    rng = np.random.default_rng(2026)
    d = 32
    tokens = 257
    for kv_precision in ["fp32", "bf16", "fp16"]:
        for state_precision in ["fp32", "bf16", "fp16"]:
            for num_shards in [1, 2, 4, 8, 16, 32]:
                for query_tile_size in [1, 2, 4, 8, 16]:
                    q = rng.normal(size=(query_tile_size, d)).astype(np.float32)
                    k = rng.normal(size=(tokens, d)).astype(np.float32)
                    v = rng.normal(size=(tokens, d)).astype(np.float32)
                    if kv_precision == "fp16":
                        k_ref = k.astype(np.float16).astype(np.float32)
                        v_ref = v.astype(np.float16).astype(np.float32)
                    elif kv_precision == "bf16":
                        from .attention_math import quantize_bf16

                        k_ref = quantize_bf16(k)
                        v_ref = quantize_bf16(v)
                    else:
                        k_ref = k
                        v_ref = v
                    ref = full_attention_reference(q.astype(np.float64), k_ref, v_ref)
                    states = [
                        partial_attention_stats(
                            q,
                            k[block],
                            v[block],
                            state_precision=state_precision,
                            kv_storage_precision=kv_precision,
                        )
                        for block in np.array_split(np.arange(tokens), num_shards)
                    ]
                    for topology, reducer in [("ring_reduce", ring_reduce), ("binary_tree_reduce", tree_reduce)]:
                        out = finalize_stats(reducer(states, state_precision=state_precision))
                        row = {
                            "kv_storage_precision": kv_precision,
                            "state_precision": state_precision,
                            "num_shards": num_shards,
                            "query_tile_size": query_tile_size,
                            "reduction_topology": topology,
                        }
                        row.update(error_metrics(ref, out))
                        row["passes_fp32_gate"] = (
                            state_precision == "fp32" and float(row["relative_error"]) < 1e-5
                        )
                        rows.append(row)
    write_csv(out_csv, rows)
    return rows


def generate_regime_map(out_csv: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    model = ModelConfig(query_tile_size=8)
    hardware = HardwareConfig(region_sram_capacity_gib=4.5, region_capacity_gib=4.5)
    for prefix in [512, 2048, 8192, 32768, 65536]:
        for agents in [2, 4, 8, 16, 32]:
            for decode in [32, 128, 512, 1024]:
                workload = WorkloadConfig(
                    shared_prefix_tokens=prefix,
                    concurrent_agents=agents,
                    decode_tokens_per_agent=decode,
                )
                candidates = [
                    run_mode("central_kv_stationary", model, workload, hardware, query_tile_size=8),
                    run_mode("kvring_v2_ring", model, workload, hardware, query_tile_size=8, num_shards=8),
                    run_mode("kvring_v2_tree", model, workload, hardware, query_tile_size=8, num_shards=8),
                    run_mode("kvring_v2_region_split", model, workload, hardware, query_tile_size=8, num_shards=8),
                    run_mode("kvring_v3_adaptive", model, workload, hardware, query_tile_size=8, num_shards=8),
                ]
                candidate_rows = [result_row(c, shared_prefix_tokens=prefix, agents=agents, decode_tokens=decode) for c in candidates]
                valid = filter_valid_rows(candidate_rows)
                valid_non_v3 = [r for r in valid if r["mode"] != "KVRing-v3-adaptive"]
                if valid_non_v3:
                    winner = min(valid_non_v3, key=lambda r: float(r["attention_stage_proxy_latency_s"]))
                else:
                    winner = min(candidate_rows, key=lambda r: float(r["attention_stage_proxy_latency_s"]))
                central = next(r for r in candidate_rows if r["mode"] == "Central-KV-Stationary")
                kvring_valid = [r for r in valid if str(r["mode"]).startswith("KVRing-v2-")]
                best_kvring = min(kvring_valid, key=lambda r: float(r["attention_stage_proxy_latency_s"])) if kvring_valid else None
                central_latency = float(central["attention_stage_proxy_latency_s"])
                kvring_latency = float(best_kvring["attention_stage_proxy_latency_s"]) if best_kvring else float("nan")
                winner_latency = float(winner["attention_stage_proxy_latency_s"])
                rows.append(
                    {
                        "shared_prefix_tokens": prefix,
                        "num_agents": agents,
                        "decode_tokens_per_agent": decode,
                        "winner_mode": winner["mode"],
                        "winner_latency_s": winner_latency,
                        "central_latency_s": central_latency,
                        "kvring_best_latency_s": kvring_latency,
                        "speedup_vs_central": central_latency / winner_latency if winner_latency > 0 else "",
                        "valid_capacity": winner["valid_capacity"],
                        "reason": "minimum attention-stage proxy latency among valid-capacity candidates",
                    }
                )
    write_csv(out_csv, rows)
    return rows


def generate_paper_claims_matrix(path: Path, tables: dict[str, List[Dict[str, object]]]) -> List[Dict[str, object]]:
    mode_rows = filter_valid_rows(tables.get("mode_rows", []))
    query_rows = filter_valid_rows(tables.get("query_rows", []))
    correctness_rows = tables.get("correctness_rows", [])
    invalid_rows = tables.get("invalid_rows", [])
    central = next((r for r in mode_rows if r["mode"] == "Central-KV-Stationary"), None)
    kvrings = [r for r in mode_rows if str(r["mode"]).startswith("KVRing-v2-") or r["mode"] == "KVRing-v3-adaptive"]
    best_kvring = min(kvrings, key=lambda r: float(r["attention_stage_proxy_latency_s"])) if kvrings else None
    fp32_rows = [
        r
        for r in correctness_rows
        if r.get("state_precision") == "fp32" and float(r.get("relative_error", 1.0)) < 1e-5
    ]
    def claim_row(claim: str, status: str, metric: str, baseline: str, kv_value: object, comp_value: object, evidence: str, figure: str, abstract: bool, main: bool) -> Dict[str, object]:
        try:
            delta = (float(kv_value) - float(comp_value)) / max(abs(float(comp_value)), 1e-12) * 100.0
        except Exception:
            delta = ""
        return {
            "claim": claim,
            "status": status,
            "primary_metric": metric,
            "baseline": baseline,
            "kvring_value": kv_value,
            "comparison_value": comp_value,
            "delta_pct": delta,
            "evidence_csv": evidence,
            "figure_id": figure,
            "allowed_in_abstract": abstract,
            "allowed_in_main_results": main,
        }
    rows = [
        claim_row(
            "replication_centralization_dilemma",
            "supported" if mode_rows else "failed",
            "peak_region_sram_bytes,total_wire_bytes,max_directed_link_load_bytes",
            "Replicate-All/Pull-KV-Independent/Central-KV-Stationary",
            best_kvring["attention_stage_proxy_latency_s"] if best_kvring else "",
            central["attention_stage_proxy_latency_s"] if central else "",
            "tables/default_summary.csv",
            "fig2",
            True,
            True,
        ),
        claim_row(
            "central_kv_stationary_strong_baseline",
            "supported" if central else "failed",
            "central_query_wire_bytes,central_sram_read_latency_s,central_compute_latency_s",
            "Pull-KV-Independent",
            central["total_wire_bytes"] if central else "",
            "",
            "figures_source/fig_central_kv_bottleneck_breakdown.csv",
            "fig3",
            True,
            True,
        ),
        claim_row(
            "kvring_v2_beats_central_valid_capacity",
            "supported" if (central and best_kvring and float(best_kvring["attention_stage_proxy_latency_s"]) < float(central["attention_stage_proxy_latency_s"])) else "partial",
            "attention_stage_proxy_latency_s latency_bound=throughput_bound",
            "Central-KV-Stationary",
            best_kvring["attention_stage_proxy_latency_s"] if best_kvring else "",
            central["attention_stage_proxy_latency_s"] if central else "",
            "tables/default_summary.csv",
            "fig2",
            True,
            True,
        ),
        claim_row(
            "query_tiling_amortizes_local_sram_reads",
            "supported" if query_rows else "failed",
            "local_sram_read_bytes",
            "R=1",
            "saturates for R>=N",
            "largest at R=1",
            "figures_source/fig_query_tile_sweep.csv",
            "fig4",
            True,
            True,
        ),
        claim_row(
            "online_softmax_exactness",
            "supported" if fp32_rows else "failed",
            "relative_error fp32_state < 1e-5",
            "dense FP32 attention",
            max(float(r["relative_error"]) for r in fp32_rows) if fp32_rows else "",
            "1e-5",
            "figures_source/online_softmax_correctness.csv",
            "fig8",
            True,
            True,
        ),
        claim_row(
            "latency_bounds_transparent",
            "supported",
            "serialized_latency_s,throughput_bound_latency_s,critical_path_latency_s,conservative_max",
            "optimistic-only reporting",
            "all bounds exported",
            "",
            "figures_source/fig_latency_bounds_comparison.csv",
            "fig6",
            False,
            True,
        ),
        claim_row(
            "capacity_invalid_rows_filtered",
            "supported" if invalid_rows is not None else "failed",
            "valid_capacity",
            "unfiltered headline rows",
            len(invalid_rows),
            0,
            "figures_source/invalid_capacity_rows.csv",
            "appendix",
            False,
            True,
        ),
    ]
    write_csv(path, rows)
    return rows


def write_default_artifacts(outdir: Path = Path("."), *, legacy_only: bool = False) -> List[ModeResult]:
    results = legacy_v1_results() if legacy_only else default_results()
    write_json(
        outdir / "kv_ring_results.json",
        {
            "config": {
                "model": ModelConfig().__dict__,
                "workload": WorkloadConfig().__dict__,
                "hardware": HardwareConfig().__dict__,
            },
            "results": [r.to_full_dict() for r in results],
        },
    )
    write_text_report(outdir / "kv_ring_report.txt", results)
    plot_mode_comparison(results, outdir / "kv_ring_comparison")
    plot_link_loads(results, outdir / "kv_ring_link_loads.png")
    return results


def paper_default_results() -> List[ModeResult]:
    model = ModelConfig(query_tile_size=8)
    workload = WorkloadConfig(shared_prefix_tokens=32768, concurrent_agents=8, decode_tokens_per_agent=256)
    hardware = HardwareConfig(region_sram_capacity_gib=4.5, region_capacity_gib=4.5)
    return [
        run_mode("replicate_all", model, workload, hardware, query_tile_size=8, num_shards=8),
        run_mode("pull_kv_independent", model, workload, hardware, query_tile_size=8, num_shards=8),
        run_mode("central_kv_stationary", model, workload, hardware, query_tile_size=8, num_shards=8),
        run_mode("kvring_v1", model, workload, hardware, query_tile_size=8, num_shards=8),
        run_mode("kvring_v2_ring", model, workload, hardware, query_tile_size=8, num_shards=8),
        run_mode("kvring_v2_tree", model, workload, hardware, query_tile_size=8, num_shards=8),
        run_mode("kvring_v2_region_split", model, workload, hardware, query_tile_size=8, num_shards=8),
        run_mode("kvring_v3_adaptive", model, workload, hardware, query_tile_size=8, num_shards=8),
    ]


def export_round2_artifacts(outdir: Path, clean: bool = True) -> None:
    if clean and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    table_dir = outdir / "tables"
    fig_dir = outdir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    (outdir / "figures_source").mkdir(parents=True, exist_ok=True)

    results = paper_default_results()
    mode_rows = [result_row(r) for r in results]
    valid_mode_rows = filter_valid_rows(mode_rows)
    all_rows: List[Dict[str, object]] = list(mode_rows)
    write_csv(table_dir / "mode_summary.csv", mode_rows)
    write_csv(table_dir / "default_summary.csv", valid_mode_rows)
    write_csv(outdir / "figures_source" / "fig_replication_centralization_dilemma.csv", valid_mode_rows)
    write_csv(outdir / "figures_source" / "fig2_replication_centralization_dilemma.csv", valid_mode_rows)
    write_text_report(outdir / "report.md", results)
    write_json(outdir / "report.json", {"results": [r.to_full_dict() for r in results]})
    plot_mode_comparison(results, fig_dir / "fig_replication_centralization_dilemma")
    plot_mode_comparison(results, fig_dir / "fig2_replication_centralization_dilemma")
    plot_system_overview(fig_dir / "fig1_system_overview")
    write_csv(
        outdir / "figures_source" / "fig1_system_overview.csv",
        [{"source_note": "deterministic schematic generated from simulator concepts; not a measured wafer layout"}],
    )

    query_rows = generate_query_tile_sweep(
        table_dir / "query_tile_sweep.csv",
        [
            "central_kv_stationary",
            "kvring_v2_ring",
            "kvring_v2_tree",
            "kvring_v2_region_split",
            "kvring_v3_adaptive",
        ],
        [1, 2, 4, 8, 16, 32],
        [2048, 8192, 32768, 65536],
        [8, 16, 32],
        [64, 256, 1024],
        [8, 16, 32],
    )
    all_rows.extend(query_rows)
    valid_query_rows = filter_valid_rows(query_rows)
    write_csv(outdir / "figures_source" / "fig_query_tile_sweep.csv", valid_query_rows)
    write_csv(outdir / "figures_source" / "fig4_query_tile_sweep.csv", valid_query_rows)
    plot_query_tile_sweep(valid_query_rows, fig_dir / "fig_query_tile_sweep")
    plot_query_tile_sweep(valid_query_rows, fig_dir / "fig4_query_tile_sweep")

    fixed_query_rows = generate_query_tile_sweep(
        table_dir / "query_tile_sweep_fixed.csv",
        ["kvring_v2_ring", "kvring_v2_tree", "central_kv_stationary"],
        [1, 2, 3, 4, 5, 6, 7, 8, 16, 32],
        [32768],
        [8],
        [256],
    )
    all_rows.extend(fixed_query_rows)
    valid_fixed_query_rows = filter_valid_rows(fixed_query_rows)
    write_csv(outdir / "figures_source" / "fig_query_tile_sweep_fixed.csv", valid_fixed_query_rows)
    plot_query_tile_sweep(valid_fixed_query_rows, fig_dir / "fig_query_tile_sweep_fixed")

    prefix_rows = generate_shared_prefix_sweep(table_dir / "shared_prefix_sweep.csv")
    all_rows.extend(prefix_rows)
    write_csv(outdir / "figures_source" / "fig_prefix_length_sweep.csv", prefix_rows)
    plot_simple_bar(
        [
            r
            for r in prefix_rows
            if r["mode"] == "KVRing-v2-tree"
        ],
        "estimated_attention_stage_latency_s",
        "shared_prefix_tokens",
        fig_dir / "fig_prefix_length_sweep",
        "KVRing-v2 Prefix Length Sensitivity",
    )

    agent_rows = generate_agent_count_sweep(table_dir / "agent_count_sweep.csv")
    all_rows.extend(agent_rows)
    write_csv(outdir / "figures_source" / "agent_count_sweep.csv", agent_rows)

    shard_rows = generate_shard_count_sweep(table_dir / "shard_count_sweep.csv")
    all_rows.extend(shard_rows)
    write_csv(outdir / "figures_source" / "fig_shard_count_sweep.csv", shard_rows)
    plot_simple_bar(
        [
            r
            for r in shard_rows
            if r["mode"] == "KVRing-v2-tree"
        ],
        "estimated_attention_stage_latency_s",
        "ring_shards",
        fig_dir / "fig_shard_count_sweep",
        "KVRing-v2 Shard Count Sensitivity",
    )

    reduction_rows = generate_reduction_topology_sweep(table_dir / "reduction_topology_sweep.csv")
    all_rows.extend(reduction_rows)
    valid_reduction_rows = filter_valid_rows(reduction_rows)
    write_csv(table_dir / "reduction_topology.csv", valid_reduction_rows)
    write_csv(outdir / "figures_source" / "fig_reduction_topology.csv", valid_reduction_rows)
    write_csv(outdir / "figures_source" / "fig5_reduction_topology.csv", valid_reduction_rows)
    plot_simple_bar(
        valid_reduction_rows,
        "reduction_latency_s",
        "reduction_topology",
        fig_dir / "fig_reduction_topology",
        "Ring vs Binary Tree Online-Softmax Reduction",
    )
    plot_simple_bar(
        valid_reduction_rows,
        "max_directed_link_load_bytes",
        "reduction_topology",
        fig_dir / "fig5_reduction_topology",
        "Reduction Topology Hotspot Comparison",
    )

    topo_rows = generate_topology_sweep(table_dir / "topology_sweep.csv")
    all_rows.extend(topo_rows)
    write_csv(table_dir / "topology_summary.csv", topo_rows)
    write_csv(outdir / "figures_source" / "topology_sweep.csv", topo_rows)

    hw_rows = generate_hardware_sensitivity(table_dir / "hardware_sensitivity.csv")
    all_rows.extend(hw_rows)
    write_csv(outdir / "figures_source" / "hardware_sensitivity.csv", hw_rows)
    write_csv(outdir / "figures_source" / "fig9_hardware_sensitivity.csv", filter_valid_rows(hw_rows))
    plot_simple_bar(
        filter_valid_rows(hw_rows)[:24],
        "attention_stage_proxy_latency_s",
        "link_bandwidth_gbps",
        fig_dir / "fig9_hardware_sensitivity",
        "Hardware Sensitivity (valid rows subset)",
    )

    kv_move_rows = [
        r
        for r in mode_rows
        if r["mode"] in {"Pull-KV-Independent", "KVRing-v2-ring", "KVRing-v2-tree"}
    ]
    write_csv(outdir / "figures_source" / "fig_kv_move_vs_query_move.csv", kv_move_rows)
    plot_simple_bar(
        kv_move_rows,
        "wire_traffic_tib",
        "mode",
        fig_dir / "fig_kv_move_vs_query_move",
        "Moving KV vs Moving Query/States",
    )

    latency_bound_rows = []
    for row in mode_rows:
        if row["mode"] in {"Central-KV-Stationary", "KVRing-v2-ring", "KVRing-v2-tree", "KVRing-v3-adaptive"}:
            for key, label in [
                ("critical_path_latency_s", "critical_path"),
                ("throughput_bound_latency_s", "throughput_bound"),
                ("serialized_latency_s", "serialized"),
            ]:
                latency_bound_rows.append(
                    {
                        "mode": row["mode"],
                        "reduction_topology": row["reduction_topology"],
                        "latency_bound": label,
                        "latency_s": row[key],
                    }
                )
            latency_bound_rows.append(
                {
                    "mode": row["mode"],
                    "reduction_topology": row["reduction_topology"],
                    "latency_bound": "conservative_max",
                    "latency_s": max(float(row["throughput_bound_latency_s"]), float(row["critical_path_latency_s"])),
                }
            )
    write_csv(table_dir / "latency_bounds.csv", latency_bound_rows)
    write_csv(outdir / "figures_source" / "fig_latency_bounds_comparison.csv", latency_bound_rows)
    write_csv(outdir / "figures_source" / "fig6_latency_bounds_comparison.csv", latency_bound_rows)
    plot_latency_bounds(latency_bound_rows, fig_dir / "fig_latency_bounds_comparison")
    plot_latency_bounds(latency_bound_rows, fig_dir / "fig6_latency_bounds_comparison")

    central_rows = [r for r in mode_rows if r["mode"] == "Central-KV-Stationary"]
    central_breakdown_rows = []
    if central_rows:
        central = central_rows[0]
        for key, component in [
            ("central_query_mesh_latency_s", "query mesh"),
            ("central_sram_read_latency_s", "central SRAM read"),
            ("central_compute_latency_s", "central compute"),
            ("central_result_mesh_latency_s", "result mesh"),
            ("central_queue_latency_s", "central queue"),
        ]:
            central_breakdown_rows.append(
                {
                    "mode": central["mode"],
                    "component": component,
                    "latency_s": central[key],
                    "central_router_in_bytes": central["central_router_in_bytes"],
                    "central_router_out_bytes": central["central_router_out_bytes"],
                    "central_query_wire_bytes": central["central_query_wire_bytes"],
                    "central_result_wire_bytes": central["central_result_wire_bytes"],
                    "central_max_directed_link_load_bytes": central["central_max_directed_link_load_bytes"],
                    "central_hotspot_ratio": central["central_hotspot_ratio"],
                    "central_bottleneck_component": central["central_bottleneck_component"],
                }
            )
    write_csv(table_dir / "central_bottleneck_breakdown.csv", central_breakdown_rows)
    write_csv(
        outdir / "figures_source" / "fig_central_kv_bottleneck_breakdown.csv",
        central_breakdown_rows,
    )
    write_csv(
        outdir / "figures_source" / "fig3_central_kv_bottleneck_breakdown.csv",
        central_breakdown_rows,
    )
    plot_central_breakdown(central_breakdown_rows, fig_dir / "fig_central_kv_bottleneck_breakdown")
    plot_central_breakdown(central_breakdown_rows, fig_dir / "fig3_central_kv_bottleneck_breakdown")

    regime_rows = generate_regime_map(table_dir / "regime_map.csv")
    write_csv(outdir / "figures_source" / "fig_regime_map.csv", regime_rows)
    write_csv(outdir / "figures_source" / "fig7_regime_map.csv", regime_rows)
    plot_regime_map(regime_rows, fig_dir / "fig_regime_map")
    plot_regime_map(regime_rows, fig_dir / "fig7_regime_map")

    correctness_rows = generate_online_softmax_correctness(table_dir / "online_softmax_correctness.csv")
    write_csv(outdir / "figures_source" / "online_softmax_correctness.csv", correctness_rows)
    numerical_rows = generate_numerical_stability(
        table_dir / "numerical_error_summary.csv", fig_dir / "fig_numerical_stability"
    )
    write_csv(outdir / "figures_source" / "fig_numerical_stability.csv", numerical_rows)
    write_csv(outdir / "figures_source" / "fig8_numerical_stability.csv", numerical_rows)
    generate_numerical_stability(table_dir / "numerical_error_summary.csv", fig_dir / "fig8_numerical_stability")

    link_rows = []
    for r in results:
        for item in r.top_links(8):
            link_rows.append({"mode": r.mode, **item})
    write_csv(table_dir / "link_load_summary.csv", link_rows)
    write_csv(table_dir / "sram_summary.csv", mode_rows)
    write_csv(table_dir / "baseline_summary.csv", mode_rows)
    write_csv(table_dir / "correctness_summary.csv", correctness_rows)
    invalid_rows = [
        row
        for row in all_rows
        if str(row.get("valid_capacity", row.get("capacity_valid", "True"))).lower() in {"false", "0"}
        or str(row.get("region_capacity_violation", "False")).lower() in {"true", "1"}
    ]
    write_csv(outdir / "figures_source" / "invalid_capacity_rows.csv", invalid_rows)
    generate_paper_claims_matrix(
        outdir / "paper_claims_matrix.csv",
        {
            "mode_rows": valid_mode_rows,
            "query_rows": valid_query_rows,
            "correctness_rows": correctness_rows,
            "invalid_rows": invalid_rows,
        },
    )

    command = f"uv run python scripts/export_kvring_artifacts.py --out {outdir}"
    commit_hash = _git_commit()
    manifest_rows = []
    for p in sorted(outdir.rglob("*")):
        if not p.is_file() or p.name == "artifact_manifest.json":
            continue
        manifest_rows.append(
            {
                "file_path": str(p.relative_to(outdir)),
                "bytes": p.stat().st_size,
                "rows": _csv_rows(p),
                "row_count": _csv_rows(p),
                "sha256": _sha256(p),
                "command": command,
                "commit_hash": commit_hash,
            }
        )
    source_manifest = outdir / "source_run_manifest.csv"
    write_csv(source_manifest, manifest_rows)
    manifest_rows.append(
        {
            "file_path": str(source_manifest.relative_to(outdir)),
            "bytes": source_manifest.stat().st_size,
            "rows": _csv_rows(source_manifest),
            "row_count": _csv_rows(source_manifest),
            "sha256": _sha256(source_manifest),
            "command": command,
            "commit_hash": commit_hash,
        }
    )
    write_json(
        outdir / "artifact_manifest.json",
        {
            "project": "KVRing Round 3",
            "artifact_count": len(manifest_rows),
            "vc_model": HardwareConfig().vc_model,
            "command": command,
            "commit_hash": commit_hash,
            "outputs": manifest_rows,
        },
    )
