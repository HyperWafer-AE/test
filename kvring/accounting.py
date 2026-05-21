"""Traffic and mode result accounting."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict, Dict, List, Mapping, Optional

import numpy as np

from .config import Coord, Edge
from .units import gib, tib


def edge_key(edge: Edge) -> str:
    (r0, c0), (r1, c1) = edge
    return f"({r0},{c0})->({r1},{c1})"


@dataclass
class TraceStats:
    payload_bytes: float = 0.0
    link_loads: DefaultDict[Edge, float] = field(default_factory=lambda: defaultdict(float))

    def add_payload(self, bytes_: float) -> None:
        self.payload_bytes += bytes_

    @property
    def total_wire_bytes(self) -> float:
        return float(sum(self.link_loads.values()))

    @property
    def active_link_loads(self) -> List[float]:
        return [v for v in self.link_loads.values() if v > 0.0]

    @property
    def max_link(self) -> Optional[Edge]:
        if not self.active_link_loads:
            return None
        return max(self.link_loads.items(), key=lambda kv: kv[1])[0]

    @property
    def max_link_load_bytes(self) -> float:
        edge = self.max_link
        return 0.0 if edge is None else float(self.link_loads[edge])

    @property
    def mean_active_link_load_bytes(self) -> float:
        active = self.active_link_loads
        return float(np.mean(active)) if active else 0.0

    @property
    def hotspot_ratio(self) -> float:
        mean = self.mean_active_link_load_bytes
        return self.max_link_load_bytes / mean if mean > 0.0 else 0.0


@dataclass
class ModeResult:
    mode: str
    description: str
    total_sram_bytes: float
    peak_region_sram_bytes: float
    payload_bytes: float
    total_wire_bytes: float
    max_link_load_bytes: float
    mean_active_link_load_bytes: float
    hotspot_ratio: float
    mesh_seconds: float
    compute_seconds: float
    estimated_latency_seconds: float
    estimated_cycles: int
    sram_port_bytes: float
    network_cycles: int
    compute_cycles: int
    link_loads: Mapping[Edge, float] = field(repr=False)
    region_sram_bytes: Mapping[Coord, float] = field(repr=False)
    max_link: Optional[Edge] = None
    extra: Dict[str, object] = field(default_factory=dict)

    def top_links(self, n: int = 10) -> List[Dict[str, object]]:
        items = sorted(self.link_loads.items(), key=lambda kv: kv[1], reverse=True)
        return [{"edge": edge_key(edge), "bytes": load, "gib": gib(load)} for edge, load in items[:n]]

    def to_summary_dict(self) -> Dict[str, object]:
        return {
            "mode": self.mode,
            "total_sram_gib": gib(self.total_sram_bytes),
            "peak_region_sram_gib": gib(self.peak_region_sram_bytes),
            "payload_gib": gib(self.payload_bytes),
            "total_wire_tib": tib(self.total_wire_bytes),
            "max_link_load_gib": gib(self.max_link_load_bytes),
            "hotspot_ratio_max_over_mean_active": self.hotspot_ratio,
            "mesh_time_s": self.mesh_seconds,
            "compute_time_s": self.compute_seconds,
            "estimated_latency_s": self.estimated_latency_seconds,
            "estimated_cycles": self.estimated_cycles,
            "sram_port_gib": gib(self.sram_port_bytes),
            "network_cycles": self.network_cycles,
            "compute_cycles": self.compute_cycles,
            "max_directed_link": edge_key(self.max_link) if self.max_link else None,
            "extra": self.extra,
        }

    def to_full_dict(self) -> Dict[str, object]:
        data = self.to_summary_dict()
        data["description"] = self.description
        data["top_links"] = self.top_links(16)
        data["region_sram_gib"] = {
            f"({r},{c})": gib(v)
            for (r, c), v in sorted(self.region_sram_bytes.items(), key=lambda kv: kv[0])
            if v > 0.0
        }
        data["link_load_gib"] = {
            edge_key(edge): gib(load)
            for edge, load in sorted(self.link_loads.items(), key=lambda kv: edge_key(kv[0]))
            if load > 0.0
        }
        return data


def result_from_stats(
    *,
    mode: str,
    description: str,
    stats: TraceStats,
    region_sram: Mapping[Coord, float],
    sram_port_bytes: float,
    propagation_hops: int,
    link_bytes_per_cycle: float,
    sram_bytes_per_cycle: float,
    clock_hz: float,
    hop_latency_cycles: int,
    network_cycles_override: int | None = None,
    compute_cycles_override: int | None = None,
    extra: Dict[str, object] | None = None,
) -> ModeResult:
    if network_cycles_override is None:
        network_cycles = int(np.ceil(stats.max_link_load_bytes / link_bytes_per_cycle))
        network_cycles += propagation_hops * hop_latency_cycles
    else:
        network_cycles = int(network_cycles_override)
    if compute_cycles_override is None:
        compute_cycles = int(np.ceil(sram_port_bytes / sram_bytes_per_cycle))
    else:
        compute_cycles = int(compute_cycles_override)
    total_cycles = network_cycles + compute_cycles
    return ModeResult(
        mode=mode,
        description=description,
        total_sram_bytes=float(sum(region_sram.values())),
        peak_region_sram_bytes=float(max(region_sram.values()) if region_sram else 0.0),
        payload_bytes=stats.payload_bytes,
        total_wire_bytes=stats.total_wire_bytes,
        max_link_load_bytes=stats.max_link_load_bytes,
        mean_active_link_load_bytes=stats.mean_active_link_load_bytes,
        hotspot_ratio=stats.hotspot_ratio,
        mesh_seconds=network_cycles / clock_hz,
        compute_seconds=compute_cycles / clock_hz,
        estimated_latency_seconds=total_cycles / clock_hz,
        estimated_cycles=total_cycles,
        sram_port_bytes=sram_port_bytes,
        network_cycles=network_cycles,
        compute_cycles=compute_cycles,
        link_loads=dict(stats.link_loads),
        region_sram_bytes=dict(region_sram),
        max_link=stats.max_link,
        extra=extra or {},
    )
