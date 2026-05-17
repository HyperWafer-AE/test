from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class MeshConfig:
    rows: int
    cols: int
    tile_sram_mb: float
    tile_prefill_tflops: float
    tile_decode_tflops: float
    link_bandwidth_GBps: float
    link_latency_us: float
    multicast_supported: bool
    energy_per_flop_pJ: float
    energy_per_byte_pJ: float
    h100_prefill_to_wafer_scale: float = 0.35
    h100_decode_to_wafer_scale: float = 0.50
    prefill_tile_fraction: float = 0.50
    decode_tile_fraction: float = 0.50

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MeshConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    @property
    def total_tiles(self) -> int:
        return self.rows * self.cols

    @property
    def tile_sram_bytes(self) -> int:
        return int(self.tile_sram_mb * 1024 * 1024)

    @property
    def total_sram_bytes(self) -> int:
        return self.tile_sram_bytes * self.total_tiles


class Mesh:
    def __init__(self, config: MeshConfig):
        self.config = config
        self.link_loads: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}

    def manhattan_distance(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def clamp_tile(self, pos: tuple[int, int]) -> tuple[int, int]:
        return (
            max(0, min(self.config.rows - 1, int(pos[0]))),
            max(0, min(self.config.cols - 1, int(pos[1]))),
        )

    def route(self, src: tuple[int, int], dst: tuple[int, int], bytes_moved: float) -> float:
        src = self.clamp_tile(src)
        dst = self.clamp_tile(dst)
        r, c = src
        dr = 1 if dst[0] >= r else -1
        while r != dst[0]:
            nxt = (r + dr, c)
            self._add_link((r, c), nxt, bytes_moved)
            r += dr
        dc = 1 if dst[1] >= c else -1
        while c != dst[1]:
            nxt = (r, c + dc)
            self._add_link((r, c), nxt, bytes_moved)
            c += dc
        return self.comm_time_ms(src, dst, bytes_moved)

    def multicast(self, src: tuple[int, int], dsts: list[tuple[int, int]], bytes_moved: float) -> float:
        if not dsts:
            return 0.0
        if self.config.multicast_supported:
            # Approximate tree cost with the farthest distance and one payload copy.
            farthest = max(self.manhattan_distance(src, d) for d in dsts)
            for dst in dsts:
                self.route(src, dst, bytes_moved / max(1, len(dsts)))
            return self._bandwidth_time_ms(bytes_moved) + farthest * self.config.link_latency_us / 1000.0
        return sum(self.route(src, dst, bytes_moved) for dst in dsts)

    def _add_link(self, a: tuple[int, int], b: tuple[int, int], bytes_moved: float) -> None:
        key = tuple(sorted([a, b]))  # type: ignore[arg-type]
        self.link_loads[key] = self.link_loads.get(key, 0.0) + float(bytes_moved)

    def _bandwidth_time_ms(self, bytes_moved: float) -> float:
        bytes_per_ms = self.config.link_bandwidth_GBps * 1e9 / 1000.0
        return float(bytes_moved) / bytes_per_ms if bytes_per_ms else 0.0

    def comm_time_ms(self, src: tuple[int, int], dst: tuple[int, int], bytes_moved: float) -> float:
        distance = self.manhattan_distance(src, dst)
        return self._bandwidth_time_ms(bytes_moved) + distance * self.config.link_latency_us / 1000.0

    def stats(self) -> dict[str, float]:
        loads = list(self.link_loads.values())
        total = sum(loads)
        avg = total / len(loads) if loads else 0.0
        max_load = max(loads) if loads else 0.0
        return {
            "mesh_total_traffic_bytes": total,
            "mesh_avg_link_load_bytes": avg,
            "mesh_max_link_load_bytes": max_load,
            "mesh_hotspot_ratio": max_load / avg if avg else 1.0,
        }
