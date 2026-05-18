from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from waferagent.mesh import MeshConfig


@dataclass
class ResourcePool:
    name: str
    tile_count: int
    available_heap: list[float] = field(default_factory=list)
    busy_tile_ms: float = 0.0
    idle_tile_ms: float = 0.0

    def __post_init__(self) -> None:
        self.tile_count = max(1, int(self.tile_count))
        if not self.available_heap:
            self.available_heap = [0.0 for _ in range(self.tile_count)]
            heapq.heapify(self.available_heap)

    def reserve(self, ready_ms: float, duration_ms: float, requested_tiles: int) -> tuple[float, float, int]:
        k = max(1, min(int(requested_tiles), self.tile_count))
        popped = [heapq.heappop(self.available_heap) for _ in range(k)]
        start = max(ready_ms, max(popped))
        effective_duration = duration_ms / max(1.0, k ** 0.72)
        end = start + effective_duration
        for _ in popped:
            heapq.heappush(self.available_heap, end)
        self.busy_tile_ms += effective_duration * k
        return start, end, k

    def utilization(self, makespan_ms: float) -> float:
        denom = max(1e-9, makespan_ms * self.tile_count)
        return min(1.0, self.busy_tile_ms / denom)


@dataclass
class ResourceModel:
    mesh_config: MeshConfig
    prefill_pool: ResourcePool
    decode_pool: ResourcePool

    @classmethod
    def from_config(cls, cfg: MeshConfig, dynamic_pd_partition: bool = False, prefill_pressure: float = 0.5) -> "ResourceModel":
        total = cfg.total_tiles
        if dynamic_pd_partition:
            frac = min(0.8, max(0.2, prefill_pressure))
        else:
            frac = cfg.prefill_tile_fraction
        prefill_tiles = max(1, int(total * frac))
        decode_tiles = max(1, total - prefill_tiles)
        return cls(cfg, ResourcePool("prefill", prefill_tiles), ResourcePool("decode", decode_tiles))

    def reserve_stage(self, pool: str, ready_ms: float, duration_ms: float, requested_tiles: int) -> tuple[float, float, int]:
        if pool == "prefill":
            return self.prefill_pool.reserve(ready_ms, duration_ms, requested_tiles)
        if pool == "decode":
            return self.decode_pool.reserve(ready_ms, duration_ms, requested_tiles)
        return ready_ms, ready_ms + duration_ms, 0

    def stats(self, makespan_ms: float) -> dict[str, float]:
        busy = self.prefill_pool.busy_tile_ms + self.decode_pool.busy_tile_ms
        total = max(1e-9, makespan_ms * self.mesh_config.total_tiles)
        return {
            "prefill_tile_utilization": self.prefill_pool.utilization(makespan_ms),
            "decode_tile_utilization": self.decode_pool.utilization(makespan_ms),
            "tile_busy_time_ms": busy,
            "tile_idle_time_ms": max(0.0, total - busy),
            "pd_interference_ms": 0.0,
        }
