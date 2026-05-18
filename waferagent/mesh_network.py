from __future__ import annotations

from dataclasses import dataclass

from waferagent.mesh import MeshConfig


@dataclass
class MeshLinkEvent:
    baseline: str
    job_id: str
    stage_id: str
    src: str
    dst: str
    bytes: int
    wait_ms: float
    transfer_ms: float
    start_ms: float
    end_ms: float

    def to_dict(self) -> dict[str, str | int | float]:
        return self.__dict__.copy()


class MeshNetwork:
    def __init__(self, cfg: MeshConfig, baseline: str, congestion_enabled: bool = True):
        self.cfg = cfg
        self.baseline = baseline
        self.congestion_enabled = congestion_enabled
        self.link_available: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}
        self.link_loads: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}
        self.events: list[MeshLinkEvent] = []

    def route(
        self,
        job_id: str,
        stage_id: str,
        src: tuple[int, int],
        dst: tuple[int, int],
        bytes_moved: int,
        ready_ms: float,
    ) -> tuple[float, float, int]:
        bytes_moved = int(max(0, bytes_moved))
        if src == dst or bytes_moved == 0:
            return 0.0, 0.0, 0
        path = self._xy_path(src, dst)
        bytes_per_ms = max(1e-9, self.cfg.link_bandwidth_GBps * 1e9 / 1000.0)
        per_link_transfer = bytes_moved / bytes_per_ms + self.cfg.link_latency_us / 1000.0
        t = ready_ms
        total_wait = 0.0
        for a, b in path:
            key = tuple(sorted([a, b]))  # type: ignore[arg-type]
            available = self.link_available.get(key, 0.0)
            wait = max(0.0, available - t) if self.congestion_enabled else 0.0
            start = t + wait
            end = start + per_link_transfer
            self.link_available[key] = end
            self.link_loads[key] = self.link_loads.get(key, 0.0) + bytes_moved
            self.events.append(
                MeshLinkEvent(
                    baseline=self.baseline,
                    job_id=job_id,
                    stage_id=stage_id,
                    src=f"{a[0]},{a[1]}",
                    dst=f"{b[0]},{b[1]}",
                    bytes=bytes_moved,
                    wait_ms=wait,
                    transfer_ms=per_link_transfer,
                    start_ms=start,
                    end_ms=end,
                )
            )
            total_wait += wait
            t = end
        return total_wait, max(0.0, t - ready_ms), bytes_moved * len(path)

    def multicast(
        self,
        job_id: str,
        stage_id: str,
        src: tuple[int, int],
        dsts: list[tuple[int, int]],
        bytes_moved: int,
        ready_ms: float,
    ) -> tuple[float, float, int]:
        if not dsts:
            return 0.0, 0.0, 0
        if not self.cfg.multicast_supported:
            waits = times = traffic = 0.0
            for dst in dsts:
                w, t, b = self.route(job_id, stage_id, src, dst, bytes_moved, ready_ms)
                waits += w
                times = max(times, t)
                traffic += b
            return waits, times, int(traffic)
        # Approximate multicast tree by routing once to each unique destination, not by dividing bytes.
        waits = 0.0
        end_delta = 0.0
        traffic = 0
        for dst in sorted(set(dsts)):
            w, t, b = self.route(job_id, stage_id, src, dst, bytes_moved, ready_ms)
            waits += w
            end_delta = max(end_delta, t)
            traffic += b
        return waits, end_delta, traffic

    def _xy_path(self, src: tuple[int, int], dst: tuple[int, int]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        r, c = src
        path = []
        step = 1 if dst[0] >= r else -1
        while r != dst[0]:
            nxt = (r + step, c)
            path.append(((r, c), nxt))
            r += step
        step = 1 if dst[1] >= c else -1
        while c != dst[1]:
            nxt = (r, c + step)
            path.append(((r, c), nxt))
            c += step
        return path

    def stats(self) -> dict[str, float]:
        loads = list(self.link_loads.values())
        total = sum(loads)
        avg = total / len(loads) if loads else 0.0
        max_load = max(loads) if loads else 0.0
        wait = sum(ev.wait_ms for ev in self.events)
        transfer = sum(ev.transfer_ms for ev in self.events)
        return {
            "mesh_total_traffic_bytes": total,
            "mesh_avg_link_load_bytes": avg,
            "mesh_max_link_load_bytes": max_load,
            "mesh_hotspot_ratio": max_load / avg if avg else 1.0,
            "mesh_wait_ms": wait,
            "communication_time_ms": wait + transfer,
        }
