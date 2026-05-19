from __future__ import annotations

from dataclasses import dataclass, field

from waferagent.kv_model import PrefixBlock, ttl_priority
from waferagent.mesh import MeshConfig


@dataclass
class SRAMEvent:
    baseline: str
    job_id: str
    stage_id: str
    prefix_id: str
    event_type: str
    bytes: int
    used_bytes: int
    capacity_bytes: int
    region_id: str = "global"
    source_region: str = ""

    def to_dict(self) -> dict[str, str | int]:
        return self.__dict__.copy()


@dataclass
class SRAMRegion:
    region_id: str
    capacity_bytes: int
    resident_blocks: dict[str, PrefixBlock] = field(default_factory=dict)
    used_bytes: int = 0
    eviction_count: int = 0
    reload_bytes: int = 0
    spill_bytes: int = 0
    hits: int = 0
    misses: int = 0


class SRAMManager:
    def __init__(self, capacity_bytes: int, policy: str, baseline: str):
        self.region = SRAMRegion("global", max(1, int(capacity_bytes)))
        self.policy = policy
        self.baseline = baseline
        self.events: list[SRAMEvent] = []
        self.prefix_blocks: dict[str, PrefixBlock] = {}
        self.pinned: set[str] = set()
        self.ttl_eviction_avoided_count = 0
        self.ttl_pinned_bytes = 0

    def access(
        self,
        job_id: str,
        stage_id: str,
        prefix_id: str,
        token_len: int,
        kv_bytes: int,
        step: int,
        criticality: float,
        tool_resume_probability: float = 0.0,
        pin: bool = False,
        compute_store: bool = False,
    ) -> tuple[bool, int, int]:
        if not prefix_id or kv_bytes <= 0:
            return True, 0, 0
        block = self.prefix_blocks.get(prefix_id)
        if block is None:
            block = PrefixBlock(prefix_id, token_len, kv_bytes, [], 0, step, step, 0, criticality, tool_resume_probability)
            self.prefix_blocks[prefix_id] = block
        block.ref_count += 1
        block.last_use_step = step
        block.reuse_distance = max(0, block.last_use_step - block.first_use_step)
        block.criticality_score = max(block.criticality_score, criticality)
        block.tool_resume_probability = max(block.tool_resume_probability, tool_resume_probability)
        if pin:
            self.pinned.add(prefix_id)
            self.ttl_pinned_bytes += kv_bytes

        if prefix_id in self.region.resident_blocks:
            self.region.hits += 1
            self.events.append(self._event(job_id, stage_id, prefix_id, "hit", 0))
            return True, 0, 0

        self.region.misses += 1
        reload_bytes = 0 if compute_store else kv_bytes
        self.region.reload_bytes += reload_bytes
        evicted_bytes = self._ensure_capacity(job_id, stage_id, kv_bytes)
        if kv_bytes > self.region.capacity_bytes:
            spill = kv_bytes - self.region.capacity_bytes
            self.region.spill_bytes += spill
            self.events.append(self._event(job_id, stage_id, prefix_id, "spill", spill))
            return False, reload_bytes + spill, evicted_bytes
        self.region.resident_blocks[prefix_id] = block
        self.region.used_bytes += kv_bytes
        self.events.append(self._event(job_id, stage_id, prefix_id, "store" if compute_store else "reload", reload_bytes))
        return False, reload_bytes, evicted_bytes

    def _ensure_capacity(self, job_id: str, stage_id: str, needed: int) -> int:
        evicted = 0
        while self.region.used_bytes + needed > self.region.capacity_bytes and self.region.resident_blocks:
            candidates = list(self.region.resident_blocks.values())
            non_pinned = [b for b in candidates if b.prefix_id not in self.pinned]
            if not non_pinned:
                self.ttl_eviction_avoided_count += 1
                break
            victim = min(non_pinned, key=lambda b: ttl_priority(b, self.policy))
            self.region.resident_blocks.pop(victim.prefix_id, None)
            self.region.used_bytes = max(0, self.region.used_bytes - victim.kv_bytes)
            self.region.eviction_count += 1
            evicted += victim.kv_bytes
            self.events.append(self._event(job_id, stage_id, victim.prefix_id, "evict", victim.kv_bytes))
        return evicted

    def _event(self, job_id: str, stage_id: str, prefix_id: str, event_type: str, bytes_: int) -> SRAMEvent:
        return SRAMEvent(
            baseline=self.baseline,
            job_id=job_id,
            stage_id=stage_id,
            prefix_id=prefix_id,
            event_type=event_type,
            bytes=int(bytes_),
            used_bytes=int(self.region.used_bytes),
            capacity_bytes=int(self.region.capacity_bytes),
        )

    def stats(self) -> dict[str, float]:
        accesses = self.region.hits + self.region.misses
        reuse = sum(max(0, b.ref_count - 1) for b in self.prefix_blocks.values())
        return {
            "sram_evictions": float(self.region.eviction_count),
            "sram_reload_bytes": float(self.region.reload_bytes),
            "sram_spill_bytes": float(self.region.spill_bytes),
            "sram_hit_rate": self.region.hits / accesses if accesses else 0.0,
            "prefix_block_reuse_count": float(reuse),
            "ttl_pinned_bytes": float(self.ttl_pinned_bytes),
            "ttl_eviction_avoided_count": float(self.ttl_eviction_avoided_count),
        }

    def prefix_block_rows(self) -> list[dict[str, str | int | float]]:
        return [
            {
                "prefix_id": b.prefix_id,
                "token_len": b.token_len,
                "kv_bytes": b.kv_bytes,
                "resident_regions": "global" if b.prefix_id in self.region.resident_blocks else "",
                "owner_nodes": ",".join(b.owner_nodes),
                "ref_count": b.ref_count,
                "first_use_step": b.first_use_step,
                "last_use_step": b.last_use_step,
                "next_use_step": -1,
                "criticality_score": b.criticality_score,
                "tool_resume_probability": b.tool_resume_probability,
            }
            for b in self.prefix_blocks.values()
        ]


@dataclass
class DistributedSRAMAccess:
    hit: bool
    same_region_hit: bool
    cross_region_hit: bool
    reload_bytes: int
    evicted_bytes: int
    spill_bytes: int
    source_region: str
    source_tile: tuple[int, int] | None


class DistributedSRAMManager:
    def __init__(self, cfg: MeshConfig, policy: str, baseline: str):
        self.cfg = cfg
        self.policy = policy
        self.baseline = baseline
        self.region_rows = max(1, int(cfg.sram_region_rows))
        self.region_cols = max(1, int(cfg.sram_region_cols))
        self.regions: dict[str, SRAMRegion] = {}
        self.prefix_blocks: dict[str, PrefixBlock] = {}
        self.block_regions: dict[str, set[str]] = {}
        self.pinned: set[str] = set()
        self.events: list[SRAMEvent] = []
        self.ttl_eviction_avoided_count = 0
        self.ttl_pinned_bytes = 0

    def region_id_for_tile(self, tile: tuple[int, int]) -> str:
        rr = min(self.cfg.rows - 1, max(0, int(tile[0]))) // self.region_rows
        cc = min(self.cfg.cols - 1, max(0, int(tile[1]))) // self.region_cols
        return f"r{rr}c{cc}"

    def region_center_tile(self, region_id: str) -> tuple[int, int]:
        rr = int(region_id.split("c", 1)[0][1:])
        cc = int(region_id.split("c", 1)[1])
        return (
            min(self.cfg.rows - 1, rr * self.region_rows + self.region_rows // 2),
            min(self.cfg.cols - 1, cc * self.region_cols + self.region_cols // 2),
        )

    def _region(self, region_id: str) -> SRAMRegion:
        if region_id not in self.regions:
            tiles = self.region_rows * self.region_cols
            self.regions[region_id] = SRAMRegion(
                region_id=region_id,
                capacity_bytes=max(1, tiles * self.cfg.tile_sram_bytes),
            )
        return self.regions[region_id]

    def materialize(
        self,
        job_id: str,
        stage_id: str,
        prefix_id: str,
        token_len: int,
        kv_bytes: int,
        step: int,
        criticality: float,
        region_id: str,
        event_type: str = "planned_replica",
    ) -> tuple[int, int]:
        """Place a block in a specific SRAM region and charge capacity/evictions.

        This is used by shared-KV placement planning. It intentionally does not
        imply a compute hit; it only updates physical residency.
        """
        if not prefix_id or kv_bytes <= 0:
            return 0, 0
        region = self._region(region_id)
        block = self.prefix_blocks.get(prefix_id)
        if block is None:
            block = PrefixBlock(prefix_id, token_len, kv_bytes, [], 0, step, step, 0, criticality, 0.0)
            self.prefix_blocks[prefix_id] = block
        block.ref_count += 1
        block.last_use_step = step
        block.criticality_score = max(block.criticality_score, criticality)
        if prefix_id in region.resident_blocks:
            region.hits += 1
            self.events.append(self._event(job_id, stage_id, prefix_id, "planned_hit", 0, region_id))
            return 0, 0
        region.misses += 1
        evicted = self._ensure_capacity(region, job_id, stage_id, kv_bytes)
        spill = 0
        if kv_bytes > region.capacity_bytes:
            spill = kv_bytes - region.capacity_bytes
            region.spill_bytes += spill
            self.events.append(self._event(job_id, stage_id, prefix_id, "spill", spill, region_id))
            return evicted, spill
        region.resident_blocks[prefix_id] = block
        region.used_bytes += kv_bytes
        self.block_regions.setdefault(prefix_id, set()).add(region_id)
        self.events.append(self._event(job_id, stage_id, prefix_id, event_type, kv_bytes, region_id))
        return evicted, spill

    def access(
        self,
        job_id: str,
        stage_id: str,
        prefix_id: str,
        token_len: int,
        kv_bytes: int,
        step: int,
        criticality: float,
        tile: tuple[int, int],
        tool_resume_probability: float = 0.0,
        pin: bool = False,
        compute_store: bool = False,
        allow_cross_region_materialize: bool = True,
    ) -> DistributedSRAMAccess:
        if not prefix_id or kv_bytes <= 0:
            return DistributedSRAMAccess(True, True, False, 0, 0, 0, "", None)
        region_id = self.region_id_for_tile(tile)
        region = self._region(region_id)
        block = self.prefix_blocks.get(prefix_id)
        if block is None:
            block = PrefixBlock(
                prefix_id,
                token_len,
                kv_bytes,
                [],
                0,
                step,
                step,
                0,
                criticality,
                tool_resume_probability,
            )
            self.prefix_blocks[prefix_id] = block
        block.ref_count += 1
        block.last_use_step = step
        block.reuse_distance = max(0, block.last_use_step - block.first_use_step)
        block.criticality_score = max(block.criticality_score, criticality)
        block.tool_resume_probability = max(block.tool_resume_probability, tool_resume_probability)
        if pin:
            self.pinned.add(prefix_id)
            self.ttl_pinned_bytes += kv_bytes

        if prefix_id in region.resident_blocks:
            region.hits += 1
            self.events.append(self._event(job_id, stage_id, prefix_id, "hit", 0, region_id))
            return DistributedSRAMAccess(True, True, False, 0, 0, 0, region_id, tile)

        resident = sorted(self.block_regions.get(prefix_id, set()))
        if resident:
            src = resident[0]
            region.misses += 1
            if not allow_cross_region_materialize:
                self.events.append(self._event(job_id, stage_id, prefix_id, "remote_region_hit", kv_bytes, region_id, src))
                return DistributedSRAMAccess(
                    True,
                    False,
                    True,
                    0,
                    0,
                    0,
                    src,
                    self.region_center_tile(src),
                )
            evicted = self._ensure_capacity(region, job_id, stage_id, kv_bytes)
            spill = 0
            if kv_bytes > region.capacity_bytes:
                spill = kv_bytes - region.capacity_bytes
                region.spill_bytes += spill
                self.events.append(self._event(job_id, stage_id, prefix_id, "spill", spill, region_id, src))
            else:
                region.resident_blocks[prefix_id] = block
                region.used_bytes += kv_bytes
                self.block_regions.setdefault(prefix_id, set()).add(region_id)
                self.events.append(self._event(job_id, stage_id, prefix_id, "cross_region_hit", kv_bytes, region_id, src))
            return DistributedSRAMAccess(
                True,
                False,
                True,
                0,
                evicted,
                spill,
                src,
                self.region_center_tile(src),
            )

        region.misses += 1
        reload_bytes = 0 if compute_store else kv_bytes
        region.reload_bytes += reload_bytes
        evicted = self._ensure_capacity(region, job_id, stage_id, kv_bytes)
        spill = 0
        if kv_bytes > region.capacity_bytes:
            spill = kv_bytes - region.capacity_bytes
            region.spill_bytes += spill
            self.events.append(self._event(job_id, stage_id, prefix_id, "spill", spill, region_id))
            return DistributedSRAMAccess(False, False, False, reload_bytes + spill, evicted, spill, "", None)
        region.resident_blocks[prefix_id] = block
        region.used_bytes += kv_bytes
        self.block_regions.setdefault(prefix_id, set()).add(region_id)
        self.events.append(self._event(job_id, stage_id, prefix_id, "store" if compute_store else "reload", reload_bytes, region_id))
        return DistributedSRAMAccess(False, False, False, reload_bytes, evicted, spill, "", None)

    def _ensure_capacity(self, region: SRAMRegion, job_id: str, stage_id: str, needed: int) -> int:
        evicted = 0
        while region.used_bytes + needed > region.capacity_bytes and region.resident_blocks:
            candidates = list(region.resident_blocks.values())
            non_pinned = [b for b in candidates if b.prefix_id not in self.pinned]
            if not non_pinned:
                self.ttl_eviction_avoided_count += 1
                break
            victim = min(non_pinned, key=lambda b: ttl_priority(b, self.policy))
            region.resident_blocks.pop(victim.prefix_id, None)
            self.block_regions.get(victim.prefix_id, set()).discard(region.region_id)
            region.used_bytes = max(0, region.used_bytes - victim.kv_bytes)
            region.eviction_count += 1
            evicted += victim.kv_bytes
            self.events.append(self._event(job_id, stage_id, victim.prefix_id, "evict", victim.kv_bytes, region.region_id))
        return evicted

    def _event(
        self,
        job_id: str,
        stage_id: str,
        prefix_id: str,
        event_type: str,
        bytes_: int,
        region_id: str,
        source_region: str = "",
    ) -> SRAMEvent:
        region = self._region(region_id)
        return SRAMEvent(
            baseline=self.baseline,
            job_id=job_id,
            stage_id=stage_id,
            prefix_id=prefix_id,
            event_type=event_type,
            bytes=int(bytes_),
            used_bytes=int(region.used_bytes),
            capacity_bytes=int(region.capacity_bytes),
            region_id=region_id,
            source_region=source_region,
        )

    def stats(self) -> dict[str, float]:
        hits = sum(r.hits for r in self.regions.values())
        misses = sum(r.misses for r in self.regions.values())
        accesses = hits + misses
        reuse = sum(max(0, b.ref_count - 1) for b in self.prefix_blocks.values())
        return {
            "sram_evictions": float(sum(r.eviction_count for r in self.regions.values())),
            "sram_reload_bytes": float(sum(r.reload_bytes for r in self.regions.values())),
            "sram_spill_bytes": float(sum(r.spill_bytes for r in self.regions.values())),
            "sram_hit_rate": hits / accesses if accesses else 0.0,
            "prefix_block_reuse_count": float(reuse),
            "ttl_pinned_bytes": float(self.ttl_pinned_bytes),
            "ttl_eviction_avoided_count": float(self.ttl_eviction_avoided_count),
        }

    def prefix_block_rows(self) -> list[dict[str, str | int | float]]:
        rows = []
        for b in self.prefix_blocks.values():
            regions = sorted(self.block_regions.get(b.prefix_id, set()))
            rows.append(
                {
                    "prefix_id": b.prefix_id,
                    "token_len": b.token_len,
                    "kv_bytes": b.kv_bytes,
                    "home_region": regions[0] if regions else "",
                    "replica_regions": ",".join(regions[1:]),
                    "resident_regions": ",".join(regions),
                    "resident_tiles": "",
                    "owner_nodes": ",".join(b.owner_nodes),
                    "ref_count": b.ref_count,
                    "first_use_step": b.first_use_step,
                    "last_use_step": b.last_use_step,
                    "next_use_step": -1,
                    "criticality_score": b.criticality_score,
                    "tool_resume_probability": b.tool_resume_probability,
                    "pin_until_step": -1 if b.prefix_id not in self.pinned else 10**9,
                }
            )
        return rows
