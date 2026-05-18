from __future__ import annotations

from dataclasses import dataclass, field

from waferagent.kv_model import PrefixBlock, ttl_priority


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
        reload_bytes = kv_bytes
        self.region.reload_bytes += reload_bytes
        evicted_bytes = self._ensure_capacity(job_id, stage_id, kv_bytes)
        if kv_bytes > self.region.capacity_bytes:
            spill = kv_bytes - self.region.capacity_bytes
            self.region.spill_bytes += spill
            self.events.append(self._event(job_id, stage_id, prefix_id, "spill", spill))
            return False, reload_bytes + spill, evicted_bytes
        self.region.resident_blocks[prefix_id] = block
        self.region.used_bytes += kv_bytes
        self.events.append(self._event(job_id, stage_id, prefix_id, "reload", reload_bytes))
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
