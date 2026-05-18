from __future__ import annotations

from waferagent.sram_manager import SRAMManager


def _exercise(capacity: int) -> int:
    mgr = SRAMManager(capacity, "lru", "unit")
    for i in range(4):
        mgr.access("j", f"s{i}", f"p{i}", 10, 80, i, 0.1)
    return int(mgr.stats()["sram_evictions"])


def test_lower_sram_capacity_increases_evictions():
    assert _exercise(120) > _exercise(400)
