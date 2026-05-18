from __future__ import annotations

from waferagent.mesh import MeshConfig
from waferagent.sram_manager import DistributedSRAMManager


def _exercise(tile_sram_mb: float, policy: str = "lru") -> float:
    cfg = MeshConfig(8, 8, tile_sram_mb, 1, 1, 50, 1, True, 1, 1, sram_region_rows=4, sram_region_cols=4)
    mgr = DistributedSRAMManager(cfg, policy, "unit")
    for i in range(8):
        mgr.access("j", f"s{i}", f"p{i}", 128, 2 * 1024 * 1024, i, 1.0, (0, 0))
    return mgr.stats()["sram_evictions"]


def test_low_region_sram_eviction_is_observed():
    assert _exercise(0.25) > 0
    assert _exercise(0.25) >= _exercise(4.0)
