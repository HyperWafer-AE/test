from __future__ import annotations

from waferagent.mesh import MeshConfig
from waferagent.sram_manager import DistributedSRAMManager


def test_shared_kv_replica_consumes_sram_capacity():
    cfg = MeshConfig(4, 4, 1, 1, 1, 25, 1, True, 1, 1, sram_region_rows=1, sram_region_cols=1)
    sram = DistributedSRAMManager(cfg, "lru", "test")
    region = "r0c0"
    sram.materialize("j", "s", "p0", 1, cfg.tile_sram_bytes // 2, 0, 0.0, region)
    used_after_first = sram._region(region).used_bytes
    sram.materialize("j", "s", "p1", 1, cfg.tile_sram_bytes // 2, 1, 0.0, region)
    assert sram._region(region).used_bytes >= used_after_first
    sram.materialize("j", "s", "p2", 1, cfg.tile_sram_bytes, 2, 0.0, region)
    assert sram.stats()["sram_evictions"] > 0

