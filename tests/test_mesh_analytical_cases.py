from __future__ import annotations

from waferagent.mesh import MeshConfig
from waferagent.mesh_network import MeshNetwork


def test_one_hop_transfer_time():
    cfg = MeshConfig(4, 4, 1, 1, 1, 1, 1000, True, 1, 1)
    mesh = MeshNetwork(cfg, "unit")
    _wait, t, traffic = mesh.route("j", "s", (0, 0), (0, 1), 1_000_000, 0.0, "message_tokens")
    assert traffic == 1_000_000
    assert abs(t - 2.0) < 1e-6


def test_two_flows_share_link_waits():
    cfg = MeshConfig(4, 4, 1, 1, 1, 1, 1000, True, 1, 1)
    mesh = MeshNetwork(cfg, "unit")
    mesh.route("j", "a", (0, 0), (0, 1), 1_000_000, 0.0, "message_tokens")
    wait, _t, _traffic = mesh.route("j", "b", (0, 0), (0, 1), 1_000_000, 0.0, "message_tokens")
    assert wait > 0


def test_multicast_records_less_or_equal_traffic_than_unicast_tree():
    cfg = MeshConfig(4, 4, 1, 1, 1, 10, 1, True, 1, 1)
    multi = MeshNetwork(cfg, "unit")
    _w, _t, mtraffic = multi.multicast("j", "m", (0, 0), [(0, 3), (3, 0)], 1024, 0.0)
    uni = MeshNetwork(MeshConfig(4, 4, 1, 1, 1, 10, 1, False, 1, 1), "unit")
    _w, _t, utraffic = uni.multicast("j", "u", (0, 0), [(0, 3), (3, 0)], 1024, 0.0)
    assert mtraffic <= utraffic
