from __future__ import annotations

from waferagent.baselines import assert_neutral_multipliers, get_baseline


def test_neutral_mode_all_multipliers_are_one():
    for name in ["wafer_naive", "kvflow_like", "continuum_like", "waferagent_full", "oracle"]:
        assert_neutral_multipliers(get_baseline(name, neutral=True))


def test_legacy_profile_is_labeled():
    cfg = get_baseline("waferagent_full", neutral=False)
    assert cfg.mechanism_profile == "legacy_heuristic"
    assert cfg.prefill_time_multiplier != 1.0
