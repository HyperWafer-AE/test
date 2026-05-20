import pandas as pd

from waferagent.adaptive_semantics import audit_policy_stage_map, infer_stage_semantics


def test_adaptive_semantics_apc_stage_is_apc_faithful():
    sem = infer_stage_semantics(
        {
            "global_stage_id": "j:s",
            "stage_type": "decode",
            "prefix_id": "p",
            "selected_policy": "apc_like",
        }
    )
    assert sem.kv_sharing_applied is True
    assert sem.decode_cohort_applied is False
    assert sem.shared_kv_placement_applied is False
    assert sem.is_semantically_faithful is True


def test_adaptive_semantics_pat_and_waferagent_rules():
    detail, summary = audit_policy_stage_map(
        pd.DataFrame(
            [
                {"global_stage_id": "s1", "stage_type": "decode", "prefix_id": "p1", "selected_policy": "pat_like_traffic_only"},
                {"global_stage_id": "s2", "stage_type": "decode", "prefix_id": "p2", "selected_policy": "waferagent_latency_safe"},
            ]
        )
    )
    assert set(detail["selected_policy"]) == {"pat_like_traffic_only", "waferagent_latency_safe"}
    pat = detail[detail["selected_policy"] == "pat_like_traffic_only"].iloc[0]
    waf = detail[detail["selected_policy"] == "waferagent_latency_safe"].iloc[0]
    assert bool(pat["decode_cohort_applied"]) is True
    assert bool(pat["shared_kv_placement_applied"]) is False
    assert bool(waf["decode_cohort_applied"]) is True
    assert bool(waf["shared_kv_placement_applied"]) is True
    assert summary["pass"].astype(bool).all()
