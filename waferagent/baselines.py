from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class BaselineConfig:
    name: str
    placement_policy: str = "round_robin"
    scheduling_policy: str = "fifo_topological"
    kv_sharing: bool = False
    ttl_policy: str = "lru"
    tool_ttl: bool = False
    critical_path: bool = False
    dynamic_pd_partition: bool = False
    aggregator_placement: bool = False
    mesh_congestion_penalty: bool = False
    hotspot_aware_placement: bool = False
    shared_kv_decode_cohort: bool = False
    shared_kv_placement: bool = False
    shared_kv_replication_policy: str = "none"
    cohort_admission_policy: str = "none"
    future_reuse_policy: bool = False
    distributed_sram_policy: bool = True
    oracle: bool = False
    prefill_time_multiplier: float = 1.0
    decode_time_multiplier: float = 1.0
    comm_time_multiplier: float = 1.0
    parallelism_multiplier: float = 1.0
    mechanism_profile: str = "neutral"


NEUTRAL_BASELINES: dict[str, BaselineConfig] = {
    "no_cache": BaselineConfig(
        name="no_cache",
        placement_policy="round_robin",
        scheduling_policy="fifo_topological",
        kv_sharing=False,
        ttl_policy="lru",
        distributed_sram_policy=False,
    ),
    "wafer_naive": BaselineConfig(
        name="wafer_naive",
        placement_policy="round_robin",
        scheduling_policy="fifo_topological",
        kv_sharing=False,
        ttl_policy="lru",
    ),
    "apc_like": BaselineConfig(
        name="apc_like",
        placement_policy="round_robin",
        scheduling_policy="fifo_topological",
        kv_sharing=True,
        ttl_policy="lru",
        shared_kv_replication_policy="no_replication",
    ),
    "kvflow_like": BaselineConfig(
        name="kvflow_like",
        placement_policy="round_robin",
        scheduling_policy="kvflow_like_steps_to_execution",
        kv_sharing=True,
        ttl_policy="steps_to_execution",
        future_reuse_policy=True,
        shared_kv_replication_policy="no_replication",
    ),
    "pat_like": BaselineConfig(
        name="pat_like",
        placement_policy="round_robin",
        scheduling_policy="fifo_topological",
        kv_sharing=True,
        ttl_policy="lru",
        shared_kv_decode_cohort=True,
        shared_kv_replication_policy="no_replication",
        cohort_admission_policy="traffic_only",
    ),
    "pat_like_traffic_only": BaselineConfig(
        name="pat_like_traffic_only",
        placement_policy="round_robin",
        scheduling_policy="fifo_topological",
        kv_sharing=True,
        ttl_policy="lru",
        shared_kv_decode_cohort=True,
        shared_kv_replication_policy="no_replication",
        cohort_admission_policy="traffic_only",
    ),
    "continuum_like": BaselineConfig(
        name="continuum_like",
        placement_policy="layer_contiguous",
        scheduling_policy="continuum_like_tool_ttl",
        kv_sharing=False,
        ttl_policy="lru",
        tool_ttl=True,
    ),
    "waferagent_full": BaselineConfig(
        name="waferagent_full",
        placement_policy="communication_affinity",
        scheduling_policy="critical_path",
        kv_sharing=True,
        ttl_policy="graph_ttl_criticality",
        tool_ttl=True,
        critical_path=True,
        dynamic_pd_partition=True,
        aggregator_placement=True,
        mesh_congestion_penalty=True,
        hotspot_aware_placement=True,
        shared_kv_decode_cohort=True,
        shared_kv_placement=True,
        shared_kv_replication_policy="benefit_cost",
        cohort_admission_policy="latency_safe",
        future_reuse_policy=True,
    ),
    "waferagent_traffic_only": BaselineConfig(
        name="waferagent_traffic_only",
        placement_policy="communication_affinity",
        scheduling_policy="critical_path",
        kv_sharing=True,
        ttl_policy="graph_ttl_criticality",
        tool_ttl=True,
        critical_path=True,
        dynamic_pd_partition=True,
        aggregator_placement=True,
        mesh_congestion_penalty=True,
        hotspot_aware_placement=True,
        shared_kv_decode_cohort=True,
        shared_kv_placement=True,
        shared_kv_replication_policy="benefit_cost",
        cohort_admission_policy="traffic_only",
        future_reuse_policy=True,
    ),
    "waferagent_latency_safe": BaselineConfig(
        name="waferagent_latency_safe",
        placement_policy="communication_affinity",
        scheduling_policy="critical_path",
        kv_sharing=True,
        ttl_policy="graph_ttl_criticality",
        tool_ttl=True,
        critical_path=True,
        dynamic_pd_partition=True,
        aggregator_placement=True,
        mesh_congestion_penalty=True,
        hotspot_aware_placement=True,
        shared_kv_decode_cohort=True,
        shared_kv_placement=True,
        shared_kv_replication_policy="benefit_cost",
        cohort_admission_policy="latency_safe",
        future_reuse_policy=True,
    ),
    "oracle": BaselineConfig(
        name="oracle",
        placement_policy="communication_affinity",
        scheduling_policy="critical_path",
        kv_sharing=True,
        ttl_policy="oracle_next_use",
        tool_ttl=True,
        critical_path=True,
        dynamic_pd_partition=True,
        aggregator_placement=True,
        mesh_congestion_penalty=True,
        hotspot_aware_placement=True,
        shared_kv_decode_cohort=True,
        shared_kv_placement=True,
        shared_kv_replication_policy="oracle",
        cohort_admission_policy="traffic_only",
        future_reuse_policy=True,
        oracle=True,
    ),
    "ideal_next_use_cache": BaselineConfig(
        name="ideal_next_use_cache",
        placement_policy="communication_affinity",
        scheduling_policy="critical_path",
        kv_sharing=True,
        ttl_policy="oracle_next_use",
        tool_ttl=True,
        critical_path=True,
        dynamic_pd_partition=True,
        aggregator_placement=True,
        mesh_congestion_penalty=True,
        hotspot_aware_placement=True,
        shared_kv_decode_cohort=True,
        shared_kv_placement=True,
        shared_kv_replication_policy="no_replication",
        cohort_admission_policy="traffic_only",
        future_reuse_policy=True,
    ),
}


def _legacy(cfg: BaselineConfig, **kwargs) -> BaselineConfig:
    return replace(cfg, mechanism_profile="legacy_heuristic", **kwargs)


LEGACY_BASELINES: dict[str, BaselineConfig] = {
    "wafer_naive": _legacy(NEUTRAL_BASELINES["wafer_naive"]),
    "no_cache": _legacy(NEUTRAL_BASELINES["no_cache"]),
    "apc_like": _legacy(NEUTRAL_BASELINES["apc_like"]),
    "kvflow_like": _legacy(
        NEUTRAL_BASELINES["kvflow_like"],
        prefill_time_multiplier=0.82,
        parallelism_multiplier=0.9,
    ),
    "continuum_like": _legacy(
        NEUTRAL_BASELINES["continuum_like"],
        prefill_time_multiplier=0.95,
        decode_time_multiplier=0.95,
    ),
    "pat_like": _legacy(NEUTRAL_BASELINES["pat_like"]),
    "pat_like_traffic_only": _legacy(NEUTRAL_BASELINES["pat_like_traffic_only"]),
    "waferagent_full": _legacy(
        NEUTRAL_BASELINES["waferagent_full"],
        prefill_time_multiplier=0.62,
        decode_time_multiplier=0.82,
        comm_time_multiplier=0.70,
        parallelism_multiplier=1.35,
    ),
    "waferagent_traffic_only": _legacy(NEUTRAL_BASELINES["waferagent_traffic_only"]),
    "waferagent_latency_safe": _legacy(NEUTRAL_BASELINES["waferagent_latency_safe"]),
    "oracle": _legacy(NEUTRAL_BASELINES["oracle"]),
    "ideal_next_use_cache": _legacy(NEUTRAL_BASELINES["ideal_next_use_cache"]),
}


def _ablation(base: BaselineConfig, name: str, **kwargs) -> BaselineConfig:
    return replace(base, name=name, **kwargs)


def ablations(neutral: bool = True) -> dict[str, BaselineConfig]:
    base = get_baseline("waferagent_full", neutral=neutral)
    return {
        "waferagent_full": base,
        "no_kv_sharing": _ablation(base, "no_kv_sharing", kv_sharing=False),
        "no_affinity_placement": _ablation(
            base, "no_affinity_placement", placement_policy="round_robin", aggregator_placement=False
        ),
        "no_hotspot_aware_placement": _ablation(base, "no_hotspot_aware_placement", hotspot_aware_placement=False),
        "no_critical_path_scheduling": _ablation(
            base, "no_critical_path_scheduling", critical_path=False, scheduling_policy="fifo_topological"
        ),
        "no_tool_ttl": _ablation(base, "no_tool_ttl", tool_ttl=False, ttl_policy="lru"),
        "no_dynamic_pd_partition": _ablation(base, "no_dynamic_pd_partition", dynamic_pd_partition=False),
        "no_aggregator_placement": _ablation(base, "no_aggregator_placement", aggregator_placement=False),
        "no_mesh_congestion_penalty": _ablation(base, "no_mesh_congestion_penalty", mesh_congestion_penalty=False),
        "no_shared_kv_decode_cohort": _ablation(base, "no_shared_kv_decode_cohort", shared_kv_decode_cohort=False),
        "no_shared_kv_replication": _ablation(base, "no_shared_kv_replication", shared_kv_replication_policy="no_replication"),
        "no_distributed_sram_policy": _ablation(base, "no_distributed_sram_policy", distributed_sram_policy=False, ttl_policy="lru"),
        "no_future_reuse_policy": _ablation(base, "no_future_reuse_policy", future_reuse_policy=False, ttl_policy="lru"),
    }


def get_baseline(name: str, neutral: bool = True) -> BaselineConfig:
    table = NEUTRAL_BASELINES if neutral else LEGACY_BASELINES
    if name in table:
        return table[name]
    abl = ablations(neutral=neutral)
    if name in abl:
        return abl[name]
    raise ValueError(f"Unknown baseline/variant: {name}")


def assert_neutral_multipliers(cfg: BaselineConfig) -> None:
    values = [
        cfg.prefill_time_multiplier,
        cfg.decode_time_multiplier,
        cfg.comm_time_multiplier,
        cfg.parallelism_multiplier,
    ]
    if any(v != 1.0 for v in values):
        raise AssertionError(f"{cfg.name} is not neutral: {values}")


BASELINES = NEUTRAL_BASELINES
ABLATIONS = ablations(neutral=True)
