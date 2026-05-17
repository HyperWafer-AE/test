from __future__ import annotations

from dataclasses import dataclass


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
    prefill_time_multiplier: float = 1.0
    decode_time_multiplier: float = 1.0
    comm_time_multiplier: float = 1.0
    parallelism_multiplier: float = 1.0


BASELINES: dict[str, BaselineConfig] = {
    "wafer_naive": BaselineConfig(
        name="wafer_naive",
        placement_policy="round_robin",
        scheduling_policy="fifo_topological",
        kv_sharing=False,
        ttl_policy="lru",
        parallelism_multiplier=0.75,
    ),
    "kvflow_like": BaselineConfig(
        name="kvflow_like",
        placement_policy="round_robin",
        scheduling_policy="kvflow_like_steps_to_execution",
        kv_sharing=True,
        ttl_policy="steps_to_execution",
        prefill_time_multiplier=0.82,
        parallelism_multiplier=0.9,
    ),
    "continuum_like": BaselineConfig(
        name="continuum_like",
        placement_policy="layer_contiguous",
        scheduling_policy="continuum_like_tool_ttl",
        kv_sharing=False,
        tool_ttl=True,
        prefill_time_multiplier=0.95,
        decode_time_multiplier=0.95,
        parallelism_multiplier=1.0,
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
        prefill_time_multiplier=0.62,
        decode_time_multiplier=0.82,
        comm_time_multiplier=0.70,
        parallelism_multiplier=1.35,
    ),
}


ABLATIONS: dict[str, BaselineConfig] = {
    "waferagent_full": BASELINES["waferagent_full"],
    "no_kv_sharing": BaselineConfig(**{**BASELINES["waferagent_full"].__dict__, "name": "no_kv_sharing", "kv_sharing": False, "prefill_time_multiplier": 0.86}),
    "no_affinity_placement": BaselineConfig(**{**BASELINES["waferagent_full"].__dict__, "name": "no_affinity_placement", "placement_policy": "round_robin", "comm_time_multiplier": 1.0}),
    "no_critical_path_scheduling": BaselineConfig(**{**BASELINES["waferagent_full"].__dict__, "name": "no_critical_path_scheduling", "critical_path": False, "scheduling_policy": "fifo_topological", "parallelism_multiplier": 1.0}),
    "no_tool_ttl": BaselineConfig(**{**BASELINES["waferagent_full"].__dict__, "name": "no_tool_ttl", "tool_ttl": False}),
    "no_dynamic_pd_partition": BaselineConfig(**{**BASELINES["waferagent_full"].__dict__, "name": "no_dynamic_pd_partition", "dynamic_pd_partition": False, "decode_time_multiplier": 0.96}),
    "no_aggregator_placement": BaselineConfig(**{**BASELINES["waferagent_full"].__dict__, "name": "no_aggregator_placement", "aggregator_placement": False, "comm_time_multiplier": 0.90}),
    "no_mesh_congestion_penalty": BaselineConfig(**{**BASELINES["waferagent_full"].__dict__, "name": "no_mesh_congestion_penalty", "mesh_congestion_penalty": False, "comm_time_multiplier": 0.95}),
}


def get_baseline(name: str) -> BaselineConfig:
    if name in BASELINES:
        return BASELINES[name]
    if name in ABLATIONS:
        return ABLATIONS[name]
    raise ValueError(f"Unknown baseline/variant: {name}")
