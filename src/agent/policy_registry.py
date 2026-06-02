from dataclasses import dataclass
from typing import Dict, Iterable, Tuple


@dataclass(frozen=True)
class PolicySpec:
    name: str
    backend_policy_name: str
    retention_policy: str
    mapping_policy: str
    prefetch_policy: str
    estimator_mode: str
    baseline_class: str
    topology_enabled: bool
    prefetch_enabled: bool
    static_replica_enabled: bool
    common_compression_enabled: bool = False
    asg_compression_enabled: bool = False


_POLICIES: Dict[str, PolicySpec] = {
    "nocache": PolicySpec("nocache", "nocache", "none", "none", "none", "none", "basic", False, False, False),
    "lru": PolicySpec("lru", "lru", "lru", "none", "none", "none", "system", False, False, True),
    "lru-basic": PolicySpec("lru-basic", "lru", "lru", "none", "none", "none", "basic", False, False, False),
    "lru-system": PolicySpec("lru-system", "lru", "lru", "static_replica", "none", "none", "system", False, False, True),
    "kvflow": PolicySpec("kvflow", "kvflow", "kvflow", "none", "none", "none", "system", False, False, True),
    "kvflow-like": PolicySpec("kvflow-like", "kvflow", "kvflow", "none", "none", "none", "system", False, False, True),
    "asg-retention": PolicySpec("asg-retention", "asg-retention", "asg", "none", "none", "none", "asg", False, False, True),
    "asg-placement": PolicySpec("asg-placement", "asg-placement", "asg", "topology", "none", "none", "asg", True, False, True),
    "asg-prefetch": PolicySpec("asg-prefetch", "asg-prefetch", "asg", "topology", "greedy", "none", "asg", True, True, True),
    "asg-retention-v2": PolicySpec("asg-retention-v2", "asg-retention-v2", "asg-v2", "none", "none", "none", "asg", False, False, True),
    "asg-placement-v2": PolicySpec("asg-placement-v2", "asg-placement-v2", "asg-v2", "topology", "none", "none", "asg", True, False, True),
    "asg-prefetch-v2": PolicySpec("asg-prefetch-v2", "asg-prefetch-v2", "asg-v2", "topology", "windowed", "none", "asg", True, True, True),
    "asg-retention-v2-graph-only": PolicySpec("asg-retention-v2-graph-only", "asg-retention-v2", "asg-v2", "none", "none", "none", "asg", False, False, True),
    "asg-retention-v2-online": PolicySpec("asg-retention-v2-online", "asg-retention-v2", "asg-v2", "none", "none", "heuristic", "asg", False, False, True),
    "asg-retention-v2-trace-stats": PolicySpec("asg-retention-v2-trace-stats", "asg-retention-v2", "asg-v2", "none", "none", "trace_stats", "asg", False, False, True),
    "asg-retention-v2-oracle": PolicySpec("asg-retention-v2-oracle", "asg-oracle-retention", "asg-v2", "none", "none", "oracle", "oracle", False, False, True),
    "asg-placement-v2-online": PolicySpec("asg-placement-v2-online", "asg-placement-v2", "asg-v2", "asg_cost_aware", "none", "heuristic", "asg", True, False, True),
    "asg-prefetch-v2-online": PolicySpec("asg-prefetch-v2-online", "asg-prefetch-v2", "asg-v2", "asg_cost_aware", "windowed", "heuristic", "asg", True, True, True),
    "asg-oracle-retention": PolicySpec("asg-oracle-retention", "asg-oracle-retention", "asg-v2", "none", "none", "oracle", "oracle", False, False, True),
    "asg-oracle-placement": PolicySpec("asg-oracle-placement", "asg-oracle-placement", "asg-v2", "topology", "none", "oracle", "oracle", True, False, True),
    "asg-oracle-prefetch": PolicySpec("asg-oracle-prefetch", "asg-oracle-prefetch", "asg-v2", "topology", "windowed", "oracle", "oracle", True, True, True),
    "asg": PolicySpec("asg", "asg-prefetch-v2", "asg-v2", "asg_cost_aware", "windowed", "heuristic", "asg", True, True, True),
}


REAL_V2_POLICY_NAMES: Tuple[str, ...] = (
    "nocache",
    "lru-basic",
    "lru-system",
    "kvflow-like",
    "asg-retention-v2-graph-only",
    "asg-retention-v2-online",
    "asg-retention-v2-trace-stats",
    "asg-retention-v2-oracle",
    "asg-placement-v2-online",
    "asg-prefetch-v2-online",
)


def get_policy_spec(name: str) -> PolicySpec:
    try:
        return _POLICIES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown policy: {name}") from exc


def list_policy_specs(names: Iterable[str]):
    return [get_policy_spec(name) for name in names]
