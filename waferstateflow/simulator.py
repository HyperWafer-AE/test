"""Execution simulator for WaferStateFlow baselines and proposed scheduling."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from .hotness import HotnessTracker, initialize_static_hotness
from .ir import OperatorNode, StateAccessGraph, StateNode
from .schedulers import OperatorCentricScheduler, RequestCentricScheduler, StateCentricWaveScheduler, Wave
from .state_policy import PolicyDecision, decide_state_policy
from .wafer_topology import StatePlacement, WaferTopology


BASELINES = (
    "flat_sequential",
    "request_parallel_gpu_like",
    "prefix_cache_like",
    "helium_like_operator_schedule",
    "kvflow_like_future_eviction",
    "wafer_request_centric",
    "replicate_all_hot_states",
    "single_pin_hot_state",
    "WaferStateFlow",
)


@dataclass(frozen=True)
class BackendProfile:
    prefill_time_per_token: float = 0.002
    decode_time_per_token: float = 0.006
    local_op_time: float = 0.1
    batch_prefill_multiplier: float = 1.0


@dataclass(frozen=True)
class SimulationConfig:
    worker_count: int = 8
    backend_profile: BackendProfile = field(default_factory=BackendProfile)
    wave_launch_overhead: float = 0.05
    gpu_worker_cache_bytes: int = 128 * 1024 * 1024
    global_cache_bytes: int = 512 * 1024 * 1024
    critical_wait_threshold: float = 4.0


@dataclass(frozen=True)
class SimulationResult:
    baseline: str
    workflow_latency: float
    state_materialization_bytes: int
    state_movement_byte_hop: int
    max_link_utilization: float
    max_link_load: int
    p95_link_load: float
    hotspot_region: str
    region_memory_pressure: float
    critical_path_wait: float
    average_wave_batch_size: float
    wave_count: int
    duplicate_state_materialization_bytes: int
    notes: str = ""

    def to_row(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SimulationRun:
    result: SimulationResult
    wave_schedule: list[dict[str, object]]
    policy_decisions: list[PolicyDecision]
    state_access_events: list[dict[str, object]]


def run_all_baselines(
    graph: StateAccessGraph,
    topology: WaferTopology | None = None,
    config: SimulationConfig | None = None,
    state_policy: str = "dynamic",
    baselines: tuple[str, ...] = BASELINES,
) -> list[SimulationRun]:
    return [
        simulate_workflow(graph, baseline, topology=topology, config=config, state_policy=state_policy)
        for baseline in baselines
    ]


def simulate_workflow(
    graph: StateAccessGraph,
    baseline: str,
    topology: WaferTopology | None = None,
    config: SimulationConfig | None = None,
    state_policy: str = "dynamic",
) -> SimulationRun:
    if baseline not in BASELINES:
        raise ValueError(f"unknown baseline {baseline}")
    cfg = config or SimulationConfig()
    topo = topology or WaferTopology()
    graph.update_operator_input_tokens()
    graph.compute_lifetimes()
    initialize_static_hotness(graph)
    tracker = HotnessTracker(graph)

    policies = _decide_policies(graph, baseline, state_policy)
    policy_by_state = {decision.state_id: decision for decision in policies}
    worker_regions = _worker_regions(topo, cfg.worker_count)
    hot_state_ids = _hot_state_ids(graph)

    scheduler = _scheduler_for_baseline(baseline, cfg)
    completed: set[str] = set()
    ready_since: dict[str, float] = {}
    current_time = 0.0
    wave_rows: list[dict[str, object]] = []
    wave_sizes: list[int] = []
    materialization_bytes = 0
    movement_byte_hop = 0
    link_loads: dict[tuple[str, str], int] = {}
    state_access_events: list[dict[str, object]] = []
    critical_wait = 0.0
    worker_caches: dict[str, set[str]] = {region: set() for region in worker_regions}
    global_cache: set[str] = set()
    placements: dict[str, StatePlacement] = {}
    region_memory_used: dict[str, int] = {region.region_id: 0 for region in topo.regions}

    while len(completed) < len(graph.operators):
        ready = graph.ready_operators(completed)
        if not ready:
            raise RuntimeError("workflow has no ready operators but is incomplete")
        for op_id in ready:
            ready_since.setdefault(op_id, current_time)

        wave = _next_wave_for_baseline(baseline, scheduler, graph, completed, ready, current_time)
        if not wave.operator_ids:
            break
        selected = list(wave.operator_ids)
        selected_regions = _assign_regions(baseline, topo, worker_regions, selected, wave, placements)

        wave_mat = 0
        wave_move = 0
        wave_link_load_start = dict(link_loads)
        for op_id, region in zip(selected, selected_regions):
            op = graph.operators[op_id]
            for state_id in op.input_states:
                state = graph.states[state_id]
                hotness_before = state.dynamic_hotness
                policy_before = policy_by_state[state_id].policy
                tracker.observe_access(state_id, op)
                if baseline == "WaferStateFlow" and state_policy == "dynamic":
                    policy_by_state[state_id] = decide_state_policy(
                        state,
                        hotness=tracker.dynamic_hotness.get(state_id, 0.0),
                        memory_pressure=_memory_pressure(region_memory_used, topo),
                    )
                charge, cache_note = _materialization_charge(
                    baseline,
                    state,
                    region,
                    worker_caches,
                    global_cache,
                    hot_state_ids,
                    policy_by_state,
                    wave.seed_state_id,
                    selected,
                    graph,
                )
                wave_mat += charge
                placement = _placement_for_state(
                    baseline,
                    topo,
                    state,
                    policy_by_state,
                    placements,
                    selected_regions,
                    hot_state_ids,
                    region_memory_used,
                )
                if _uses_wafer_movement(baseline):
                    state_move = topo.add_state_transfer(link_loads, state, placement, region)
                    wave_move += state_move
                elif charge > 0:
                    state_move = charge
                    wave_move += charge
                else:
                    state_move = 0
                if cache_note == "global":
                    _admit_global_cache(baseline, state_id, global_cache, graph, cfg, completed)
                elif cache_note == "worker":
                    worker_caches[region].add(state_id)
                state_access_events.append(
                    {
                        "baseline": baseline,
                        "wave_id": len(wave_rows),
                        "time": current_time,
                        "operator_id": op_id,
                        "state_id": state_id,
                        "region_id": region,
                        "policy_before": policy_before,
                        "policy_after": policy_by_state[state_id].policy,
                        "dynamic_hotness_before": hotness_before,
                        "dynamic_hotness_after": state.dynamic_hotness,
                        "materialization_bytes": int(charge),
                        "movement_byte_hop": int(state_move),
                        "cache_hit": bool(charge == 0),
                    }
                )

        op_times = [_operator_time(graph.operators[op_id], baseline, len(selected), cfg) for op_id in selected]
        movement_time = _movement_time(wave_move, topo)
        wave_time = max(op_times) + movement_time + cfg.wave_launch_overhead
        start_time = current_time
        end_time = current_time + wave_time

        for op_id in ready:
            if op_id not in selected and graph.operators[op_id].criticality >= 2.0:
                critical_wait += wave_time
        critical_wait += wave.wait_penalty

        completed.update(selected)
        current_time = end_time
        materialization_bytes += int(wave_mat)
        movement_byte_hop += int(wave_move)
        wave_sizes.append(len(selected))
        wave_rows.append(
            {
                "wave_id": len(wave_rows),
                "baseline": baseline,
                "scheduler": wave.scheduler,
                "seed_state_id": wave.seed_state_id or "",
                "operator_ids": json.dumps(selected),
                "region_ids": json.dumps(selected_regions),
                "start_time": start_time,
                "end_time": end_time,
                "batch_size": len(selected),
                "materialization_bytes": int(wave_mat),
                "movement_byte_hop": int(wave_move),
                "max_link_load_delta": int(_max_link_delta(wave_link_load_start, link_loads)),
                "benefit": wave.benefit,
                "wait_penalty": wave.wait_penalty,
            }
        )

    duplicate_bytes = sum(
        state.text_size_bytes * max(0, len(state.consumers) - 1) for state in graph.states.values()
    )
    if baseline == "flat_sequential":
        region_pressure = 0.0
    elif _uses_wafer_movement(baseline):
        region_pressure = max(region_memory_used.values(), default=0) / max(1, topo.region_memory_capacity)
    else:
        region_pressure = max(
            (
                sum(graph.states[state_id].materialized_size_bytes for state_id in cache)
                / max(1, cfg.gpu_worker_cache_bytes)
                for cache in worker_caches.values()
            ),
            default=0.0,
        )

    link_summary = topo.link_load_summary(link_loads)
    result = SimulationResult(
        baseline=baseline,
        workflow_latency=current_time,
        state_materialization_bytes=materialization_bytes,
        state_movement_byte_hop=movement_byte_hop,
        max_link_utilization=_max_link_utilization(
            int(link_summary["max_link_load"]), topo, max(current_time, 1e-9)
        ),
        max_link_load=int(link_summary["max_link_load"]),
        p95_link_load=float(link_summary["p95_link_load"]),
        hotspot_region=str(link_summary["hotspot_region"]),
        region_memory_pressure=region_pressure,
        critical_path_wait=critical_wait,
        average_wave_batch_size=sum(wave_sizes) / len(wave_sizes) if wave_sizes else 0.0,
        wave_count=len(wave_rows),
        duplicate_state_materialization_bytes=duplicate_bytes,
        notes=_baseline_note(baseline),
    )
    return SimulationRun(result, wave_rows, list(policy_by_state.values()), state_access_events)


def _scheduler_for_baseline(baseline: str, cfg: SimulationConfig):
    if baseline == "WaferStateFlow":
        return StateCentricWaveScheduler(critical_wait_threshold=cfg.critical_wait_threshold)
    if baseline in {"helium_like_operator_schedule", "kvflow_like_future_eviction"}:
        return OperatorCentricScheduler(max_wave_size=cfg.worker_count)
    return RequestCentricScheduler()


def _next_wave_for_baseline(
    baseline: str,
    scheduler: object,
    graph: StateAccessGraph,
    completed: set[str],
    ready: list[str],
    current_time: float,
) -> Wave:
    if baseline == "flat_sequential":
        return Wave(len(completed), baseline, (ready[0],), None, 0.0, 0.0)
    if baseline in {
        "request_parallel_gpu_like",
        "prefix_cache_like",
        "wafer_request_centric",
        "replicate_all_hot_states",
        "single_pin_hot_state",
    }:
        selected = tuple(ready[:8])
        return Wave(len(completed), baseline, selected, None, float(len(selected)), 0.0)
    if baseline == "helium_like_operator_schedule":
        groups: dict[tuple[str, ...], list[str]] = {}
        for op_id in ready:
            template = tuple(
                state_id
                for state_id in graph.operators[op_id].input_states
                if graph.states[state_id].prefix_compatible
            )
            groups.setdefault(template, []).append(op_id)
        selected = max(groups.values(), key=lambda op_ids: (len(op_ids), -ready.index(op_ids[0])))
        selected = selected[:8]
        return Wave(len(completed), baseline, tuple(selected), None, float(len(selected)), 0.0)
    return scheduler.next_wave(graph, completed, current_time)


def _operator_time(op: OperatorNode, baseline: str, batch_size: int, cfg: SimulationConfig) -> float:
    if op.kind != "llm":
        return cfg.backend_profile.local_op_time
    prefill = (
        op.estimated_input_tokens
        * cfg.backend_profile.prefill_time_per_token
        * cfg.backend_profile.batch_prefill_multiplier
    )
    decode = op.estimated_output_tokens * cfg.backend_profile.decode_time_per_token
    return prefill + decode


def _materialization_charge(
    baseline: str,
    state: StateNode,
    region: str,
    worker_caches: dict[str, set[str]],
    global_cache: set[str],
    hot_state_ids: set[str],
    policy_by_state: dict[str, PolicyDecision],
    seed_state_id: str | None,
    wave_ops: list[str],
    graph: StateAccessGraph,
) -> tuple[int, str]:
    size = state.text_size_bytes
    if baseline == "flat_sequential":
        return size, ""
    if baseline in {"request_parallel_gpu_like", "wafer_request_centric"}:
        if state.state_id in worker_caches[region]:
            return 0, "worker"
        return size, "worker"
    if baseline == "prefix_cache_like":
        is_prefix_like = state.producer is None and state.prefix_compatible
        if is_prefix_like and state.state_id in global_cache:
            return 0, "global"
        return size, "global" if is_prefix_like else ""
    if baseline == "helium_like_operator_schedule":
        if state.state_id in global_cache:
            return 0, "global"
        return size if state.state_id in _unique_inputs_for_wave(wave_ops, graph) else 0, "global"
    if baseline == "kvflow_like_future_eviction":
        if state.state_id in global_cache:
            return 0, "global"
        if len(state.consumers) > 1:
            return size, "global"
        return size, ""
    if baseline == "replicate_all_hot_states":
        if state.state_id in hot_state_ids and state.state_id in global_cache:
            return 0, "global"
        return size, "global" if state.state_id in hot_state_ids else ""
    if baseline == "single_pin_hot_state":
        if state.state_id in hot_state_ids and state.state_id in global_cache:
            return 0, "global"
        return size, "global" if state.state_id in hot_state_ids else ""
    if baseline == "WaferStateFlow":
        policy = policy_by_state[state.state_id].policy
        if policy in {"inline", "recompute", "evict"}:
            return size, ""
        if state.state_id in global_cache:
            return 0, "global"
        if seed_state_id == state.state_id:
            return size, "global"
        decision_inputs = _unique_inputs_for_wave(wave_ops, graph)
        return (size if state.state_id in decision_inputs else 0), "global"
    return size, ""


def _unique_inputs_for_wave(wave_ops: list[str], graph: StateAccessGraph) -> set[str]:
    inputs: set[str] = set()
    for op_id in wave_ops:
        inputs.update(graph.operators[op_id].input_states)
    return inputs


def _placement_for_state(
    baseline: str,
    topology: WaferTopology,
    state: StateNode,
    policy_by_state: dict[str, PolicyDecision],
    placements: dict[str, StatePlacement],
    selected_regions: list[str],
    hot_state_ids: set[str],
    region_memory_used: dict[str, int],
) -> StatePlacement | None:
    if not _uses_wafer_movement(baseline):
        return None
    if state.state_id in placements:
        return placements[state.state_id]
    if baseline == "replicate_all_hot_states" and state.state_id in hot_state_ids:
        policy = "replicate"
    elif baseline == "single_pin_hot_state" and state.state_id in hot_state_ids:
        policy = "pin"
    elif baseline == "WaferStateFlow":
        policy = policy_by_state.get(state.state_id).policy if state.state_id in policy_by_state else "inline"
    else:
        policy = "inline"
    placement = topology.place_state(state, policy, selected_regions)
    placements[state.state_id] = placement
    for region, bytes_ in topology.placement_region_memory_bytes(state, policy, placement.regions).items():
        region_memory_used[region] = region_memory_used.get(region, 0) + bytes_
    return placement


def _assign_regions(
    baseline: str,
    topology: WaferTopology,
    worker_regions: list[str],
    selected: list[str],
    wave: Wave,
    placements: dict[str, StatePlacement],
) -> list[str]:
    if baseline == "WaferStateFlow" and wave.seed_state_id and wave.seed_state_id in placements:
        regions = placements[wave.seed_state_id].regions
        if regions:
            return [regions[i % len(regions)] for i in range(len(selected))]
    return [worker_regions[i % len(worker_regions)] for i in range(len(selected))]


def _worker_regions(topology: WaferTopology, worker_count: int) -> list[str]:
    coords = [
        (0, 0),
        (0, topology.mesh_y - 1),
        (topology.mesh_x - 1, 0),
        (topology.mesh_x - 1, topology.mesh_y - 1),
        (topology.mesh_x // 2, topology.mesh_y // 2),
        (0, topology.mesh_y // 2),
        (topology.mesh_x // 2, 0),
        (topology.mesh_x - 1, topology.mesh_y // 2),
    ]
    regions = []
    for coord in coords:
        region = topology.region_id(coord)
        if region not in regions:
            regions.append(region)
    all_regions = [region.region_id for region in topology.regions]
    for region in all_regions:
        if len(regions) >= worker_count:
            break
        if region not in regions:
            regions.append(region)
    return regions[:worker_count]


def _decide_policies(
    graph: StateAccessGraph,
    baseline: str,
    state_policy: str,
) -> list[PolicyDecision]:
    decisions = []
    for state in graph.states.values():
        hotness = state.static_hotness
        if state.metadata.get("dynamic_hot_candidate"):
            hotness = min(hotness, state.token_size * 0.1)
        decision = decide_state_policy(state, hotness=hotness, memory_pressure=0.0)
        decisions.append(decision)
    return decisions


def _hot_state_ids(graph: StateAccessGraph) -> set[str]:
    rows = sorted(
        graph.states.values(),
        key=lambda state: state.token_size * max(0, len(state.consumers) - 1),
        reverse=True,
    )
    cutoff = max(1, min(5, len(rows) // 4 or 1))
    return {state.state_id for state in rows[:cutoff] if len(state.consumers) > 1}


def _movement_time(byte_hop: int, topology: WaferTopology) -> float:
    if byte_hop <= 0:
        return 0.0
    avg_hops = max(1, topology.mesh_x + topology.mesh_y)
    bytes_moved = byte_hop / avg_hops
    return bytes_moved / topology.link_bandwidth_bytes + topology.hop_latency * avg_hops


def _max_link_utilization(max_link_load: int, topology: WaferTopology, latency: float) -> float:
    byte_seconds_capacity = topology.link_bandwidth_bytes * latency
    return min(1.0, max_link_load / max(1.0, byte_seconds_capacity))


def _memory_pressure(region_memory_used: dict[str, int], topology: WaferTopology) -> float:
    return max(region_memory_used.values(), default=0) / max(1, topology.region_memory_capacity)


def _max_link_delta(
    before: dict[tuple[str, str], int],
    after: dict[tuple[str, str], int],
) -> int:
    return max((after.get(link, 0) - before.get(link, 0) for link in after), default=0)


def _admit_global_cache(
    baseline: str,
    state_id: str,
    global_cache: set[str],
    graph: StateAccessGraph,
    config: SimulationConfig,
    completed: set[str],
) -> None:
    if baseline != "kvflow_like_future_eviction":
        global_cache.add(state_id)
        return

    candidates = set(global_cache)
    candidates.add(state_id)
    ordered = sorted(
        candidates,
        key=lambda sid: (
            _future_access_count(graph, sid, completed) * graph.states[sid].materialized_size_bytes,
            graph.states[sid].materialized_size_bytes,
        ),
        reverse=True,
    )
    global_cache.clear()
    used = 0
    for sid in ordered:
        size = graph.states[sid].materialized_size_bytes
        if _future_access_count(graph, sid, completed) <= 0:
            continue
        if used + size <= config.global_cache_bytes:
            global_cache.add(sid)
            used += size


def _future_access_count(graph: StateAccessGraph, state_id: str, completed: set[str]) -> int:
    return sum(1 for op_id in graph.states[state_id].consumers if op_id not in completed)


def _uses_wafer_movement(baseline: str) -> bool:
    return baseline in {
        "wafer_request_centric",
        "replicate_all_hot_states",
        "single_pin_hot_state",
        "WaferStateFlow",
    }


def _baseline_note(baseline: str) -> str:
    notes = {
        "flat_sequential": "approximate: sequential flat prompt materialization; no cache or parallelism",
        "request_parallel_gpu_like": "approximate: ready requests assigned to worker-local caches",
        "prefix_cache_like": "approximate: only exact root prefix-compatible states are globally reused",
        "helium_like_operator_schedule": "approximate: groups ready operators by shared prefix/state template",
        "kvflow_like_future_eviction": "approximate: future-use cache admission/eviction under capacity",
        "wafer_request_centric": "approximate: wafer backend without state-centric wave formation",
        "replicate_all_hot_states": "ablation: blindly replicates hot states",
        "single_pin_hot_state": "ablation: pins hot states centrally and exposes hotspots",
        "WaferStateFlow": "approximate: hot-state-seeded wave scheduling with policy-driven placement",
    }
    return notes[baseline]
