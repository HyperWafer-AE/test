from typing import Dict, Iterable, List, Optional

from .topology import bfs_region, choose_spread_anchors, compute_nodes, dist, weighted_medoid
from .types import AgentSpec, KRD, KVStateSpec, PlacementPlan, Policy


SHARED_KINDS = {"global", "workflow_shared"}
PRIVATE_KINDS = {"private_hot", "private_warm", "private_cold"}


def unique_nodes(nodes: Iterable):
    seen = set()
    out = []
    for node in nodes:
        if node not in seen:
            seen.add(node)
            out.append(node)
    return out


def _place_decode_nodes(krds: List[KRD], hardware_platform, region_size: int) -> tuple[Dict[int, tuple], List[KRD]]:
    allowed = compute_nodes(hardware_platform)
    if not allowed:
        raise ValueError("hardware platform has no compute nodes")
    anchors = choose_spread_anchors(hardware_platform, len(krds), allowed)
    allowed_set = set(allowed)

    agent_decode_nodes: Dict[int, tuple] = {}
    placed_krds: List[KRD] = []
    for idx, krd in enumerate(krds):
        anchor = anchors[idx % len(anchors)]
        region = bfs_region(hardware_platform, anchor, region_size, allowed_set)
        if not region:
            region = [anchor]
        for local_idx, agent_id in enumerate(sorted(krd.agent_ids)):
            agent_decode_nodes[agent_id] = region[local_idx % len(region)]
        placed_krds.append(
            KRD(
                krd_id=krd.krd_id,
                agent_ids=list(krd.agent_ids),
                workflow_ids=list(krd.workflow_ids),
                regions=region,
                anchor=anchor,
            )
        )
    return agent_decode_nodes, placed_krds


def replication_gain(
    state: KVStateSpec,
    krd: KRD,
    agents_in_krd: List[AgentSpec],
    old_locations: List[tuple],
    new_location: tuple,
    agent_decode_nodes: Dict[int, tuple],
    hardware_platform,
    dijkstra: bool = True,
    current_region_used_bytes: int = 0,
    region_capacity_bytes: Optional[int] = None,
    pressure_penalty_scale: float = 1.0,
) -> float:
    saved = 0.0
    for agent in agents_in_krd:
        if state.state_id not in agent.required_state_ids:
            continue
        consumer = agent_decode_nodes[agent.agent_id]
        old_d = min(dist(hardware_platform, consumer, old, dijkstra) for old in old_locations)
        new_d = dist(hardware_platform, consumer, new_location, dijkstra)
        saved += agent.expected_decode_tokens * state.total_bytes * max(0.0, old_d - new_d)
    copy_cost = state.total_bytes * min(dist(hardware_platform, old, new_location, dijkstra) for old in old_locations)
    pressure_penalty = 0.0
    if region_capacity_bytes and region_capacity_bytes > 0:
        projected = current_region_used_bytes + state.total_bytes
        occupancy_penalty = state.total_bytes * (current_region_used_bytes / region_capacity_bytes)
        overflow_penalty = max(0, projected - region_capacity_bytes) * 1024.0
        pressure_penalty = pressure_penalty_scale * (occupancy_penalty + overflow_penalty)
    return saved - copy_cost - pressure_penalty


def _unique_state_bytes(states: List[KVStateSpec]) -> int:
    return int(sum(state.total_bytes for state in states))


def _resident_bytes(states: List[KVStateSpec], state_locations: Dict[int, List[tuple]]) -> int:
    states_by_id = {state.state_id: state for state in states}
    return int(sum(states_by_id[state_id].total_bytes * len(locs) for state_id, locs in state_locations.items()))


def _sram_capacity_bytes(hardware_platform) -> int:
    if hasattr(hardware_platform, "sram_cfg"):
        return int(hardware_platform.sram_cfg.get("sram_capacity", 0))
    if hasattr(hardware_platform, "sram_dict") and hardware_platform.sram_dict:
        return int(next(iter(hardware_platform.sram_dict.values())).sram_capacity)
    return 0


def _region_id_for_location(location, krds: List[KRD], hardware_platform, dijkstra: bool = True):
    for krd in krds:
        if location in set(krd.regions):
            return krd.krd_id
    anchored = [krd for krd in krds if krd.anchor is not None]
    if not anchored:
        return None
    return min(
        anchored,
        key=lambda krd: (dist(hardware_platform, location, krd.anchor, dijkstra), krd.krd_id),
    ).krd_id


def _region_capacity_bytes(krd: KRD, sram_capacity_bytes: int) -> int:
    return int(len(krd.regions) * sram_capacity_bytes)


def _region_usage(
    states: List[KVStateSpec],
    state_locations: Dict[int, List[tuple]],
    krds: List[KRD],
    hardware_platform,
    dijkstra: bool = True,
) -> Dict[int, int]:
    states_by_id = {state.state_id: state for state in states}
    usage = {krd.krd_id: 0 for krd in krds}
    for state_id, locations in state_locations.items():
        state = states_by_id[state_id]
        for location in unique_nodes(locations):
            region_id = _region_id_for_location(location, krds, hardware_platform, dijkstra)
            if region_id is not None:
                usage[region_id] += state.total_bytes
    return usage


def _finalize_plan(
    policy: str,
    states: List[KVStateSpec],
    state_locations: Dict[int, List[tuple]],
    agent_decode_nodes: Dict[int, tuple],
    krds: List[KRD],
    hardware_platform,
    dijkstra: bool = True,
) -> PlacementPlan:
    unique_state_bytes = _unique_state_bytes(states)
    resident_bytes = _resident_bytes(states, state_locations)
    region_used_bytes = _region_usage(states, state_locations, krds, hardware_platform, dijkstra)
    sram_capacity_bytes = _sram_capacity_bytes(hardware_platform)
    capacity_violations = 0
    for krd in krds:
        if sram_capacity_bytes > 0 and region_used_bytes.get(krd.krd_id, 0) > _region_capacity_bytes(krd, sram_capacity_bytes):
            capacity_violations += 1
    used_values = list(region_used_bytes.values())
    return PlacementPlan(
        policy=policy,
        state_locations=state_locations,
        agent_decode_nodes=agent_decode_nodes,
        krds=krds,
        resident_bytes=resident_bytes,
        unique_state_bytes=unique_state_bytes,
        extra_replica_bytes=max(0, resident_bytes - unique_state_bytes),
        region_used_bytes=region_used_bytes,
        capacity_violations=capacity_violations,
        max_region_used_bytes=int(max(used_values) if used_values else 0),
        avg_region_used_bytes=float(sum(used_values) / len(used_values) if used_values else 0.0),
        sram_capacity_bytes=sram_capacity_bytes,
    )


def place_states(
    agents: List[AgentSpec],
    states: List[KVStateSpec],
    krds: List[KRD],
    hardware_platform,
    policy: Policy,
    dijkstra: bool = True,
    gain_threshold: float = 0.0,
    region_size: int = 8,
    pressure_penalty_scale: float = 1.0,
) -> PlacementPlan:
    if policy not in {"central", "full_replication", "krd_selective"}:
        raise ValueError(f"unknown placement policy: {policy}")

    agent_decode_nodes, placed_krds = _place_decode_nodes(krds, hardware_platform, region_size)
    agents_by_id = {agent.agent_id: agent for agent in agents}
    allowed = compute_nodes(hardware_platform)
    consumers = [agent_decode_nodes[agent.agent_id] for agent in agents]
    weights = [max(1, agent.expected_decode_tokens) for agent in agents]
    global_medoid = weighted_medoid(hardware_platform, allowed, consumers, weights, dijkstra=dijkstra)

    state_locations: Dict[int, List[tuple]] = {}
    for state in states:
        if state.kind in PRIVATE_KINDS and state.owner_agent_id is not None:
            state_locations[state.state_id] = [agent_decode_nodes[state.owner_agent_id]]
        elif state.kind in SHARED_KINDS:
            if policy == "central":
                state_locations[state.state_id] = [global_medoid]
            elif policy == "full_replication":
                state_locations[state.state_id] = unique_nodes(
                    [krd.anchor for krd in placed_krds if krd.anchor is not None]
                )
            else:
                state_locations[state.state_id] = [global_medoid]
        else:
            raise ValueError(f"cannot place state {state.state_id} with kind {state.kind}")

    if policy == "krd_selective":
        sram_capacity_bytes = _sram_capacity_bytes(hardware_platform)
        region_used_bytes = _region_usage(states, state_locations, placed_krds, hardware_platform, dijkstra)
        for state in states:
            if state.kind not in SHARED_KINDS:
                continue
            locations = state_locations[state.state_id]
            for krd in placed_krds:
                agents_in_krd = [agents_by_id[agent_id] for agent_id in krd.agent_ids]
                consumers_in_krd = [
                    agent_decode_nodes[agent.agent_id]
                    for agent in agents_in_krd
                    if state.state_id in agent.required_state_ids
                ]
                if not consumers_in_krd:
                    continue
                krd_weights = [
                    agent.expected_decode_tokens
                    for agent in agents_in_krd
                    if state.state_id in agent.required_state_ids
                ]
                new_location = weighted_medoid(
                    hardware_platform,
                    krd.regions or [krd.anchor],
                    consumers_in_krd,
                    krd_weights,
                    dijkstra=dijkstra,
                )
                if new_location in locations:
                    continue
                region_id = _region_id_for_location(new_location, placed_krds, hardware_platform, dijkstra)
                current_region_used = region_used_bytes.get(region_id, 0)
                region_capacity = _region_capacity_bytes(krd, sram_capacity_bytes) if sram_capacity_bytes else None
                gain = replication_gain(
                    state=state,
                    krd=krd,
                    agents_in_krd=agents_in_krd,
                    old_locations=locations,
                    new_location=new_location,
                    agent_decode_nodes=agent_decode_nodes,
                    hardware_platform=hardware_platform,
                    dijkstra=dijkstra,
                    current_region_used_bytes=current_region_used,
                    region_capacity_bytes=region_capacity,
                    pressure_penalty_scale=pressure_penalty_scale,
                )
                if gain > gain_threshold:
                    locations.append(new_location)
                    if region_id is not None:
                        region_used_bytes[region_id] = current_region_used + state.total_bytes
            state_locations[state.state_id] = unique_nodes(locations)

    return _finalize_plan(
        policy=policy,
        states=states,
        state_locations=state_locations,
        agent_decode_nodes=agent_decode_nodes,
        krds=placed_krds,
        hardware_platform=hardware_platform,
        dijkstra=dijkstra,
    )


def apply_placement(data_dict: Dict, states: List[KVStateSpec], plan: PlacementPlan) -> None:
    for state in states:
        tensor = data_dict[state.data_tag]
        locs = plan.state_locations[state.state_id]
        assert locs, f"state {state.state_id} has no placement"
        for split in tensor.generated_splitted_tag_dict:
            tensor.generated_split_location[split] = list(locs)
