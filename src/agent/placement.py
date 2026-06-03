from typing import Dict, Iterable, List

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
    return saved - copy_cost - pressure_penalty


def _placement_bytes(states: List[KVStateSpec], state_locations: Dict[int, List[tuple]]) -> int:
    states_by_id = {state.state_id: state for state in states}
    return int(sum(states_by_id[state_id].total_bytes * len(locs) for state_id, locs in state_locations.items()))


def place_states(
    agents: List[AgentSpec],
    states: List[KVStateSpec],
    krds: List[KRD],
    hardware_platform,
    policy: Policy,
    dijkstra: bool = True,
    gain_threshold: float = 0.0,
    region_size: int = 8,
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
                locations = [global_medoid]
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
                    gain = replication_gain(
                        state=state,
                        krd=krd,
                        agents_in_krd=agents_in_krd,
                        old_locations=locations,
                        new_location=new_location,
                        agent_decode_nodes=agent_decode_nodes,
                        hardware_platform=hardware_platform,
                        dijkstra=dijkstra,
                    )
                    if gain > gain_threshold:
                        locations.append(new_location)
                state_locations[state.state_id] = unique_nodes(locations)
        else:
            raise ValueError(f"cannot place state {state.state_id} with kind {state.kind}")

    replica_bytes = _placement_bytes(states, state_locations)
    return PlacementPlan(
        policy=policy,
        state_locations=state_locations,
        agent_decode_nodes=agent_decode_nodes,
        krds=placed_krds,
        replica_bytes=replica_bytes,
    )


def apply_placement(data_dict: Dict, states: List[KVStateSpec], plan: PlacementPlan) -> None:
    for state in states:
        tensor = data_dict[state.data_tag]
        locs = plan.state_locations[state.state_id]
        assert locs, f"state {state.state_id} has no placement"
        for split in tensor.generated_splitted_tag_dict:
            tensor.generated_split_location[split] = list(locs)

