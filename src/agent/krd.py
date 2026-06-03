from itertools import combinations
from typing import Dict, List, Sequence

from .types import AgentSpec, KRD, KVStateSpec


def agent_affinity(
    ai: AgentSpec,
    aj: AgentSpec,
    states_by_id: Dict[int, KVStateSpec],
    alpha: float = 0.1,
    beta: float = 0.05,
    global_weight: float = 0.1,
    workflow_bonus: float = 1.0,
) -> float:
    shared_ids = set(ai.required_state_ids) & set(aj.required_state_ids)
    weighted_shared = 0.0
    normalizer = 0.0
    private_block_bytes = 0
    for state_id in shared_ids:
        state = states_by_id[state_id]
        if state.owner_agent_id is None:
            contribution = state.reuse_count * state.total_bytes
            weight = global_weight if state.kind == "global" else 1.0
            weighted_shared += weight * contribution
            normalizer += contribution
        else:
            private_block_bytes = max(private_block_bytes, state.block_bytes)
    if private_block_bytes == 0:
        private_block_bytes = max((state.block_bytes for state in states_by_id.values()), default=2)
    shared_score = weighted_shared / normalizer if normalizer else 0.0
    time_score = 1.0 / (1.0 + abs(ai.expected_return_time - aj.expected_return_time))
    same_workflow_bonus = workflow_bonus if ai.workflow_id == aj.workflow_id else 0.0
    pressure_bytes = (ai.private_blocks + aj.private_blocks) * private_block_bytes
    pressure_normalizer = max((state.total_bytes for state in states_by_id.values()), default=1)
    pressure_penalty = pressure_bytes / pressure_normalizer
    return shared_score + alpha * time_score + same_workflow_bonus - beta * pressure_penalty


def _chunks(items: Sequence[AgentSpec], size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _krd_from_agents(krd_id: int, agents: Sequence[AgentSpec]) -> KRD:
    return KRD(
        krd_id=krd_id,
        agent_ids=[agent.agent_id for agent in agents],
        workflow_ids=sorted({agent.workflow_id for agent in agents}),
    )


def _workflow_krds(agents: List[AgentSpec], max_agents_per_krd: int) -> List[KRD]:
    by_workflow: Dict[int, List[AgentSpec]] = {}
    for agent in sorted(agents, key=lambda a: (a.workflow_id, a.agent_id)):
        by_workflow.setdefault(agent.workflow_id, []).append(agent)

    krds: List[KRD] = []
    for workflow_id in sorted(by_workflow):
        for chunk in _chunks(by_workflow[workflow_id], max_agents_per_krd):
            krds.append(_krd_from_agents(len(krds), chunk))
    return krds


def _average_cluster_affinity(
    left: Sequence[AgentSpec],
    right: Sequence[AgentSpec],
    states_by_id: Dict[int, KVStateSpec],
) -> float:
    scores = [agent_affinity(ai, aj, states_by_id) for ai in left for aj in right]
    return sum(scores) / len(scores) if scores else float("-inf")


def _affinity_krds(
    agents: List[AgentSpec],
    states: List[KVStateSpec],
    max_agents_per_krd: int,
    max_private_blocks_per_krd: int | None = None,
) -> List[KRD]:
    states_by_id = {state.state_id: state for state in states}
    clusters: List[List[AgentSpec]] = [[agent] for agent in sorted(agents, key=lambda a: a.agent_id)]
    blocked = set()

    while True:
        best_pair = None
        best_score = float("-inf")
        for i, j in combinations(range(len(clusters)), 2):
            if (i, j) in blocked:
                continue
            if len(clusters[i]) + len(clusters[j]) > max_agents_per_krd:
                blocked.add((i, j))
                continue
            if max_private_blocks_per_krd is not None:
                private_blocks = sum(agent.private_blocks for agent in clusters[i] + clusters[j])
                if private_blocks > max_private_blocks_per_krd:
                    blocked.add((i, j))
                    continue
            score = _average_cluster_affinity(clusters[i], clusters[j], states_by_id)
            if score > best_score:
                best_score = score
                best_pair = (i, j)
        if best_pair is None or best_score <= 0:
            break
        i, j = best_pair
        merged = sorted(clusters[i] + clusters[j], key=lambda a: a.agent_id)
        clusters = [cluster for idx, cluster in enumerate(clusters) if idx not in best_pair]
        clusters.append(merged)
        clusters.sort(key=lambda cluster: cluster[0].agent_id)
        blocked = set()

    return [_krd_from_agents(idx, cluster) for idx, cluster in enumerate(clusters)]


def build_krds(
    agents: List[AgentSpec],
    states: List[KVStateSpec],
    mode: str = "workflow",
    max_agents_per_krd: int = 4,
    max_private_blocks_per_krd: int | None = None,
) -> List[KRD]:
    if max_agents_per_krd <= 0:
        raise ValueError("max_agents_per_krd must be positive")
    if mode == "workflow":
        return _workflow_krds(agents, max_agents_per_krd)
    if mode == "affinity":
        return _affinity_krds(agents, states, max_agents_per_krd, max_private_blocks_per_krd)
    raise ValueError(f"unknown KRD mode: {mode}")
