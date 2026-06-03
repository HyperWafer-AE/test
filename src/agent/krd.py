from itertools import combinations
from typing import Dict, List, Sequence

from .types import AgentSpec, KRD, KVStateSpec


def agent_affinity(
    ai: AgentSpec,
    aj: AgentSpec,
    states_by_id: Dict[int, KVStateSpec],
    alpha: float = 1.0,
    beta: float = 1e-6,
) -> float:
    shared_ids = set(ai.required_state_ids) & set(aj.required_state_ids)
    shared_score = 0.0
    private_block_bytes = 0
    for state_id in shared_ids:
        state = states_by_id[state_id]
        if state.owner_agent_id is None:
            shared_score += state.reuse_count * state.total_bytes
        else:
            private_block_bytes = max(private_block_bytes, state.block_bytes)
    if private_block_bytes == 0:
        private_block_bytes = max((state.block_bytes for state in states_by_id.values()), default=2)
    time_score = 1.0 / (1.0 + abs(ai.expected_return_time - aj.expected_return_time))
    pressure_penalty = (ai.private_blocks + aj.private_blocks) * private_block_bytes
    return shared_score + alpha * time_score - beta * pressure_penalty


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
) -> List[KRD]:
    if max_agents_per_krd <= 0:
        raise ValueError("max_agents_per_krd must be positive")
    if mode == "workflow":
        return _workflow_krds(agents, max_agents_per_krd)
    if mode == "affinity":
        return _affinity_krds(agents, states, max_agents_per_krd)
    raise ValueError(f"unknown KRD mode: {mode}")

