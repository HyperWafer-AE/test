from typing import Dict, List

from beha_notation import beha_notation

from .types import AgentSpec, KVStateSpec, PlacementPlan


def build_kv_read_behaviors(
    beha_dict: Dict,
    data_dict: Dict,
    agents: List[AgentSpec],
    states: List[KVStateSpec],
    plan: PlacementPlan,
) -> None:
    states_by_id = {state.state_id: state for state in states}
    for agent in agents:
        decode_node = plan.agent_decode_nodes[agent.agent_id]
        for local_idx, state_id in enumerate(agent.required_state_ids):
            state = states_by_id[state_id]
            tensor = data_dict[state.data_tag]
            splits = set(tensor.generated_splitted_tag_dict.keys())
            beha_tag = (0, agent.workflow_id, agent.agent_id, state_id, local_idx)
            behavior = beha_notation(
                beha_name=f"agent{agent.agent_id}_{state.name}_kvread",
                beha_tag=beha_tag,
                beha_type="lookup",
                needed_data_split_dict={state.data_tag: splits},
                needed_tag_size_dict={},
            )
            behavior.device = "vectorunit"
            behavior.location = decode_node + (0,)
            beha_dict[beha_tag] = behavior
            for split in splits:
                tensor.used_splitted_tag_dict[split].add(beha_tag)

