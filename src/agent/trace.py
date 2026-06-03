import random
from typing import List, Tuple

from .types import AgentSpec, KVStateSpec


def make_coding_trace(
    num_workflows: int = 2,
    agents_per_workflow: int = 3,
    repo_blocks: int = 8,
    issue_blocks: int = 2,
    private_blocks: int = 4,
    block_elems: int = 65536,
    seed: int = 123,
) -> Tuple[List[AgentSpec], List[KVStateSpec]]:
    rng = random.Random(seed)
    roles = ["planner", "coder", "tester", "reviewer"]
    decode_tokens = {
        "planner": 32,
        "coder": 64,
        "tester": 48,
        "reviewer": 32,
    }

    states: List[KVStateSpec] = []
    agents: List[AgentSpec] = []
    next_state_id = 0

    system_state_id = next_state_id
    states.append(
        KVStateSpec(
            state_id=system_state_id,
            name="global_system_prompt",
            kind="global",
            owner_agent_id=None,
            workflow_id=None,
            num_blocks=2,
            block_elems=block_elems,
            reuse_count=float(num_workflows * agents_per_workflow),
        )
    )
    next_state_id += 1

    tool_schema_state_id = next_state_id
    states.append(
        KVStateSpec(
            state_id=tool_schema_state_id,
            name="global_tool_schema",
            kind="global",
            owner_agent_id=None,
            workflow_id=None,
            num_blocks=2,
            block_elems=block_elems,
            reuse_count=float(num_workflows * agents_per_workflow),
        )
    )
    next_state_id += 1

    workflow_repo_state_ids = {}
    workflow_issue_state_ids = {}
    for workflow_id in range(num_workflows):
        repo_state_id = next_state_id
        states.append(
            KVStateSpec(
                state_id=repo_state_id,
                name=f"workflow{workflow_id}_repo_context",
                kind="workflow_shared",
                owner_agent_id=None,
                workflow_id=workflow_id,
                num_blocks=repo_blocks,
                block_elems=block_elems,
                reuse_count=float(agents_per_workflow),
            )
        )
        workflow_repo_state_ids[workflow_id] = repo_state_id
        next_state_id += 1

        issue_state_id = next_state_id
        states.append(
            KVStateSpec(
                state_id=issue_state_id,
                name=f"workflow{workflow_id}_issue_context",
                kind="workflow_shared",
                owner_agent_id=None,
                workflow_id=workflow_id,
                num_blocks=issue_blocks,
                block_elems=block_elems,
                reuse_count=float(agents_per_workflow),
            )
        )
        workflow_issue_state_ids[workflow_id] = issue_state_id
        next_state_id += 1

    next_agent_id = 0
    for workflow_id in range(num_workflows):
        for local_agent_idx in range(agents_per_workflow):
            role = roles[local_agent_idx % len(roles)]
            private_state_id = next_state_id
            states.append(
                KVStateSpec(
                    state_id=private_state_id,
                    name=f"agent{next_agent_id}_{role}_private_hot",
                    kind="private_hot",
                    owner_agent_id=next_agent_id,
                    workflow_id=workflow_id,
                    num_blocks=private_blocks,
                    block_elems=block_elems,
                    reuse_count=1.0,
                )
            )
            next_state_id += 1

            required_state_ids = (
                system_state_id,
                tool_schema_state_id,
                workflow_repo_state_ids[workflow_id],
                workflow_issue_state_ids[workflow_id],
                private_state_id,
            )
            agents.append(
                AgentSpec(
                    agent_id=next_agent_id,
                    workflow_id=workflow_id,
                    role=role,
                    expected_decode_tokens=decode_tokens[role],
                    expected_return_time=workflow_id * 100.0 + local_agent_idx * 8.0 + rng.random(),
                    private_blocks=private_blocks,
                    required_state_ids=required_state_ids,
                )
            )
            next_agent_id += 1

    return agents, states


if __name__ == "__main__":
    demo_agents, demo_states = make_coding_trace()
    assert len(demo_agents) == 6
    assert all(len(agent.required_state_ids) == 5 for agent in demo_agents)
    print(f"agents={len(demo_agents)} states={len(demo_states)}")

