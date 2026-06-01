import json
import random
from pathlib import Path
from typing import List


def _positive_int(rng: random.Random, mean: int, min_value: int = 1) -> int:
    if mean <= min_value:
        return min_value
    value = int(rng.gauss(mean, max(1.0, mean * 0.25)))
    return max(min_value, value)


def _phase_tool(turn: int, turns_per_agent: int, rng: random.Random) -> str:
    if turn < turns_per_agent / 3:
        return rng.choice(["Read", "Grep", "WebFetch", "WebSearch"])
    if turn < 2 * turns_per_agent / 3:
        return rng.choice(["Edit", "Write", "Bash"])
    return rng.choice(["Bash", "pytest", "test", "Verifier"])


def generate_synthetic_trace(
    num_agents: int,
    turns_per_agent: int,
    shared_prefix_tokens: int,
    role_tokens: int,
    append_tokens_mean: int,
    output_tokens_mean: int,
    tool_probability: float,
    tool_latency_mean: int,
    tool_output_tokens_mean: int,
    failure_probability: float,
    seed: int,
    shared_state_probability: float = 0.15,
    cross_agent_share_probability: float = 0.10,
    tool_output_shared_probability: float = 0.20,
    agent_handoff_probability: float = 0.08,
    large_observation_probability: float = 0.10,
) -> List[dict]:
    rng = random.Random(seed)
    trace = [
        {
            "type": "state",
            "state_id": "system",
            "state_type": "system_prefix",
            "owner": "shared",
            "tokens": max(1, shared_prefix_tokens // 2),
        },
        {
            "type": "state",
            "state_id": "task",
            "state_type": "task_prefix",
            "owner": "shared",
            "tokens": max(1, shared_prefix_tokens - shared_prefix_tokens // 2),
        },
    ]
    agent_history = {}
    global_shared_pool = []
    for agent_idx in range(num_agents):
        agent = f"agent_{agent_idx}"
        role_id = f"{agent}_role"
        agent_history[agent] = ["system", "task", role_id]
        trace.append(
            {
                "type": "state",
                "state_id": role_id,
                "state_type": "agent_role",
                "owner": agent,
                "tokens": role_tokens,
            }
        )

    for turn in range(turns_per_agent):
        for agent_idx in range(num_agents):
            agent = f"agent_{agent_idx}"
            output_state = f"{agent}_turn{turn}_assistant"
            input_state_ids = list(agent_history[agent])
            if global_shared_pool and rng.random() <= shared_state_probability:
                input_state_ids.append(rng.choice(global_shared_pool))
            if global_shared_pool and rng.random() <= cross_agent_share_probability:
                input_state_ids.append(rng.choice(global_shared_pool))
            if num_agents > 1 and rng.random() <= agent_handoff_probability:
                donor = f"agent_{(agent_idx - 1) % num_agents}"
                donor_candidates = [
                    state_id for state_id in agent_history.get(donor, [])
                    if state_id not in {"system", "task", f"{donor}_role"}
                ]
                if donor_candidates:
                    input_state_ids.append(donor_candidates[-1])
            input_state_ids = list(dict.fromkeys(input_state_ids))
            trace.append(
                {
                    "type": "llm",
                    "agent": agent,
                    "turn": turn,
                    "input_state_ids": input_state_ids,
                    "append_tokens": _positive_int(rng, append_tokens_mean),
                    "output_tokens": _positive_int(rng, output_tokens_mean),
                    "new_state_id": output_state,
                    "new_state_type": "assistant_delta",
                }
            )
            agent_history[agent].append(output_state)

            if rng.random() <= tool_probability:
                tool = _phase_tool(turn, turns_per_agent, rng)
                failed = rng.random() <= failure_probability
                if failed and tool in {"pytest", "test", "Verifier", "Bash"}:
                    state_type = "test_failure_summary"
                elif failed:
                    state_type = "failure_summary"
                elif tool in {"Read", "Grep", "WebFetch", "WebSearch"}:
                    state_type = "file_context"
                elif tool in {"Edit", "Write"}:
                    state_type = "edit_diff"
                else:
                    state_type = "tool_observation"
                tool_state = f"{agent}_turn{turn}_{tool.lower()}_obs"
                tool_output_tokens = _positive_int(rng, tool_output_tokens_mean)
                if rng.random() <= large_observation_probability:
                    tool_output_tokens *= rng.choice([4, 8])
                trace.append(
                    {
                        "type": "tool",
                        "agent": agent,
                        "turn": turn,
                        "tool": tool,
                        "latency": _positive_int(rng, tool_latency_mean),
                        "output_tokens": tool_output_tokens,
                        "status": "failed" if failed else "ok",
                        "new_state_id": tool_state,
                        "new_state_type": state_type,
                    }
                )
                agent_history[agent].append(tool_state)
                if rng.random() <= tool_output_shared_probability or state_type in {
                    "file_context",
                    "test_failure_summary",
                    "failure_summary",
                }:
                    global_shared_pool.append(tool_state)

    return trace


def save_trace_json(trace: List[dict], path):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")


def load_trace_json(path) -> List[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
