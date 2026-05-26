"""Synthetic agent workflow generators for WaferStateFlow experiments."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from .ir import OperatorNode, StateAccessGraph, StateNode


@dataclass(frozen=True)
class WorkflowConfig:
    num_agents: int = 4
    num_branches: int = 4
    num_rounds: int = 2
    batch_size: int = 1
    shared_state_size: int = 4000
    unique_state_size: int = 400
    dynamic_hot_state_probability: float = 0.25
    output_token_mean: int = 250
    seed: int = 0


WORKFLOW_NAMES = (
    "mapreduce",
    "debate",
    "reflection",
    "iterative",
    "parallel_chains",
    "trading",
    "software_dev",
)


def generate_workflow(workflow: str, **kwargs: object) -> StateAccessGraph:
    config = WorkflowConfig(**kwargs)
    generators = {
        "mapreduce": generate_mapreduce,
        "debate": generate_debate,
        "reflection": generate_reflection,
        "iterative": generate_iterative,
        "parallel_chains": generate_parallel_chains,
        "trading": generate_trading,
        "software_dev": generate_software_dev,
    }
    try:
        return generators[workflow](config)
    except KeyError as exc:
        raise ValueError(f"unknown workflow {workflow!r}; expected one of {WORKFLOW_NAMES}") from exc


def generate_mapreduce(config: WorkflowConfig) -> StateAccessGraph:
    rng = random.Random(config.seed)
    graph = _base_graph("mapreduce", config, ("task", "document", "rubric", "tool_schema"))
    expert_outputs = []
    n = max(1, config.num_agents or config.num_branches)
    for i in range(n):
        unique = _add_state(graph, f"S_unique_expert_{i}", "unique_context", _jitter(rng, config.unique_state_size))
        op = _add_op(graph, f"O_expert_{i}", "llm", f"expert_{i}", criticality=1.0)
        _consume_many(graph, op.op_id, ["S_task", "S_document", "S_rubric", unique.state_id])
        out = _add_output(graph, f"S_expert_{i}_summary", "intermediate_summary", rng, config)
        graph.connect_operator_to_state(op.op_id, out.state_id)
        expert_outputs.append(out.state_id)
    summarizer = _add_op(graph, "O_summarizer", "llm", "summarizer", criticality=2.0)
    _consume_many(graph, summarizer.op_id, ["S_task", "S_rubric", *expert_outputs])
    final = _add_output(graph, "S_final_summary", "output", rng, config, deterministic=True)
    graph.connect_operator_to_state(summarizer.op_id, final.state_id)
    _finalize(graph)
    return graph


def generate_debate(config: WorkflowConfig) -> StateAccessGraph:
    rng = random.Random(config.seed)
    graph = _base_graph("debate", config, ("task", "document", "rubric", "role_instruction"))
    previous_round: list[str] = []
    transcript_states: list[str] = []
    for round_idx in range(max(1, config.num_rounds)):
        current_round = []
        for agent_idx in range(max(2, config.num_agents)):
            role = f"debater_{agent_idx}"
            stance = _add_state(
                graph,
                f"S_round_{round_idx}_stance_{agent_idx}",
                "unique_context",
                _jitter(rng, config.unique_state_size),
            )
            op = _add_op(graph, f"O_debate_r{round_idx}_a{agent_idx}", "llm", role, criticality=1.0)
            inputs = ["S_task", "S_document", "S_rubric", "S_role_instruction", stance.state_id]
            inputs.extend(previous_round)
            _consume_many(graph, op.op_id, inputs)
            out = _add_output(graph, f"S_debate_r{round_idx}_a{agent_idx}", "debate_turn", rng, config)
            graph.connect_operator_to_state(op.op_id, out.state_id)
            current_round.append(out.state_id)
        transcript = _add_state(
            graph,
            f"S_transcript_r{round_idx}",
            "intermediate_summary",
            token_size=sum(graph.states[s].token_size for s in current_round),
            kv_size_bytes=_kv_bytes(sum(graph.states[s].token_size for s in current_round)),
            materialized_form="text",
            metadata={"synthetic": True, "aggregated_from": current_round},
        )
        aggregate = _add_op(
            graph,
            f"O_transcript_r{round_idx}",
            "aggregate",
            "transcript_builder",
            deterministic=True,
            criticality=0.8,
        )
        _consume_many(graph, aggregate.op_id, current_round)
        graph.connect_operator_to_state(aggregate.op_id, transcript.state_id)
        previous_round = [transcript.state_id]
        transcript_states.append(transcript.state_id)
    judge = _add_op(graph, "O_judge", "llm", "judge", criticality=2.2)
    _consume_many(graph, judge.op_id, ["S_task", "S_rubric", *previous_round, *transcript_states[-1:]])
    verdict = _add_output(graph, "S_verdict", "output", rng, config, deterministic=True)
    graph.connect_operator_to_state(judge.op_id, verdict.state_id)
    _finalize(graph)
    return graph


def generate_reflection(config: WorkflowConfig) -> StateAccessGraph:
    rng = random.Random(config.seed)
    graph = _base_graph("reflection", config, ("task", "document", "rubric", "style_guide"))
    expert = _add_op(graph, "O_expert_draft", "llm", "expert", criticality=1.6)
    _consume_many(graph, expert.op_id, ["S_task", "S_document", "S_style_guide"])
    draft = _add_output(graph, "S_draft", "intermediate_summary", rng, config)
    graph.connect_operator_to_state(expert.op_id, draft.state_id)

    feedback = []
    for i in range(max(1, config.num_agents - 1)):
        critic = _add_op(graph, f"O_critic_{i}", "llm", f"critic_{i}", criticality=1.0)
        lens = _add_state(graph, f"S_critic_lens_{i}", "unique_context", _jitter(rng, config.unique_state_size))
        _consume_many(graph, critic.op_id, ["S_task", "S_rubric", draft.state_id, lens.state_id])
        out = _add_output(graph, f"S_critic_{i}_feedback", "critic_feedback", rng, config)
        graph.connect_operator_to_state(critic.op_id, out.state_id)
        feedback.append(out.state_id)

    revise = _add_op(graph, "O_revise", "llm", "reviser", criticality=2.0)
    _consume_many(graph, revise.op_id, ["S_task", "S_document", draft.state_id, *feedback])
    final = _add_output(graph, "S_revised_answer", "output", rng, config, deterministic=True)
    graph.connect_operator_to_state(revise.op_id, final.state_id)
    _finalize(graph)
    return graph


def generate_iterative(config: WorkflowConfig) -> StateAccessGraph:
    rng = random.Random(config.seed)
    graph = _base_graph("iterative", config, ("task", "rubric", "global_context"))
    previous: str | None = None
    for i in range(max(1, config.num_rounds * max(1, config.num_branches))):
        chunk = _add_state(graph, f"S_chunk_{i}", "document", _jitter(rng, config.unique_state_size * 2))
        op = _add_op(graph, f"O_refine_{i}", "llm", "iterative_refiner", criticality=1.0 + i * 0.1)
        inputs = ["S_task", "S_rubric", "S_global_context", chunk.state_id]
        if previous is not None:
            inputs.append(previous)
        _consume_many(graph, op.op_id, inputs)
        out = _add_output(graph, f"S_running_summary_{i}", "intermediate_summary", rng, config)
        graph.connect_operator_to_state(op.op_id, out.state_id)
        previous = out.state_id
    _finalize(graph)
    return graph


def generate_parallel_chains(config: WorkflowConfig) -> StateAccessGraph:
    rng = random.Random(config.seed)
    graph = _base_graph("parallel_chains", config, ("task", "document", "rubric", "tool_schema"))
    chain_tails = []
    for branch in range(max(1, config.num_branches)):
        prev: str | None = None
        branch_context = _add_state(
            graph,
            f"S_chain_{branch}_context",
            "unique_context",
            _jitter(rng, config.unique_state_size),
        )
        for round_idx in range(max(1, config.num_rounds)):
            op = _add_op(
                graph,
                f"O_chain_{branch}_{round_idx}",
                "llm",
                f"chain_{branch}_worker",
                criticality=1.0 + round_idx * 0.15,
            )
            inputs = ["S_task", "S_document", "S_rubric", branch_context.state_id]
            if prev is not None:
                inputs.append(prev)
            _consume_many(graph, op.op_id, inputs)
            out = _add_output(graph, f"S_chain_{branch}_{round_idx}_out", "intermediate_summary", rng, config)
            graph.connect_operator_to_state(op.op_id, out.state_id)
            prev = out.state_id
        if prev is not None:
            chain_tails.append(prev)
    writer = _add_op(graph, "O_writer", "llm", "writer", criticality=2.0)
    _consume_many(graph, writer.op_id, ["S_task", "S_rubric", *chain_tails])
    final = _add_output(graph, "S_parallel_report", "output", rng, config, deterministic=True)
    graph.connect_operator_to_state(writer.op_id, final.state_id)
    _finalize(graph)
    return graph


def generate_trading(config: WorkflowConfig) -> StateAccessGraph:
    rng = random.Random(config.seed)
    graph = _base_graph(
        "trading",
        config,
        ("task", "market_context", "risk_policy", "tool_schema", "role_instruction"),
    )
    decisions = []
    batch = max(1, config.batch_size, config.num_branches)
    for i in range(batch):
        signal = _add_state(graph, f"S_market_signal_{i}", "retrieval", _jitter(rng, config.unique_state_size))
        analyst = _add_op(graph, f"O_analyst_{i}", "llm", "analyst", criticality=1.1)
        research = _add_op(graph, f"O_researcher_{i}", "llm", "researcher", criticality=1.0)
        _consume_many(
            graph,
            analyst.op_id,
            ["S_task", "S_market_context", "S_role_instruction", signal.state_id],
        )
        _consume_many(
            graph,
            research.op_id,
            ["S_task", "S_market_context", "S_tool_schema", signal.state_id],
        )
        analyst_out = _add_output(graph, f"S_analyst_{i}_view", "intermediate_summary", rng, config)
        research_out = _add_output(graph, f"S_research_{i}_evidence", "retrieval_result", rng, config)
        graph.connect_operator_to_state(analyst.op_id, analyst_out.state_id)
        graph.connect_operator_to_state(research.op_id, research_out.state_id)

        trader = _add_op(graph, f"O_trader_{i}", "llm", "trader", criticality=1.5)
        _consume_many(
            graph,
            trader.op_id,
            ["S_task", "S_market_context", analyst_out.state_id, research_out.state_id],
        )
        trade = _add_output(graph, f"S_trade_{i}", "planner_output", rng, config)
        graph.connect_operator_to_state(trader.op_id, trade.state_id)

        risk = _add_op(graph, f"O_risk_{i}", "llm", "risk", criticality=1.7)
        _consume_many(graph, risk.op_id, ["S_task", "S_risk_policy", "S_market_context", trade.state_id])
        risk_out = _add_output(graph, f"S_risk_{i}", "critic_feedback", rng, config)
        graph.connect_operator_to_state(risk.op_id, risk_out.state_id)
        decisions.extend([trade.state_id, risk_out.state_id])

    manager = _add_op(graph, "O_manager", "llm", "manager", criticality=2.4)
    _consume_many(graph, manager.op_id, ["S_task", "S_market_context", "S_risk_policy", *decisions])
    final = _add_output(graph, "S_portfolio_decision", "output", rng, config, deterministic=True)
    graph.connect_operator_to_state(manager.op_id, final.state_id)
    _finalize(graph)
    return graph


def generate_software_dev(config: WorkflowConfig) -> StateAccessGraph:
    rng = random.Random(config.seed)
    graph = _base_graph(
        "software_dev",
        config,
        ("task", "repo_context", "coding_standards", "tool_schema", "test_plan"),
    )
    planner = _add_op(graph, "O_planner", "llm", "planner", criticality=1.8)
    _consume_many(graph, planner.op_id, ["S_task", "S_repo_context", "S_coding_standards"])
    plan = _add_output(graph, "S_plan", "planner_output", rng, config)
    graph.connect_operator_to_state(planner.op_id, plan.state_id)

    branch_outputs = []
    for i in range(max(1, config.num_branches)):
        task_slice = _add_state(graph, f"S_feature_slice_{i}", "unique_context", _jitter(rng, config.unique_state_size))
        coder = _add_op(graph, f"O_coder_{i}", "llm", "coder", criticality=1.2)
        _consume_many(
            graph,
            coder.op_id,
            ["S_task", "S_repo_context", "S_tool_schema", plan.state_id, task_slice.state_id],
        )
        code = _add_output(graph, f"S_code_{i}", "output", rng, config)
        graph.connect_operator_to_state(coder.op_id, code.state_id)

        tester = _add_op(graph, f"O_tester_{i}", "llm", "tester", criticality=1.0)
        _consume_many(graph, tester.op_id, ["S_task", "S_test_plan", "S_repo_context", code.state_id])
        test_result = _add_output(graph, f"S_test_result_{i}", "critic_feedback", rng, config)
        graph.connect_operator_to_state(tester.op_id, test_result.state_id)

        debugger = _add_op(graph, f"O_debugger_{i}", "llm", "debugger", criticality=1.5)
        _consume_many(
            graph,
            debugger.op_id,
            ["S_task", "S_repo_context", code.state_id, test_result.state_id],
        )
        fixed = _add_output(graph, f"S_fixed_code_{i}", "output", rng, config)
        graph.connect_operator_to_state(debugger.op_id, fixed.state_id)
        branch_outputs.append(fixed.state_id)

    reviewer = _add_op(graph, "O_reviewer", "llm", "reviewer", criticality=2.2)
    _consume_many(
        graph,
        reviewer.op_id,
        ["S_task", "S_repo_context", "S_coding_standards", *branch_outputs],
    )
    review = _add_output(graph, "S_reviewed_patch", "output", rng, config, deterministic=True)
    graph.connect_operator_to_state(reviewer.op_id, review.state_id)
    _finalize(graph)
    return graph


def _base_graph(name: str, config: WorkflowConfig, shared_kinds: tuple[str, ...]) -> StateAccessGraph:
    graph = StateAccessGraph(name, metadata={"config": asdict(config), "workflow": name})
    per_state = max(1, config.shared_state_size // max(1, len(shared_kinds)))
    for idx, kind in enumerate(shared_kinds):
        tokens = per_state + (config.shared_state_size % len(shared_kinds) if idx == 0 else 0)
        _add_state(graph, f"S_{kind}", kind, tokens, materialized_form="text")
    return graph


def _add_state(
    graph: StateAccessGraph,
    state_id: str,
    kind: str,
    token_size: int,
    kv_size_bytes: int | None = None,
    materialized_form: str = "inline",
    deterministic: bool = True,
    metadata: dict[str, object] | None = None,
) -> StateNode:
    state = StateNode(
        state_id=state_id,
        kind=kind,
        token_size=max(1, int(token_size)),
        kv_size_bytes=_kv_bytes(token_size) if kv_size_bytes is None else kv_size_bytes,
        materialized_form=materialized_form,
        deterministic=deterministic,
        metadata=metadata or {},
    )
    graph.add_state(state)
    return state


def _add_output(
    graph: StateAccessGraph,
    state_id: str,
    kind: str,
    rng: random.Random,
    config: WorkflowConfig,
    deterministic: bool = False,
) -> StateNode:
    dynamic = rng.random() < config.dynamic_hot_state_probability
    token_size = _jitter(rng, config.output_token_mean)
    return _add_state(
        graph,
        state_id,
        kind,
        token_size,
        materialized_form="output",
        deterministic=deterministic,
        metadata={"dynamic_hot_candidate": dynamic},
    )


def _add_op(
    graph: StateAccessGraph,
    op_id: str,
    kind: str,
    role: str,
    deterministic: bool = False,
    criticality: float = 1.0,
) -> OperatorNode:
    op = OperatorNode(
        op_id=op_id,
        kind=kind,
        role=role,
        deterministic=deterministic,
        criticality=criticality,
        phase_profile="prefill_heavy" if kind == "llm" else "mixed",
    )
    graph.add_operator(op)
    return op


def _consume_many(graph: StateAccessGraph, op_id: str, state_ids: list[str]) -> None:
    seen: set[str] = set()
    for state_id in state_ids:
        if state_id not in seen:
            graph.connect_state_to_operator(state_id, op_id)
            seen.add(state_id)


def _finalize(graph: StateAccessGraph) -> None:
    graph.update_operator_input_tokens()
    graph.compute_lifetimes()
    for state in graph.states.values():
        future_accesses = max(0, len(state.consumers))
        state.static_hotness = state.token_size * max(0, future_accesses - 1)
        if state.metadata.get("dynamic_hot_candidate"):
            state.dynamic_hotness = 0.0


def _jitter(rng: random.Random, mean: int) -> int:
    if mean <= 1:
        return 1
    return max(1, int(rng.gauss(mean, max(1, mean * 0.12))))


def _kv_bytes(tokens: int) -> int:
    return int(max(1, tokens) * 1024)
