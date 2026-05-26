"""State materialization policy decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .ir import StateNode


@dataclass(frozen=True)
class PolicyConfig:
    text_cache_token_threshold: int = 512
    small_state_token_threshold: int = 2048
    large_state_token_threshold: int = 8192
    hotness_threshold: float = 2500.0
    kv_cache_multiplier: float = 1.35
    storage_cost_per_byte: float = 0.00001
    movement_cost_per_byte_hop: float = 0.00002
    memory_pressure_high: float = 0.8
    memory_pressure_critical: float = 0.95


@dataclass(frozen=True)
class PolicyDecision:
    state_id: str
    policy: str
    expected_saved_cost: float
    materialization_cost: float
    storage_cost: float
    movement_cost: float
    contention_cost: float
    reason: str

    def to_row(self) -> dict[str, object]:
        return asdict(self)


def decide_state_policy(
    state: StateNode,
    consumer_count: int | None = None,
    hotness: float | None = None,
    memory_pressure: float = 0.0,
    average_hops_to_consumers: float = 1.0,
    deterministic_operator: bool = False,
    config: PolicyConfig | None = None,
) -> PolicyDecision:
    cfg = config or PolicyConfig()
    fanout = len(state.consumers) if consumer_count is None else consumer_count
    state_hotness = max(state.static_hotness, state.dynamic_hotness) if hotness is None else hotness
    duplicate_accesses = max(0, fanout - 1)
    materialization_cost = state.text_size_bytes * max(1, fanout) * 0.001
    storage_cost = state.materialized_size_bytes * cfg.storage_cost_per_byte
    movement_cost = state.materialized_size_bytes * average_hops_to_consumers * cfg.movement_cost_per_byte_hop
    contention_cost = movement_cost * min(2.0, fanout / 4.0)
    expected_saved = state.text_size_bytes * duplicate_accesses * 0.001 + state_hotness

    if fanout <= 1 or state_hotness < cfg.hotness_threshold * 0.25:
        if deterministic_operator and state.kind == "output" and memory_pressure >= cfg.memory_pressure_high:
            return _decision(
                state,
                "recompute",
                expected_saved,
                materialization_cost,
                storage_cost,
                movement_cost,
                contention_cost,
                "low reuse deterministic output under memory pressure",
            )
        return _decision(
            state,
            "inline",
            expected_saved,
            materialization_cost,
            storage_cost,
            movement_cost,
            contention_cost,
            "low fanout or low hotness",
        )

    total_cost = materialization_cost + storage_cost + movement_cost + contention_cost
    if expected_saved <= total_cost and memory_pressure >= cfg.memory_pressure_high:
        return _decision(
            state,
            "evict",
            expected_saved,
            materialization_cost,
            storage_cost,
            movement_cost,
            contention_cost,
            "saved cost does not justify storage under memory pressure",
        )

    if memory_pressure >= cfg.memory_pressure_critical:
        return _decision(
            state,
            "shard" if state.token_size >= cfg.large_state_token_threshold else "pin",
            expected_saved,
            materialization_cost,
            storage_cost,
            movement_cost,
            contention_cost,
            "critical memory pressure prevents replication",
        )

    if state.token_size <= cfg.text_cache_token_threshold and fanout >= 2:
        return _decision(
            state,
            "cache_text",
            expected_saved,
            materialization_cost,
            storage_cost,
            movement_cost,
            contention_cost,
            "small reusable text state",
        )

    if state.token_size <= cfg.small_state_token_threshold and fanout >= 3 and memory_pressure < cfg.memory_pressure_high:
        return _decision(
            state,
            "replicate",
            expected_saved,
            materialization_cost,
            storage_cost,
            movement_cost,
            contention_cost,
            "small hot state with broad fanout",
        )

    if state.token_size >= cfg.large_state_token_threshold and fanout >= 2:
        return _decision(
            state,
            "shard",
            expected_saved,
            materialization_cost,
            storage_cost,
            movement_cost,
            contention_cost,
            "large hot state should avoid full replication",
        )

    if state.materialized_form in {"inline", "text", "output"} and state.kv_size_bytes > state.text_size_bytes:
        return _decision(
            state,
            "cache_kv",
            expected_saved * cfg.kv_cache_multiplier,
            materialization_cost,
            storage_cost,
            movement_cost,
            contention_cost,
            "prefill-heavy reusable state can be represented as compatible KV",
        )

    return _decision(
        state,
        "pin",
        expected_saved,
        materialization_cost,
        storage_cost,
        movement_cost,
        contention_cost,
        "moderately hot state benefits from stable locality",
    )


def _decision(
    state: StateNode,
    policy: str,
    expected_saved_cost: float,
    materialization_cost: float,
    storage_cost: float,
    movement_cost: float,
    contention_cost: float,
    reason: str,
) -> PolicyDecision:
    return PolicyDecision(
        state_id=state.state_id,
        policy=policy,
        expected_saved_cost=expected_saved_cost,
        materialization_cost=materialization_cost,
        storage_cost=storage_cost,
        movement_cost=movement_cost,
        contention_cost=contention_cost,
        reason=reason,
    )


def decide_policies(
    states: list[StateNode],
    memory_pressure: float = 0.0,
    config: PolicyConfig | None = None,
) -> list[PolicyDecision]:
    return [
        decide_state_policy(state, memory_pressure=memory_pressure, config=config)
        for state in states
    ]
