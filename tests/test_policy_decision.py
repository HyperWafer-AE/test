from waferstateflow.ir import StateNode
from waferstateflow.state_policy import PolicyConfig, decide_state_policy


def _state(token_size, consumers, hotness):
    state = StateNode("S", "doc", token_size=token_size)
    state.consumers = [f"O_{i}" for i in range(consumers)]
    state.static_hotness = hotness
    return state


def test_small_hot_state_replicates():
    decision = decide_state_policy(_state(1000, 6, 8000), memory_pressure=0.2)
    assert decision.policy == "replicate"


def test_large_hot_state_shards():
    decision = decide_state_policy(_state(12000, 4, 50000), memory_pressure=0.2)
    assert decision.policy == "shard"


def test_low_reuse_state_stays_inline():
    decision = decide_state_policy(_state(1000, 1, 100), memory_pressure=0.1)
    assert decision.policy == "inline"


def test_memory_pressure_prevents_blind_replication():
    cfg = PolicyConfig(memory_pressure_critical=0.9)
    decision = decide_state_policy(_state(1000, 8, 12000), memory_pressure=0.96, config=cfg)
    assert decision.policy in {"pin", "shard"}


def test_cache_kv_requires_prefix_compatibility():
    state = _state(3000, 2, 10000)
    state.materialized_form = "text"
    state.kv_size_bytes = state.token_size * 1024
    state.kv_cacheable = True
    state.prefix_compatible = True
    assert decide_state_policy(state).policy == "cache_kv"


def test_non_prefix_intermediate_output_cannot_cache_kv():
    state = _state(3000, 2, 10000)
    state.kind = "output"
    state.materialized_form = "output"
    state.kv_size_bytes = state.token_size * 1024
    state.kv_cacheable = True
    state.prefix_compatible = False
    assert decide_state_policy(state).policy != "cache_kv"
