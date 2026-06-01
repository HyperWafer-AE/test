from dataclasses import asdict, dataclass


@dataclass
class AgentMetrics:
    total_input_tokens: int = 0
    append_tokens: int = 0
    effective_prefill_tokens: int = 0
    decode_tokens: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    kv_migration_bytes: int = 0
    evicted_kv_bytes: int = 0
    tool_wait_cycles: int = 0
    num_llm_steps: int = 0
    num_tool_steps: int = 0

    @property
    def cache_hit_ratio(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total else 0.0

    @property
    def effective_prefill_reduction(self) -> float:
        baseline = self.total_input_tokens + self.append_tokens
        if baseline == 0:
            return 0.0
        return 1.0 - (self.effective_prefill_tokens / baseline)

    @property
    def decode_to_prefill_ratio(self) -> float:
        if self.effective_prefill_tokens == 0:
            return float("inf") if self.decode_tokens else 0.0
        return self.decode_tokens / self.effective_prefill_tokens

    def to_dict(self) -> dict:
        data = asdict(self)
        data["cache_hit_ratio"] = self.cache_hit_ratio
        data["effective_prefill_reduction"] = self.effective_prefill_reduction
        data["decode_to_prefill_ratio"] = self.decode_to_prefill_ratio
        return data

