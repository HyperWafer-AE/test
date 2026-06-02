from dataclasses import asdict, dataclass


@dataclass
class AgentMetrics:
    total_input_tokens: int = 0
    append_tokens: int = 0
    effective_prefill_tokens: int = 0
    decode_tokens: int = 0
    llm_output_tokens: int = 0
    tool_output_tokens: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_byte_miss: int = 0
    kv_migration_bytes: int = 0
    demand_migration_bytes: int = 0
    prefetch_migration_bytes: int = 0
    num_kv_migrations: int = 0
    num_demand_migrations: int = 0
    num_prefetch_migrations: int = 0
    num_prefetch_events: int = 0
    prefetch_kv_bytes: int = 0
    prefetch_hidden_cycles: int = 0
    prefetch_exposed_cycles: int = 0
    prefetch_total_cycles: int = 0
    evicted_kv_bytes: int = 0
    tool_wait_cycles: int = 0
    local_state_hits: int = 0
    remote_state_hits: int = 0
    state_misses: int = 0
    num_remote_accesses: int = 0
    remote_access_bytes: int = 0
    remote_read_bytes: int = 0
    num_remote_reads: int = 0
    remote_read_cycles: int = 0
    migration_skipped_by_cost: int = 0
    num_action_local: int = 0
    num_action_remote_read: int = 0
    num_action_migrate: int = 0
    num_action_replicate: int = 0
    num_action_static_hit: int = 0
    static_replica_bytes: int = 0
    num_static_replicas: int = 0
    unused_prefetch_events: int = 0
    migration_cost_estimate_cycles: int = 0
    remote_read_cost_estimate_cycles: int = 0
    external_wait_cycles: int = 0
    model_compute_cycles: int = 0
    model_comm_cycles: int = 0
    exposed_migration_cycles: int = 0
    exposed_prefetch_cycles: int = 0
    approximate_non_wait_cycles: int = 0
    llm_side_cycles: int = 0
    compressed_observation_tokens_saved: int = 0
    num_compressed_observations: int = 0
    num_agents: int = 0
    num_states_resident_final: int = 0
    num_states_total: int = 0
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

    @property
    def tool_wait_overlap_ratio(self) -> float:
        total = self.prefetch_hidden_cycles + self.prefetch_exposed_cycles
        return self.prefetch_hidden_cycles / total if total else 0.0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["cache_hit_ratio"] = self.cache_hit_ratio
        data["effective_prefill_reduction"] = self.effective_prefill_reduction
        data["decode_to_prefill_ratio"] = self.decode_to_prefill_ratio
        data["tool_wait_overlap_ratio"] = self.tool_wait_overlap_ratio
        return data


def postprocess_agent_events(
    events_dict,
    metrics: AgentMetrics,
    total_cycles: int = None,
    pure_comp_cycles: int = None,
    pure_comm_cycles: int = None,
):
    wait_events = {
        event.event_tag: event
        for event in events_dict.values()
        if getattr(event, "event_type", None) == "external_wait"
    }
    metrics.prefetch_hidden_cycles = 0
    metrics.prefetch_exposed_cycles = 0
    metrics.prefetch_total_cycles = 0
    metrics.external_wait_cycles = 0
    metrics.exposed_migration_cycles = 0

    for event in events_dict.values():
        if getattr(event, "event_type", None) == "external_wait":
            metrics.external_wait_cycles += max(0, int(event.end_time) - int(event.start_time))
            continue
        if getattr(event, "event_type", None) != "communication":
            continue
        metadata = getattr(event, "metadata", {}) or {}
        if metadata.get("reason") == "remote_read":
            metrics.remote_read_cycles += max(0, int(event.end_time) - int(event.start_time))
        if metadata.get("reason") in {"demand", "replicate"}:
            metrics.exposed_migration_cycles += max(0, int(event.end_time) - int(event.start_time))
        if metadata.get("reason") != "prefetch":
            continue
        duration = max(0, int(event.end_time) - int(event.start_time))
        wait = wait_events.get(metadata.get("associated_wait_tag"))
        hidden = 0
        if wait is not None:
            overlap_start = max(int(event.start_time), int(wait.start_time))
            overlap_end = min(int(event.end_time), int(wait.end_time))
            hidden = max(0, overlap_end - overlap_start)
        exposed = max(0, duration - hidden)
        metrics.prefetch_hidden_cycles += hidden
        metrics.prefetch_exposed_cycles += exposed
        metrics.prefetch_total_cycles += duration
    metrics.exposed_prefetch_cycles = metrics.prefetch_exposed_cycles
    if pure_comp_cycles is not None:
        metrics.model_compute_cycles = int(pure_comp_cycles)
    if pure_comm_cycles is not None:
        metrics.model_comm_cycles = int(pure_comm_cycles)
    if total_cycles is not None:
        metrics.approximate_non_wait_cycles = max(0, int(total_cycles) - metrics.external_wait_cycles)
        metrics.llm_side_cycles = metrics.approximate_non_wait_cycles
