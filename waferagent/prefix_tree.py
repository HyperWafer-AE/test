from __future__ import annotations

from dataclasses import dataclass, field

from waferagent.baselines import BaselineConfig
from waferagent.stage_ir import Stage


@dataclass
class PrefixComputeDecision:
    logical_shared_tokens: int
    shared_tokens_computed: int
    shared_tokens_saved: int
    private_tokens_computed: int
    full_input_tokens: int
    computed_input_tokens: int
    hit: bool


@dataclass
class PrefixComputeTracker:
    computed_prefixes: set[str] = field(default_factory=set)
    prefix_owner_job: dict[str, str] = field(default_factory=dict)
    logical_shared_prefix_tokens: int = 0
    unique_shared_prefix_tokens_computed: int = 0
    shared_prefill_tokens_saved: int = 0
    private_prefill_tokens_computed: int = 0
    prefix_compute_hits: int = 0
    prefix_compute_accesses: int = 0
    cross_job_prefix_compute_hits: int = 0
    cross_job_prefix_compute_accesses: int = 0

    def decide(self, stage: Stage, baseline: BaselineConfig) -> PrefixComputeDecision:
        shared = max(0, int(stage.shared_prefix_token_len))
        full = max(0, int(stage.input_tokens))
        private = max(0, full - shared)
        if stage.stage_type != "prefill" or not baseline.kv_sharing or shared <= 0 or not stage.shared_prefix_ids:
            self.private_prefill_tokens_computed += full
            return PrefixComputeDecision(0, 0, 0, full, full, full, False)

        prefix_id = stage.shared_prefix_ids[0]
        self.prefix_compute_accesses += 1
        self.logical_shared_prefix_tokens += shared
        if prefix_id in self.computed_prefixes:
            self.prefix_compute_hits += 1
            owner = self.prefix_owner_job.get(prefix_id)
            if owner and owner != stage.job_id:
                self.cross_job_prefix_compute_hits += 1
            if owner:
                self.cross_job_prefix_compute_accesses += 1
            self.shared_prefill_tokens_saved += shared
            self.private_prefill_tokens_computed += private
            return PrefixComputeDecision(shared, 0, shared, private, full, private, True)

        self.computed_prefixes.add(prefix_id)
        self.prefix_owner_job[prefix_id] = stage.job_id
        self.unique_shared_prefix_tokens_computed += shared
        self.private_prefill_tokens_computed += private
        return PrefixComputeDecision(shared, shared, 0, private, full, shared + private, False)

    def stats(self, saved_ms: float) -> dict[str, float]:
        return {
            "shared_prefill_compute_ms_saved": float(saved_ms),
            "shared_prefill_tokens_saved": float(self.shared_prefill_tokens_saved),
            "private_prefill_tokens_computed": float(self.private_prefill_tokens_computed),
            "unique_shared_prefix_tokens_computed": float(self.unique_shared_prefix_tokens_computed),
            "logical_shared_prefix_tokens": float(self.logical_shared_prefix_tokens),
            "prefix_compute_hit_rate": self.prefix_compute_hits / self.prefix_compute_accesses
            if self.prefix_compute_accesses
            else 0.0,
            "cross_job_prefix_compute_hits": float(self.cross_job_prefix_compute_hits),
            "cross_job_prefix_hit_rate": self.cross_job_prefix_compute_hits
            / self.cross_job_prefix_compute_accesses
            if self.cross_job_prefix_compute_accesses
            else 0.0,
        }
