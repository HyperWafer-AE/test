from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StagingDecision:
    prefix_id: str
    source_region: str
    staging_region: str
    bytes: int
    start_ms: float
    release_ms: float
    expected_saved_remote_reads: int
    staging_cost_ms: float
    accepted: bool
    reason: str

    def to_dict(self) -> dict:
        return {
            "prefix_id": self.prefix_id,
            "source_region": self.source_region,
            "staging_region": self.staging_region,
            "bytes": int(self.bytes),
            "start_ms": float(self.start_ms),
            "release_ms": float(self.release_ms),
            "expected_saved_remote_reads": int(self.expected_saved_remote_reads),
            "staging_cost_ms": float(self.staging_cost_ms),
            "accepted": bool(self.accepted),
            "reason": self.reason,
        }


def decide_transient_staging(
    prefix_id: str,
    prefix_bytes: int,
    cohort_size: int,
    shared_prefix_tokens: int,
    source_region: str,
    staging_region: str,
    start_ms: float,
    duration_ms: float,
    bytes_per_ms: float,
    sram_slack_bytes: int,
    threshold_tokens: int = 1024,
) -> StagingDecision:
    if cohort_size < 2:
        reason = "cohort_too_small"
        accepted = False
    elif shared_prefix_tokens < threshold_tokens:
        reason = "prefix_too_short"
        accepted = False
    elif prefix_bytes > sram_slack_bytes:
        reason = "insufficient_sram_slack"
        accepted = False
    else:
        staging_cost = prefix_bytes / max(1.0, bytes_per_ms)
        saved_remote_reads = max(0, cohort_size - 1)
        saved_ms = saved_remote_reads * prefix_bytes / max(1.0, bytes_per_ms)
        accepted = saved_ms > staging_cost
        reason = "accepted" if accepted else "not_amortized"
    return StagingDecision(
        prefix_id=prefix_id,
        source_region=source_region,
        staging_region=staging_region,
        bytes=prefix_bytes,
        start_ms=start_ms,
        release_ms=start_ms + duration_ms,
        expected_saved_remote_reads=max(0, cohort_size - 1),
        staging_cost_ms=prefix_bytes / max(1.0, bytes_per_ms),
        accepted=accepted,
        reason=reason,
    )

