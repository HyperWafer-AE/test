# WaferAgent Round 10 Paper-Facing Artifacts

All wafer numbers are trace-driven wafer-scale simulator results, not real wafer hardware measurements.

- export_commit: `a42177d10f933389920202a6b62f3971b4bbb880`
- shared_attention_accounting_main_mode: `cohort_stage`
- paper_writing_allowed: `False`

## Claim Gate
```json
{
  "artifact_ready": {
    "figures_exported": true,
    "pass": true,
    "tables_exported": true
  },
  "claim_ready": {
    "affinity_placement_supported": true,
    "latency_safe_cohort_supported_or_traffic_only_demoted": true,
    "replication_demoted": true
  },
  "demoted": {
    "critical_path_scheduling": true,
    "dynamic_pd_partition": true,
    "persistent_shared_kv_replication": true,
    "tool_ttl": true
  },
  "evidence_ready": {
    "existing_prefix_cache_gap_supported": true,
    "hf_mini_trace_completed": true,
    "hf_or_vllm_mini_trace_completed_or_formally_missing_with_timeout_logs": true,
    "regime_map_has_non_low_reuse_beneficial_region": false,
    "vllm_mini_trace_completed": true
  },
  "method_ready": {
    "main_sim_uses_cohort_stage_or_conservative_accounting": true,
    "shared_attention_accounting_main_mode": "cohort_stage",
    "shared_attention_fit_validated": true
  },
  "paper_writing_allowed": false
}
```
