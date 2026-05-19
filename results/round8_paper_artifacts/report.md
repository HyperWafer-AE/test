# WaferAgent Round 7 Paper Artifacts

All wafer numbers in this bundle are trace-driven wafer-scale simulator results, not real wafer hardware measurements.

## Artifact Export Provenance

- artifact_export_commit: `b0854cf9c00d8affd8a1bd3c8ffaca27b8a61162`
- artifact_export_command: `python scripts/export_paper_artifacts.py --source-results results/round8_global_main_neutral,results/round8_existing_cache_gap,results/round8_ablation,results/round8_cohort_admission,results/round8_decode_cohort_sweep,results/round8_replication_tradeoff,results/round8_prefix_realism_sensitivity,results/round8_shared_attention_microbench_h100 --out results/round8_paper_artifacts`
- oracle semantics: `ideal_next_use_cache` is a cache upper-bound style baseline, not a full-system oracle upper bound.
- replication: benefit-cost replication is demoted unless the numeric claim matrix reports a non-zero supported delta.

## Source Runs

| artifact_file | source_result_dir | source_git_commit | source_duration_source | source_arrival_mode | source_arrival_rates |
| --- | --- | --- | --- | --- | --- |
| global_simulation_summary.csv | results/round8_global_main_neutral | ba5bd02c448e60b6e0894ba6e36f0928c8d40b5f | synthetic |  |  |
| global_job_metrics_sample.csv | results/round8_global_main_neutral | ba5bd02c448e60b6e0894ba6e36f0928c8d40b5f | synthetic |  |  |
| slo_goodput.csv | results/round8_global_main_neutral | ba5bd02c448e60b6e0894ba6e36f0928c8d40b5f | synthetic |  |  |
| existing_cache_gap_summary.csv | results/round8_existing_cache_gap | ba5bd02c448e60b6e0894ba6e36f0928c8d40b5f | synthetic |  |  |
| ablation_global_summary.csv | results/round8_ablation | ba5bd02c448e60b6e0894ba6e36f0928c8d40b5f | synthetic |  |  |
| ablation_delta_summary.csv | results/round8_ablation | ba5bd02c448e60b6e0894ba6e36f0928c8d40b5f | synthetic |  |  |
| cohort_admission_summary.csv | results/round8_cohort_admission | ba5bd02c448e60b6e0894ba6e36f0928c8d40b5f | synthetic |  |  |
| cohort_admission_decisions.csv | results/round8_cohort_admission | ba5bd02c448e60b6e0894ba6e36f0928c8d40b5f | synthetic |  |  |
| decode_cohorts_event_driven.csv | results/round8_cohort_admission | ba5bd02c448e60b6e0894ba6e36f0928c8d40b5f | synthetic |  |  |
| decode_cohort_analytical_sweep.csv | results/round8_decode_cohort_sweep | ba5bd02c448e60b6e0894ba6e36f0928c8d40b5f |  |  |  |
| replication_tradeoff_summary.csv | results/round8_replication_tradeoff | ba5bd02c448e60b6e0894ba6e36f0928c8d40b5f |  |  |  |
| prefix_realism_sensitivity.csv | results/round8_prefix_realism_sensitivity | ba5bd02c448e60b6e0894ba6e36f0928c8d40b5f | synthetic |  |  |
| prefix_realism_prefix_stats.csv | results/round8_prefix_realism_sensitivity | ba5bd02c448e60b6e0894ba6e36f0928c8d40b5f | synthetic |  |  |
| planning_overhead_summary.csv | results/round8_global_main_neutral | ba5bd02c448e60b6e0894ba6e36f0928c8d40b5f | synthetic |  |  |
| shared_attention_microbench_summary.csv | results/round8_shared_attention_microbench_h100 | ba5bd02c448e60b6e0894ba6e36f0928c8d40b5f |  |  |  |
| shared_attention_microbench_raw_sample.csv | results/round8_shared_attention_microbench_h100 | ba5bd02c448e60b6e0894ba6e36f0928c8d40b5f |  |  |  |

## Readiness

```json
{
  "demoted": {
    "aggregator_placement_headline_claim": true,
    "critical_path_scheduling": true,
    "dynamic_pd_partition": true,
    "replication_headline_claim": true,
    "tool_ttl": true
  },
  "hf_trace_status": "missing",
  "oracle_renamed_not_upper_bound": true,
  "paper_ready": {
    "ablation_artifact_not_identical_to_main": true,
    "ablation_delta_summary_present": true,
    "analytical_cohort_not_used_as_main_evidence": true,
    "artifact_tables_exported": true,
    "benefit_cost_replication_nonzero_or_demoted": true,
    "cohort_latency_safe_or_traffic_only": true,
    "cost_aware_cohort_admission_recorded": true,
    "event_driven_cohort_artifact_exported": true,
    "existing_cache_gap_units_correct": true,
    "global_serving_results_present": true,
    "oracle_monotonic_upper_bound_or_renamed": true,
    "oracle_renamed_not_upper_bound": true,
    "planning_overhead_recorded": true,
    "prefix_realism_exported": true,
    "shared_attention_microbench_exported": true
  },
  "pass": {
    "overall": true,
    "paper_ready": true,
    "sanity": true
  },
  "replication_headline_claim": false,
  "sanity": {
    "neutral_default": true,
    "no_semantic_fallback_in_export": true,
    "no_silent_fallback": true,
    "wafer_results_marked_simulation": true
  },
  "vllm_trace_status": "explicitly_missing"
}
```

## Paper-Ready Claim Matrix

| claim | status | primary_metric | baseline | waferagent_value | comparison_value | delta | delta_pct | threshold | evidence_file | figure_id | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Existing prefix cache gap | supported | decode_shared_kv_read_bytes | apc_like | 27395448897391.0 | 35540854374400.0 | -8145405477009.0 | -0.2291842900342913 | -0.05 | existing_cache_gap_summary.csv | Fig2 | nan |
| Global serving tail latency | supported | jct_p99_ms | apc_like | 13103.759480653272 | 33477.3995576944 | -20373.640077041127 | -0.6085789322414225 | -0.05 | global_simulation_summary.csv | Fig4 | nan |
| Decode cohort traffic reduction | supported | decode_shared_kv_read_bytes | no_shared_kv_decode_cohort | 28329604284274.0 | 35540854374400.0 | -7211250090126.0 | -0.202900302118799 | 0.05 | ablation_delta_summary.csv | Fig7 | nan |
| Cost-aware cohort latency safety | partial | jct_p99_delta_pct_vs_no_cohort | no_shared_kv_decode_cohort | 0.1135934095607177 | 0.0 | 0.1135934095607177 | 0.1135934095607177 | 0.05 | cohort_admission_summary.csv | Fig7 | Supported only when byte savings do not regress p99 JCT by more than 5%. |
| Shared-KV replication | demoted | mesh_traffic_bytes | no_replication | 9959223853056.0 | 9998683865088.0 | -39460012032.0 | -0.0039465206185567 | -0.05 | replication_tradeoff_summary.csv | Appendix | Benefit-cost replication is not a headline claim unless it beats no_replication. |
| Oracle semantics | demoted | nan | oracle | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | report.json | Limitations | Renamed to ideal_next_use_cache; not claimed as full-system upper bound. |
