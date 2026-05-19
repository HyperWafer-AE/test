# WaferAgent Round 9 Paper Artifacts

All wafer numbers in this bundle are trace-driven wafer-scale simulator results, not real wafer hardware measurements.

## Artifact Export Provenance

- artifact_export_commit: `e8acd20a7b5c6cbc3f2e9b83209346d5eb5bb5aa`
- artifact_export_command: `python scripts/export_paper_artifacts.py --source-results results/round9_global_main_h100sharedfit,results/round9_existing_cache_gap,results/round9_ablation,results/round9_cohort_policy_comparison,results/round9_cohort_admission,results/round9_decode_cohort_sweep,results/round9_replication_tradeoff,results/round9_prefix_realism_sensitivity,results/round9_shared_attention_microbench_h100,results/round9_shared_attention_fit,results/round9_characterization_h100_hf_20jobs,results/round9_characterization_h100_vllm_20jobs --out results/round9_paper_artifacts`
- oracle semantics: `ideal_next_use_cache` is a cache upper-bound style baseline, not a full-system oracle upper bound.
- replication: benefit-cost replication is demoted unless the numeric claim matrix reports a non-zero supported delta.

## Source Runs

| artifact_file | source_result_dir | source_git_commit | source_duration_source | source_arrival_mode | source_arrival_rates |
| --- | --- | --- | --- | --- | --- |
| global_simulation_summary.csv | results/round9_global_main_h100sharedfit | e1fda3377365586db6adf13d97fb4fb3152574d2 | synthetic | poisson | 2,4,8,16 |
| global_job_metrics_sample.csv | results/round9_global_main_h100sharedfit | e1fda3377365586db6adf13d97fb4fb3152574d2 | synthetic | poisson | 2,4,8,16 |
| slo_goodput.csv | results/round9_global_main_h100sharedfit | e1fda3377365586db6adf13d97fb4fb3152574d2 | synthetic | poisson | 2,4,8,16 |
| existing_cache_gap_summary.csv | results/round9_existing_cache_gap | e1fda3377365586db6adf13d97fb4fb3152574d2 | synthetic |  |  |
| ablation_global_summary.csv | results/round9_ablation | e1fda3377365586db6adf13d97fb4fb3152574d2 | synthetic | poisson | 4,8,16 |
| ablation_delta_summary.csv | results/round9_ablation | e1fda3377365586db6adf13d97fb4fb3152574d2 | synthetic | poisson | 4,8,16 |
| cohort_admission_summary.csv | results/round9_cohort_admission | e1fda3377365586db6adf13d97fb4fb3152574d2 | synthetic | burst | 16 |
| cohort_admission_decisions.csv | results/round9_cohort_admission | e1fda3377365586db6adf13d97fb4fb3152574d2 | synthetic | burst | 16 |
| cohort_policy_comparison.csv | results/round9_cohort_policy_comparison | e1fda3377365586db6adf13d97fb4fb3152574d2 | synthetic | poisson | 2,4,8,16 |
| decode_cohorts_event_driven.csv | results/round9_cohort_admission | e1fda3377365586db6adf13d97fb4fb3152574d2 | synthetic | burst | 16 |
| decode_cohort_analytical_sweep.csv | results/round9_decode_cohort_sweep | e1fda3377365586db6adf13d97fb4fb3152574d2 |  |  |  |
| replication_tradeoff_summary.csv | results/round9_replication_tradeoff | e1fda3377365586db6adf13d97fb4fb3152574d2 |  |  |  |
| prefix_realism_sensitivity.csv | results/round9_prefix_realism_sensitivity | e1fda3377365586db6adf13d97fb4fb3152574d2 | synthetic |  |  |
| prefix_realism_prefix_stats.csv | results/round9_prefix_realism_sensitivity | e1fda3377365586db6adf13d97fb4fb3152574d2 | synthetic |  |  |
| regime_classification.csv | results/round9_prefix_realism_sensitivity | e1fda3377365586db6adf13d97fb4fb3152574d2 | synthetic |  |  |
| planning_overhead_summary.csv | results/round9_global_main_h100sharedfit | e1fda3377365586db6adf13d97fb4fb3152574d2 | synthetic | poisson | 2,4,8,16 |
| shared_attention_microbench_summary.csv | results/round9_shared_attention_microbench_h100 | c208041d0ec9aed8c0cf2dbf8d2c0ff263372d04 |  |  |  |
| shared_attention_microbench_raw_sample.csv | results/round9_shared_attention_microbench_h100 | c208041d0ec9aed8c0cf2dbf8d2c0ff263372d04 |  |  |  |
| shared_attention_cost_fit.json | results/round9_shared_attention_fit | c208041d0ec9aed8c0cf2dbf8d2c0ff263372d04 |  |  |  |
| shared_attention_fit_quality.json | results/round9_shared_attention_fit | c208041d0ec9aed8c0cf2dbf8d2c0ff263372d04 |  |  |  |

## Readiness

```json
{
  "artifact_ready": {
    "artifact_tables_exported": true,
    "pass": true,
    "report_title_round9": true,
    "shared_attention_fit_exported": true,
    "source_manifest_arrival_fields": true
  },
  "claim_ready": {
    "affinity_placement_supported": true,
    "cohort_latency_improvement": true,
    "decode_cohort_traffic_reduction": true,
    "existing_prefix_cache_gap": true,
    "global_tail_latency_vs_apc": true,
    "h100_shared_attention_fit_used": true,
    "pass": true,
    "prefix_regime_classification_present": true,
    "replication_headline_claim": false
  },
  "cohort_latency_improvement_claim_allowed": true,
  "cohort_traffic_only_claim": false,
  "demoted": {
    "aggregator_placement_headline_claim": true,
    "critical_path_scheduling": true,
    "dynamic_pd_partition": true,
    "replication_headline_claim": true,
    "tool_ttl": true
  },
  "hf_trace_status": "explicitly_missing",
  "oracle_renamed_not_upper_bound": true,
  "paper_ready": {
    "ablation_artifact_not_identical_to_main": true,
    "ablation_delta_summary_present": true,
    "analytical_cohort_not_used_as_main_evidence": true,
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
    "shared_attention_fit_drives_main_simulator": true,
    "shared_attention_microbench_exported": true
  },
  "pass": {
    "artifact_ready": true,
    "claim_ready": true,
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
| Existing prefix cache gap | supported | decode_shared_kv_read_bytes | apc_like | 28249288395863.0 | 35540854374400.0 | -7291565978537.0 | -0.2051601208492359 | -0.05 | existing_cache_gap_summary.csv | Fig2 | nan |
| Global serving tail latency | supported | jct_p99_ms | apc_like | 5503.162522199244 | 29358.29083459689 | -23855.128312397646 | -0.8125516722617205 | -0.05 | global_simulation_summary.csv | Fig4 | nan |
| Decode cohort traffic reduction | supported | decode_shared_kv_read_bytes | no_shared_kv_decode_cohort | 27145052304032.0 | 35540854374400.0 | -8395802070368.0 | -0.236229607254897 | 0.05 | ablation_delta_summary.csv | Fig7 | nan |
| Cost-aware cohort latency safety | supported | jct_p99_delta_pct_vs_no_cohort | no_shared_kv_decode_cohort | -0.0187042807397592 | 0.0 | -0.0187042807397592 | -0.0187042807397592 | 0.05 | cohort_admission_summary.csv | Fig7 | Supported only when byte savings do not regress p99 JCT by more than 5%. |
| Shared-KV replication | demoted | mesh_traffic_bytes | no_replication | 9959223853056.0 | 9998683865088.0 | -39460012032.0 | -0.0039465206185567 | -0.05 | replication_tradeoff_summary.csv | Appendix | Benefit-cost replication is not a headline claim unless it beats no_replication. |
| Oracle semantics | demoted | nan | oracle | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | report.json | Limitations | Renamed to ideal_next_use_cache; not claimed as full-system upper bound. |
| H100 shared-attention cost fit drives simulator | supported | shared_attention_cost_model_source | global_main | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 | global_simulation_summary.csv | Fig8 | Supported only when the main global simulation reports h100_microbench_fit and a non-empty fit hash. |
