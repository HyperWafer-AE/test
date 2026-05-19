# WaferAgent Round 7 Paper Artifacts

All wafer numbers in this bundle are trace-driven wafer-scale simulator results, not real wafer hardware measurements.

## Artifact Export Provenance

- artifact_export_commit: `72f97912e294fc12470bbbf2862b503d0349ef58`
- artifact_export_command: `python scripts/export_paper_artifacts.py --source-results results/round7_global_main_neutral,results/round7_existing_cache_gap,results/round7_ablation,results/round7_decode_cohort_targeted,results/round7_decode_cohort_sweep,results/round7_replication_tradeoff,results/round7_prefix_realism_sensitivity,results/round7_shared_kv_microbench_h100 --out results/round7_paper_artifacts`
- oracle semantics: `ideal_next_use_cache` is a cache upper-bound style baseline, not a full-system oracle upper bound.
- replication: benefit-cost replication is demoted unless the numeric claim matrix reports a non-zero supported delta.

## Source Runs

| artifact_file | source_result_dir | source_git_commit | source_duration_source | source_arrival_mode | source_arrival_rates |
| --- | --- | --- | --- | --- | --- |
| global_simulation_summary.csv | results/round7_global_main_neutral | 455d3234c48daeab0cfd924549dc385568e0205e | synthetic |  |  |
| global_job_metrics_sample.csv | results/round7_global_main_neutral | 455d3234c48daeab0cfd924549dc385568e0205e | synthetic |  |  |
| slo_goodput.csv | results/round7_global_main_neutral | 455d3234c48daeab0cfd924549dc385568e0205e | synthetic |  |  |
| existing_cache_gap_summary.csv | results/round7_existing_cache_gap | 455d3234c48daeab0cfd924549dc385568e0205e | synthetic |  |  |
| ablation_global_summary.csv | results/round7_ablation | 72f97912e294fc12470bbbf2862b503d0349ef58 | synthetic |  |  |
| ablation_delta_summary.csv | results/round7_ablation | 72f97912e294fc12470bbbf2862b503d0349ef58 | synthetic |  |  |
| decode_cohorts_event_driven.csv | results/round7_decode_cohort_targeted | 455d3234c48daeab0cfd924549dc385568e0205e | synthetic |  |  |
| decode_cohort_analytical_sweep.csv | results/round7_decode_cohort_sweep | 455d3234c48daeab0cfd924549dc385568e0205e |  |  |  |
| replication_tradeoff_summary.csv | results/round7_replication_tradeoff | 455d3234c48daeab0cfd924549dc385568e0205e |  |  |  |
| prefix_realism_sensitivity.csv | results/round7_prefix_realism_sensitivity | 455d3234c48daeab0cfd924549dc385568e0205e | synthetic |  |  |
| prefix_realism_prefix_stats.csv | results/round7_prefix_realism_sensitivity | 455d3234c48daeab0cfd924549dc385568e0205e | synthetic |  |  |
| planning_overhead_summary.csv | results/round7_global_main_neutral | 455d3234c48daeab0cfd924549dc385568e0205e | synthetic |  |  |
| shared_kv_microbench_summary.csv | results/round7_shared_kv_microbench_h100 | 455d3234c48daeab0cfd924549dc385568e0205e |  |  |  |
| shared_kv_microbench_raw_sample.csv | results/round7_shared_kv_microbench_h100 | 455d3234c48daeab0cfd924549dc385568e0205e |  |  |  |

## Readiness

```json
{
  "demoted": {
    "critical_path_scheduling": true,
    "dynamic_pd_partition": true,
    "replication_headline_claim": true,
    "tool_ttl": true
  },
  "oracle_renamed_not_upper_bound": true,
  "paper_ready": {
    "ablation_artifact_not_identical_to_main": true,
    "ablation_delta_summary_present": true,
    "analytical_cohort_not_used_as_main_evidence": true,
    "artifact_tables_exported": true,
    "benefit_cost_replication_nonzero_or_demoted": true,
    "event_driven_cohort_artifact_exported": true,
    "existing_cache_gap_units_correct": true,
    "global_serving_results_present": true,
    "oracle_monotonic_upper_bound_or_renamed": true,
    "oracle_renamed_not_upper_bound": true,
    "planning_overhead_recorded": true,
    "prefix_realism_exported": true,
    "shared_kv_microbench_exported": true
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
  }
}
```

## Paper-Ready Claim Matrix

| claim | status | primary_metric | baseline | waferagent_value | comparison_value | delta | threshold | evidence_file | paper_section | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Existing prefix cache gap | supported | decode_shared_kv_read_bytes | apc_like | 15357299811897.0 | 35540854374400.0 | -20183554562503.0 | -0.05 | existing_cache_gap_summary.csv | Existing prefix-cache gap | nan |
| Global serving tail latency | supported | jct_p99_ms | apc_like | 16590.930477492242 | 33477.3995576944 | -16886.469080202158 | -0.05 | global_simulation_summary.csv | Global serving results | nan |
| Decode cohort event-driven benefit | supported | decode_shared_kv_read_bytes | no_shared_kv_decode_cohort | 22277995364268.0 | 35540854374400.0 | -13262859010132.0 | 0.05 | ablation_delta_summary.csv | Ablation | nan |
| Shared-KV replication | demoted | mesh_traffic_bytes | no_replication | 9959223853056.0 | 9998683865088.0 | -39460012032.0 | -0.05 | replication_tradeoff_summary.csv | Replication design-space | Benefit-cost replication is not a headline claim unless it beats no_replication. |
| Oracle semantics | demoted | nan | oracle | 0.0 | 0.0 | 0.0 | 0.0 | report.json | Limitations | Renamed to ideal_next_use_cache; not claimed as full-system upper bound. |
