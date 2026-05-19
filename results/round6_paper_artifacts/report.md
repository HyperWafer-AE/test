# WaferAgent Round 6 Report

## Environment

- GPU: `NVIDIA H100 PCIe, 81559 MiB, 570.211.01
NVIDIA H100 PCIe, 81559 MiB, 570.211.01`
- Python: `3.11.14 (main, Oct 28 2025, 12:11:26) [Clang 20.1.4 ]`
- PyTorch: `2.5.1+cu124`
- CUDA available: `True`
- command: `scripts/run_global_serving_sweep.py --traces results/round5_workload_opportunity/traces/*.jsonl --wafer-config configs/wafer/wse_like.yaml --arrival-mode poisson --arrival-rate-jobs-per-s 1,2,4,8,16,32 --baselines no_cache,apc_like,kvflow_like,pat_like,waferagent_full,oracle --out results/round5_global_main_neutral --duration-source synthetic --seed 17 --clean-required`
- git commit: `60420c85b560439a45e1a502db86cba4490031b5`

## Readiness Check

### paper_ready
- PASS: artifact_tables_exported
- PASS: event_driven_decode_cohort
- PASS: replication_affects_actual_route
- PASS: ttft_tpot_correct
- PASS: existing_cache_gap_units_correct
- PASS: realistic_prefix_sensitivity
- PASS: planning_overhead_recorded
- PASS: global_serving_results_present
- PASS: ablation_nonzero_for_main_mechanisms
### sanity
- PASS: clean_git_tree
- PASS: no_silent_fallback
- PASS: neutral_default
- PASS: wafer_results_marked_simulation
### demoted
- PASS: dynamic_pd_partition
- PASS: tool_ttl
- PASS: critical_path_scheduling
- FAIL: replication_if_no_delta
- FAIL: distributed_sram_if_no_delta
### pass
- PASS: paper_ready
- PASS: sanity
- PASS: overall


## 1. Paper Goal and Claims

Round 6 focuses on paper-grade shared-KV execution semantics: event-driven decode cohorts, real shared-KV residency routing through distributed SRAM/mesh, correct TTFT/TPOT token accounting, realistic cross-job prefix sharing, and exportable artifact tables.

## 2. Global Serving Results

| baseline | arrival_rate_jobs_per_s | completed_jobs | jobs_per_s | tokens_per_s | jct_p50_ms | jct_p90_ms | jct_p99_ms | ttft_p50_ms | ttft_p90_ms | ttft_p99_ms | tpot_p50_ms | tpot_p90_ms | tpot_p99_ms | queue_wait_ms | mesh_wait_ms | cross_job_prefix_hit_rate | cross_job_prefix_compute_hits | sram_hit_rate | sram_evictions | sram_reload_bytes | mesh_total_traffic_bytes | mesh_hotspot_ratio | computed_prefill_tokens | avoided_prefill_tokens | compute_energy_j | mesh_energy_j | sram_energy_j | offwafer_energy_j | energy_per_job_j | decode_shared_kv_read_bytes | decode_shared_kv_read_bytes_without_cohort | decode_kv_read_reduction_ratio | cross_region_kv_transfer_bytes | num_decode_cohorts | avg_cohort_size | replica_bytes_total | saved_mesh_traffic_bytes | replication_transfer_bytes | replication_actual_transfer_bytes | shared_prefill_compute_ms_saved | sram_read_bytes | sram_write_bytes | offwafer_reload_bytes | shared_kv_extraction_overhead_ms | placement_planning_overhead_ms | replication_planning_overhead_ms | decode_cohort_planning_overhead_ms | scheduling_loop_overhead_ms | total_runtime_overhead_ms | overhead_per_job_ms | overhead_fraction_of_jct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| apc_like | 2 | 100 | 1.089 | 4.771e+04 | 1.968e+04 | 3.347e+04 | 3.348e+04 | 11.61 | 17.39 | 23.7 | 4.232 | 6.573 | 6.573 | 2.463e+04 | 0 | 0.9107 | 765 | 0.7506 | 114 | 8.388e+10 | 1.232e+14 | 4.069 | 2.163e+06 | 2.112e+06 | 0.8921 | 24.63 | 0.01401 | 0.06844 | 0.2561 | 3.554e+13 | 3.554e+13 | 0 | 3.554e+13 | 0 | 0 | 0 | 0 | 0 | 0 | 1.319e+04 | 1.929e+11 | 8.723e+10 | 8.556e+10 | 131.4 | 236.1 | 0.0257 | 0.9148 | 54.74 | 558.4 | 5.584 | 0.0002832 |
| kvflow_like | 2 | 100 | 1.089 | 4.771e+04 | 1.968e+04 | 3.347e+04 | 3.348e+04 | 11.61 | 17.39 | 23.7 | 4.232 | 6.573 | 6.573 | 2.463e+04 | 0 | 0.9107 | 765 | 0.7494 | 115 | 8.402e+10 | 1.232e+14 | 4.069 | 2.163e+06 | 2.112e+06 | 0.8921 | 24.63 | 0.01401 | 0.06856 | 0.2561 | 3.554e+13 | 3.554e+13 | 0 | 3.554e+13 | 0 | 0 | 0 | 0 | 0 | 0 | 1.319e+04 | 1.928e+11 | 8.738e+10 | 8.57e+10 | 77.77 | 236.9 | 0.02597 | 0.9897 | 54.55 | 507.1 | 5.071 | 0.0002572 |
| no_cache | 2 | 100 | 1.087 | 4.762e+04 | 1.979e+04 | 3.357e+04 | 3.357e+04 | 18.72 | 27.65 | 27.65 | 4.232 | 6.573 | 6.573 | 2.47e+04 | 0 | 0 | 0 | 0 | 449 | 0 | 1.235e+14 | 2.036 | 4.275e+06 | 0 | 1.723 | 24.7 | 0.02802 | 0.1745 | 0.2662 | 3.554e+13 | 3.554e+13 | 0 | 3.554e+13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5.604e+11 | 2.181e+11 | 77.64 | 235.6 | 0.04181 | 0.9425 | 57.22 | 508.8 | 5.088 | 0.0002571 |
| oracle | 2 | 100 | 1.649 | 7.225e+04 | 1133 | 2583 | 3627 | 4.591 | 10.76 | 14.59 | 1.457 | 3.075 | 3.832 | 34.23 | 13.49 | 0.9107 | 765 | 0.9104 | 85 | 5.157e+10 | 1.203e+11 | 2.607 | 2.163e+06 | 2.112e+06 | 0.8921 | 0.02405 | 0.995 | 0.04204 | 0.01953 | 2.206e+13 | 3.554e+13 | 0.3793 | 2.206e+13 | 153 | 3.791 | 0 | 0 | 0 | 0 | 1.319e+04 | 1.986e+13 | 4.369e+10 | 5.255e+10 | 79.78 | 244.2 | 0.03815 | 9.679 | 53.66 | 515.5 | 5.155 | 0.003887 |
| pat_like | 2 | 100 | 1.136 | 4.975e+04 | 1.781e+04 | 3.062e+04 | 3.099e+04 | 9.11 | 17.59 | 23.9 | 2.8 | 5.065 | 5.07 | 1.524e+04 | 0 | 0.9107 | 765 | 0.7565 | 120 | 8.03e+10 | 7.618e+13 | 3.812 | 2.163e+06 | 2.112e+06 | 0.8921 | 15.24 | 0.01401 | 0.0652 | 0.1621 | 2.251e+13 | 3.554e+13 | 0.3668 | 2.251e+13 | 149 | 3.785 | 0 | 0 | 0 | 0 | 1.319e+04 | 1.965e+11 | 8.365e+10 | 8.15e+10 | 77.79 | 232 | 0.02277 | 10.51 | 63.81 | 510.5 | 5.105 | 0.0003044 |
| wafer_naive | 2 | 100 | 1.087 | 4.762e+04 | 1.979e+04 | 3.357e+04 | 3.357e+04 | 18.72 | 27.65 | 27.65 | 4.232 | 6.573 | 6.573 | 2.47e+04 | 0 | 0 | 0 | 0 | 449 | 0 | 1.235e+14 | 2.036 | 4.275e+06 | 0 | 1.723 | 24.7 | 0.02802 | 0.1745 | 0.2662 | 3.554e+13 | 3.554e+13 | 0 | 3.554e+13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5.604e+11 | 2.181e+11 | 78.47 | 306.5 | 0.02389 | 0.9089 | 58.48 | 582.1 | 5.821 | 0.0002941 |
| waferagent_full | 2 | 100 | 1.611 | 7.059e+04 | 2169 | 4273 | 5087 | 9.048 | 17.59 | 23.9 | 2.698 | 5.065 | 5.065 | 74.53 | 9.087 | 0.9107 | 765 | 0.8836 | 110 | 7.07e+10 | 3.312e+11 | 3.047 | 2.163e+06 | 2.112e+06 | 0.8921 | 0.06625 | 0.9464 | 0.05761 | 0.01962 | 2.216e+13 | 3.554e+13 | 0.3764 | 2.216e+13 | 149 | 3.826 | 0 | 0 | 0 | 0 | 1.319e+04 | 1.887e+13 | 5.516e+10 | 7.201e+10 | 79.38 | 238.1 | 0.02545 | 9.817 | 52.58 | 510.8 | 5.108 | 0.002299 |
| apc_like | 4 | 100 | 1.597 | 6.998e+04 | 1.968e+04 | 3.347e+04 | 3.348e+04 | 11.61 | 17.39 | 23.7 | 4.232 | 6.573 | 6.573 | 2.462e+04 | 0 | 0.9107 | 765 | 0.7835 | 94 | 7.291e+10 | 1.231e+14 | 4.07 | 2.163e+06 | 2.112e+06 | 0.8921 | 24.62 | 0.01401 | 0.05967 | 0.2558 | 3.554e+13 | 3.554e+13 | 0 | 3.554e+13 | 0 | 0 | 0 | 0 | 0 | 0 | 1.319e+04 | 2.039e+11 | 7.627e+10 | 7.459e+10 | 77.42 | 232.4 | 0.03088 | 0.9028 | 54 | 498.3 | 4.983 | 0.0002529 |
| kvflow_like | 4 | 100 | 1.597 | 6.998e+04 | 1.968e+04 | 3.347e+04 | 3.348e+04 | 11.61 | 17.39 | 23.7 | 4.232 | 6.573 | 6.573 | 2.462e+04 | 0 | 0.9107 | 765 | 0.7824 | 95 | 7.306e+10 | 1.231e+14 | 4.07 | 2.163e+06 | 2.112e+06 | 0.8921 | 24.62 | 0.01401 | 0.05979 | 0.2558 | 3.554e+13 | 3.554e+13 | 0 | 3.554e+13 | 0 | 0 | 0 | 0 | 0 | 0 | 1.319e+04 | 2.038e+11 | 7.641e+10 | 7.473e+10 | 77.72 | 315.9 | 0.02212 | 0.9623 | 55.34 | 583.7 | 5.837 | 0.0002963 |
| no_cache | 4 | 100 | 1.593 | 6.978e+04 | 1.979e+04 | 3.357e+04 | 3.357e+04 | 18.72 | 27.65 | 27.65 | 4.232 | 6.573 | 6.573 | 2.47e+04 | 0 | 0 | 0 | 0 | 449 | 0 | 1.235e+14 | 2.036 | 4.275e+06 | 0 | 1.723 | 24.7 | 0.02802 | 0.1745 | 0.2662 | 3.554e+13 | 3.554e+13 | 0 | 3.554e+13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5.604e+11 | 2.181e+11 | 77.08 | 291.5 | 0.02319 | 0.9063 | 57.24 | 566.8 | 5.668 | 0.0002863 |
| oracle | 4 | 100 | 2.47 | 1.082e+05 | 1168 | 9694 | 1.174e+04 | 5.532 | 10.76 | 14.59 | 1.405 | 3.075 | 3.151 | 2101 | 2141 | 0.9107 | 765 | 0.8211 | 192 | 1.13e+11 | 1.926e+12 | 2.18 | 2.163e+06 | 2.112e+06 | 0.8921 | 0.3853 | 0.8512 | 0.09138 | 0.0222 | 2.19e+13 | 3.554e+13 | 0.3837 | 2.19e+13 | 165 | 3.606 | 0 | 0 | 0 | 0 | 1.319e+04 | 1.695e+13 | 7.228e+10 | 1.142e+11 | 78.98 | 239.2 | 0.02322 | 11.03 | 55.21 | 508.3 | 5.083 | 0.001508 |
| pat_like | 4 | 100 | 1.675 | 7.339e+04 | 9193 | 3.062e+04 | 3.064e+04 | 9.048 | 17.59 | 23.9 | 2.558 | 5.065 | 5.065 | 1.499e+04 | 0 | 0.9107 | 765 | 0.7588 | 116 | 8.1e+10 | 7.493e+13 | 3.847 | 2.163e+06 | 2.112e+06 | 0.8921 | 14.99 | 0.01401 | 0.0659 | 0.1596 | 2.236e+13 | 3.554e+13 | 0.371 | 2.236e+13 | 150 | 3.76 | 0 | 0 | 0 | 0 | 1.319e+04 | 1.958e+11 | 8.436e+10 | 8.238e+10 | 77.68 | 236.6 | 0.02174 | 10.81 | 63.42 | 513.2 | 5.132 | 0.0003115 |


## 3. Event-Driven Decode Cohorts

| cohort_id | shared_kv_id | node_ids | planned_start_ms | max_wait_ms | shared_kv_region | expected_shared_kv_bytes_read | expected_private_kv_bytes_read | expected_query_transfer_bytes | expected_merge_bytes | cohort_size | baseline | event_driven | arrival_rate_jobs_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| event_cohort_0 | sha256:189cf4f6f9e04f1e4e11d36545c0a19aca2cbcb501f9dbcb9c4f9dc74210cfca | debate_job_0_r0_proposer_1,debate_job_0_r0_proposer_2,debate_job_0_r0_proposer_3,debate_job_1_r0_proposer_2,debate_job_1_r0_proposer_0,debate_job_2_r0_proposer_1,debate_job_1_r0_proposer_1,debate_job_1_r0_proposer_3,debate_job_2_r0_proposer_3,debate_job_2_r0_proposer_2,debate_job_2_r0_proposer_0 | 3.734 | 2 | r1c0 | 17179869184 | 1441792 | 360448 | 90112 | 11 | pat_like | True | 16 |
| event_cohort_1 | sha256:c9ae75070adf4480e75e3ce89087a7329723b6fe299ec24d04b46275e59c92aa | debate_job_10_r0_proposer_1,debate_job_10_r0_proposer_2,debate_job_10_r0_proposer_3,debate_job_11_r0_proposer_1,debate_job_11_r0_proposer_0,debate_job_11_r0_proposer_3,debate_job_13_r0_proposer_1,debate_job_14_r0_proposer_2,debate_job_17_r0_proposer_0,debate_job_11_r0_proposer_2,debate_job_12_r0_proposer_3,debate_job_12_r0_proposer_0,debate_job_13_r0_proposer_2,debate_job_13_r0_proposer_3,debate_job_16_r0_proposer_1,debate_job_15_r0_proposer_2 | 3.734 | 2 | r1c0 | 17179869184 | 2097152 | 524288 | 131072 | 16 | pat_like | True | 16 |
| event_cohort_2 | sha256:c9ae75070adf4480e75e3ce89087a7329723b6fe299ec24d04b46275e59c92aa | debate_job_12_r0_proposer_1,debate_job_12_r0_proposer_2,debate_job_13_r0_proposer_0,debate_job_15_r0_proposer_0,debate_job_14_r0_proposer_1,debate_job_14_r0_proposer_0,debate_job_16_r0_proposer_2,debate_job_15_r0_proposer_3,debate_job_16_r0_proposer_0,debate_job_15_r0_proposer_1,debate_job_14_r0_proposer_3,debate_job_17_r0_proposer_1,debate_job_18_r0_proposer_3,debate_job_19_r0_proposer_1,debate_job_16_r0_proposer_3,debate_job_19_r0_proposer_0 | 3.734 | 2 | r1c0 | 17179869184 | 2097152 | 524288 | 131072 | 16 | pat_like | True | 16 |
| event_cohort_3 | sha256:c9ae75070adf4480e75e3ce89087a7329723b6fe299ec24d04b46275e59c92aa | debate_job_17_r0_proposer_2,debate_job_17_r0_proposer_3,debate_job_18_r0_proposer_0,debate_job_19_r0_proposer_2,debate_job_18_r0_proposer_2,debate_job_18_r0_proposer_1,debate_job_19_r0_proposer_3 | 3.734 | 2 | r1c0 | 17179869184 | 917504 | 229376 | 57344 | 7 | pat_like | True | 16 |
| event_cohort_4 | sha256:1bf945a34e5780e4b08114ef9815ece70617d5f164919d5257e61771a1b5741c | debate_job_20_r0_proposer_1,debate_job_20_r0_proposer_2,debate_job_20_r0_proposer_3,debate_job_22_r0_proposer_1,debate_job_21_r0_proposer_2,debate_job_21_r0_proposer_0,debate_job_22_r0_proposer_3,debate_job_21_r0_proposer_3,debate_job_22_r0_proposer_0,debate_job_21_r0_proposer_1,debate_job_22_r0_proposer_2 | 3.734 | 2 | r1c0 | 17179869184 | 1441792 | 360448 | 90112 | 11 | pat_like | True | 16 |
| event_cohort_5 | sha256:1bf945a34e5780e4b08114ef9815ece70617d5f164919d5257e61771a1b5741c | debate_job_23_r0_proposer_0,debate_job_23_r0_proposer_1,debate_job_24_r0_proposer_2,debate_job_23_r0_proposer_2,debate_job_25_r0_proposer_0,debate_job_26_r0_proposer_0,debate_job_24_r0_proposer_1,debate_job_23_r0_proposer_3,debate_job_26_r0_proposer_2,debate_job_25_r0_proposer_3,debate_job_27_r0_proposer_2,debate_job_27_r0_proposer_1,debate_job_24_r0_proposer_3,debate_job_29_r0_proposer_2,debate_job_24_r0_proposer_0,debate_job_28_r0_proposer_1 | 13.73 | 2 | r1c0 | 17179869184 | 2097152 | 524288 | 131072 | 16 | pat_like | True | 16 |
| event_cohort_6 | sha256:1bf945a34e5780e4b08114ef9815ece70617d5f164919d5257e61771a1b5741c | debate_job_25_r0_proposer_1,debate_job_25_r0_proposer_2,debate_job_26_r0_proposer_3,debate_job_26_r0_proposer_1,debate_job_28_r0_proposer_3,debate_job_28_r0_proposer_0,debate_job_27_r0_proposer_0,debate_job_29_r0_proposer_0,debate_job_28_r0_proposer_2,debate_job_27_r0_proposer_3,debate_job_29_r0_proposer_1,debate_job_29_r0_proposer_3 | 13.73 | 2 | r1c0 | 17179869184 | 1572864 | 393216 | 98304 | 12 | pat_like | True | 16 |
| event_cohort_7 | sha256:2cdbca1506dbbaea86db8efae9b02d70acf80ad24a8792db0ff6fc9027e0d116 | debate_job_30_r0_proposer_1,debate_job_30_r0_proposer_2,debate_job_30_r0_proposer_3,debate_job_31_r0_proposer_0,debate_job_31_r0_proposer_1,debate_job_32_r0_proposer_0,debate_job_32_r0_proposer_1,debate_job_34_r0_proposer_1,debate_job_31_r0_proposer_3,debate_job_31_r0_proposer_2,debate_job_33_r0_proposer_2,debate_job_33_r0_proposer_1,debate_job_32_r0_proposer_3,debate_job_33_r0_proposer_0,debate_job_35_r0_proposer_2,debate_job_36_r0_proposer_0 | 13.73 | 2 | r1c0 | 17179869184 | 2097152 | 524288 | 131072 | 16 | pat_like | True | 16 |
| event_cohort_8 | sha256:2cdbca1506dbbaea86db8efae9b02d70acf80ad24a8792db0ff6fc9027e0d116 | debate_job_32_r0_proposer_2,debate_job_33_r0_proposer_3,debate_job_34_r0_proposer_0,debate_job_34_r0_proposer_3,debate_job_35_r0_proposer_0,debate_job_34_r0_proposer_2,debate_job_35_r0_proposer_3,debate_job_36_r0_proposer_2,debate_job_37_r0_proposer_3,debate_job_35_r0_proposer_1,debate_job_37_r0_proposer_1,debate_job_37_r0_proposer_0,debate_job_36_r0_proposer_3,debate_job_36_r0_proposer_1,debate_job_37_r0_proposer_2 | 13.73 | 2 | r1c0 | 17179869184 | 1966080 | 491520 | 122880 | 15 | pat_like | True | 16 |
| event_cohort_9 | sha256:189cf4f6f9e04f1e4e11d36545c0a19aca2cbcb501f9dbcb9c4f9dc74210cfca | debate_job_3_r0_proposer_0,debate_job_3_r0_proposer_1,debate_job_3_r0_proposer_2,debate_job_3_r0_proposer_3 | 13.73 | 2 | r1c0 | 17179869184 | 524288 | 131072 | 32768 | 4 | pat_like | True | 16 |


## 4. Prefix Realism Sensitivity

| baseline | arrival_rate_jobs_per_s | completed_jobs | jobs_per_s | tokens_per_s | jct_p50_ms | jct_p90_ms | jct_p99_ms | ttft_p50_ms | ttft_p90_ms | ttft_p99_ms | tpot_p50_ms | tpot_p90_ms | tpot_p99_ms | queue_wait_ms | mesh_wait_ms | cross_job_prefix_hit_rate | cross_job_prefix_compute_hits | sram_hit_rate | sram_evictions | sram_reload_bytes | mesh_total_traffic_bytes | mesh_hotspot_ratio | computed_prefill_tokens | avoided_prefill_tokens | compute_energy_j | mesh_energy_j | sram_energy_j | offwafer_energy_j | energy_per_job_j | decode_shared_kv_read_bytes | decode_shared_kv_read_bytes_without_cohort | decode_kv_read_reduction_ratio | cross_region_kv_transfer_bytes | num_decode_cohorts | avg_cohort_size | replica_bytes_total | saved_mesh_traffic_bytes | replication_transfer_bytes | replication_actual_transfer_bytes | shared_prefill_compute_ms_saved | sram_read_bytes | sram_write_bytes | offwafer_reload_bytes | shared_kv_extraction_overhead_ms | placement_planning_overhead_ms | replication_planning_overhead_ms | decode_cohort_planning_overhead_ms | scheduling_loop_overhead_ms | total_runtime_overhead_ms | overhead_per_job_ms | overhead_fraction_of_jct | cross_job_task_group_size | unique_task_ratio | cross_job_prefix_hit_rate_observed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| apc_like | 1 | 15 | 0.4493 | 1.643e+04 | 6522 | 3.333e+04 | 3.338e+04 | 9.792 | 13.43 | 22.26 | 2.034 | 6.573 | 6.573 | 2.084e+04 | 0 | 0.2689 | 32 | 0.5923 | 45 | 9.127e+09 | 1.563e+13 | 3.966 | 2.806e+05 | 2.519e+05 | 0.1166 | 3.126 | 0.001745 | 0.007436 | 0.2168 | 4.381e+12 | 4.381e+12 | 0 | 4.381e+12 | 0 | 0 | 0 | 0 | 0 | 0 | 1493 | 2.389e+10 | 1.101e+10 | 9.295e+09 | 12.93 | 39.59 | 0.03676 | 0.135 | 9.331 | 89.2 | 5.947 | 0.0003886 | 1 | 0 | 0.9154 |
| waferagent_full | 1 | 15 | 3.535 | 1.293e+05 | 1523 | 3421 | 4161 | 4.466 | 8.785 | 21.78 | 1.349 | 3.637 | 4.958 | 492.7 | 359.6 | 0.2689 | 32 | 0.4977 | 100 | 1.819e+10 | 9.986e+10 | 3.978 | 2.806e+05 | 2.519e+05 | 0.1166 | 0.01997 | 0.05447 | 0.01471 | 0.01372 | 2.375e+12 | 4.381e+12 | 0.4579 | 2.375e+12 | 25 | 3.32 | 0 | 0 | 0 | 0 | 1493 | 1.079e+12 | 1.011e+10 | 1.839e+10 | 13.17 | 90.57 | 0.02173 | 1.167 | 7.968 | 134.4 | 8.962 | 0.004853 | 1 | 0 | 0.9154 |
| apc_like | 1 | 15 | 0.4493 | 1.643e+04 | 6522 | 3.333e+04 | 3.338e+04 | 9.792 | 13.43 | 22.26 | 2.034 | 6.573 | 6.573 | 2.084e+04 | 0 | 0.2689 | 32 | 0.5923 | 45 | 9.127e+09 | 1.563e+13 | 3.966 | 2.806e+05 | 2.519e+05 | 0.1166 | 3.126 | 0.001745 | 0.007436 | 0.2168 | 4.381e+12 | 4.381e+12 | 0 | 4.381e+12 | 0 | 0 | 0 | 0 | 0 | 0 | 1493 | 2.389e+10 | 1.101e+10 | 9.295e+09 | 12.88 | 39.31 | 0.02132 | 0.1306 | 9.032 | 82.18 | 5.478 | 0.000358 | 1 | 0.25 | 0.9154 |
| waferagent_full | 1 | 15 | 3.535 | 1.293e+05 | 1523 | 3421 | 4161 | 4.466 | 8.785 | 21.78 | 1.349 | 3.637 | 4.958 | 492.7 | 359.6 | 0.2689 | 32 | 0.4977 | 100 | 1.819e+10 | 9.986e+10 | 3.978 | 2.806e+05 | 2.519e+05 | 0.1166 | 0.01997 | 0.05447 | 0.01471 | 0.01372 | 2.375e+12 | 4.381e+12 | 0.4579 | 2.375e+12 | 25 | 3.32 | 0 | 0 | 0 | 0 | 1493 | 1.079e+12 | 1.011e+10 | 1.839e+10 | 13.27 | 40.22 | 0.01892 | 1.114 | 7.743 | 83.12 | 5.541 | 0.003001 | 1 | 0.25 | 0.9154 |
| apc_like | 1 | 15 | 0.4493 | 1.643e+04 | 6522 | 3.333e+04 | 3.338e+04 | 9.792 | 13.43 | 22.26 | 2.034 | 6.573 | 6.573 | 2.084e+04 | 0 | 0.2689 | 32 | 0.5923 | 45 | 9.127e+09 | 1.563e+13 | 3.966 | 2.806e+05 | 2.519e+05 | 0.1166 | 3.126 | 0.001745 | 0.007436 | 0.2168 | 4.381e+12 | 4.381e+12 | 0 | 4.381e+12 | 0 | 0 | 0 | 0 | 0 | 0 | 1493 | 2.389e+10 | 1.101e+10 | 9.295e+09 | 12.92 | 39.26 | 0.0218 | 0.1294 | 8.682 | 82.39 | 5.493 | 0.0003589 | 1 | 0.5 | 0.9154 |
| waferagent_full | 1 | 15 | 3.535 | 1.293e+05 | 1523 | 3421 | 4161 | 4.466 | 8.785 | 21.78 | 1.349 | 3.637 | 4.958 | 492.7 | 359.6 | 0.2689 | 32 | 0.4977 | 100 | 1.819e+10 | 9.986e+10 | 3.978 | 2.806e+05 | 2.519e+05 | 0.1166 | 0.01997 | 0.05447 | 0.01471 | 0.01372 | 2.375e+12 | 4.381e+12 | 0.4579 | 2.375e+12 | 25 | 3.32 | 0 | 0 | 0 | 0 | 1493 | 1.079e+12 | 1.011e+10 | 1.839e+10 | 13.43 | 41.04 | 0.02069 | 1.146 | 8.195 | 84.79 | 5.652 | 0.003061 | 1 | 0.5 | 0.9154 |
| apc_like | 1 | 15 | 0.4493 | 1.643e+04 | 6522 | 3.333e+04 | 3.338e+04 | 9.792 | 13.43 | 22.26 | 2.034 | 6.573 | 6.573 | 2.084e+04 | 0 | 0.2689 | 32 | 0.5923 | 45 | 9.127e+09 | 1.563e+13 | 3.966 | 2.806e+05 | 2.519e+05 | 0.1166 | 3.126 | 0.001745 | 0.007436 | 0.2168 | 4.381e+12 | 4.381e+12 | 0 | 4.381e+12 | 0 | 0 | 0 | 0 | 0 | 0 | 1493 | 2.389e+10 | 1.101e+10 | 9.295e+09 | 12.87 | 39.33 | 0.01989 | 0.1329 | 8.74 | 82.25 | 5.483 | 0.0003583 | 1 | 0.75 | 0.9154 |
| waferagent_full | 1 | 15 | 3.535 | 1.293e+05 | 1523 | 3421 | 4161 | 4.466 | 8.785 | 21.78 | 1.349 | 3.637 | 4.958 | 492.7 | 359.6 | 0.2689 | 32 | 0.4977 | 100 | 1.819e+10 | 9.986e+10 | 3.978 | 2.806e+05 | 2.519e+05 | 0.1166 | 0.01997 | 0.05447 | 0.01471 | 0.01372 | 2.375e+12 | 4.381e+12 | 0.4579 | 2.375e+12 | 25 | 3.32 | 0 | 0 | 0 | 0 | 1493 | 1.079e+12 | 1.011e+10 | 1.839e+10 | 13.18 | 40.34 | 0.02214 | 1.15 | 7.993 | 83.45 | 5.563 | 0.003012 | 1 | 0.75 | 0.9154 |
| apc_like | 1 | 15 | 0.4493 | 1.643e+04 | 6522 | 3.333e+04 | 3.338e+04 | 9.792 | 13.43 | 22.26 | 2.034 | 6.573 | 6.573 | 2.084e+04 | 0 | 0.2689 | 32 | 0.5923 | 45 | 9.127e+09 | 1.563e+13 | 3.966 | 2.806e+05 | 2.519e+05 | 0.1166 | 3.126 | 0.001745 | 0.007436 | 0.2168 | 4.381e+12 | 4.381e+12 | 0 | 4.381e+12 | 0 | 0 | 0 | 0 | 0 | 0 | 1493 | 2.389e+10 | 1.101e+10 | 9.295e+09 | 13.46 | 40.12 | 0.01926 | 0.1301 | 8.779 | 83.74 | 5.582 | 0.0003648 | 1 | 1 | 0.9154 |
| waferagent_full | 1 | 15 | 3.535 | 1.293e+05 | 1523 | 3421 | 4161 | 4.466 | 8.785 | 21.78 | 1.349 | 3.637 | 4.958 | 492.7 | 359.6 | 0.2689 | 32 | 0.4977 | 100 | 1.819e+10 | 9.986e+10 | 3.978 | 2.806e+05 | 2.519e+05 | 0.1166 | 0.01997 | 0.05447 | 0.01471 | 0.01372 | 2.375e+12 | 4.381e+12 | 0.4579 | 2.375e+12 | 25 | 3.32 | 0 | 0 | 0 | 0 | 1493 | 1.079e+12 | 1.011e+10 | 1.839e+10 | 13.07 | 40.03 | 0.01976 | 1.111 | 7.854 | 82.55 | 5.504 | 0.00298 | 1 | 1 | 0.9154 |


## 5. Planning Overhead

| baseline | arrival_rate_jobs_per_s | shared_kv_extraction_overhead_ms | placement_planning_overhead_ms | replication_planning_overhead_ms | decode_cohort_planning_overhead_ms | scheduling_loop_overhead_ms | total_runtime_overhead_ms | overhead_per_job_ms | overhead_fraction_of_jct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| apc_like | 2 | 131.4 | 236.1 | 0.0257 | 0.9148 | 54.74 | 558.4 | 5.584 | 0.0002832 |
| kvflow_like | 2 | 77.77 | 236.9 | 0.02597 | 0.9897 | 54.55 | 507.1 | 5.071 | 0.0002572 |
| no_cache | 2 | 77.64 | 235.6 | 0.04181 | 0.9425 | 57.22 | 508.8 | 5.088 | 0.0002571 |
| oracle | 2 | 79.78 | 244.2 | 0.03815 | 9.679 | 53.66 | 515.5 | 5.155 | 0.003887 |
| pat_like | 2 | 77.79 | 232 | 0.02277 | 10.51 | 63.81 | 510.5 | 5.105 | 0.0003044 |
| wafer_naive | 2 | 78.47 | 306.5 | 0.02389 | 0.9089 | 58.48 | 582.1 | 5.821 | 0.0002941 |
| waferagent_full | 2 | 79.38 | 238.1 | 0.02545 | 9.817 | 52.58 | 510.8 | 5.108 | 0.002299 |
| apc_like | 4 | 77.42 | 232.4 | 0.03088 | 0.9028 | 54 | 498.3 | 4.983 | 0.0002529 |
| kvflow_like | 4 | 77.72 | 315.9 | 0.02212 | 0.9623 | 55.34 | 583.7 | 5.837 | 0.0002963 |
| no_cache | 4 | 77.08 | 291.5 | 0.02319 | 0.9063 | 57.24 | 566.8 | 5.668 | 0.0002863 |


## 6. Artifact Export

Lightweight paper-facing artifacts are exported under `results/round6_paper_artifacts/` for independent review. Wafer results remain trace-driven wafer-scale simulation, not real wafer hardware measurements.


## Main Neutral Results

| baseline | arrival_rate_jobs_per_s | completed_jobs | jobs_per_s | tokens_per_s | jct_p50_ms | jct_p90_ms | jct_p99_ms | ttft_p50_ms | ttft_p90_ms | ttft_p99_ms | tpot_p50_ms | tpot_p90_ms | tpot_p99_ms | queue_wait_ms | mesh_wait_ms | cross_job_prefix_hit_rate | cross_job_prefix_compute_hits | sram_hit_rate | sram_evictions | sram_reload_bytes | mesh_total_traffic_bytes | mesh_hotspot_ratio | computed_prefill_tokens | avoided_prefill_tokens | compute_energy_j | mesh_energy_j | sram_energy_j | offwafer_energy_j | energy_per_job_j | decode_shared_kv_read_bytes | decode_shared_kv_read_bytes_without_cohort | decode_kv_read_reduction_ratio | cross_region_kv_transfer_bytes | num_decode_cohorts | avg_cohort_size | replica_bytes_total | saved_mesh_traffic_bytes | replication_transfer_bytes | replication_actual_transfer_bytes | shared_prefill_compute_ms_saved | sram_read_bytes | sram_write_bytes | offwafer_reload_bytes | shared_kv_extraction_overhead_ms | placement_planning_overhead_ms | replication_planning_overhead_ms | decode_cohort_planning_overhead_ms | scheduling_loop_overhead_ms | total_runtime_overhead_ms | overhead_per_job_ms | overhead_fraction_of_jct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| apc_like | 2 | 100 | 1.089 | 4.771e+04 | 1.968e+04 | 3.347e+04 | 3.348e+04 | 11.61 | 17.39 | 23.7 | 4.232 | 6.573 | 6.573 | 2.463e+04 | 0 | 0.9107 | 765 | 0.7506 | 114 | 8.388e+10 | 1.232e+14 | 4.069 | 2.163e+06 | 2.112e+06 | 0.8921 | 24.63 | 0.01401 | 0.06844 | 0.2561 | 3.554e+13 | 3.554e+13 | 0 | 3.554e+13 | 0 | 0 | 0 | 0 | 0 | 0 | 1.319e+04 | 1.929e+11 | 8.723e+10 | 8.556e+10 | 131.4 | 236.1 | 0.0257 | 0.9148 | 54.74 | 558.4 | 5.584 | 0.0002832 |
| kvflow_like | 2 | 100 | 1.089 | 4.771e+04 | 1.968e+04 | 3.347e+04 | 3.348e+04 | 11.61 | 17.39 | 23.7 | 4.232 | 6.573 | 6.573 | 2.463e+04 | 0 | 0.9107 | 765 | 0.7494 | 115 | 8.402e+10 | 1.232e+14 | 4.069 | 2.163e+06 | 2.112e+06 | 0.8921 | 24.63 | 0.01401 | 0.06856 | 0.2561 | 3.554e+13 | 3.554e+13 | 0 | 3.554e+13 | 0 | 0 | 0 | 0 | 0 | 0 | 1.319e+04 | 1.928e+11 | 8.738e+10 | 8.57e+10 | 77.77 | 236.9 | 0.02597 | 0.9897 | 54.55 | 507.1 | 5.071 | 0.0002572 |
| no_cache | 2 | 100 | 1.087 | 4.762e+04 | 1.979e+04 | 3.357e+04 | 3.357e+04 | 18.72 | 27.65 | 27.65 | 4.232 | 6.573 | 6.573 | 2.47e+04 | 0 | 0 | 0 | 0 | 449 | 0 | 1.235e+14 | 2.036 | 4.275e+06 | 0 | 1.723 | 24.7 | 0.02802 | 0.1745 | 0.2662 | 3.554e+13 | 3.554e+13 | 0 | 3.554e+13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5.604e+11 | 2.181e+11 | 77.64 | 235.6 | 0.04181 | 0.9425 | 57.22 | 508.8 | 5.088 | 0.0002571 |
| oracle | 2 | 100 | 1.649 | 7.225e+04 | 1133 | 2583 | 3627 | 4.591 | 10.76 | 14.59 | 1.457 | 3.075 | 3.832 | 34.23 | 13.49 | 0.9107 | 765 | 0.9104 | 85 | 5.157e+10 | 1.203e+11 | 2.607 | 2.163e+06 | 2.112e+06 | 0.8921 | 0.02405 | 0.995 | 0.04204 | 0.01953 | 2.206e+13 | 3.554e+13 | 0.3793 | 2.206e+13 | 153 | 3.791 | 0 | 0 | 0 | 0 | 1.319e+04 | 1.986e+13 | 4.369e+10 | 5.255e+10 | 79.78 | 244.2 | 0.03815 | 9.679 | 53.66 | 515.5 | 5.155 | 0.003887 |
| pat_like | 2 | 100 | 1.136 | 4.975e+04 | 1.781e+04 | 3.062e+04 | 3.099e+04 | 9.11 | 17.59 | 23.9 | 2.8 | 5.065 | 5.07 | 1.524e+04 | 0 | 0.9107 | 765 | 0.7565 | 120 | 8.03e+10 | 7.618e+13 | 3.812 | 2.163e+06 | 2.112e+06 | 0.8921 | 15.24 | 0.01401 | 0.0652 | 0.1621 | 2.251e+13 | 3.554e+13 | 0.3668 | 2.251e+13 | 149 | 3.785 | 0 | 0 | 0 | 0 | 1.319e+04 | 1.965e+11 | 8.365e+10 | 8.15e+10 | 77.79 | 232 | 0.02277 | 10.51 | 63.81 | 510.5 | 5.105 | 0.0003044 |
| wafer_naive | 2 | 100 | 1.087 | 4.762e+04 | 1.979e+04 | 3.357e+04 | 3.357e+04 | 18.72 | 27.65 | 27.65 | 4.232 | 6.573 | 6.573 | 2.47e+04 | 0 | 0 | 0 | 0 | 449 | 0 | 1.235e+14 | 2.036 | 4.275e+06 | 0 | 1.723 | 24.7 | 0.02802 | 0.1745 | 0.2662 | 3.554e+13 | 3.554e+13 | 0 | 3.554e+13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5.604e+11 | 2.181e+11 | 78.47 | 306.5 | 0.02389 | 0.9089 | 58.48 | 582.1 | 5.821 | 0.0002941 |
| waferagent_full | 2 | 100 | 1.611 | 7.059e+04 | 2169 | 4273 | 5087 | 9.048 | 17.59 | 23.9 | 2.698 | 5.065 | 5.065 | 74.53 | 9.087 | 0.9107 | 765 | 0.8836 | 110 | 7.07e+10 | 3.312e+11 | 3.047 | 2.163e+06 | 2.112e+06 | 0.8921 | 0.06625 | 0.9464 | 0.05761 | 0.01962 | 2.216e+13 | 3.554e+13 | 0.3764 | 2.216e+13 | 149 | 3.826 | 0 | 0 | 0 | 0 | 1.319e+04 | 1.887e+13 | 5.516e+10 | 7.201e+10 | 79.38 | 238.1 | 0.02545 | 9.817 | 52.58 | 510.8 | 5.108 | 0.002299 |
| apc_like | 4 | 100 | 1.597 | 6.998e+04 | 1.968e+04 | 3.347e+04 | 3.348e+04 | 11.61 | 17.39 | 23.7 | 4.232 | 6.573 | 6.573 | 2.462e+04 | 0 | 0.9107 | 765 | 0.7835 | 94 | 7.291e+10 | 1.231e+14 | 4.07 | 2.163e+06 | 2.112e+06 | 0.8921 | 24.62 | 0.01401 | 0.05967 | 0.2558 | 3.554e+13 | 3.554e+13 | 0 | 3.554e+13 | 0 | 0 | 0 | 0 | 0 | 0 | 1.319e+04 | 2.039e+11 | 7.627e+10 | 7.459e+10 | 77.42 | 232.4 | 0.03088 | 0.9028 | 54 | 498.3 | 4.983 | 0.0002529 |
| kvflow_like | 4 | 100 | 1.597 | 6.998e+04 | 1.968e+04 | 3.347e+04 | 3.348e+04 | 11.61 | 17.39 | 23.7 | 4.232 | 6.573 | 6.573 | 2.462e+04 | 0 | 0.9107 | 765 | 0.7824 | 95 | 7.306e+10 | 1.231e+14 | 4.07 | 2.163e+06 | 2.112e+06 | 0.8921 | 24.62 | 0.01401 | 0.05979 | 0.2558 | 3.554e+13 | 3.554e+13 | 0 | 3.554e+13 | 0 | 0 | 0 | 0 | 0 | 0 | 1.319e+04 | 2.038e+11 | 7.641e+10 | 7.473e+10 | 77.72 | 315.9 | 0.02212 | 0.9623 | 55.34 | 583.7 | 5.837 | 0.0002963 |
| no_cache | 4 | 100 | 1.593 | 6.978e+04 | 1.979e+04 | 3.357e+04 | 3.357e+04 | 18.72 | 27.65 | 27.65 | 4.232 | 6.573 | 6.573 | 2.47e+04 | 0 | 0 | 0 | 0 | 449 | 0 | 1.235e+14 | 2.036 | 4.275e+06 | 0 | 1.723 | 24.7 | 0.02802 | 0.1745 | 0.2662 | 3.554e+13 | 3.554e+13 | 0 | 3.554e+13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5.604e+11 | 2.181e+11 | 77.08 | 291.5 | 0.02319 | 0.9063 | 57.24 | 566.8 | 5.668 | 0.0002863 |
| oracle | 4 | 100 | 2.47 | 1.082e+05 | 1168 | 9694 | 1.174e+04 | 5.532 | 10.76 | 14.59 | 1.405 | 3.075 | 3.151 | 2101 | 2141 | 0.9107 | 765 | 0.8211 | 192 | 1.13e+11 | 1.926e+12 | 2.18 | 2.163e+06 | 2.112e+06 | 0.8921 | 0.3853 | 0.8512 | 0.09138 | 0.0222 | 2.19e+13 | 3.554e+13 | 0.3837 | 2.19e+13 | 165 | 3.606 | 0 | 0 | 0 | 0 | 1.319e+04 | 1.695e+13 | 7.228e+10 | 1.142e+11 | 78.98 | 239.2 | 0.02322 | 11.03 | 55.21 | 508.3 | 5.083 | 0.001508 |
| pat_like | 4 | 100 | 1.675 | 7.339e+04 | 9193 | 3.062e+04 | 3.064e+04 | 9.048 | 17.59 | 23.9 | 2.558 | 5.065 | 5.065 | 1.499e+04 | 0 | 0.9107 | 765 | 0.7588 | 116 | 8.1e+10 | 7.493e+13 | 3.847 | 2.163e+06 | 2.112e+06 | 0.8921 | 14.99 | 0.01401 | 0.0659 | 0.1596 | 2.236e+13 | 3.554e+13 | 0.371 | 2.236e+13 | 150 | 3.76 | 0 | 0 | 0 | 0 | 1.319e+04 | 1.958e+11 | 8.436e+10 | 8.238e+10 | 77.68 | 236.6 | 0.02174 | 10.81 | 63.42 | 513.2 | 5.132 | 0.0003115 |


## H100 Trace And Calibration

- HF trace model: `/data2/model_zoo/Qwen2.5-7B-Instruct`
- HF trace engine: `hf`
- HF calibration model: `/data2/model_zoo/Qwen2.5-7B-Instruct`
- HF calibration engine: `hf`
- vLLM smoke model: `/data2/model_zoo/Qwen2.5-7B-Instruct`
- vLLM smoke engine: `vllm`
- vLLM smoke real trace: `True`

## H100-Calibrated Wafer Simulation

Not available.


## Failures / Missing Baselines

- vLLM install: `vLLM 0.6.4.post1 install retry exit_code=0 finished: 2026-05-18T20:20:07+08:00`.
- vLLM full baseline: if `real_vllm_full_or_explicitly_missing` is FAIL, do not use vLLM as a paper-grade baseline.
- Dynamic P/D partition is demoted unless targeted ablation shows >=5% JCT benefit.
- Wafer: all wafer results are trace-driven wafer-scale simulator results, not real wafer hardware measurements.

## Interpretation

Main wafer numbers are trace-driven simulation results. Real H100 traces and H100 calibration are recorded separately when the HF/vLLM engines complete. Synthetic traces are used only for controlled mechanism stress tests unless explicitly marked otherwise.

## Output Layout

- JSON readiness: `/home/duzc/data/agent_wafer/results/round6_final_report/report.json`
- Tables: `/home/duzc/data/agent_wafer/results/round6_final_report/tables`
- Figures: `/home/duzc/data/agent_wafer/results/round6_final_report/figures`
