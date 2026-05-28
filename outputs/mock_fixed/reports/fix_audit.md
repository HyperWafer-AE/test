# Agent Trace Profiling Fix Audit

## Run

- commit hash: `6e00f4aae611620db4f38ebc8d16adaf0a06951b`
- command: `python scripts/run_all.py --datasets mock --sample-size 10  --outdir /home/duzc/data/agent_wafer/outputs/mock_fixed`
- used_mocks: `{'mock': True}`
- warnings_summary: `mock: using bundled TerminalBench/SWE-agent style rows.`

## Correctness Fixes

1. Fixed wrapper vs semantic tool bias.
   - Files: `src/normalize/command_parser.py`, `src/normalize/schema.py`, `src/normalize/normalizer.py`, `src/analysis/transitions.py`.
   - Before: `bash_command -> bash_command` and `execute_bash -> execute_bash` dominated tool locality.
   - After: canonical steps include `tool_wrapper`, `semantic_tool`, and `command_string`; H1 is judged on semantic/phase/collapsed views.

2. Fixed command-aware phase classification.
   - Files: `src/normalize/command_parser.py`, `src/normalize/normalizer.py`.
   - Before: natural-language messages mentioning “test/create/verify” could change phase.
   - After: phase uses `command_string`/`semantic_tool` first, wrapper second, terse command-like text last.

3. Fixed path regex.
   - File: `src/normalize/normalizer.py`.
   - Before: `.html` could be truncated to `.h`.
   - After: extension ordering and boundary checks preserve `/app/out.html` and common extensions.

4. Fixed access_type and object provenance.
   - Files: `src/normalize/schema.py`, `src/normalize/normalizer.py`.
   - Before: any path in edit/write phase could be labeled `write`.
   - After: write access requires explicit write command; observation paths are `mention`, `execute_result`, or `retrieve_result`. Added `object_source` and `stable_object`.

5. Separated stable object reuse from synthetic bucket locality.
   - File: `src/analysis/object_locality.py`.
   - Before: `large_observation_bucket` could manufacture reuse evidence.
   - After: stable file/url/test_case reuse is reported separately from synthetic observation/test-log/large buckets.

6. Added transition stratification.
   - File: `src/analysis/transitions.py`.
   - Outputs: wrapper, semantic, collapsed semantic, phase recall and heatmaps.

7. Added fingerprint baselines and bootstrap CIs.
   - File: `src/analysis/fingerprint.py`.
   - Baselines: global mean, task, harness, task+harness, early fingerprint model.

8. Added prediction-aware and capacity-aware wafer proxy strategies.
   - File: `src/sim/wafer_proxy.py`.
   - Oracle island is reported only as an upper bound; feasible claims use early/capacity strategies.

9. Added strict-real mode.
   - File: `scripts/run_all.py`.
   - `--strict-real`/`--no-mock-fallback` refuses mock fallback for real experiments.

## Hypothesis Verdicts After Fix

| Hypothesis | Verdict |
|---|---:|
| H1 semantic/phase locality | 支持 |
| H2 early fingerprint vs baselines | 不支持 |
| H3 stable object locality | 部分支持 |
| H4 success/failure differences | 支持 |
| H5 prediction-aware wafer proxy | 支持 |

## Transition Recall By View

| view                    |   k |   conditional_recall |   global_recall |   delta_vs_global |   n_test |
|:------------------------|----:|---------------------:|----------------:|------------------:|---------:|
| wrapper_tool            |   1 |                0.375 |           0.375 |             0     |        8 |
| wrapper_tool            |   3 |                0.375 |           0.625 |            -0.25  |        8 |
| wrapper_tool            |   5 |                0.375 |           0.625 |            -0.25  |        8 |
| semantic_tool           |   1 |                0.125 |           0     |             0.125 |        8 |
| semantic_tool           |   3 |                0.125 |           0     |             0.125 |        8 |
| semantic_tool           |   5 |                0.125 |           0     |             0.125 |        8 |
| collapsed_semantic_tool |   1 |                0.125 |           0     |             0.125 |        8 |
| collapsed_semantic_tool |   3 |                0.125 |           0     |             0.125 |        8 |
| collapsed_semantic_tool |   5 |                0.125 |           0     |             0.125 |        8 |
| phase                   |   1 |                0.375 |           0.375 |             0     |        8 |
| phase                   |   3 |                0.75  |           0.75  |             0     |        8 |
| phase                   |   5 |                0.875 |           1     |            -0.125 |        8 |
| collapsed_phase         |   1 |                0.375 |           0.375 |             0     |        8 |
| collapsed_phase         |   3 |                0.75  |           0.75  |             0     |        8 |
| collapsed_phase         |   5 |                0.875 |           1     |            -0.125 |        8 |

## Fingerprint Metrics By Baseline

|   K | baseline                |   n_train |   n_test |   future_tool_cosine |   future_tool_cosine_ci_low |   future_tool_cosine_ci_high |   future_tool_top3_recall |   future_tool_top3_recall_ci_low |   future_tool_top3_recall_ci_high |   remaining_steps_mae |   remaining_steps_mae_ci_low |   remaining_steps_mae_ci_high |   remaining_steps_r2 |   remaining_steps_r2_ci_low |   remaining_steps_r2_ci_high |   future_obs_spearman |   future_obs_mae |   future_obs_mae_ci_low |   future_obs_mae_ci_high |   long_auc |   long_accuracy |   long_accuracy_ci_low |   long_accuracy_ci_high |   future_tool_cosine_delta_global |   future_tool_top3_recall_delta_global |   remaining_steps_mae_delta_global |   future_obs_mae_delta_global |   long_auc_delta_global |   long_accuracy_delta_global |   long_auc_ci_low |   long_auc_ci_high |   future_obs_spearman_ci_low |   future_obs_spearman_ci_high |
|----:|:------------------------|----------:|---------:|---------------------:|----------------------------:|-----------------------------:|--------------------------:|---------------------------------:|----------------------------------:|----------------------:|-----------------------------:|------------------------------:|---------------------:|----------------------------:|-----------------------------:|----------------------:|-----------------:|------------------------:|-------------------------:|-----------:|----------------:|-----------------------:|------------------------:|----------------------------------:|---------------------------------------:|-----------------------------------:|------------------------------:|------------------------:|-----------------------------:|------------------:|-------------------:|-----------------------------:|------------------------------:|
|   1 | global_mean_baseline    |         4 |        2 |            0.313013  |                    0.293962 |                    0.332065  |                       0   |                                0 |                                 0 |                  1.75 |                         0.75 |                          2.75 |            -3.0625   |                   -3.0625   |                    -3.0625   |                   nan |            79.5  |                   37.25 |                   121.75 |      nan   |             0   |                      0 |                       0 |                          0        |                                    0   |                               0    |                          0    |                   nan   |                          0   |             nan   |              nan   |                          nan |                           nan |
|   1 | task_baseline           |         4 |        2 |            0.313013  |                    0.293962 |                    0.332065  |                       0   |                                0 |                                 0 |                  1.75 |                         0.75 |                          2.75 |            -3.0625   |                   -3.0625   |                    -3.0625   |                   nan |            79.5  |                   37.25 |                   121.75 |      nan   |             0   |                      0 |                       0 |                          0        |                                    0   |                               0    |                          0    |                   nan   |                          0   |             nan   |              nan   |                          nan |                           nan |
|   1 | harness_baseline        |         4 |        2 |            0.313013  |                    0.293962 |                    0.332065  |                       0   |                                0 |                                 0 |                  1.75 |                         0.75 |                          2.75 |            -3.0625   |                   -3.0625   |                    -3.0625   |                   nan |            79.5  |                   37.25 |                   121.75 |      nan   |             0   |                      0 |                       0 |                          0        |                                    0   |                               0    |                          0    |                   nan   |                          0   |             nan   |              nan   |                          nan |                           nan |
|   1 | task_harness_baseline   |         4 |        2 |            0.313013  |                    0.293962 |                    0.332065  |                       0   |                                0 |                                 0 |                  1.75 |                         0.75 |                          2.75 |            -3.0625   |                   -3.0625   |                    -3.0625   |                   nan |            79.5  |                   37.25 |                   121.75 |      nan   |             0   |                      0 |                       0 |                          0        |                                    0   |                               0    |                          0    |                   nan   |                          0   |             nan   |              nan   |                          nan |                           nan |
|   1 | early_fingerprint_model |         4 |        2 |            0.313013  |                    0.293962 |                    0.332065  |                       0   |                                0 |                                 0 |                  1.75 |                         0.75 |                          2.75 |            -3.0625   |                   -3.0625   |                    -3.0625   |                   nan |            79.5  |                   37.25 |                   121.75 |      nan   |             0   |                      0 |                       0 |                          0        |                                    0   |                               0    |                          0    |                   nan   |                          0   |             nan   |              nan   |                          nan |                           nan |
|   2 | global_mean_baseline    |         4 |        2 |            0.0394464 |                    0        |                    0.0788928 |                       0   |                                0 |                                 0 |                  1.25 |                         1.25 |                          1.25 |           nan        |                  nan        |                   nan        |                   nan |            22    |                    1.25 |                    42.75 |      nan   |             0   |                      0 |                       0 |                          0        |                                    0   |                               0    |                          0    |                   nan   |                          0   |             nan   |              nan   |                          nan |                           nan |
|   2 | task_baseline           |         4 |        2 |            0.0394464 |                    0        |                    0.0788928 |                       0   |                                0 |                                 0 |                  1.25 |                         1.25 |                          1.25 |           nan        |                  nan        |                   nan        |                   nan |            22    |                    1.25 |                    42.75 |      nan   |             0   |                      0 |                       0 |                          0        |                                    0   |                               0    |                          0    |                   nan   |                          0   |             nan   |              nan   |                          nan |                           nan |
|   2 | harness_baseline        |         4 |        2 |            0.0394464 |                    0        |                    0.0788928 |                       0   |                                0 |                                 0 |                  1.25 |                         1.25 |                          1.25 |           nan        |                  nan        |                   nan        |                   nan |            22    |                    1.25 |                    42.75 |      nan   |             0   |                      0 |                       0 |                          0        |                                    0   |                               0    |                          0    |                   nan   |                          0   |             nan   |              nan   |                          nan |                           nan |
|   2 | task_harness_baseline   |         4 |        2 |            0.0394464 |                    0        |                    0.0788928 |                       0   |                                0 |                                 0 |                  1.25 |                         1.25 |                          1.25 |           nan        |                  nan        |                   nan        |                   nan |            22    |                    1.25 |                    42.75 |      nan   |             0   |                      0 |                       0 |                          0        |                                    0   |                               0    |                          0    |                   nan   |                          0   |             nan   |              nan   |                          nan |                           nan |
|   2 | early_fingerprint_model |         4 |        2 |            0.0394464 |                    0        |                    0.0788928 |                       0   |                                0 |                                 0 |                  1.25 |                         1.25 |                          1.25 |           nan        |                  nan        |                   nan        |                   nan |            22    |                    1.25 |                    42.75 |      nan   |             0   |                      0 |                       0 |                          0        |                                    0   |                               0    |                          0    |                   nan   |                          0   |             nan   |              nan   |                          nan |                           nan |
|   3 | global_mean_baseline    |         4 |        2 |            0.46751   |                    0.387298 |                    0.547723  |                       0.5 |                                0 |                                 1 |                  1.5  |                         0.5  |                          2.5  |            -0.444444 |                   -0.444444 |                    -0.444444 |                   nan |            63.5  |                   18.5  |                   108.5  |        0.5 |             0.5 |                      0 |                       1 |                          0        |                                    0   |                               0    |                          0    |                     0   |                          0   |               0.5 |                0.5 |                          nan |                           nan |
|   3 | task_baseline           |         4 |        2 |            0.46751   |                    0.387298 |                    0.547723  |                       0.5 |                                0 |                                 1 |                  1.5  |                         0.5  |                          2.5  |            -0.444444 |                   -0.444444 |                    -0.444444 |                   nan |            63.5  |                   18.5  |                   108.5  |        0.5 |             0.5 |                      0 |                       1 |                          0        |                                    0   |                               0    |                          0    |                     0   |                          0   |               0.5 |                0.5 |                          nan |                           nan |
|   3 | harness_baseline        |         4 |        2 |            0.773861  |                    0.547723 |                    1         |                       1   |                                1 |                                 1 |                  1.25 |                         0    |                          2.5  |            -0.388889 |                   -0.388889 |                    -0.388889 |                    -1 |            71.25 |                   34    |                   108.5  |        1   |             1   |                      1 |                       1 |                          0.306351 |                                    0.5 |                               0.25 |                         -7.75 |                     0.5 |                          0.5 |               1   |                1   |                           -1 |                            -1 |
|   3 | task_harness_baseline   |         4 |        2 |            0.46751   |                    0.387298 |                    0.547723  |                       0.5 |                                0 |                                 1 |                  1.5  |                         0.5  |                          2.5  |            -0.444444 |                   -0.444444 |                    -0.444444 |                   nan |            63.5  |                   18.5  |                   108.5  |        0.5 |             0.5 |                      0 |                       1 |                          0        |                                    0   |                               0    |                          0    |                     0   |                          0   |               0.5 |                0.5 |                          nan |                           nan |
|   3 | early_fingerprint_model |         4 |        2 |            0.46751   |                    0.387298 |                    0.547723  |                       0.5 |                                0 |                                 1 |                  1.5  |                         0.5  |                          2.5  |            -0.444444 |                   -0.444444 |                    -0.444444 |                   nan |            63.5  |                   18.5  |                   108.5  |        0.5 |             0.5 |                      0 |                       1 |                          0        |                                    0   |                               0    |                          0    |                     0   |                          0   |               0.5 |                0.5 |                          nan |                           nan |
|   5 | global_mean_baseline    |         4 |        2 |            0.288675  |                    0.288675 |                    0.288675  |                       1   |                                1 |                                 1 |                  0.5  |                         0    |                          1    |            -1        |                   -1        |                    -1        |                   nan |            18    |                   14    |                    22    |        0.5 |             0.5 |                      0 |                       1 |                          0        |                                    0   |                               0    |                          0    |                     0   |                          0   |               0.5 |                0.5 |                          nan |                           nan |
|   5 | task_baseline           |         4 |        2 |            0.288675  |                    0.288675 |                    0.288675  |                       1   |                                1 |                                 1 |                  0.5  |                         0    |                          1    |            -1        |                   -1        |                    -1        |                   nan |            18    |                   14    |                    22    |        0.5 |             0.5 |                      0 |                       1 |                          0        |                                    0   |                               0    |                          0    |                     0   |                          0   |               0.5 |                0.5 |                          nan |                           nan |
|   5 | harness_baseline        |         4 |        2 |            0.288675  |                    0.288675 |                    0.288675  |                       1   |                                1 |                                 1 |                  0    |                         0    |                          0    |             1        |                    1        |                     1        |                     1 |             7    |                    0    |                    14    |        1   |             1   |                      1 |                       1 |                          0        |                                    0   |                               0.5  |                         11    |                     0.5 |                          0.5 |               1   |                1   |                            1 |                             1 |
|   5 | task_harness_baseline   |         4 |        2 |            0.288675  |                    0.288675 |                    0.288675  |                       1   |                                1 |                                 1 |                  0.5  |                         0    |                          1    |            -1        |                   -1        |                    -1        |                   nan |            18    |                   14    |                    22    |        0.5 |             0.5 |                      0 |                       1 |                          0        |                                    0   |                               0    |                          0    |                     0   |                          0   |               0.5 |                0.5 |                          nan |                           nan |
|   5 | early_fingerprint_model |         4 |        2 |            0.288675  |                    0.288675 |                    0.288675  |                       1   |                                1 |                                 1 |                  0.5  |                         0    |                          1    |            -1        |                   -1        |                    -1        |                   nan |            18    |                   14    |                    22    |        0.5 |             0.5 |                      0 |                       1 |                          0        |                                    0   |                               0    |                          0    |                     0   |                          0   |               0.5 |                0.5 |                          nan |                           nan |

## Object Reuse Summary

| reuse_class   |   object_accesses |   unique_objects |   reuse_events |   median_reuse_distance |   short_reuse_fraction_le3 |
|:--------------|------------------:|-----------------:|---------------:|------------------------:|---------------------------:|
| stable        |                41 |               21 |             20 |                       2 |                          1 |
| synthetic     |                33 |               29 |              3 |                       2 |                          1 |

## Success/Failure Metrics

| metric                |   success_mean |   failure_mean |   success_median |   failure_median |   delta_failure_minus_success |   mannwhitney_p |
|:----------------------|---------------:|---------------:|-----------------:|-----------------:|------------------------------:|----------------:|
| total_steps           |        5.5     |        6.5     |          5.5     |          6.5     |                      1        |       0.802587  |
| tool_entropy          |        2.28678 |        2.03878 |          2.32193 |          2.03878 |                     -0.247995 |       0.48112   |
| phase_entropy         |        1.80532 |        2.03878 |          1.92011 |          2.03878 |                      0.233459 |       0.218819  |
| observation_bytes     |      131.5     |      216       |        133.5     |        216       |                     84.5      |       0.533333  |
| repeated_action_ratio |        0.1     |        0       |          0       |          0       |                     -0.1      |       0.723674  |
| error_rate            |        0.05    |        0.55    |          0       |          0.55    |                      0.5      |       0.0851524 |

## Wafer Proxy Results

| strategy                 | mesh   |   island_size |   object_accesses |   island_homed_accesses |   avg_hops |   remote_object_accesses |   moved_bytes |   estimated_latency_units |   movement_reduction_vs_random |   latency_reduction_vs_random |
|:-------------------------|:-------|--------------:|------------------:|------------------------:|-----------:|-------------------------:|--------------:|--------------------------:|-------------------------------:|------------------------------:|
| baseline_random          | 5x5    |             0 |                74 |                       0 |    3.64865 |                       73 |          8987 |                   169.399 |                      0         |                     0         |
| session_affinity         | 5x5    |             0 |                74 |                       0 |    4.01351 |                       74 |          9768 |                   178.927 |                     -0.0869033 |                    -0.0562466 |
| oracle_island            | 5x5    |             3 |                74 |                      74 |    1.68919 |                       63 |          4088 |                   118.159 |                      0.545121  |                     0.302481  |
| early_fingerprint_island | 5x5    |             3 |                74 |                      73 |    1.68919 |                       63 |          4088 |                   118.159 |                      0.545121  |                     0.302481  |
| capacity_limited_island  | 5x5    |             3 |                74 |                      73 |    1.68919 |                       63 |          4088 |                   118.159 |                      0.545121  |                     0.302481  |
| wrong_prediction_stress  | 5x5    |             3 |                74 |                      70 |    1.71622 |                       63 |          4128 |                   118.863 |                      0.54067   |                     0.298325  |

## Conclusions That Became Weaker

- Wrapper locality is no longer treated as semantic locality. Strong wrapper self-loops are reported as harness behavior.
- Oracle island is explicitly labeled as an upper bound and is not used alone to support H5.
- Synthetic observation buckets are not counted as stable object reuse evidence.

## Still-Robust Signals

- If semantic/collapsed/phase recall remains above global baseline, H1 survives the wrapper-bias fix.
- If early fingerprint rows beat task/harness/global baselines for at least two K values, H2 has real early-state signal.
- If stable reuse events remain non-trivial with short median distance, H3 is supported by stable objects, not buckets.

## Remaining Limits

- Public traces are not hardware traces and lack KV/MoE expert IDs.
- Object inference is still rule-based; stable file/url/test-case IDs are approximate.
- Wafer proxy model is a relative movement estimator, not calibrated hardware replay.

## Next Steps

- Add real runtime KV/object IDs and MoE expert traces.
- Calibrate movement and latency terms against replay or simulator data.
- Add confidence intervals to transition and wafer metrics, and stratify by agent/model/task family.
