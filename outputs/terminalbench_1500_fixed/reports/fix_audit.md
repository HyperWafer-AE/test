# Agent Trace Profiling Fix Audit

## Run

- commit hash: `6e00f4aae611620db4f38ebc8d16adaf0a06951b`
- command: `python scripts/run_all.py --datasets terminalbench --sample-size 1500 --strict-real --outdir /home/duzc/data/agent_wafer/outputs/terminalbench_1500_fixed`
- used_mocks: `{'terminalbench': False}`
- warnings_summary: `terminalbench: skipped 773 rows with empty/missing steps.`

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
| H2 early fingerprint vs baselines | 支持 |
| H3 stable object locality | 支持 |
| H4 success/failure differences | 支持 |
| H5 prediction-aware wafer proxy | 支持 |

## Transition Recall By View

| view                    |   k |   conditional_recall |   global_recall |   delta_vs_global |   n_test |
|:------------------------|----:|---------------------:|----------------:|------------------:|---------:|
| wrapper_tool            |   1 |             0.864777 |        0.453254 |        0.411524   |    24419 |
| wrapper_tool            |   3 |             0.955567 |        0.843073 |        0.112494   |    24419 |
| wrapper_tool            |   5 |             0.970474 |        0.912404 |        0.0580695  |    24419 |
| semantic_tool           |   1 |             0.630124 |        0.462058 |        0.168066   |    24419 |
| semantic_tool           |   3 |             0.737377 |        0.547402 |        0.189975   |    24419 |
| semantic_tool           |   5 |             0.779434 |        0.619149 |        0.160285   |    24419 |
| collapsed_semantic_tool |   1 |             0.314428 |        0.235821 |        0.078607   |     4020 |
| collapsed_semantic_tool |   3 |             0.507711 |        0.358209 |        0.149502   |     4020 |
| collapsed_semantic_tool |   5 |             0.599502 |        0.466915 |        0.132587   |     4020 |
| phase                   |   1 |             0.744912 |        0.523977 |        0.220935   |    24419 |
| phase                   |   3 |             0.942872 |        0.900037 |        0.0428355  |    24419 |
| phase                   |   5 |             0.997666 |        0.997666 |        0          |    24419 |
| collapsed_phase         |   1 |             0.433831 |        0.449005 |       -0.0151741  |     4020 |
| collapsed_phase         |   3 |             0.879353 |        0.852985 |        0.0263682  |     4020 |
| collapsed_phase         |   5 |             0.995522 |        0.99801  |       -0.00248756 |     4020 |

## Fingerprint Metrics By Baseline

|   K | baseline                |   n_train |   n_test |   future_tool_cosine |   future_tool_cosine_ci_low |   future_tool_cosine_ci_high |   future_tool_top3_recall |   future_tool_top3_recall_ci_low |   future_tool_top3_recall_ci_high |   remaining_steps_mae |   remaining_steps_mae_ci_low |   remaining_steps_mae_ci_high |   remaining_steps_r2 |   remaining_steps_r2_ci_low |   remaining_steps_r2_ci_high |   future_obs_spearman |   future_obs_mae |   future_obs_mae_ci_low |   future_obs_mae_ci_high |   long_auc |   long_auc_ci_low |   long_auc_ci_high |   long_accuracy |   long_accuracy_ci_low |   long_accuracy_ci_high |   future_tool_cosine_delta_global |   future_tool_top3_recall_delta_global |   remaining_steps_mae_delta_global |   future_obs_mae_delta_global |   long_auc_delta_global |   long_accuracy_delta_global |   future_obs_spearman_ci_low |   future_obs_spearman_ci_high |
|----:|:------------------------|----------:|---------:|---------------------:|----------------------------:|-----------------------------:|--------------------------:|---------------------------------:|----------------------------------:|----------------------:|-----------------------------:|------------------------------:|---------------------:|----------------------------:|-----------------------------:|----------------------:|-----------------:|------------------------:|-------------------------:|-----------:|------------------:|-------------------:|----------------:|-----------------------:|------------------------:|----------------------------------:|---------------------------------------:|-----------------------------------:|------------------------------:|------------------------:|-----------------------------:|-----------------------------:|------------------------------:|
|   1 | global_mean_baseline    |      1050 |      450 |             0.531631 |                    0.514496 |                     0.552836 |                  0.523385 |                         0.486582 |                          0.57018  |               52.815  |                      35.8877 |                       74.9431 |         -0.00061847  |                 -0.0578061  |                 -5.01571e-06 |           nan         |          8447.52 |                 7192.16 |                  9707.9  |   0.5      |          0.5      |           0.5      |        0.748889 |               0.713333 |                0.791222 |                         0         |                              0         |                           0        |                         0     |                0        |                   0          |                  nan         |                    nan        |
|   1 | task_baseline           |      1050 |      450 |             0.603885 |                    0.588309 |                     0.620688 |                  0.710468 |                         0.668815 |                          0.750014 |               53.0659 |                      37.8144 |                       75.5596 |         -0.00773079  |                 -0.780325   |                  0.0141807   |             0.273796  |          7941.18 |                 6744.43 |                  9176.29 |   0.690252 |          0.635231 |           0.740835 |        0.746667 |               0.704444 |                0.791167 |                         0.072254  |                              0.187082  |                          -0.250937 |                       506.341 |                0.190252 |                  -0.00222222 |                    0.179191  |                      0.359564 |
|   1 | harness_baseline        |      1050 |      450 |             0.606192 |                    0.589064 |                     0.625859 |                  0.603563 |                         0.555773 |                          0.653425 |               47.0314 |                      31.3559 |                       68.5556 |          0.0627787   |                  0.00151756 |                  0.0727991   |             0.157903  |          8288.97 |                 7050.38 |                  9545.94 |   0.778433 |          0.73025  |           0.812338 |        0.793333 |               0.764278 |                0.8245   |                         0.0745605 |                              0.0801782 |                           5.78357  |                       158.551 |                0.278433 |                   0.0444444  |                    0.0800268 |                      0.260145 |
|   1 | task_harness_baseline   |      1050 |      450 |             0.658749 |                    0.639551 |                     0.678579 |                  0.672606 |                         0.628027 |                          0.71558  |               46.6896 |                      30.8817 |                       70.1794 |          0.0472516   |                 -1.87238    |                  0.234914    |             0.318208  |          7411.94 |                 6303.98 |                  8548.88 |   0.834721 |          0.781377 |           0.872159 |        0.813333 |               0.779944 |                0.842278 |                         0.127117  |                              0.14922   |                           6.12535  |                      1035.58  |                0.334721 |                   0.0644444  |                    0.240754  |                      0.399022 |
|   1 | early_fingerprint_model |      1050 |      450 |             0.584714 |                    0.565215 |                     0.602709 |                  0.547884 |                         0.507795 |                          0.6      |               50.778  |                      33.3466 |                       72.8061 |          0.00907235  |                 -0.00458542 |                  0.0140074   |             0.305056  |          5837.4  |                 5051.95 |                  6729.15 |   0.683005 |          0.633078 |           0.723896 |        0.78     |               0.746667 |                0.811222 |                         0.0530831 |                              0.0244989 |                           2.03697  |                      2610.12  |                0.183005 |                   0.0311111  |                    0.207428  |                      0.398095 |
|   2 | global_mean_baseline    |      1050 |      450 |             0.439144 |                    0.421456 |                     0.455189 |                  0.462054 |                         0.422819 |                          0.507891 |               49.3836 |                      35.128  |                       69.3934 |         -3.5776e-05  |                 -0.0866995  |                 -6.81061e-06 |           nan         |          8299.84 |                 6873.16 |                  9731.35 |   0.5      |          0.5      |           0.5      |        0.757778 |               0.72     |                0.793333 |                         0         |                              0         |                           0        |                         0     |                0        |                   0          |                  nan         |                    nan        |
|   2 | task_baseline           |      1050 |      450 |             0.553432 |                    0.536044 |                     0.570733 |                  0.607143 |                         0.563751 |                          0.654072 |               49.67   |                      35.0001 |                       70.1127 |         -0.00500365  |                 -0.567619   |                  0.0254377   |             0.248389  |          8030.09 |                 6702.71 |                  9443.57 |   0.709529 |          0.655487 |           0.755895 |        0.771111 |               0.735556 |                0.811167 |                         0.114287  |                              0.145089  |                          -0.286448 |                       269.749 |                0.209529 |                   0.0133333  |                    0.151085  |                      0.343137 |
|   2 | harness_baseline        |      1050 |      450 |             0.561385 |                    0.544453 |                     0.583714 |                  0.517857 |                         0.478667 |                          0.564788 |               44.4189 |                      30.5449 |                       64.9918 |          0.0587763   |                 -0.300833   |                  0.070486    |             0.124054  |          8045.81 |                 6644.38 |                  9307.69 |   0.739164 |          0.68637  |           0.784525 |        0.775556 |               0.739889 |                0.813389 |                         0.122241  |                              0.0558036 |                           4.96463  |                       254.028 |                0.239164 |                   0.0177778  |                    0.036426  |                      0.214661 |
|   2 | task_harness_baseline   |      1050 |      450 |             0.638937 |                    0.619873 |                     0.660889 |                  0.6875   |                         0.647318 |                          0.728944 |               38.2494 |                      23.1988 |                       59.184  |          0.0567076   |                 -0.75053    |                  0.155786    |             0.354485  |          7662.52 |                 6409.05 |                  9003.24 |   0.8218   |          0.778541 |           0.86307  |        0.815556 |               0.784444 |                0.846722 |                         0.199792  |                              0.225446  |                          11.1341   |                       637.323 |                0.3218   |                   0.0577778  |                    0.251062  |                      0.44536  |
|   2 | early_fingerprint_model |      1050 |      450 |             0.582356 |                    0.562245 |                     0.603131 |                  0.564732 |                         0.523378 |                          0.610356 |               49.5019 |                      34.6273 |                       69.5139 |          0.00094135  |                 -1.00253    |                  0.0490392   |             0.425426  |          5699.36 |                 4851.95 |                  6471.65 |   0.713108 |          0.654058 |           0.766576 |        0.762222 |               0.722222 |                0.800056 |                         0.143212  |                              0.102679  |                          -0.118332 |                      2600.48  |                0.213108 |                   0.00444444 |                    0.338616  |                      0.502911 |
|   3 | global_mean_baseline    |      1050 |      450 |             0.424079 |                    0.4043   |                     0.440646 |                  0.460137 |                         0.405033 |                          0.503416 |               48.8581 |                      37.1531 |                       66.5126 |         -1.82633e-05 |                 -0.0701967  |                 -1.04837e-06 |           nan         |          7636.41 |                 6602.92 |                  8652.62 |   0.5      |          0.5      |           0.5      |        0.746667 |               0.706667 |                0.784444 |                         0         |                              0         |                           0        |                         0     |                0        |                   0          |                  nan         |                    nan        |
|   3 | task_baseline           |      1050 |      450 |             0.525795 |                    0.504596 |                     0.545689 |                  0.596811 |                         0.537555 |                          0.6419   |               50.4835 |                      38.3595 |                       69.6689 |          0.00401656  |                 -0.46694    |                  0.0458821   |             0.287974  |          7090.38 |                 6114.02 |                  8063.4  |   0.662046 |          0.613924 |           0.721599 |        0.755556 |               0.7155   |                0.793389 |                         0.101716  |                              0.136674  |                          -1.6254   |                       546.03  |                0.162046 |                   0.00888889 |                    0.219475  |                      0.374539 |
|   3 | harness_baseline        |      1050 |      450 |             0.553797 |                    0.53088  |                     0.572402 |                  0.551253 |                         0.498776 |                          0.597702 |               44.749  |                      32.3296 |                       62.7712 |          0.0690291   |                 -0.105967   |                  0.0928133   |             0.0687051 |          7496.98 |                 6488.97 |                  8569.85 |   0.778874 |          0.73186  |           0.826713 |        0.808889 |               0.766611 |                0.844444 |                         0.129718  |                              0.0911162 |                           4.10916  |                       139.43  |                0.278874 |                   0.0622222  |                   -0.0244547 |                      0.168796 |
|   3 | task_harness_baseline   |      1050 |      450 |             0.620064 |                    0.595654 |                     0.641002 |                  0.660592 |                         0.623174 |                          0.698475 |               37.7489 |                      25.941  |                       55.3648 |          0.117848    |                 -0.57659    |                  0.357878    |             0.284229  |          7170.73 |                 6195.75 |                  8103.95 |   0.830174 |          0.776781 |           0.873344 |        0.84     |               0.806667 |                0.877778 |                         0.195985  |                              0.200456  |                          11.1092   |                       465.686 |                0.330174 |                   0.0933333  |                    0.209202  |                      0.373176 |
|   3 | early_fingerprint_model |      1050 |      450 |             0.60171  |                    0.57644  |                     0.622296 |                  0.603645 |                         0.559745 |                          0.643239 |               41.1172 |                      29.0398 |                       59.9896 |          0.00199629  |                 -0.611322   |                  0.0732524   |             0.501521  |          5201.35 |                 4478.68 |                  5881.82 |   0.848828 |          0.802152 |           0.891894 |        0.826667 |               0.795444 |                0.855667 |                         0.177631  |                              0.143508  |                           7.74088  |                      2435.06  |                0.348828 |                   0.08       |                    0.417072  |                      0.563266 |
|   5 | global_mean_baseline    |      1050 |      450 |             0.414512 |                    0.398816 |                     0.428943 |                  0.416873 |                         0.374206 |                          0.460443 |               53.1717 |                      35.779  |                       73.7898 |         -0.000867753 |                 -0.053775   |                 -3.61952e-06 |           nan         |          7251.7  |                 6293.66 |                  8384.9  |   0.5      |          0.5      |           0.5      |        0.751111 |               0.7155   |                0.791167 |                         0         |                              0         |                           0        |                         0     |                0        |                   0          |                  nan         |                    nan        |
|   5 | task_baseline           |      1050 |      450 |             0.499155 |                    0.476406 |                     0.519062 |                  0.513648 |                         0.458751 |                          0.556702 |               52.4125 |                      34.8953 |                       72.7817 |         -0.0102608   |                 -0.534279   |                  0.0148895   |             0.255229  |          6834.72 |                 5895.29 |                  7940.13 |   0.669947 |          0.617312 |           0.721436 |        0.766667 |               0.735444 |                0.806722 |                         0.084643  |                              0.0967742 |                           0.759152 |                       416.986 |                0.169947 |                   0.0155556  |                    0.16506   |                      0.346238 |
|   5 | harness_baseline        |      1050 |      450 |             0.550367 |                    0.530405 |                     0.57407  |                  0.560794 |                         0.517218 |                          0.606673 |               48.4717 |                      31.5854 |                       68.6614 |          0.0612196   |                 -0.160566   |                  0.0699319   |             0.0979851 |          7113.44 |                 6124.52 |                  8184.94 |   0.797192 |          0.752483 |           0.847123 |        0.813333 |               0.782167 |                0.848889 |                         0.135854  |                              0.143921  |                           4.69996  |                       138.267 |                0.297192 |                   0.0622222  |                    0.0203089 |                      0.195179 |
|   5 | task_harness_baseline   |      1050 |      450 |             0.593001 |                    0.56995  |                     0.616516 |                  0.647643 |                         0.599975 |                          0.695024 |               43.7517 |                      25.3119 |                       66.1351 |          0.0201751   |                 -0.322817   |                  0.105796    |             0.280047  |          6653.36 |                 5750.92 |                  7519.05 |   0.852375 |          0.809441 |           0.897377 |        0.857778 |               0.824333 |                0.888889 |                         0.178489  |                              0.230769  |                           9.41995  |                       598.345 |                0.352375 |                   0.106667   |                    0.202588  |                      0.36717  |
|   5 | early_fingerprint_model |      1050 |      450 |             0.619668 |                    0.59734  |                     0.641837 |                  0.667494 |                         0.61807  |                          0.715064 |               40.1033 |                      23.4074 |                       59.8992 |          0.0704374   |                  0.0262094  |                  0.127401    |             0.574591  |          4649.26 |                 3912.25 |                  5297.18 |   0.858992 |          0.821593 |           0.901102 |        0.842222 |               0.808833 |                0.875556 |                         0.205156  |                              0.25062   |                          13.0683   |                      2602.44  |                0.358992 |                   0.0911111  |                    0.512385  |                      0.65444  |

## Object Reuse Summary

| reuse_class   |   object_accesses |   unique_objects |   reuse_events |   median_reuse_distance |   short_reuse_fraction_le3 |
|:--------------|------------------:|-----------------:|---------------:|------------------------:|---------------------------:|
| stable        |             37945 |             4503 |          25524 |                       1 |                   0.790668 |
| synthetic     |             38281 |            19457 |           6592 |                       2 |                   0.634405 |

## Success/Failure Metrics

| metric                |   success_mean |   failure_mean |   success_median |   failure_median |   delta_failure_minus_success |   mannwhitney_p |
|:----------------------|---------------:|---------------:|-----------------:|-----------------:|------------------------------:|----------------:|
| total_steps           |     30.987     |      61.9168   |       21         |        23        |                    30.9299    |     0.266876    |
| tool_entropy          |      2.59527   |       2.44729  |        2.64136   |         2.52164  |                    -0.147979  |     0.00572749  |
| phase_entropy         |      1.48658   |       1.31332  |        1.56448   |         1.41085  |                    -0.173264  |     3.60846e-12 |
| observation_bytes     |   5607.16      |    9242.99     |     3515.5       |      3164        |                  3635.84      |     0.774321    |
| repeated_action_ratio |      0.324226  |       0.36185  |        0.274159  |         0.333333 |                     0.0376239 |     0.0281111   |
| error_rate            |      0.0872028 |       0.118396 |        0.0526316 |         0.065942 |                     0.0311934 |     0.00592715  |

## Wafer Proxy Results

| strategy                 | mesh   |   island_size |   object_accesses |   island_homed_accesses |   avg_hops |   remote_object_accesses |   moved_bytes |   estimated_latency_units |   movement_reduction_vs_random |   latency_reduction_vs_random |
|:-------------------------|:-------|--------------:|------------------:|------------------------:|-----------:|-------------------------:|--------------:|--------------------------:|-------------------------------:|------------------------------:|
| baseline_random          | 5x5    |             0 |             76226 |                       0 |    3.2033  |                    73181 |   1.21898e+08 |                    173877 |                     0          |                    0          |
| session_affinity         | 5x5    |             0 |             76226 |                       0 |    3.25563 |                    73154 |   1.23052e+08 |                    175389 |                    -0.00947043 |                   -0.00869346 |
| oracle_island            | 5x5    |             3 |             76226 |                   76226 |    1.78063 |                    67685 |   6.75622e+07 |                    130488 |                     0.445747   |                    0.24954    |
| early_fingerprint_island | 5x5    |             3 |             76226 |                   46548 |    2.23384 |                    69826 |   8.23118e+07 |                    144054 |                     0.324748   |                    0.171517   |
| capacity_limited_island  | 5x5    |             3 |             76226 |                   40331 |    2.33035 |                    70305 |   8.8801e+07  |                    147278 |                     0.271513   |                    0.152978   |
| wrong_prediction_stress  | 5x5    |             3 |             76226 |                   43245 |    2.2777  |                    70047 |   8.5341e+07  |                    145527 |                     0.299897   |                    0.163046   |

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
