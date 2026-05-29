# Problem Existence Report

## 1. Research Question

We test, from scratch, whether dynamic agent workflows contain reusable static stateflow skeletons. No previous profile result is used as a premise.

## 2. Data and Cleaning

| dataset       |   num_traces |   num_steps |   num_tool_action_steps |   num_object_accesses | used_mock   | strict_real   |   skipped_rows |   unknown_rate |   artifact_rate |   command_parse_success_rate | source                | profile_mode       |
|:--------------|-------------:|------------:|------------------------:|----------------------:|:------------|:--------------|---------------:|---------------:|----------------:|-----------------------------:|:----------------------|:-------------------|
| terminalbench |         1500 |       81593 |                   48402 |                 80790 | False       | True          |              0 |       0.365583 |        0.364517 |                     0.656858 | datasets_server_cache | skeleton_existence |

Strict-real is `True` and `used_mock=false`.

## 3. Hypothesis Results

| hypothesis                 | verdict   |
|:---------------------------|:----------|
| H1 Tool/Phase Skeleton     | supported |
| H2 Cross-task Motifs       | supported |
| H3 Data Dependency         | supported |
| H4 Object Working Set      | supported |
| H5 Early Skeleton Matching | supported |
| H6 Failure-loop Skeleton   | supported |

H1 metric: semantic tool-action-only Top-3 conditional recall delta vs global = `0.2468`, CI = `[0.2273, 0.2697]`. Baseline: global next-tool frequency. Negative controls: within-trace shuffle, global next-label shuffle, frequency-preserving temporal break.

H2 metric: held-out motif reproduction from train-mined motifs. Baseline/control: held-out trace split and harness support count. See `outputs/skeleton_existence_tb1500/tables/motif_generalization.csv`.

H3 metric: dependency count `47452`, median dependency distance `5.00`. Baseline/control: observation-only path mentions are not counted; dependency requires later command use.

H4 metric: object prediction baselines at exact object-id and path-prefix granularity. Main comparison: last-K exact objects vs random and object_type-only. See `outputs/skeleton_existence_tb1500/tables/object_prediction_baselines.csv`.

H5 metric: early K=1/2/3/5/8 prediction delta vs best metadata baseline. See `outputs/skeleton_existence_tb1500/tables/early_vs_metadata_baseline.csv`.

H6 metric: failure_loop_score success vs failure, Mann-Whitney p-value `0.01198`. Baseline: success traces.

## 4. Static Skeleton Evidence

- Phase/tool skeleton: see H1 transition recall, entropy/MI, permutation tests.
- Cross-task motif skeleton: see frequent motifs and held-out support.
- Data dependency skeleton: file/url/test/error dependencies require future command use.
- Object working set skeleton: exact object_id and path_prefix are analyzed separately from object_type.
- KV skeleton opportunity: estimated only as repeated prompt/prefix opportunity proxy, not as proven KV reuse.
- Failure-loop skeleton: compared with success/failure statistics, not observation-byte volume.

## 5. Problem Severity

| strategy                    |   estimated_cost |   reduction_vs_independent |
|:----------------------------|-----------------:|---------------------------:|
| independent_call            |      2.08095e+08 |                0           |
| session_affinity            |      2.07865e+08 |           229518           |
| skeleton_oracle_upper_bound |      1.50743e+08 |                5.73514e+07 |
| skeleton_predicted          |      1.89653e+08 |                1.84412e+07 |
| wrong_prediction_stress     |      2.079e+08   |           195090           |
| object_type_only            |      2.07877e+08 |           218042           |

Predicted skeleton-aware opportunity is `18441182.74` vs session-affinity `229517.60` and wrong-prediction stress `195089.96`.

## 6. What Is Proven

Only hypotheses with `supported` in the table above should be treated as proven for this TerminalBench sample.

## 7. What Is Not Proven

- Real wafer acceleration is not proven.
- Object/KV reuse in a production runtime is not proven.
- Generality across all agent domains is not proven.
- Speculation safety is not proven.
- End-to-end task success preservation is not proven.

## 8. Next Step

问题存在。动态 agent workflow 中存在可利用的静态 stateflow skeleton，可以进入 SkeletonFlow 算法设计阶段。
