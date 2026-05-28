# Agent Trace Profile Report

生成时间由本地运行决定。命令参数：

```bash
python scripts/run_all.py --datasets mock --sample-size 10 --outdir /home/duzc/data/agent_wafer/outputs/mock_fixed
```

## 数据集和样本量

请求数据集：`mock`  
加载模式：`streaming=True`, `offline_mock=True`  
真实/兜底状态：`mock=mock`  
Canonical rows：traces=6, steps=35, object_accesses=74

| dataset       |   num_traces |   num_steps |   success_rate |   median_steps |   mean_steps |   median_obs_chars_per_trace |   mean_obs_chars_per_trace |
|:--------------|-------------:|------------:|---------------:|---------------:|-------------:|-----------------------------:|---------------------------:|
| swe_agent     |            2 |          10 |           0.5  |              5 |         5    |                          123 |                        123 |
| terminalbench |            4 |          25 |           0.75 |              6 |         6.25 |                          157 |                        178 |

Canonical CSV：

- `../data/normalized/traces.csv`
- `../data/normalized/steps.csv`
- `../data/normalized/object_accesses.csv`

## Hypothesis Verdicts

| Hypothesis | Verdict | Evidence |
|---|---:|---|
| H1: tool/phase 时间局部性 | 支持 | 最高频 semantic-tool 转移是 `edit -> pytest`，count=2，P=1.000。 wrapper/semantic/collapsed/phase 分层见 `../tables/transition_recall_by_view.csv` 和 transition heatmaps。 |
| H2: early fingerprint 可预测未来 | 不支持 | 对比 global/task/harness/task+harness baselines，见 `../tables/fingerprint_metrics_by_baseline.csv` 和 `../figures/early_fingerprint_vs_baselines.png`。 |
| H3: stable object locality | 部分支持 | observation p50=32.0 chars, p95=53.0 chars；stable reuse events=20, median=2.00；synthetic bucket reuse events=3, median=2.00。见 `../tables/object_reuse_summary.csv`。 |
| H4: 成功/失败轨迹状态成本不同 | 支持 | step 数、tool entropy、observation bytes、重复 action、error rate 的分组统计见 `../tables/success_failure_metrics.csv` 和 `../figures/success_vs_failure_bars.png`。 |
| H5: prediction-aware Agent Island | 支持 | oracle_island 是上界；真正判断 early_fingerprint/capacity_limited/wrong_prediction，见 `../tables/wafer_proxy_results.csv` 和 `../figures/wafer_proxy_strategy_comparison.png`。 |

## 关键发现

1. Tool locality：条件转移预测的 top-k recall 如下。H1 只根据 semantic_tool/phase/collapsed 视图判断；wrapper_tool 只作为 harness bias 对照。

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

2. Early fingerprint：前 1/2/3/5 步模型必须优于 global/task/harness/task+harness baseline 才算证据。

|   K | baseline                |   future_tool_cosine |   future_tool_cosine_delta_global |   future_tool_top3_recall |   long_auc |   long_auc_delta_global |   long_accuracy |
|----:|:------------------------|---------------------:|----------------------------------:|--------------------------:|-----------:|------------------------:|----------------:|
|   1 | global_mean_baseline    |            0.313013  |                          0        |                       0   |      nan   |                   nan   |             0   |
|   1 | task_baseline           |            0.313013  |                          0        |                       0   |      nan   |                   nan   |             0   |
|   1 | harness_baseline        |            0.313013  |                          0        |                       0   |      nan   |                   nan   |             0   |
|   1 | task_harness_baseline   |            0.313013  |                          0        |                       0   |      nan   |                   nan   |             0   |
|   1 | early_fingerprint_model |            0.313013  |                          0        |                       0   |      nan   |                   nan   |             0   |
|   2 | global_mean_baseline    |            0.0394464 |                          0        |                       0   |      nan   |                   nan   |             0   |
|   2 | task_baseline           |            0.0394464 |                          0        |                       0   |      nan   |                   nan   |             0   |
|   2 | harness_baseline        |            0.0394464 |                          0        |                       0   |      nan   |                   nan   |             0   |
|   2 | task_harness_baseline   |            0.0394464 |                          0        |                       0   |      nan   |                   nan   |             0   |
|   2 | early_fingerprint_model |            0.0394464 |                          0        |                       0   |      nan   |                   nan   |             0   |
|   3 | global_mean_baseline    |            0.46751   |                          0        |                       0.5 |        0.5 |                     0   |             0.5 |
|   3 | task_baseline           |            0.46751   |                          0        |                       0.5 |        0.5 |                     0   |             0.5 |
|   3 | harness_baseline        |            0.773861  |                          0.306351 |                       1   |        1   |                     0.5 |             1   |
|   3 | task_harness_baseline   |            0.46751   |                          0        |                       0.5 |        0.5 |                     0   |             0.5 |
|   3 | early_fingerprint_model |            0.46751   |                          0        |                       0.5 |        0.5 |                     0   |             0.5 |
|   5 | global_mean_baseline    |            0.288675  |                          0        |                       1   |        0.5 |                     0   |             0.5 |
|   5 | task_baseline           |            0.288675  |                          0        |                       1   |        0.5 |                     0   |             0.5 |
|   5 | harness_baseline        |            0.288675  |                          0        |                       1   |        1   |                     0.5 |             1   |
|   5 | task_harness_baseline   |            0.288675  |                          0        |                       1   |        0.5 |                     0   |             0.5 |
|   5 | early_fingerprint_model |            0.288675  |                          0        |                       1   |        0.5 |                     0   |             0.5 |

3. Object locality：stable object reuse 只统计 file/url/test_case 等稳定对象；large_observation_bucket、observation hash、test_log hash 单独归为 synthetic bucket，不作为真实对象复用证据。

4. 成功/失败差异：

| metric                |   success_mean |   failure_mean |   success_median |   failure_median |   delta_failure_minus_success |   mannwhitney_p |
|:----------------------|---------------:|---------------:|-----------------:|-----------------:|------------------------------:|----------------:|
| total_steps           |        5.5     |        6.5     |          5.5     |          6.5     |                      1        |       0.802587  |
| tool_entropy          |        2.28678 |        2.03878 |          2.32193 |          2.03878 |                     -0.247995 |       0.48112   |
| phase_entropy         |        1.80532 |        2.03878 |          1.92011 |          2.03878 |                      0.233459 |       0.218819  |
| observation_bytes     |      131.5     |      216       |        133.5     |        216       |                     84.5      |       0.533333  |
| repeated_action_ratio |        0.1     |        0       |          0       |          0       |                     -0.1      |       0.723674  |
| error_rate            |        0.05    |        0.55    |          0       |          0.55    |                      0.5      |       0.0851524 |

5. Wafer proxy movement：oracle_island 是 co-location 上界，不代表可实现 predictor。重点看 early_fingerprint_island 和 capacity_limited_island 是否优于 session_affinity。

| strategy                 | mesh   |   island_size |   object_accesses |   island_homed_accesses |   avg_hops |   remote_object_accesses |   moved_bytes |   estimated_latency_units |   movement_reduction_vs_random |   latency_reduction_vs_random |
|:-------------------------|:-------|--------------:|------------------:|------------------------:|-----------:|-------------------------:|--------------:|--------------------------:|-------------------------------:|------------------------------:|
| baseline_random          | 5x5    |             0 |                74 |                       0 |    3.64865 |                       73 |          8987 |                   169.399 |                      0         |                     0         |
| session_affinity         | 5x5    |             0 |                74 |                       0 |    4.01351 |                       74 |          9768 |                   178.927 |                     -0.0869033 |                    -0.0562466 |
| oracle_island            | 5x5    |             3 |                74 |                      74 |    1.68919 |                       63 |          4088 |                   118.159 |                      0.545121  |                     0.302481  |
| early_fingerprint_island | 5x5    |             3 |                74 |                      73 |    1.68919 |                       63 |          4088 |                   118.159 |                      0.545121  |                     0.302481  |
| capacity_limited_island  | 5x5    |             3 |                74 |                      73 |    1.68919 |                       63 |          4088 |                   118.159 |                      0.545121  |                     0.302481  |
| wrong_prediction_stress  | 5x5    |             3 |                74 |                      70 |    1.71622 |                       63 |          4128 |                   118.863 |                      0.54067   |                     0.298325  |

## 输出索引

Figures：

- `../figures/trajectory_length_cdf.png`
- `../figures/tool_calls_per_trace_cdf.png`
- `../figures/observation_size_cdf.png`
- `../figures/wrapper_tool_transition_heatmap.png`
- `../figures/semantic_tool_transition_heatmap.png`
- `../figures/collapsed_semantic_tool_transition_heatmap.png`
- `../figures/tool_transition_heatmap.png`
- `../figures/phase_transition_heatmap.png`
- `../figures/topk_next_tool_recall_by_view.png`
- `../figures/early_fingerprint_predictability.png`
- `../figures/early_fingerprint_vs_baselines.png`
- `../figures/stable_object_reuse_distance_cdf.png`
- `../figures/synthetic_bucket_reuse_distance_cdf.png`
- `../figures/success_vs_failure_bars.png`
- `../figures/wafer_proxy_movement_reduction.png`
- `../figures/wafer_proxy_strategy_comparison.png`

Tables：

- `../tables/dataset_summary.csv`
- `../tables/top_tools.csv`
- `../tables/wrapper_tool_transition_top_pairs.csv`
- `../tables/semantic_tool_transition_top_pairs.csv`
- `../tables/collapsed_semantic_tool_transition_top_pairs.csv`
- `../tables/phase_transition_matrix.csv`
- `../tables/transition_recall_by_view.csv`
- `../tables/fingerprint_metrics_by_baseline.csv`
- `../tables/object_reuse_summary.csv`
- `../tables/success_failure_metrics.csv`
- `../tables/wafer_proxy_results.csv`

## 限制

- 公开 agent trace 不是生产底层硬件 trace，不能直接推出真实 wafer runtime 的延迟。
- 当前数据没有真实 KV cache object、MoE expert ID、die placement 或 D2D traffic counter。
- Observation 可能被 harness 截断或格式化，`observation_text` 里的对象识别是规则近似。
- Wafer simulator 是 trace-level proxy model，只估计相对 movement/hops，不代表物理网络、cache coherence、DMA 或调度开销。
- 若本次运行使用 mock fallback，报告只验证 pipeline 可运行，不应作为论文证据。

## 下一步

1. 补采开源 MoE expert trace 和 KV/cache access trace，把 `ObjectAccess` 从文本规则升级为真实 runtime object。
2. 加入真实 latency replay：按工具调用、observation size、KV movement、D2D hops 校准 simulator。
3. 扩展到 ClawsBench 和 Applied Compute released jsonl，并把 schema adapter 做成插件式 loader。
4. 对 TerminalBench/SWE-agent 做更大样本的 bootstrap 置信区间和跨模型/跨 agent 分层分析。

## Warnings

- mock: using bundled TerminalBench/SWE-agent style rows.
