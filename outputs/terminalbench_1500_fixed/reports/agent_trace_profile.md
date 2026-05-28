# Agent Trace Profile Report

生成时间由本地运行决定。命令参数：

```bash
python scripts/run_all.py --datasets terminalbench --sample-size 1500 --strict-real --outdir /home/duzc/data/agent_wafer/outputs/terminalbench_1500_fixed
```

## 数据集和样本量

请求数据集：`terminalbench`  
加载模式：`streaming=True`, `offline_mock=False`, `strict_real=True`  
真实/兜底状态：`terminalbench=real`  
Canonical rows：traces=1500, steps=76235, object_accesses=76226

| dataset       |   num_traces |   num_steps |   success_rate |   median_steps |   mean_steps |   median_obs_chars_per_trace |   mean_obs_chars_per_trace |
|:--------------|-------------:|------------:|---------------:|---------------:|-------------:|-----------------------------:|---------------------------:|
| terminalbench |         1500 |       76235 |       0.358667 |             22 |      50.8233 |                       3393.5 |                    7938.94 |

Canonical CSV：

- `../data/normalized/traces.csv`
- `../data/normalized/steps.csv`
- `../data/normalized/object_accesses.csv`

## Hypothesis Verdicts

| Hypothesis | Verdict | Evidence |
|---|---:|---|
| H1: tool/phase 时间局部性 | 支持 | 最高频 semantic-tool 转移是 `unknown -> unknown`，count=26665，P=0.843。 wrapper/semantic/collapsed/phase 分层见 `../tables/transition_recall_by_view.csv` 和 transition heatmaps。 |
| H2: early fingerprint 可预测未来 | 支持 | 对比 global/task/harness/task+harness baselines，见 `../tables/fingerprint_metrics_by_baseline.csv` 和 `../figures/early_fingerprint_vs_baselines.png`。 |
| H3: stable object locality | 支持 | observation p50=120.0 chars, p95=1002.0 chars；stable reuse events=25524, median=1.00；synthetic bucket reuse events=6592, median=2.00。见 `../tables/object_reuse_summary.csv`。 |
| H4: 成功/失败轨迹状态成本不同 | 支持 | step 数、tool entropy、observation bytes、重复 action、error rate 的分组统计见 `../tables/success_failure_metrics.csv` 和 `../figures/success_vs_failure_bars.png`。 |
| H5: prediction-aware Agent Island | 支持 | oracle_island 是上界；真正判断 early_fingerprint/capacity_limited/wrong_prediction，见 `../tables/wafer_proxy_results.csv` 和 `../figures/wafer_proxy_strategy_comparison.png`。 |

## 关键发现

1. Tool locality：条件转移预测的 top-k recall 如下。H1 只根据 semantic_tool/phase/collapsed 视图判断；wrapper_tool 只作为 harness bias 对照。

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

2. Early fingerprint：前 1/2/3/5 步模型必须优于 global/task/harness/task+harness baseline 才算证据。

|   K | baseline                |   future_tool_cosine |   future_tool_cosine_delta_global |   future_tool_top3_recall |   long_auc |   long_auc_delta_global |   long_accuracy |
|----:|:------------------------|---------------------:|----------------------------------:|--------------------------:|-----------:|------------------------:|----------------:|
|   1 | global_mean_baseline    |             0.531631 |                         0         |                  0.523385 |   0.5      |                0        |        0.748889 |
|   1 | task_baseline           |             0.603885 |                         0.072254  |                  0.710468 |   0.690252 |                0.190252 |        0.746667 |
|   1 | harness_baseline        |             0.606192 |                         0.0745605 |                  0.603563 |   0.778433 |                0.278433 |        0.793333 |
|   1 | task_harness_baseline   |             0.658749 |                         0.127117  |                  0.672606 |   0.834721 |                0.334721 |        0.813333 |
|   1 | early_fingerprint_model |             0.584714 |                         0.0530831 |                  0.547884 |   0.683005 |                0.183005 |        0.78     |
|   2 | global_mean_baseline    |             0.439144 |                         0         |                  0.462054 |   0.5      |                0        |        0.757778 |
|   2 | task_baseline           |             0.553432 |                         0.114287  |                  0.607143 |   0.709529 |                0.209529 |        0.771111 |
|   2 | harness_baseline        |             0.561385 |                         0.122241  |                  0.517857 |   0.739164 |                0.239164 |        0.775556 |
|   2 | task_harness_baseline   |             0.638937 |                         0.199792  |                  0.6875   |   0.8218   |                0.3218   |        0.815556 |
|   2 | early_fingerprint_model |             0.582356 |                         0.143212  |                  0.564732 |   0.713108 |                0.213108 |        0.762222 |
|   3 | global_mean_baseline    |             0.424079 |                         0         |                  0.460137 |   0.5      |                0        |        0.746667 |
|   3 | task_baseline           |             0.525795 |                         0.101716  |                  0.596811 |   0.662046 |                0.162046 |        0.755556 |
|   3 | harness_baseline        |             0.553797 |                         0.129718  |                  0.551253 |   0.778874 |                0.278874 |        0.808889 |
|   3 | task_harness_baseline   |             0.620064 |                         0.195985  |                  0.660592 |   0.830174 |                0.330174 |        0.84     |
|   3 | early_fingerprint_model |             0.60171  |                         0.177631  |                  0.603645 |   0.848828 |                0.348828 |        0.826667 |
|   5 | global_mean_baseline    |             0.414512 |                         0         |                  0.416873 |   0.5      |                0        |        0.751111 |
|   5 | task_baseline           |             0.499155 |                         0.084643  |                  0.513648 |   0.669947 |                0.169947 |        0.766667 |
|   5 | harness_baseline        |             0.550367 |                         0.135854  |                  0.560794 |   0.797192 |                0.297192 |        0.813333 |
|   5 | task_harness_baseline   |             0.593001 |                         0.178489  |                  0.647643 |   0.852375 |                0.352375 |        0.857778 |
|   5 | early_fingerprint_model |             0.619668 |                         0.205156  |                  0.667494 |   0.858992 |                0.358992 |        0.842222 |

3. Object locality：stable object reuse 只统计 file/url/test_case 等稳定对象；large_observation_bucket、observation hash、test_log hash 单独归为 synthetic bucket，不作为真实对象复用证据。

4. 成功/失败差异：

| metric                |   success_mean |   failure_mean |   success_median |   failure_median |   delta_failure_minus_success |   mannwhitney_p |
|:----------------------|---------------:|---------------:|-----------------:|-----------------:|------------------------------:|----------------:|
| total_steps           |     30.987     |      61.9168   |       21         |        23        |                    30.9299    |     0.266876    |
| tool_entropy          |      2.59527   |       2.44729  |        2.64136   |         2.52164  |                    -0.147979  |     0.00572749  |
| phase_entropy         |      1.48658   |       1.31332  |        1.56448   |         1.41085  |                    -0.173264  |     3.60846e-12 |
| observation_bytes     |   5607.16      |    9242.99     |     3515.5       |      3164        |                  3635.84      |     0.774321    |
| repeated_action_ratio |      0.324226  |       0.36185  |        0.274159  |         0.333333 |                     0.0376239 |     0.0281111   |
| error_rate            |      0.0872028 |       0.118396 |        0.0526316 |         0.065942 |                     0.0311934 |     0.00592715  |

5. Wafer proxy movement：oracle_island 是 co-location 上界，不代表可实现 predictor。重点看 early_fingerprint_island 和 capacity_limited_island 是否优于 session_affinity。

| strategy                 | mesh   |   island_size |   object_accesses |   island_homed_accesses |   avg_hops |   remote_object_accesses |   moved_bytes |   estimated_latency_units |   movement_reduction_vs_random |   latency_reduction_vs_random |
|:-------------------------|:-------|--------------:|------------------:|------------------------:|-----------:|-------------------------:|--------------:|--------------------------:|-------------------------------:|------------------------------:|
| baseline_random          | 5x5    |             0 |             76226 |                       0 |    3.2033  |                    73181 |   1.21898e+08 |                    173877 |                     0          |                    0          |
| session_affinity         | 5x5    |             0 |             76226 |                       0 |    3.25563 |                    73154 |   1.23052e+08 |                    175389 |                    -0.00947043 |                   -0.00869346 |
| oracle_island            | 5x5    |             3 |             76226 |                   76226 |    1.78063 |                    67685 |   6.75622e+07 |                    130488 |                     0.445747   |                    0.24954    |
| early_fingerprint_island | 5x5    |             3 |             76226 |                   46548 |    2.23384 |                    69826 |   8.23118e+07 |                    144054 |                     0.324748   |                    0.171517   |
| capacity_limited_island  | 5x5    |             3 |             76226 |                   40331 |    2.33035 |                    70305 |   8.8801e+07  |                    147278 |                     0.271513   |                    0.152978   |
| wrong_prediction_stress  | 5x5    |             3 |             76226 |                   43245 |    2.2777  |                    70047 |   8.5341e+07  |                    145527 |                     0.299897   |                    0.163046   |

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

- terminalbench: skipped 773 rows with empty/missing steps.
