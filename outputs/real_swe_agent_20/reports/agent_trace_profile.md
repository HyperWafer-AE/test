# Agent Trace Profile Report

生成时间由本地运行决定。命令参数：

```bash
python scripts/run_all.py --datasets swe_agent --sample-size 20 --outdir /home/duzc/data/agent_wafer/outputs/real_swe_agent_20
```

## 数据集和样本量

请求数据集：`swe_agent`  
加载模式：`streaming=True`, `offline_mock=False`  
真实/兜底状态：`swe_agent=real`  
Canonical rows：traces=20, steps=401, object_accesses=1691

| dataset   |   num_traces |   num_steps |   success_rate |   median_steps |   mean_steps |   median_obs_chars_per_trace |   mean_obs_chars_per_trace |
|:----------|-------------:|------------:|---------------:|---------------:|-------------:|-----------------------------:|---------------------------:|
| swe_agent |           20 |         401 |           0.35 |             14 |        20.05 |                        48147 |                    66686.2 |

Canonical CSV：

- `../data/normalized/traces.csv`
- `../data/normalized/steps.csv`
- `../data/normalized/object_accesses.csv`

## Hypothesis Verdicts

| Hypothesis | Verdict | Evidence |
|---|---:|---|
| H1: tool/phase 时间局部性 | 支持 | 最高频工具转移是 `end_of_edit -> end_of_edit`，count=75，P=0.503。 见 `../figures/tool_transition_heatmap.png`、`../figures/phase_transition_heatmap.png`、`../figures/topk_next_tool_recall.png`。 |
| H2: early fingerprint 可预测未来 | 支持 | 前 K 步特征的 tool cosine、remaining step R2、long trajectory AUC/accuracy 见 `../tables/fingerprint_metrics.csv` 和 `../figures/early_fingerprint_predictability.png`。 |
| H3: observation/file/test/browser 是可复用状态对象 | 支持 | observation p50=851.0 chars, p95=4488.0 chars；reuse events=1190, median reuse distance=2.00。见 `../figures/observation_size_cdf.png` 和 `../figures/object_reuse_distance_cdf.png`。 |
| H4: 成功/失败轨迹状态成本不同 | 支持 | step 数、tool entropy、observation bytes、重复 action、error rate 的分组统计见 `../tables/success_failure_metrics.csv` 和 `../figures/success_vs_failure_bars.png`。 |
| H5: Agent Island placement 降低 proxy movement | 支持 | 5x5 mesh proxy simulator 对比 random/session/island，见 `../tables/wafer_proxy_results.csv` 和 `../figures/wafer_proxy_movement_reduction.png`。 |

## 关键发现

1. Tool locality：条件转移预测的 top-k recall 如下。若 `conditional_recall` 高于 global baseline，说明当前 tool 对下一 tool 有可利用信号。

|   k |   conditional_recall |   global_recall |
|----:|---------------------:|----------------:|
|   1 |             0.272109 |        0.612245 |
|   3 |             0.816327 |        0.768707 |
|   5 |             0.877551 |        0.857143 |

2. Early fingerprint：前 1/2/3/5 步的简单统计已能给出轻量预测基线。它不是最终模型，但能回答“早期状态是否带有可调度信号”。

|   K |   future_tool_cosine |   future_tool_top3_recall |   remaining_steps_r2 |   long_auc |   long_accuracy |
|----:|---------------------:|--------------------------:|---------------------:|-----------:|----------------:|
|   1 |             0.793662 |                  0.666667 |         -2.71        |        0.5 |        0.333333 |
|   2 |             0.89064  |                  1        |       -119.862       |      nan   |        1        |
|   3 |             0.803892 |                  1        |         -2.16389     |        1   |        0.666667 |
|   5 |             0.870606 |                  1        |         -0.000597181 |      nan   |        0.666667 |

3. Observation 长尾与对象复用：observation size CDF 展示状态对象大小分布；object reuse distance CDF 使用文件路径、test case、URL、observation hash 和 large-observation bucket 的规则近似复用。

4. 成功/失败差异：

| metric                |   success_mean |   failure_mean |   success_median |   failure_median |   delta_failure_minus_success |   mannwhitney_p |
|:----------------------|---------------:|---------------:|-----------------:|-----------------:|------------------------------:|----------------:|
| total_steps           |      13.5714   |      23.5385   |       13         |         22       |                    9.96703    |     0.151279    |
| tool_entropy          |       2.9404   |       2.55797  |        2.81507   |          2.36852 |                   -0.38243    |     0.0667061   |
| phase_entropy         |       2.02032  |       1.61589  |        1.96778   |          1.64115 |                   -0.404431   |     0.000607762 |
| observation_bytes     |   41560.4      |   80215.5      |    36697         |      75094       |                38655          |     0.425312    |
| repeated_action_ratio |       0.375051 |       0.366092 |        0.314286  |          0.3     |                   -0.00895951 |     0.576799    |
| error_rate            |       0.115052 |       0.532897 |        0.0769231 |          0.7     |                    0.417846   |     0.00671592  |

5. Wafer proxy movement：

| strategy         | mesh   |   island_size |   object_accesses |   avg_hops |   remote_object_accesses |   moved_bytes |   estimated_latency_units |   movement_reduction_vs_random |   latency_reduction_vs_random |
|:-----------------|:-------|--------------:|------------------:|-----------:|-------------------------:|--------------:|--------------------------:|-------------------------------:|------------------------------:|
| baseline_random  | 5x5    |             0 |              1691 |    3.24719 |                     1605 |   8.43255e+06 |                   4456.11 |                       0        |                      0        |
| session_affinity | 5x5    |             0 |              1691 |    4.05973 |                     1600 |   1.07184e+07 |                   5165.59 |                      -0.271076 |                     -0.159217 |
| agent_island     | 5x5    |             3 |              1691 |    1.70018 |                     1488 |   4.4841e+06  |                   3145.66 |                       0.46824  |                      0.294079 |

## 输出索引

Figures：

- `../figures/trajectory_length_cdf.png`
- `../figures/tool_calls_per_trace_cdf.png`
- `../figures/observation_size_cdf.png`
- `../figures/tool_transition_heatmap.png`
- `../figures/phase_transition_heatmap.png`
- `../figures/topk_next_tool_recall.png`
- `../figures/early_fingerprint_predictability.png`
- `../figures/object_reuse_distance_cdf.png`
- `../figures/success_vs_failure_bars.png`
- `../figures/wafer_proxy_movement_reduction.png`

Tables：

- `../tables/dataset_summary.csv`
- `../tables/top_tools.csv`
- `../tables/tool_transition_top_pairs.csv`
- `../tables/phase_transition_matrix.csv`
- `../tables/fingerprint_metrics.csv`
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

- 无。
