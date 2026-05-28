# Agent Trace Profile Report

生成时间由本地运行决定。命令参数：

```bash
python scripts/run_all.py --datasets terminalbench --sample-size 100 --outdir /home/duzc/data/agent_wafer/outputs/real_terminalbench_100
```

## 数据集和样本量

请求数据集：`terminalbench`  
加载模式：`streaming=True`, `offline_mock=False`  
真实/兜底状态：`terminalbench=real`  
Canonical rows：traces=100, steps=5095, object_accesses=8958

| dataset       |   num_traces |   num_steps |   success_rate |   median_steps |   mean_steps |   median_obs_chars_per_trace |   mean_obs_chars_per_trace |
|:--------------|-------------:|------------:|---------------:|---------------:|-------------:|-----------------------------:|---------------------------:|
| terminalbench |          100 |        5095 |           0.29 |           29.5 |        50.95 |                       5317.5 |                      12290 |

Canonical CSV：

- `../data/normalized/traces.csv`
- `../data/normalized/steps.csv`
- `../data/normalized/object_accesses.csv`

## Hypothesis Verdicts

| Hypothesis | Verdict | Evidence |
|---|---:|---|
| H1: tool/phase 时间局部性 | 支持 | 最高频工具转移是 `bash_command -> bash_command`，count=1632，P=0.953。 见 `../figures/tool_transition_heatmap.png`、`../figures/phase_transition_heatmap.png`、`../figures/topk_next_tool_recall.png`。 |
| H2: early fingerprint 可预测未来 | 支持 | 前 K 步特征的 tool cosine、remaining step R2、long trajectory AUC/accuracy 见 `../tables/fingerprint_metrics.csv` 和 `../figures/early_fingerprint_predictability.png`。 |
| H3: observation/file/test/browser 是可复用状态对象 | 部分支持 | observation p50=122.0 chars, p95=968.6 chars；reuse events=4815, median reuse distance=1.00。见 `../figures/observation_size_cdf.png` 和 `../figures/object_reuse_distance_cdf.png`。 |
| H4: 成功/失败轨迹状态成本不同 | 支持 | step 数、tool entropy、observation bytes、重复 action、error rate 的分组统计见 `../tables/success_failure_metrics.csv` 和 `../figures/success_vs_failure_bars.png`。 |
| H5: Agent Island placement 降低 proxy movement | 支持 | 5x5 mesh proxy simulator 对比 random/session/island，见 `../tables/wafer_proxy_results.csv` 和 `../figures/wafer_proxy_movement_reduction.png`。 |

## 关键发现

1. Tool locality：条件转移预测的 top-k recall 如下。若 `conditional_recall` 高于 global baseline，说明当前 tool 对下一 tool 有可利用信号。

|   k |   conditional_recall |   global_recall |
|----:|---------------------:|----------------:|
|   1 |             0.799483 |        0.452781 |
|   3 |             0.938551 |        0.81436  |
|   5 |             0.958603 |        0.896507 |

2. Early fingerprint：前 1/2/3/5 步的简单统计已能给出轻量预测基线。它不是最终模型，但能回答“早期状态是否带有可调度信号”。

|   K |   future_tool_cosine |   future_tool_top3_recall |   remaining_steps_r2 |   long_auc |   long_accuracy |
|----:|---------------------:|--------------------------:|---------------------:|-----------:|----------------:|
|   1 |             0.719783 |                  0.9      |            0.0976963 |   0.758523 |        0.7      |
|   2 |             0.862795 |                  0.965517 |            0.227719  |   0.784    |        0.733333 |
|   3 |             0.87969  |                  0.931034 |           -0.0732115 |   0.52     |        0.733333 |
|   5 |             0.831632 |                  0.925926 |            0.46812   |   0.83     |        0.833333 |

3. Observation 长尾与对象复用：observation size CDF 展示状态对象大小分布；object reuse distance CDF 使用文件路径、test case、URL、observation hash 和 large-observation bucket 的规则近似复用。

4. 成功/失败差异：

| metric                |   success_mean |   failure_mean |   success_median |   failure_median |   delta_failure_minus_success |   mannwhitney_p |
|:----------------------|---------------:|---------------:|-----------------:|-----------------:|------------------------------:|----------------:|
| total_steps           |      37.3793   |      56.493    |       18         |        34        |                    19.1136    |       0.246509  |
| tool_entropy          |       1.03854  |       0.882656 |        0.921928  |         0.918296 |                    -0.155888  |       0.280664  |
| phase_entropy         |       1.64959  |       1.70945  |        1.75      |         1.81228  |                     0.0598558 |       0.454301  |
| observation_bytes     |    5713.9      |   14975.9      |     3382         |      6053        |                  9262.05      |       0.0262885 |
| repeated_action_ratio |       0.721047 |       0.766262 |        0.8       |         0.857143 |                     0.0452147 |       0.164405  |
| error_rate            |       0.151435 |       0.176236 |        0.0909091 |         0.142857 |                     0.0248015 |       0.0535232 |

5. Wafer proxy movement：

| strategy         | mesh   |   island_size |   object_accesses |   avg_hops |   remote_object_accesses |   moved_bytes |   estimated_latency_units |   movement_reduction_vs_random |   latency_reduction_vs_random |
|:-----------------|:-------|--------------:|------------------:|-----------:|-------------------------:|--------------:|--------------------------:|-------------------------------:|------------------------------:|
| baseline_random  | 5x5    |             0 |              8958 |    3.22583 |                     8614 |   1.56551e+07 |                   20637.5 |                      0         |                    0          |
| session_affinity | 5x5    |             0 |              8958 |    3.19547 |                     8710 |   1.56209e+07 |                   20538.8 |                      0.0021881 |                    0.00477895 |
| agent_island     | 5x5    |             3 |              8958 |    1.7745  |                     7928 |   8.5318e+06  |                   15374.8 |                      0.455016  |                    0.255006   |

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

- terminalbench: skipped 70 rows with empty/missing steps.
