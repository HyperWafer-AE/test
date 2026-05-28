# Agent Trace Profiling for Agent-on-Wafer

一条命令复现实验：

```bash
python scripts/run_all.py --datasets terminalbench --sample-size 1500 --strict-real --outdir outputs/terminalbench_1500_fixed
```

本项目验证一个核心问题：真实 agent trajectory 是否存在可预测的状态流规律，从而支持未来 Agent-on-Wafer 的 locality-aware runtime 设计。Pipeline 会下载/抽样公开 agent traces，统一成 canonical trace schema，生成统计图、表格和 Markdown 报告。

## 数据集

主数据集：

- `yoonholee/terminalbench-trajectories`：Terminal-Bench 2.0 trajectories，HuggingFace `train` split，`steps` 是 JSON 序列化 step list。

可选第二数据集：

- `nebius/SWE-agent-trajectories`：SWE-agent trajectories，`trajectory` 是 list 字段，包含 `instance_id/model_name/target/exit_status` 等元数据。该数据集可能包含很大的 patch/log 字段，默认用 bounded sample。

无网/CI fallback：

- `--offline-mock` 使用内置小样本 mock traces，保证 normalize、analysis、figures、tables、report 全链路可运行。
- `--strict-real`/`--no-mock-fallback` 禁止真实实验 fallback 到 mock；真实数据加载失败时会直接退出。

## 安装

Python 3.11+：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果使用现有项目环境，也可以直接安装 pyproject：

```bash
pip install -e .
```

## 运行

只跑 TerminalBench，默认 streaming 抽样 5000 条：

```bash
python scripts/run_all.py --datasets terminalbench --sample-size 5000 --strict-real --outdir outputs/run1
```

同时跑 TerminalBench 和 SWE-agent：

```bash
python scripts/run_all.py --datasets terminalbench,swe_agent --sample-size 2000 --outdir outputs/run_tb_swe
```

无网 smoke test：

```bash
python scripts/run_all.py --datasets mock --sample-size 10 --offline-mock --outdir outputs/mock_smoke
```

常用参数：

- `--sample-size`：每个真实数据集最多抽样多少 trajectory，默认 `5000`。
- `--datasets`：逗号分隔，支持 `terminalbench`、`swe_agent`、`mock`。
- `--outdir`：输出目录。目录下会生成 `figures/`、`tables/`、`reports/`、`data/normalized/`。
- `--cache-dir`：HuggingFace cache 目录，默认 `.cache/huggingface`。
- `--no-streaming`：禁用 HuggingFace streaming。
- `--offline-mock`：跳过网络，使用内置 mock。
- `--strict-real`：真实数据集不允许 mock fallback，用于论文/报告实验。
- `--bootstrap-samples`：fingerprint 指标 bootstrap 次数，默认 `200`。

## Canonical Schema

`data/normalized/traces.csv`：

- `trace_id, dataset, task_id, model, agent_or_harness`
- `success, reward, resolved, duration_s`
- `input_tokens, output_tokens, cache_tokens, total_steps`

`data/normalized/steps.csv`：

- `trace_id, step_id, role, phase, tool_name`
- `tool_wrapper, semantic_tool, command_string`
- `message_text, message_tokens_est, tool_args_len`
- `observation_text, observation_len_chars, observation_tokens_est, error_flag`

`data/normalized/object_accesses.csv`：

- `trace_id, step_id, object_type, object_id`
- `size_chars, access_type, phase, tool_name`
- `object_source, stable_object, tool_wrapper, semantic_tool`

`tool_name` 保留兼容字段，当前等价于 `semantic_tool`；原始包装器在 `tool_wrapper` 中，例如 `bash_command`、`execute_bash`、`str_replace_editor`。

Phase 是规则分类，不依赖大模型，优先级为 `command_string/semantic_tool`、`tool_wrapper`、最后才是简短 command-like message fallback：

- `explore/read`：`ls/cat/grep/find/search/open/read/view`
- `edit/write`：`edit/write/patch/sed/create/modify`
- `execute/test`：`bash/run/pytest/test/compile/python/make`
- `retrieve/browser`：`search/browser/fetch/web/http`
- `verify/final`：`final/submit/answer/verifier`
- `unknown`

## 输出

核心图表：

- `figures/trajectory_length_cdf.png`
- `figures/tool_calls_per_trace_cdf.png`
- `figures/observation_size_cdf.png`
- `figures/wrapper_tool_transition_heatmap.png`
- `figures/semantic_tool_transition_heatmap.png`
- `figures/collapsed_semantic_tool_transition_heatmap.png`
- `figures/phase_transition_heatmap.png`
- `figures/topk_next_tool_recall_by_view.png`
- `figures/early_fingerprint_vs_baselines.png`
- `figures/stable_object_reuse_distance_cdf.png`
- `figures/synthetic_bucket_reuse_distance_cdf.png`
- `figures/success_vs_failure_bars.png`
- `figures/wafer_proxy_movement_reduction.png`
- `figures/wafer_proxy_strategy_comparison.png`

核心表格：

- `tables/dataset_summary.csv`
- `tables/top_tools.csv`
- `tables/wrapper_tool_transition_top_pairs.csv`
- `tables/semantic_tool_transition_top_pairs.csv`
- `tables/collapsed_semantic_tool_transition_top_pairs.csv`
- `tables/phase_transition_matrix.csv`
- `tables/transition_recall_by_view.csv`
- `tables/fingerprint_metrics_by_baseline.csv`
- `tables/object_reuse_summary.csv`
- `tables/success_failure_metrics.csv`
- `tables/wafer_proxy_results.csv`

报告：

- `reports/agent_trace_profile.md`
- `reports/fix_audit.md`
- `reports/warnings.log`

报告会逐条判断 H1 到 H5：

- H1：当前 tool/phase 是否能预测下一步 tool/phase。
- H2：前 K 步 fingerprint 是否能预测未来工具分布、剩余步数、未来 observation bytes、long trajectory。
- H3：observation/file/test/browser output 是否体现对象复用和长尾状态。
- H4：成功和失败 trajectories 是否在状态成本上有差异。
- H5：Agent Island proxy placement 是否减少 estimated remote object movement / D2D hops。

## 代码结构

```text
src/loaders/terminalbench.py
src/loaders/swe_agent.py
src/normalize/schema.py
src/normalize/normalizer.py
src/analysis/basic_stats.py
src/analysis/transitions.py
src/analysis/fingerprint.py
src/analysis/object_locality.py
src/analysis/success_failure.py
src/sim/wafer_proxy.py
scripts/run_all.py
reports/
figures/
tables/
data/normalized/
```

## 注意

这个 pipeline 是 hypothesis-screening 工具。公开 trace 不是底层硬件 trace，没有真实 KV/MoE expert ID；stable object locality 只由文件路径、test id、URL 等稳定对象估计，large observation bucket 单独报告为 synthetic bucket；wafer simulator 是 proxy model，不能直接作为真实硬件加速结论。`oracle_island` 只是 co-location 上界，真正可实现趋势应看 `early_fingerprint_island` 和 `capacity_limited_island`。
