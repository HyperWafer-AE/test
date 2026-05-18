# WaferAgent Round 2 Completion Report

## 1. 本轮改了什么

- 修复环境路径约束：脚本统一使用项目内 `.cache`、`tmp`、`results`，模型只读 `/data2/model_zoo`。
- 修复 PyTorch CUDA 环境：当前 `.venv` 中 `torch==2.5.1+cu124`，两张 `NVIDIA H100 PCIe` 均可见。
- 增加 Round 2 reproducibility metadata：`metadata.json`、`metadata.yaml`、`environment.txt`、`command.txt`、`run_manifest.json`，包含 git commit、branch、remote、dirty files、GPU/driver/PyTorch 信息。
- 禁止 silent synthetic fallback：`hf`/`vllm` engine 失败时默认非零退出，只有显式 `--allow-synthetic-fallback` 才 fallback。
- 增加 neutral mechanism multipliers：`--neutral-mechanism-multipliers` 下所有 arbitrary time multiplier 为 `1.0`，旧启发式只能作为 legacy heuristic。
- 新增 stage/resource/SRAM/mesh 机制模块：
  - `waferagent/stage_ir.py`
  - `waferagent/resource_model.py`
  - `waferagent/sram_manager.py`
  - `waferagent/mesh_network.py`
  - `waferagent/tool_ttl.py`
  - `waferagent/statistics.py`
  - `waferagent/prompts.py`
  - `waferagent/real_benchmark.py`
  - `waferagent/timing.py`
- 新增 targeted workloads：`long_context_swe_stress`、`mesh_stress_moa`、`sram_pressure_debate`、`tool_pause_resume_loop`。
- 增强 simulator 输出：`simulation_metrics.csv`、`simulation_summary.csv`、`summary_with_ci.csv`、`stage_schedule.csv`、`sram_events.csv`、`mesh_link_events.csv`、`prefix_blocks.csv`。
- 增强 final report readiness check：检查真实 H100 trace、no silent fallback、neutral multipliers、SRAM sensitivity、bandwidth sensitivity、critical-path ablation、placement ablation、git commit。
- 新增 Round 2 tests：no silent fallback、neutral multipliers、SRAM eviction、mesh contention、tool TTL semantics、stage IR、prefix tree、report readiness。

## 2. 哪些实验成功

- 环境验证成功：
  - 路径：`results/env_validation/torch_cuda_check.txt`
  - 输出：`torch 2.5.1+cu124`，`cuda_available=True`，`device_count=2`。
- 真实 HF H100 calibration 成功：
  - 路径：`results/h100_calibration_real_hf`
  - 模型：`/data2/model_zoo/Qwen2.5-7B-Instruct`
  - 完成：20 个 real HF calibration cases，0 failures。
  - 注意：这是 `--max-cases 20` 子集，不是完整 8 x 5 x 5 calibration matrix。
- 真实 HF multi-agent trace 成功：
  - 路径：`results/characterization_h100_hf_v2`
  - 工作负载：`debate,moa,planner_worker_tool,swe_like,rag_like`
  - 规模：5 workloads x 20 jobs，共 960 trace records。
  - `engine_used=hf`，`fallback_count=0`。
  - 860 条 LLM/aggregate/verify/summarize 是真实 HF trace；100 条 tool call 是 synthetic tool latency，且明确标注为 tool call。
- synthetic neutral characterization / main / ablation / sensitivity 成功：
  - `results/characterization_synthetic_v2`
  - `results/main_wafer_sim_neutral_v2`
  - `results/ablation_neutral_v2`
  - `results/sensitivity_neutral_v2`
- H100-calibrated wafer simulation 成功：
  - 路径：`results/main_wafer_sim_h100cal_v2`
  - 使用 `results/h100_calibration_real_hf/h100_fit.json`
  - neutral multipliers 开启。
- Final readiness report 成功：
  - `results/final_report_v2/report.md`
  - `results/final_report_v2/report.json`
  - `report.json` 中 readiness 全 PASS。
- vLLM 安装和 smoke 成功：
  - 安装版本：`vllm==0.6.4.post1`
  - 兼容版本调整：`transformers==4.46.3`、`numpy==1.26.4`
  - 安装日志：`results/env_validation/vllm_install_064post1_aliyun_retry.log`
  - smoke trace：`results/characterization_h100_vllm_smoke`
  - 规模：`debate` 1 job，共 9 条 vLLM real trace，`fallback_count=0`
- 测试成功：
  - 命令：`pytest tests -q`
  - 结果：`17 passed`。

## 3. 哪些实验失败及原因

- 完整 vLLM baseline 未完成：
  - `vllm==0.6.4.post1` 已安装并通过 1-job real smoke。
  - 还没有跑完整 `characterization_h100_vllm_v2` 或 vLLM calibration matrix。
  - 因此只能声明 vLLM 环境与 smoke trace 可用，不能把 vLLM 全量 baseline 写进论文主结果。
- 完整 H100 calibration matrix 未完成：
  - 当前完成 20-case subset，用于 sanity 和 H100-calibrated simulation。
  - 还没有完成 `[128..32768] x [1,16,32,128,256] x [1,2,4,8,16]` 全矩阵。
- dynamic P/D partition 的 targeted ablation 目前未显示收益：
  - `no_dynamic_pd_partition` 与 `waferagent_full` JCT 相同。
  - 该机制当前不能作为论文贡献证据。

## 4. 当前最可信的 5 个数据结论

1. 在 synthetic neutral main 上，WaferAgent full 相对 wafer naive 的 JCT speedup 为约 `1763.17 / 793.24 = 2.22x`，且没有使用 hardcoded time multiplier。
2. 在 synthetic neutral main 上，cross-agent KV sharing 带来约 `49.5%` KV memory saving；`no_kv_sharing` ablation 将 saving 降为 `0`，JCT 从 `793.24 ms` 增至 `877.53 ms`。
3. 在 `mesh_stress_moa` bandwidth sensitivity 上，link bandwidth 从 `10` 到 `200 GB/s` 时，wafer naive JCT 从 `17612.75 ms` 降到 `1018.81 ms`，说明 mesh bandwidth sensitivity 非平坦。
4. 在 SRAM sensitivity 上，WaferAgent full 在 `2/4/8 MB per tile` 下 JCT 分别约为 `1405.81 / 1235.53 / 186.95 ms`，`sram_spill_bytes` 和 hit rate 明显变化，说明 SRAM capacity 真实影响结果。
5. 真实 H100 trace + H100-calibrated simulation 已可作为 sanity check：`results/main_wafer_sim_h100cal_v2` 中 WaferAgent full 保持 KV saving 约 `47.2%`，mesh traffic 明显低于 wafer naive，但 JCT 相对 KVFlow-like 只有很小差距。

## 5. 当前还不能写进论文的结论

- 不能声明完整 vLLM baseline 已完成；当前只有 vLLM 安装验证和 1-job real smoke trace。
- 不能声明真实 wafer 硬件结果；所有 wafer 结果仍是 trace-driven wafer-scale simulator。
- 不能把 dynamic P/D partition 写成有强收益的机制；当前 targeted ablation 是 flat。
- 不能把 H100-calibrated simulation 的 JCT 结论写得过强；它当前更支持 KV/mesh 指标 sanity，而不是强 JCT speedup。
- 不能声明完整 calibration matrix 已完成；当前真实 calibration 是 20-case subset。

## 6. 下一轮建议

- 完整跑完 H100 calibration matrix，并加入更多 seeds。
- 优先修 dynamic P/D partition，使队列压力和 tile pool conflict 能在 targeted workload 中产生可测差异。
- 继续加强 H100-calibrated cost model，让真实 trace 的 prefill/decode 拟合更稳定。
- 如果要保留 vLLM baseline，建议基于当前已验证组合 `torch==2.5.1+cu124`、`transformers==4.46.3`、`vllm==0.6.4.post1` 跑完整 vLLM characterization/calibration。
- 对 final paper 主结果，建议暂时使用 synthetic neutral stress tests 做机制证据，用真实 H100 trace + H100-calibrated simulation 做 sanity check，而不是把后者过度包装成主 JCT 结论。
