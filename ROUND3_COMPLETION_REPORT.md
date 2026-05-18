# WaferAgent Round 3 Completion Report

## 1. 成功项

- 修正 H100 calibration timing：新增 `waferagent/h100_forward_timer.py`，使用 causal LM forward 测 prefill，并用 `past_key_values` 逐 token decode；不再用两次 `generate()` 伪造 TTFT。
- 修正 simulator duration：新增 `waferagent/calibrated_cost_model.py`，`--duration-source calibrated` 会实际使用 `h100_fit.json` 系数，并在 metrics 中记录 `calibration_loaded=1` 与 `calibration_fit_hash`。
- 修正 cross-agent KV sharing compute 语义：新增 `waferagent/prefix_tree.py`，shared prefix 命中后只计算 private prefill tokens，并输出 `shared_prefill_compute_ms_saved`、`shared_prefill_tokens_saved`、`prefix_compute_hit_rate`。
- SRAM 改为 placement-aware region SRAM：新增 distributed SRAM access，按 tile placement 映射 region，记录 same-region/cross-region/reload/spill/evict 事件。
- Mesh transfer 改为默认阻塞 resource start：stage 先完成 dependency/KV transfer，再 reserve compute resource；`--allow-mesh-compute-overlap` 作为显式选项保留。
- Simulation 默认 neutral multipliers；legacy heuristic 只能通过 `--legacy-heuristic-multipliers` 显式启用。
- 加入 clean tree gating：支持 `--clean-required/--allow-dirty`，Round 3 主要实验均在 clean tree 下启动。
- vLLM runner 使用和 HF 一致的 `prompt_for_node()`，支持 DAG layer batch generate；vLLM TTFT/TPOT 不可用时标记为 unavailable，不再写成 `ttft=total, decode=0`。
- 新增测试并通过：`pytest tests -q` 为 `23 passed`。

## 2. 成功实验

- Environment validation:
  - `results/round3_env_validation`
  - PyTorch `2.5.1+cu124`，CUDA available，2 张 `NVIDIA H100 PCIe`，vLLM `0.6.4.post1` 可 import。
- HF forward calibration 子集:
  - `results/round3_h100_calibration_full_hf`
  - 20 个 `(input_len, output_len, batch_size)` cases，每个 3 reps，共 60 条 raw timing。
  - `timing_sanity.json`: `impossible_rows=0`。
  - 注意：这不是完整 8 x 5 x 5 matrix。
- Synthetic mechanism stress:
  - `results/round3_characterization_synthetic`
  - `results/round3_main_neutral`
  - `results/round3_ablation`
  - `results/round3_sensitivity`
- H100-calibrated wafer simulation:
  - `results/round3_main_h100cal`
  - `simulation_metrics.csv` 中 `duration_source=calibrated`，`calibration_loaded=1`，并记录 calibration fit hash。
- Real HF traces:
  - `results/round3_characterization_h100_hf`
  - 5 workloads x 5 jobs，共 240 条 trace，`engine_used=hf`，`fallback_count=0`，trace 中 `real_trace=true`。
- vLLM sanity traces:
  - `results/round3_characterization_h100_vllm`
  - 5 workloads x 1 job，共 48 条 trace，`engine_used=vllm`，`fallback_count=0`。
- Final report:
  - `results/round3_final_report/report.md`
  - `results/round3_final_report/report.json`
  - `report.json` readiness: `pass=true`。

## 3. 失败项 / 未完成项

- 完整 HF calibration matrix 未完成：本轮只跑了 20-case 子集，不是 8 x 5 x 5 x 3 全矩阵。不能把该 calibration 称为 full production calibration。
- 真实 HF characterization 未达到 20 jobs：当前是 5 jobs，不是任务书要求的 20 jobs。
- vLLM 未达到完整 20-job characterization：当前是 1-job sanity trace，不能作为完整 vLLM baseline。
- Critical-path scheduling 和 dynamic P/D partition 当前没有形成稳定 headline 证据，已在 report 中按 Round 3 规则 demote，不建议写成主贡献。
- Tool TTL 在当前 aggregate ablation 中未显示独立收益，暂时只能作为机制实现/secondary option，不适合作为主结论。

## 4. 当前最可信的 5 个数据结论

1. Neutral synthetic main 中，WaferAgent full 相对 wafer naive 的 JCT 从 `1286.66 ms` 降到 `773.00 ms`，约 `1.66x` speedup。
2. Neutral synthetic main 中，WaferAgent full 的 KV saving 为 `44.72%`，同时 shared prefill compute saving 为 `308.44 ms/job`。
3. Prefix sensitivity 中，shared prefix ratio 从 `0.0` 增至 `0.9` 时，WaferAgent JCT 从 `131.67 ms` 降至 `89.42 ms`，shared prefill compute saving 从 `0` 增至 `133.73 ms`。
4. SRAM sensitivity 中，tile SRAM 从 `2 MB` 增至 `8 MB` 后，WaferAgent JCT 从 `1350.37 ms` 降至 `227.22 ms`，reload bytes 从 `3.54e10` 降至 `0`。
5. Mesh bandwidth sensitivity 中，link bandwidth 从 `10 GB/s` 到 `200 GB/s` 时，mesh wait 从 `3.62 ms` 降至 `0.38 ms`，说明 finite mesh contention 已进入 JCT。

## 5. 可以写进论文的结论

- Multi-agent graph-level shared prefix reuse 可以同时降低 KV footprint 和 repeated prefill compute。
- WaferAgent 的 communication/aggregator-aware placement 在 mesh traffic 与 JCT 上有明显影响；`no_affinity_placement` JCT 为 `1190.69 ms`，full 为 `773.00 ms`。
- Distributed SRAM capacity 对 reload/JCT 有非平坦趋势，说明 SRAM pressure 已经进入 simulator。
- Calibration coefficients 已真实进入 simulator duration path，可作为 H100-calibrated sanity simulation。

## 6. 不能写进论文的结论

- 不能声称完成 full H100 calibration matrix。
- 不能把当前 vLLM 结果当完整 baseline；当前只是 vLLM sanity trace。
- 不能声称 dynamic P/D partition 是 headline contribution。
- 不能声称 tool TTL 有稳定主结果收益。
- 不能声称 wafer 结果是真实硬件测量；全部 wafer 结果都是 trace-driven wafer-scale simulator。

## 7. Claim 到 artifact 映射

- Prefix compute reuse:
  - CSV: `results/round3_sensitivity/simulation/sensitivity_shared_prefix_ratio.csv`
  - Figure: `results/round3_sensitivity/figures/fig_sensitivity_prefix_ratio.{png,pdf}`
- Main neutral speedup:
  - CSV: `results/round3_main_neutral/simulation/simulation_summary.csv`
  - Figure: `results/round3_main_neutral/figures/fig_main_speedup.{png,pdf}`
- H100-calibrated simulation:
  - CSV: `results/round3_main_h100cal/simulation/simulation_metrics.csv`
  - Fit: `results/round3_h100_calibration_full_hf/h100_fit.json`
- SRAM sensitivity:
  - CSV: `results/round3_sensitivity/simulation/sensitivity_sram_per_tile_mb.csv`
  - Figure: `results/round3_sensitivity/figures/fig_sensitivity_sram_capacity.{png,pdf}`
- Mesh bandwidth sensitivity:
  - CSV: `results/round3_sensitivity/simulation/sensitivity_link_bandwidth_GBps.csv`
  - Figure: `results/round3_sensitivity/figures/fig_sensitivity_mesh_bandwidth.{png,pdf}`
- Placement ablation:
  - CSV: `results/round3_ablation/simulation/simulation_summary.csv`
  - Figure: `results/round3_ablation/figures/fig_ablation_speedup.{png,pdf}`
- Real HF trace sanity:
  - Trace: `results/round3_characterization_h100_hf/traces/traces.jsonl`
  - Model selection: `results/round3_characterization_h100_hf/model_selection.json`
- vLLM sanity:
  - Trace: `results/round3_characterization_h100_vllm/traces/traces.jsonl`
  - Model selection: `results/round3_characterization_h100_vllm/model_selection.json`

## 8. 主要命令

```bash
uv run pytest tests -q
uv run python scripts/validate_environment.py --out results/round3_env_validation
uv run python scripts/calibrate_h100_real.py --engine hf --model Qwen2.5-7B-Instruct --gpus 0 --out results/round3_h100_calibration_full_hf --reps 3 --max-cases 20 --clean-required
uv run python scripts/collect_h100_traces.py --engine synthetic --workloads debate,moa,planner_worker_tool,swe_like,rag_like,long_context_swe_stress,mesh_stress_moa,sram_pressure_debate,tool_pause_resume_loop --num-jobs 50 --out results/round3_characterization_synthetic --seed 11 --clean-required
uv run python scripts/run_simulation_sweep.py --traces "results/round3_characterization_synthetic/traces/*.jsonl" --wafer-config configs/wafer/wse_like.yaml --baselines wafer_naive,continuum_like,kvflow_like,waferagent_full,oracle --out results/round3_main_neutral --duration-source synthetic --clean-required
uv run python scripts/run_simulation_sweep.py --traces "results/round3_characterization_synthetic/traces/*.jsonl" --wafer-config configs/wafer/wse_like.yaml --baselines wafer_naive,continuum_like,kvflow_like,waferagent_full,oracle --out results/round3_main_h100cal --duration-source calibrated --calibration results/round3_h100_calibration_full_hf/h100_fit.json --clean-required
uv run python scripts/run_ablation.py --traces "results/round3_characterization_synthetic/traces/*.jsonl" --wafer-config configs/wafer/wse_like.yaml --out results/round3_ablation --duration-source synthetic --clean-required
uv run python scripts/run_sensitivity.py --engine synthetic --wafer-config configs/wafer/wse_like.yaml --out results/round3_sensitivity --duration-source synthetic --clean-required
uv run python scripts/collect_h100_traces.py --model Qwen2.5-7B-Instruct --engine hf --workloads debate,moa,planner_worker_tool,swe_like,rag_like --num-jobs 5 --gpus 1 --out results/round3_characterization_h100_hf --seed 11 --clean-required
uv run python scripts/collect_h100_traces.py --model Qwen2.5-7B-Instruct --engine vllm --workloads debate,moa,planner_worker_tool,swe_like,rag_like --num-jobs 1 --gpus 0 --out results/round3_characterization_h100_vllm --seed 31 --clean-required
uv run python scripts/make_report.py --results results/round3_main_neutral --out results/round3_final_report/report.md
```

## 9. 环境与 git commit

- Latest code commit after Round 3 patches: `4ef8f8a` before report generation, plus this completion report should be committed next.
- Key experiment commits recorded in metadata:
  - `round3_main_neutral`: `95e07f76ff2eacf8765c4b0907cf40b1507e85d6`
  - `round3_h100_calibration_full_hf`: `655970328fda0037819183e12ecbb9f53d24b9b0`
  - `round3_characterization_h100_hf`: `70d011192a49648f9ab4f349a74b5cdb7881984c`
- GPU: 2 x NVIDIA H100 PCIe, 81559 MiB each, driver `570.211.01`
- Python: `3.11.14`
- PyTorch: `2.5.1+cu124`
- vLLM: `0.6.4.post1`

## 10. 下一轮建议

- 跑完整 HF calibration matrix，或定义可接受的 stratified calibration subset 并在论文中解释。
- 把 HF characterization 扩到 20 jobs，并把 vLLM characterization 扩到 20 jobs。
- 为 tool TTL 构造更强 targeted workload，让 resume reload cost 成为可解释收益。
- 若要保留 dynamic P/D partition，必须加入 burst/tail workload 并证明 >=5% targeted JCT benefit；否则继续 demote。
