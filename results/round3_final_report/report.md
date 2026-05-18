# WaferAgent Round 3 Report

## Environment

- GPU: `NVIDIA H100 PCIe, 81559 MiB, 570.211.01
NVIDIA H100 PCIe, 81559 MiB, 570.211.01`
- Python: `3.11.14 (main, Oct 28 2025, 12:11:26) [Clang 20.1.4 ]`
- PyTorch: `2.5.1+cu124`
- CUDA available: `True`
- command: `scripts/collect_h100_traces.py --model Qwen2.5-7B-Instruct --engine hf --workloads debate,moa,planner_worker_tool,swe_like,rag_like --num-jobs 5 --gpus 1 --out results/round3_characterization_h100_hf --seed 11 --clean-required`
- git commit: `70d011192a49648f9ab4f349a74b5cdb7881984c`

## Readiness Check

- PASS: clean_git_tree
- PASS: no_silent_fallback
- PASS: neutral_default
- PASS: legacy_not_used
- PASS: h100_forward_calibration_full_or_oom_recorded
- PASS: calibration_coefficients_used_by_simulator
- PASS: no_impossible_timing_rows
- PASS: real_hf_traces_available
- PASS: real_vllm_full_or_explicitly_missing
- PASS: prefix_ratio_affects_prefill_compute_and_jct
- PASS: sram_evictions_observed_under_pressure
- PASS: sram_policy_ablation_nonzero
- PASS: mesh_bandwidth_sensitivity_nonflat
- PASS: placement_ablation_nonzero
- PASS: critical_path_ablation_nonzero_or_demoted
- PASS: dynamic_pd_nonzero_or_demoted
- PASS: all_final_tables_have_ci

## Main Neutral Results

| baseline | job_completion_time_ms | p50_latency_ms | p90_latency_ms | p99_latency_ms | goodput_jobs_per_s | kv_bytes_total | kv_saving_ratio | sram_evictions | sram_reload_bytes | sram_hit_rate | shared_prefill_compute_ms_saved | shared_prefill_tokens_saved | prefix_compute_hit_rate | mesh_total_traffic_bytes | mesh_wait_ms | mesh_hotspot_ratio | prefill_tile_utilization | decode_tile_utilization | energy_per_job_j |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| continuum_like | 970.2 | 14.13 | 234.7 | 331.9 | 2360 | 1.256e+10 | 0 | 5.556 | 0 | 0 | 0 | 0 | 0 | 4.51e+10 | 0 | 4.191 | 0.0005694 | 0.001486 | 0.0484 |
| kvflow_like | 1135 | 11.67 | 233.8 | 330.8 | 2373 | 6.599e+09 | 0.4472 | 0 | 0 | 0.7348 | 308.4 | 4.617e+04 | 0.9085 | 9.387e+10 | 0 | 3.449 | 0.0002361 | 0.001055 | 0.05815 |
| oracle | 736.1 | 7.088 | 229.2 | 326.6 | 5761 | 6.599e+09 | 0.4472 | 0 | 0 | 0.9004 | 308.4 | 4.617e+04 | 0.9085 | 6.235e+08 | 0.1007 | 1.709 | 0.002231 | 0.009114 | 0.03951 |
| wafer_naive | 1287 | 14.48 | 235.9 | 333.1 | 2226 | 1.256e+10 | 0 | 6.889 | 7.419e+08 | 0 | 0 | 0 | 0 | 1.437e+11 | 0 | 1.854 | 0.000386 | 0.0009588 | 0.06812 |
| waferagent_full | 773 | 11.67 | 233.8 | 330.8 | 3585 | 6.599e+09 | 0.4472 | 0 | 0 | 0.9004 | 308.4 | 4.617e+04 | 0.9085 | 6.235e+08 | 0.1007 | 1.709 | 0.001151 | 0.004665 | 0.03951 |


## H100 Trace And Calibration

- HF trace model: `/data2/model_zoo/Qwen2.5-7B-Instruct`
- HF trace engine: `hf`
- HF calibration model: `/data2/model_zoo/Qwen2.5-7B-Instruct`
- HF calibration engine: `hf`
- vLLM smoke model: `/data2/model_zoo/Qwen2.5-7B-Instruct`
- vLLM smoke engine: `vllm`
- vLLM smoke real trace: `True`

## H100-Calibrated Wafer Simulation

| baseline | job_completion_time_ms | p50_latency_ms | p90_latency_ms | p99_latency_ms | goodput_jobs_per_s | kv_bytes_total | kv_saving_ratio | sram_evictions | sram_reload_bytes | sram_hit_rate | shared_prefill_compute_ms_saved | shared_prefill_tokens_saved | prefix_compute_hit_rate | mesh_total_traffic_bytes | mesh_wait_ms | mesh_hotspot_ratio | prefill_tile_utilization | decode_tile_utilization | energy_per_job_j |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| continuum_like | 1.149e+05 | 7848 | 2.357e+04 | 2.555e+04 | 7.236 | 1.256e+10 | 0 | 5.556 | 0 | 0 | 0 | 0 | 0 | 4.51e+10 | 0 | 4.191 | 0.0004599 | 0.008339 | 0.0484 |
| kvflow_like | 1.068e+05 | 7291 | 2.357e+04 | 2.555e+04 | 7.68 | 6.599e+09 | 0.4472 | 0 | 0 | 0.7348 | 1.212e+05 | 4.617e+04 | 0.9085 | 9.387e+10 | 0 | 3.449 | 0.0001359 | 0.008913 | 0.05815 |
| oracle | 6.49e+04 | 4553 | 1.431e+04 | 1.551e+04 | 12.57 | 6.599e+09 | 0.4472 | 0 | 0 | 0.9004 | 1.212e+05 | 4.617e+04 | 0.9085 | 6.235e+08 | 0.1007 | 1.709 | 0.0003224 | 0.01429 | 0.03951 |
| wafer_naive | 1.152e+05 | 7849 | 2.357e+04 | 2.555e+04 | 7.228 | 1.256e+10 | 0 | 6.889 | 7.419e+08 | 0 | 0 | 0 | 0 | 1.437e+11 | 0 | 1.854 | 0.0004584 | 0.0083 | 0.06812 |
| waferagent_full | 1.065e+05 | 7291 | 2.357e+04 | 2.555e+04 | 7.687 | 6.599e+09 | 0.4472 | 0 | 0 | 0.9004 | 1.212e+05 | 4.617e+04 | 0.9085 | 6.235e+08 | 0.1007 | 1.709 | 0.0001615 | 0.007156 | 0.03951 |


## Failures / Missing Baselines

- vLLM install: `vLLM 0.6.4.post1 install retry exit_code=0 finished: 2026-05-18T20:20:07+08:00`.
- vLLM full baseline: if `real_vllm_full_or_explicitly_missing` is FAIL, do not use vLLM as a paper-grade baseline.
- Dynamic P/D partition is demoted unless targeted ablation shows >=5% JCT benefit.
- Wafer: all wafer results are trace-driven wafer-scale simulator results, not real wafer hardware measurements.

## Interpretation

Main wafer numbers are trace-driven simulation results. Real H100 traces and H100 calibration are recorded separately when the HF/vLLM engines complete. Synthetic traces are used only for controlled mechanism stress tests unless explicitly marked otherwise.

## Output Layout

- JSON readiness: `/home/duzc/data/agent_wafer/results/round3_final_report/report.json`
- Tables: `/home/duzc/data/agent_wafer/results/round3_final_report/tables`
- Figures: `/home/duzc/data/agent_wafer/results/round3_final_report/figures`
