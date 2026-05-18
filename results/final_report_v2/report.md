# WaferAgent Round 2 Report

## Environment

- GPU: `NVIDIA H100 PCIe, 81559 MiB, 570.211.01
NVIDIA H100 PCIe, 81559 MiB, 570.211.01`
- Python: `3.11.14 (main, Oct 28 2025, 12:11:26) [Clang 20.1.4 ]`
- PyTorch: `2.5.1+cu124`
- CUDA available: `True`
- command: `scripts/collect_h100_traces.py --model auto --engine hf --workloads debate,moa,planner_worker_tool,swe_like,rag_like --num-jobs 20 --gpus 0,1 --out results/characterization_h100_hf_v2 --seed 11`
- git commit: `dd740f49c74599d5a04a494d56e996ab70dd164d`

## Readiness Check

- PASS: real H100 traces available
- PASS: no silent fallback
- PASS: neutral multipliers used
- PASS: SRAM sensitivity non-flat
- PASS: bandwidth sensitivity non-flat on mesh_stress_moa
- PASS: critical-path ablation non-zero on targeted workload
- PASS: placement ablation affects mesh metrics and targeted JCT
- PASS: git commit available

## Main Neutral Results

| baseline | job_completion_time_ms | p50_latency_ms | p90_latency_ms | p99_latency_ms | goodput_jobs_per_s | kv_bytes_total | kv_saving_ratio | sram_evictions | sram_reload_bytes | sram_hit_rate | mesh_total_traffic_bytes | mesh_wait_ms | mesh_hotspot_ratio | prefill_tile_utilization | decode_tile_utilization | energy_per_job_j |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| continuum_like | 1203 | 17.03 | 318.7 | 442.9 | 5016 | 1.256e+10 | 0 | 0 | 0 | 0 | 9.075e+10 | 0 | 5.535 | 0.0003647 | 0.0008424 | 0.05753 |
| kvflow_like | 847.8 | 14.46 | 249.1 | 348.7 | 1.064e+04 | 6.338e+09 | 0.4949 | 0 | 8.939e+08 | 0.8113 | 3.442e+09 | 0 | 2.838 | 0.001821 | 0.004534 | 0.04007 |
| oracle | 740 | 8.473 | 229.6 | 327 | 1.934e+04 | 6.338e+09 | 0.4949 | 0 | 2.619e+08 | 0.9085 | 4.313e+05 | 0.1007 | 1.619 | 0.004549 | 0.007367 | 0.03938 |
| wafer_naive | 1763 | 24.67 | 445.7 | 588.4 | 3551 | 1.256e+10 | 0 | 0 | 0 | 0 | 2.531e+11 | 0 | 1.749 | 0.000258 | 0.0004334 | 0.09001 |
| waferagent_full | 793.2 | 14.06 | 235.9 | 334 | 1.157e+04 | 6.338e+09 | 0.4949 | 0 | 8.939e+08 | 0.8113 | 2.39e+08 | 0.1007 | 1.73 | 0.002257 | 0.003652 | 0.03943 |


## H100 Trace And Calibration

- HF trace model: `/data2/model_zoo/Qwen2.5-7B-Instruct`
- HF trace engine: `hf`
- HF calibration model: `/data2/model_zoo/Qwen2.5-7B-Instruct`
- HF calibration engine: `hf`
- vLLM smoke model: `/data2/model_zoo/Qwen2.5-7B-Instruct`
- vLLM smoke engine: `vllm`
- vLLM smoke real trace: `True`

## H100-Calibrated Wafer Simulation

| baseline | job_completion_time_ms | p50_latency_ms | p90_latency_ms | p99_latency_ms | goodput_jobs_per_s | kv_bytes_total | kv_saving_ratio | sram_evictions | sram_reload_bytes | sram_hit_rate | mesh_total_traffic_bytes | mesh_wait_ms | mesh_hotspot_ratio | prefill_tile_utilization | decode_tile_utilization | energy_per_job_j |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| continuum_like | 3597 | 548.8 | 860.2 | 921.6 | 30.21 | 2.469e+09 | 0 | 0 | 0 | 0 | 6.516e+09 | 0 | 4.09 | 3.097e-05 | 0.002486 | 0.009143 |
| kvflow_like | 3527 | 539 | 860.2 | 921.6 | 30.93 | 1.309e+09 | 0.4716 | 0 | 1.493e+08 | 0.879 | 2.857e+07 | 0 | 3.044 | 3.054e-05 | 0.002547 | 0.007846 |
| oracle | 2297 | 327.2 | 600.8 | 707.2 | 48.83 | 1.309e+09 | 0.4716 | 0 | 1.493e+08 | 0.879 | 4506 | 0.5643 | 1.273 | 0.000175 | 0.002451 | 0.00784 |
| wafer_naive | 3679 | 554.8 | 860.2 | 921.6 | 29.62 | 2.469e+09 | 0 | 0 | 0 | 0 | 1.174e+10 | 0 | 1.721 | 3.48e-05 | 0.002436 | 0.01019 |
| waferagent_full | 3524 | 539 | 860.2 | 921.6 | 30.95 | 1.309e+09 | 0.4716 | 0 | 1.493e+08 | 0.879 | 4506 | 0.9295 | 1.273 | 9.086e-05 | 0.001274 | 0.00784 |


## Failures / Missing Baselines

- vLLM install: `vLLM 0.6.4.post1 install retry exit_code=0 finished: 2026-05-18T20:20:07+08:00`.
- vLLM full baseline: only a 1-job smoke trace has been rerun after installation; the full vLLM characterization matrix should still be run before using vLLM as a paper baseline.
- Wafer: all wafer results are trace-driven wafer-scale simulator results, not real wafer hardware measurements.

## Interpretation

Main wafer numbers are trace-driven simulation results. Real H100 traces and H100 calibration are recorded separately when the HF/vLLM engines complete. Synthetic traces are used only for controlled mechanism stress tests unless explicitly marked otherwise.

## Output Layout

- JSON readiness: `/home/duzc/data/agent_wafer/results/final_report_v2/report.json`
- Tables: `/home/duzc/data/agent_wafer/results/final_report_v2/tables`
- Figures: `/home/duzc/data/agent_wafer/results/final_report_v2/figures`
