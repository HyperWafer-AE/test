# WaferAgent Round 5 Completion Report

## Scope

Round 5 moves the project from a prefix-cache/wafer simulator prototype toward graph-aware shared-KV execution. The implemented path focuses on synthetic mechanism stress, global multi-job serving, APC/KVFlow/PAT-like baselines, decode cohorts, shared-KV placement metrics, and final paper figure/report generation.

All wafer numbers in this round are trace-driven wafer-scale simulation results, not real wafer hardware measurements.

## Implemented

- `SharedKVObject` extraction with safe strict-prefix accounting.
- APC-like, KVFlow-like, PAT-like, no-cache, WaferAgent, oracle, and Round5 ablation baselines.
- Decode cohort scheduler and shared-attention traffic model. Cohorts reduce repeated shared-KV reads/transfers; attention outputs are not reused.
- Shared-KV replication/placement metrics and replication tradeoff sweep.
- Global multi-job serving simulator outputs for job metrics, stage schedule, SLO goodput, resource utilization, SRAM events, mesh events, shared-KV objects, and decode cohorts.
- Round5 figure generation for Fig. 1-20 style paper inputs.
- Round5 final report and strict readiness JSON.

## Experiments Completed

- E0 tests and smoke:
  - `uv run pytest tests -q`
  - `uv run python scripts/run_smoke_test.py --engine synthetic --out results/round5_smoke --clean-required`
- E1 workload opportunity:
  - `results/round5_workload_opportunity`
- E2 existing prefix-cache gap:
  - `results/round5_existing_cache_gap`
- E3 decode cohort sweep:
  - `results/round5_decode_cohort_sweep`
- E4 replication tradeoff:
  - `results/round5_replication_tradeoff`
- E5 global serving neutral:
  - `results/round5_global_main_neutral`
- E5 H100-calibrated global serving using Round4 calibration:
  - `results/round5_global_main_h100cal`
- E6 ablation:
  - `results/round5_ablation`
- Final report:
  - `results/round5_final_report/report.md`
  - `results/round5_final_report/report.json`
  - `results/round5_final_report/tables/*.csv`
  - `results/round5_final_report/figures/*.{png,pdf}`

## Key Observations

1. APC-like prefix cache saves repeated prefill compute but leaves decode shared-KV read bytes unchanged relative to no-cache.
2. PAT-like decode grouping reduces decode shared-KV reads, but without WaferAgent placement it keeps much higher mesh traffic and worse tail JCT.
3. WaferAgent full reduces decode shared-KV read bytes versus APC-like and sharply reduces mesh traffic versus PAT-like in the global serving runs.
4. Global serving shows strong queueing/tail effects that per-job simulation cannot expose.
5. Ablation supports decode cohort, affinity placement, and aggregator placement. Shared-KV replication, distributed SRAM policy, and future-reuse policy did not show enough ablation delta in this synthetic pressure setting.

## Paper-Ready Claims

Supported by current Round5 outputs:

- Existing prefix cache is insufficient for wafer-scale multi-agent serving because it does not reduce decode-side shared-KV reads or solve physical placement.
- Decode cohorts reduce shared-KV read traffic without reusing attention outputs.
- Wafer-aware placement materially affects mesh traffic and tail latency.
- Global multi-job serving evaluation is now available and should replace per-job JCT as the main serving result.

## Not Paper-Ready Yet

- Planning overhead is not instrumented; the readiness gate marks it as missing.
- Round5 did not rerun real HF/vLLM 20-job traces. Round4 HF 20-job traces exist; Round5 vLLM paper-grade characterization is still missing.
- Round5 did not rerun prefix-extension calibration; Round4 stratified H100 calibration and prefix-extension fit are reused for H100-calibrated simulation.
- Dynamic P/D partition, tool TTL, and critical-path scheduling remain demoted.
- Shared-KV replication and distributed SRAM should not be headline claims until a targeted workload shows non-zero ablation delta.

## Main Artifact Paths

- Workload opportunity: `results/round5_workload_opportunity/simulation/shared_kv_opportunity.csv`
- Prefix-cache gap: `results/round5_existing_cache_gap/simulation/existing_cache_gap_summary.csv`
- Decode cohort sweep: `results/round5_decode_cohort_sweep/simulation/decode_cohort_sweep.csv`
- Replication tradeoff: `results/round5_replication_tradeoff/simulation/replication_tradeoff_summary.csv`
- Global neutral serving: `results/round5_global_main_neutral/simulation/global_simulation_summary.csv`
- Global H100-calibrated serving: `results/round5_global_main_h100cal/simulation/global_simulation_summary.csv`
- Ablation: `results/round5_ablation/simulation/global_simulation_summary.csv`
- Final readiness: `results/round5_final_report/report.json`
- Final report: `results/round5_final_report/report.md`

## Git

- Final code commit is recorded in `results/round5_final_report/metadata.json`.
- The result directories are ignored by git and recorded via per-run metadata/artifact hashes.
