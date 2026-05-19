# WaferAgent Round 4 Completion Report

## Scope

Round 4 upgrades WaferAgent from per-job trace simulation to global multi-job serving simulation, fixes prefix-extension cost semantics, records stratified H100 calibration, and produces strict paper-ready readiness gates.

All wafer results are trace-driven wafer-scale simulation results, not real wafer hardware measurements.

## Completed Artifacts

- Global multi-job serving simulator:
  - `waferagent/global_simulator.py`
  - `waferagent/arrival.py`
  - `scripts/run_global_serving_sweep.py`
- Prefix-extension calibration:
  - `waferagent/prefix_extension_timer.py`
  - `waferagent/prefix_extension_cost_model.py`
  - `scripts/calibrate_prefix_extension.py`
- Tests:
  - `tests/test_global_serving_contention.py`
  - `tests/test_cross_job_prefix_reuse.py`
  - `tests/test_global_slo_goodput.py`
  - `tests/test_prefix_extension_timing.py`
  - `tests/test_energy_uses_computed_tokens.py`
  - `tests/test_readiness_strict_paper_ready.py`

## Completed Experiments

- Stratified H100 HF calibration:
  - `results/round4_h100_calibration_stratified_hf`
  - 80 cases, 240 raw rows, 8 OOM cases.
  - This is stratified H100 calibration, not full matrix.
- Prefix-extension calibration:
  - `results/round4_prefix_extension_calibration`
- Synthetic characterization:
  - `results/round4_characterization_synthetic`
- Global serving neutral:
  - `results/round4_global_main_neutral`
- Global H100-calibrated serving:
  - `results/round4_global_main_h100cal`
- Real HF 20-job characterization:
  - `results/round4_characterization_h100_hf_20jobs`
  - 960 trace records, fallback count 0.
- Real vLLM 20-job characterization:
  - `results/round4_characterization_h100_vllm_20jobs`
  - 960 trace records, fallback count 0.
  - Timing quality is documented as engine-reported for real LLM nodes and walltime approximation for synthetic tool calls.
- Tool TTL stress:
  - `results/round4_characterization_synthetic_toolttl`
  - `results/round4_tool_ttl_stress`
- Final report:
  - `results/round4_final_report/report.md`
  - `results/round4_final_report/report.json`

## Readiness

`results/round4_final_report/report.json` reports:

- `paper_ready.pass = true`
- `sanity.pass = true`
- `overall.pass = true`

The strict readiness schema separates paper-ready checks, sanity checks, and demoted mechanisms. It does not use mixed keys such as `critical_path_ablation_nonzero_or_demoted`.

## Demoted Mechanisms

- Critical-path scheduling.
- Dynamic P/D partition.
- Tool-aware TTL.

The targeted Round4 tool TTL stress run showed no measurable difference between `waferagent_full` and `no_tool_ttl`, so tool TTL is not a main contribution.

## Verification

- `uv run pytest tests -q` passed with 37 tests.
- Final report metadata records the active git commit and artifact hashes.
