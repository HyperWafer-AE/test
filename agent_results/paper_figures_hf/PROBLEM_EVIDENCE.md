# Agent-on-Wafer Problem Evidence

This directory contains ISCA-style motivation figures for the Agent-on-Wafer paper line. The figures are generated from the current local Hugging Face public-trace evaluation, using paper-usable traces selected before policy replay.

## Evidence Chain

1. Real public traces contain persistent agent state pressure.
   - The selector loaded 181 candidate traces.
   - It found 2 paper-usable opportunity traces and 0 matched controls.
   - The 2 paper-usable traces have estimated live KV pressure of 9.82x and 18.56x the replay memory budget.

2. LRU-style state retention hits a memory cliff.
   - Under the tight memory sweep, `lru-system` produces up to 86.17 GB of KV miss traffic.
   - As state memory increases, the LRU miss traffic falls, which shows the bottleneck is state residency rather than an artifact of replay parsing.

3. Delayed-reuse state is not well captured by plain recency.
   - At the tightest budgets, LRU preserves 0 delayed-reuse regret tokens, while ASG-style retention preserves 1008 tokens.
   - The current online ASG prefill-token reduction is still small, with a maximum of about 0.57% versus `lru-system`. This is problem-existence evidence, not yet a strong algorithm-superiority claim.

4. Wafer topology matters once state placement is modeled.
   - `asg-placement-v2-online` exposes up to 24.73 GB of remote-read traffic in the current sweep.
   - `asg-prefetch-v2-online` introduces up to 3.70 GB of prefetch migration traffic.
   - This motivates topology-aware persistent-state placement and decode-centric scheduling instead of treating KV state as a flat cache.

## Generated Figures

- `fig_problem_motivation.*`: core 2x2 motivation figure covering state pressure, LRU miss traffic, delayed-reuse retention, and topology traffic.
- `fig_workload_characterization.*`: real-trace selection funnel and intrinsic opportunity metrics.
- `fig_concurrency_memory_sweep.*`: memory-capacity cliff and serving replay pressure.
- `fig_online_vs_oracle_gap.*`: delayed-reuse token retention and the current online/oracle prefill gap.
- `fig_opportunity_vs_control.*`: opportunity-vs-smoke intrinsic metrics plus the current lack of matched controls.
- `fig_mapping_ablation.*`: remote-read bytes and communication cycles from state placement.
- `fig_prefetch_ablation.*`: hidden/exposed prefetch cycles and migration risk.

Each figure has `.pdf`, `.svg`, `.png`, and a source `.csv`. `figure_report.json` records the exact evidence numbers and limitations.

## Reproduction

```bash
.venv\Scripts\python.exe -m src.agent.plot_paper_eval --input-dir agent_results\paper_eval_hf_tight_memory --output-dir agent_results\paper_figures_hf --trace-selection-dir agent_results\trace_selection_hf
```

## Current Limitation

The current public-trace sample is enough to show that the problem exists, but it is not enough for broad workload claims or a "far beyond LRU" result. The next paper step is to obtain more paper-usable opportunity/control traces or build a clearly labeled real-trace-derived serving stress harness.
