# Agent-on-Wafer MVP

This package adds an agent-level replay frontend over BusyBarn. It converts normalized agent trajectories into BusyBarn-compatible compute, communication, KV movement, and external-wait events, then uses the existing analytical event driver for wafer timing.

## Algorithm Boundaries

- ASG Builder: online graph construction from the replayed trace. It records states, LLM/tool executions, dependencies, affinities, phases, and live access counters. It is not a learning model and does not train from future events.
- State Demand Estimator: optional online demand prediction. `heuristic` uses hand-coded state/phase features; `trace_stats` fits replay-trace buckets from a training trace set; `oracle` is only an upper-bound replay mode.
- Persistent State Planner / KV Manager: decides which state KV entries remain resident under the memory budget. `asg-retention-v2-graph-only` uses graph state and recency/live-prompt terms without a demand estimator. `asg-retention-v2-online` adds the estimator. `asg-retention-v2-oracle` uses future replay access and should be read as a diagnostic ceiling.
- Wafer Mapper: maps agents and resident states onto wafer nodes, chooses local/remote/migrate/replicate actions, and optionally emits tool-wait prefetch events.

## Policy Names

- `nocache`: no persistent KV residency; every LLM step pays prefill for input state tokens.
- `lru-basic`: LRU retention without static shared-prefix replicas.
- `lru-system`: LRU retention with the same static shared-prefix replica support available to stronger system policies.
- `kvflow-like`: workflow-aware retention approximation using next-use/reuse hints, without ASG v2 knapsack planning.
- `asg-retention-v2-graph-only`: ASG graph and v2 retention planner without a demand estimator.
- `asg-retention-v2-online`: graph-only plus the selected State Demand Estimator.
- `asg-retention-v2-oracle`: oracle future-demand upper bound for diagnosis.
- `asg-placement-v2-online`: online retention plus topology-aware execution/state placement.
- `asg-prefetch-v2-online`: placement plus windowed prefetch during tool waits.

## Common Commands

```bash
python -m compileall src
```

```bash
python -m src.agent.experiment --policy asg --cfg src/platform/cfgs/wamis_hd_distributed.cfg --topology wamis --num-agents 2 --turns 4 --memory-budget-gb 1 --seed 1 --output agent_results/smoke_asg.json
```

```bash
python -m src.agent.experiment --run-all-policies --policy-suite v2 --cfg src/platform/cfgs/wamis_hd_distributed.cfg --topology wamis --num-agents 4 --turns 8 --memory-budget-gb 1 --seed 1 --output-dir agent_results/basic/
```

```bash
python -m src.agent.trace_sources --output agent_results/trace_sources_report.json
python -m src.agent.trace_audit --trace-dir traces/real --trace-format auto --output agent_results/real_trace_audit.json
python -m src.agent.trace_profile --trace-dir traces/real --trace-format auto --output agent_results/real_trace_profile.json
```

```bash
python -m src.agent.real_trace_experiment --trace-dir traces/real --trace-format auto --prediction-mode heuristic --policy-suite v2 --memory-budget-gb 0.5 --concurrency 8 --output-dir agent_results/real_online_vs_oracle/
```

```bash
python -m src.agent.real_trace_experiment --trace-dir traces/real --trace-format auto --prediction-mode trace_stats --train-trace-dir traces/real --policy-suite v2 --memory-budget-gb 0.5 --concurrency 8 --output-dir agent_results/real_trace_stats/
```

```bash
python -m src.agent.run_real_sweeps --trace-dir traces/real --trace-format auto --output-dir agent_results/real_sweeps
```

## Useful CLI Flags

- `--agent-placement {round_robin,compact}` controls deterministic agent home assignment.
- `--per-node-memory-mb` sets a lightweight per-node KV placement budget.
- `--policy-suite {basic,v2,oracle,all}` chooses synthetic experiment policies; real-trace replay currently uses the v2 suite.
- `--future-horizon` and `--oracle-future` enable synthetic-trace future demand for ASG v2/oracle policies.
- `--horizon` controls the real-trace estimator/oracle lookahead window.
- `--prediction-mode {heuristic,trace_stats}` selects the online State Demand Estimator for real-trace online policies.
- `--reject-accumulated-fallback` rejects traces whose prompt reconstruction mostly comes from accumulated fallback history.
- `--enable-common-observation-compression` explicitly enables observation compression for all real-trace policies.
- `--enable-asg-observation-compression` explicitly enables ASG-specific observation compression and labels it in outputs.
- `--disable-observation-compression` is a global off switch. Real-trace replay does not enable ASG-specific compression by default.

## Real Trace Evaluation

`trace_loader.py` accepts normalized JSON/JSONL and best-effort public trajectory adapters for SWE-Gym, CodeTraceBench/CodeTracer, AgentLens/OpenHands, and generic ReAct JSONL. Source-specific adapters preserve raw payloads in metadata and fall back to message/tool/observation detection when public schemas are incomplete or drift.

`trace_normalizer.py` labels each LLM event with prompt reconstruction metadata:

- `explicit`: the uploaded or normalized trace already contains explicit state references.
- `message_history`: input states were reconstructed from message history.
- `accumulated_fallback`: the adapter had to build prompts from accumulated thought/action/observation text.

It also reports `full_history_likely` and a monotonic growth score. Full-history-like traces can still be replayed, but they should not be presented as high-quality evidence about exact prompt composition.

`trace_audit.py` reports source quality, prompt reconstruction risk, nontrivial tool/state reuse, high-quality trace count, low-quality trace count, and exclusion reasons. `trace_profile.py` reports reuse-distance, state-type, delayed-reuse, cross-agent-reuse, LRU-regret, tool, and phase statistics, and writes both state and reuse CSVs.

`real_trace_experiment.py` writes one JSON per policy plus `summary.csv`. The summary separates:

- `baseline_class`: `basic`, `system`, `asg`, or `oracle`.
- `estimator_mode`: `none`, `heuristic`, `trace_stats`, or `oracle`.
- `asg_builder_enabled`: whether the policy uses ASG graph/planner logic.
- `oracle_future`: whether future replay access was used.
- `prompt_reconstruction_quality`: a compact label for the replayed trace set.

## Baseline Interpretation

Use `lru-system`, not `lru-basic`, as the main practical LRU baseline when judging system-level gains. `lru-basic` is retained for ablation because it disables static shared-prefix replicas.

Compare `asg-retention-v2-graph-only` against `lru-system` to isolate graph/planner value without a demand estimator. Compare `asg-retention-v2-online` against graph-only to isolate estimator value. Compare online against `asg-retention-v2-oracle` only as an oracle-gap diagnostic.

Any run with common or ASG-specific observation compression must be labeled by `observation_compression_class`; do not mix compressed and uncompressed policies when claiming cache-policy superiority.

## Metrics

- `effective_prefill_tokens`: tokens paid as prefill after cache hits.
- `cache_hit_ratio`: fraction of input states served from resident KV.
- `state_misses`: non-resident input states that force prefill.
- `LRU_regret_states_preserved` / `LRU_regret_tokens_preserved`: high-value delayed-reuse candidates still resident at the end of replay.
- `oracle_gap`: normalized distance between `lru-system` and `asg-retention-v2-oracle` in effective prefill tokens.
- `remote_read_bytes`, `demand_migration_bytes`, `prefetch_migration_bytes`: KV movement/access bytes.
- `model_compute_cycles` / `model_comm_cycles`: BusyBarn pure compute/communication timing breakdown.
- `compressed_observation_tokens_saved` / `num_compressed_observations`: explicit observation compression effect.

## Limitations

- Real-trace adapters are best-effort and depend on public dataset schemas that may drift.
- Message-history reconstruction is not exact tokenizer-level prompt composition; full-history-like traces are flagged and should be interpreted cautiously.
- `semantic_key` helps analysis but does not imply exact KV equality; exact reuse requires `exact_token_hash`.
- No real KV tensor layout or physical DMA implementation is modeled.
- KV bytes and prefill/decode costs use a configurable analytical model.
- Tool execution is modeled as fixed external wait.
- Topology-aware placement and prefetch are heuristic, not globally optimal.
- The oracle policy uses future replay access and is not an online deployable algorithm.
- Trace-stat demand estimation can overfit if training and evaluation use the same small trace set; report that condition when it happens.
