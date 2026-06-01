# Agent-on-Wafer MVP

This package adds an agent-level frontend over BusyBarn. It converts synthetic multi-agent traces into BusyBarn-compatible fixed compute, communication, and external wait events, then uses the existing analytical event driver for wafer timing.

## Policies

- `nocache`: no persistent KV residency; every LLM step prefills full historical input plus appended tokens.
- `lru`: keeps states within the memory budget by most recent access.
- `kvflow`: approximates workflow-aware retention with next-use distance and reuse score, without topology-aware placement or prefetch.
- `asg-retention`: ASG phase-aware retention only, using agent home placement and no KV migration.
- `asg-placement`: ASG retention plus topology-aware execution/state placement and blocking demand KV migration.
- `asg-prefetch`: ASG retention and placement plus tool-wait KV prefetch.
- `asg-retention-v2`: Future-Demand-aware knapsack retention that optimizes expected saved prefill cycles.
- `asg-placement-v2`: v2 retention plus explicit local/remote-read/migrate/replicate action selection.
- `asg-prefetch-v2`: v2 placement plus windowed prefetch knapsack during tool waits.
- `asg-oracle-retention`, `asg-oracle-placement`, `asg-oracle-prefetch`: oracle future-demand upper-bound policies for diagnosis.
- `asg`: alias for `asg-prefetch-v2`.

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
python -m src.agent.experiment --run-all-policies --policy-suite v2 --cfg src/platform/cfgs/wamis_hd_distributed.cfg --topology wamis --num-agents 8 --turns 16 --memory-budget-gb 0.5 --per-node-memory-mb 64 --stress-placement --agent-placement round_robin --tool-latency-scale 1000000 --enable-observation-compression --seed 2 --scheduler-seed 0 --oracle-future --output-dir agent_results/stress/
```

```bash
python -m src.agent.comm_benchmark --cfg src/platform/cfgs/wamis_hd_distributed.cfg --topology wamis --output agent_results/comm_benchmark.json
```

```bash
python -m src.agent.trace_profile --trace-dir traces/real --trace-format auto --output agent_results/real_trace_profile.json
```

```bash
python -m src.agent.real_trace_experiment --trace-dir traces/real --trace-format auto --prediction-mode heuristic --policy-suite v2 --memory-budget-gb 0.5 --concurrency 8 --output-dir agent_results/real_online_vs_oracle/
```

## Useful CLI Flags

- `--agent-placement {round_robin,compact}` controls deterministic agent home assignment.
- `--per-node-memory-mb` sets a lightweight per-node KV placement budget.
- `--stress-placement` enables a workload preset with stronger cross-agent sharing, handoff, large observations, and tool use.
- `--cross-agent-share-probability`, `--agent-handoff-probability`, `--large-observation-probability`, and `--tool-output-shared-probability` tune synthetic sharing pressure.
- `--max-prefetch-states` caps prefetch migrations emitted during each tool wait.
- `--disable-topology-placement` disables ASG topology-aware placement.
- `--disable-prefetch` disables tool-wait prefetch.
- `--policy-suite {basic,v2,oracle,all}` chooses the policies used by `--run-all-policies`; the default is `v2`.
- `--future-horizon` and `--oracle-future` enable synthetic-trace future demand for ASG v2/oracle policies.
- `--knapsack-granularity-mb`, `--max-future-access-cap`, `--storage-penalty`, and `--knapsack-max-candidates` control v2 cost-weighted retention.
- `--scheduler-seed` seeds BusyBarn's randomized communication scheduling for repeatable agent experiments.
- `--tool-latency-scale` converts synthetic tool latency units into cycles. The default is `1000000`, so a trace latency of `1000` becomes `1e9` external wait cycles.
- `--comm-cost-model {heuristic,backend}` selects either the old distance/bandwidth estimate or a backend-calibrated link-time estimate. `backend` is the default.
- `--effective-bandwidth-bytes-per-cycle` controls the fallback heuristic migration/remote-read cost model.
- `--prefetch-reuse-threshold`, `--prefetch-next-use-threshold`, `--max-prefetch-bytes`, and `--prefetch-wait-fraction` gate prefetch candidates.
- `--enable-observation-compression` compresses large tool observations with `--large-observation-token-threshold` and `--observation-compression-ratio`.

## Real Trace Evaluation

Round5 adds a trace-driven path that does not change the recorded agent trajectory. `trace_loader.py` accepts normalized JSON/JSONL and best-effort public trajectory adapters for SWE-Gym, CodeTraceBench/CodeTracer, AgentLens/OpenHands, and generic ReAct JSONL. Source-specific adapters preserve raw payloads in metadata and fall back to robust message/tool/observation detection when public schemas are incomplete or drift.

Normalized trace events are:

- `state`: persistent prompt/state segment with `state_id`, `state_type`, `owner`, `tokens`, optional `semantic_key`, and optional `exact_token_hash`.
- `llm`: recorded LLM step with `input_state_ids`, `append_tokens`, `output_tokens`, and `new_state_id`.
- `tool`: recorded tool/action wait with `tool`, `latency`, `output_tokens`, `status`, and `new_state_id`.

`prompt_segmenter.py` maps system/developer prompts, task statements, roles, assistant deltas, file reads, edits, test failures, raw errors, web results, summaries, and subagent outputs into state types. Exact KV reuse is keyed by `exact_token_hash`; semantic keys are retained for analysis but do not imply KV equality.

`real_trace_experiment.py` runs `nocache`, `lru-basic`, `lru-system`, `kvflow-like`, online ASG v2, and oracle ASG v2. `lru-basic` disables static replicas and observation compression; `lru-system` enables both. The summary includes `oracle_gap`, retained state type distribution, LRU regret preservation, and delayed-reuse subset metadata.

## Metrics

- `effective_prefill_tokens`: tokens paid as prefill after cache hits.
- `llm_output_tokens`: LLM decode output tokens.
- `tool_output_tokens`: tokens generated by synthetic tool observations.
- `kv_migration_bytes`: total KV movement bytes.
- `demand_migration_bytes`: blocking KV movement before an LLM step.
- `prefetch_migration_bytes` / `prefetch_kv_bytes`: KV movement issued during tool-wait prefetch.
- `num_prefetch_events`: number of prefetch migration events.
- `remote_read_bytes`: bytes read remotely when migration is not cost-effective.
- `migration_skipped_by_cost`: count of remote reads selected by the migration cost model.
- `static_replica_bytes` / `num_static_replicas`: zero-cost MVP replicas for pinned static shared prefixes at agent homes.
- `unused_prefetch_events`: prefetches not consumed by the next LLM step for that agent.
- `migration_cost_estimate_cycles` / `remote_read_cost_estimate_cycles`: deterministic estimates used by policy decisions.
- `local_state_hits`: resident states already at the chosen execution node.
- `remote_state_hits`: resident states that required remote movement/access before compute.
- `state_misses`: non-resident input states that force prefill.
- `prefetch_hidden_cycles` / `prefetch_exposed_cycles`: postprocessed overlap of prefetch migration with external tool wait.
- `external_wait_cycles` / `llm_side_cycles`: approximate split between tool latency and model-side system time.
- `model_compute_cycles` / `model_comm_cycles`: BusyBarn pure compute/communication timing breakdown.
- `exposed_migration_cycles` / `exposed_prefetch_cycles`: communication time attributed to blocking movement and unhidden prefetch.
- `num_action_local`, `num_action_remote_read`, `num_action_migrate`, `num_action_replicate`, `num_action_static_hit`: v2 placement action counts.
- `compressed_observation_tokens_saved` / `num_compressed_observations`: stress-run observation compression effect.

## Limitations

- Synthetic trace only.
- No real tokenizer or prompt parsing yet.
- No real KV tensor layout.
- No real model kernel execution.
- KV bytes and prefill/decode costs use a configurable analytical model.
- Tool execution is modeled as fixed external wait.
- Topology-aware placement is heuristic, not globally optimal.
- Migration versus remote read uses a calibrated estimate of backend link time, but contention is still only measured after event-driver execution.
- KV migration is modeled as state movement events, not physical DMA implementation.
- Speculation is not implemented yet.
- Skill routing is not implemented yet.
