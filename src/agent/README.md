# Agent-on-Wafer MVP

This package adds an agent-level replay frontend over BusyBarn. It converts normalized agent trajectories into BusyBarn-compatible fixed compute, communication, KV movement, and external-wait events, then uses the existing analytical event driver for wafer timing.

## Backend Contract

Agent-on-Wafer uses BusyBarn as an event-level wafer backend.

- LLM prefill/decode are `fixed_compute` analytical events. They occupy a BusyBarn compute module, but their durations come from `ModelProfile`, not from BusyBarn's original operator-level LLM partition pipeline.
- KV remote-read, migration, replication, and prefetch are `communication` events. They use BusyBarn NoC routing, link scheduling, and communication contention.
- Tool/environment latency is an `external_wait` event. It occupies no compute module and no link, but it creates dependency delay.

This artifact does not claim full BusyBarn operator-level LLM integration for prefill/decode. The explicit contract lives in `backend_contract.py`; event creation is centralized through `EventCompiler`.

## Algorithm Boundaries

- ASG Builder: online graph construction from the replayed trace. It records states, LLM/tool executions, dependencies, affinities, phases, and live access counters. It is not a learning model and does not train from future events.
- State Demand Estimator: optional online demand prediction. `heuristic` uses hand-coded state/phase features; `trace_stats` fits replay-trace buckets from a training trace set; `oracle` is only an upper-bound replay mode.
- Persistent State Planner / KV Manager: decides which state KV entries remain resident under the memory budget. `asg-retention-v2-graph-only` uses graph state and recency/live-prompt terms without a demand estimator. `asg-retention-v2-online` adds the estimator. `asg-retention-v2-oracle` uses future replay access and should be read as a diagnostic ceiling.
- Wafer Mapper: maps agents and resident states onto wafer nodes, chooses local/remote/migrate/replicate actions, and optionally emits tool-wait prefetch events.
- Event Compiler: turns planner/mapper decisions into BusyBarn fixed-compute, communication, and external-wait events.
- BusyBarn Backend Runner: runs the event graph with `collective_event_driver` and reports compute/communication timing.

## Runtime Components

- `asg_builder_runtime.py`: ASG state/exec insertion, dependencies, generation edges, and affinity edges. It creates no BusyBarn events.
- `state_planner_runtime.py`: persistent-state planning and node-memory synchronization around `state_manager.update`.
- `wafer_mapper_runtime.py`: execution/state placement decisions over wafer topology. It creates no BusyBarn events.
- `event_compiler.py`: BusyBarn event creation and dependency wiring through `backend_contract.py`.
- `policy_registry.py`: centralized policy definitions for retention, mapping, prefetch, estimator, baseline class, topology, prefetch, and static replica flags.
- `invariant_checks.py`: optional checks for event graph validity, ASG consistency, state locations, and policy output sanity.

## Policy Names

- `nocache`: no persistent KV residency; every LLM step pays prefill for input state tokens.
- `lru-basic`: LRU retention without static shared-prefix replicas.
- `lru-system`: LRU retention with the same static shared-prefix replica support available to stronger system policies.
- `kvflow-like`: workflow-aware retention approximation using next-use/reuse hints, without ASG v2 knapsack planning.
- `asg-retention-v2-graph-only`: ASG graph and v2 retention planner without a demand estimator.
- `asg-retention-v2-online`: graph-only plus the heuristic State Demand Estimator.
- `asg-retention-v2-trace-stats`: graph-only plus an offline trace-stat State Demand Estimator.
- `asg-retention-v2-oracle`: oracle future-demand upper bound for diagnosis.
- `asg-placement-v2-online`: online retention plus topology-aware execution/state placement.
- `asg-prefetch-v2-online`: placement plus windowed prefetch during tool waits.

## Common Commands

```bash
python -m compileall src
```

```bash
python -m src.agent.backend_smoke --cfg src/platform/cfgs/wamis_hd_distributed.cfg --topology wamis --output agent_results/backend_smoke.json
python -m src.agent.artifact_status --output agent_results/artifact_status.json
```

```bash
python -m src.agent.experiment --policy asg --cfg src/platform/cfgs/wamis_hd_distributed.cfg --topology wamis --num-agents 2 --turns 4 --memory-budget-gb 1 --seed 1 --output agent_results/smoke_asg.json --check-invariants
```

```bash
python -m src.agent.experiment --run-all-policies --policy-suite v2 --cfg src/platform/cfgs/wamis_hd_distributed.cfg --topology wamis --num-agents 4 --turns 8 --memory-budget-gb 1 --seed 1 --output-dir agent_results/basic/
```

```bash
python -m src.agent.trace_sources --output agent_results/trace_sources_report.json
python -m src.agent.fetch_public_traces --source agentlens --output-dir traces/downloaded/agentlens --max-files 50 --report agent_results/fetch_agentlens_report.json
python -m src.agent.fetch_public_traces --source codetracer --output-dir traces/downloaded/codetracer --max-files 50 --report agent_results/fetch_codetracer_report.json
python -m src.agent.fetch_public_traces --source swe_gym --output-dir traces/downloaded/swe_gym --max-files 50 --report agent_results/fetch_swegym_report.json
python -m src.agent.fetch_hf_traces --source exgentic_otel --output-dir traces/downloaded/hf/exgentic_otel --report agent_results/fetch_hf_exgentic_report.json --max-files 2 --max-parquet-rows 40
python -m src.agent.fetch_hf_traces --source pagarsky_agent_trace --output-dir traces/downloaded/hf/pagarsky_agent_trace --report agent_results/fetch_hf_pagarsky_report.json --max-files 5 --max-parquet-rows 40
python -m src.agent.fetch_hf_traces --source itbench_trajectories --output-dir traces/downloaded/hf/itbench_trajectories --report agent_results/fetch_hf_itbench_report.json --max-files 30
python -m src.agent.fetch_hf_traces --source codetracebench_hf --output-dir traces/downloaded/hf/codetracebench_hf --report agent_results/fetch_hf_codetracebench_report.json --max-files 40
python -m src.agent.extract_trace_archives --input-dir traces/real_raw/codetracebench --output-dir traces/real_extracted/codetracebench --max-archives 20 --report agent_results/archive_extraction_report.json
python -m src.agent.inspect_trace_artifacts --input-dir traces/real_extracted/codetracebench --output agent_results/codetracer_artifact_inventory.json
python -m src.agent.trace_audit --trace-dir traces/real --trace-format auto --output agent_results/real_trace_audit.json
python -m src.agent.trace_profile --trace-dir traces/real --trace-format auto --output agent_results/real_trace_profile.json
```

```bash
python -m src.agent.trace_opportunity_selector --trace-dirs traces/real traces/real_extracted traces/downloaded --trace-format auto --output-dir agent_results/trace_selection --max-traces 200 --min-turns 4 --delayed-reuse-k 8 --memory-budget-gb 0.5
python -m src.agent.build_real_trace_set --input-dirs traces/real traces/real_extracted traces/downloaded --output-dir traces/real_high_quality --trace-format auto --min-turns 4 --max-traces 200 --audit-output agent_results/high_quality_trace_audit.json --manifest agent_results/high_quality_trace_manifest.json --selection-report agent_results/trace_selection/selection_report.json
```

```bash
python -m src.agent.synthetic_trace_generator --output-dir traces/synthetic_agent --opportunity-count 12 --control-count 6 --stress-count 4 --turns 28 --agents 4 --seed 1
python -m src.agent.trace_opportunity_selector --trace-dirs traces/synthetic_agent/raw --trace-format normalized_json --output-dir agent_results/trace_selection_synthetic --max-traces 64 --min-turns 4 --delayed-reuse-k 8 --memory-budget-gb 0.05
python -m src.agent.run_paper_evaluation --opportunity-trace-dir traces/synthetic_agent/opportunity --control-trace-dir traces/synthetic_agent/control --output-dir agent_results/paper_eval_synthetic --memory-budgets 0.01 0.02 0.05 0.1 --concurrency-levels 1 2 4 8 --arrival-models round_robin burst poisson tool_wait_aware --prediction-modes heuristic trace_stats --evidence-class synthetic
```

```bash
python -m src.agent.serving_workload --trace-dir traces/real_high_quality/opportunity --trace-format normalized_json --arrival-model round_robin --concurrency 8 --output traces/serving/opportunity_rr_c8.json
```

```bash
python -m src.agent.run_paper_evaluation --opportunity-trace-dir traces/real_high_quality/opportunity --control-trace-dir traces/real_high_quality/control --output-dir agent_results/paper_eval --memory-budgets 0.25 0.5 1 2 --concurrency-levels 1 2 4 8 16 --arrival-models round_robin burst poisson tool_wait_aware --prediction-modes heuristic trace_stats
python -m src.agent.plot_paper_eval --input-dir agent_results/paper_eval --output-dir agent_results/paper_figures
```

## Useful CLI Flags

- `--agent-placement {round_robin,compact}` controls deterministic agent home assignment.
- `--per-node-memory-mb` sets a lightweight per-node KV placement budget.
- `--policy-suite {basic,v2,oracle,all}` chooses synthetic experiment policies; real-trace replay currently uses the v2 suite.
- `--future-horizon` and `--oracle-future` enable synthetic-trace future demand for ASG v2/oracle policies.
- `--horizon` controls the real-trace estimator/oracle lookahead window.
- `--prediction-mode {heuristic,trace_stats}` selects the online State Demand Estimator for real-trace online policies.
- `--reject-accumulated-fallback` rejects traces whose prompt reconstruction mostly comes from accumulated fallback history.
- `--require-high-quality` requires paper-usable traces and writes a skipped report if none are available.
- `--allow-smoke-traces` permits medium/low-quality traces for developer smoke tests only and labels outputs `paper_usable=false`.
- `--enable-common-observation-compression` explicitly enables observation compression for all real-trace policies.
- `--enable-asg-observation-compression` explicitly enables ASG-specific observation compression and labels it in outputs.
- `--disable-observation-compression` is a global off switch. Real-trace replay does not enable ASG-specific compression by default.
- `--check-invariants` validates event graph dependencies, BusyBarn event locations, ASG state/exec links, state locations, and selected policy outputs. It defaults off for quick dev runs and is enabled by `run_paper_evaluation.py`.

## Real Trace Evaluation

`trace_loader.py` accepts normalized JSON/JSONL and best-effort public trajectory adapters for SWE-Gym, CodeTraceBench/CodeTracer, AgentLens/OpenHands, OpenTelemetry span traces, and generic ReAct JSONL. Source-specific adapters preserve raw payloads in metadata and fall back to message/tool/observation detection when public schemas are incomplete or drift.

`trace_normalizer.py` labels each LLM event with prompt reconstruction metadata:

- `explicit`: the uploaded or normalized trace already contains explicit state references.
- `state_tree`: source-specific adapter found persistent state tree or memory fields.
- `step_context`: source-specific adapter found per-step state/context fields.
- `exact_otel_messages`: OpenTelemetry spans expose exact `gen_ai.input.messages` and `gen_ai.output.messages`.
- `message_history`: input states were reconstructed from message history.
- `accumulated_fallback`: the adapter had to build prompts from accumulated thought/action/observation text.

It also reports `full_history_likely` and a monotonic growth score. Lossy full-history-like traces can still be replayed only with `--allow-smoke-traces`, but they must not be presented as paper evidence. Exact OpenTelemetry prompt snapshots are treated as real prompt evidence, not reconstructed history.

`trace_audit.py` reports source quality, prompt reconstruction risk, nontrivial tool/state reuse, paper-usable trace count, smoke-only trace count, unusable trace count, and exclusion reasons. `trace_profile.py` reports reuse-distance, state-type, delayed-reuse, cross-agent-reuse, LRU-regret, tool, and phase statistics, and writes both state and reuse CSVs.

`inspect_trace_artifacts.py` inventories extracted files and reports whether richer state-tree, memory, tool-call, prompt, patch, or test artifacts exist. `extract_trace_archives.py` attempts `.tar.zst` extraction and records candidate trace files. `fetch_public_traces.py` records whether AgentLens/OpenHands, SWE-Gym, or CodeTracer samples are locally available or require manual download. `fetch_hf_traces.py` fetches public Hugging Face trace datasets through a mirror endpoint when possible and converts Parquet rows into one JSON trace per row.

`trace_opportunity_selector.py` is the pre-registered selector. It uses only workload-intrinsic metrics before any ASG policy replay: delayed high-value reuse, LRU-regret candidates, phase-dependent reuse, cross-agent reuse, reusable file/failure/edit state, memory pressure, and prompt reconstruction quality. It writes the full candidate scores, opportunity manifest, matched-control manifest, smoke-only manifest, unusable manifest, and `selection_protocol.md`. ASG/LRU/KVFlow performance is not used for selection.

Quality gates are:

- `paper_usable`: non-full-history traces with tool/state structure and intrinsic ASG opportunity.
- `opportunity_rich`: paper-usable traces with high pre-registered ASG opportunity score.
- `matched_control`: paper-usable traces with low opportunity score, retained as a control set.
- `smoke_only`: useful for pipeline testing, not paper claims.
- `unusable`: cannot be normalized or lacks agent/tool/state structure.

`build_real_trace_set.py` writes normalized opportunity traces into `traces/real_high_quality/opportunity/`, matched controls into `traces/real_high_quality/control/`, and smoke traces into `traces/real_smoke/`. If no paper-usable traces exist, it writes `agent_results/high_quality_trace_missing_report.json` instead of fabricating data.

Public traces are often single-workflow trajectories. `serving_workload.py` composes them into controlled concurrent serving workloads with `round_robin`, `burst`, `poisson`, `staggered`, or `tool_wait_aware` arrival models while preserving each workflow's internal event order. `run_paper_evaluation.py` uses the in-memory `compose_traces(...)` path, so `arrival_model` changes event order rather than only labeling CSV rows.

`run_paper_evaluation.py` runs full-set, opportunity-subset, matched-control, memory, concurrency, and arrival-model evaluations only when paper-usable opportunity traces exist. Otherwise it writes `agent_results/paper_eval/skipped_missing_opportunity_traces.json`. Mapping and prefetch ablations are not fabricated: if the dedicated ablation policy runner is not implemented, the artifact writes `mapping_ablation_skipped.json` and `prefetch_ablation_skipped.json` rather than empty CSVs that look like results. `plot_paper_eval.py` generates figures only from paper-usable CSVs; otherwise it writes `agent_results/paper_figures/skipped_missing_paper_usable_data.json`.

`real_trace_experiment.py` writes one JSON per policy plus `summary.csv`. The summary separates:

- `baseline_class`: `basic`, `system`, `asg`, or `oracle`.
- `estimator_mode`: `none`, `heuristic`, `trace_stats`, or `oracle`.
- `asg_builder_enabled`: whether the policy uses ASG graph/planner logic.
- `oracle_future`: whether future replay access was used.
- `prompt_reconstruction_quality`: a compact label for the replayed trace set.
- `data_quality`, `paper_usable`, `num_high_quality_traces`, `num_smoke_traces`: whether the replay can support paper claims.

Current local CodeTraceBench samples remain message-history/full-history reconstructed and have zero delayed reuse, zero cross-agent reuse, and zero LRU-regret candidates. Artifact inspection found no richer state-tree or per-step state context files in the extracted archives. They are smoke-only and should not be used for speedup claims.

Current Hugging Face sample run (`agent_results/trace_selection_hf/selection_report.json`) loaded 181 traces from Exgentic, AgentTrace, ITBench, and CodeTraceBench-HF samples. The selector found 2 paper-usable opportunity traces and 0 matched-control traces. On those two opportunity traces, `asg-retention-v2-online` reduces effective prefill by about 0.41% versus `lru-system` at 0.5GB and 8-way serving, and by about 0.57% under the tightest 0.05GB stress setting. This is a valid end-to-end real-trace run, but it is not evidence that ASG is far beyond LRU; more opportunity-rich/control traces or a stronger retention objective are required before making that claim.

## Synthetic Trace Testing

`synthetic_trace_generator.py` creates controlled normalized traces for algorithm and pipeline testing. The generated suite includes delayed-reuse opportunity traces, short-reuse controls, and memory-pressure stress traces. Every generated event has `metadata.synthetic=true` plus a `scenario` label, and the generator writes `synthetic_manifest.json`, `synthetic_audit.json`, and `synthetic_profile.json`.

Synthetic traces are useful for validating whether ASG retention, placement, prefetch, event compilation, invariant checks, and serving-workload interleaving react to the expected workload structure. They must not be mixed with real-trace paper evidence. Use `run_paper_evaluation.py --evidence-class synthetic` for these runs; the resulting CSVs include `evidence_class=synthetic` and `can_support_real_paper_claims=False`.

## Baseline Interpretation

Use `lru-system`, not `lru-basic`, as the main practical LRU baseline when judging system-level gains. `lru-basic` is retained for ablation because it disables static shared-prefix replicas.

Compare `asg-retention-v2-graph-only` against `lru-system` to isolate graph/planner value without a demand estimator. Compare `asg-retention-v2-online` against graph-only to isolate heuristic estimator value. Compare `asg-retention-v2-trace-stats` separately because it is trained from trace statistics. Compare online against `asg-retention-v2-oracle` only as an oracle-gap diagnostic.

Any run with common or ASG-specific observation compression must be labeled by `observation_compression_class`; do not mix compressed and uncompressed policies when claiming cache-policy superiority.

## Metrics

- `effective_prefill_tokens`: tokens paid as prefill after cache hits.
- `cache_hit_ratio`: fraction of input states served from resident KV.
- `state_misses`: non-resident input states that force prefill.
- `LRU_regret_states_preserved` / `LRU_regret_tokens_preserved`: high-value delayed-reuse candidates still resident at the end of replay.
- `oracle_gap`: normalized distance between `lru-system` and `asg-retention-v2-oracle` in effective prefill tokens.
- `remote_read_bytes`, `demand_migration_bytes`, `prefetch_migration_bytes`: KV movement/access bytes.
- `model_compute_cycles` / `model_comm_cycles`: BusyBarn pure compute/communication timing breakdown.
- `exposed_migration_cycles` / `exposed_prefetch_cycles`: KV movement cycles that were not hidden by tool waits.
- `compressed_observation_tokens_saved` / `num_compressed_observations`: explicit observation compression effect.
- `approximate_non_wait_cycles`: total cycles minus external wait cycles. This is a coarse diagnostic, not a paper-critical LLM-side timing metric.

## Limitations

- Real-trace adapters are best-effort and depend on public dataset schemas that may drift.
- Message-history reconstruction is not exact tokenizer-level prompt composition; full-history-like traces are flagged and should be interpreted cautiously.
- Prompt segmentation is heuristic unless the source provides exact prompt state or state-tree fields.
- `semantic_key` helps analysis but does not imply exact KV equality; `exact_token_hash` is approximate unless tokenizer ids/token ids are provided.
- No real KV tensor layout or physical DMA implementation is modeled.
- KV bytes and prefill/decode costs use a configurable analytical model.
- Tool execution is replayed from trace latency or a fixed fallback latency when missing.
- Wafer mapping remains analytical; topology-aware placement and prefetch are heuristic, not globally optimal.
- The oracle policy uses future replay access and is not an online deployable algorithm.
- Trace-stat demand estimation can overfit if training and evaluation use the same small trace set; report that condition when it happens.
- Current local HF selection has 2 paper-usable opportunity traces and 0 matched controls. This is enough for problem-existence and pipeline validation, but not enough to claim full paper-readiness or broad speedup superiority.
