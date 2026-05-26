# WaferStateFlow Goals

## Current Thesis

Agent workflows should be represented as state access graphs rather than only as
request streams. The first objective is to verify whether synthetic agent
workflows exhibit token/state redundancy and skewed hot-state fanout. Only after
that evidence exists should wafer-aware placement and wave scheduling be treated
as a promising backend mechanism.

## Goals

### G0: Repository reconnaissance
- Status: done
- Evidence:
  - Current checkout physically contains `pyproject.toml` and `uv.lock`; tracked
    historical `kvring`, configs, scripts, tests, and result artifacts are
    deleted in the working tree. These deletions are preserved and not reverted.
  - Historical `kvring` code in git contains wafer mesh, routing, placement, and
    byte-hop concepts, but it is KV-ring specific and does not expose
    AgentGraph, TraceRecord, State Access Graph, workflow generators, or
    state-centric workflow scheduling.
  - `pyproject.toml` already has useful dependencies: numpy, pandas,
    networkx, matplotlib, pytest.
- Next action: Build a new `waferstateflow` package with a small IR first, then
  verify it with tests before implementing analyzers and schedulers.

### G1: Build State Access Graph IR
- Status: done
- Evidence:
  - Implemented `StateNode`, `OperatorNode`, `AccessEdge`,
    `WorkflowTrace`, and `StateAccessGraph`.
  - Supports mutation, state/operator queries, producer/consumer fanout,
    operator and full-graph topological order, lifetime estimation, CSV export,
    and JSON export.
  - `python -m pytest tests/test_state_graph.py -q` passed with the required
    `S_task -> Analyst -> S_summary -> Reviewer` graph.
- Next action: Use the IR as the execution substrate for synthetic workflows.

### G2: Implement synthetic workflow generators
- Status: done
- Evidence:
  - Implemented reproducible generators for mapreduce, debate, reflection,
    iterative, parallel_chains, trading, and software_dev.
  - Each generator emits a `StateAccessGraph` with shared states, unique branch
    states, LLM/local operators, produced intermediate states, lifetimes, and
    synthetic dynamic-hot candidate metadata.
  - `python -m pytest tests/test_state_graph.py tests/test_workflow_generators.py -q`
    passed.
- Next action: Characterize redundancy and hot-state skew on generated graphs.

### G3: Redundancy characterization
- Status: done
- Evidence:
  - Implemented `redundancy_analyzer.py` with input redundancy ratio,
    token-weighted fanout, hotness skew, dynamic hot-state rate, and duplicate
    materialization bytes.
  - Ran `python -m waferstateflow.experiments.run_problem_characterization
    --workflow trading --batch-size 16 --seed 0 --out
    results/waferstateflow_characterization`.
  - Trading synthetic run reports redundancy ratio 6.73, duplicate
    materialization 597,964 bytes, and top hot states dominated by task and
    market context.
- Next action: Use characterization as the gate for scheduler claims.

### G4: Implement baseline execution models
- Status: done
- Evidence:
  - Implemented `flat_sequential`, `request_parallel_gpu_like`,
    `prefix_cache_like`, `helium_like_operator_schedule`,
    `kvflow_like_future_eviction`, `wafer_request_centric`,
    `replicate_all_hot_states`, `single_pin_hot_state`, and `WaferStateFlow` in
    `simulator.py`.
  - `results/waferstateflow_scheduler/simulation_summary.csv` contains all
    baselines with latency, materialization, byte-hop, link utilization, memory
    pressure, critical wait, and wave size metrics.
- Next action: Treat these as approximate baselines and refine calibration with
  real traces/hardware later.

### G5: Hotness model and dynamic state policy
- Status: done
- Evidence:
  - Implemented EWMA dynamic hotness, static fanout/criticality hotness, and
    hysteresis promotion/demotion in `hotness.py`.
  - Implemented inline/cache_text/cache_kv/pin/replicate/shard/evict/recompute
    decisions in `state_policy.py`.
  - `tests/test_hotness.py` and `tests/test_policy_decision.py` pass.
- Next action: Use dynamic-hotness sweep output to decide whether H3 is worth
  claiming.

### G6: State-centric wave scheduler
- Status: done
- Evidence:
  - Implemented request-centric, operator-centric, and state-centric wave
    schedulers in `schedulers.py`.
  - State-centric scheduler chooses seed hot states, groups ready consumers,
    scores reuse/batching/wait benefit, and avoids over-waiting critical ops.
  - `tests/test_wave_scheduler.py` covers high fanout waves, critical-path
    protection, and low-hotness non-waves.
- Next action: Extend with richer wave placement and NoC contention modeling in
  future work.

### G7: Wafer topology and physical mapping simulator
- Status: done
- Evidence:
  - Implemented `WaferTopology`, region IDs, Manhattan distance, byte-hop,
    state placement, movement cost, and memory pressure in
    `wafer_topology.py`.
  - `tests/test_simulator_sanity.py` verifies near placement, replication
    benefit for small states, and no blind replication for large states under
    memory pressure.
- Next action: Calibrate bandwidth/capacity constants against real hardware
  when traces are available.

### G8: End-to-end experiments
- Status: done
- Evidence:
  - Added CLI entrypoints:
    - `waferstateflow.experiments.run_problem_characterization`
    - `waferstateflow.experiments.run_scheduler_comparison`
    - `waferstateflow.experiments.run_dynamic_hotness_sweep`
    - `waferstateflow.experiments.run_wafer_sensitivity`
  - Generated complete result directories:
    - `results/waferstateflow_characterization`
    - `results/waferstateflow_scheduler`
    - `results/waferstateflow_scheduler_lowfanout`
    - `results/waferstateflow_dynamic_hotness`
    - `results/waferstateflow_sensitivity`
  - Every result directory contains metadata/config/state/operator/edge/hotness/
    policy/wave/simulation/report/figures artifacts.
  - Dynamic sweep: `p(dynamic)=0` gives no dynamic benefit; higher probabilities
    reduce materialization bytes but not latency in this model.
  - Sensitivity sweep found negative cases: small shared state and branch width
    1 can favor request-centric baselines, so wafer is not universally better.
- Next action: Use real workflow traces and calibrated serving measurements
  before making paper-grade speedup claims.

### G9: Prototype hardening after first completion
- Status: done
- Evidence:
  - User requested continuation after the first complete prototype pass.
  - Added `waferstateflow.experiments.run_workflow_suite`, which accepts
    `--workflows all`, runs characterization and scheduler comparison, writes
    per-workflow artifacts, and writes aggregate CSV/report/figures.
  - Added `tests/test_experiment_artifacts.py` to verify metadata/config/state/
    operator/edge/hotness/policy/wave/simulation/report files for
    characterization, scheduler comparison, and suite outputs.
  - `python -m pytest tests -q` now reports 17 passing tests.
  - Ran `python -m waferstateflow.experiments.run_workflow_suite --workflows
    all --mode both --batch-size 8 --mesh 16x16 --state-policy dynamic --seed
    0 --out results/waferstateflow_suite`.
  - Suite result: all 7 workflow types show H1/H2 support in this synthetic
    setting; scheduler rows include a tie/counterexample where `iterative` has
    `wafer_request_centric` no worse than WaferStateFlow.
- Next action: Next useful iteration is trace ingestion or calibration, not more
  synthetic scaffolding.

### G10: Credibility hardening and correctness fixes
- Status: done
- Evidence:
  - Added `README.md` and `CLAIMS.md` documenting that WaferStateFlow is a
    synthetic problem-discovery prototype, not paper-grade evidence.
  - Preserved unrelated historical project files; current git status shows only
    WaferStateFlow-related modifications/untracked docs, not deleted historical
    files from this pass.
  - Added `StateNode.prefix_compatible`, `StateNode.kv_cacheable`, and
    `StateNode.prompt_position`. KV policy now permits `cache_kv` only when both
    `prefix_compatible` and `kv_cacheable` are true.
  - Removed baseline-name hardcoded prefill multipliers (`WaferStateFlow=0.72`,
    `Helium=0.85`) and replaced timing with `BackendProfile`, shared by all
    baselines.
  - Fixed placement memory accounting: replicas add one full state per replica
    region; shards add `ceil(size / shard_count)` per shard region.
  - Added link-level XY NoC routing and directed link-load accumulation. Reports
    now include `max_link_load`, `p95_link_load`, and `hotspot_region`; link
    utilization is computed from the max directed link load.
  - Removed dynamic hotness injection from static fanout. Dynamic hotness is
    updated from actual state accesses during simulation. Added
    `state_access_events.csv`.
  - Split approximate Helium-like and KVFlow-like behavior: Helium-like groups
    ready operators by prefix/state template; KVFlow-like uses future-use cache
    admission/eviction under capacity.
  - Added tests for KV prefix constraints, scheduler-name-independent operator
    time, memory accounting, directed XY link loads, state access events, and
    non-identical Helium/KVFlow behavior.
  - `python -m pytest tests -q` passed with 24 tests.
  - Regenerated:
    - `results/waferstateflow_characterization`
    - `results/waferstateflow_scheduler`
    - `results/waferstateflow_scheduler_lowfanout`
    - `results/waferstateflow_dynamic_hotness`
    - `results/waferstateflow_sensitivity`
    - `results/waferstateflow_suite`
  - Negative results are retained: dynamic hotness sweep reports H3 not
    supported in this synthetic sweep; sensitivity sweep is mostly won by
    non-WaferStateFlow baselines after bias removal.
- Next action: The next credible step is real trace ingestion and calibration.

### G11: Residual redundancy decision analysis
- Status: done
- Evidence:
  - Added `waferstateflow/residual_redundancy_analyzer.py`.
  - The analyzer decomposes raw duplicated state tokens into exact-prefix cache
    coverage, deterministic operator-output cache coverage, KVFlow-like
    future-cache coverage under capacity, and residual uncovered redundancy.
  - Per-state derived labels are exported: `prefix_covered`,
    `operator_cache_covered`, `kvflow_cache_covered`, `residual_candidate`, and
    `reason`.
  - Workflow-level metrics are exported: `raw_redundancy_ratio`,
    `residual_redundancy_ratio`, `dynamic_hot_residual_fraction`,
    `residual_token_weighted_fanout`, and `wafer_opportunity_score`.
  - Added `waferstateflow.experiments.run_residual_analysis`; it does not add
    scheduling or placement features.
  - Added `tests/test_residual_redundancy.py` for decomposition and KVFlow
    capacity behavior.
  - Ran `python -m pytest tests -q`; result: 26 passed.
  - Ran `python -m waferstateflow.experiments.run_residual_analysis --workflows
    all --seed 0 --out results/waferstateflow_residual_analysis`.
  - Generated:
    - `results/waferstateflow_residual_analysis/report.md`
    - `results/waferstateflow_residual_analysis/residual_state_table.csv`
    - `results/waferstateflow_residual_analysis/residual_summary.csv`
  - Negative result retained: with default KVFlow-like capacity, all seven
    synthetic workflows have residual redundancy ratio 1.00 and zero residual
    token-weighted fanout. The report recommends abandon/pivot for the
    hot-state wafer-mapping paper direction under these synthetic settings.
- Next action: If continuing, collect real traces or constrain baseline cache
  capacity with justified measurements before revisiting wafer-specific claims.
