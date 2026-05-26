# WaferStateFlow Report

## 1. Executive Summary

Workflow `software_dev` redundancy ratio is 4.18. Best simulated scheduler is `WaferStateFlow` with latency 23.389. `WaferStateFlow` latency is 23.389.

## 2. Problem Characterization

- Input redundancy ratio: 4.18
- Hotness top-share: 1.00
- Dynamic hot-state count: 0

## 3. Method

- State Access Graph keeps task, document, tool, intermediate, and output states explicit.
- Hotness combines token size, future accesses, access cost, and criticality.
- State policy chooses inline/cache/pin/replicate/shard/evict/recompute using expected saved cost.
- State-centric wave scheduling forms ready-operator waves around hot input states.
- Wafer placement uses a mesh byte-hop model with region memory pressure.

## 4. Baselines

| baseline | modeled behavior |
| --- | --- |
| `flat_sequential` | sequential flat prompt materialization; no cache or parallelism |
| `request_parallel_gpu_like` | ready requests assigned to worker-local caches |
| `prefix_cache_like` | only exact root prefix-like states are globally reused |
| `helium_like_operator_schedule` | operator-centric wave with per-wave unique-state reuse |
| `kvflow_like_future_eviction` | future-aware cache admits highest token-weighted fanout states |
| `wafer_request_centric` | wafer backend without state-centric wave formation |
| `replicate_all_hot_states` | ablation that blindly replicates hot states |
| `single_pin_hot_state` | ablation that pins hot states centrally and exposes hotspots |
| `WaferStateFlow` | hot-state-seeded wave scheduling with policy-driven placement |

## 5. Results

| baseline | latency | materialization bytes | byte-hop | max link util | memory pressure | crit wait | avg wave |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `WaferStateFlow` | 23.389 | 39340 | 0 | 0.000 | 0.034 | 114.565 | 2.80 |
| `flat_sequential` | 74.676 | 147952 | 147952 | 0.000 | 0.000 | 0.000 | 1.00 |
| `helium_like_operator_schedule` | 25.462 | 35372 | 35372 | 0.000 | 0.000 | 0.000 | 2.80 |
| `kvflow_like_future_eviction` | 25.462 | 35372 | 35372 | 0.000 | 0.000 | 0.000 | 2.80 |
| `prefix_cache_like` | 27.854 | 55152 | 55152 | 0.000 | 0.000 | 0.000 | 2.80 |
| `replicate_all_hot_states` | 27.854 | 42352 | 380456960 | 0.000 | 0.019 | 0.000 | 2.80 |
| `request_parallel_gpu_like` | 27.854 | 76784 | 76784 | 0.000 | 0.046 | 0.000 | 2.80 |
| `single_pin_hot_state` | 27.854 | 42352 | 478760960 | 0.000 | 0.021 | 0.000 | 2.80 |
| `wafer_request_centric` | 27.854 | 76784 | 478760960 | 0.000 | 0.021 | 0.000 | 2.80 |

## 6. Ablations

`replicate_all_hot_states` gives latency 27.854 with memory pressure 0.019; `single_pin_hot_state` gives latency 27.854 and byte-hop 478760960.

## 7. Failure Cases

No baseline beat WaferStateFlow in this run, but this does not prove the platform claim. Run low-fanout or memory-rich sweeps to find negative cases.

## 8. What This Means for the Paper

- H1: supported for this workflow.
- H2: supported for this workflow.
- H3: requires dynamic-hotness sweep; this single run is not enough.
- H4: supported only if WaferStateFlow beats request/operator-centric baselines outside cache-rich regimes.
- Next experiments: dynamic-hotness probability sweep, fanout sweep, and memory-capacity sensitivity.
