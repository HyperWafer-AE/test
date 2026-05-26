# WaferStateFlow Report

## 1. Executive Summary

Workflow `trading` redundancy ratio is 6.73. Best simulated scheduler is `WaferStateFlow` with latency 51.461. `WaferStateFlow` latency is 51.461.

## 2. Problem Characterization

- Input redundancy ratio: 6.73
- Hotness top-share: 1.00
- Dynamic hot-state count: 2

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
| `WaferStateFlow` | 51.461 | 143824 | 677729280 | 0.000 | 0.069 | 1608.362 | 7.22 |
| `flat_sequential` | 354.446 | 702392 | 702392 | 0.000 | 0.000 | 0.000 | 1.00 |
| `helium_like_operator_schedule` | 56.952 | 104428 | 104428 | 0.000 | 0.000 | 0.000 | 7.22 |
| `kvflow_like_future_eviction` | 56.952 | 104428 | 104428 | 0.000 | 0.000 | 0.000 | 7.22 |
| `prefix_cache_like` | 63.288 | 196792 | 196792 | 0.000 | 0.000 | 0.000 | 7.22 |
| `replicate_all_hot_states` | 63.288 | 145592 | 677729280 | 0.000 | 0.069 | 0.000 | 7.22 |
| `request_parallel_gpu_like` | 63.288 | 230040 | 230040 | 0.000 | 0.104 | 0.000 | 7.22 |
| `single_pin_hot_state` | 63.288 | 145592 | 3905377280 | 0.000 | 0.084 | 0.000 | 7.22 |
| `wafer_request_centric` | 63.288 | 230040 | 3905377280 | 0.000 | 0.084 | 0.000 | 7.22 |

## 6. Ablations

`replicate_all_hot_states` gives latency 63.288 with memory pressure 0.069; `single_pin_hot_state` gives latency 63.288 and byte-hop 3905377280.

## 7. Failure Cases

No baseline beat WaferStateFlow in this run, but this does not prove the platform claim. Run low-fanout or memory-rich sweeps to find negative cases.

## 8. What This Means for the Paper

- H1: supported for this workflow.
- H2: supported for this workflow.
- H3: requires dynamic-hotness sweep; this single run is not enough.
- H4: supported only if WaferStateFlow beats request/operator-centric baselines outside cache-rich regimes.
- Next experiments: dynamic-hotness probability sweep, fanout sweep, and memory-capacity sensitivity.
