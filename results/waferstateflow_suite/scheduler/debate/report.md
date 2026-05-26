# WaferStateFlow Report

## 1. Executive Summary

Workflow `debate` redundancy ratio is 3.98. Best simulated scheduler is `WaferStateFlow` with latency 20.566. `WaferStateFlow` latency is 20.566.

## 2. Problem Characterization

- Input redundancy ratio: 3.98
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
| `WaferStateFlow` | 20.566 | 44052 | 48479232 | 0.000 | 0.030 | 30.403 | 2.20 |
| `flat_sequential` | 84.538 | 175272 | 175272 | 0.000 | 0.000 | 0.000 | 1.00 |
| `helium_like_operator_schedule` | 23.119 | 44052 | 44052 | 0.000 | 0.000 | 0.000 | 2.20 |
| `kvflow_like_future_eviction` | 23.119 | 44052 | 44052 | 0.000 | 0.000 | 0.000 | 2.20 |
| `prefix_cache_like` | 26.064 | 87272 | 87272 | 0.000 | 0.000 | 0.000 | 2.20 |
| `replicate_all_hot_states` | 26.064 | 44052 | 48479232 | 0.000 | 0.030 | 0.000 | 2.20 |
| `request_parallel_gpu_like` | 26.064 | 103272 | 103272 | 0.000 | 0.066 | 0.000 | 2.20 |
| `single_pin_hot_state` | 26.064 | 44052 | 626117632 | 0.000 | 0.031 | 0.000 | 2.20 |
| `wafer_request_centric` | 26.064 | 103272 | 626117632 | 0.000 | 0.031 | 0.000 | 2.20 |

## 6. Ablations

`replicate_all_hot_states` gives latency 26.064 with memory pressure 0.030; `single_pin_hot_state` gives latency 26.064 and byte-hop 626117632.

## 7. Failure Cases

No baseline beat WaferStateFlow in this run, but this does not prove the platform claim. Run low-fanout or memory-rich sweeps to find negative cases.

## 8. What This Means for the Paper

- H1: supported for this workflow.
- H2: supported for this workflow.
- H3: requires dynamic-hotness sweep; this single run is not enough.
- H4: supported only if WaferStateFlow beats request/operator-centric baselines outside cache-rich regimes.
- Next experiments: dynamic-hotness probability sweep, fanout sweep, and memory-capacity sensitivity.
