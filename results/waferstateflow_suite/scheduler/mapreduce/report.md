# WaferStateFlow Report

## 1. Executive Summary

Workflow `mapreduce` redundancy ratio is 3.00. Best simulated scheduler is `WaferStateFlow` with latency 10.931. `WaferStateFlow` latency is 10.931.

## 2. Problem Characterization

- Input redundancy ratio: 3.00
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
| `WaferStateFlow` | 10.931 | 21972 | 23791616 | 0.000 | 0.015 | 4.000 | 2.50 |
| `flat_sequential` | 33.236 | 65972 | 65972 | 0.000 | 0.000 | 0.000 | 1.00 |
| `helium_like_operator_schedule` | 11.827 | 21972 | 21972 | 0.000 | 0.000 | 0.000 | 2.50 |
| `kvflow_like_future_eviction` | 11.827 | 21972 | 21972 | 0.000 | 0.000 | 0.000 | 2.50 |
| `prefix_cache_like` | 12.860 | 37972 | 37972 | 0.000 | 0.000 | 0.000 | 2.50 |
| `replicate_all_hot_states` | 12.860 | 21972 | 23791616 | 0.000 | 0.015 | 0.000 | 2.50 |
| `request_parallel_gpu_like` | 12.860 | 57972 | 57972 | 0.000 | 0.033 | 0.000 | 2.50 |
| `single_pin_hot_state` | 12.860 | 21972 | 236783616 | 0.000 | 0.017 | 0.000 | 2.50 |
| `wafer_request_centric` | 12.860 | 57972 | 236783616 | 0.000 | 0.017 | 0.000 | 2.50 |

## 6. Ablations

`replicate_all_hot_states` gives latency 12.860 with memory pressure 0.015; `single_pin_hot_state` gives latency 12.860 and byte-hop 236783616.

## 7. Failure Cases

No baseline beat WaferStateFlow in this run, but this does not prove the platform claim. Run low-fanout or memory-rich sweeps to find negative cases.

## 8. What This Means for the Paper

- H1: supported for this workflow.
- H2: supported for this workflow.
- H3: requires dynamic-hotness sweep; this single run is not enough.
- H4: supported only if WaferStateFlow beats request/operator-centric baselines outside cache-rich regimes.
- Next experiments: dynamic-hotness probability sweep, fanout sweep, and memory-capacity sensitivity.
