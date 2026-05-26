# WaferStateFlow Report

## 1. Executive Summary

Workflow `trading` redundancy ratio is 5.93. Best simulated scheduler is `WaferStateFlow` with latency 28.073. `WaferStateFlow` latency is 28.073.

## 2. Problem Characterization

- Input redundancy ratio: 5.93
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
| `WaferStateFlow` | 28.073 | 80808 | 164954112 | 0.000 | 0.034 | 197.114 | 6.60 |
| `flat_sequential` | 179.654 | 356008 | 356008 | 0.000 | 0.000 | 0.000 | 1.00 |
| `helium_like_operator_schedule` | 30.818 | 60036 | 60036 | 0.000 | 0.000 | 0.000 | 6.60 |
| `kvflow_like_future_eviction` | 30.818 | 60036 | 60036 | 0.000 | 0.000 | 0.000 | 6.60 |
| `prefix_cache_like` | 33.986 | 106408 | 106408 | 0.000 | 0.000 | 0.000 | 6.60 |
| `replicate_all_hot_states` | 33.986 | 80808 | 164954112 | 0.000 | 0.034 | 0.000 | 6.60 |
| `request_parallel_gpu_like` | 33.986 | 178752 | 178752 | 0.000 | 0.067 | 0.000 | 6.60 |
| `single_pin_hot_state` | 33.986 | 80808 | 967770112 | 0.000 | 0.050 | 0.000 | 6.60 |
| `wafer_request_centric` | 33.986 | 178752 | 967770112 | 0.000 | 0.050 | 0.000 | 6.60 |

## 6. Ablations

`replicate_all_hot_states` gives latency 33.986 with memory pressure 0.034; `single_pin_hot_state` gives latency 33.986 and byte-hop 967770112.

## 7. Failure Cases

No baseline beat WaferStateFlow in this run, but this does not prove the platform claim. Run low-fanout or memory-rich sweeps to find negative cases.

## 8. What This Means for the Paper

- H1: supported for this workflow.
- H2: supported for this workflow.
- H3: requires dynamic-hotness sweep; this single run is not enough.
- H4: supported only if WaferStateFlow beats request/operator-centric baselines outside cache-rich regimes.
- Next experiments: dynamic-hotness probability sweep, fanout sweep, and memory-capacity sensitivity.
