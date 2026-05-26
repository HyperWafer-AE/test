# WaferStateFlow Report

## 1. Executive Summary

Workflow `iterative` redundancy ratio is 1.33. Best simulated scheduler is `wafer_request_centric` with latency 5.064. `WaferStateFlow` latency is 5.064.

## 2. Problem Characterization

- Input redundancy ratio: 1.33
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
| `WaferStateFlow` | 5.064 | 9728 | 0 | 0.000 | 0.007 | 0.000 | 1.00 |
| `flat_sequential` | 5.064 | 9728 | 9728 | 0.000 | 0.000 | 0.000 | 1.00 |
| `helium_like_operator_schedule` | 5.064 | 7328 | 7328 | 0.000 | 0.000 | 0.000 | 1.00 |
| `kvflow_like_future_eviction` | 5.064 | 7328 | 7328 | 0.000 | 0.000 | 0.000 | 1.00 |
| `prefix_cache_like` | 5.064 | 8912 | 8912 | 0.000 | 0.000 | 0.000 | 1.00 |
| `replicate_all_hot_states` | 5.064 | 8120 | 0 | 0.000 | 0.007 | 0.000 | 1.00 |
| `request_parallel_gpu_like` | 5.064 | 7328 | 7328 | 0.000 | 0.014 | 0.000 | 1.00 |
| `single_pin_hot_state` | 5.064 | 8120 | 0 | 0.000 | 0.007 | 0.000 | 1.00 |
| `wafer_request_centric` | 5.064 | 7328 | 0 | 0.000 | 0.007 | 0.000 | 1.00 |

## 6. Ablations

`replicate_all_hot_states` gives latency 5.064 with memory pressure 0.007; `single_pin_hot_state` gives latency 5.064 and byte-hop 0.

## 7. Failure Cases

In this run `wafer_request_centric` is no worse than WaferStateFlow. This is a required negative case: when hot states are small enough for simpler cache-aware baselines, wafer placement is not necessary.

## 8. What This Means for the Paper

- H1: unsupported for this workflow.
- H2: supported for this workflow.
- H3: requires dynamic-hotness sweep; this single run is not enough.
- H4: supported only if WaferStateFlow beats request/operator-centric baselines outside cache-rich regimes.
- Next experiments: dynamic-hotness probability sweep, fanout sweep, and memory-capacity sensitivity.
