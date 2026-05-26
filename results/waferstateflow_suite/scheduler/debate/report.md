# WaferStateFlow Report

## 1. Executive Summary

Workflow `debate` redundancy ratio is 3.98. Best simulated scheduler is `replicate_all_hot_states` with latency 26.064. `WaferStateFlow` latency is 26.064.

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
| `flat_sequential` | approximate: sequential flat prompt materialization; no cache or parallelism |
| `request_parallel_gpu_like` | approximate: ready requests assigned to worker-local caches |
| `prefix_cache_like` | approximate: only exact root prefix-compatible states are globally reused |
| `helium_like_operator_schedule` | approximate: groups ready operators by shared prefix/state template |
| `kvflow_like_future_eviction` | approximate: future-use cache admission/eviction under capacity |
| `wafer_request_centric` | approximate: wafer backend without state-centric wave formation |
| `replicate_all_hot_states` | ablation: blindly replicates hot states |
| `single_pin_hot_state` | ablation: pins hot states centrally and exposes hotspots |
| `WaferStateFlow` | approximate: hot-state-seeded wave scheduling with policy-driven placement |

## 5. Results

| baseline | latency | materialization bytes | byte-hop | max link load | p95 link load | hotspot | max link util | memory pressure | crit wait | avg wave |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `WaferStateFlow` | 26.064 | 44052 | 48479232 | 1661952 | 1661952.0 | R_6_7 | 0.000 | 0.030 | 40.360 | 2.20 |
| `flat_sequential` | 84.538 | 175272 | 175272 | 0 | 0.0 | - | 0.000 | 0.000 | 0.000 | 1.00 |
| `helium_like_operator_schedule` | 26.064 | 44052 | 44052 | 0 | 0.0 | - | 0.000 | 0.000 | 0.000 | 2.20 |
| `kvflow_like_future_eviction` | 26.064 | 44052 | 44052 | 0 | 0.0 | - | 0.000 | 0.000 | 0.000 | 2.20 |
| `prefix_cache_like` | 26.064 | 55272 | 55272 | 0 | 0.0 | - | 0.000 | 0.000 | 0.000 | 2.20 |
| `replicate_all_hot_states` | 26.064 | 44052 | 48479232 | 1661952 | 1661952.0 | R_6_7 | 0.000 | 0.030 | 0.000 | 2.20 |
| `request_parallel_gpu_like` | 26.064 | 103272 | 103272 | 0 | 0.0 | - | 0.000 | 0.066 | 0.000 | 2.20 |
| `single_pin_hot_state` | 26.064 | 44052 | 626117632 | 22008832 | 22008832.0 | R_6_7 | 0.000 | 0.031 | 0.000 | 2.20 |
| `wafer_request_centric` | 26.064 | 103272 | 626117632 | 22008832 | 22008832.0 | R_6_7 | 0.000 | 0.031 | 0.000 | 2.20 |

## 6. Ablations

`replicate_all_hot_states` gives latency 26.064 with memory pressure 0.030; `single_pin_hot_state` gives latency 26.064 and byte-hop 626117632.

## 7. Failure Cases

In this run `replicate_all_hot_states` is no worse than WaferStateFlow. This is a required negative case: when hot states are small enough for simpler cache-aware baselines, wafer placement is not necessary.

## 8. What This Means for the Paper

- H1: supported for this workflow.
- H2: supported for this workflow.
- H3: requires dynamic-hotness sweep; this single run is not enough.
- H4: supported only if WaferStateFlow beats request/operator-centric baselines outside cache-rich regimes.
- Next experiments: dynamic-hotness probability sweep, fanout sweep, and memory-capacity sensitivity.
