# WaferStateFlow Report

## 1. Executive Summary

Workflow `trading` redundancy ratio is 5.93. Best simulated scheduler is `helium_like_operator_schedule` with latency 33.986. `WaferStateFlow` latency is 33.986.

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
| `WaferStateFlow` | 33.986 | 80808 | 164954112 | 6871040 | 6871040.0 | R_8_7 | 0.000 | 0.034 | 271.280 | 6.60 |
| `flat_sequential` | 179.654 | 356008 | 356008 | 0 | 0.0 | - | 0.000 | 0.000 | 0.000 | 1.00 |
| `helium_like_operator_schedule` | 33.986 | 60036 | 60036 | 0 | 0.0 | - | 0.000 | 0.000 | 0.000 | 6.60 |
| `kvflow_like_future_eviction` | 33.986 | 60036 | 60036 | 0 | 0.0 | - | 0.000 | 0.000 | 0.000 | 6.60 |
| `prefix_cache_like` | 33.986 | 80808 | 80808 | 0 | 0.0 | - | 0.000 | 0.000 | 0.000 | 6.60 |
| `replicate_all_hot_states` | 33.986 | 80808 | 164954112 | 6871040 | 6871040.0 | R_8_7 | 0.000 | 0.034 | 0.000 | 6.60 |
| `request_parallel_gpu_like` | 33.986 | 178752 | 178752 | 0 | 0.0 | - | 0.000 | 0.067 | 0.000 | 6.60 |
| `single_pin_hot_state` | 33.986 | 80808 | 967770112 | 36362240 | 36362240.0 | R_8_7 | 0.000 | 0.050 | 0.000 | 6.60 |
| `wafer_request_centric` | 33.986 | 178752 | 967770112 | 36362240 | 36362240.0 | R_8_7 | 0.000 | 0.050 | 0.000 | 6.60 |

## 6. Ablations

`replicate_all_hot_states` gives latency 33.986 with memory pressure 0.034; `single_pin_hot_state` gives latency 33.986 and byte-hop 967770112.

## 7. Failure Cases

In this run `helium_like_operator_schedule` is no worse than WaferStateFlow. This is a required negative case: when hot states are small enough for simpler cache-aware baselines, wafer placement is not necessary.

## 8. What This Means for the Paper

- H1: supported for this workflow.
- H2: supported for this workflow.
- H3: requires dynamic-hotness sweep; this single run is not enough.
- H4: supported only if WaferStateFlow beats request/operator-centric baselines outside cache-rich regimes.
- Next experiments: dynamic-hotness probability sweep, fanout sweep, and memory-capacity sensitivity.
