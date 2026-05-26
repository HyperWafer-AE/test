# WaferStateFlow Report

## 1. Executive Summary

Workflow `parallel_chains` redundancy ratio is 4.72. Best simulated scheduler is `helium_like_operator_schedule` with latency 20.358. `WaferStateFlow` latency is 20.358.

## 2. Problem Characterization

- Input redundancy ratio: 4.72
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
| `WaferStateFlow` | 20.358 | 33136 | 66095104 | 2230272 | 2230272.0 | R_9_7 | 0.000 | 0.015 | 31.776 | 3.00 |
| `flat_sequential` | 63.018 | 125136 | 125136 | 0 | 0.0 | - | 0.000 | 0.000 | 0.000 | 1.00 |
| `helium_like_operator_schedule` | 20.358 | 26484 | 26484 | 0 | 0.0 | - | 0.000 | 0.000 | 0.000 | 3.00 |
| `kvflow_like_future_eviction` | 20.358 | 26484 | 26484 | 0 | 0.0 | - | 0.000 | 0.000 | 0.000 | 3.00 |
| `prefix_cache_like` | 20.358 | 33136 | 33136 | 0 | 0.0 | - | 0.000 | 0.000 | 0.000 | 3.00 |
| `replicate_all_hot_states` | 20.358 | 31348 | 52363264 | 2179072 | 2179072.0 | R_6_7 | 0.000 | 0.017 | 0.000 | 3.00 |
| `request_parallel_gpu_like` | 20.358 | 62484 | 62484 | 0 | 0.0 | - | 0.000 | 0.035 | 0.000 | 3.00 |
| `single_pin_hot_state` | 20.358 | 31348 | 463407104 | 16515072 | 16515072.0 | R_6_7 | 0.000 | 0.022 | 0.000 | 3.00 |
| `wafer_request_centric` | 20.358 | 62484 | 463407104 | 16515072 | 16515072.0 | R_6_7 | 0.000 | 0.022 | 0.000 | 3.00 |

## 6. Ablations

`replicate_all_hot_states` gives latency 20.358 with memory pressure 0.017; `single_pin_hot_state` gives latency 20.358 and byte-hop 463407104.

## 7. Failure Cases

In this run `helium_like_operator_schedule` is no worse than WaferStateFlow. This is a required negative case: when hot states are small enough for simpler cache-aware baselines, wafer placement is not necessary.

## 8. What This Means for the Paper

- H1: supported for this workflow.
- H2: supported for this workflow.
- H3: requires dynamic-hotness sweep; this single run is not enough.
- H4: supported only if WaferStateFlow beats request/operator-centric baselines outside cache-rich regimes.
- Next experiments: dynamic-hotness probability sweep, fanout sweep, and memory-capacity sensitivity.
