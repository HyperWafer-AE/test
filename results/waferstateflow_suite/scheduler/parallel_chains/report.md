# WaferStateFlow Report

## 1. Executive Summary

Workflow `parallel_chains` redundancy ratio is 4.72. Best simulated scheduler is `WaferStateFlow` with latency 16.367. `WaferStateFlow` latency is 16.367.

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
| `WaferStateFlow` | 16.367 | 33136 | 66095104 | 0.000 | 0.015 | 24.055 | 3.00 |
| `flat_sequential` | 63.018 | 125136 | 125136 | 0.000 | 0.000 | 0.000 | 1.00 |
| `helium_like_operator_schedule` | 18.220 | 26484 | 26484 | 0.000 | 0.000 | 0.000 | 3.00 |
| `kvflow_like_future_eviction` | 18.220 | 26484 | 26484 | 0.000 | 0.000 | 0.000 | 3.00 |
| `prefix_cache_like` | 20.358 | 65136 | 65136 | 0.000 | 0.000 | 0.000 | 3.00 |
| `replicate_all_hot_states` | 20.358 | 31348 | 52363264 | 0.000 | 0.017 | 0.000 | 3.00 |
| `request_parallel_gpu_like` | 20.358 | 62484 | 62484 | 0.000 | 0.035 | 0.000 | 3.00 |
| `single_pin_hot_state` | 20.358 | 31348 | 463407104 | 0.000 | 0.022 | 0.000 | 3.00 |
| `wafer_request_centric` | 20.358 | 62484 | 463407104 | 0.000 | 0.022 | 0.000 | 3.00 |

## 6. Ablations

`replicate_all_hot_states` gives latency 20.358 with memory pressure 0.017; `single_pin_hot_state` gives latency 20.358 and byte-hop 463407104.

## 7. Failure Cases

No baseline beat WaferStateFlow in this run, but this does not prove the platform claim. Run low-fanout or memory-rich sweeps to find negative cases.

## 8. What This Means for the Paper

- H1: supported for this workflow.
- H2: supported for this workflow.
- H3: requires dynamic-hotness sweep; this single run is not enough.
- H4: supported only if WaferStateFlow beats request/operator-centric baselines outside cache-rich regimes.
- Next experiments: dynamic-hotness probability sweep, fanout sweep, and memory-capacity sensitivity.
