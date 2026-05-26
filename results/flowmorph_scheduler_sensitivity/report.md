# FlowMorph-v1 Scheduler Sensitivity

## Executive Summary

This is an abstract scheduler robustness audit before any wafer mapping. It does not add wafer placement and makes no wafer performance claims.

Decision gate: **continue FlowMorph-v1 robustness work**.
Robust frontier-positive workflows under nonzero switch overhead: 4/6.

## Sweep

- consolidated_speedup_exponent: [0.2, 0.35, 0.5, 0.65, 0.8]
- mode_switch_overhead: [0.0, 0.5, 1.0, 2.0, 5.0]
- worker_count: [4, 8, 16]
- criticality_threshold: [0.8, 1.5, 2.0]
- Workflow generators were not tuned to force success.
- `iterative` is kept as a negative control.

## Regret By Workflow

| workflow | status | cases | mean regret | p95 regret | max regret | nonzero-overhead robust | flowmorph wins | best static wins |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `debate` | frontier_positive | 225 | 0.271 | 1.430 | 2.653 | no | 84 | 141 |
| `iterative` | weak | 225 | 2.308 | 8.190 | 8.190 | no | 0 | 225 |
| `mapreduce` | frontier_positive | 225 | 0.177 | 1.242 | 2.278 | yes | 113 | 112 |
| `parallel_chains` | frontier_positive | 225 | 0.205 | 1.263 | 2.343 | yes | 110 | 115 |
| `reflection` | frontier_positive | 225 | 0.436 | 1.947 | 3.884 | no | 70 | 155 |
| `software_dev` | frontier_positive | 225 | 0.193 | 1.327 | 2.492 | yes | 120 | 105 |
| `trading` | frontier_positive | 225 | -0.104 | 0.063 | 0.253 | yes | 188 | 37 |

## Winner Counts

| frontier status | winner | count |
| --- | --- | ---: |
| frontier_positive | always_consolidated | 453 |
| frontier_positive | fixed_worker_pool | 196 |
| frontier_positive | frontier_aware_morphing | 685 |
| frontier_positive | static_split_resource | 16 |
| weak | always_consolidated | 225 |

## Interpretation Rules

- `best_static_oracle` is the minimum latency among `fixed_worker_pool`, `always_parallel`, `always_consolidated`, and `static_split_resource`.
- `regret = FlowMorph latency / best_static_oracle latency - 1`.
- FlowMorph is considered robust for a frontier-positive workflow only when most nonzero-overhead cases have regret within the configured tolerance.
- If `always_consolidated` or `static_split_resource` dominates, the report treats FlowMorph-v1 as weak.
- Negative results are retained; no workflow generators were tuned.
