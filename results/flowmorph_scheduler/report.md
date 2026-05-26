# FlowMorph-v1 Frontier Scheduler Prototype

## Executive Summary

This experiment uses an abstract worker-resource model over PhaseDAG inputs. It does not implement wafer-specific placement and makes no wafer performance claims.

Selected frontier-positive workflows: 6. Negative controls: 1.

## Workflow Selection

| workflow | taxonomy | selected | reason |
| --- | --- | --- | --- |
| `debate` | frontier_and_phase | yes | frontier_positive |
| `iterative` | weak | yes | negative_control |
| `mapreduce` | frontier_only | yes | frontier_positive |
| `parallel_chains` | frontier_only | yes | frontier_positive |
| `reflection` | frontier_only | yes | frontier_positive |
| `software_dev` | frontier_only | yes | frontier_positive |
| `trading` | frontier_only | yes | frontier_positive |

## Scheduler Metrics

| workflow | scheduler | policy | taxonomy | latency | idle | critical path delay | mode switches | wide utilization | narrow latency |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `debate` | always_consolidated | always_consolidated | frontier_and_phase | 31.273 | 0.00 | 0.000 | 0 | 1.00 | 9.709 |
| `debate` | always_parallel | always_parallel | frontier_and_phase | 31.850 | 0.61 | 0.000 | 0 | 0.48 | 8.790 |
| `debate` | fixed_worker_pool | fixed_worker_pool | frontier_and_phase | 31.850 | 0.61 | 0.000 | 0 | 0.48 | 8.790 |
| `debate` | frontier_aware_morphing | frontier_aware_morphing | frontier_and_phase | 25.861 | 0.46 | 0.000 | 3 | 0.48 | 2.801 |
| `debate` | static_full_resource | static_full_resource | frontier_and_phase | 31.850 | 0.61 | 0.000 | 0 | 0.48 | 8.790 |
| `debate` | static_split_resource | static_split_resource | frontier_and_phase | 26.697 | 0.41 | 0.000 | 0 | 0.60 | 4.101 |
| `iterative` | always_consolidated | always_consolidated | weak | 29.215 | 0.00 | 0.000 | 0 | 0.00 | 29.215 |
| `iterative` | always_parallel | always_parallel | weak | 91.688 | 0.88 | 0.000 | 0 | 0.00 | 91.688 |
| `iterative` | fixed_worker_pool | fixed_worker_pool | weak | 91.688 | 0.88 | 0.000 | 0 | 0.00 | 91.688 |
| `iterative` | frontier_aware_morphing | fallback_fixed_worker_pool | weak | 91.688 | 0.88 | 0.000 | 0 | 0.00 | 91.688 |
| `iterative` | static_full_resource | static_full_resource | weak | 91.688 | 0.88 | 0.000 | 0 | 0.00 | 91.688 |
| `iterative` | static_split_resource | static_split_resource | weak | 42.774 | 0.50 | 0.000 | 0 | 0.00 | 42.774 |
| `mapreduce` | always_consolidated | always_consolidated | frontier_only | 12.778 | 0.00 | 0.000 | 0 | 1.00 | 7.467 |
| `mapreduce` | always_parallel | always_parallel | frontier_only | 15.878 | 0.68 | 0.000 | 0 | 0.48 | 7.376 |
| `mapreduce` | fixed_worker_pool | fixed_worker_pool | frontier_only | 15.878 | 0.68 | 0.000 | 0 | 0.48 | 7.376 |
| `mapreduce` | frontier_aware_morphing | frontier_aware_morphing | frontier_only | 10.852 | 0.41 | 0.000 | 1 | 0.48 | 2.350 |
| `mapreduce` | static_full_resource | static_full_resource | frontier_only | 15.878 | 0.68 | 0.000 | 0 | 0.48 | 7.376 |
| `mapreduce` | static_split_resource | static_split_resource | frontier_only | 11.607 | 0.42 | 0.000 | 0 | 0.61 | 3.441 |
| `parallel_chains` | always_consolidated | always_consolidated | frontier_only | 24.141 | 0.00 | 0.000 | 0 | 1.00 | 7.698 |
| `parallel_chains` | always_parallel | always_parallel | frontier_only | 24.798 | 0.62 | 0.000 | 0 | 0.48 | 16.660 |
| `parallel_chains` | fixed_worker_pool | fixed_worker_pool | frontier_only | 24.798 | 0.62 | 0.000 | 0 | 0.48 | 16.660 |
| `parallel_chains` | frontier_aware_morphing | frontier_aware_morphing | frontier_only | 19.755 | 0.45 | 0.000 | 2 | 0.48 | 11.617 |
| `parallel_chains` | static_full_resource | static_full_resource | frontier_only | 24.798 | 0.62 | 0.000 | 0 | 0.48 | 16.660 |
| `parallel_chains` | static_split_resource | static_split_resource | frontier_only | 20.633 | 0.37 | 0.000 | 2 | 0.61 | 16.611 |
| `reflection` | always_consolidated | always_consolidated | frontier_only | 11.208 | 0.00 | 0.000 | 0 | 1.00 | 6.811 |
| `reflection` | always_parallel | always_parallel | frontier_only | 21.938 | 0.80 | 0.000 | 0 | 0.36 | 14.860 |
| `reflection` | fixed_worker_pool | fixed_worker_pool | frontier_only | 21.938 | 0.80 | 0.000 | 0 | 0.36 | 14.860 |
| `reflection` | frontier_aware_morphing | frontier_aware_morphing | frontier_only | 11.813 | 0.38 | 0.000 | 2 | 0.36 | 4.735 |
| `reflection` | static_full_resource | static_full_resource | frontier_only | 21.938 | 0.80 | 0.000 | 0 | 0.36 | 14.860 |
| `reflection` | static_split_resource | static_split_resource | frontier_only | 13.654 | 0.50 | 0.000 | 0 | 0.49 | 6.932 |
| `software_dev` | always_consolidated | always_consolidated | frontier_only | 30.198 | 0.00 | 0.000 | 0 | 1.00 | 8.582 |
| `software_dev` | always_parallel | always_parallel | frontier_only | 35.220 | 0.66 | 0.000 | 0 | 0.48 | 27.906 |
| `software_dev` | fixed_worker_pool | fixed_worker_pool | frontier_only | 35.220 | 0.66 | 0.000 | 0 | 0.48 | 27.906 |
| `software_dev` | frontier_aware_morphing | frontier_aware_morphing | frontier_only | 25.263 | 0.42 | 0.000 | 3 | 0.48 | 17.949 |
| `software_dev` | static_full_resource | static_full_resource | frontier_only | 35.220 | 0.66 | 0.000 | 0 | 0.48 | 27.906 |
| `software_dev` | static_split_resource | static_split_resource | frontier_only | 27.088 | 0.40 | 0.000 | 4 | 0.58 | 23.512 |
| `trading` | always_consolidated | always_consolidated | frontier_only | 71.962 | 0.00 | 0.000 | 0 | 1.00 | 10.586 |
| `trading` | always_parallel | always_parallel | frontier_only | 46.868 | 0.40 | 0.000 | 0 | 0.75 | 33.222 |
| `trading` | fixed_worker_pool | fixed_worker_pool | frontier_only | 46.868 | 0.40 | 0.000 | 0 | 0.75 | 33.222 |
| `trading` | frontier_aware_morphing | frontier_aware_morphing | frontier_only | 37.326 | 0.17 | 0.000 | 2 | 0.75 | 23.680 |
| `trading` | static_full_resource | static_full_resource | frontier_only | 46.868 | 0.40 | 0.000 | 0 | 0.75 | 33.222 |
| `trading` | static_split_resource | static_split_resource | frontier_only | 48.778 | 0.23 | 0.000 | 12 | 0.86 | 22.038 |

## Interpretation Rules

- `frontier_aware_morphing` uses parallel mode on wide frontiers.
- It uses consolidated fast-lane mode for narrow ready sets with critical operators.
- Weak-frontier workflows fall back to `fixed_worker_pool`; `iterative` is kept as a negative control.
- Baselines include `fixed_worker_pool`, `static_full_resource`, `static_split_resource`, `always_parallel`, and `always_consolidated`.
- These results are scheduler-prototype evidence only, not wafer placement or wafer speedup evidence.
