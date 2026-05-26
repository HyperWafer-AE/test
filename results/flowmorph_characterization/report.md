# FlowMorph Problem Characterization

## Executive Summary

FlowMorph measures frontier-aware phase/resource irregularity in agent workflows using a PhaseDAG. This report does not make WaferStateFlow claims and does not implement wafer scheduling.

FlowMorph-v1 gate: **continue FlowMorph-v1** (6 workflows are frontier_only or frontier_and_phase).
FlowMorph-v2 gate: **do not continue FlowMorph-v2 from this synthetic evidence alone** (1 workflows are frontier_and_phase).

## Workflow Metrics

| workflow | taxonomy | frontier_morphing_opportunity | phase_morphing_opportunity | combined_opportunity | frontier CV | max frontier | median frontier | width drop | wide work | narrow critical | serial frac | parallel slack | phase variation | idle | partition imbalance |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `debate` | frontier_and_phase | yes | yes | yes | 0.67 | 4 | 1.00 | 4.00 | 0.91 | 0.28 | 0.32 | 3.08 | 0.63 | 0.61 | 0.71 |
| `iterative` | weak | no | no | no | 0.00 | 1 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.03 | 0.88 | 0.75 |
| `mapreduce` | frontier_only | yes | no | no | 0.60 | 4 | 2.50 | 1.60 | 0.82 | 0.46 | 0.40 | 2.53 | 0.02 | 0.68 | 0.65 |
| `parallel_chains` | frontier_only | yes | no | no | 0.75 | 4 | 1.00 | 4.00 | 0.44 | 0.65 | 0.33 | 3.06 | 0.02 | 0.62 | 0.65 |
| `reflection` | frontier_only | yes | no | no | 0.57 | 3 | 1.00 | 3.00 | 0.58 | 0.68 | 0.62 | 1.60 | 0.01 | 0.80 | 0.58 |
| `software_dev` | frontier_only | yes | no | no | 0.68 | 4 | 1.00 | 4.00 | 0.32 | 0.78 | 0.37 | 2.69 | 0.04 | 0.66 | 0.56 |
| `trading` | frontier_only | yes | no | no | 1.87 | 16 | 1.00 | 16.00 | 0.50 | 0.79 | 0.15 | 6.65 | 0.05 | 0.31 | 0.58 |

## Decision Gate

- frontier_only: frontier width varies strongly or parallel slack is high, while phase mix is stable.
- phase_only: frontier variation is low, while phase mix varies strongly.
- frontier_and_phase: both frontier opportunity and phase opportunity are present.
- weak: neither frontier nor phase opportunity crosses the thresholds.
- Continue FlowMorph-v1 only if multiple workflows are frontier_only or frontier_and_phase.
- Continue FlowMorph-v2 only if multiple workflows are frontier_and_phase.
- This run is characterization only; no wafer scheduling or placement is implemented.
- Workflow generators were not tuned to force success; negative results remain in the table.

## Interpretation

At least one workflow does not support frontier-aware morphing. Treat FlowMorph as conditional, not as an assumed win across all agent workflows.
