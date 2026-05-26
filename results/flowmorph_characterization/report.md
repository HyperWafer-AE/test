# FlowMorph Problem Characterization

## Executive Summary

FlowMorph measures phase/resource irregularity in agent workflows using a PhaseDAG. This report does not make WaferStateFlow claims and does not implement wafer scheduling.

Overall gate: **continue to FlowMorph scheduling for workflows with strong irregularity**.

## Workflow Metrics

| workflow | max frontier | frontier CV | phase entropy | phase variation | critical path | work/CP | idle | partition imbalance | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `debate` | 4 | 0.67 | 0.37 | 0.63 | 31.850 | 3.08 | 0.61 | 0.71 | continue_to_flowmorph_scheduling |
| `iterative` | 1 | 0.00 | 0.54 | 0.03 | 91.688 | 1.00 | 0.88 | 0.75 | weak_direction |
| `mapreduce` | 4 | 0.60 | 0.70 | 0.02 | 15.878 | 2.53 | 0.68 | 0.65 | weak_direction |
| `parallel_chains` | 4 | 0.75 | 0.66 | 0.02 | 24.798 | 3.06 | 0.62 | 0.65 | weak_direction |
| `reflection` | 3 | 0.57 | 0.73 | 0.01 | 21.938 | 1.60 | 0.80 | 0.58 | weak_direction |
| `software_dev` | 4 | 0.68 | 0.77 | 0.04 | 35.220 | 2.69 | 0.66 | 0.56 | weak_direction |
| `trading` | 16 | 1.87 | 0.77 | 0.05 | 33.946 | 6.65 | 0.31 | 0.58 | weak_direction |

## Decision Gate

- If frontier width is mostly constant and phase mix is stable, this direction is weak.
- If frontier width and phase mix both vary strongly, continue to FlowMorph scheduling.
- This run is characterization only; no wafer scheduling or placement is implemented.

## Interpretation

At least one workflow fails the irregularity gate. Treat FlowMorph scheduling as conditional, not as an assumed win across all agent workflows.
