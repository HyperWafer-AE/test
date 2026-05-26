# WaferStateFlow Workflow Suite

## Executive Summary

Characterized 7 workflows and collected 63 scheduler rows.

## Characterization

| workflow | redundancy ratio | duplicate bytes | H1 | H2 |
| --- | ---: | ---: | --- | --- |
| `debate` | 3.98 | 131220 | True | True |
| `iterative` | 3.32 | 112000 | True | True |
| `mapreduce` | 3.00 | 44000 | True | True |
| `parallel_chains` | 4.72 | 98652 | True | True |
| `reflection` | 2.25 | 31012 | True | True |
| `software_dev` | 4.18 | 112580 | True | True |
| `trading` | 5.93 | 295972 | True | True |

## Scheduler Winners

| workflow | winner | latency | WaferStateFlow latency |
| --- | --- | ---: | ---: |
| `debate` | `replicate_all_hot_states` | 26.064 | 26.064 |
| `iterative` | `wafer_request_centric` | 80.544 | 80.544 |
| `mapreduce` | `prefix_cache_like` | 12.860 | 12.860 |
| `parallel_chains` | `helium_like_operator_schedule` | 20.358 | 20.358 |
| `reflection` | `WaferStateFlow` | 17.528 | 17.528 |
| `software_dev` | `WaferStateFlow` | 27.854 | 27.854 |
| `trading` | `helium_like_operator_schedule` | 33.986 | 33.986 |

## Failure Cases

`debate` has `replicate_all_hot_states` no worse than WaferStateFlow in this suite, so the prototype preserves counterexamples and ties instead of forcing WaferStateFlow to win every setting.

## What This Means for the Paper

The suite is useful for screening hypotheses across workflow shapes. It remains synthetic and should be followed by real trace replay.
