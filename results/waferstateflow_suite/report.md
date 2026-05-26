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
| `debate` | `WaferStateFlow` | 20.566 | 20.566 |
| `iterative` | `wafer_request_centric` | 80.544 | 80.544 |
| `mapreduce` | `WaferStateFlow` | 10.931 | 10.931 |
| `parallel_chains` | `WaferStateFlow` | 16.367 | 16.367 |
| `reflection` | `WaferStateFlow` | 16.015 | 16.015 |
| `software_dev` | `WaferStateFlow` | 23.389 | 23.389 |
| `trading` | `WaferStateFlow` | 28.073 | 28.073 |

## Failure Cases

`iterative` has `wafer_request_centric` no worse than WaferStateFlow in this suite, so the prototype preserves counterexamples and ties instead of forcing WaferStateFlow to win every setting.

## What This Means for the Paper

The suite is useful for screening hypotheses across workflow shapes. It remains synthetic and should be followed by real trace replay.
