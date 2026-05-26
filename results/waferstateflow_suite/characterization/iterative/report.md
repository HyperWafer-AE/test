# WaferStateFlow Report

## 1. Executive Summary

Workflow `iterative` has input redundancy ratio 3.32. H1 is supported under this synthetic setting. Top-state cumulative hotness share is 1.00; H2 is supported.

## 2. Problem Characterization

- Input redundancy ratio: 3.32
- Total prompt tokens: 40072
- Unique consumed state tokens: 12072
- Duplicate materialization bytes: 112000
- Dynamic hot states: 0

### Top Hot States

| rank | state | kind | consumers | hotness | cumulative share |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | `S_task` | task | 8 | 50425.2 | 0.33 |
| 2 | `S_rubric` | rubric | 8 | 50387.4 | 0.67 |
| 3 | `S_global_context` | global_context | 8 | 50387.4 | 1.00 |
| 4 | `S_chunk_0` | document | 1 | 0.0 | 1.00 |
| 5 | `S_chunk_6` | document | 1 | 0.0 | 1.00 |
| 6 | `S_chunk_2` | document | 1 | 0.0 | 1.00 |
| 7 | `S_chunk_5` | document | 1 | 0.0 | 1.00 |
| 8 | `S_chunk_1` | document | 1 | 0.0 | 1.00 |
| 9 | `S_chunk_7` | document | 1 | 0.0 | 1.00 |
| 10 | `S_chunk_4` | document | 1 | 0.0 | 1.00 |

### State Fanout Table

| state | kind | tokens | consumers | token-weighted fanout |
| --- | --- | ---: | ---: | ---: |
| `S_task` | task | 1334 | 8 | 9338 |
| `S_rubric` | rubric | 1333 | 8 | 9331 |
| `S_global_context` | global_context | 1333 | 8 | 9331 |
| `S_chunk_0` | document | 890 | 1 | 0 |
| `S_chunk_6` | document | 876 | 1 | 0 |
| `S_chunk_2` | document | 817 | 1 | 0 |
| `S_chunk_5` | document | 799 | 1 | 0 |
| `S_chunk_1` | document | 793 | 1 | 0 |
| `S_chunk_7` | document | 774 | 1 | 0 |
| `S_chunk_4` | document | 767 | 1 | 0 |

## 3. Method

This run builds a State Access Graph and measures state-level fanout before any wafer claim is made.

## 7. Failure Cases

If the redundancy ratio is close to 1.0 or hotness share is flat, state-centric scheduling is not a strong paper claim for this workflow.

## 8. What This Means for the Paper

Use this characterization as a gate before scheduler results. Unsupported hypotheses should be reported rather than hidden.
