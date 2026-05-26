# WaferStateFlow Report

## 1. Executive Summary

Workflow `mapreduce` has input redundancy ratio 3.00. H1 is supported under this synthetic setting. Top-state cumulative hotness share is 1.00; H2 is supported.

## 2. Problem Characterization

- Input redundancy ratio: 3.00
- Total prompt tokens: 16493
- Unique consumed state tokens: 5493
- Duplicate materialization bytes: 44000
- Dynamic hot states: 0

### Top Hot States

| rank | state | kind | consumers | hotness | cumulative share |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | `S_task` | task | 5 | 12000.0 | 0.40 |
| 2 | `S_rubric` | rubric | 5 | 12000.0 | 0.80 |
| 3 | `S_document` | document | 4 | 6000.0 | 1.00 |
| 4 | `S_tool_schema` | tool_schema | 0 | 0.0 | 1.00 |
| 5 | `S_unique_expert_0` | unique_context | 1 | 0.0 | 1.00 |
| 6 | `S_unique_expert_2` | unique_context | 1 | 0.0 | 1.00 |
| 7 | `S_unique_expert_1` | unique_context | 1 | 0.0 | 1.00 |
| 8 | `S_unique_expert_3` | unique_context | 1 | 0.0 | 1.00 |
| 9 | `S_expert_1_summary` | intermediate_summary | 1 | 0.0 | 1.00 |
| 10 | `S_final_summary` | output | 0 | 0.0 | 1.00 |

### State Fanout Table

| state | kind | tokens | consumers | token-weighted fanout |
| --- | --- | ---: | ---: | ---: |
| `S_task` | task | 1000 | 5 | 4000 |
| `S_rubric` | rubric | 1000 | 5 | 4000 |
| `S_document` | document | 1000 | 4 | 3000 |
| `S_tool_schema` | tool_schema | 1000 | 0 | 0 |
| `S_unique_expert_0` | unique_context | 445 | 1 | 0 |
| `S_unique_expert_2` | unique_context | 408 | 1 | 0 |
| `S_unique_expert_1` | unique_context | 396 | 1 | 0 |
| `S_unique_expert_3` | unique_context | 309 | 1 | 0 |
| `S_expert_1_summary` | intermediate_summary | 285 | 1 | 0 |
| `S_final_summary` | output | 251 | 0 | 0 |

## 3. Method

This run builds a State Access Graph and measures state-level fanout before any wafer claim is made.

## 7. Failure Cases

If the redundancy ratio is close to 1.0 or hotness share is flat, state-centric scheduling is not a strong paper claim for this workflow.

## 8. What This Means for the Paper

Use this characterization as a gate before scheduler results. Unsupported hypotheses should be reported rather than hidden.
