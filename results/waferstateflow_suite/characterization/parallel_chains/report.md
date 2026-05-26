# WaferStateFlow Report

## 1. Executive Summary

Workflow `parallel_chains` has input redundancy ratio 4.72. H1 is supported under this synthetic setting. Top-state cumulative hotness share is 1.00; H2 is supported.

## 2. Problem Characterization

- Input redundancy ratio: 4.72
- Total prompt tokens: 31284
- Unique consumed state tokens: 6621
- Duplicate materialization bytes: 98652
- Dynamic hot states: 0

### Top Hot States

| rank | state | kind | consumers | hotness | cumulative share |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | `S_task` | task | 9 | 42400.0 | 0.36 |
| 2 | `S_rubric` | rubric | 9 | 42400.0 | 0.73 |
| 3 | `S_document` | document | 8 | 30100.0 | 0.98 |
| 4 | `S_chain_2_context` | unique_context | 2 | 480.5 | 0.99 |
| 5 | `S_chain_0_context` | unique_context | 2 | 478.4 | 0.99 |
| 6 | `S_chain_1_context` | unique_context | 2 | 425.7 | 1.00 |
| 7 | `S_chain_3_context` | unique_context | 2 | 403.1 | 1.00 |
| 8 | `S_tool_schema` | tool_schema | 0 | 0.0 | 1.00 |
| 9 | `S_chain_3_0_out` | intermediate_summary | 1 | 0.0 | 1.00 |
| 10 | `S_chain_1_1_out` | intermediate_summary | 1 | 0.0 | 1.00 |

### State Fanout Table

| state | kind | tokens | consumers | token-weighted fanout |
| --- | --- | ---: | ---: | ---: |
| `S_task` | task | 1000 | 9 | 8000 |
| `S_rubric` | rubric | 1000 | 9 | 8000 |
| `S_document` | document | 1000 | 8 | 7000 |
| `S_chain_2_context` | unique_context | 447 | 2 | 447 |
| `S_chain_0_context` | unique_context | 445 | 2 | 445 |
| `S_chain_1_context` | unique_context | 396 | 2 | 396 |
| `S_chain_3_context` | unique_context | 375 | 2 | 375 |
| `S_tool_schema` | tool_schema | 1000 | 0 | 0 |
| `S_chain_3_0_out` | intermediate_summary | 304 | 1 | 0 |
| `S_chain_1_1_out` | intermediate_summary | 282 | 1 | 0 |

## 3. Method

This run builds a State Access Graph and measures state-level fanout before any wafer claim is made.

## 7. Failure Cases

If the redundancy ratio is close to 1.0 or hotness share is flat, state-centric scheduling is not a strong paper claim for this workflow.

## 8. What This Means for the Paper

Use this characterization as a gate before scheduler results. Unsupported hypotheses should be reported rather than hidden.
