# WaferStateFlow Report

## 1. Executive Summary

Workflow `debate` has input redundancy ratio 3.98. H1 is supported under this synthetic setting. Top-state cumulative hotness share is 1.00; H2 is supported.

## 2. Problem Characterization

- Input redundancy ratio: 3.98
- Total prompt tokens: 43818
- Unique consumed state tokens: 11013
- Duplicate materialization bytes: 131220
- Dynamic hot states: 0

### Top Hot States

| rank | state | kind | consumers | hotness | cumulative share |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | `S_task` | task | 9 | 40800.0 | 0.28 |
| 2 | `S_rubric` | rubric | 9 | 40800.0 | 0.57 |
| 3 | `S_document` | document | 8 | 28000.0 | 0.77 |
| 4 | `S_role_instruction` | role_instruction | 8 | 28000.0 | 0.96 |
| 5 | `S_transcript_r0` | intermediate_summary | 4 | 5610.0 | 1.00 |
| 6 | `S_transcript_r1` | intermediate_summary | 1 | 0.0 | 1.00 |
| 7 | `S_round_0_stance_0` | unique_context | 1 | 0.0 | 1.00 |
| 8 | `S_round_1_stance_2` | unique_context | 1 | 0.0 | 1.00 |
| 9 | `S_round_0_stance_2` | unique_context | 1 | 0.0 | 1.00 |
| 10 | `S_round_1_stance_1` | unique_context | 1 | 0.0 | 1.00 |

### State Fanout Table

| state | kind | tokens | consumers | token-weighted fanout |
| --- | --- | ---: | ---: | ---: |
| `S_task` | task | 1000 | 9 | 8000 |
| `S_rubric` | rubric | 1000 | 9 | 8000 |
| `S_document` | document | 1000 | 8 | 7000 |
| `S_role_instruction` | role_instruction | 1000 | 8 | 7000 |
| `S_transcript_r0` | intermediate_summary | 935 | 4 | 2805 |
| `S_transcript_r1` | intermediate_summary | 989 | 1 | 0 |
| `S_round_0_stance_0` | unique_context | 445 | 1 | 0 |
| `S_round_1_stance_2` | unique_context | 438 | 1 | 0 |
| `S_round_0_stance_2` | unique_context | 408 | 1 | 0 |
| `S_round_1_stance_1` | unique_context | 399 | 1 | 0 |

## 3. Method

This run builds a State Access Graph and measures state-level fanout before any wafer claim is made.

## 7. Failure Cases

If the redundancy ratio is close to 1.0 or hotness share is flat, state-centric scheduling is not a strong paper claim for this workflow.

## 8. What This Means for the Paper

Use this characterization as a gate before scheduler results. Unsupported hypotheses should be reported rather than hidden.
