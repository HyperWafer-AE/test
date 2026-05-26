# WaferStateFlow Report

## 1. Executive Summary

Workflow `reflection` has input redundancy ratio 2.25. H1 is supported under this synthetic setting. Top-state cumulative hotness share is 1.00; H2 is supported.

## 2. Problem Characterization

- Input redundancy ratio: 2.25
- Total prompt tokens: 13936
- Unique consumed state tokens: 6183
- Duplicate materialization bytes: 31012
- Dynamic hot states: 0

### Top Hot States

| rank | state | kind | consumers | hotness | cumulative share |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | `S_task` | task | 5 | 13200.0 | 0.66 |
| 2 | `S_rubric` | rubric | 3 | 3000.0 | 0.81 |
| 3 | `S_draft` | intermediate_summary | 4 | 1882.5 | 0.91 |
| 4 | `S_document` | document | 2 | 1800.0 | 1.00 |
| 5 | `S_style_guide` | style_guide | 1 | 0.0 | 1.00 |
| 6 | `S_critic_lens_2` | unique_context | 1 | 0.0 | 1.00 |
| 7 | `S_critic_lens_1` | unique_context | 1 | 0.0 | 1.00 |
| 8 | `S_critic_lens_0` | unique_context | 1 | 0.0 | 1.00 |
| 9 | `S_critic_2_feedback` | critic_feedback | 1 | 0.0 | 1.00 |
| 10 | `S_critic_1_feedback` | critic_feedback | 1 | 0.0 | 1.00 |

### State Fanout Table

| state | kind | tokens | consumers | token-weighted fanout |
| --- | --- | ---: | ---: | ---: |
| `S_task` | task | 1000 | 5 | 4000 |
| `S_rubric` | rubric | 1000 | 3 | 2000 |
| `S_document` | document | 1000 | 2 | 1000 |
| `S_draft` | intermediate_summary | 251 | 4 | 753 |
| `S_style_guide` | style_guide | 1000 | 1 | 0 |
| `S_critic_lens_2` | unique_context | 451 | 1 | 0 |
| `S_critic_lens_1` | unique_context | 396 | 1 | 0 |
| `S_critic_lens_0` | unique_context | 349 | 1 | 0 |
| `S_critic_2_feedback` | critic_feedback | 279 | 1 | 0 |
| `S_critic_1_feedback` | critic_feedback | 238 | 1 | 0 |

## 3. Method

This run builds a State Access Graph and measures state-level fanout before any wafer claim is made.

## 7. Failure Cases

If the redundancy ratio is close to 1.0 or hotness share is flat, state-centric scheduling is not a strong paper claim for this workflow.

## 8. What This Means for the Paper

Use this characterization as a gate before scheduler results. Unsupported hypotheses should be reported rather than hidden.
