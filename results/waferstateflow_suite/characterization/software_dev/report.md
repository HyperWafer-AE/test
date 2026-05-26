# WaferStateFlow Report

## 1. Executive Summary

Workflow `software_dev` has input redundancy ratio 4.18. H1 is supported under this synthetic setting. Top-state cumulative hotness share is 1.00; H2 is supported.

## 2. Problem Characterization

- Input redundancy ratio: 4.18
- Total prompt tokens: 36988
- Unique consumed state tokens: 8843
- Duplicate materialization bytes: 112580
- Dynamic hot states: 0

### Top Hot States

| rank | state | kind | consumers | hotness | cumulative share |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | `S_task` | task | 14 | 97760.0 | 0.46 |
| 2 | `S_repo_context` | repo_context | 14 | 97760.0 | 0.93 |
| 3 | `S_tool_schema` | tool_schema | 4 | 5760.0 | 0.96 |
| 4 | `S_test_plan` | test_plan | 4 | 4800.0 | 0.98 |
| 5 | `S_plan` | planner_output | 4 | 1807.2 | 0.99 |
| 6 | `S_coding_standards` | coding_standards | 2 | 1600.0 | 0.99 |
| 7 | `S_code_3` | output | 2 | 343.8 | 1.00 |
| 8 | `S_code_2` | output | 2 | 341.2 | 1.00 |
| 9 | `S_code_1` | output | 2 | 281.2 | 1.00 |
| 10 | `S_code_0` | output | 2 | 273.8 | 1.00 |

### State Fanout Table

| state | kind | tokens | consumers | token-weighted fanout |
| --- | --- | ---: | ---: | ---: |
| `S_task` | task | 800 | 14 | 10400 |
| `S_repo_context` | repo_context | 800 | 14 | 10400 |
| `S_tool_schema` | tool_schema | 800 | 4 | 2400 |
| `S_test_plan` | test_plan | 800 | 4 | 2400 |
| `S_coding_standards` | coding_standards | 800 | 2 | 800 |
| `S_plan` | planner_output | 251 | 4 | 753 |
| `S_code_3` | output | 275 | 2 | 275 |
| `S_code_2` | output | 273 | 2 | 273 |
| `S_code_1` | output | 225 | 2 | 225 |
| `S_code_0` | output | 219 | 2 | 219 |

## 3. Method

This run builds a State Access Graph and measures state-level fanout before any wafer claim is made.

## 7. Failure Cases

If the redundancy ratio is close to 1.0 or hotness share is flat, state-centric scheduling is not a strong paper claim for this workflow.

## 8. What This Means for the Paper

Use this characterization as a gate before scheduler results. Unsupported hypotheses should be reported rather than hidden.
