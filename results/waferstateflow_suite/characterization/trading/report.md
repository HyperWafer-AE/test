# WaferStateFlow Report

## 1. Executive Summary

Workflow `trading` has input redundancy ratio 5.93. H1 is supported under this synthetic setting. Top-state cumulative hotness share is 1.00; H2 is supported.

## 2. Problem Characterization

- Input redundancy ratio: 5.93
- Total prompt tokens: 89002
- Unique consumed state tokens: 15009
- Duplicate materialization bytes: 295972
- Dynamic hot states: 0

### Top Hot States

| rank | state | kind | consumers | hotness | cumulative share |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | `S_task` | task | 33 | 573440.0 | 0.46 |
| 2 | `S_market_context` | market_context | 33 | 573440.0 | 0.92 |
| 3 | `S_risk_policy` | risk_policy | 9 | 51200.0 | 0.96 |
| 4 | `S_role_instruction` | role_instruction | 8 | 24640.0 | 0.98 |
| 5 | `S_tool_schema` | tool_schema | 8 | 22400.0 | 0.99 |
| 6 | `S_trade_3` | planner_output | 2 | 537.1 | 0.99 |
| 7 | `S_trade_4` | planner_output | 2 | 512.5 | 0.99 |
| 8 | `S_trade_1` | planner_output | 2 | 510.4 | 1.00 |
| 9 | `S_market_signal_5` | retrieval | 2 | 510.3 | 1.00 |
| 10 | `S_trade_0` | planner_output | 2 | 506.3 | 1.00 |

### State Fanout Table

| state | kind | tokens | consumers | token-weighted fanout |
| --- | --- | ---: | ---: | ---: |
| `S_task` | task | 800 | 33 | 25600 |
| `S_market_context` | market_context | 800 | 33 | 25600 |
| `S_risk_policy` | risk_policy | 800 | 9 | 6400 |
| `S_tool_schema` | tool_schema | 800 | 8 | 5600 |
| `S_role_instruction` | role_instruction | 800 | 8 | 5600 |
| `S_market_signal_5` | retrieval | 486 | 2 | 486 |
| `S_market_signal_0` | retrieval | 445 | 2 | 445 |
| `S_market_signal_2` | retrieval | 438 | 2 | 438 |
| `S_market_signal_3` | retrieval | 426 | 2 | 426 |
| `S_market_signal_1` | retrieval | 409 | 2 | 409 |

## 3. Method

This run builds a State Access Graph and measures state-level fanout before any wafer claim is made.

## 7. Failure Cases

If the redundancy ratio is close to 1.0 or hotness share is flat, state-centric scheduling is not a strong paper claim for this workflow.

## 8. What This Means for the Paper

Use this characterization as a gate before scheduler results. Unsupported hypotheses should be reported rather than hidden.
