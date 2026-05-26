# WaferStateFlow Report

## 1. Executive Summary

Workflow `trading` has input redundancy ratio 6.73. H1 is supported under this synthetic setting. Top-state cumulative hotness share is 1.00; H2 is supported.

## 2. Problem Characterization

- Input redundancy ratio: 6.73
- Total prompt tokens: 175598
- Unique consumed state tokens: 26107
- Duplicate materialization bytes: 597964
- Dynamic hot states: 2

### Top Hot States

| rank | state | kind | consumers | hotness | cumulative share |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | `S_task` | task | 65 | 2232320.0 | 0.46 |
| 2 | `S_market_context` | market_context | 65 | 2232320.0 | 0.92 |
| 3 | `S_risk_policy` | risk_policy | 17 | 189440.0 | 0.96 |
| 4 | `S_role_instruction` | role_instruction | 16 | 105600.0 | 0.98 |
| 5 | `S_tool_schema` | tool_schema | 16 | 96000.0 | 1.00 |
| 6 | `S_trade_9` | planner_output | 2 | 588.3 | 1.00 |
| 7 | `S_trade_15` | planner_output | 2 | 539.1 | 1.00 |
| 8 | `S_trade_3` | planner_output | 2 | 537.1 | 1.00 |
| 9 | `S_trade_12` | planner_output | 2 | 533.0 | 1.00 |
| 10 | `S_trade_11` | planner_output | 2 | 530.9 | 1.00 |

### State Fanout Table

| state | kind | tokens | consumers | token-weighted fanout |
| --- | --- | ---: | ---: | ---: |
| `S_task` | task | 800 | 65 | 51200 |
| `S_market_context` | market_context | 800 | 65 | 51200 |
| `S_risk_policy` | risk_policy | 800 | 17 | 12800 |
| `S_tool_schema` | tool_schema | 800 | 16 | 12000 |
| `S_role_instruction` | role_instruction | 800 | 16 | 12000 |
| `S_market_signal_5` | retrieval | 486 | 2 | 486 |
| `S_market_signal_10` | retrieval | 460 | 2 | 460 |
| `S_market_signal_0` | retrieval | 445 | 2 | 445 |
| `S_market_signal_2` | retrieval | 438 | 2 | 438 |
| `S_market_signal_14` | retrieval | 429 | 2 | 429 |

## 3. Method

This run builds a State Access Graph and measures state-level fanout before any wafer claim is made.

## 7. Failure Cases

If the redundancy ratio is close to 1.0 or hotness share is flat, state-centric scheduling is not a strong paper claim for this workflow.

## 8. What This Means for the Paper

Use this characterization as a gate before scheduler results. Unsupported hypotheses should be reported rather than hidden.
