# WaferStateFlow Claims

## Allowed Claims

- WaferStateFlow builds a first-class State Access Graph for synthetic agent
  workflows.
- The prototype reports input redundancy, state fanout, hotness skew, duplicate
  materialization bytes, policy decisions, wave schedules, and approximate
  simulator metrics.
- Synthetic runs can support or refute hypotheses for the chosen synthetic
  workflow configuration.
- Negative and tie results are valid outcomes and should be retained.

## Disallowed Claims

- Do not claim paper-grade speedups from the synthetic simulator alone.
- Do not claim WaferStateFlow is universally better than GPU-style,
  request-centric, operator-centric, Helium-like, or KVFlow-like baselines.
- Do not claim real wafer hardware performance or calibrated NoC behavior.
- Do not claim KV cache reuse unless `kv_cacheable` and `prefix_compatible` are
  both true for the state, with compatible prompt position/model assumptions.
- Do not merge semantically different agent outputs. Shared inputs may be reused
  or colocated; distinct outputs remain distinct states.
