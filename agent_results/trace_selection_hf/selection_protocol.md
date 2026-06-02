# Trace Selection Protocol

- Traces are selected by workload-intrinsic metrics only.
- ASG/LRU/KVFlow performance outputs are not read by this selector.
- The full candidate set, ASG-opportunity subset, matched control subset, smoke-only set, and unusable set are all reported.
- Negative/control results must not be hidden.
- Smoke-only/full-history traces are not used for paper claims.

selection_timestamp: 2026-06-02T04:43:42.077861+00:00
selection_signature_sha256: df400793ee6141ec46158131a8e2d24ac38177956ebec94c6d92ed60756ccc24
num_candidates: 181
num_opportunity_rich: 2
num_matched_control: 0
num_smoke_only: 112
num_unusable: 67
