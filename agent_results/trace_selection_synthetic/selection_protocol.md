# Trace Selection Protocol

- Traces are selected by workload-intrinsic metrics only.
- ASG/LRU/KVFlow performance outputs are not read by this selector.
- The full candidate set, ASG-opportunity subset, matched control subset, smoke-only set, and unusable set are all reported.
- Negative/control results must not be hidden.
- Smoke-only/full-history traces are not used for paper claims.

selection_timestamp: 2026-06-02T08:21:00.024380+00:00
selection_signature_sha256: fe1ba62eb0587c942acafbe78e8dcdf0b075dba0e9f9f6ff3af8092106db718b
num_candidates: 22
num_opportunity_rich: 7
num_matched_control: 15
num_smoke_only: 0
num_unusable: 0
