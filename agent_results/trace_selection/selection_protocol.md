# Trace Selection Protocol

- Traces are selected by workload-intrinsic metrics only.
- ASG/LRU/KVFlow performance outputs are not read by this selector.
- The full candidate set, ASG-opportunity subset, matched control subset, smoke-only set, and unusable set are all reported.
- Negative/control results must not be hidden.
- Smoke-only/full-history traces are not used for paper claims.

selection_timestamp: 2026-06-02T04:13:20.451459+00:00
selection_signature_sha256: 3ebfe9f5b92e2556862f65ccf795056d2b635804de87620e0753cb0d19fc57cf
num_candidates: 12
num_opportunity_rich: 0
num_matched_control: 0
num_smoke_only: 12
num_unusable: 0
