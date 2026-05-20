# WaferAgent Round13 Paper Artifacts

All wafer performance numbers are trace-driven wafer-scale simulator results, not real wafer hardware measurements.

The stage-level semantic audit CSVs in this artifact directory are bounded samples; full stage maps are included as `.csv.gz` files.

```json
{
  "claims_allowed": {
    "adaptive_non_worse": true,
    "existing_prefix_cache_gap": true,
    "high_opportunity_speedup": true,
    "replication": false,
    "universal_superiority": false
  },
  "evaluation_ready": {
    "adaptive_semantics_audit_pass": true,
    "controlled_validation_nonempty_and_all_pass": true,
    "mechanism_attribution_unexplained_under_20pct": true,
    "randomized_heldout_policy_pass": true,
    "real_trace_scope_consistent": true
  },
  "paper_writing_allowed": true
}
```
