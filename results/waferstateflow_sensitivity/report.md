# WaferStateFlow Wafer Sensitivity

## Executive Summary

Swept 72 mesh/memory/state-size/branch-width cases. Win counts: {'wafer_request_centric': 6, 'request_parallel_gpu_like': 6, 'WaferStateFlow': 60}.

## Failure Cases

Example non-WaferStateFlow winner: `wafer_request_centric` on mesh 8x8, memory 67108864, shared size 200, branch width 1. This indicates wafer-aware waves are not always necessary.

## What This Means for the Paper

The sensitivity sweep is a prototype sanity check, not a calibrated hardware study. It is useful for finding regimes and counterexamples.
