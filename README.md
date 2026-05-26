# WaferStateFlow

WaferStateFlow is a synthetic research prototype for discovering whether
agent-style workflows contain state-level redundancy and hot-state skew. It
represents workflows as a State Access Graph, then runs analyzers and a simple
simulator over synthetic workflow families.

## Scope

- This repository is a problem-discovery tool, not a production serving system.
- The current experiments are synthetic and are useful for hypothesis screening.
- The simulator is intentionally approximate. It should not be used as
  paper-grade speedup evidence without real traces and calibrated hardware data.
- Wafer-aware results must be interpreted with counterexamples. If a
  request-centric or operator-centric baseline is no worse, that result should
  remain in the report.

## Current Entry Points

```bash
python -m waferstateflow.experiments.run_problem_characterization --workflow trading --batch-size 16 --seed 0 --out results/waferstateflow_characterization
python -m waferstateflow.experiments.run_scheduler_comparison --workflow trading --batch-size 16 --mesh 32x32 --state-policy dynamic --seed 0 --out results/waferstateflow_scheduler
python -m waferstateflow.experiments.run_workflow_suite --workflows all --mode both --batch-size 8 --mesh 16x16 --state-policy dynamic --seed 0 --out results/waferstateflow_suite
```

## Hygiene

The repository contains historical KV-ring work in git history. WaferStateFlow
does not require deleting or reverting those files. Preserve unrelated user or
historical changes unless a task explicitly asks otherwise.
