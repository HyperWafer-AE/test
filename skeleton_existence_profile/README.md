# Skeleton Existence Profiling

This subproject profiles whether dynamic agent workflows contain reusable static stateflow skeletons. It does not implement an Agent-on-Wafer runtime and does not reuse any previous profile conclusion as an assumption.

## Setup

```bash
cd /home/duzc/data/agent_wafer/skeleton_existence_profile
pip install -r requirements.txt
```

## Strict-real TerminalBench run

The loader uses the Hugging Face dataset server for `yoonholee/terminalbench-trajectories`. It never falls back to mock traces. If the network fails before any usable row is loaded, the command exits with a clear error.

```bash
python scripts/run_all.py \
  --datasets terminalbench \
  --sample-size 1500 \
  --strict-real \
  --profile-mode skeleton_existence \
  --outdir outputs/skeleton_existence_tb1500 \
  --page-size 25
```

The same runner supports `--sample-size 3000` if more time is available.

## Tests

```bash
PYTHONPATH=src pytest -q
```

The tests cover parser behavior, clean transition filtering, artifact filtering, dependency extraction, object granularity separation, permutation controls, and strict-real no-mock behavior.

## Main outputs

- Canonical tables: `outputs/skeleton_existence_tb1500/data/`
- Metrics: `outputs/skeleton_existence_tb1500/tables/`
- Figures: `outputs/skeleton_existence_tb1500/figures/`
- Reports: `outputs/skeleton_existence_tb1500/reports/`
- Run metadata: `outputs/skeleton_existence_tb1500/metadata.json`

The report verdict is intentionally allowed to be negative. The current 1500-trace run should be read from `reports/problem_existence_report.md`, not from any older profile.
