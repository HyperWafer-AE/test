# Problem Existence Audit

- Required strict-real run: True
- Mock fallback used: False
- Tool-action-only filters exclude unknown/no-tool/artifact steps.
- Object reuse is reported at exact object-id and path-prefix levels.
- Negative controls are in `tables/permutation_tests.csv` and cost stress rows.
- Main command: `python scripts/run_all.py --datasets terminalbench --sample-size 1500 --strict-real --profile-mode skeleton_existence --outdir outputs/skeleton_existence_tb1500`
