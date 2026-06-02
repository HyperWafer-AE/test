Bug report: safe_divide should return default on zero divisor.

Run `python -m pytest -q` and patch `math_utils.py` so the test passes.