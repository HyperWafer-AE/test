#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from waferagent.utils import configure_project_env, init_run_dir, write_json


TORCH_CHECK = r"""
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
"""


def main() -> None:
    configure_project_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/env_validation")
    parser.add_argument("--engine", default="synthetic")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = init_run_dir(args.out, {"run_type": "env_validation", "engine": args.engine, "model": args.model, "gpus": args.gpus, "seed": args.seed})
    try:
        proc = subprocess.run(
            [sys.executable, "-c", TORCH_CHECK],
            cwd=str(Path.cwd()),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=60,
        )
        text = proc.stdout
        code = proc.returncode
    except Exception as exc:
        text = f"torch check failed: {exc}\n"
        code = 1
    (out / "torch_cuda_check.txt").write_text(text, encoding="utf-8")
    write_json(out / "validation.json", {"torch_cuda_check_exit_code": code, "torch_cuda_check": text})
    print(text)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
