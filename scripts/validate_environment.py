#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from waferagent.utils import (
    configure_project_env,
    enforce_clean_git_tree,
    finalize_run_dir,
    init_run_dir,
    write_json,
)


TORCH_CHECK = r"""
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
"""


PACKAGE_CHECK = r"""
import importlib
import importlib.metadata as metadata

packages = [
    "torch",
    "transformers",
    "vllm",
    "xformers",
    "numpy",
    "pandas",
    "pytest",
]
for package in packages:
    try:
        print(f"{package}={metadata.version(package)}")
    except Exception as exc:
        print(f"{package}=MISSING:{exc}")

for module in ["torch", "transformers", "vllm"]:
    try:
        importlib.import_module(module)
        print(f"import:{module}=ok")
    except Exception as exc:
        print(f"import:{module}=failed:{exc!r}")
        raise
"""


def main() -> None:
    configure_project_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/env_validation")
    parser.add_argument("--engine", default="synthetic")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clean-required", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    enforce_clean_git_tree(args.clean_required, args.allow_dirty)
    out = init_run_dir(
        args.out,
        {
            "run_type": "env_validation",
            "engine": args.engine,
            "model": args.model,
            "gpus": args.gpus,
            "seed": args.seed,
            "clean_required": bool(args.clean_required),
        },
    )
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
    try:
        pkg_proc = subprocess.run(
            [sys.executable, "-c", PACKAGE_CHECK],
            cwd=str(Path.cwd()),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=120,
        )
        package_text = pkg_proc.stdout
        package_code = pkg_proc.returncode
    except Exception as exc:
        package_text = f"package check failed: {exc}\n"
        package_code = 1
    (out / "package_check.txt").write_text(package_text, encoding="utf-8")
    write_json(
        out / "validation.json",
        {
            "torch_cuda_check_exit_code": code,
            "torch_cuda_check": text,
            "package_check_exit_code": package_code,
            "package_check": package_text,
        },
    )
    print(text)
    print(package_text)
    finalize_run_dir(out)
    raise SystemExit(code or package_code)


if __name__ == "__main__":
    main()
