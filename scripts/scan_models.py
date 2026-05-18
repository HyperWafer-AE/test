#!/usr/bin/env python
from __future__ import annotations

import argparse

from waferagent.model_discovery import write_model_index
from waferagent.utils import configure_project_env


def main() -> None:
    configure_project_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data2/model_zoo")
    parser.add_argument("--output", default="configs/models.local.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="")
    parser.add_argument("--engine", default="synthetic")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--gpus", default="")
    args = parser.parse_args()
    index = write_model_index(args.root, args.output)
    print(f"Wrote {args.output} with {len(index['models'])} models")


if __name__ == "__main__":
    main()
