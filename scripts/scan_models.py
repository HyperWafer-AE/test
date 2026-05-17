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
    args = parser.parse_args()
    index = write_model_index(args.root, args.output)
    print(f"Wrote {args.output} with {len(index['models'])} models")


if __name__ == "__main__":
    main()
