from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests


DATASET_ID = "yoonholee/terminalbench-trajectories"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_terminalbench_strict_real(
    sample_size: int,
    raw_path: Path,
    page_size: int = 10,
    resume: bool = True,
    base_url: str = "https://datasets-server.huggingface.co",
    max_retries: int = 6,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load TerminalBench rows through the public datasets-server API.

    There is intentionally no mock fallback here. A failure raises unless a
    cache with enough real rows already exists.
    """
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_jsonl(raw_path) if resume else []
    if len(existing) >= sample_size:
        return existing[:sample_size], {
            "dataset": "terminalbench",
            "source": "datasets_server_cache",
            "used_mock": False,
            "skipped_rows": 0,
        }

    mode = "a" if existing else "w"
    written = len(existing)
    offset = len(existing)
    skipped_rows = 0
    encoded = quote(DATASET_ID, safe="")
    with raw_path.open(mode, encoding="utf-8") as f:
        while written < sample_size:
            length = page_size
            url = f"{base_url.rstrip('/')}/rows?dataset={encoded}&config=default&split=train&offset={offset}&length={length}"
            payload = None
            for attempt in range(1, max_retries + 1):
                try:
                    r = requests.get(url, timeout=120)
                    r.raise_for_status()
                    payload = r.json()
                    break
                except Exception as exc:
                    if attempt == max_retries:
                        if _read_jsonl(raw_path):
                            rows = _read_jsonl(raw_path)
                            if rows:
                                print(f"warning: using partial strict-real cache with {len(rows)} rows after {exc}", file=sys.stderr)
                                return rows[:sample_size], {
                                    "dataset": "terminalbench",
                                    "source": "datasets_server_partial_cache",
                                    "used_mock": False,
                                    "skipped_rows": skipped_rows,
                                    "partial_load_error": repr(exc),
                                }
                        raise RuntimeError(f"strict-real TerminalBench load failed at offset {offset}: {exc}") from exc
                    time.sleep(min(30, 2**attempt))
            rows = payload.get("rows", []) if payload else []
            if not rows:
                raise RuntimeError(f"strict-real TerminalBench load returned no rows at offset {offset}")
            for item in rows:
                row = item.get("row", item)
                steps = row.get("steps")
                offset += 1
                if not steps or steps == "null":
                    skipped_rows += 1
                    continue
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
                written += 1
                if written >= sample_size:
                    break
            print(f"Fetched {written}/{sample_size} usable TerminalBench rows (scanned offset {offset})", file=sys.stderr)

    rows = _read_jsonl(raw_path)
    if not rows:
        raise RuntimeError("strict-real TerminalBench load produced zero usable rows; mock fallback is disabled")
    return rows[:sample_size], {
        "dataset": "terminalbench",
        "source": "datasets_server",
        "used_mock": False,
        "skipped_rows": skipped_rows,
    }
