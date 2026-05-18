from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArrivalConfig:
    mode: str = "closed_loop"
    rate_jobs_per_s: float = 1.0
    seed: int = 0
    max_jobs: int | None = None
    replay_path: str = ""


def generate_arrivals(job_ids: list[str], cfg: ArrivalConfig) -> dict[str, float]:
    ids = job_ids[: cfg.max_jobs] if cfg.max_jobs else list(job_ids)
    rng = random.Random(cfg.seed)
    if cfg.mode == "closed_loop":
        return {job_id: 0.0 for job_id in ids}
    if cfg.mode == "burst":
        return {job_id: float((i // max(1, int(cfg.rate_jobs_per_s))) * 10.0) for i, job_id in enumerate(ids)}
    if cfg.mode == "poisson":
        t = 0.0
        out: dict[str, float] = {}
        rate_per_ms = max(1e-9, cfg.rate_jobs_per_s / 1000.0)
        for job_id in ids:
            t += rng.expovariate(rate_per_ms)
            out[job_id] = t
        return out
    if cfg.mode == "replay":
        rows = {}
        with Path(cfg.replay_path).open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows[str(row["job_id"])] = float(row["arrival_ms"])
        return {job_id: rows.get(job_id, 0.0) for job_id in ids}
    raise ValueError(f"Unknown arrival mode: {cfg.mode}")
