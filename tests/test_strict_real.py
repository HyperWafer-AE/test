from __future__ import annotations

import pytest

from scripts import run_all
from src.loaders.terminalbench import LoadResult


def test_strict_real_refuses_mock_fallback(monkeypatch) -> None:
    def fake_load_terminalbench(**kwargs):
        return LoadResult("yoonholee/terminalbench-trajectories", [], ["using mock rows"], used_mock=True)

    monkeypatch.setattr(run_all, "load_terminalbench", fake_load_terminalbench)
    with pytest.raises(RuntimeError):
        run_all._load_dataset_rows(
            dataset="terminalbench",
            sample_size=1,
            cache_dir=None,
            streaming=True,
            offline_mock=False,
            seed=0,
            strict_real=True,
        )
