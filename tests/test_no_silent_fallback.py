from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_hf_bad_model_fails_without_synthetic_fallback():
    root = Path("/home/duzc/data/agent_wafer")
    env = os.environ.copy()
    env.update(
        {
            "PROJECT_ROOT": str(root),
            "HF_HOME": str(root / ".cache" / "huggingface"),
            "TRANSFORMERS_CACHE": str(root / ".cache" / "huggingface"),
            "XDG_CACHE_HOME": str(root / ".cache"),
            "UV_CACHE_DIR": str(root / ".cache" / "uv"),
            "UV_PYTHON_INSTALL_DIR": str(root / ".cache" / "uv" / "python"),
            "TMPDIR": str(root / "tmp"),
            "TOKENIZERS_PARALLELISM": "false",
            "MODEL_ZOO": "/data2/model_zoo",
        }
    )
    out = root / "tmp" / "test_no_silent_fallback"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/collect_h100_traces.py",
            "--engine",
            "hf",
            "--model",
            "__definitely_missing_model__",
            "--workloads",
            "debate",
            "--num-jobs",
            "1",
            "--out",
            str(out),
        ],
        cwd=str(root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.returncode != 0
