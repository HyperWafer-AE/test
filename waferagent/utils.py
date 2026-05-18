from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/home/duzc/data/agent_wafer")).resolve()
MODEL_ZOO = Path(os.environ.get("MODEL_ZOO", "/data2/model_zoo")).resolve()


def configure_project_env() -> None:
    os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))
    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / ".cache" / "huggingface"))
    os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
    os.environ.setdefault("UV_CACHE_DIR", str(PROJECT_ROOT / ".cache" / "uv"))
    os.environ.setdefault("UV_PYTHON_INSTALL_DIR", str(PROJECT_ROOT / ".cache" / "uv" / "python"))
    os.environ.setdefault("TMPDIR", str(PROJECT_ROOT / "tmp"))
    os.environ.setdefault("CUDA_CACHE_PATH", str(PROJECT_ROOT / ".cache" / "cuda"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(PROJECT_ROOT / ".cache" / "triton"))
    os.environ.setdefault("RAY_TMPDIR", str(PROJECT_ROOT / "tmp" / "ray"))
    os.environ.setdefault("VLLM_CACHE_ROOT", str(PROJECT_ROOT / ".cache" / "vllm"))
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("MODEL_ZOO", str(MODEL_ZOO))
    (PROJECT_ROOT / "tmp").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / ".cache" / "cuda").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / ".cache" / "triton").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / ".cache" / "vllm").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "tmp" / "ray").mkdir(parents=True, exist_ok=True)


def require_project_path(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    if resolved == PROJECT_ROOT or PROJECT_ROOT in resolved.parents:
        return resolved
    if MODEL_ZOO == resolved or MODEL_ZOO in resolved.parents:
        return resolved
    raise ValueError(f"Refusing to write outside project/model roots: {resolved}")


def ensure_dir(path: str | Path) -> Path:
    p = require_project_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":")))


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def now_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


def run_command_capture(cmd: list[str], cwd: Path | None = None, timeout_s: int = 20) -> str:
    try:
        out = subprocess.check_output(
            cmd,
            cwd=str(cwd or PROJECT_ROOT),
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            text=True,
        )
        return out.strip()
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"unavailable: {exc}"


def git_commit() -> str:
    return run_command_capture(["git", "rev-parse", "HEAD"], timeout_s=5)


def git_metadata() -> dict[str, Any]:
    commit = git_commit()
    status = run_command_capture(["git", "status", "--short"], timeout_s=5)
    branch = run_command_capture(["git", "branch", "--show-current"], timeout_s=5)
    remote = run_command_capture(["git", "remote", "get-url", "origin"], timeout_s=5)
    dirty_files: list[str] = []
    for line in status.splitlines():
        if not line.strip():
            continue
        if len(line) > 3 and line[2] == " ":
            dirty_files.append(line[3:])
        else:
            parts = line.split(maxsplit=1)
            dirty_files.append(parts[-1])
    return {
        "git_commit": commit,
        "git_status_short": status,
        "branch": branch,
        "remote_url": remote,
        "dirty_files": dirty_files,
    }


def nvidia_smi_text() -> str:
    query = "name,memory.total,driver_version"
    return run_command_capture(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"], timeout_s=15
    )


def torch_environment() -> dict[str, str]:
    try:
        import torch

        return {
            "torch_version": str(torch.__version__),
            "cuda_available": str(torch.cuda.is_available()),
            "torch_cuda_version": str(torch.version.cuda),
            "gpu_count": str(torch.cuda.device_count()),
        }
    except Exception as exc:  # pragma: no cover - depends on install
        return {"torch": f"unavailable: {exc}"}


def environment_dict(command: str | None = None) -> dict[str, Any]:
    env = {
        "project_root": str(PROJECT_ROOT),
        "model_zoo": str(MODEL_ZOO),
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "nvidia_smi": nvidia_smi_text(),
        "command": command or " ".join(sys.argv),
        "env": {
            k: os.environ.get(k, "")
            for k in [
                "PROJECT_ROOT",
                "HF_HOME",
                "TRANSFORMERS_CACHE",
                "XDG_CACHE_HOME",
                "UV_CACHE_DIR",
                "UV_PYTHON_INSTALL_DIR",
                "TMPDIR",
                "CUDA_CACHE_PATH",
                "TRITON_CACHE_DIR",
                "RAY_TMPDIR",
                "VLLM_CACHE_ROOT",
                "VLLM_WORKER_MULTIPROC_METHOD",
                "TOKENIZERS_PARALLELISM",
                "MODEL_ZOO",
            ]
        },
    }
    env.update(torch_environment())
    return env


def write_json(path: str | Path, obj: Any) -> Path:
    p = require_project_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    p = require_project_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    return p


def append_text(path: str | Path, text: str) -> None:
    p = require_project_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(text)


def init_run_dir(out: str | Path, metadata: dict[str, Any] | None = None) -> Path:
    configure_project_env()
    out_path = ensure_dir(out)
    for sub in ["traces", "calibration", "simulation", "figures"]:
        ensure_dir(out_path / sub)
    metadata = metadata or {}
    metadata.setdefault("created_unix", time.time())
    metadata.setdefault("command", " ".join(sys.argv))
    metadata.update({k: v for k, v in git_metadata().items() if k not in metadata})
    write_json(out_path / "metadata.json", metadata)
    try:
        import yaml

        (out_path / "metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=True), encoding="utf-8")
    except Exception:
        (out_path / "metadata.yaml").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    command = metadata.get("command", " ".join(sys.argv))
    (out_path / "command.txt").write_text(str(command) + "\n", encoding="utf-8")
    (out_path / "commands.log").write_text(str(command) + "\n", encoding="utf-8")
    env = environment_dict(metadata.get("command"))
    write_json(out_path / "environment.json", env)
    env_text = "\n".join(f"{k}: {v}" for k, v in env.items())
    (out_path / "environment.txt").write_text(env_text + "\n", encoding="utf-8")
    manifest = {
        "metadata": metadata,
        "environment": env,
        "created_unix": metadata["created_unix"],
        "schema_version": "2.0",
    }
    write_json(out_path / "run_manifest.json", manifest)
    return out_path


def stable_rng_seed(seed: int, *parts: Any) -> int:
    digest = hashlib.sha256(("|".join(map(str, (seed,) + parts))).encode()).hexdigest()
    return int(digest[:8], 16)
