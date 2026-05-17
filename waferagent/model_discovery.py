from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
}


def _model_name(path: Path) -> str:
    return f"{path.name}-local"


def _head_dim(cfg: dict[str, Any]) -> int | None:
    if "head_dim" in cfg:
        return int(cfg["head_dim"])
    hidden = cfg.get("hidden_size")
    heads = cfg.get("num_attention_heads")
    if hidden and heads:
        return int(hidden) // int(heads)
    return None


def scan_model_dir(path: Path) -> dict[str, Any] | None:
    config_path = path / "config.json"
    if not config_path.exists():
        return None
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    has_tokenizer = any((path / f).exists() for f in TOKENIZER_FILES)
    architectures = cfg.get("architectures") or []
    dtype_hint = cfg.get("torch_dtype") or cfg.get("dtype")
    return {
        "name": _model_name(path),
        "path": str(path),
        "has_tokenizer": bool(has_tokenizer),
        "architectures": architectures,
        "model_type": cfg.get("model_type"),
        "num_hidden_layers": cfg.get("num_hidden_layers") or cfg.get("n_layer"),
        "hidden_size": cfg.get("hidden_size") or cfg.get("n_embd"),
        "num_attention_heads": cfg.get("num_attention_heads") or cfg.get("n_head"),
        "num_key_value_heads": cfg.get("num_key_value_heads")
        or cfg.get("multi_query_group_num")
        or cfg.get("num_attention_heads")
        or cfg.get("n_head"),
        "head_dim": _head_dim(cfg),
        "max_position_embeddings": cfg.get("max_position_embeddings")
        or cfg.get("n_positions")
        or cfg.get("seq_length"),
        "dtype_hint": str(dtype_hint) if dtype_hint else None,
    }


def discover_models(root: str | Path) -> dict[str, list[dict[str, Any]]]:
    root_path = Path(root)
    models: list[dict[str, Any]] = []
    if root_path.exists():
        for config_path in sorted(root_path.rglob("config.json")):
            item = scan_model_dir(config_path.parent)
            if item:
                models.append(item)
    return {"models": models}


def write_model_index(root: str | Path, output: str | Path) -> dict[str, Any]:
    index = discover_models(root)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return index


def _size_score(name: str) -> int:
    low = name.lower()
    match = re.search(r"(\d+(?:\.\d+)?)\s*b", low)
    if not match:
        return 100
    size = float(match.group(1))
    if 7 <= size <= 14:
        return 0
    if size < 7:
        return 1
    return 2


def select_model(index: dict[str, Any]) -> dict[str, Any] | None:
    candidates = [m for m in index.get("models", []) if m.get("has_tokenizer")]
    if not candidates:
        candidates = list(index.get("models", []))
    if not candidates:
        return None

    def score(m: dict[str, Any]) -> tuple[int, int, str]:
        name = (m.get("name") or m.get("path") or "").lower()
        instruct = 0 if any(k in name for k in ["instruct", "chat", "it"]) else 1
        return (instruct, _size_score(name), name)

    return sorted(candidates, key=score)[0]


def load_or_scan(root: str | Path, index_path: str | Path) -> dict[str, Any]:
    path = Path(index_path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return write_model_index(root, path)
