from __future__ import annotations

from pathlib import Path


def whitespace_token_count(text: str) -> int:
    return max(1, len(text.split()))


def load_tokenizer(model_path: str | Path | None = None):
    if not model_path:
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    except Exception:
        return None


def count_tokens(text: str, tokenizer=None) -> int:
    if tokenizer is None:
        return whitespace_token_count(text)
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return whitespace_token_count(text)
