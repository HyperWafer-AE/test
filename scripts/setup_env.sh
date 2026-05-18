#!/usr/bin/env bash
set -euo pipefail

export PROJECT_ROOT=/home/duzc/data/agent_wafer
export HF_HOME=/home/duzc/data/agent_wafer/.cache/huggingface
export TRANSFORMERS_CACHE=/home/duzc/data/agent_wafer/.cache/huggingface
export XDG_CACHE_HOME=/home/duzc/data/agent_wafer/.cache
export UV_CACHE_DIR=/home/duzc/data/agent_wafer/.cache/uv
export UV_PYTHON_INSTALL_DIR=/home/duzc/data/agent_wafer/.cache/uv/python
export TMPDIR=/home/duzc/data/agent_wafer/tmp
export TOKENIZERS_PARALLELISM=false
export MODEL_ZOO=/data2/model_zoo

mkdir -p "$PROJECT_ROOT/.cache/huggingface" "$PROJECT_ROOT/.cache/uv" "$UV_PYTHON_INSTALL_DIR" "$TMPDIR"
cd "$PROJECT_ROOT"

if [ ! -d .venv ]; then
  uv venv --python 3.11
fi

source .venv/bin/activate
uv pip install -e ".[dev]"
