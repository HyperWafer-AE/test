# WaferAgent

Trace-driven experimental framework for **WaferAgent: Graph-Aware Wafer-Scale
Serving for LLM Multi-Agent Systems**.

This repository treats wafer-scale results as simulation results. It does not
claim measurements on real wafer hardware unless a future experiment explicitly
connects calibrated hardware data.

## Quick Start

```bash
cd /home/duzc/data/agent_wafer
export PROJECT_ROOT=/home/duzc/data/agent_wafer
export HF_HOME=/home/duzc/data/agent_wafer/.cache/huggingface
export TRANSFORMERS_CACHE=/home/duzc/data/agent_wafer/.cache/huggingface
export XDG_CACHE_HOME=/home/duzc/data/agent_wafer/.cache
export UV_CACHE_DIR=/home/duzc/data/agent_wafer/.cache/uv
export UV_PYTHON_INSTALL_DIR=/home/duzc/data/agent_wafer/.cache/uv/python
export TOKENIZERS_PARALLELISM=false
export MODEL_ZOO=/data2/model_zoo

uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
# Optional, only for HF/vLLM runs:
# uv pip install -e ".[hf]"
# uv pip install vllm

pytest tests -q
python scripts/run_smoke_test.py --engine synthetic --out results/smoke
```

## Main Synthetic Flow

```bash
python scripts/scan_models.py --root /data2/model_zoo --output configs/models.local.json
python scripts/collect_h100_traces.py --engine synthetic --workloads debate,moa,planner_worker_tool,swe_like,rag_like --num-jobs 20 --out results/characterization_synthetic
python scripts/run_simulation_sweep.py --traces "results/characterization_synthetic/traces/*.jsonl" --wafer-config configs/wafer/wse_like.yaml --baselines wafer_naive,kvflow_like,continuum_like,waferagent_full --out results/main_wafer_sim_synthetic
python scripts/run_ablation.py --traces "results/characterization_synthetic/traces/*.jsonl" --wafer-config configs/wafer/wse_like.yaml --out results/ablation_synthetic
python scripts/run_sensitivity.py --engine synthetic --wafer-config configs/wafer/wse_like.yaml --out results/sensitivity_synthetic
python scripts/make_report.py --results results/main_wafer_sim_synthetic --out results/main_wafer_sim_synthetic/report.md
```

## H100/HF Flow

The HF and vLLM runners are best-effort. If a model cannot be found or loaded,
the scripts record the reason and fall back to synthetic traces.

```bash
python scripts/calibrate_h100.py --model auto --engine hf --gpus 0,1 --out results/h100_calibration
python scripts/collect_h100_traces.py --model auto --engine hf --workloads debate,moa --num-jobs 5 --gpus 0,1 --out results/characterization_mini
```
