# Missing HF 10-job Trace Baseline

- command attempted: `uv run python scripts/collect_h100_traces.py --model Qwen2.5-7B-Instruct --engine hf --workloads debate,moa,planner_worker_tool,swe_like,rag_like --num-jobs 10 --gpus 0 --max-new-tokens 32 --max-input-tokens 4096 --stop-after-minutes 60 --out results/round12_characterization_h100_hf_10jobs --seed 11 --clean-required`
- model selected: `Qwen2.5-7B-Instruct-local`
- model path: `/data2/model_zoo/Qwen2.5-7B-Instruct`
- GPU requested: `0`
- fallback used: `false`
- exclusion reason: `/data2/model_zoo` is not present in this execution environment, so the local model path cannot be read. Models must not be downloaded into the project directory.
- error excerpt: `HF model load failed: Incorrect path_or_model_id: '/data2/model_zoo/Qwen2.5-7B-Instruct'.`

This is a missing 10-job baseline artifact, not a synthetic fallback.
