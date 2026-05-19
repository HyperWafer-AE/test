# Missing HF 20-job Trace Baseline

- status: attempted_not_completed
- fallback_used: false
- command_attempted: `uv run python scripts/collect_h100_traces.py --model Qwen2.5-7B-Instruct --engine hf --workloads debate,moa,planner_worker_tool,swe_like,rag_like --num-jobs 20 --gpus 0 --out results/round9_characterization_h100_hf_20jobs --seed 11 --clean-required`
- model_selected: `/data2/model_zoo/Qwen2.5-7B-Instruct`
- gpu: `0`
- environment: `torch 2.5.1+cu124`, `transformers 4.46.3`, CUDA available with 2 x NVIDIA H100 PCIe
- observed_behavior: model checkpoint loaded successfully, then the 20-job HF generation trace did not complete within the local execution window.
- termination: the attempted run was terminated after about 11 minutes to avoid leaving a long-running foreground process without finalized trace files.
- reason_for_exclusion: no complete `traces/traces.jsonl` or characterization tables were produced, so this is not reported as a completed HF 20-job baseline.

