# Missing vLLM 20-job Trace Baseline

- status: explicitly_missing
- fallback_used: false
- command_expected: `uv run python scripts/collect_h100_traces.py --model Qwen2.5-7B-Instruct --engine vllm --workloads debate,moa,planner_worker_tool,swe_like,rag_like --num-jobs 20 --gpus 1 --out results/round9_characterization_h100_vllm_20jobs --seed 31 --clean-required`
- model_selected: `/data2/model_zoo/Qwen2.5-7B-Instruct`
- gpu: `1`
- environment: `torch 2.5.1+cu124`, `vllm 0.6.4.post1`, CUDA available with 2 x NVIDIA H100 PCIe
- error_message: not run in this Round 9 artifact build after the HF 20-job attempt exceeded the local execution window.
- reason_for_exclusion: no completed vLLM 20-job trace artifact exists; vLLM is excluded from paper-ready claims for this artifact bundle.

