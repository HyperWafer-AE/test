# CodeTraceBench Real Trace Sample

Date pulled: 2026-06-02

Source dataset: https://huggingface.co/datasets/NJU-LINK/CodeTraceBench

Selection:
- Split: verified
- Rows: first 3 available solved Mini-SWE-agent trajectories from the Hugging Face rows API
- Purpose: small high-quality real-code-agent replay sample for Agent-on-Wafer policy validation

Artifacts:
- `miniswe-OpenAI__GPT-5-clap-rs__clap-2501-1c7ee8f3.tar.zst`
  - Extracted trace: `swe_raw/mini_swe_agent__multi/clap-rs__clap-2501/clap-rs__clap-2501.traj.json`
  - Report: `swe_raw/mini_swe_agent__multi/clap-rs__clap-2501/report.json`
  - Solved: true
  - Steps: 25
- `miniswe-OpenAI__GPT-5-clap-rs__clap-2758-8f87bc0d.tar.zst`
  - Extracted trace: `swe_raw/mini_swe_agent__multi/clap-rs__clap-2758/clap-rs__clap-2758.traj.json`
  - Report: `swe_raw/mini_swe_agent__multi/clap-rs__clap-2758/report.json`
  - Solved: true
  - Steps: 41
- `miniswe-OpenAI__GPT-5-clap-rs__clap-3179-107b6c1d.tar.zst`
  - Extracted trace: `swe_raw/mini_swe_agent__multi/clap-rs__clap-3179/clap-rs__clap-3179.traj.json`
  - Report: `swe_raw/mini_swe_agent__multi/clap-rs__clap-3179/report.json`
  - Solved: true
  - Steps: 37

Raw downloads are stored in `traces/real_raw/codetracebench/`.

Replay commands:

```powershell
python -m src.agent.trace_profile --trace-dir traces/real/codetracebench_solved --trace-format codetracer --output agent_results/real_codetracebench_profile.json

python -m src.agent.real_trace_experiment --trace-dir traces/real/codetracebench_solved --trace-format codetracer --prediction-mode heuristic --policy-suite v2 --memory-budget-gb 0.5 --concurrency 3 --output-dir agent_results/real_codetracebench_v2_heuristic
```

Notes:
- `report.json` files are kept for audit but skipped by the trace loader.
- The trajectory files are full message-level Mini-SWE-agent conversations. The normalizer replays assistant turns and user/tool observations sequentially.
