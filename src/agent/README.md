# Agent-on-Wafer MVP

This package adds an agent-level frontend over BusyBarn. It converts synthetic multi-agent traces into BusyBarn-compatible fixed compute, communication, and external wait events, then uses the existing analytical event driver for wafer timing.

## Policies

- `nocache`: no persistent KV residency; every LLM step prefills full historical input plus appended tokens.
- `lru`: keeps states within the memory budget by most recent access.
- `kvflow`: approximates workflow-aware retention with next-use distance and reuse score.
- `asg`: uses Agent State Graph reuse prediction, phase-aware value, topology-aware placement, and KV migration events.

## Smoke Tests

```bash
python -m src.agent.experiment --policy asg --cfg src/platform/cfgs/wamis_hd_distributed.cfg --topology wamis --num-agents 2 --turns 4 --memory-budget-gb 1 --seed 1 --output agent_results/smoke_asg.json
```

```bash
python -m src.agent.experiment --run-all-policies --cfg src/platform/cfgs/wamis_hd_distributed.cfg --topology wamis --num-agents 4 --turns 8 --memory-budget-gb 1 --seed 1 --output-dir agent_results/
```

## Metrics

- `total_input_tokens`: historical state tokens presented to LLM steps.
- `append_tokens`: newly appended prompt tokens.
- `effective_prefill_tokens`: tokens actually paid as prefill after cache hits.
- `decode_tokens`: generated decode tokens.
- `cache_hit_ratio`: fraction of input state references found resident.
- `effective_prefill_reduction`: reduction against full historical prefill.
- `kv_migration_bytes`: bytes moved through BusyBarn communication events.
- `tool_wait_cycles`: fixed external wait latency from tool calls.

## Limitations

- Synthetic agent trace only.
- No real tokenizer or prompt parsing yet.
- No real model kernel execution.
- KV bytes and prefill/decode costs use a configurable analytical model.
- Tool execution is modeled as fixed external wait.
- Speculation is not implemented yet.
- Skill routing is not implemented yet.

