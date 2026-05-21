# KVRing Round 3 Report

Scope: attention-only shared-prefix plus local suffix modeling; non-attention LLM layers are not included.
This is a microarchitectural shared-prefix attention-stage simulator, not a full LLM serving system and not a real wafer hardware measurement.
Model path: `/data1/duzc/model/model/LLM-Research/Meta-Llama-3___1-8B-Instruct`
KV/token: 128.000 KiB
Shared prefix: 32768 tokens = 4.000 GiB
Agents: 8; decode tokens/agent: 256
NoC channel model: directed bidirectional; VC model: `not_modeled_for_performance`

Headline latency uses `throughput_bound_latency_s`; serialized and critical-path bounds are exported separately.

## Main Headline Table (capacity-valid rows only)

| Mode | Peak SRAM | Wire traffic | Max directed link | SRAM port | Attention-stage proxy latency |
|---|---:|---:|---:|---:|---:|
| Replicate-All | 4.031 GiB | 368.000 GiB | 16.000 GiB | 1.000 TiB | 221.810980 ms |
| Pull-KV-Independent | 4.000 GiB | 92.000 TiB | 4.000 TiB | 8.000 TiB | 44.380582 s |
| Central-KV-Stationary | 4.000 GiB | 679.000 MiB | 32.500 MiB | 1.000 TiB | 84.359739 ms |
| KVRing-v1-sequential-pipeline | 544.000 MiB | 9.000 GiB | 36.000 MiB | 1.000 TiB | 50.012208 ms |
| KVRing-v2-ring | 544.000 MiB | 3.527 GiB | 144.500 MiB | 1.031 TiB | 6.253100 ms |
| KVRing-v2-tree | 544.000 MiB | 3.273 GiB | 112.000 MiB | 1.031 TiB | 6.253100 ms |
| KVRing-v2-region-split | 544.000 MiB | 2.512 GiB | 144.500 MiB | 1.031 TiB | 6.253100 ms |
| KVRing-v3-adaptive | 544.000 MiB | 3.527 GiB | 144.500 MiB | 1.031 TiB | 6.253100 ms |

KVRing-v2 packet accounting is symbolic: `Q_tile + FP32(m,l,z)`; exact online-softmax state is reduced by ring/tree collectives.

## Baseline accounting notes
- Replicate-All separates setup replication (32.0 GiB payload, 0.359375 TiB directed byte-hop) from steady-state local shared-KV reads (1024.0 GiB).
- Central-KV-Stationary keeps KV off the mesh during decode; its bottleneck is central SRAM/compute queueing (0.084359739 s).
- KVRing-v2-ring reports query scatter, shard compute, exact online-softmax reduction, suffix, and merge components separately.
- KVRing-v2-tree reports query scatter, shard compute, exact online-softmax reduction, suffix, and merge components separately.
- KVRing-v2-region-split reports query scatter, shard compute, exact online-softmax reduction, suffix, and merge components separately.