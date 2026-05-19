from __future__ import annotations

from dataclasses import dataclass

from waferagent.graph_ir import AgentGraph
from waferagent.workloads import WorkloadParams, generate_workload


@dataclass(frozen=True)
class ControlledRegimeConfig:
    workload: str = "controlled_shared_kv_reuse"
    num_jobs: int = 100
    reuse_group_size: int = 8
    shared_prefix_tokens: int = 8192
    private_suffix_tokens: int = 512
    decode_tokens: int = 128
    num_agents_per_job: int = 8
    dependency_pattern: str = "debate"
    cross_job_reuse_probability: float = 1.0
    seed: int = 11


def generate_controlled_regime_graphs(cfg: ControlledRegimeConfig) -> list[AgentGraph]:
    graphs: list[AgentGraph] = []
    input_len = int(cfg.shared_prefix_tokens + cfg.private_suffix_tokens)
    ratio = cfg.shared_prefix_tokens / max(1, input_len)
    unique_ratio = max(0.0, min(1.0, 1.0 - cfg.cross_job_reuse_probability))
    if cfg.workload == "controlled_low_reuse":
        unique_ratio = max(unique_ratio, 0.9)
    elif cfg.workload == "controlled_mixed_reuse":
        unique_ratio = max(unique_ratio, 0.5)
    base_workload = "decode_heavy_shared_prefix" if cfg.dependency_pattern in {"debate", "decode_heavy"} else "moa_decode_cohort_stress"
    for j in range(cfg.num_jobs):
        params = WorkloadParams(
            workload=base_workload,
            job_id=f"{cfg.workload}_job_{j}",
            seed=cfg.seed + j,
            num_agents=cfg.num_agents_per_job,
            input_len=input_len,
            output_len=cfg.decode_tokens,
            shared_prefix_ratio=ratio,
            cross_job_task_group_size=max(1, cfg.reuse_group_size),
            unique_task_ratio=unique_ratio,
            prefix_namespace=f"controlled|{cfg.workload}|g{cfg.reuse_group_size}|s{cfg.shared_prefix_tokens}|d{cfg.decode_tokens}",
        )
        graph = generate_workload(params)
        graph.workload = cfg.workload
        graphs.append(graph)
    return graphs

