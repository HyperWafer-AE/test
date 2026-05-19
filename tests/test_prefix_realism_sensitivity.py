from __future__ import annotations

from waferagent.llm_runner import RunnerConfig
from waferagent.trace_collector import collect_graph_traces
from waferagent.workloads import WorkloadParams, generate_workload


def _prefix_hits(group_size: int, unique_ratio: float) -> int:
    graphs = [
        generate_workload(
            WorkloadParams(
                workload="debate",
                job_id=f"prefix_job_{i}",
                num_agents=2,
                cross_job_task_group_size=group_size,
                unique_task_ratio=unique_ratio,
            )
        )
        for i in range(6)
    ]
    traces = collect_graph_traces(graphs, "prefix_realism", RunnerConfig(engine="synthetic"))
    prefixes = [tr.shared_prefix_ids[0] for tr in traces if tr.shared_prefix_ids]
    return len(prefixes) - len(set(prefixes))


def test_prefix_realism_group_size_and_unique_ratio_affect_reuse():
    low_group = _prefix_hits(group_size=1, unique_ratio=0.0)
    high_group = _prefix_hits(group_size=6, unique_ratio=0.0)
    all_unique = _prefix_hits(group_size=6, unique_ratio=1.0)
    assert high_group > low_group
    assert all_unique < high_group

