from __future__ import annotations

from dataclasses import dataclass

from waferagent.graph_ir import AgentNode, NodeType
from waferagent.llm_runner import RunnerConfig, make_runner
from waferagent.trace_schema import TraceRecord


@dataclass(frozen=True)
class BenchmarkCase:
    input_len: int
    output_len: int
    batch_size: int
    seed: int


def run_case_with_runner(run_id: str, runner, case: BenchmarkCase) -> list[TraceRecord]:
    rows: list[TraceRecord] = []
    for b in range(case.batch_size):
        node = AgentNode(
            node_id=f"calib_i{case.input_len}_o{case.output_len}_b{case.batch_size}_{b}",
            job_id=f"calib_i{case.input_len}_o{case.output_len}_b{case.batch_size}",
            agent_id=f"batch_{b}",
            round_id=0,
            node_type=NodeType.LLM_CALL,
            role="calibration",
            input_token_len=case.input_len,
            expected_output_token_len=case.output_len,
            actual_output_token_len=case.output_len,
            shared_prefix_token_len=0,
            private_prefix_token_len=case.input_len,
        )
        rows.append(runner.run_node(run_id, "calibration", node))
    return rows


def run_case(run_id: str, runner_config: RunnerConfig, case: BenchmarkCase) -> list[TraceRecord]:
    return run_case_with_runner(run_id, make_runner(runner_config), case)
