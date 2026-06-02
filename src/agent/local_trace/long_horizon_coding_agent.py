import argparse
import json
import shutil
import time
from pathlib import Path

from src.agent.local_serving.openai_client_wrapper import LocalOpenAIClientWrapper
from src.agent.local_trace.long_benchmark_tasks import ensure_local_long_debug_tasks
from src.agent.local_trace.prompt_state_builder import PromptStateBuilder
from src.agent.local_trace.tool_wrappers import ToolLogger
from src.agent.trace_schema import validate_trace


DEFAULT_MODEL_PATH = "/data1/dg123_data/Qwen-32B"


def load_long_tasks(tasks_dir: str):
    ensure_local_long_debug_tasks(tasks_dir)
    manifest = json.loads((Path(tasks_dir) / "manifest.json").read_text(encoding="utf-8"))
    return manifest["tasks"]


def run_long_workflow(task: dict, workflow_id: str, args):
    agent_id = "agent_0"
    task_dir = Path(task["path"])
    workspace = Path(args.workspace_root) / workflow_id / "repo"
    if workspace.exists():
        shutil.rmtree(workspace)
    shutil.copytree(task_dir / "repo", workspace)
    issue = (task_dir / "issue.md").read_text(encoding="utf-8")
    metadata = json.loads((task_dir / "metadata.json").read_text(encoding="utf-8"))
    builder = PromptStateBuilder(workflow_id, agent_id, args.history_mode, model_path=args.model_path, recent_k=2)
    tool = ToolLogger(
        workflow_id,
        agent_id,
        builder,
        raw_dir=args.raw_tool_dir,
        artifact_dir=getattr(args, "tool_artifact_dir", "traces/local_h100_long/tool_outputs"),
    )
    client = LocalOpenAIClientWrapper(args.server_url, args.model, output_dir=args.raw_request_dir, server_backend="sglang")
    trace = []
    trace.extend(builder.bootstrap(issue, agent_role="long_horizon_coding_agent"))
    start = time.time()

    relevant = [Path(path) for path in metadata["expected_relevant_files"]]
    opportunity = metadata.get("expected_control_or_opportunity") == "opportunity"
    first_file_state = None
    second_file_state = None
    third_file_state = None
    failure_state = None
    first_edit_state = None
    final_edit_state = None
    glob_state_id = None
    grep_bug_state_id = None
    grep_threshold_state_id = None
    llm_plan_state_id = None
    llm_after_second_state_id = None

    def add(items):
        trace.extend(items)

    def finalize(success_event):
        success = success_event["status"] == "ok"
        for item in trace:
            item.setdefault("metadata", {})
            item["metadata"].setdefault("workflow_id", workflow_id)
            item["metadata"].setdefault("framework", "long_horizon_coding_agent")
            item["metadata"].setdefault("trace_success", success)
            item["metadata"].setdefault("benchmark", "local_long_debug_tasks")
            item["metadata"].setdefault("task_family", metadata.get("task_family"))
            item["metadata"].setdefault("expected_control_or_opportunity", metadata.get("expected_control_or_opportunity"))
            item["metadata"].setdefault("global_timestamp", item.get("timestamp_start") or item.get("timestamp_end") or start)
        validate_trace(trace)
        return trace, {
            "success": success,
            "wall_clock_s": time.time() - start,
            "num_events": len(trace),
            "task_family": metadata.get("task_family"),
            "expected_control_or_opportunity": metadata.get("expected_control_or_opportunity"),
        }

    events = call_llm(client, builder, workflow_id, agent_id, "llm_00_plan", "Plan a long-horizon debugging investigation. Name which source states should be preserved for delayed reuse.", "planning", args, [])
    llm_plan_state_id = events[0]["state_id"]
    add(events)
    state_events, event, _ = tool.glob("**/*.py", workspace, "tool_01_glob")
    glob_state_id = event["new_state_id"]
    add(state_events + [event])
    add(call_llm(client, builder, workflow_id, agent_id, "llm_02_after_glob", "Choose the first two files to inspect and explain why they may interact.", "localization", args, [event["new_state_id"]]))

    state_events, event, first_file_state = tool.read(relevant[0], workspace, "tool_03_read_first")
    add(state_events + [event])
    add(call_llm(client, builder, workflow_id, agent_id, "llm_04_after_first_read", "Inspect this early file_context. Record invariants that may matter much later.", "localization", args, [first_file_state.state_id], include_recent=False, include_failures=False))

    second_path = relevant[1] if len(relevant) > 1 else relevant[0]
    state_events, event, second_file_state = tool.read(second_path, workspace, "tool_05_read_second")
    add(state_events + [event])
    events = call_llm(client, builder, workflow_id, agent_id, "llm_06_after_second_read", "Compare the second file with the earlier file, but do not restate the whole history.", "hypothesis", args, [second_file_state.state_id], include_recent=False, include_failures=False)
    llm_after_second_state_id = events[0]["state_id"]
    add(events)

    grep_pattern = "BUG" if opportunity else "return"
    state_events, event, _ = tool.grep(grep_pattern, workspace, "tool_07_grep_bug_markers")
    grep_bug_state_id = event["new_state_id"]
    add(state_events + [event])
    add(call_llm(client, builder, workflow_id, agent_id, "llm_08_hypothesis", "Propose the first minimal edit. Keep the early file out of this prompt unless explicitly referenced.", "hypothesis", args, [event["new_state_id"]], include_recent=True, include_failures=False))

    first_patch = metadata["first_patch"]
    state_events, event, first_edit_state = tool.edit_replace(Path(first_patch["path"]), workspace, first_patch["old"], first_patch["new"], "tool_09_first_edit")
    add(state_events + [event])
    add(call_llm(client, builder, workflow_id, agent_id, "llm_10_after_first_edit", "Review the first edit diff and predict what the tests will still complain about.", "editing", args, [first_edit_state.state_id], include_recent=False, include_failures=False))

    state_events, event, failure_state = tool.bash(metadata["expected_test_command"], workspace, "tool_11_pytest_after_first_edit", timeout=45, tool_name="Pytest")
    add(state_events + [event])
    add(call_llm(client, builder, workflow_id, agent_id, "llm_12_after_failure", "Summarize the failed test as a reusable state. Identify which earlier source state should be revisited after more inspection.", "diagnosis", args, [failure_state.state_id], include_recent=False, include_failures=False))

    if not opportunity:
        final_patch = metadata["final_patch"]
        state_events, event, final_edit_state = tool.edit_replace(Path(final_patch["path"]), workspace, final_patch["old"], final_patch["new"], "tool_13_control_final_edit")
        add(state_events + [event])
        add(call_llm(client, builder, workflow_id, agent_id, "llm_14_control_after_final_edit", "Review the local fix for this recency-control task.", "review", args, [final_edit_state.state_id], include_recent=True, include_failures=False))
        state_events, event, _ = tool.bash(metadata["expected_test_command"], workspace, "tool_15_control_pytest_final", timeout=45, tool_name="Pytest")
        add(state_events + [event])
        add(call_llm(client, builder, workflow_id, agent_id, "llm_16_control_final_summary", "Finalize the short local fix using only recent states.", "final", args, [final_edit_state.state_id, event["new_state_id"]], include_recent=True, include_failures=False))
        return finalize(event)

    third_path = relevant[2] if len(relevant) > 2 else second_path
    state_events, event, third_file_state = tool.read(third_path, workspace, "tool_13_read_third")
    add(state_events + [event])
    add(call_llm(client, builder, workflow_id, agent_id, "llm_14_after_third_read", "Use this fresh file_context to refine the diagnosis. Keep the failure state out for one turn.", "diagnosis", args, [third_file_state.state_id], include_recent=False, include_failures=False))

    state_events, event, _ = tool.grep("threshold", workspace, "tool_15_grep_thresholds")
    grep_threshold_state_id = event["new_state_id"]
    add(state_events + [event])
    add(call_llm(client, builder, workflow_id, agent_id, "llm_16_revisit_early_file", "Now explicitly revisit the early file_context after the delayed interval and connect it to the failed hypothesis.", "diagnosis", args, [first_file_state.state_id, event["new_state_id"]], include_recent=False, include_failures=False))

    add(call_llm(client, builder, workflow_id, agent_id, "llm_17_revisit_failure", "Reintroduce the delayed test_failure_summary and decide the corrected patch.", "editing", args, [failure_state.state_id, first_file_state.state_id], include_recent=False, include_failures=False))

    final_patch = metadata["final_patch"]
    state_events, event, final_edit_state = tool.edit_replace(Path(final_patch["path"]), workspace, final_patch["old"], final_patch["new"], "tool_18_final_edit")
    add(state_events + [event])
    add(call_llm(client, builder, workflow_id, agent_id, "llm_19_after_final_edit", "Review the corrected edit diff against the delayed failure and source context.", "review", args, [final_edit_state.state_id, failure_state.state_id], include_recent=False, include_failures=False))

    state_events, event, _ = tool.bash(metadata["expected_test_command"], workspace, "tool_20_pytest_final", timeout=45, tool_name="Pytest")
    add(state_events + [event])
    add(call_llm(client, builder, workflow_id, agent_id, "llm_21_final_summary", "Finalize with explicit state IDs for the early file, failure summary, first edit, final edit, and final test observation.", "final", args, [first_file_state.state_id, second_file_state.state_id, failure_state.state_id, first_edit_state.state_id, final_edit_state.state_id, event["new_state_id"]], include_recent=False, include_failures=False))

    add(call_llm(client, builder, workflow_id, agent_id, "llm_22_delayed_state_postmortem", "Revisit early plan/search states, the delayed failure summary, and both edit diffs after final verification. Explain which state was most valuable.", "postmortem", args, [glob_state_id, grep_bug_state_id, llm_plan_state_id, llm_after_second_state_id, first_file_state.state_id, failure_state.state_id, first_edit_state.state_id, final_edit_state.state_id], include_recent=False, include_failures=False))
    add(call_llm(client, builder, workflow_id, agent_id, "llm_23_cross_file_regression_guard", "Build a regression guard from the delayed search result, first, second, and third file_context states plus the delayed failure summary.", "postmortem", args, [grep_threshold_state_id, first_file_state.state_id, second_file_state.state_id, third_file_state.state_id, failure_state.state_id], include_recent=False, include_failures=False))

    return finalize(event)


def call_llm(client, builder, workflow_id, agent_id, step_id, instruction, phase, args, force_state_ids, include_recent=True, include_failures=True):
    segments = builder.assemble_segments(
        phase=phase,
        force_state_ids=force_state_ids,
        include_recent=include_recent,
        include_failures=include_failures,
    )
    messages = builder.messages_from_segments(segments, instruction)
    result = client.chat_completion(
        messages,
        workflow_id=workflow_id,
        agent_id=agent_id,
        step_id=step_id,
        input_segments=segments,
        temperature=0.0,
        top_p=1.0,
        max_tokens=getattr(args, "max_tokens", 32),
        stream=getattr(args, "stream", False),
    )
    record = result["record"]
    output = result["content"] or ("ERROR: " + json.dumps(result["error"], ensure_ascii=False) if result["error"] else "")
    state_event, llm_event = builder.make_llm_event(
        event_id=f"{step_id}_{workflow_id}",
        segments=segments,
        output_text=output,
        append_tokens=record.get("prompt_tokens"),
        output_tokens=record.get("completion_tokens"),
        timestamp_start=record.get("timestamp_start"),
        timestamp_end=record.get("timestamp_end"),
        phase=phase,
        metadata={"request_id": result["request_id"], "error": result["error"], "step_id": step_id},
    )
    return [state_event, llm_event]


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--server-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", default="local-qwen-32b")
    parser.add_argument("--tasks-dir", default="benchmarks/local_long_debug_tasks")
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--history-mode", choices=("selective_state", "full_history"), default="selective_state")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--raw-request-dir", default="traces/local_h100_long/raw_requests")
    parser.add_argument("--raw-tool-dir", default="traces/local_h100_long/raw_tools")
    parser.add_argument("--tool-artifact-dir", default="traces/local_h100_long/tool_outputs")
    parser.add_argument("--workspace-root", default="traces/local_h100_long/workspaces")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--stream", action="store_true")
