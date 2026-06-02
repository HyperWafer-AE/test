import argparse
import json
import random
import shutil
import time
from pathlib import Path

from src.agent.local_serving.openai_client_wrapper import LocalOpenAIClientWrapper
from src.agent.local_trace.benchmark_tasks import ensure_local_debug_tasks
from src.agent.local_trace.prompt_state_builder import PromptStateBuilder
from src.agent.local_trace.tool_wrappers import ToolLogger
from src.agent.trace_schema import validate_trace


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run local coding-agent workflows and collect explicit state traces.")
    parser.add_argument("--server-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", default="local-qwen3-27b")
    parser.add_argument("--framework", default="local_react_coding_agent")
    parser.add_argument("--benchmark", default="local_debug_tasks")
    parser.add_argument("--tasks-dir", default="benchmarks/local_debug_tasks")
    parser.add_argument("--output-dir", default="traces/local_h100/workflows")
    parser.add_argument("--num-tasks", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--history-mode", choices=("selective_state", "full_history"), default="selective_state")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model-path", default="/data1/dg123_data/Qwen-32B")
    parser.add_argument("--workspace-root", default="traces/local_h100/workspaces")
    parser.add_argument("--raw-request-dir", default="traces/local_h100/raw_requests")
    parser.add_argument("--raw-tool-dir", default="traces/local_h100/raw_tools")
    parser.add_argument("--stream", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    ensure_local_debug_tasks(args.tasks_dir)
    tasks = load_tasks(args.tasks_dir)[: args.num_tasks]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, task in enumerate(tasks):
        workflow_id = f"wf_{index:03d}_{task['task_id']}"
        trace, result = run_workflow(task, workflow_id, args)
        out = output_dir / f"{workflow_id}.json"
        out.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
        results.append({"workflow_id": workflow_id, "output": str(out), **result})
    manifest = {"framework": args.framework, "history_mode": args.history_mode, "num_workflows": len(results), "workflows": results}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {len(results)} workflow traces to {output_dir}")
    return results


def load_tasks(tasks_dir: str):
    ensure_local_debug_tasks(tasks_dir)
    manifest = json.loads((Path(tasks_dir) / "manifest.json").read_text(encoding="utf-8"))
    return manifest["tasks"]


def run_workflow(task: dict, workflow_id: str, args, global_recorder=None):
    agent_id = "agent_0"
    task_dir = Path(task["path"])
    workspace = Path(args.workspace_root) / workflow_id / "repo"
    if workspace.exists():
        shutil.rmtree(workspace)
    shutil.copytree(task_dir / "repo", workspace)
    issue = (task_dir / "issue.md").read_text(encoding="utf-8")
    metadata = json.loads((task_dir / "metadata.json").read_text(encoding="utf-8"))
    builder = PromptStateBuilder(workflow_id, agent_id, args.history_mode, model_path=args.model_path)
    tool = ToolLogger(workflow_id, agent_id, builder, raw_dir=args.raw_tool_dir)
    client = LocalOpenAIClientWrapper(args.server_url, args.model, output_dir=args.raw_request_dir, server_backend="sglang")
    trace = []
    trace.extend(builder.bootstrap(issue, agent_role=args.framework))
    start = time.time()

    source_state_id = None
    failure_state_id = None
    edit_state_id = None
    phases = [
        ("plan", "Summarize the bug and list the first inspection actions.", []),
        ("after_glob", "Choose the most relevant source file from the file listing.", []),
        ("after_read", "Inspect the source snippet and identify the likely faulty line.", []),
        ("before_test", "Predict what the failing unit test will show.", []),
        ("after_failure", "Use the failure summary to propose the minimal patch.", []),
        ("before_patch", "Confirm the edit target and explain why old behavior is wrong.", []),
        ("after_patch", "Review the patch diff and decide which test to run.", []),
        ("final", "Use the delayed file/test/edit states to write a final status summary.", []),
    ]

    trace.extend(call_llm(client, builder, workflow_id, agent_id, "llm_plan", phases[0][1], "planning", args, phases[0][2]))

    state_events, event, _ = tool.glob("**/*.py", workspace, "tool_glob")
    trace.extend(state_events + [event])
    trace.extend(call_llm(client, builder, workflow_id, agent_id, "llm_after_glob", phases[1][1], "localization", args, [event["new_state_id"]]))

    state_events, event, file_state = tool.read(Path(metadata["patch"]["path"]), workspace, "tool_read_source")
    source_state_id = file_state.state_id
    trace.extend(state_events + [event])
    trace.extend(call_llm(client, builder, workflow_id, agent_id, "llm_after_read", phases[2][1], "localization", args, [source_state_id]))

    state_events, event, _ = tool.grep(metadata["patch"]["old"].split("(", 1)[0].strip().split()[0], workspace, "tool_grep")
    trace.extend(state_events + [event])
    trace.extend(call_llm(client, builder, workflow_id, agent_id, "llm_before_test", phases[3][1], "testing", args, [source_state_id]))

    state_events, event, failure_state = tool.bash(metadata["expected_command"], workspace, "tool_pytest_fail", timeout=30, tool_name="Pytest")
    failure_state_id = failure_state.state_id
    trace.extend(state_events + [event])
    trace.extend(call_llm(client, builder, workflow_id, agent_id, "llm_after_failure", phases[4][1], "diagnosis", args, [failure_state_id]))

    trace.extend(call_llm(client, builder, workflow_id, agent_id, "llm_before_patch", phases[5][1], "editing", args, [failure_state_id]))
    patch = metadata["patch"]
    state_events, event, edit_state = tool.edit_replace(Path(patch["path"]), workspace, patch["old"], patch["new"], "tool_edit")
    edit_state_id = edit_state.state_id
    trace.extend(state_events + [event])
    trace.extend(call_llm(client, builder, workflow_id, agent_id, "llm_after_patch", phases[6][1], "review", args, [edit_state_id]))

    state_events, event, _ = tool.bash(metadata["expected_command"], workspace, "tool_pytest_final", timeout=30, tool_name="Pytest")
    trace.extend(state_events + [event])
    trace.extend(call_llm(client, builder, workflow_id, agent_id, "llm_final", phases[7][1], "final", args, [source_state_id, failure_state_id, edit_state_id]))

    success = event["status"] == "ok"
    for item in trace:
        item.setdefault("metadata", {})
        item["metadata"].setdefault("workflow_id", workflow_id)
        item["metadata"].setdefault("framework", args.framework)
        item["metadata"].setdefault("trace_success", success)
        item["metadata"].setdefault("benchmark", args.benchmark)
    validate_trace(trace)
    return trace, {"success": success, "wall_clock_s": time.time() - start, "num_events": len(trace)}


def call_llm(client, builder, workflow_id, agent_id, step_id, instruction, phase, args, force_state_ids):
    segments = builder.assemble_segments(phase=phase, force_state_ids=force_state_ids, include_recent=True, include_failures=True)
    messages = builder.messages_from_segments(segments, instruction)
    result = client.chat_completion(
        messages,
        workflow_id=workflow_id,
        agent_id=agent_id,
        step_id=step_id,
        input_segments=segments,
        temperature=0.0,
        top_p=1.0,
        max_tokens=48,
        stream=args.stream,
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


if __name__ == "__main__":
    main()

