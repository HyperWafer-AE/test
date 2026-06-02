import argparse
import json
import random
from pathlib import Path

from src.agent.local_trace.long_benchmark_tasks import ensure_local_long_debug_tasks
from src.agent.local_trace.long_horizon_coding_agent import add_common_args, load_long_tasks, run_long_workflow


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Collect long-horizon local H100 coding-agent traces.")
    add_common_args(parser)
    parser.add_argument("--output-dir", default="traces/local_h100_long/workflows")
    parser.add_argument("--num-tasks", type=int, default=40)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    ensure_local_long_debug_tasks(args.tasks_dir)
    tasks = load_long_tasks(args.tasks_dir)[: args.num_tasks]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for idx, task in enumerate(tasks):
        workflow_id = f"long_wf_{idx:03d}_{task['task_id']}"
        trace, result = run_long_workflow(task, workflow_id, args)
        path = out / f"{workflow_id}.json"
        path.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
        results.append({"workflow_id": workflow_id, "output": str(path), **result})
        print(f"{idx + 1}/{len(tasks)} {workflow_id} success={result['success']} events={result['num_events']}", flush=True)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "framework": "long_horizon_coding_agent",
                "history_mode": args.history_mode,
                "num_workflows": len(results),
                "workflows": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {len(results)} long workflow traces to {out}")


if __name__ == "__main__":
    main()
