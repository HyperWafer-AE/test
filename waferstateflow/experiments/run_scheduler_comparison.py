"""Run scheduler and placement comparison for one synthetic workflow."""

from __future__ import annotations

import argparse

from waferstateflow.hotness import initialize_static_hotness
from waferstateflow.redundancy_analyzer import analyze_redundancy
from waferstateflow.reporting import write_scheduler_outputs
from waferstateflow.simulator import BASELINES, SimulationConfig, run_all_baselines
from waferstateflow.wafer_topology import WaferTopology
from waferstateflow.workflow_generators import WORKFLOW_NAMES, generate_workflow


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", choices=WORKFLOW_NAMES, default="trading")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--num-branches", type=int, default=4)
    parser.add_argument("--num-rounds", type=int, default=2)
    parser.add_argument("--shared-state-size", type=int, default=4000)
    parser.add_argument("--unique-state-size", type=int, default=400)
    parser.add_argument("--dynamic-hot-state-probability", type=float, default=0.25)
    parser.add_argument("--output-token-mean", type=int, default=250)
    parser.add_argument("--mesh", default="32x32")
    parser.add_argument("--state-policy", choices=["static", "dynamic"], default="dynamic")
    parser.add_argument("--worker-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    config = vars(args)
    graph = generate_workflow(
        args.workflow,
        batch_size=args.batch_size,
        num_agents=args.num_agents,
        num_branches=args.num_branches,
        num_rounds=args.num_rounds,
        shared_state_size=args.shared_state_size,
        unique_state_size=args.unique_state_size,
        dynamic_hot_state_probability=args.dynamic_hot_state_probability,
        output_token_mean=args.output_token_mean,
        seed=args.seed,
    )
    initialize_static_hotness(graph)
    analysis = analyze_redundancy(graph)
    topology = WaferTopology.from_mesh(args.mesh)
    sim_config = SimulationConfig(worker_count=args.worker_count)
    runs = run_all_baselines(
        graph,
        topology=topology,
        config=sim_config,
        state_policy=args.state_policy,
        baselines=BASELINES,
    )
    write_scheduler_outputs(graph, analysis, runs, args.out, config)


if __name__ == "__main__":
    main()
