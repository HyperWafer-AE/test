import argparse
import json
from pathlib import Path

from .backend_contract import (
    add_dependency,
    emit_communication_event,
    emit_external_wait_event,
    emit_fixed_compute_event,
)
from .experiment import build_hardware
from .placement import available_compute_locations, node_of_module
from event_driver import collective_event_driver


def _locations(hardware):
    compute = available_compute_locations(hardware)
    if not compute:
        raise ValueError("No tensorcore compute locations available.")
    nodes = sorted(getattr(hardware, "nodes_set", []))
    if len(nodes) < 2:
        node0 = node_of_module(compute[0])
        return compute[0], node0, node0
    return compute[0], tuple(nodes[0]), tuple(nodes[1])


def _run_case(name, cfg, topology, builder, checker):
    hardware = build_hardware(cfg, topology)
    compute_loc, node0, node1 = _locations(hardware)
    events = {}
    builder(events, compute_loc, node0, node1)
    total, pure_comp, pure_comm = collective_event_driver(events, hardware)
    passed, reason = checker(events, total, pure_comp, pure_comm)
    return {
        "test_name": name,
        "passed": bool(passed),
        "reason": reason,
        "total_cycles": int(total),
        "pure_comp_cycles": int(pure_comp),
        "pure_comm_cycles": int(pure_comm),
    }


def run_backend_smoke(cfg: str, topology: str):
    tests = []

    def fixed_only(events, compute_loc, node0, node1):
        emit_fixed_compute_event(events, 0, "fixed_only", "tensorcore", compute_loc, 100, {"test": "fixed_only"})

    tests.append(
        _run_case(
            "fixed_compute_only",
            cfg,
            topology,
            fixed_only,
            lambda events, total, comp, comm: (comp > 0 and comm == 0 and total >= 100, "fixed compute should consume compute cycles only"),
        )
    )

    def comm_only(events, compute_loc, node0, node1):
        emit_communication_event(events, 0, "comm_only", node0, node1, 4096, {"test": "comm_only"})

    tests.append(
        _run_case(
            "communication_only",
            cfg,
            topology,
            comm_only,
            lambda events, total, comp, comm: (comm > 0 and comp == 0 and total > 0, "communication should consume link cycles only"),
        )
    )

    def wait_then_compute(events, compute_loc, node0, node1):
        wait = emit_external_wait_event(events, 0, "tool_wait", 50, {"test": "wait_then_compute"})
        comp = emit_fixed_compute_event(events, 1, "after_wait", "tensorcore", compute_loc, 100, {"test": "wait_then_compute"})
        add_dependency(events, wait.event_tag, comp)

    tests.append(
        _run_case(
            "external_wait_then_fixed_compute",
            cfg,
            topology,
            wait_then_compute,
            lambda events, total, comp, comm: (
                events[1].start_time >= events[0].end_time and total >= 150,
                "compute should start after external wait dependency",
            ),
        )
    )

    def comm_then_compute(events, compute_loc, node0, node1):
        comm = emit_communication_event(events, 0, "kv_move", node0, node1, 4096, {"test": "comm_then_compute"})
        comp = emit_fixed_compute_event(events, 1, "after_comm", "tensorcore", compute_loc, 100, {"test": "comm_then_compute"})
        add_dependency(events, comm.event_tag, comp)

    tests.append(
        _run_case(
            "communication_then_fixed_compute",
            cfg,
            topology,
            comm_then_compute,
            lambda events, total, comp, comm: (
                events[1].start_time >= events[0].end_time and comm > 0,
                "compute should depend on communication completion",
            ),
        )
    )

    def unused_prefetch(events, compute_loc, node0, node1):
        emit_communication_event(events, 0, "unused_prefetch", node0, node1, 4096, {"reason": "prefetch", "blocking": False})
        emit_fixed_compute_event(events, 1, "independent_compute", "tensorcore", compute_loc, 100, {"test": "unused_prefetch"})

    tests.append(
        _run_case(
            "unused_prefetch_does_not_block_compute",
            cfg,
            topology,
            unused_prefetch,
            lambda events, total, comp, comm: (
                events[1].start_time == 0 and 1 not in events[0].issue_set,
                "unused prefetch should not be a dependency of compute",
            ),
        )
    )
    return tests


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run Agent-on-Wafer BusyBarn backend smoke tests.")
    parser.add_argument("--cfg", default="src/platform/cfgs/wamis_hd_distributed.cfg")
    parser.add_argument("--topology", choices=("wamis", "tlm"), default="wamis")
    parser.add_argument("--output", default="agent_results/backend_smoke.json")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    tests = run_backend_smoke(args.cfg, args.topology)
    report = {
        "status": "passed" if all(test["passed"] for test in tests) else "failed",
        "backend_mode": "event-level BusyBarn backend with fixed-duration analytical LLM compute events",
        "tests": tests,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"backend_smoke status={report['status']} output={output}")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
