import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _maybe_reexec_in_venv():
    repo_root = _repo_root()
    venv_python = repo_root / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        return
    if Path(sys.executable).resolve() == venv_python.resolve():
        return
    try:
        import numpy  # noqa: F401
    except ModuleNotFoundError:
        os.execv(str(venv_python), [str(venv_python), "-m", "src.agent.comm_benchmark"] + sys.argv[1:])


_maybe_reexec_in_venv()


def _setup_paths():
    repo_root = _repo_root()
    paths = [
        repo_root,
        repo_root / "src",
        repo_root / "utils",
        repo_root / "src" / "scheduling",
        repo_root / "src" / "scheduling" / "communication" / "topology",
        repo_root / "src" / "backend" / "analytical",
    ]
    for path in paths:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_setup_paths()

from event_driver import collective_event_driver
from src.scheduling.event_notation import communication_notation

from .cost_model import estimate_comm_cycles_backend
from .experiment import build_hardware
from .placement import node_distance


def _node_pairs(hardware, count: int = 4):
    nodes = sorted(getattr(hardware, "nodes_set", []))
    if len(nodes) < 2:
        return []
    source = nodes[0]
    ranked = sorted(
        [node for node in nodes if node != source],
        key=lambda node: (node_distance(hardware, source, node), node),
    )
    picks = []
    for idx in [0, max(0, len(ranked) // 4), max(0, len(ranked) // 2), len(ranked) - 1]:
        target = ranked[idx]
        pair = (source, target)
        if pair not in picks:
            picks.append(pair)
    return picks[:count]


def run_benchmark(args):
    probe_hardware = build_hardware(args.cfg, args.topology)
    pairs = _node_pairs(probe_hardware)
    byte_sizes = [1 * 1024 * 1024, 16 * 1024 * 1024, 128 * 1024 * 1024]
    records = []
    ratios = []

    for source, target in pairs:
        for byte_size in byte_sizes:
            hardware = build_hardware(args.cfg, args.topology)
            state = SimpleNamespace(kv_bytes=byte_size)
            event = communication_notation(
                comm_name="comm_benchmark",
                comm_tag=0,
                source_location=source,
                target_location=target,
                comm_bytes=byte_size,
            )
            actual_cycles, _, _ = collective_event_driver({0: event}, hardware)
            estimated_cycles = estimate_comm_cycles_backend(state, source, target, hardware)
            ratio = estimated_cycles / actual_cycles if actual_cycles else None
            if ratio is not None:
                ratios.append(ratio)
            records.append(
                {
                    "source": list(source),
                    "target": list(target),
                    "distance": node_distance(hardware, source, target),
                    "bytes": byte_size,
                    "estimated_cycles": int(estimated_cycles),
                    "actual_cycles": int(actual_cycles),
                    "estimated_actual_ratio": ratio,
                }
            )

    median_ratio = statistics.median(ratios) if ratios else None
    return {
        "cfg": args.cfg,
        "topology": args.topology,
        "median_estimated_actual_ratio": median_ratio,
        "within_2x": bool(median_ratio is not None and 0.5 <= median_ratio <= 2.0),
        "records": records,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Microbenchmark backend communication cost estimates.")
    parser.add_argument("--cfg", default="src/platform/cfgs/wamis_hd_distributed.cfg")
    parser.add_argument("--topology", choices=("wamis", "tlm"), default="wamis")
    parser.add_argument("--output", default="agent_results/comm_benchmark.json")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = run_benchmark(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        "Wrote communication benchmark to {path}; median estimated/actual ratio={ratio}".format(
            path=output,
            ratio=result["median_estimated_actual_ratio"],
        )
    )


if __name__ == "__main__":
    main()
