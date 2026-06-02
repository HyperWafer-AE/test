import argparse
import json
import subprocess
from pathlib import Path


SOURCE_SPECS = [
    {
        "source_name": "AgentLens / OpenHands trajectories",
        "expected_repo": "https://github.com/microsoft/code-agent-state-trajectories",
        "local_candidates": [
            "traces/real/agentlens",
            "traces/downloaded/agentlens",
            "traces/real/openhands",
            "traces/real/code-agent-state-trajectories",
        ],
        "supported_format": "agentlens",
        "notes": "Adapter expects OpenHands/AgentLens JSON or JSONL trajectories with actions, observations, and success metadata.",
    },
    {
        "source_name": "CodeTracer / CodeTraceBench",
        "expected_repo": None,
        "local_candidates": [
            "traces/real/codetracebench_solved",
            "traces/real/codetracebench",
            "traces/real_extracted/codetracebench",
            "traces/downloaded/codetracer",
            "traces/real_raw/codetracebench",
        ],
        "supported_format": "codetracer",
        "notes": "Local sample traces are accepted when trajectory JSON files are present; raw archives require extraction before evaluation.",
    },
    {
        "source_name": "SWE-Gym / SWE-agent trajectories",
        "expected_repo": None,
        "local_candidates": [
            "traces/real/swe_gym",
            "traces/downloaded/swe_gym",
            "traces/real/swegym",
            "traces/real/swe-agent",
            "traces/real/r2egym",
        ],
        "supported_format": "swe_gym",
        "notes": "Adapter supports SWE-Gym/SWE-agent style JSON or JSONL trajectory records when placed locally.",
    },
    {
        "source_name": "Generic local normalized traces",
        "expected_repo": None,
        "local_candidates": [
            "traces/real",
        ],
        "supported_format": "auto / normalized_json / normalized_jsonl / generic_react_jsonl",
        "notes": "Catches local normalized traces and best-effort generic ReAct-like JSON/JSONL logs.",
    },
]


def build_report(download_sample: bool = False, max_files: int = 10, require_data: bool = False) -> dict:
    sources = [_inspect_source(spec, download_sample=download_sample, max_files=max_files) for spec in SOURCE_SPECS]
    actual_files = sum(source["num_files_detected"] for source in sources)
    report = {
        "download_sample": bool(download_sample),
        "max_files": int(max_files),
        "actual_public_or_local_trace_files_found": actual_files > 0,
        "num_sources_available": sum(1 for source in sources if source["status"] == "available"),
        "sources": sources,
    }
    if require_data and not report["actual_public_or_local_trace_files_found"]:
        raise RuntimeError("No public/local trace data found and --require-data was set.")
    return report


def _inspect_source(spec: dict, download_sample: bool, max_files: int) -> dict:
    detected_files = []
    detected_paths = []
    for candidate in spec["local_candidates"]:
        path = Path(candidate)
        if not path.exists():
            continue
        files = _trace_like_files(path)
        if files:
            detected_paths.append(str(path))
            detected_files.extend(files[:max_files])

    status = "available" if detected_files else "requires_manual_download"
    notes = spec["notes"]
    remote_probe = None
    if not detected_files and spec.get("expected_repo"):
        remote_probe = _probe_remote(spec["expected_repo"]) if download_sample else "not_checked"
        if remote_probe == "accessible":
            status = "adapter_only"
            notes += " Remote repository is reachable, but sample download is intentionally not automatic for this large/unknown dataset."
        elif remote_probe not in {None, "not_checked"}:
            notes += f" Remote probe result: {remote_probe}."
    if not detected_files and not spec.get("expected_repo"):
        status = "adapter_only" if spec["source_name"] != "Generic local normalized traces" else "not_found"

    return {
        "source_name": spec["source_name"],
        "status": status,
        "expected_repo": spec.get("expected_repo"),
        "local_path": ";".join(detected_paths),
        "num_files_detected": len(detected_files),
        "sample_files": [str(path) for path in detected_files[:max_files]],
        "supported_format": spec["supported_format"],
        "notes": notes,
    }


def _trace_like_files(path: Path):
    if path.is_file():
        return [path] if path.suffix.lower() in {".json", ".jsonl", ".zst", ".tar"} else []
    files = [
        item
        for item in path.rglob("*")
        if item.is_file()
        and item.suffix.lower() in {".json", ".jsonl", ".zst", ".tar"}
        and item.name.lower() not in {"profile.json", "summary.json", "report.json"}
        and not item.name.lower().endswith("_state_stats.json")
    ]
    return sorted(files)


def _probe_remote(repo_url: str) -> str:
    try:
        proc = subprocess.run(
            ["git", "ls-remote", "--heads", repo_url],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception as exc:
        return f"probe_failed:{type(exc).__name__}:{exc}"
    if proc.returncode == 0 and proc.stdout.strip():
        return "accessible"
    detail = (proc.stderr or proc.stdout or "empty response").strip().splitlines()
    return f"not_accessible:{detail[0] if detail else 'unknown'}"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Report Agent-on-Wafer public trace source availability.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--download-sample", action="store_true")
    parser.add_argument("--max-files", type=int, default=10)
    parser.add_argument("--require-data", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = build_report(
        download_sample=args.download_sample,
        max_files=args.max_files,
        require_data=args.require_data,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote trace source report to {output}")
    print(f"available_sources={report['num_sources_available']} data_found={report['actual_public_or_local_trace_files_found']}")


if __name__ == "__main__":
    main()
