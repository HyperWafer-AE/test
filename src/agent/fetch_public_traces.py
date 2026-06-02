import argparse
import json
import shutil
import subprocess
from pathlib import Path


SOURCE_SPECS = {
    "agentlens": {
        "name": "AgentLens / OpenHands trajectories",
        "repo": "https://github.com/microsoft/code-agent-state-trajectories/",
        "local_candidates": ["traces/real/agentlens", "traces/downloaded/agentlens"],
        "manual_instructions": [
            "Check the repository README for dataset storage or LFS instructions.",
            "Download a small JSON/JSONL trajectory sample into traces/real/agentlens.",
            "Re-run trace_audit and build_real_trace_set after files are present.",
        ],
    },
    "swe_gym": {
        "name": "SWE-Gym / SWE-agent trajectories",
        "repo": None,
        "local_candidates": ["traces/real/swe_gym", "traces/downloaded/swe_gym"],
        "manual_instructions": [
            "Place SWE-Gym or SWE-agent trajectory JSON/JSONL files under traces/real/swe_gym.",
            "Use --trace-format swe_gym or auto for audit/build_real_trace_set.",
        ],
    },
    "codetracer": {
        "name": "CodeTracer / CodeTraceBench",
        "repo": None,
        "local_candidates": [
            "traces/real_raw/codetracebench",
            "traces/real_extracted/codetracebench",
            "traces/downloaded/codetracer",
            "traces/real/codetracebench_solved",
        ],
        "manual_instructions": [
            "Place CodeTraceBench .traj.json files under traces/real/codetracebench or raw .tar.zst archives under traces/real_raw/codetracebench.",
            "Run extract_trace_archives before build_real_trace_set.",
        ],
    },
}


def fetch_source(source: str, output_dir, max_files: int = 20, force_download: bool = False) -> dict:
    spec = SOURCE_SPECS[source]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "source": source,
        "source_name": spec["name"],
        "output_dir": str(output_dir),
        "repo": spec.get("repo"),
        "remote_status": "not_applicable",
        "download_status": "not_attempted",
        "num_files_downloaded": 0,
        "files_downloaded": 0,
        "local_files": [str(path) for path in _trace_like_files(output_dir, max_files=max_files)],
        "local_sample_files": [],
        "sample_candidate_class": "unknown",
        "manual_download_required": False,
        "manual_instructions": spec["manual_instructions"],
    }
    if report["local_files"]:
        report["download_status"] = "already_available"
        report["num_files_downloaded"] = len(report["local_files"])
        report["files_downloaded"] = report["num_files_downloaded"]
        report["local_sample_files"] = report["local_files"]
        report["sample_candidate_class"] = _candidate_class(report["local_files"])
        return report

    local_candidates = _local_candidate_files(spec.get("local_candidates", []), max_files=max_files)
    if local_candidates:
        copied = _copy_sample_files(local_candidates, output_dir, max_files=max_files)
        report["download_status"] = "local_sample_copied"
        report["num_files_downloaded"] = len(copied)
        report["files_downloaded"] = len(copied)
        report["local_files"] = copied
        report["local_sample_files"] = copied
        report["manual_download_required"] = False
        report["sample_candidate_class"] = _candidate_class(copied)
        return report

    repo = spec.get("repo")
    if not repo:
        report["manual_download_required"] = True
        report["download_status"] = "manual_only_source"
        return report

    report["remote_status"] = _git_ls_remote(repo)
    if report["remote_status"] != "accessible":
        report["manual_download_required"] = True
        report["download_status"] = "remote_unavailable"
        return report

    if not force_download:
        report["manual_download_required"] = True
        report["download_status"] = "remote_accessible_not_downloaded"
        report["notes"] = "Remote repository is reachable, but automatic bulk download is disabled to avoid LFS/large-data pulls."
        return report

    clone_dir = output_dir / "_repo_sample"
    if clone_dir.exists():
        shutil.rmtree(clone_dir)
    completed = subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", repo, str(clone_dir)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        report["manual_download_required"] = True
        report["download_status"] = "clone_failed"
        report["failure_reason"] = (completed.stderr or completed.stdout or "").strip()
        return report
    files = _trace_like_files(clone_dir, max_files=max_files)
    copied = []
    for idx, path in enumerate(files[:max_files]):
        target = output_dir / f"sample_{idx:03d}_{path.name}"
        shutil.copy2(path, target)
        copied.append(str(target))
    report["download_status"] = "downloaded_sample" if copied else "clone_succeeded_no_trace_files_found"
    report["num_files_downloaded"] = len(copied)
    report["files_downloaded"] = len(copied)
    report["local_files"] = copied
    report["local_sample_files"] = copied
    report["manual_download_required"] = len(copied) == 0
    report["sample_candidate_class"] = _candidate_class(copied)
    return report


def _git_ls_remote(repo: str) -> str:
    if shutil.which("git") is None:
        return "git_not_found"
    try:
        completed = subprocess.run(["git", "ls-remote", repo], check=False, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return "timeout"
    if completed.returncode == 0 and completed.stdout.strip():
        return "accessible"
    return "inaccessible: " + (completed.stderr or completed.stdout or "").strip()


def _trace_like_files(root: Path, max_files: int = 20):
    if not root.exists():
        return []
    suffixes = (".json", ".jsonl", ".traj.json", ".trajectory.json", ".tar.zst")
    files = []
    for path in root.rglob("*"):
        name = path.name.lower()
        if name in {"report.json", "profile.json", "summary.json"}:
            continue
        if path.is_file() and any(name.endswith(suffix) for suffix in suffixes):
            files.append(path)
        if len(files) >= max_files:
            break
    return sorted(files)


def _local_candidate_files(paths, max_files: int = 20):
    files = []
    seen = set()
    for item in paths:
        for path in _trace_like_files(Path(item), max_files=max_files):
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(path)
            if len(files) >= max_files:
                return files
    return files


def _copy_sample_files(files, output_dir: Path, max_files: int = 20):
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for idx, path in enumerate(files[:max_files]):
        target = output_dir / f"sample_{idx:03d}_{path.name}"
        if path.resolve() != target.resolve():
            shutil.copy2(path, target)
        copied.append(str(target))
    return copied


def _candidate_class(files) -> str:
    names = [str(path).lower() for path in files]
    if not names:
        return "none"
    if any(name.endswith(".tar.zst") for name in names):
        return "raw_source_requires_extraction"
    if any("traj.json" in name for name in names):
        return "trajectory_candidate_requires_audit"
    return "json_candidate_requires_audit"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Fetch or report public real-trace source availability.")
    parser.add_argument("--source", choices=sorted(SOURCE_SPECS), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-files", type=int, default=20)
    parser.add_argument("--report", required=True)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = fetch_source(args.source, args.output_dir, max_files=args.max_files, force_download=args.force_download)
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote public trace fetch report to {output}")
    print(
        "source={source} remote={remote} download={download} files={files}".format(
            source=args.source,
            remote=report["remote_status"],
            download=report["download_status"],
            files=report["num_files_downloaded"],
        )
    )


if __name__ == "__main__":
    main()
