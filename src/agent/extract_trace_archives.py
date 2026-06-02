import argparse
import io
import json
import shutil
import subprocess
import tarfile
from collections import Counter
from pathlib import Path


TRACE_SUFFIXES = (".json", ".jsonl", ".traj.json", ".trajectory.json")


def extract_archives(input_dir, output_dir, max_archives=None, require_success=False) -> dict:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    archives = sorted(input_dir.glob("*.tar.zst"))
    if max_archives is not None:
        archives = archives[: int(max_archives)]

    extracted = []
    failures = []
    for archive in archives:
        target = output_dir / archive.stem.replace(".tar", "")
        target.mkdir(parents=True, exist_ok=True)
        result = _extract_one(archive, target)
        entry = {
            "archive": str(archive),
            "output_path": str(target),
            "method": result.get("method"),
            "status": result["status"],
            "candidate_trace_files": [str(path) for path in _candidate_trace_files(target)],
        }
        if result["status"] == "extracted":
            extracted.append(entry)
        else:
            entry["reason"] = result.get("reason", "unknown")
            failures.append(entry)
            if require_success:
                raise RuntimeError(f"Failed to extract {archive}: {entry['reason']}")

    candidate_files = []
    for path in _candidate_trace_files(output_dir):
        candidate_files.append(str(path))
    failure_reasons = Counter(item.get("reason", "unknown") for item in failures)
    return {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_archives_found": len(archives),
        "num_archives_extracted": len(extracted),
        "num_archives_failed": len(failures),
        "num_candidate_trace_files": len(candidate_files),
        "candidate_trace_files": candidate_files,
        "extracted_paths": extracted,
        "failures": failures,
        "failure_reasons": dict(failure_reasons),
    }


def _extract_one(archive: Path, target: Path) -> dict:
    zstd_result = _extract_with_python_zstandard(archive, target)
    if zstd_result["status"] == "extracted":
        return zstd_result
    tar_result = _extract_with_system_tar(archive, target)
    if tar_result["status"] == "extracted":
        return tar_result
    return {
        "status": "failed",
        "method": "python_zstandard_then_system_tar",
        "reason": f"python_zstandard={zstd_result.get('reason')}; system_tar={tar_result.get('reason')}",
    }


def _extract_with_python_zstandard(archive: Path, target: Path) -> dict:
    try:
        import zstandard as zstd
    except Exception as exc:
        return {"status": "failed", "method": "python_zstandard", "reason": f"zstandard unavailable: {exc}"}
    try:
        with archive.open("rb") as compressed:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(compressed) as reader:
                data = reader.read()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as handle:
            _safe_extract_tar(handle, target)
        return {"status": "extracted", "method": "python_zstandard"}
    except Exception as exc:
        return {"status": "failed", "method": "python_zstandard", "reason": str(exc)}


def _extract_with_system_tar(archive: Path, target: Path) -> dict:
    if shutil.which("tar") is None:
        return {"status": "failed", "method": "system_tar", "reason": "tar not found on PATH"}
    try:
        completed = subprocess.run(
            ["tar", "-xf", str(archive), "-C", str(target)],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        return {"status": "failed", "method": "system_tar", "reason": str(exc)}
    if completed.returncode == 0:
        return {"status": "extracted", "method": "system_tar"}
    stderr = (completed.stderr or completed.stdout or "").strip()
    return {"status": "failed", "method": "system_tar", "reason": stderr or f"tar exited {completed.returncode}"}


def _safe_extract_tar(handle: tarfile.TarFile, target: Path):
    target = target.resolve()
    for member in handle.getmembers():
        member_path = (target / member.name).resolve()
        if not str(member_path).startswith(str(target)):
            raise RuntimeError(f"Refusing unsafe archive member path: {member.name}")
    handle.extractall(target)


def _candidate_trace_files(root: Path):
    if not root.exists():
        return []
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name in {"report.json", "profile.json", "summary.json"}:
            continue
        if any(name.endswith(suffix) for suffix in TRACE_SUFFIXES):
            files.append(path)
    return sorted(files)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Extract public real-trace .tar.zst archives.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-archives", type=int)
    parser.add_argument("--report", required=True)
    parser.add_argument("--require-success", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = extract_archives(
        args.input_dir,
        args.output_dir,
        max_archives=args.max_archives,
        require_success=args.require_success,
    )
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote archive extraction report to {output}")
    print(
        "archives={found} extracted={extracted} candidates={candidates}".format(
            found=report["num_archives_found"],
            extracted=report["num_archives_extracted"],
            candidates=report["num_candidate_trace_files"],
        )
    )


if __name__ == "__main__":
    main()
