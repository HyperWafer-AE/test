import argparse
import fnmatch
import json
import shutil
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


HF_SOURCE_SPECS = {
    "exgentic_otel": {
        "repo_id": "Exgentic/agent-llm-traces",
        "patterns": ["data/train-*.parquet"],
        "trace_format": "otel_spans",
        "notes": "OpenTelemetry LLM-agent spans; Parquet rows are converted to one JSON trace per row.",
    },
    "pagarsky_agent_trace": {
        "repo_id": "pagarsky/agent-trace",
        "patterns": ["datasets/*.jsonl", "data/agenttrace.parquet"],
        "trace_format": "generic_react_jsonl",
        "notes": "AgentTrace JSONL/parquet traces with tool-use metadata.",
    },
    "itbench_trajectories": {
        "repo_id": "ibm-research/ITBench-Trajectories",
        "patterns": ["*/session.jsonl", "*/*/session.jsonl", "*/*/*/session.jsonl", "*/*/*/*/session.jsonl"],
        "trace_format": "generic_react_jsonl",
        "notes": "ITBench SRE ReAct sessions; session.jsonl files are directly loadable as generic ReAct logs.",
    },
    "codetracebench_hf": {
        "repo_id": "NJU-LINK/CodeTraceBench",
        "patterns": ["swe_raw/*/*/*.traj.json"],
        "trace_format": "codetracer",
        "notes": "CodeTraceBench trajectory JSON files from Hugging Face. Full archives are intentionally excluded by default.",
    },
}


def fetch_hf_source(
    source: str,
    output_dir,
    max_files: int = 20,
    max_parquet_rows: int = 50,
    mirror_endpoint: str = "https://hf-mirror.com",
    api_endpoint: str = "https://huggingface.co",
    force: bool = False,
) -> dict:
    spec = HF_SOURCE_SPECS[source]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    siblings = _dataset_siblings(spec["repo_id"], api_endpoint=api_endpoint)
    selected = _select_files(siblings, spec["patterns"], max_files=max_files)
    report = {
        "source": source,
        "repo_id": spec["repo_id"],
        "trace_format": spec["trace_format"],
        "output_dir": str(output_dir),
        "api_endpoint": api_endpoint,
        "mirror_endpoint": mirror_endpoint,
        "patterns": spec["patterns"],
        "num_remote_files": len(siblings),
        "num_selected_files": len(selected),
        "downloaded_files": [],
        "converted_trace_files": [],
        "failed_files": [],
        "notes": spec["notes"],
    }

    for filename in selected:
        target = _target_path(output_dir, filename)
        try:
            if target.exists() and not force:
                status = "already_available"
            else:
                _download_dataset_file(
                    repo_id=spec["repo_id"],
                    filename=filename,
                    output_path=target,
                    mirror_endpoint=mirror_endpoint,
                )
                status = "downloaded"
            item = {
                "remote_file": filename,
                "local_file": str(target),
                "status": status,
                "bytes": target.stat().st_size if target.exists() else 0,
            }
            if filename.endswith(".parquet"):
                converted = _convert_parquet_rows(target, output_dir, max_rows=max_parquet_rows)
                item["converted_trace_files"] = converted
                report["converted_trace_files"].extend(converted)
            report["downloaded_files"].append(item)
        except Exception as exc:
            report["failed_files"].append(
                {
                    "remote_file": filename,
                    "local_file": str(target),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    report["num_downloaded_files"] = len(report["downloaded_files"])
    report["num_failed_files"] = len(report["failed_files"])
    report["num_converted_trace_files"] = len(report["converted_trace_files"])
    report["trace_like_files"] = [str(path) for path in _trace_like_files(output_dir)]
    return report


def _dataset_siblings(repo_id: str, api_endpoint: str) -> list:
    url = f"{api_endpoint.rstrip('/')}/api/datasets/{repo_id}"
    with _open_url(url) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return sorted(item.get("rfilename", "") for item in payload.get("siblings", []) if item.get("rfilename"))


def _select_files(siblings: list, patterns: list, max_files: int) -> list:
    selected = []
    seen = set()
    for pattern in patterns:
        for filename in siblings:
            if filename in seen:
                continue
            if fnmatch.fnmatch(filename, pattern):
                selected.append(filename)
                seen.add(filename)
                if len(selected) >= max_files:
                    return selected
    return selected


def _target_path(output_dir: Path, filename: str) -> Path:
    safe_parts = [part for part in Path(filename).parts if part not in {"..", "/", "\\"}]
    return output_dir.joinpath(*safe_parts)


def _download_dataset_file(repo_id: str, filename: str, output_path: Path, mirror_endpoint: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quoted = urllib.parse.quote(filename.replace("\\", "/"), safe="/")
    url = f"{mirror_endpoint.rstrip('/')}/datasets/{repo_id}/resolve/main/{quoted}"
    with _open_url(url) as response, output_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _open_url(url: str):
    current = url
    for _ in range(8):
        request = urllib.request.Request(current, headers={"User-Agent": "agent-wafer-fetch/1.0"})
        try:
            return urllib.request.urlopen(request, timeout=300)
        except urllib.error.HTTPError as exc:
            if exc.code in {301, 302, 303, 307, 308} and exc.headers.get("Location"):
                current = exc.headers["Location"]
                continue
            raise
    raise RuntimeError(f"Too many redirects while opening {url}")


def _convert_parquet_rows(path: Path, output_dir: Path, max_rows: int) -> list:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("Parquet conversion requires pyarrow in the active Python environment") from exc

    table = pq.read_table(path)
    rows = table.slice(0, min(max_rows, table.num_rows)).to_pylist()
    converted_dir = output_dir / "converted"
    converted_dir.mkdir(parents=True, exist_ok=True)
    converted = []
    for idx, row in enumerate(rows):
        target = converted_dir / f"{path.stem}_row{idx:04d}.json"
        target.write_text(json.dumps(row, ensure_ascii=True, default=str), encoding="utf-8")
        converted.append(str(target))
    return converted


def _trace_like_files(root: Path):
    if not root.exists():
        return []
    files = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}:
            name = path.name.lower()
            if name in {"report.json", "summary.json", "profile.json"}:
                continue
            files.append(path)
    return sorted(files)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Fetch public Hugging Face real-agent traces through a mirror endpoint.")
    parser.add_argument("--source", choices=sorted(HF_SOURCE_SPECS), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-files", type=int, default=20)
    parser.add_argument("--max-parquet-rows", type=int, default=50)
    parser.add_argument("--mirror-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--api-endpoint", default="https://huggingface.co")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = fetch_hf_source(
        args.source,
        args.output_dir,
        max_files=args.max_files,
        max_parquet_rows=args.max_parquet_rows,
        mirror_endpoint=args.mirror_endpoint,
        api_endpoint=args.api_endpoint,
        force=args.force,
    )
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote Hugging Face trace fetch report to {output}")
    print(
        "source={source} selected={selected} downloaded={downloaded} converted={converted} failed={failed}".format(
            source=args.source,
            selected=report["num_selected_files"],
            downloaded=report["num_downloaded_files"],
            converted=report["num_converted_trace_files"],
            failed=report["num_failed_files"],
        )
    )


if __name__ == "__main__":
    main()
