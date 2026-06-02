import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from .trace_audit import audit_single_trace, audit_traces
from .trace_loader import load_trace_file


TRACE_SUFFIXES = (".json", ".jsonl", ".traj.json", ".trajectory.json")


def build_trace_set(
    input_dirs,
    output_dir,
    trace_format: str = "auto",
    min_turns: int = 4,
    max_traces: int = 100,
    audit_output=None,
    manifest_path=None,
) -> dict:
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = _candidate_files([Path(path) for path in input_dirs], max_files=max_traces)
    loaded_traces = []
    manifest = []
    high_quality = []
    failures = []

    for source_idx, path in enumerate(candidates):
        try:
            trace = load_trace_file(path, trace_format=trace_format)
        except Exception as exc:
            failures.append({"source_file": str(path), "reason": str(exc)})
            manifest.append({"source_file": str(path), "quality_label": "load_failed", "failure_reason": str(exc)})
            continue
        loaded_traces.append(trace)
        report = audit_single_trace(
            trace,
            trace_idx=source_idx,
            min_turns=min_turns,
            allow_reconstructed_full_history=False,
            delayed_reuse_k=8,
        )
        entry = {
            "source_file": str(path),
            "quality_label": report["quality_label"],
            "exclusion_reasons": report["exclusion_reasons"],
            "delayed_reuse_ratio": report["reuse_quality"]["delayed_reuse_ratio"],
            "LRU_regret_candidate_count": report["reuse_quality"]["LRU_regret_candidate_count"],
            "cross_agent_reuse_ratio": report["reuse_quality"]["cross_agent_reuse_ratio"],
            "prompt_reconstruction_quality": report["prompt_reconstruction_quality"],
            "state_type_distribution": report["state_quality"]["state_type_distribution"],
        }
        if report["quality_label"] == "high":
            output_file = output_dir / f"{len(high_quality):04d}_{_safe_stem(path)}.normalized.json"
            output_file.write_text(json.dumps(trace, indent=2), encoding="utf-8")
            entry["normalized_output_file"] = str(output_file)
            high_quality.append(entry)
        manifest.append(entry)

    audit = audit_traces(loaded_traces, min_turns=min_turns, allow_reconstructed_full_history=False, delayed_reuse_k=8)
    audit["num_candidate_files"] = len(candidates)
    audit["num_load_failures"] = len(failures)
    audit["load_failures"] = failures

    if audit_output:
        audit_path = Path(audit_output)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if manifest_path:
        manifest_file = Path(manifest_path)
        manifest_file.parent.mkdir(parents=True, exist_ok=True)
        manifest_file.write_text(json.dumps({"traces": manifest}, indent=2), encoding="utf-8")

    missing_report_path = None
    if not high_quality:
        missing_report_path = Path("agent_results/high_quality_trace_missing_report.json")
        missing_report_path.parent.mkdir(parents=True, exist_ok=True)
        missing_report = _missing_report(input_dirs, candidates, manifest, failures)
        missing_report_path.write_text(json.dumps(missing_report, indent=2), encoding="utf-8")

    return {
        "num_candidate_files": len(candidates),
        "num_loaded_traces": len(loaded_traces),
        "num_high_quality_traces": len(high_quality),
        "output_dir": str(output_dir),
        "audit_output": str(audit_output) if audit_output else "",
        "manifest": str(manifest_path) if manifest_path else "",
        "missing_report": str(missing_report_path) if missing_report_path else "",
    }


def _candidate_files(input_dirs, max_files: int):
    files = []
    seen = set()
    for root in input_dirs:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            name = path.name.lower()
            if name in {"report.json", "profile.json", "summary.json"}:
                continue
            if not any(name.endswith(suffix) for suffix in TRACE_SUFFIXES):
                continue
            if "agent_results" in path.parts:
                continue
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(path)
            if len(files) >= max_files:
                return sorted(files)
    return sorted(files)


def _safe_stem(path: Path) -> str:
    stem = path.name
    for suffix in (".normalized.json", ".traj.json", ".trajectory.json", ".jsonl", ".json"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    keep = []
    for char in stem:
        keep.append(char if char.isalnum() or char in {"-", "_"} else "_")
    return "".join(keep)[:120] or "trace"


def _missing_report(input_dirs, candidates, manifest, failures) -> dict:
    reason_counts = Counter()
    for item in manifest:
        for reason in item.get("exclusion_reasons", []):
            reason_counts[reason] += 1
        if item.get("quality_label") == "load_failed":
            reason_counts["load_failed"] += 1
    return {
        "status": "missing_high_quality_traces",
        "input_dirs": [str(path) for path in input_dirs],
        "num_candidate_files": len(candidates),
        "num_load_failures": len(failures),
        "failure_reasons": dict(reason_counts),
        "source_reports": _source_report_summary(),
        "manual_steps_required": [
            "Extract any raw .tar.zst archives with python -m src.agent.extract_trace_archives.",
            "Obtain traces that expose per-step prompt context, state tree, persistent memory, or exact input state ids.",
            "Avoid message-history-only transcripts for paper claims because they are full-history reconstructions.",
            "Place usable JSON/JSONL traces under traces/real or traces/real_extracted and rerun build_real_trace_set.",
        ],
        "candidate_summaries": manifest,
    }


def _source_report_summary() -> dict:
    trace_sources = _load_optional_json(Path("agent_results/trace_sources_report.json"))
    archive = _load_optional_json(Path("agent_results/archive_extraction_report.json"))
    agentlens = _load_optional_json(Path("agent_results/fetch_agentlens_report.json"))
    return {
        "trace_sources_report": {
            "path": "agent_results/trace_sources_report.json",
            "actual_public_or_local_trace_files_found": trace_sources.get("actual_public_or_local_trace_files_found"),
            "sources": [
                {
                    "source_name": item.get("source_name"),
                    "status": item.get("status"),
                    "num_files_detected": item.get("num_files_detected"),
                    "local_path": item.get("local_path"),
                }
                for item in trace_sources.get("sources", [])
            ],
        },
        "archive_extraction_report": {
            "path": "agent_results/archive_extraction_report.json",
            "num_archives_found": archive.get("num_archives_found"),
            "num_archives_extracted": archive.get("num_archives_extracted"),
            "num_candidate_trace_files": archive.get("num_candidate_trace_files"),
            "failure_reasons": archive.get("failure_reasons"),
        },
        "fetch_agentlens_report": {
            "path": "agent_results/fetch_agentlens_report.json",
            "remote_status": agentlens.get("remote_status"),
            "download_status": agentlens.get("download_status"),
            "manual_download_required": agentlens.get("manual_download_required"),
            "num_files_downloaded": agentlens.get("num_files_downloaded"),
        },
    }


def _load_optional_json(path: Path) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build a paper-usable real trace set if high-quality traces exist.")
    parser.add_argument("--input-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trace-format", default="auto")
    parser.add_argument("--min-turns", type=int, default=4)
    parser.add_argument("--max-traces", type=int, default=100)
    parser.add_argument("--audit-output", required=True)
    parser.add_argument("--manifest", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = build_trace_set(
        args.input_dirs,
        args.output_dir,
        trace_format=args.trace_format,
        min_turns=args.min_turns,
        max_traces=args.max_traces,
        audit_output=args.audit_output,
        manifest_path=args.manifest,
    )
    print(
        "candidate_files={num_candidate_files} loaded={num_loaded_traces} high_quality={num_high_quality_traces}".format(
            **report
        )
    )
    if report["missing_report"]:
        print(f"Wrote missing high-quality trace report to {report['missing_report']}")


if __name__ == "__main__":
    main()
