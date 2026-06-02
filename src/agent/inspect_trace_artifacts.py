import argparse
import json
from collections import Counter
from pathlib import Path


TEXT_HINT_SUFFIXES = (".log", ".txt", ".md", ".patch", ".diff", ".out", ".err")
JSON_SUFFIXES = (".json", ".jsonl", ".traj.json", ".trajectory.json")


def inspect_artifacts(input_dir, output):
    input_dir = Path(input_dir)
    files = []
    summary = Counter()
    for path in sorted(input_dir.rglob("*")) if input_dir.exists() else []:
        if not path.is_file():
            continue
        if not _is_candidate(path):
            continue
        info = inspect_file(path)
        files.append(info)
        for key, value in info.items():
            if key.startswith("contains_") and value:
                summary[key] += 1
        summary[f"schema:{info['likely_schema']}"] += 1
    only_messages = files and all(
        item.get("likely_schema") in {"message_history_only", "report_or_metadata", "unknown_text"}
        for item in files
    )
    report = {
        "input_dir": str(input_dir),
        "num_candidate_files": len(files),
        "feature_counts": dict(summary),
        "richer_state_artifacts_found": any(
            item.get("likely_schema") in {"stateful_codetracer_artifact", "step_trajectory"}
            for item in files
        ),
        "message_history_only": bool(only_messages),
        "notes": (
            "Only message-history/report-like artifacts were found; local CodeTracer traces are not paper-usable."
            if only_messages
            else "Some structured artifacts may exist; inspect candidate files before claiming paper usability."
        ),
        "files": files,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def inspect_file(path: Path) -> dict:
    text = _read_prefix(path)
    payload = _decode_json(text, path)
    feature_text = text.lower()
    if payload is not None:
        feature_text = json.dumps(payload, ensure_ascii=True).lower()[:200000]
    info = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "contains_messages": _has_key(payload, "messages") or "messages" in feature_text,
        "contains_steps": any(_has_key(payload, key) for key in ("steps", "trajectory", "trajectory_steps", "history")),
        "contains_actions": any(token in feature_text for token in ("action", "command", "bash", "tool_input")),
        "contains_observations": any(token in feature_text for token in ("observation", "<output>", "<returncode>", "stdout", "stderr")),
        "contains_prompt": any(token in feature_text for token in ("prompt", "llm_input", "input_state")),
        "contains_state_tree": any(token in feature_text for token in ("state_tree", "state tree")),
        "contains_memory": any(token in feature_text for token in ("persistent_memory", "\"memory\"", "summary_state")),
        "contains_tool_calls": any(token in feature_text for token in ("tool_calls", "tool_call", "\"tool\"", "tool_name")),
        "contains_file_paths": any(token in feature_text for token in ("/src/", ".rs", ".py", "file_path", "path")),
        "contains_tests": any(token in feature_text for token in ("pytest", "cargo test", "test_result", "failed_count", "passed_count")),
    }
    info["likely_schema"] = _likely_schema(payload, info, path)
    return info


def _likely_schema(payload, info: dict, path: Path) -> str:
    name = path.name.lower()
    if name == "report.json":
        return "report_or_metadata"
    if info["contains_state_tree"] or info["contains_memory"]:
        return "stateful_codetracer_artifact"
    if info["contains_steps"] and (info["contains_actions"] or info["contains_observations"]):
        return "step_trajectory"
    if info["contains_messages"] and not info["contains_steps"]:
        return "message_history_only"
    if path.suffix.lower() in TEXT_HINT_SUFFIXES:
        return "text_artifact"
    return "unknown_json" if payload is not None else "unknown_text"


def _is_candidate(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in JSON_SUFFIXES + TEXT_HINT_SUFFIXES)


def _read_prefix(path: Path, limit: int = 200000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except Exception:
        return ""


def _decode_json(text: str, path: Path):
    if not path.name.lower().endswith(JSON_SUFFIXES):
        return None
    try:
        if path.suffix.lower() == ".jsonl":
            first = next((line for line in text.splitlines() if line.strip()), "")
            return json.loads(first) if first else None
        return json.loads(text)
    except Exception:
        return None


def _has_key(payload, key: str) -> bool:
    if isinstance(payload, dict):
        if key in payload:
            return True
        return any(_has_key(value, key) for value in payload.values())
    if isinstance(payload, list):
        return any(_has_key(item, key) for item in payload[:20])
    return False


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Inventory extracted trace artifacts and schema hints.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = inspect_artifacts(args.input_dir, args.output)
    print(
        "artifact_files={num_candidate_files} richer_state_artifacts_found={richer_state_artifacts_found}".format(
            **report
        )
    )


if __name__ == "__main__":
    main()
