import json
from pathlib import Path
from typing import List, Optional

from .trace_normalizer import normalize_jsonl_records, normalize_payload


SUPPORTED_TRACE_FORMATS = {
    "auto",
    "normalized_jsonl",
    "normalized_json",
    "swe_gym",
    "codetracer",
    "agentlens",
    "otel_spans",
    "generic_react_jsonl",
}


def load_trace_file(path, trace_format: str = "auto", reject_accumulated_fallback: bool = False) -> List[dict]:
    path = Path(path)
    if trace_format not in SUPPORTED_TRACE_FORMATS:
        raise ValueError(f"Unsupported trace format: {trace_format}")
    if not path.exists():
        raise FileNotFoundError(path)
    fmt = detect_trace_format(path, trace_format)
    if path.suffix.lower() == ".jsonl":
        records = _read_jsonl(path)
        if fmt == "normalized_jsonl":
            trace = normalize_jsonl_records(records, trace_format=fmt, source_id=path.stem)
        else:
            trace = normalize_jsonl_records(records, trace_format=fmt, source_id=path.stem)
        _reject_if_accumulated_fallback(path, trace, reject_accumulated_fallback)
        return trace
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        trace = normalize_payload(payload, trace_format=fmt, source_id=path.stem)
        _reject_if_accumulated_fallback(path, trace, reject_accumulated_fallback)
        return trace
    raise ValueError(
        f"Unsupported file extension for {path}. Convert source trajectories to JSON/JSONL first."
    )


def load_trace_dir(
    path,
    trace_format: str = "auto",
    max_traces: Optional[int] = None,
    min_turns: int = 0,
    filter_success: str = "all",
    reject_accumulated_fallback: bool = False,
) -> List[List[dict]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    files = sorted(
        item
        for item in path.rglob("*")
        if item.is_file()
        and item.suffix.lower() in _allowed_suffixes(trace_format)
        and not _looks_like_generated_output(item)
    )
    traces = []
    for file_path in files:
        try:
            trace = load_trace_file(
                file_path,
                trace_format=trace_format,
                reject_accumulated_fallback=reject_accumulated_fallback,
            )
        except Exception as exc:
            if trace_format != "auto":
                raise
            print(f"Skipping {file_path}: {exc}")
            continue
        if len([event for event in trace if event.get("type") == "llm"]) < min_turns:
            continue
        if not _success_filter_ok(trace, filter_success):
            continue
        traces.append(trace)
        if max_traces is not None and len(traces) >= max_traces:
            break
    return traces


def _allowed_suffixes(trace_format: str) -> set:
    if trace_format == "normalized_jsonl" or trace_format == "generic_react_jsonl":
        return {".jsonl"}
    if trace_format == "normalized_json":
        return {".json"}
    return {".json", ".jsonl"}


def _looks_like_generated_output(path: Path) -> bool:
    name = path.name.lower()
    return name in {"profile.json", "summary.json", "report.json"} or name.endswith("_state_stats.json")


def detect_trace_format(path: Path, trace_format: str = "auto") -> str:
    if trace_format != "auto":
        return trace_format
    name = str(path).lower()
    if "swe-gym" in name or "swegym" in name or "r2egym" in name:
        return "swe_gym"
    if "codetrace" in name or "codetracer" in name:
        return "codetracer"
    if "agentlens" in name or "openhands" in name:
        return "agentlens"
    if "exgentic" in name or "otel" in name or "opentelemetry" in name:
        return "otel_spans"
    if path.suffix.lower() == ".jsonl":
        first = _first_jsonl(path)
        if isinstance(first, dict) and first.get("type") in {"state", "llm", "tool"}:
            return "normalized_jsonl"
        return _detect_payload_format(first)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(payload, list) and payload and isinstance(payload[0], dict) and payload[0].get("type"):
            return "normalized_json"
        return _detect_payload_format(payload)
    return "generic_react_jsonl"


def _detect_payload_format(payload) -> str:
    text = json.dumps(payload, ensure_ascii=True).lower()[:4000]
    if "swe-gym" in text or "r2e" in text or "swe_agent" in text:
        return "swe_gym"
    if "codetrace" in text or "stage_label" in text or "failure localization" in text:
        return "codetracer"
    if "agentlens" in text or "openhands" in text:
        return "agentlens"
    if "gen_ai.input.messages" in text or "span_id" in text or "opentelemetry" in text:
        return "otel_spans"
    return "generic_react_jsonl"


def _read_jsonl(path: Path) -> List[dict]:
    records = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _first_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                return json.loads(line)
    return None


def _success_filter_ok(trace: List[dict], filter_success: str) -> bool:
    if filter_success == "all":
        return True
    values = []
    for event in trace:
        metadata = event.get("metadata") or {}
        if "trace_success" in metadata:
            values.append(metadata["trace_success"])
    if not values:
        return True
    success = any(bool(value) for value in values)
    return success if filter_success == "success" else not success


def _reject_if_accumulated_fallback(path: Path, trace: List[dict], reject: bool):
    if not reject:
        return
    llm_events = [event for event in trace if event.get("type") == "llm"]
    if not llm_events:
        return
    fallback = [
        event
        for event in llm_events
        if (event.get("metadata") or {}).get("prompt_reconstruction") == "accumulated_fallback"
    ]
    if len(fallback) / len(llm_events) > 0.5:
        raise ValueError(
            f"{path} uses accumulated_fallback prompt reconstruction for "
            f"{len(fallback)}/{len(llm_events)} LLM events"
        )


def expected_source_format_notes() -> dict:
    return {
        "swe_gym": "JSON/JSONL trajectory records with messages/steps, actions, observations, status, and optional instance_id.",
        "codetracer": "CodeTraceBench-style JSON/JSONL records with step trajectory, tool/action fields, observations, and stage labels if present.",
        "agentlens": "OpenHands/AgentLens JSON/JSONL records with actions, observations, tool names, and success metadata.",
        "otel_spans": "OpenTelemetry-style agent traces with spans, gen_ai.input.messages, gen_ai.output.messages, token usage, and optional tool messages.",
        "generic_react_jsonl": "One JSON object per step, or a JSON object/list containing messages plus tool/action/observation fields.",
    }
