from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


STATE_TYPES = {
    "system_prefix",
    "task_prefix",
    "agent_role",
    "shared_prefix",
    "dialogue_delta",
    "assistant_delta",
    "tool_observation",
    "web_result",
    "file_context",
    "edit_diff",
    "test_failure_summary",
    "failure_summary",
    "raw_error_log",
    "summary_state",
    "subagent_output",
    "skill_core",
    "speculative_state",
}


@dataclass
class StateEvent:
    state_id: str
    state_type: str
    owner: str
    tokens: int
    kv_bytes: Optional[int] = None
    semantic_key: Optional[str] = None
    exact_token_hash: Optional[str] = None
    producer_event_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    type: str = "state"

    def to_dict(self) -> dict:
        return _drop_none(asdict(self))


@dataclass
class LLMEvent:
    event_id: str
    agent: str
    turn: int
    input_state_ids: List[str]
    append_tokens: int
    output_tokens: int
    new_state_id: str
    new_state_type: str = "assistant_delta"
    timestamp_start: Optional[float] = None
    timestamp_end: Optional[float] = None
    input_segments: Optional[List[dict]] = None
    phase: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    type: str = "llm"

    def to_dict(self) -> dict:
        return _drop_none(asdict(self))


@dataclass
class ToolEvent:
    event_id: str
    agent: str
    turn: int
    tool: str
    latency: int
    output_tokens: int
    status: str
    new_state_id: str
    new_state_type: str
    phase: Optional[str] = None
    timestamp_start: Optional[float] = None
    timestamp_end: Optional[float] = None
    metadata: Dict = field(default_factory=dict)
    type: str = "tool"

    def to_dict(self) -> dict:
        return _drop_none(asdict(self))


def _drop_none(data: dict) -> dict:
    return {key: value for key, value in data.items() if value is not None}


def is_normalized_event(event: dict) -> bool:
    if not isinstance(event, dict):
        return False
    event_type = event.get("type")
    if event_type == "state":
        return all(key in event for key in ("state_id", "state_type", "owner"))
    if event_type == "llm":
        return all(key in event for key in ("agent", "input_state_ids", "new_state_id"))
    if event_type == "tool":
        return all(key in event for key in ("agent", "tool", "new_state_id"))
    return False


def normalize_event_aliases(event: dict) -> dict:
    event = dict(event)
    metadata = dict(event.get("metadata") or {})
    if event.get("type") == "state":
        if "semantic_id" not in event and "semantic_key" in event:
            event["semantic_id"] = event["semantic_key"]
        if "exact_kv_id" not in event and "exact_token_hash" in event:
            event["exact_kv_id"] = event["exact_token_hash"]
        if "semantic_key" in event:
            metadata.setdefault("semantic_key", event["semantic_key"])
        if "exact_token_hash" in event:
            metadata.setdefault("exact_token_hash", event["exact_token_hash"])
    event["metadata"] = metadata
    return event


def validate_trace(trace: List[dict]) -> List[dict]:
    if not isinstance(trace, list):
        raise TypeError("A normalized trace must be a list of event dictionaries.")
    normalized = []
    for idx, event in enumerate(trace):
        if not is_normalized_event(event):
            raise ValueError(f"Event {idx} is not a normalized Agent-on-Wafer event: {event}")
        item = normalize_event_aliases(event)
        if item["type"] == "state":
            item["tokens"] = int(item.get("tokens", item.get("token_len", 1)))
        elif item["type"] == "llm":
            item["turn"] = int(item.get("turn", idx))
            item["append_tokens"] = int(item.get("append_tokens", 0))
            item["output_tokens"] = int(item.get("output_tokens", 1))
        elif item["type"] == "tool":
            item["turn"] = int(item.get("turn", idx))
            item["latency"] = int(item.get("latency", item.get("tool_latency", 0)))
            item["output_tokens"] = int(item.get("output_tokens", 1))
            item["status"] = item.get("status", "ok")
        normalized.append(item)
    return normalized
