import hashlib
import re
from typing import Iterable, List, Optional

from .trace_schema import StateEvent


def estimate_tokens(text: object) -> int:
    if text is None:
        return 1
    text = str(text)
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return max(1, len(enc.encode(text)))
    except Exception:
        return max(1, int(len(re.findall(r"\S+", text)) * 1.3) or len(text) // 4 or 1)


def exact_token_hash(text: object, model_id: str = "unknown", tokenizer_id: str = "whitespace") -> str:
    payload = f"{model_id}\0{tokenizer_id}\0{'' if text is None else str(text)}"
    return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()


def stable_id(*parts: object, prefix: str = "state") -> str:
    payload = "\0".join("" if part is None else str(part) for part in parts)
    return f"{prefix}_{hashlib.sha1(payload.encode('utf-8', errors='ignore')).hexdigest()[:16]}"


def state_type_for_message(message: dict, idx: int = 0) -> str:
    role = str(message.get("role", "")).lower()
    content = str(message.get("content", ""))
    lowered = content.lower()
    if role in {"system", "developer"}:
        return "system_prefix"
    if role == "assistant":
        return "assistant_delta"
    if role in {"tool", "observation"}:
        return state_type_for_tool(message.get("name") or message.get("tool") or "", content, message.get("status"))
    if idx <= 1 or "problem statement" in lowered or "issue" in lowered:
        return "task_prefix"
    if "summary" in lowered or "memory" in lowered:
        return "summary_state"
    return "dialogue_delta"


def state_type_for_tool(tool: str, content: object, status: Optional[str] = None) -> str:
    tool_lower = str(tool or "").lower()
    text = str(content or "")
    lowered = text.lower()
    failed = str(status or "").lower() in {"fail", "failed", "error"} or "traceback" in lowered
    if any(name in tool_lower for name in ("read", "grep", "search", "find", "ls", "cat")):
        return "file_context"
    if any(name in tool_lower for name in ("edit", "write", "patch", "apply")) or "diff --git" in lowered:
        return "edit_diff"
    if any(name in tool_lower for name in ("pytest", "test", "unit")) and failed:
        return "test_failure_summary"
    if "bash" in tool_lower or "shell" in tool_lower or "cmd" in tool_lower:
        if failed:
            return "raw_error_log" if len(text) > 4000 else "failure_summary"
        return "tool_observation"
    if "web" in tool_lower or "browser" in tool_lower:
        return "web_result"
    if "subagent" in tool_lower or "delegate" in tool_lower:
        return "subagent_output"
    return "failure_summary" if failed else "tool_observation"


def semantic_key_for_tool(tool: str, content: object, metadata: Optional[dict] = None) -> Optional[str]:
    metadata = metadata or {}
    if metadata.get("path"):
        version = metadata.get("version", metadata.get("commit", "unknown"))
        line_range = metadata.get("range", metadata.get("line_range", "all"))
        return f"file:{metadata['path']}:{line_range}:{version}"
    if metadata.get("test"):
        error_type = metadata.get("error_type", metadata.get("status", "unknown"))
        return f"test:{metadata['test']}:{error_type}"
    text = str(content or "")
    match = re.search(r"([A-Za-z0-9_./\\-]+\.(py|js|ts|java|go|rs|cpp|h|md|txt))", text)
    if match:
        return f"file:{match.group(1)}:all:unknown"
    return None


def make_state_event(
    text: object,
    state_type: str,
    owner: str,
    source_id: str,
    ordinal: int,
    model_id: str = "unknown",
    tokenizer_id: str = "whitespace",
    semantic_key: Optional[str] = None,
    producer_event_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> dict:
    token_hash = exact_token_hash(text, model_id=model_id, tokenizer_id=tokenizer_id)
    base_key = semantic_key or token_hash
    state_id = stable_id(source_id.split(":turn", 1)[0].split(":tool", 1)[0], state_type, owner, base_key, prefix="state")
    return StateEvent(
        state_id=state_id,
        state_type=state_type,
        owner=owner,
        tokens=estimate_tokens(text),
        semantic_key=semantic_key,
        exact_token_hash=token_hash,
        producer_event_id=producer_event_id,
        metadata=metadata or {},
    ).to_dict()


def segment_messages(
    messages: Iterable[dict],
    source_id: str,
    agent: str = "agent_0",
    model_id: str = "unknown",
    tokenizer_id: str = "whitespace",
) -> List[dict]:
    states = []
    for idx, message in enumerate(messages or []):
        content = message.get("content", "")
        state_type = state_type_for_message(message, idx)
        owner = "shared" if state_type in {"system_prefix", "task_prefix", "shared_prefix"} else agent
        states.append(
            make_state_event(
                content,
                state_type,
                owner,
                source_id,
                idx,
                model_id=model_id,
                tokenizer_id=tokenizer_id,
                metadata={"role": message.get("role"), "raw": message},
            )
        )
    return states


def segment_tool_output(
    tool: str,
    output: object,
    source_id: str,
    agent: str,
    ordinal: int,
    status: Optional[str] = None,
    producer_event_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> dict:
    metadata = dict(metadata or {})
    state_type = state_type_for_tool(tool, output, status=status)
    semantic_key = semantic_key_for_tool(tool, output, metadata)
    return make_state_event(
        output,
        state_type,
        agent,
        source_id,
        ordinal,
        semantic_key=semantic_key,
        producer_event_id=producer_event_id,
        metadata={"tool": tool, "status": status, **metadata},
    )
