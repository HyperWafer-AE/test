import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from src.agent.trace_schema import LLMEvent, StateEvent


STATE_TYPES = {
    "system_prefix",
    "task_prefix",
    "agent_role",
    "file_context",
    "edit_diff",
    "test_failure_summary",
    "failure_summary",
    "raw_error_log",
    "tool_observation",
    "summary_state",
    "assistant_delta",
    "subagent_output",
}


@dataclass
class PromptState:
    state_id: str
    state_type: str
    owner: str
    text: str
    tokens: int
    text_hash: str
    semantic_key: str
    producer_event_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    created_turn: int = 0
    last_used_turn: int = -1
    use_count: int = 0

    def segment(self, include_text: bool = True) -> dict:
        data = {
            "state_id": self.state_id,
            "state_type": self.state_type,
            "owner": self.owner,
            "tokens": self.tokens,
            "text_hash": self.text_hash,
            "semantic_key": self.semantic_key,
            "producer_event_id": self.producer_event_id,
            "metadata": self.metadata,
        }
        if include_text:
            data["text"] = self.text
        return data

    def event(self) -> dict:
        return StateEvent(
            state_id=self.state_id,
            state_type=self.state_type,
            owner=self.owner,
            tokens=self.tokens,
            semantic_key=self.semantic_key,
            exact_token_hash=self.text_hash,
            producer_event_id=self.producer_event_id,
            metadata={**self.metadata, "text_hash": self.text_hash, "text_preview": self.text[:512]},
        ).to_dict()


class PromptStateBuilder:
    """Builds explicit state-level prompt segments for local coding-agent traces."""

    def __init__(
        self,
        workflow_id: str,
        agent_id: str = "agent_0",
        history_mode: str = "selective_state",
        model_path: str = "/data1/dg123_data/Qwen-32B",
        recent_k: int = 4,
    ):
        if history_mode not in {"selective_state", "full_history"}:
            raise ValueError(f"Unsupported history mode: {history_mode}")
        self.workflow_id = workflow_id
        self.agent_id = agent_id
        self.history_mode = history_mode
        self.model_path = model_path
        self.recent_k = int(recent_k)
        self.tokenizer_name, self._tokenizer = self._load_tokenizer(model_path)
        self.states: Dict[str, PromptState] = {}
        self.state_events: List[dict] = []
        self.timeline: List[str] = []
        self.turn = 0
        self._bootstrap_done = False

    def bootstrap(self, issue_text: str, agent_role: str = "local_react_coding_agent") -> List[dict]:
        if self._bootstrap_done:
            return []
        self._bootstrap_done = True
        created = [
            self.create_state(
                "system_prefix",
                "You are a coding agent. Use explicit state references, inspect files, run tests, and repair the bug.",
                owner="shared",
                semantic_key=f"system:{self.workflow_id}",
                metadata={"scope": "static_prefix"},
            ),
            self.create_state(
                "task_prefix",
                issue_text,
                owner="shared",
                semantic_key=f"task:{self.workflow_id}",
                metadata={"scope": "issue"},
            ),
            self.create_state(
                "agent_role",
                f"Agent role: {agent_role}. Prefer concise plans, concrete file references, and test-driven repair.",
                owner=self.agent_id,
                semantic_key=f"role:{self.workflow_id}:{self.agent_id}",
                metadata={"framework": agent_role},
            ),
        ]
        return [state.event() for state in created]

    def create_state(
        self,
        state_type: str,
        text: object,
        owner: Optional[str] = None,
        semantic_key: Optional[str] = None,
        producer_event_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> PromptState:
        if state_type not in STATE_TYPES:
            raise ValueError(f"Unsupported state type: {state_type}")
        text = "" if text is None else str(text)
        owner = owner or self.agent_id
        text_hash = self._hash_text(text)
        semantic_key = semantic_key or f"{state_type}:{text_hash[:16]}"
        state_id = f"state_{hashlib.sha1(f'{self.workflow_id}\\0{state_type}\\0{semantic_key}\\0{text_hash}'.encode()).hexdigest()[:18]}"
        if state_id in self.states:
            return self.states[state_id]
        state = PromptState(
            state_id=state_id,
            state_type=state_type,
            owner=owner,
            text=text,
            tokens=self.count_tokens(text),
            text_hash=text_hash,
            semantic_key=semantic_key,
            producer_event_id=producer_event_id,
            metadata={**(metadata or {}), "tokenizer": self.tokenizer_name},
            created_turn=self.turn,
        )
        self.states[state_id] = state
        self.state_events.append(state.event())
        self.timeline.append(state_id)
        return state

    def register_file_context(
        self,
        path: str,
        text: str,
        start_line: int = 1,
        end_line: Optional[int] = None,
        version: str = "workspace",
        producer_event_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> PromptState:
        end_line = end_line if end_line is not None else start_line + max(0, len(text.splitlines()) - 1)
        return self.create_state(
            "file_context",
            text,
            semantic_key=f"file:{path}:{start_line}-{end_line}:{version}",
            producer_event_id=producer_event_id,
            metadata={"path": path, "start_line": start_line, "end_line": end_line, "version": version, **(metadata or {})},
        )

    def register_edit_diff(self, diff: str, path: str, producer_event_id: Optional[str] = None) -> PromptState:
        return self.create_state(
            "edit_diff",
            diff,
            semantic_key=f"edit:{path}:{self._hash_text(diff)[:16]}",
            producer_event_id=producer_event_id,
            metadata={"path": path},
        )

    def register_tool_output(
        self,
        tool: str,
        output: str,
        status: str = "ok",
        state_type: Optional[str] = None,
        producer_event_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> List[PromptState]:
        state_type = state_type or infer_state_type(tool, output, status)
        states = [
            self.create_state(
                state_type,
                output,
                semantic_key=semantic_key_for_tool(tool, output, metadata or {}, status),
                producer_event_id=producer_event_id,
                metadata={"tool": tool, "status": status, **(metadata or {})},
            )
        ]
        if self.count_tokens(output) > 512:
            summary = summarize_text(output)
            states.append(
                self.create_state(
                    "summary_state",
                    summary,
                    semantic_key=f"summary:{states[0].state_id}",
                    producer_event_id=producer_event_id,
                    metadata={"source_state_id": states[0].state_id, "tool": tool},
                )
            )
        return states

    def assemble_segments(
        self,
        phase: str,
        force_state_ids: Optional[Iterable[str]] = None,
        include_recent: bool = True,
        include_failures: bool = True,
    ) -> List[dict]:
        force_state_ids = list(force_state_ids or [])
        selected = []
        selected.extend(self._states_by_type("system_prefix"))
        selected.extend(self._states_by_type("task_prefix"))
        selected.extend(self._states_by_type("agent_role"))
        if self.history_mode == "full_history":
            selected.extend(self.states.values())
        else:
            selected.extend(self.states[sid] for sid in force_state_ids if sid in self.states)
            if include_failures:
                selected.extend(self._latest_by_type({"test_failure_summary", "failure_summary", "raw_error_log"}, limit=2))
            selected.extend(self._latest_by_type({"summary_state"}, limit=2))
            if include_recent:
                selected.extend(self._recent_short_term(limit=self.recent_k))
        deduped = []
        seen = set()
        for state in selected:
            if state.state_id in seen:
                continue
            seen.add(state.state_id)
            state.use_count += 1
            state.last_used_turn = self.turn
            deduped.append(state.segment(include_text=True))
        return deduped

    def messages_from_segments(self, segments: List[dict], user_instruction: str) -> List[dict]:
        body = ["Use the following explicit prompt states. Do not assume unseen full history."]
        for item in segments:
            body.append(
                f"[{item['state_id']}|{item['state_type']}|tokens={item['tokens']}|key={item.get('semantic_key','')}]\n{item.get('text','')}"
            )
        body.append(f"Current instruction:\n{user_instruction}")
        # Qwen2.5-VL's OpenAI path expects content-list even for text-only calls.
        return [{"role": "user", "content": [{"type": "text", "text": "\n\n".join(body)}]}]

    def make_llm_event(
        self,
        event_id: str,
        segments: List[dict],
        output_text: str,
        append_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        timestamp_start: Optional[float] = None,
        timestamp_end: Optional[float] = None,
        phase: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> tuple:
        self.turn += 1
        assistant_state = self.create_state(
            "assistant_delta",
            output_text,
            semantic_key=f"assistant:{event_id}",
            producer_event_id=event_id,
            metadata={"phase": phase or "unknown"},
        )
        input_state_ids = [item["state_id"] for item in segments]
        llm = LLMEvent(
            event_id=event_id,
            agent=self.agent_id,
            turn=self.turn,
            input_state_ids=input_state_ids,
            append_tokens=append_tokens if append_tokens is not None else sum(int(item.get("tokens", 0)) for item in segments),
            output_tokens=output_tokens if output_tokens is not None else self.count_tokens(output_text),
            new_state_id=assistant_state.state_id,
            new_state_type="assistant_delta",
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            input_segments=[{k: v for k, v in item.items() if k != "text"} for item in segments],
            phase=phase,
            metadata={
                "workflow_id": self.workflow_id,
                "history_mode": self.history_mode,
                "prompt_reconstruction": "explicit_selective_state" if self.history_mode == "selective_state" else "explicit_full_history",
                "full_history_likely": self.history_mode == "full_history",
                "tokenizer": self.tokenizer_name,
                **(metadata or {}),
            },
        )
        return assistant_state.event(), llm.to_dict()

    def count_tokens(self, text: object) -> int:
        text = "" if text is None else str(text)
        if self._tokenizer is not None:
            try:
                return max(1, len(self._tokenizer.encode(text)))
            except Exception:
                pass
        return max(1, len(text.split()) or len(text) // 4 or 1)

    def _latest_by_type(self, state_types: set, limit: int) -> List[PromptState]:
        states = [self.states[sid] for sid in self.timeline if self.states[sid].state_type in state_types]
        return states[-limit:]

    def _recent_short_term(self, limit: int) -> List[PromptState]:
        short = {"tool_observation", "summary_state", "assistant_delta", "edit_diff"}
        states = [self.states[sid] for sid in self.timeline if self.states[sid].state_type in short]
        return states[-limit:]

    def _states_by_type(self, state_type: str) -> List[PromptState]:
        return [self.states[sid] for sid in self.timeline if self.states[sid].state_type == state_type]

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256((self.tokenizer_name + "\0" + text).encode("utf-8", errors="ignore")).hexdigest()

    def _load_tokenizer(self, model_path: str):
        try:
            from transformers import AutoTokenizer

            return f"hf:{model_path}", AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            return "fallback:whitespace", None


def infer_state_type(tool: str, output: str, status: str) -> str:
    tool_lower = tool.lower()
    failed = status.lower() not in {"ok", "success", "passed"}
    lowered = output.lower()
    if tool_lower in {"read", "grep", "glob"}:
        return "file_context"
    if tool_lower in {"edit", "write", "patch"}:
        return "edit_diff"
    if "pytest" in tool_lower or "test" in tool_lower:
        return "test_failure_summary" if failed else "tool_observation"
    if failed:
        return "raw_error_log" if len(output) > 4000 or "traceback" in lowered else "failure_summary"
    return "tool_observation"


def semantic_key_for_tool(tool: str, output: str, metadata: dict, status: str) -> str:
    if metadata.get("path"):
        return f"file:{metadata['path']}:{metadata.get('line_range', 'all')}:{metadata.get('version', 'workspace')}"
    if metadata.get("command"):
        return f"command:{metadata['command']}:{status}"
    return f"tool:{tool}:{hashlib.sha1(output.encode('utf-8', errors='ignore')).hexdigest()[:16]}"


def summarize_text(text: str, limit: int = 1600) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    head = "\n".join(lines[:12])
    tail = "\n".join(lines[-12:]) if len(lines) > 12 else ""
    summary = head if not tail else head + "\n...\n" + tail
    return summary[:limit]


def write_json(path: Path, payload: object):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def new_event_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}_{int(time.time() * 1000)}"

