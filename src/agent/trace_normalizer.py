import json
from pathlib import Path
from typing import Iterable, List, Optional

from .prompt_segmenter import estimate_tokens, make_state_event, segment_messages, segment_tool_output, stable_id
from .trace_schema import LLMEvent, ToolEvent, is_normalized_event, validate_trace


LLM_KEYS = {"messages", "prompt", "input", "llm_input", "response", "completion", "assistant"}
TOOL_KEYS = {"tool", "tool_name", "action", "observation", "output", "result", "command"}


def normalize_payload(payload, trace_format: str = "auto", source_id: str = "trace") -> List[dict]:
    payload = _decode_maybe_json(payload)
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list) and payload["rows"]:
        first_row = payload["rows"][0]
        payload = first_row.get("row", first_row)
    if _looks_like_normalized_trace(payload):
        return _annotate_prompt_quality(validate_trace(list(payload)), default_reconstruction="explicit")
    if isinstance(payload, dict) and _looks_like_normalized_trace(payload.get("events")):
        return _annotate_prompt_quality(validate_trace(payload["events"]), default_reconstruction="explicit")
    if trace_format == "normalized_json":
        return _annotate_prompt_quality(
            validate_trace(payload if isinstance(payload, list) else payload.get("events", [])),
            default_reconstruction="explicit",
        )
    if trace_format == "otel_spans" or _looks_like_otel_spans_payload(payload):
        return normalize_otel_spans_payload(payload, source_id=source_id)
    if trace_format == "codetracer" or _looks_like_codetracer_payload(payload):
        return normalize_codetracer_payload(payload, source_id=source_id)
    return normalize_generic_react(payload, source_id=source_id, trace_format=trace_format)


def normalize_jsonl_records(records: Iterable[dict], trace_format: str = "auto", source_id: str = "trace") -> List[dict]:
    records = list(records)
    if _looks_like_normalized_trace(records):
        return _annotate_prompt_quality(validate_trace(records), default_reconstruction="explicit")
    if trace_format == "otel_spans" or (len(records) == 1 and _looks_like_otel_spans_payload(records[0])):
        payload = records[0] if len(records) == 1 else {"spans": records}
        return normalize_otel_spans_payload(payload, source_id=source_id)
    if trace_format == "codetracer":
        return normalize_codetracer_payload(records, source_id=source_id)
    return normalize_generic_react(records, source_id=source_id, trace_format=trace_format)


def _looks_like_normalized_trace(obj) -> bool:
    return isinstance(obj, list) and obj and all(is_normalized_event(item) for item in obj)


def _candidate_steps(payload):
    payload = _decode_maybe_json(payload)
    if isinstance(payload, list):
        return [_decode_maybe_json(item.get("row", item)) if isinstance(item, dict) else _decode_maybe_json(item) for item in payload]
    if not isinstance(payload, dict):
        return [{"content": str(payload)}]
    for key in ("events", "steps", "trajectory_steps", "trajectory", "trajectories", "history", "records", "logs", "rows"):
        value = _decode_maybe_json(payload.get(key))
        if isinstance(value, list):
            return [_decode_maybe_json(item.get("row", item)) if isinstance(item, dict) else _decode_maybe_json(item) for item in value]
    if isinstance(payload.get("messages"), list):
        return [payload]
    return [payload]


def _as_messages(value):
    value = _decode_maybe_json(value)
    if isinstance(value, list):
        messages = []
        for item in value:
            if isinstance(item, dict):
                role = item.get("role") or item.get("speaker") or item.get("type") or "user"
                content = item.get("content", item.get("text", item.get("message", "")))
                messages.append({"role": role, "content": content, **item})
            else:
                messages.append({"role": "user", "content": str(item)})
        return messages
    if value is None:
        return []
    return [{"role": "user", "content": str(value)}]


def _extract_messages(step: dict):
    step = _decode_maybe_json(step)
    if not isinstance(step, dict):
        return [{"role": "user", "content": str(step)}]
    for key in ("messages", "prompt", "input", "llm_input", "state"):
        if key in step:
            return _as_messages(step[key])
    return []


def _extract_assistant_text(step: dict) -> str:
    step = _decode_maybe_json(step)
    if not isinstance(step, dict):
        return ""
    for key in ("response", "completion", "assistant", "llm_output", "thought"):
        if key in step and step[key] is not None:
            return str(step[key])
    messages = step.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and str(message.get("role", "")).lower() == "assistant":
                return str(message.get("content", ""))
    return ""


def _extract_tool(step: dict):
    step = _decode_maybe_json(step)
    if not isinstance(step, dict):
        return None
    tool = step.get("tool") or step.get("tool_name") or step.get("action") or step.get("command")
    output = step.get("observation", step.get("output", step.get("result")))
    if isinstance(tool, dict):
        output = output if output is not None else tool.get("output", tool.get("result", ""))
        tool = tool.get("name", tool.get("tool", tool.get("type", "tool")))
    if tool is None and output is None:
        return None
    status = step.get("status") or step.get("exit_status") or ("failed" if step.get("error") else "ok")
    latency = step.get("latency", step.get("duration", step.get("elapsed", 1)))
    try:
        latency = int(float(latency))
    except Exception:
        latency = 1
    return {
        "tool": str(tool or "tool"),
        "output": "" if output is None else output,
        "status": str(status or "ok"),
        "latency": max(0, latency),
    }


def _success_from_payload(payload) -> Optional[bool]:
    if not isinstance(payload, dict):
        return None
    for key in ("success", "passed", "resolved"):
        if key in payload:
            return bool(payload[key])
    status = str(payload.get("status", payload.get("exit_status", ""))).lower()
    if status:
        if any(word in status for word in ("success", "passed", "resolved", "submit")):
            return True
        if any(word in status for word in ("fail", "error", "timeout")):
            return False
    return None


def _looks_like_codetracer_payload(payload) -> bool:
    payload = _decode_maybe_json(payload)
    if isinstance(payload, list):
        return any(_looks_like_codetracer_payload(item) for item in payload[:3])
    if not isinstance(payload, dict):
        return False
    if str(payload.get("trajectory_format", "")).lower().startswith("mini-swe-agent"):
        return True
    if payload.get("instance_id") and payload.get("messages") and payload.get("info"):
        return True
    return any(key in payload for key in ("state_tree", "persistent_memory", "trajectory_steps"))


def _looks_like_otel_spans_payload(payload) -> bool:
    payload = _decode_maybe_json(payload)
    if isinstance(payload, dict) and isinstance(payload.get("spans"), list):
        return True
    if isinstance(payload, list) and payload:
        return all(isinstance(item, dict) and ("span_id" in item or "attributes" in item) for item in payload[:3])
    if not isinstance(payload, dict):
        return False
    text = json.dumps(payload, ensure_ascii=True).lower()[:4000]
    return "gen_ai.input.messages" in text or "gen_ai.output.messages" in text


def normalize_otel_spans_payload(payload, source_id: str) -> List[dict]:
    payload = _decode_maybe_json(payload)
    if isinstance(payload, list):
        payload = {"spans": payload}
    if not isinstance(payload, dict):
        return normalize_generic_react(payload, source_id=source_id, trace_format="otel_spans")

    spans = _otel_spans(payload)
    agent = _agent_name(payload, source_id)
    trace_success = _otel_trace_success(payload, spans)
    model_id = _otel_model_id(payload, spans)
    events = []
    seen_states = set()
    seen_tool_states = set()
    turn = 0

    for span in spans:
        attrs = span.get("attributes") or {}
        input_messages = _otel_messages(attrs.get("gen_ai.input.messages"))
        output_messages = _otel_messages(attrs.get("gen_ai.output.messages"))
        input_ids = []

        for msg_idx, message in enumerate(input_messages):
            normalized_message = _otel_normalized_message(message)
            content = normalized_message.get("content", "")
            if not str(content).strip():
                continue
            if _otel_is_tool_response(message):
                tool_name = _otel_tool_name(message)
                status = "failed" if _looks_failed(content) else "ok"
                tool_state = segment_tool_output(
                    tool_name,
                    content,
                    source_id=f"{source_id}:oteltool",
                    agent=agent,
                    ordinal=msg_idx,
                    status=status,
                    producer_event_id=_event_id(source_id, "tool", len(seen_tool_states)),
                    metadata={
                        "raw": message,
                        "trace_format": "otel_spans",
                        "trace_source": "huggingface_otel",
                        "span_id": span.get("span_id"),
                        "tool_call_id": _otel_tool_call_id(message),
                    },
                )
                if tool_state["state_id"] not in seen_tool_states:
                    events.append(
                        ToolEvent(
                            event_id=_event_id(source_id, "tool", len(seen_tool_states)),
                            agent=agent,
                            turn=turn,
                            tool=tool_name,
                            latency=_otel_span_latency(span),
                            output_tokens=tool_state["tokens"],
                            status=status,
                            new_state_id=tool_state["state_id"],
                            new_state_type=tool_state["state_type"],
                            phase=_phase_from_text(content),
                            metadata={
                                "raw": message,
                                "trace_success": trace_success,
                                "trace_format": "otel_spans",
                                "trace_source": "huggingface_otel",
                                "span_id": span.get("span_id"),
                                "tool_call_id": _otel_tool_call_id(message),
                            },
                        ).to_dict()
                    )
                    seen_tool_states.add(tool_state["state_id"])
                input_ids.append(tool_state["state_id"])
                continue

            states = segment_messages(
                [normalized_message],
                source_id=f"{source_id}:otelmsg",
                agent=agent,
                model_id=model_id,
                tokenizer_id="otel",
            )
            for state in states:
                metadata = dict(state.get("metadata") or {})
                metadata.setdefault("trace_format", "otel_spans")
                metadata.setdefault("trace_source", "huggingface_otel")
                metadata.setdefault("span_id", span.get("span_id"))
                state["metadata"] = metadata
                if state["state_id"] not in seen_states:
                    events.append(state)
                    seen_states.add(state["state_id"])
                input_ids.append(state["state_id"])

        assistant_text = "\n".join(
            text for text in (_otel_message_content(message) for message in output_messages) if str(text).strip()
        )
        output_state_id = stable_id(source_id, "assistant", turn, assistant_text, span.get("span_id"), prefix="state")
        events.append(
            LLMEvent(
                event_id=_event_id(source_id, "llm", turn),
                agent=agent,
                turn=turn,
                input_state_ids=list(dict.fromkeys(input_ids)),
                append_tokens=_otel_int(attrs.get("gen_ai.usage.input_tokens"), estimate_tokens(input_messages[-1] if input_messages else "")),
                output_tokens=_otel_int(attrs.get("gen_ai.usage.output_tokens"), estimate_tokens(assistant_text)),
                new_state_id=output_state_id,
                phase=_phase_from_text(assistant_text),
                metadata={
                    "raw": span,
                    "trace_success": trace_success,
                    "trace_format": "otel_spans",
                    "trace_source": "huggingface_otel",
                    "span_id": span.get("span_id"),
                    "trace_id": span.get("trace_id"),
                    "session_id": payload.get("session_id") or span.get("session_id"),
                    "benchmark": payload.get("benchmark") or span.get("benchmark"),
                    "harness": payload.get("harness") or span.get("harness"),
                    "prompt_reconstruction": "exact_otel_messages",
                },
            ).to_dict()
        )
        turn += 1

    return _annotate_prompt_quality(validate_trace(events), default_reconstruction="exact_otel_messages")


def _otel_spans(payload: dict) -> list:
    spans = _decode_maybe_json(payload.get("spans")) or []
    if not isinstance(spans, list):
        return []
    llm_spans = [
        _decode_maybe_json(span)
        for span in spans
        if isinstance(_decode_maybe_json(span), dict)
        and (
            str(_decode_maybe_json(span).get("type", "")).lower() == "llm_call"
            or "gen_ai.input.messages" in (_decode_maybe_json(span).get("attributes") or {})
        )
    ]
    return sorted(llm_spans, key=lambda span: str(span.get("start_time", "")))


def _otel_messages(value) -> list:
    value = _decode_maybe_json(value)
    if isinstance(value, list):
        return [item if isinstance(item, dict) else {"role": "user", "content": str(item)} for item in value]
    if isinstance(value, dict):
        return [value]
    if value is None:
        return []
    return [{"role": "user", "content": str(value)}]


def _otel_normalized_message(message: dict) -> dict:
    role = str(message.get("role") or message.get("type") or "user")
    return {"role": role, "content": _otel_message_content(message), **message}


def _otel_message_content(message: dict) -> str:
    if not isinstance(message, dict):
        return str(message)
    if message.get("content") is not None:
        return str(message.get("content"))
    parts = _decode_maybe_json(message.get("parts"))
    if not isinstance(parts, list):
        return ""
    chunks = []
    for part in parts:
        if not isinstance(part, dict):
            chunks.append(str(part))
            continue
        part_type = str(part.get("type", "")).lower()
        if part.get("content") is not None:
            chunks.append(str(part.get("content")))
        elif part.get("result") is not None:
            chunks.append(str(part.get("result")))
        elif part_type in {"tool_call", "function_call"}:
            chunks.append(
                json.dumps(
                    {
                        "tool_call": part.get("name") or part.get("tool"),
                        "arguments": part.get("arguments"),
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
            )
    return "\n".join(chunk for chunk in chunks if str(chunk).strip())


def _otel_is_tool_response(message: dict) -> bool:
    role = str(message.get("role", "")).lower()
    if role in {"tool", "observation", "function"}:
        return True
    parts = _decode_maybe_json(message.get("parts"))
    return isinstance(parts, list) and any(
        isinstance(part, dict) and str(part.get("type", "")).lower() in {"tool_call_response", "tool_result", "function_response"}
        for part in parts
    )


def _otel_tool_name(message: dict) -> str:
    if message.get("name") or message.get("tool"):
        return str(message.get("name") or message.get("tool"))
    parts = _decode_maybe_json(message.get("parts"))
    if isinstance(parts, list):
        for part in parts:
            if isinstance(part, dict) and (part.get("name") or part.get("tool")):
                return str(part.get("name") or part.get("tool"))
    return "tool_response"


def _otel_tool_call_id(message: dict):
    if message.get("tool_call_id") or message.get("id"):
        return message.get("tool_call_id") or message.get("id")
    parts = _decode_maybe_json(message.get("parts"))
    if isinstance(parts, list):
        for part in parts:
            if isinstance(part, dict) and part.get("id"):
                return part.get("id")
    return None


def _otel_model_id(payload: dict, spans: list) -> str:
    models = payload.get("models")
    if isinstance(models, list) and models:
        return str(models[0])
    for span in spans:
        attrs = span.get("attributes") or {}
        if attrs.get("gen_ai.request.model"):
            return str(attrs["gen_ai.request.model"])
    return "unknown"


def _otel_trace_success(payload: dict, spans: list):
    direct = _success_from_payload(payload)
    if direct is not None:
        return direct
    status_codes = []
    for span in spans:
        status = span.get("status") or {}
        if isinstance(status, dict) and status.get("code") is not None:
            status_codes.append(status.get("code"))
    if not status_codes:
        return None
    return all(int(code) in {0, 1} for code in status_codes)


def _otel_int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _otel_span_latency(span: dict) -> int:
    return 1


def normalize_codetracer_payload(payload, source_id: str) -> List[dict]:
    """Normalize CodeTracer/CodeTraceBench payloads without hiding provenance.

    Some CodeTraceBench samples only expose the chat `messages` transcript; those
    are intentionally labeled `message_history` and may be rejected by audit as
    full-history reconstructions. Richer artifacts with per-step context/state
    fields are normalized as `step_context` or `state_tree`.
    """
    payload = _decode_maybe_json(payload)
    if isinstance(payload, list):
        payload = {"steps": payload, "source_id": source_id}

    if isinstance(payload, dict):
        steps = _codetracer_steps(payload)
        if steps:
            events = _normalize_codetracer_steps(payload, steps, source_id)
            return _annotate_prompt_quality(validate_trace(events), default_reconstruction="step_context")
        if _is_message_trajectory(payload.get("messages")):
            events = _normalize_message_trajectory(payload, source_id, trace_format="codetracer")
            for event in events:
                metadata = dict(event.get("metadata") or {})
                metadata.setdefault("trace_source", "codetracer")
                metadata.setdefault("codetracer_adapter_reason", "messages_only_no_step_context")
                if event.get("type") == "llm":
                    metadata["prompt_reconstruction"] = "message_history"
                event["metadata"] = metadata
            return _annotate_prompt_quality(validate_trace(events), default_reconstruction="message_history")
    return normalize_generic_react(payload, source_id=source_id, trace_format="codetracer")


def _codetracer_steps(payload: dict) -> list:
    for key in ("steps", "trajectory_steps", "trajectory", "actions", "events"):
        value = _decode_maybe_json(payload.get(key))
        if isinstance(value, list) and value:
            return [_decode_maybe_json(item.get("row", item)) if isinstance(item, dict) else _decode_maybe_json(item) for item in value]
    return []


def _normalize_codetracer_steps(payload: dict, steps: list, source_id: str) -> List[dict]:
    events = []
    seen_states = set()
    current_context = []
    agent = _agent_name(payload, source_id)
    trace_success = _success_from_payload(payload)

    initial_messages = _initial_messages(payload) or []
    for state in segment_messages(initial_messages, source_id=f"{source_id}:initial", agent=agent):
        state["metadata"].setdefault("trace_source", "codetracer")
        state["metadata"].setdefault("prompt_reconstruction", "explicit")
        if state["state_id"] not in seen_states:
            events.append(state)
            seen_states.add(state["state_id"])
        current_context.append(state["state_id"])

    for turn, step in enumerate(steps):
        if not isinstance(step, dict):
            step = {"content": str(step)}
        phase = _phase_from_step(step)
        step_states, reconstruction = _codetracer_context_states(step, source_id, agent, turn)
        input_ids = []
        for state in step_states:
            state["metadata"].setdefault("trace_source", "codetracer")
            state["metadata"].setdefault("raw_step_id", _raw_step_id(step, turn))
            if state["state_id"] not in seen_states:
                events.append(state)
                seen_states.add(state["state_id"])
            input_ids.append(state["state_id"])
        if input_ids:
            current_context = list(dict.fromkeys(input_ids))
        else:
            input_ids = list(current_context)
            reconstruction = "accumulated_fallback"

        assistant_text = _extract_assistant_text(step) or str(step.get("thought", step.get("reasoning", "")))
        if assistant_text or input_ids:
            output_state_id = stable_id(source_id, "assistant", turn, assistant_text, prefix="state")
            events.append(
                LLMEvent(
                    event_id=_event_id(source_id, "llm", turn),
                    agent=agent,
                    turn=turn,
                    input_state_ids=list(dict.fromkeys(input_ids)),
                    append_tokens=estimate_tokens(_step_append_text(step)),
                    output_tokens=estimate_tokens(assistant_text),
                    new_state_id=output_state_id,
                    phase=phase,
                    metadata={
                        "raw": step,
                        "trace_success": trace_success,
                        "trace_format": "codetracer",
                        "trace_source": "codetracer",
                        "raw_step_id": _raw_step_id(step, turn),
                        "prompt_reconstruction": reconstruction,
                        "stage": step.get("stage") or step.get("phase"),
                    },
                ).to_dict()
            )
            current_context.append(output_state_id)

        for tool_info in _codetracer_tools(step):
            event_id = _event_id(source_id, "tool", turn)
            state = segment_tool_output(
                tool_info["tool"],
                tool_info["output"],
                source_id=f"{source_id}:tool{turn}:{tool_info['ordinal']}",
                agent=agent,
                ordinal=turn,
                status=tool_info["status"],
                producer_event_id=event_id,
                metadata={
                    "raw": step,
                    "trace_source": "codetracer",
                    "raw_step_id": _raw_step_id(step, turn),
                    "file_path": tool_info.get("file_path"),
                    "test_name": tool_info.get("test_name"),
                },
            )
            events.append(
                ToolEvent(
                    event_id=f"{event_id}:{tool_info['ordinal']}",
                    agent=agent,
                    turn=turn,
                    tool=tool_info["tool"],
                    latency=tool_info["latency"],
                    output_tokens=state["tokens"],
                    status=tool_info["status"],
                    new_state_id=state["state_id"],
                    new_state_type=state["state_type"],
                    phase=phase,
                    metadata={
                        "raw": step,
                        "trace_success": trace_success,
                        "trace_format": "codetracer",
                        "trace_source": "codetracer",
                        "raw_step_id": _raw_step_id(step, turn),
                        "file_path": tool_info.get("file_path"),
                        "test_name": tool_info.get("test_name"),
                    },
                ).to_dict()
            )
            current_context.append(state["state_id"])
    return events


def _codetracer_context_states(step: dict, source_id: str, agent: str, turn: int) -> tuple[list, str]:
    context_items = []
    reconstruction = "step_context"
    for key in ("state_tree", "persistent_memory", "memory", "state"):
        value = _decode_maybe_json(step.get(key))
        if value:
            reconstruction = "state_tree" if key in {"state_tree", "persistent_memory"} else "step_context"
            context_items.extend(_flatten_context_items(value, key))
    for key in ("context", "prompt_context", "input_state", "input_states", "observations"):
        value = _decode_maybe_json(step.get(key))
        if value:
            context_items.extend(_flatten_context_items(value, key))
    states = []
    for ordinal, item in enumerate(context_items):
        text = item.get("content", "")
        if not str(text).strip():
            continue
        state_type = item.get("state_type") or _state_type_from_context_key(item.get("key", ""), text)
        states.append(
            make_state_event(
                text,
                state_type,
                agent,
                source_id=f"{source_id}:ctx{turn}",
                ordinal=ordinal,
                semantic_key=item.get("semantic_key"),
                metadata={
                    "trace_source": "codetracer",
                    "context_key": item.get("key"),
                    "file_path": item.get("file_path"),
                    "prompt_reconstruction": reconstruction,
                },
            )
        )
    return states, reconstruction if states else "accumulated_fallback"


def _flatten_context_items(value, key: str) -> list[dict]:
    value = _decode_maybe_json(value)
    if value is None:
        return []
    if isinstance(value, dict):
        items = []
        if any(name in value for name in ("content", "text", "value", "path", "file_path")):
            content = value.get("content", value.get("text", value.get("value", "")))
            if not content and value.get("path"):
                content = value.get("path")
            items.append(
                {
                    "key": key,
                    "content": content,
                    "file_path": value.get("file_path") or value.get("path"),
                    "semantic_key": value.get("semantic_key") or value.get("id") or value.get("path"),
                }
            )
        else:
            for child_key, child_value in value.items():
                for item in _flatten_context_items(child_value, str(child_key)):
                    items.append(item)
        return items
    if isinstance(value, list):
        items = []
        for idx, child in enumerate(value):
            for item in _flatten_context_items(child, f"{key}:{idx}"):
                items.append(item)
        return items
    return [{"key": key, "content": str(value)}]


def _state_type_from_context_key(key: str, content: object) -> str:
    text = f"{key} {str(content)[:500]}".lower()
    if any(word in text for word in ("patch", "diff", "edit", "apply")):
        return "edit_diff"
    if any(word in text for word in ("pytest", "cargo test", "test failure", "failed test")):
        return "test_failure_summary"
    if any(word in text for word in ("traceback", "error", "stderr", "returncode")):
        return "failure_summary"
    if any(word in text for word in ("summary", "memory")):
        return "summary_state"
    if any(word in text for word in ("subagent", "delegate")):
        return "subagent_output"
    if any(word in text for word in ("file", "read", "grep", "search", "cat", "source")):
        return "file_context"
    return "dialogue_delta"


def _codetracer_tools(step: dict) -> list[dict]:
    candidates = []
    direct = _extract_tool(step)
    if direct is not None:
        candidates.append(direct)
    for key in ("tool_calls", "tools", "actions"):
        value = _decode_maybe_json(step.get(key))
        if isinstance(value, list):
            for item in value:
                extracted = _extract_tool(item)
                if extracted is not None:
                    candidates.append(extracted)
    if not candidates:
        return []
    output = []
    for ordinal, item in enumerate(candidates):
        tool = _canonical_tool_name(item.get("tool"))
        output.append(
            {
                "ordinal": ordinal,
                "tool": tool,
                "output": item.get("output", ""),
                "status": item.get("status", "ok"),
                "latency": item.get("latency", 1),
                "file_path": item.get("file_path"),
                "test_name": item.get("test_name"),
            }
        )
    return output


def _canonical_tool_name(tool: object) -> str:
    text = str(tool or "tool")
    lower = text.lower()
    if any(word in lower for word in ("read", "grep", "search", "cat", "ls", "find")):
        return "Read"
    if any(word in lower for word in ("edit", "patch", "apply", "write")):
        return "Edit"
    if any(word in lower for word in ("test", "pytest", "cargo test")):
        return "Test"
    if any(word in lower for word in ("subagent", "delegate")):
        return "Subagent"
    if "summary" in lower:
        return "Summary"
    return text


def _raw_step_id(step: dict, turn: int):
    for key in ("id", "step_id", "idx", "index", "turn"):
        if step.get(key) is not None:
            return step.get(key)
    return turn


def _step_append_text(step: dict) -> str:
    for key in ("input", "prompt", "user", "instruction", "observation"):
        if step.get(key):
            return str(step.get(key))
    return ""


def normalize_generic_react(payload, source_id: str, trace_format: str = "auto") -> List[dict]:
    payload = _decode_maybe_json(payload)
    if isinstance(payload, dict) and _is_message_trajectory(payload.get("messages")):
        return _annotate_prompt_quality(
            validate_trace(_normalize_message_trajectory(payload, source_id, trace_format)),
            default_reconstruction="message_history",
        )
    events = []
    current_states = []
    seen_state_ids = set()
    agent = _agent_name(payload, source_id)
    trace_success = _success_from_payload(payload)
    steps = _candidate_steps(payload)

    initial_messages = _initial_messages(payload)
    if initial_messages and not any(isinstance(step, dict) and "messages" in step for step in steps[1:]):
        steps = [{"messages": initial_messages}] + [step for step in steps if step is not payload]

    for turn, step in enumerate(steps):
        messages = _extract_messages(step)
        if messages:
            state_events = segment_messages(messages, source_id=f"{source_id}:turn{turn}", agent=agent)
            input_ids = []
            for state in state_events:
                if state["state_id"] not in seen_state_ids:
                    state["metadata"].setdefault("trace_format", trace_format)
                    events.append(state)
                    seen_state_ids.add(state["state_id"])
                input_ids.append(state["state_id"])
            input_ids = list(dict.fromkeys(input_ids))

            assistant_text = _extract_assistant_text(step)
            event_id = _event_id(source_id, "llm", turn)
            output_state_id = stable_id(source_id, "assistant", turn, assistant_text, prefix="state")
            events.append(
                LLMEvent(
                    event_id=event_id,
                    agent=agent,
                    turn=turn,
                    input_state_ids=list(input_ids),
                    input_segments=state_events,
                    append_tokens=estimate_tokens(messages[-1].get("content", "")),
                    output_tokens=estimate_tokens(assistant_text),
                    new_state_id=output_state_id,
                    phase=_phase_from_step(step),
                    metadata={
                        "raw": step,
                        "trace_success": trace_success,
                        "trace_format": trace_format,
                        "prompt_reconstruction": "message_history",
                    },
                ).to_dict()
            )
            current_states = list(dict.fromkeys(input_ids + [output_state_id]))
        elif isinstance(step, dict) and (step.get("thought") or step.get("action")) and current_states:
            assistant_text = "\n".join(str(step.get(key, "")) for key in ("thought", "action") if step.get(key))
            event_id = _event_id(source_id, "llm", turn)
            output_state_id = stable_id(source_id, "assistant", turn, assistant_text, prefix="state")
            events.append(
                LLMEvent(
                    event_id=event_id,
                    agent=agent,
                    turn=turn,
                    input_state_ids=list(current_states),
                    append_tokens=0,
                    output_tokens=estimate_tokens(assistant_text),
                    new_state_id=output_state_id,
                    phase=_phase_from_step(step),
                    metadata={
                        "raw": step,
                        "trace_success": trace_success,
                        "trace_format": trace_format,
                        "prompt_reconstruction": "accumulated_fallback",
                    },
                ).to_dict()
            )
            current_states.append(output_state_id)

        tool_info = _extract_tool(step)
        if tool_info is not None:
            event_id = _event_id(source_id, "tool", turn)
            state = segment_tool_output(
                tool_info["tool"],
                tool_info["output"],
                source_id=f"{source_id}:tool{turn}",
                agent=agent,
                ordinal=turn,
                status=tool_info["status"],
                producer_event_id=event_id,
                metadata={"raw": step, "trace_format": trace_format},
            )
            events.append(
                ToolEvent(
                    event_id=event_id,
                    agent=agent,
                    turn=turn,
                    tool=tool_info["tool"],
                    latency=tool_info["latency"],
                    output_tokens=state["tokens"],
                    status=tool_info["status"],
                    new_state_id=state["state_id"],
                    new_state_type=state["state_type"],
                    phase=_phase_from_step(step),
                    metadata={"raw": step, "trace_success": trace_success, "trace_format": trace_format},
                ).to_dict()
            )
            current_states.append(state["state_id"])

    return _annotate_prompt_quality(validate_trace(events), default_reconstruction="accumulated_fallback")


def _is_message_trajectory(messages) -> bool:
    messages = _decode_maybe_json(messages)
    return (
        isinstance(messages, list)
        and len(messages) >= 3
        and any(isinstance(item, dict) and str(item.get("role", "")).lower() == "assistant" for item in messages)
    )


def _normalize_message_trajectory(payload: dict, source_id: str, trace_format: str = "auto") -> List[dict]:
    messages = _decode_maybe_json(payload.get("messages")) or []
    agent = _agent_name(payload, source_id)
    trace_success = _success_from_payload(payload)
    events = []
    current_states = []
    seen_states = set()
    last_user_text = ""
    turn = 0
    pending_initial = []

    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            message = {"role": "user", "content": str(message)}
        role = str(message.get("role", "user")).lower()
        content = message.get("content", "")
        if role in {"system", "developer"}:
            state = segment_messages([message], source_id=f"{source_id}:msg{idx}", agent=agent)[0]
            state["metadata"].setdefault("trace_format", trace_format)
            if state["state_id"] not in seen_states:
                events.append(state)
                seen_states.add(state["state_id"])
            current_states.append(state["state_id"])
            continue

        if role == "assistant":
            if pending_initial:
                state_events = segment_messages(pending_initial, source_id=f"{source_id}:initial", agent=agent)
                for state in state_events:
                    state["metadata"].setdefault("trace_format", trace_format)
                    if state["state_id"] not in seen_states:
                        events.append(state)
                        seen_states.add(state["state_id"])
                    current_states.append(state["state_id"])
                pending_initial = []
            current_states = list(dict.fromkeys(current_states))
            event_id = _event_id(source_id, "llm", turn)
            output_state_id = stable_id(source_id, "assistant", idx, content, prefix="state")
            events.append(
                LLMEvent(
                    event_id=event_id,
                    agent=agent,
                    turn=turn,
                    input_state_ids=list(current_states),
                    append_tokens=estimate_tokens(last_user_text),
                    output_tokens=estimate_tokens(content),
                    new_state_id=output_state_id,
                    phase=_phase_from_text(content),
                    metadata={
                        "raw": message,
                        "trace_success": trace_success,
                        "trace_format": trace_format,
                        "prompt_reconstruction": "message_history",
                    },
                ).to_dict()
            )
            current_states.append(output_state_id)
            turn += 1
            last_user_text = ""
            continue

        if role in {"user", "tool", "observation"}:
            if not current_states:
                pending_initial.append(message)
                last_user_text = str(content)
                continue
            tool_like = _looks_like_observation(content)
            if tool_like:
                tool_name = _infer_tool_from_previous(events)
                status = "failed" if _looks_failed(content) else "ok"
                state_type = _tool_state_type_from_text(tool_name, content, status)
                state_id = stable_id(source_id, "tool", idx, content, prefix="state")
                events.append(
                    ToolEvent(
                        event_id=_event_id(source_id, "tool", turn),
                        agent=agent,
                        turn=turn,
                        tool=tool_name,
                        latency=max(1, _latency_from_message(message)),
                        output_tokens=estimate_tokens(content),
                        status=status,
                        new_state_id=state_id,
                        new_state_type=state_type,
                        phase=_phase_from_text(content),
                        metadata={"raw": message, "trace_success": trace_success, "trace_format": trace_format},
                    ).to_dict()
                )
                current_states.append(state_id)
            else:
                state = segment_messages([message], source_id=f"{source_id}:msg{idx}", agent=agent)[0]
                state["metadata"].setdefault("trace_format", trace_format)
                if state["state_id"] not in seen_states:
                    events.append(state)
                    seen_states.add(state["state_id"])
                current_states.append(state["state_id"])
                last_user_text = str(content)
    return events


def _annotate_prompt_quality(trace: List[dict], default_reconstruction: str = "explicit") -> List[dict]:
    """Annotate LLM prompt provenance and full-history risk.

    Public agent logs vary from exact normalized prompt segments to lossy message
    histories. The evaluator must know which case it is using before making
    real-trace claims.
    """
    trace = [dict(event) for event in trace]
    llm_events = [event for event in trace if event.get("type") == "llm"]
    input_counts = [len(event.get("input_state_ids", [])) for event in llm_events]
    token_lookup = _state_token_lookup(trace)
    context_tokens = [
        sum(token_lookup.get(state_id, 1) for state_id in event.get("input_state_ids", []))
        for event in llm_events
    ]
    growth_score = _monotonic_growth_score(input_counts)
    full_history_trace = len(input_counts) >= 4 and growth_score >= 0.8 and input_counts[-1] > input_counts[0]
    token_growth_trace = (
        len(context_tokens) >= 4
        and _monotonic_growth_score(context_tokens) >= 0.8
        and context_tokens[-1] > context_tokens[0]
    )
    for event in trace:
        if event.get("type") != "llm":
            continue
        metadata = dict(event.get("metadata") or {})
        reconstruction = metadata.get("prompt_reconstruction")
        if reconstruction is None:
            reconstruction = "explicit" if event.get("input_segments") else default_reconstruction
        metadata["prompt_reconstruction"] = reconstruction
        exact_prompt_source = reconstruction in {"explicit", "exact_otel_messages"}
        metadata["monotonic_context_growth_score"] = growth_score
        metadata["full_history_likely"] = bool(
            reconstruction == "accumulated_fallback"
            or ((full_history_trace or token_growth_trace) and not exact_prompt_source)
        )
        event["metadata"] = metadata
    return trace


def _state_token_lookup(trace: List[dict]) -> dict:
    tokens = {}
    for event in trace:
        if event.get("type") == "state":
            tokens[event["state_id"]] = int(event.get("tokens", 1))
        elif event.get("type") == "tool":
            tokens[event["new_state_id"]] = int(event.get("output_tokens", 1))
        elif event.get("type") == "llm":
            tokens[event["new_state_id"]] = int(event.get("output_tokens", 1))
    return tokens


def _monotonic_growth_score(values: List[int]) -> float:
    if len(values) <= 1:
        return 0.0
    comparisons = 0
    non_decreasing = 0
    for left, right in zip(values, values[1:]):
        comparisons += 1
        if right >= left:
            non_decreasing += 1
    return non_decreasing / comparisons if comparisons else 0.0


def _looks_like_observation(content: object) -> bool:
    text = str(content or "").lower()
    return any(marker in text for marker in ("<returncode>", "<output>", "<warning>", "<explore_context>", "traceback", "error:"))


def _looks_failed(content: object) -> bool:
    text = str(content or "").lower()
    return any(marker in text for marker in ("<returncode>1", "<returncode>2", "traceback", "error:", "failed", "not found"))


def _infer_tool_from_previous(events: List[dict]) -> str:
    for event in reversed(events):
        if event.get("type") == "llm":
            raw = (event.get("metadata") or {}).get("raw", {})
            content = str(raw.get("content", "")).lower() if isinstance(raw, dict) else ""
            if "pytest" in content or "cargo test" in content or "test" in content:
                return "test"
            if "rg " in content or "grep" in content or "find " in content:
                return "Grep"
            if "sed " in content or "cat " in content or "nl " in content:
                return "Read"
            if "apply_patch" in content or "git apply" in content or "diff" in content:
                return "Edit"
            if "bash" in content or "```" in content:
                return "Bash"
    return "tool"


def _tool_state_type_from_text(tool: str, content: object, status: str) -> str:
    from .prompt_segmenter import state_type_for_tool

    return state_type_for_tool(tool, content, status=status)


def _latency_from_message(message: dict) -> int:
    for key in ("latency", "duration", "elapsed", "total_time"):
        value = message.get(key)
        if value is None:
            continue
        try:
            return int(float(value))
        except Exception:
            continue
    return 1


def _phase_from_text(content: object) -> Optional[str]:
    text = str(content or "").lower()
    if any(word in text for word in ("test", "pytest", "cargo test", "failure", "traceback", "verify")):
        return "verify"
    if any(word in text for word in ("edit", "patch", "write", "diff", "fix")):
        return "execute"
    if any(word in text for word in ("read", "grep", "search", "inspect", "find")):
        return "explore"
    return None


def _event_id(source_id: str, kind: str, turn: int) -> str:
    return f"{source_id}:{kind}:{turn}"


def _agent_name(payload, source_id: str) -> str:
    if isinstance(payload, dict):
        for key in ("agent", "agent_id", "model", "instance_id"):
            if payload.get(key):
                return str(payload[key]).replace("/", "_")[:80]
    return Path(source_id).stem or "agent_0"


def _initial_messages(payload):
    if not isinstance(payload, dict):
        return None
    if payload.get("messages"):
        return payload.get("messages")
    messages = []
    agent_args = _decode_maybe_json(payload.get("agent_args", {}))
    if isinstance(agent_args, dict) and agent_args.get("system_prompt"):
        messages.append({"role": "system", "content": agent_args["system_prompt"]})
    if payload.get("problem_statement"):
        messages.append({"role": "user", "content": payload["problem_statement"]})
    elif isinstance(payload.get("ds"), dict) and payload["ds"].get("problem_statement"):
        messages.append({"role": "user", "content": payload["ds"]["problem_statement"]})
    return messages or None


def _phase_from_step(step) -> Optional[str]:
    step = _decode_maybe_json(step)
    if not isinstance(step, dict):
        return None
    phase = step.get("phase") or step.get("stage")
    if phase:
        return str(phase).lower()
    text = json.dumps(step, ensure_ascii=True).lower()[:4000]
    if any(word in text for word in ("pytest", "test", "verify", "failure", "traceback")):
        return "verify"
    if any(word in text for word in ("edit", "write", "patch", "diff")):
        return "execute"
    if any(word in text for word in ("read", "grep", "search", "inspect")):
        return "explore"
    return None


def _decode_maybe_json(value):
    for _ in range(3):
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if stripped and stripped[0] in "[{\"":
            try:
                decoded = json.loads(stripped)
                if decoded == value:
                    return value
                value = decoded
                continue
            except Exception:
                return value
        return value
    return value
