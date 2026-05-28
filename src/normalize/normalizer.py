"""Normalize heterogeneous public agent traces into canonical tables."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections.abc import Iterable
from typing import Any

import pandas as pd

from src.normalize.command_parser import (
    classify_phase as classify_phase_command,
    clean_tool_name,
    command_has_explicit_write,
    extract_command_string,
    extract_tool_wrapper,
    extract_tool_wrapper_from_value,
    semantic_tool_from_command,
)
from src.normalize.schema import (
    OBJECT_COLUMNS,
    STEP_COLUMNS,
    TRACE_COLUMNS,
    ObjectAccess,
    Step,
    Trace,
)

LOGGER = logging.getLogger(__name__)

ERROR_RE = re.compile(
    r"(traceback|assertionerror|runtimeerror|exception|failed|error:|exit code [1-9]|timeout|not resolved)",
    re.I,
)
TEST_RE = re.compile(r"(pytest|unittest|failed|passed|assertion|test_|::test|tests?/)", re.I)
URL_RE = re.compile(r"https?://[^\s\]\)\"']+")
PATH_RE = re.compile(
    r"(?<![\w@.-])((?:/|\.{1,2}/)?(?:[\w@+.-]+/)*[\w@+.-]+\."
    r"(?:html|htm|hpp|cpp|tsx|jsx|ipynb|yaml|json|java|toml|rst|txt|css|csv|log|xml|"
    r"py|js|ts|go|rs|cc|ini|yml|sql|php|cfg|md|sh|rb|c|h)"
    r"(?=$|[^\w./-]))"
)


def estimate_tokens(text: str | None) -> int:
    if not text:
        return 0
    return int(math.ceil(len(text) / 4.0))


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def _safe_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "resolved", "pass", "passed", "submitted"}:
        return True
    if lowered in {"false", "0", "no", "unresolved", "fail", "failed", "error"}:
        return False
    return None


def _stringify(value: Any, max_chars: int | None = None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    elif isinstance(value, (int, float, bool)):
        text = str(value)
    else:
        try:
            text = json.dumps(value, ensure_ascii=True, sort_keys=True)
        except Exception:
            text = str(value)
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars] + "\n...[truncated]"
    return text


def _first_value(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def _parse_steps(raw: Any, warnings: list[str], trace_id: str) -> list[Any]:
    if raw in (None, "", "null"):
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, tuple):
        return list(raw)
    if isinstance(raw, dict):
        for key in ("steps", "trajectory", "messages", "history"):
            if key in raw:
                return _parse_steps(raw[key], warnings, trace_id)
        return [raw]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return _parse_steps(parsed, warnings, trace_id)
        except Exception as exc:
            warnings.append(f"{trace_id}: could not parse steps JSON ({exc}); treating as one text step.")
            return [{"role": "unknown", "content": raw}]
    warnings.append(f"{trace_id}: unsupported steps type {type(raw).__name__}; stringifying.")
    return [{"role": "unknown", "content": _stringify(raw)}]


def _extract_nested_name(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for path in (
            ("name",),
            ("fn",),
            ("cmd",),
            ("tool_name",),
            ("tool",),
            ("function", "name"),
            ("function_call", "name"),
        ):
            cur: Any = value
            ok = True
            for part in path:
                if not isinstance(cur, dict) or part not in cur:
                    ok = False
                    break
                cur = cur[part]
            if ok and cur:
                return _stringify(cur)
    if isinstance(value, list) and value:
        return _extract_nested_name(value[0])
    return None


def normalize_tool_name(tool: Any, message: str = "") -> str | None:
    raw = extract_tool_wrapper_from_value(tool)
    if raw:
        return raw
    return clean_tool_name(tool)


def _extract_step_fields(raw_step: Any, max_text_chars: int) -> dict[str, Any]:
    if not isinstance(raw_step, dict):
        text = _stringify(raw_step, max_text_chars)
        return {
            "role": "unknown",
            "tool_wrapper": None,
            "semantic_tool": semantic_tool_from_command(text),
            "command_string": text,
            "tool_name": semantic_tool_from_command(text),
            "message_text": text,
            "message_text_full": text,
            "tool_args_len": 0,
            "observation_text": "",
            "observation_text_full": "",
            "observation_len_chars": 0,
        }

    role = _first_value(raw_step, ["role", "src", "type", "actor", "speaker", "source", "kind"])

    message_raw = _first_value(
        raw_step,
        [
            "content",
            "msg",
            "message",
            "text",
            "thought",
            "prompt",
            "response",
            "system_prompt",
            "assistant",
            "action",
        ],
    )
    message_text_full = _stringify(message_raw)
    message_text = _stringify(message_raw, max_text_chars)

    tool_wrapper = extract_tool_wrapper(raw_step)
    command_string = extract_command_string(raw_step, message_text_full)
    semantic_tool = semantic_tool_from_command(command_string, tool_wrapper)
    tool_name = semantic_tool

    args_raw = _first_value(
        raw_step,
        ["arguments", "args", "input", "tool_input", "command", "action_input", "parameters", "tools"],
    )
    tool_args_len = len(_stringify(args_raw)) if args_raw is not None else 0
    if tool_args_len == 0 and command_string:
        tool_args_len = len(command_string)

    obs_raw = _first_value(
        raw_step,
        [
            "observation",
            "obs",
            "output",
            "result",
            "stdout",
            "stderr",
            "tool_output",
            "tool_result",
            "environment_response",
        ],
    )
    observation_full = _stringify(obs_raw)
    observation_text = _stringify(obs_raw, max_text_chars)
    return {
        "role": _stringify(role)[:64] if role is not None else None,
        "tool_name": tool_name,
        "tool_wrapper": tool_wrapper,
        "semantic_tool": semantic_tool,
        "command_string": command_string,
        "message_text": message_text,
        "message_text_full": message_text_full,
        "tool_args_len": tool_args_len,
        "observation_text": observation_text,
        "observation_text_full": observation_full,
        "observation_len_chars": len(observation_full),
    }


def _looks_like_swe_observation(role: str | None, text: str) -> bool:
    if (role or "").lower() != "user":
        return False
    markers = (
        "(Open file:",
        "(Current directory:",
        "bash-$",
        "Traceback (most recent call last):",
        "Command exited",
        "Directory ",
        "File not found",
        "No such file",
    )
    return any(marker in (text or "") for marker in markers)


def _hash_text(prefix: str, text: str, length: int = 16) -> str:
    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:length]
    return f"{prefix}:{digest}"


def _extract_paths(text: str, limit: int = 12) -> list[str]:
    seen: set[str] = set()
    paths: list[str] = []
    for match in PATH_RE.finditer(text or ""):
        path = match.group(1).strip(".,;:()[]{}'\"")
        if path and path not in seen:
            seen.add(path)
            paths.append(path)
            if len(paths) >= limit:
                break
    return paths


def _extract_test_ids(text: str, limit: int = 4) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for pat in (r"FAILED\s+([^\s]+)", r"([\w./-]+\.py::[^\s]+)", r"(tests?/[^\s:]+)"):
        for match in re.finditer(pat, text or ""):
            test_id = match.group(1).strip(".,;")
            if test_id and test_id not in seen:
                seen.add(test_id)
                ids.append(test_id)
                if len(ids) >= limit:
                    return ids
    return ids


def _message_file_access_type(step: Step, command_string: str | None) -> str:
    if command_has_explicit_write(command_string, step.semantic_tool):
        return "write"
    if step.phase == "execute/test":
        return "execute"
    if step.phase == "retrieve/browser":
        return "retrieve"
    return "read"


def _observation_file_access_type(step: Step, observation: str) -> str:
    if step.phase == "execute/test" or TEST_RE.search(observation or ""):
        return "execute_result"
    if step.phase == "retrieve/browser":
        return "retrieve_result"
    return "mention"


def _make_object(
    step: Step,
    object_type: str,
    object_id: str,
    size_chars: int,
    access_type: str,
    object_source: str,
    stable_object: bool,
) -> ObjectAccess:
    return ObjectAccess(
        trace_id=step.trace_id,
        step_id=step.step_id,
        object_type=object_type,
        object_id=object_id,
        size_chars=size_chars,
        access_type=access_type,
        object_source=object_source,
        stable_object=stable_object,
        phase=step.phase,
        tool_name=step.tool_name,
        tool_wrapper=step.tool_wrapper,
        semantic_tool=step.semantic_tool,
    )


def infer_object_accesses(step: Step, full_message: str, full_observation: str) -> list[ObjectAccess]:
    accesses: list[ObjectAccess] = []
    message_blob = f"{full_message}\n{step.command_string or ''}"
    message_paths = set(_extract_paths(message_blob))
    observation_paths = set(_extract_paths(full_observation))
    all_paths = sorted(message_paths | observation_paths)

    for path in all_paths:
        if path in message_paths and path in observation_paths:
            source = "both"
            access_type = (
                "write"
                if command_has_explicit_write(step.command_string, step.semantic_tool)
                else _observation_file_access_type(step, full_observation)
            )
        elif path in message_paths:
            source = "message"
            access_type = _message_file_access_type(step, step.command_string)
        else:
            source = "observation"
            access_type = _observation_file_access_type(step, full_observation)
        accesses.append(
            _make_object(
                step,
                object_type="file",
                object_id=f"file:{path}",
                size_chars=max(step.observation_len_chars if source != "message" else 0, len(path)),
                access_type=access_type,
                object_source=source,
                stable_object=True,
            )
        )

    if full_observation:
        obs_type = "observation"
        obs_access = "read"
        if step.phase == "retrieve/browser":
            obs_type = "browser_page"
            obs_access = "retrieve_result"
        elif step.phase == "execute/test" or TEST_RE.search(full_observation):
            obs_type = "test_log"
            obs_access = "execute_result"
        object_id = _hash_text(obs_type, full_observation)
        accesses.append(
            _make_object(
                step,
                obs_type,
                object_id,
                step.observation_len_chars,
                obs_access,
                "observation",
                False,
            )
        )

    for test_id in _extract_test_ids(full_observation):
        accesses.append(
            _make_object(
                step,
                "test_case",
                f"test:{test_id}",
                max(step.observation_len_chars, len(test_id)),
                "execute_result",
                "observation",
                True,
            )
        )

    for url in URL_RE.findall(f"{full_message}\n{full_observation}"):
        source = "both" if url in full_message and url in full_observation else "message" if url in full_message else "observation"
        accesses.append(
            _make_object(
                step,
                "browser_page",
                f"url:{url[:200]}",
                max(step.observation_len_chars if source != "message" else 0, len(url)),
                "retrieve" if source == "message" else "retrieve_result",
                source,
                True,
            )
        )

    if step.observation_len_chars >= 2000:
        accesses.append(
            _make_object(
                step,
                "large_observation_bucket",
                f"large_obs:{step.phase}:{step.semantic_tool or 'unknown'}",
                step.observation_len_chars,
                "read",
                "synthetic",
                False,
            )
        )

    return accesses


def _trace_id(dataset_short: str, row: dict[str, Any], idx: int, keys: list[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            safe = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(value))[:160]
            return f"{dataset_short}:{safe}"
    return f"{dataset_short}:row_{idx:06d}"


def _normalize_one_trace(
    dataset_short: str,
    row: dict[str, Any],
    idx: int,
    warnings: list[str],
    max_text_chars: int,
) -> tuple[Trace, list[Step], list[ObjectAccess]]:
    if dataset_short == "terminalbench":
        trace_id = _trace_id(dataset_short, row, idx, ["trial_id", "trial_name", "task_name"])
        reward = _safe_float(row.get("reward"))
        success = bool(reward and reward > 0) if reward is not None else None
        raw_steps = row.get("steps")
        trace = Trace(
            trace_id=trace_id,
            dataset="terminalbench",
            task_id=_stringify(row.get("task_name")) or None,
            model=_stringify(row.get("model")) or None,
            agent_or_harness=_stringify(row.get("agent")) or None,
            success=success,
            reward=reward,
            resolved=success,
            duration_s=_safe_float(row.get("duration_seconds") or row.get("duration_s")),
            input_tokens=_safe_float(row.get("input_tokens")),
            output_tokens=_safe_float(row.get("output_tokens")),
            cache_tokens=_safe_float(row.get("cache_tokens")),
        )
    elif dataset_short == "swe_agent":
        trace_id = _trace_id(dataset_short, row, idx, ["instance_id", "id"])
        resolved = _safe_bool(row.get("target") if "target" in row else row.get("resolved"))
        success = resolved
        raw_steps = row.get("trajectory") or row.get("steps") or row.get("messages")
        trace = Trace(
            trace_id=trace_id,
            dataset="swe_agent",
            task_id=_stringify(row.get("instance_id")) or None,
            model=_stringify(row.get("model_name") or row.get("model")) or None,
            agent_or_harness="SWE-agent",
            success=success,
            reward=1.0 if success else 0.0 if success is not None else None,
            resolved=resolved,
            duration_s=_safe_float(row.get("duration_seconds") or row.get("duration_s")),
            input_tokens=_safe_float(row.get("input_tokens")),
            output_tokens=_safe_float(row.get("output_tokens")),
            cache_tokens=_safe_float(row.get("cache_tokens")),
        )
    else:
        trace_id = _trace_id(dataset_short, row, idx, ["trace_id", "id", "task_id"])
        success = _safe_bool(row.get("success") or row.get("resolved"))
        raw_steps = row.get("steps") or row.get("trajectory") or row.get("messages")
        trace = Trace(
            trace_id=trace_id,
            dataset=dataset_short,
            task_id=_stringify(row.get("task_id") or row.get("instance_id")) or None,
            model=_stringify(row.get("model")) or None,
            agent_or_harness=_stringify(row.get("agent") or row.get("agent_or_harness")) or None,
            success=success,
            reward=_safe_float(row.get("reward")),
            resolved=_safe_bool(row.get("resolved")),
            duration_s=_safe_float(row.get("duration_s") or row.get("duration_seconds")),
            input_tokens=_safe_float(row.get("input_tokens")),
            output_tokens=_safe_float(row.get("output_tokens")),
            cache_tokens=_safe_float(row.get("cache_tokens")),
        )

    raw_step_list = _parse_steps(raw_steps, warnings, trace_id)
    steps: list[Step] = []
    objects: list[ObjectAccess] = []
    for step_id, raw_step in enumerate(raw_step_list):
        fields = _extract_step_fields(raw_step, max_text_chars=max_text_chars)
        phase = classify_phase_command(
            command_string=fields.get("command_string"),
            semantic_tool=fields.get("semantic_tool"),
            tool_wrapper=fields.get("tool_wrapper"),
            message_text=fields["message_text_full"],
        )
        full_obs = fields.get("observation_text_full", "")
        full_msg = fields.get("message_text_full", "")
        error_flag = bool(ERROR_RE.search(f"{full_msg}\n{full_obs}"))
        if dataset_short == "swe_agent" and _looks_like_swe_observation(fields["role"], full_msg) and steps:
            prev = steps[-1]
            if (prev.role or "").lower() in {"ai", "assistant", "agent"}:
                prev.observation_text = _stringify(full_msg, max_text_chars)
                prev.observation_len_chars = len(full_msg)
                prev.observation_tokens_est = estimate_tokens(full_msg)
                prev.error_flag = bool(prev.error_flag or error_flag)
                objects.extend(infer_object_accesses(prev, "", full_msg))
                continue
        step = Step(
            trace_id=trace_id,
            step_id=step_id,
            role=fields["role"],
            phase=phase,
            tool_name=fields["tool_name"],
            tool_wrapper=fields.get("tool_wrapper"),
            semantic_tool=fields.get("semantic_tool"),
            command_string=fields.get("command_string"),
            message_text=fields["message_text"],
            message_tokens_est=estimate_tokens(full_msg),
            tool_args_len=fields["tool_args_len"],
            observation_text=fields["observation_text"],
            observation_len_chars=fields["observation_len_chars"],
            observation_tokens_est=estimate_tokens(full_obs),
            error_flag=error_flag,
        )
        steps.append(step)
        objects.extend(infer_object_accesses(step, full_msg, full_obs))

    trace.total_steps = len(steps)
    return trace, steps, objects


def normalize_rows(
    dataset: str,
    rows: list[dict[str, Any]],
    max_text_chars: int = 20000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Normalize raw rows from one known dataset into canonical DataFrames."""

    warnings: list[str] = []
    dataset_l = dataset.lower()
    if "terminalbench" in dataset_l:
        dataset_short = "terminalbench"
    elif "swe-agent" in dataset_l or "swe_agent" in dataset_l:
        dataset_short = "swe_agent"
    else:
        dataset_short = re.sub(r"[^a-z0-9_]+", "_", dataset_l).strip("_") or "unknown"

    traces: list[Trace] = []
    steps: list[Step] = []
    objects: list[ObjectAccess] = []
    for idx, row in enumerate(rows):
        try:
            trace, trace_steps, trace_objects = _normalize_one_trace(
                dataset_short=dataset_short,
                row=row,
                idx=idx,
                warnings=warnings,
                max_text_chars=max_text_chars,
            )
            traces.append(trace)
            steps.extend(trace_steps)
            objects.extend(trace_objects)
        except Exception as exc:
            LOGGER.exception("Failed to normalize row %s from %s", idx, dataset)
            warnings.append(f"{dataset_short}: failed to normalize row {idx}: {exc}")

    traces_df = pd.DataFrame([t.to_dict() for t in traces], columns=TRACE_COLUMNS)
    steps_df = pd.DataFrame([s.to_dict() for s in steps], columns=STEP_COLUMNS)
    objects_df = pd.DataFrame([o.to_dict() for o in objects], columns=OBJECT_COLUMNS)
    return traces_df, steps_df, objects_df, warnings


def concat_normalized(
    parts: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not parts:
        return (
            pd.DataFrame(columns=TRACE_COLUMNS),
            pd.DataFrame(columns=STEP_COLUMNS),
            pd.DataFrame(columns=OBJECT_COLUMNS),
        )
    traces = pd.concat([p[0] for p in parts], ignore_index=True) if parts else pd.DataFrame()
    steps = pd.concat([p[1] for p in parts], ignore_index=True) if parts else pd.DataFrame()
    objects = pd.concat([p[2] for p in parts], ignore_index=True) if parts else pd.DataFrame()
    return traces[TRACE_COLUMNS], steps[STEP_COLUMNS], objects[OBJECT_COLUMNS]
