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

from src.normalize.schema import (
    OBJECT_COLUMNS,
    STEP_COLUMNS,
    TRACE_COLUMNS,
    ObjectAccess,
    Step,
    Trace,
)

LOGGER = logging.getLogger(__name__)

EXPLORE_RE = re.compile(
    r"\b(ls|dir|cat|grep|rg|find|fd|search|open|read|view|inspect|head|tail|less|tree)\b",
    re.I,
)
EDIT_RE = re.compile(
    r"\b(edit|write|patch|apply_patch|sed|create|modify|replace|insert|delete|touch|mv|cp)\b",
    re.I,
)
EXEC_RE = re.compile(
    r"\b(bash|sh|run|pytest|test|compile|python|node|npm|yarn|make|cargo|go test|mvn|gradle|unit)\b",
    re.I,
)
RETRIEVE_RE = re.compile(r"\b(browser|browse|fetch|web|url|http|https|wget|curl)\b", re.I)
VERIFY_RE = re.compile(r"\b(final|submit|answer|verifier|verify|resolved|finish)\b", re.I)

ERROR_RE = re.compile(
    r"(traceback|assertionerror|runtimeerror|exception|failed|error:|exit code [1-9]|timeout|not resolved)",
    re.I,
)
TEST_RE = re.compile(r"(pytest|unittest|failed|passed|assertion|test_|::test|tests?/)", re.I)
URL_RE = re.compile(r"https?://[^\s\]\)\"']+")
PATH_RE = re.compile(
    r"(?<![\w.-])((?:\.{1,2}/)?(?:[\w.-]+/)*[\w.-]+\."
    r"(?:py|js|ts|tsx|jsx|java|go|rs|c|cc|cpp|h|hpp|md|rst|txt|toml|yaml|yml|json|ini|sh|sql|rb|php|css|html|ipynb|cfg))"
)
FENCED_COMMAND_RE = re.compile(r"```(?:bash|sh|shell|console)?\s*\n(.*?)```", re.I | re.S)


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
    raw = _extract_nested_name(tool) if not isinstance(tool, str) else tool
    if raw is None:
        raw = ""
    raw = raw.strip()
    if not raw and message:
        first = message.strip().split(maxsplit=1)[0].strip("$`'\"") if message.strip() else ""
        if first in {
            "ls",
            "cat",
            "grep",
            "rg",
            "find",
            "sed",
            "python",
            "pytest",
            "bash",
            "sh",
            "make",
            "npm",
            "curl",
            "wget",
        }:
            raw = first
    if not raw:
        return None
    raw = raw.replace("\n", " ").strip("`'\" ")
    if len(raw) > 64 and " " in raw:
        raw = raw.split(maxsplit=1)[0]
    return raw.lower()[:80]


def _extract_fenced_command(message: str) -> str | None:
    candidates: list[str] = []
    for block in FENCED_COMMAND_RE.findall(message or ""):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if lines:
            candidates.append(lines[-1])
    if not candidates:
        return None
    command = candidates[-1].strip()
    if command.startswith("$ "):
        command = command[2:].strip()
    return command or None


def _tool_from_command(command: str | None) -> str | None:
    if not command:
        return None
    first = command.strip().split(maxsplit=1)[0].strip("`'\"")
    if not first:
        return None
    return first.lower()[:80]


def classify_phase(tool_name: str | None, message_text: str | None = None) -> str:
    combined = f"{tool_name or ''} {message_text or ''}"
    stripped = combined.strip().lower()
    tool_l = (tool_name or "").lower()
    if VERIFY_RE.search(stripped):
        return "verify/final"
    if any(x in tool_l for x in ("browser", "browse", "fetch", "web")):
        return "retrieve/browser"
    if RETRIEVE_RE.search(stripped):
        return "retrieve/browser"
    if any(x in tool_l for x in ("edit", "write", "patch", "create", "modify", "replace")):
        return "edit/write"
    if EDIT_RE.search(stripped):
        return "edit/write"
    if any(x in tool_l for x in ("read", "open", "grep", "find", "search", "glob", "list")):
        return "explore/read"
    if EXPLORE_RE.search(stripped):
        return "explore/read"
    if any(x in tool_l for x in ("bash", "pytest", "test", "run", "python", "compile")):
        return "execute/test"
    if EXEC_RE.search(stripped):
        return "execute/test"
    return "unknown"


def _extract_step_fields(raw_step: Any, max_text_chars: int) -> dict[str, Any]:
    if not isinstance(raw_step, dict):
        text = _stringify(raw_step, max_text_chars)
        return {
            "role": "unknown",
            "tool_name": normalize_tool_name(None, text),
            "message_text": text,
            "tool_args_len": 0,
            "observation_text": "",
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

    tool_raw = _first_value(
        raw_step,
        [
            "tool_name",
            "tool",
            "tools",
            "name",
            "command",
            "command_name",
            "action_name",
            "tool_call",
            "tool_calls",
            "function_call",
        ],
    )
    tool_name = normalize_tool_name(tool_raw, message_text_full)
    role_text = _stringify(role).lower() if role is not None else ""
    command_from_message = None
    if tool_name is None and role_text in {"ai", "assistant", "agent"}:
        command_from_message = _extract_fenced_command(message_text_full)
        tool_name = _tool_from_command(command_from_message)

    args_raw = _first_value(
        raw_step,
        ["arguments", "args", "input", "tool_input", "command", "action_input", "parameters", "tools"],
    )
    tool_args_len = len(_stringify(args_raw)) if args_raw is not None else 0
    if tool_args_len == 0 and command_from_message:
        tool_args_len = len(command_from_message)

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


def _extract_paths(text: str, limit: int = 6) -> list[str]:
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


def _access_type_for_phase(phase: str) -> str:
    if phase == "edit/write":
        return "write"
    if phase == "execute/test":
        return "execute"
    if phase == "retrieve/browser":
        return "retrieve"
    return "read"


def infer_object_accesses(step: Step, full_message: str, full_observation: str) -> list[ObjectAccess]:
    accesses: list[ObjectAccess] = []
    combined = f"{full_message}\n{full_observation}"
    access_type = _access_type_for_phase(step.phase)
    tool = step.tool_name or "unknown"

    for path in _extract_paths(combined):
        accesses.append(
            ObjectAccess(
                trace_id=step.trace_id,
                step_id=step.step_id,
                object_type="file",
                object_id=f"file:{path}",
                size_chars=max(step.observation_len_chars, len(path)),
                access_type=access_type,
                phase=step.phase,
                tool_name=step.tool_name,
            )
        )

    if full_observation:
        obs_type = "observation"
        if step.phase == "retrieve/browser":
            obs_type = "browser_page"
        elif step.phase == "execute/test" or TEST_RE.search(full_observation):
            obs_type = "test_log"
        object_id = _hash_text(obs_type, full_observation)
        accesses.append(
            ObjectAccess(
                trace_id=step.trace_id,
                step_id=step.step_id,
                object_type=obs_type,
                object_id=object_id,
                size_chars=step.observation_len_chars,
                access_type="read",
                phase=step.phase,
                tool_name=step.tool_name,
            )
        )

    for test_id in _extract_test_ids(full_observation):
        accesses.append(
            ObjectAccess(
                trace_id=step.trace_id,
                step_id=step.step_id,
                object_type="test_case",
                object_id=f"test:{test_id}",
                size_chars=max(step.observation_len_chars, len(test_id)),
                access_type="execute",
                phase=step.phase,
                tool_name=step.tool_name,
            )
        )

    for url in URL_RE.findall(combined):
        accesses.append(
            ObjectAccess(
                trace_id=step.trace_id,
                step_id=step.step_id,
                object_type="browser_page",
                object_id=f"url:{url[:200]}",
                size_chars=max(step.observation_len_chars, len(url)),
                access_type="retrieve",
                phase=step.phase,
                tool_name=step.tool_name,
            )
        )

    if step.observation_len_chars >= 2000:
        accesses.append(
            ObjectAccess(
                trace_id=step.trace_id,
                step_id=step.step_id,
                object_type="large_observation_bucket",
                object_id=f"large_obs:{step.phase}:{tool}",
                size_chars=step.observation_len_chars,
                access_type="read",
                phase=step.phase,
                tool_name=step.tool_name,
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
        phase = classify_phase(fields["tool_name"], fields["message_text_full"])
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
