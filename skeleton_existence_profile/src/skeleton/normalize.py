from __future__ import annotations

import hashlib
import json
import re
from pathlib import PurePosixPath
from typing import Any

import pandas as pd

from .parser import (
    command_artifact_flag,
    error_signature,
    extract_command_string,
    extract_paths,
    extract_test_ids,
    extract_urls,
    extract_wrapper,
    has_explicit_write,
    is_no_tool_step,
    is_tool_action,
    object_dir,
    object_extension,
    object_prefix,
    phase_from_tool,
    semantic_tool_clean,
    semantic_tool_from_command,
    stringify,
    unique_preserve,
)
from .schema import DEPENDENCY_COLUMNS, OBJECT_COLUMNS, STEP_COLUMNS, TRACE_COLUMNS


ERROR_RE = re.compile(r"(traceback|assertionerror|runtimeerror|exception|failed|error:|exit code [1-9]|timeout)", re.I)


def _safe_float(x: Any) -> float | None:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def _parse_steps(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    if isinstance(raw, list):
        return [x if isinstance(x, dict) else {"msg": stringify(x)} for x in raw]
    return []


def _hash(text: str | None) -> str | None:
    if not text:
        return None
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _object_record(
    trace_id: str,
    step_id: int,
    object_id: str,
    object_type: str,
    access_type: str,
    source: str,
    size: int = 0,
    path: str | None = None,
    err_sig: str | None = None,
    derived_tool: bool = False,
    derived_obs: bool = False,
) -> dict[str, Any]:
    is_stable = object_type in {"file", "url", "test_case", "error_signature"} and object_id not in {"large_observation_bucket"}
    actionable = object_type in {"file", "url", "test_case", "error_signature"}
    path = path or (object_id.replace("file:", "") if object_type == "file" else None)
    return {
        "trace_id": trace_id,
        "step_id": step_id,
        "object_id": object_id,
        "object_type": object_type,
        "object_source": source,
        "object_version": None,
        "object_size": int(size or 0),
        "access_type": access_type,
        "stable_object": bool(is_stable),
        "actionable_object": bool(actionable),
        "object_path": path,
        "object_dir": object_dir(path) if path else None,
        "object_prefix": object_prefix(path) if path else None,
        "object_extension": object_extension(path) if path else None,
        "content_hash": None,
        "error_signature": err_sig,
        "derived_from_tool": bool(derived_tool),
        "derived_from_observation": bool(derived_obs),
    }


def _access_type(tool_clean: str, command: str | None, from_obs: bool) -> str:
    if from_obs:
        return "mention"
    if has_explicit_write(command, tool_clean) or tool_clean == "edit":
        return "write"
    if tool_clean in {"test", "execute"}:
        return "execute"
    if tool_clean == "retrieve":
        return "retrieve"
    return "read"


def normalize_terminalbench_rows(rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    steps_out: list[dict[str, Any]] = []
    objects: list[dict[str, Any]] = []
    skipped_rows = 0

    for idx, row in enumerate(rows):
        trace_id = str(row.get("trial_id") or row.get("trial_name") or f"terminalbench:{idx}")
        raw_steps = _parse_steps(row.get("steps"))
        if not raw_steps:
            skipped_rows += 1
            continue
        trace_step_records: list[dict[str, Any]] = []
        for step_id, step in enumerate(raw_steps):
            msg = stringify(step.get("msg") or step.get("message") or step.get("content"))
            obs = stringify(step.get("obs") or step.get("observation") or step.get("output"))
            wrapper = extract_wrapper(step)
            command = extract_command_string(step, msg)
            semantic = semantic_tool_from_command(command, wrapper)
            clean = semantic_tool_clean(semantic)
            artifact = command_artifact_flag(command, clean)
            no_tool = is_no_tool_step(wrapper, command, clean)
            tool_action = is_tool_action(wrapper, command, clean)
            phase, phase_source = phase_from_tool(clean, command)
            err = bool(ERROR_RE.search(obs or "") or error_signature(obs))
            rec = {
                "trace_id": trace_id,
                "step_id": step_id,
                "role": stringify(step.get("src") or step.get("role")) or None,
                "raw_tool_name": wrapper,
                "tool_wrapper": wrapper,
                "semantic_tool": semantic,
                "semantic_tool_clean": clean,
                "command_string": command,
                "phase": phase,
                "phase_clean": phase,
                "phase_source": phase_source,
                "is_tool_action": bool(tool_action),
                "is_no_tool_step": bool(no_tool),
                "command_artifact_flag": bool(artifact),
                "observation_len_chars": len(obs),
                "message_len_chars": len(msg),
                "error_flag": bool(err),
                "timestamp": step.get("timestamp") or step.get("time"),
            }
            steps_out.append(rec)
            trace_step_records.append(rec)

            for path in unique_preserve(extract_paths(command, 16)):
                objects.append(
                    _object_record(
                        trace_id,
                        step_id,
                        f"file:{path}",
                        "file",
                        _access_type(clean, command, False),
                        "command",
                        size=len(path),
                        path=path,
                        derived_tool=True,
                    )
                )
            for path in unique_preserve(extract_paths(obs, 32)):
                objects.append(
                    _object_record(
                        trace_id,
                        step_id,
                        f"file:{path}",
                        "file",
                        _access_type(clean, command, True),
                        "observation",
                        size=len(path),
                        path=path,
                        derived_obs=True,
                    )
                )
            for url in unique_preserve(extract_urls(command, 8) + extract_urls(obs, 8)):
                objects.append(
                    _object_record(trace_id, step_id, f"url:{url}", "url", "retrieve", "mixed", len(url), url, derived_tool=url in (command or ""), derived_obs=url in (obs or ""))
                )
            for test_id in unique_preserve(extract_test_ids(obs, 8) + extract_test_ids(command, 4)):
                objects.append(
                    _object_record(trace_id, step_id, f"test:{test_id}", "test_case", "execute", "mixed", len(test_id), test_id, derived_obs=test_id in (obs or ""))
                )
            sig = error_signature(obs)
            if sig:
                objects.append(
                    _object_record(trace_id, step_id, f"error:{sig}", "error_signature", "error", "observation", len(sig), None, sig, derived_obs=True)
                )
            if len(obs) > 4000:
                objects.append(
                    _object_record(trace_id, step_id, "large_observation_bucket", "large_observation_bucket", "mention", "observation", len(obs), None, None, derived_obs=True)
                )

        tool_steps = sum(1 for s in trace_step_records if s["is_tool_action"])
        traces.append(
            {
                "trace_id": trace_id,
                "dataset": "terminalbench",
                "task_id": row.get("task_name"),
                "harness": row.get("agent"),
                "agent_scaffold": row.get("agent"),
                "model": row.get("model"),
                "success": bool(row.get("reward")) if row.get("reward") is not None else None,
                "reward": _safe_float(row.get("reward")),
                "resolved": bool(row.get("reward")) if row.get("reward") is not None else None,
                "total_steps": len(trace_step_records),
                "tool_action_steps": tool_steps,
                "duration": _safe_float(row.get("duration_seconds")),
                "input_tokens": _safe_float(row.get("input_tokens")),
                "output_tokens": _safe_float(row.get("output_tokens")),
            }
        )

    steps = pd.DataFrame(steps_out, columns=STEP_COLUMNS)
    trace_df = pd.DataFrame(traces, columns=TRACE_COLUMNS)
    object_df = pd.DataFrame(objects, columns=OBJECT_COLUMNS)
    deps = build_dependencies(steps, object_df)
    meta = {
        "skipped_rows": skipped_rows,
        "num_traces": int(len(trace_df)),
        "num_steps": int(len(steps)),
        "num_tool_action_steps": int(steps["is_tool_action"].sum()) if not steps.empty else 0,
        "num_object_accesses": int(len(object_df)),
        "unknown_rate": float((steps["semantic_tool_clean"].fillna("unknown") == "unknown").mean()) if not steps.empty else 0.0,
        "artifact_rate": float(steps["command_artifact_flag"].mean()) if not steps.empty else 0.0,
        "command_parse_success_rate": float(steps["command_string"].notna().mean()) if not steps.empty else 0.0,
    }
    return trace_df, steps, object_df, deps, meta


def build_dependencies(steps: pd.DataFrame, objects: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if steps.empty or objects.empty:
        return pd.DataFrame(columns=DEPENDENCY_COLUMNS)
    step_map = steps.set_index(["trace_id", "step_id"])
    obs_objects = objects[objects["derived_from_observation"].astype(bool) & objects["stable_object"].astype(bool)]
    cmd_objects = objects[objects["derived_from_tool"].astype(bool) & objects["stable_object"].astype(bool)]
    for _, src in obs_objects.iterrows():
        trace_id = src["trace_id"]
        src_step = int(src["step_id"])
        oid = src["object_id"]
        future = cmd_objects[
            (cmd_objects["trace_id"] == trace_id)
            & (cmd_objects["object_id"] == oid)
            & (cmd_objects["step_id"] > src_step)
            & (cmd_objects["step_id"] <= src_step + horizon)
        ]
        for _, dst in future.iterrows():
            dst_step = int(dst["step_id"])
            try:
                src_tool = step_map.loc[(trace_id, src_step)]["semantic_tool_clean"]
                dst_tool = step_map.loc[(trace_id, dst_step)]["semantic_tool_clean"]
            except Exception:
                src_tool = dst_tool = "unknown"
            otype = src["object_type"]
            dep_type = "object_reused"
            if otype == "file":
                dep_type = "file_path_from_output_to_next_arg"
                if str(src_tool) == "test" or str(src.get("error_signature")) not in {"", "None", "nan"}:
                    dep_type = "error_file_to_read_or_edit"
            elif otype == "url":
                dep_type = "url_from_search_to_open"
            elif otype == "test_case":
                dep_type = "test_case_to_rerun"
            rows.append(
                {
                    "trace_id": trace_id,
                    "src_step_id": src_step,
                    "dst_step_id": dst_step,
                    "dependency_type": dep_type,
                    "src_tool": src_tool,
                    "dst_tool": dst_tool,
                    "object_id": oid,
                    "value_type": otype,
                    "confidence": "medium" if dst_step - src_step > 1 else "high",
                }
            )
    return pd.DataFrame(rows, columns=DEPENDENCY_COLUMNS).drop_duplicates()
