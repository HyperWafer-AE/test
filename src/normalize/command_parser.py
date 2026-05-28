"""Command and tool parsing utilities for agent trace normalization.

The public traces often expose a wrapper-level tool name such as
``bash_command`` or ``execute_bash``.  Those wrappers are useful for debugging
the harness, but they are too coarse for research claims about agent state-flow
locality.  This module extracts a lower-level command string and maps it to a
semantic tool when possible.
"""

from __future__ import annotations

import json
import re
import shlex
from typing import Any

FENCED_COMMAND_RE = re.compile(r"```(?:bash|sh|shell|console)?\s*\n(.*?)```", re.I | re.S)

READ_TOOLS = {
    "ls",
    "dir",
    "cat",
    "grep",
    "rg",
    "find",
    "fd",
    "search",
    "search_file",
    "search_dir",
    "find_file",
    "open",
    "read",
    "view",
    "head",
    "tail",
    "less",
    "tree",
    "pwd",
}
EDIT_TOOLS = {
    "apply_patch",
    "sed",
    "perl",
    "edit",
    "write",
    "write_file",
    "replace",
    "str_replace",
    "str_replace_editor",
    "create",
    "touch",
    "mkdir",
    "mv",
    "cp",
    "rm",
    "echo",
    "tee",
    "end_of_edit",
}
EXEC_TOOLS = {
    "bash",
    "sh",
    "shell",
    "run",
    "pytest",
    "unittest",
    "python",
    "python3",
    "node",
    "npm",
    "npx",
    "yarn",
    "pnpm",
    "make",
    "cargo",
    "go",
    "mvn",
    "gradle",
    "gcc",
    "g++",
    "clang",
    "clang++",
    "java",
    "javac",
    "compile",
}
RETRIEVE_TOOLS = {"curl", "wget", "fetch", "browser", "browse", "web", "http", "https"}
VERIFY_TOOLS = {"final", "submit", "answer", "verifier", "verify", "finish"}

SHELL_WRAPPERS = {"bash", "sh", "shell", "bash_command", "execute_bash"}
TOOL_ALIASES = {
    "python3": "python",
    "py.test": "pytest",
    "ag": "grep",
    "ripgrep": "rg",
    "str_replace_editor": "edit",
    "str_replace": "edit",
    "write": "write_file",
}

COMMAND_KEY_CANDIDATES = (
    "cmd",
    "command",
    "command_string",
    "input",
    "tool_input",
    "action_input",
    "arguments",
    "args",
)
WRAPPER_KEY_PATHS = (
    ("tool_name",),
    ("tool",),
    ("name",),
    ("fn",),
    ("function", "name"),
    ("function_call", "name"),
    ("command_name",),
    ("action_name",),
)


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    except Exception:
        return str(value)


def clean_tool_name(value: Any) -> str | None:
    text = stringify(value).strip().replace("\n", " ").strip("`'\" ")
    if not text:
        return None
    if len(text) > 64 and " " in text:
        text = text.split(maxsplit=1)[0]
    return text.lower()[:80]


def _get_path(mapping: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = mapping
    for part in path:
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def extract_tool_wrapper(raw_step: dict[str, Any] | Any) -> str | None:
    if isinstance(raw_step, list):
        for item in raw_step:
            wrapper = extract_tool_wrapper(item)
            if wrapper:
                return wrapper
    if not isinstance(raw_step, dict):
        return clean_tool_name(raw_step)
    for key in ("tools", "tool_calls", "tool_call"):
        value = raw_step.get(key)
        wrapper = extract_tool_wrapper(value)
        if wrapper:
            return wrapper
    for path in WRAPPER_KEY_PATHS:
        value = _get_path(raw_step, path)
        wrapper = clean_tool_name(value)
        if wrapper:
            return wrapper
    return None


def extract_tool_wrapper_from_value(value: Any) -> str | None:
    return extract_tool_wrapper(value)


def _extract_first_tool_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                return item
    return None


def _extract_command_from_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, dict):
        for key in COMMAND_KEY_CANDIDATES:
            if key in value and value[key] not in (None, ""):
                if key in {"args", "arguments"} and isinstance(value[key], dict):
                    nested = _extract_command_from_value(value[key])
                    if nested:
                        return nested
                text = stringify(value[key]).strip()
                if text:
                    return text
        return None
    if isinstance(value, list):
        for item in value:
            text = _extract_command_from_value(item)
            if text:
                return text
    return None


def extract_fenced_command(message: str) -> str | None:
    candidates: list[str] = []
    for block in FENCED_COMMAND_RE.findall(message or ""):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if lines:
            command = lines[-1]
            if command.startswith("$ "):
                command = command[2:].strip()
            candidates.append(command)
    return candidates[-1] if candidates else None


def extract_command_string(raw_step: dict[str, Any] | Any, message_text: str = "") -> str | None:
    if isinstance(raw_step, dict):
        for key in ("tools", "tool_calls", "tool_call"):
            command = _extract_command_from_value(raw_step.get(key))
            if command:
                return command
        for key in COMMAND_KEY_CANDIDATES:
            command = _extract_command_from_value(raw_step.get(key))
            if command:
                return command
    command = extract_fenced_command(message_text)
    if command:
        return command
    return None


def _split_shell(command: str) -> list[str]:
    command = command.strip()
    if command.startswith("$ "):
        command = command[2:].strip()
    try:
        return shlex.split(command, posix=True)
    except Exception:
        return command.split()


def _basename(token: str) -> str:
    token = token.strip()
    if "/" in token:
        token = token.rsplit("/", 1)[-1]
    return token.lower()


def _strip_command_wrappers(tokens: list[str]) -> list[str]:
    while tokens and "=" in tokens[0] and not tokens[0].startswith("-"):
        tokens = tokens[1:]
    while tokens and _basename(tokens[0]) in {"env", "time", "timeout", "sudo", "xvfb-run"}:
        tokens = tokens[1:]
        while tokens and tokens[0].startswith("-"):
            tokens = tokens[1:]
    return tokens


def _inner_shell_command(tokens: list[str]) -> str | None:
    if not tokens:
        return None
    first = _basename(tokens[0])
    if first not in {"bash", "sh"}:
        return None
    for i, token in enumerate(tokens[1:], start=1):
        if token in {"-c", "-lc", "-lcx"} and i + 1 < len(tokens):
            return tokens[i + 1]
    return None


def semantic_tool_from_command(command_string: str | None, tool_wrapper: str | None = None) -> str:
    if command_string:
        tokens = _strip_command_wrappers(_split_shell(command_string))
        inner = _inner_shell_command(tokens)
        if inner:
            return semantic_tool_from_command(inner, tool_wrapper=tool_wrapper)
        if tokens:
            first = _basename(tokens[0])
            if first in {"python", "python3"}:
                if len(tokens) >= 3 and tokens[1] == "-m":
                    mod = _basename(tokens[2])
                    if mod in {"pytest", "unittest"}:
                        return "pytest" if mod == "pytest" else "unittest"
                if any("pytest" in tok for tok in tokens[1:]):
                    return "pytest"
                return "python"
            if first == "go" and len(tokens) > 1 and tokens[1] == "test":
                return "go_test"
            return TOOL_ALIASES.get(first, first)
    wrapper = clean_tool_name(tool_wrapper)
    if not wrapper:
        return "unknown"
    if wrapper in TOOL_ALIASES:
        return TOOL_ALIASES[wrapper]
    if wrapper in {"bash_command", "execute_bash"}:
        return "unknown"
    if wrapper in {"todo_write", "todowrite"}:
        return "edit"
    return wrapper


def command_has_explicit_write(command_string: str | None, semantic_tool: str | None = None) -> bool:
    command = command_string or ""
    tool = (semantic_tool or semantic_tool_from_command(command_string)).lower()
    if tool in {"apply_patch", "write_file", "edit", "replace", "str_replace", "create", "touch", "mkdir", "mv", "cp", "rm", "end_of_edit"}:
        return True
    if tool in {"sed", "perl"} and re.search(r"(^|\s)-i(\s|$|[.\w-])", command):
        return True
    if re.search(r"(^|\s)(cat|echo|printf|tee)\b.*?(>>?|tee\s+-a)\s+\S+", command, re.S):
        return True
    if re.search(r"\b(write_file|apply_patch|replace|edit)\b", command, re.I):
        return True
    return False


def classify_phase(
    command_string: str | None = None,
    semantic_tool: str | None = None,
    tool_wrapper: str | None = None,
    message_text: str | None = None,
) -> str:
    tool = (semantic_tool or semantic_tool_from_command(command_string, tool_wrapper)).lower()
    command = (command_string or "").lower()
    wrapper = (tool_wrapper or "").lower()

    if tool in VERIFY_TOOLS:
        return "verify/final"
    if tool in RETRIEVE_TOOLS or command.startswith(("curl ", "wget ")):
        return "retrieve/browser"
    if command_has_explicit_write(command_string, tool):
        return "edit/write"
    if tool in EDIT_TOOLS:
        return "edit/write"
    if tool in READ_TOOLS:
        return "explore/read"
    if tool in EXEC_TOOLS or tool == "go_test":
        return "execute/test"

    if wrapper in {"str_replace_editor", "write_file", "editor", "todowrite", "todo_write"}:
        return "edit/write"
    if wrapper in {"browser", "web_search", "fetch"}:
        return "retrieve/browser"
    if wrapper in {"final", "submit"}:
        return "verify/final"
    if wrapper in SHELL_WRAPPERS and command:
        return "execute/test"

    # Last-resort fallback: only trust terse command-like text, not arbitrary
    # reasoning paragraphs that mention "test", "create", or "verify".
    text = (message_text or "").strip()
    if text and "\n" not in text and len(text.split()) <= 8:
        fallback_tool = semantic_tool_from_command(text, tool_wrapper)
        if fallback_tool != "unknown":
            return classify_phase(text, fallback_tool, tool_wrapper, None)
    return "unknown"
