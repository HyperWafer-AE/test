from __future__ import annotations

import re
import shlex
from pathlib import PurePosixPath
from typing import Iterable


UNKNOWN = "unknown"
NO_TOOL = "no_tool"

FENCED_RE = re.compile(r"```(?:bash|sh|shell|console)?\s*\n(.*?)```", re.I | re.S)
PATH_RE = re.compile(
    r"(?<![\w@.-])((?:/|\.{1,2}/)?(?:[\w@+.-]+/)*[\w@+.-]+\."
    r"(?:py|js|ts|tsx|jsx|go|rs|java|c|cc|cpp|h|hpp|sh|md|txt|json|yaml|yml|toml|ini|cfg|csv|log|xml|html|css|sql)"
    r"(?=$|[^\w./-]))"
)
URL_RE = re.compile(r"https?://[^\s\]\)\"']+")
TEST_ID_RE = re.compile(r"(?:FAILED\s+)?((?:tests?/)?[\w./-]+\.py::[^\s]+|test_[\w.-]+)")
TRACEBACK_RE = re.compile(r"File \"([^\"]+\.py)\", line (\d+)|([A-Za-z_][\w.]*Error|Exception):?([^\n]*)")

READ_CMDS = {
    "cat",
    "head",
    "tail",
    "less",
    "more",
    "open",
    "view",
    "read",
    "grep",
    "rg",
    "ack",
    "findstr",
    "search",
    "search_file",
    "search_dir",
    "find_file",
    "ls",
    "find",
    "fd",
    "tree",
    "pwd",
}
TEST_CMDS = {"pytest", "unittest", "tox", "nox"}
EXEC_CMDS = {"python", "python3", "node", "npm", "npx", "yarn", "pnpm", "make", "go", "cargo", "mvn", "gradle", "bash", "sh"}
NET_CMDS = {"curl", "wget", "browser", "browse", "fetch", "web_search"}
FINAL_CMDS = {"submit", "final", "answer", "finish", "verify"}
WRITE_CMDS = {"apply_patch", "edit", "write_file", "create", "touch", "mkdir", "mv", "cp", "rm", "tee"}


def shell_split(command: str | None) -> list[str]:
    if not command:
        return []
    text = command.strip()
    if text.startswith("$ "):
        text = text[2:].strip()
    try:
        return shlex.split(text, posix=True)
    except Exception:
        return text.split()


def basename(token: str) -> str:
    return token.strip().strip("'\"").rsplit("/", 1)[-1].lower()


def extract_fenced_command(text: str | None) -> str | None:
    if not text:
        return None
    blocks = FENCED_RE.findall(text)
    if not blocks:
        return None
    lines = [line.strip() for line in blocks[-1].splitlines() if line.strip()]
    if not lines:
        return None
    cmd = lines[-1]
    return cmd[2:].strip() if cmd.startswith("$ ") else cmd


def extract_command_string(step: dict | None, message_text: str = "") -> str | None:
    step = step or {}
    for key in ("tools", "tool_calls", "tool_call"):
        value = step.get(key)
        if isinstance(value, list) and value:
            value = value[0]
        if isinstance(value, dict):
            for k in ("cmd", "command", "input", "arguments", "args", "tool_input"):
                if value.get(k):
                    return stringify(value[k])
        elif isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("cmd", "command", "input", "arguments", "args", "tool_input"):
        if step.get(key):
            return stringify(step[key])
    return extract_fenced_command(message_text)


def extract_wrapper(step: dict | None) -> str:
    step = step or {}
    for key in ("tools", "tool_calls", "tool_call"):
        value = step.get(key)
        if isinstance(value, list) and value:
            value = value[0]
        if isinstance(value, dict):
            for k in ("fn", "name", "tool_name", "tool"):
                if value.get(k):
                    return clean_name(value[k])
        elif isinstance(value, str) and value.strip():
            return clean_name(value)
    for key in ("tool_name", "tool", "name", "fn"):
        if step.get(key):
            return clean_name(step[key])
    return UNKNOWN


def stringify(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def clean_name(value) -> str:
    text = stringify(value).strip().strip("`'\"").replace("\n", " ")
    return (text.split()[0] if text else UNKNOWN).lower()


def command_artifact_flag(command: str | None, semantic_tool: str | None = None) -> bool:
    text = (command or "").strip()
    tool = semantic_tool or semantic_tool_from_command(command)
    if not text:
        return True
    if text.startswith((">>>", "...", "{", "}", "[", "]")):
        return True
    if len(text) > 4000:
        return True
    if "\n" in text and tool == UNKNOWN:
        return True
    return False


def has_explicit_write(command: str | None, semantic_tool: str | None = None) -> bool:
    text = command or ""
    tokens = shell_split(text)
    tool = (semantic_tool or semantic_tool_from_command(command)).lower()
    if tool in {"apply_patch", "edit", "write_file", "create", "touch", "mkdir", "mv", "cp", "rm"}:
        return True
    if tool in {"sed", "perl"} and re.search(r"(^|\s)-i(?:\s|$|[.\w-])", text):
        return True
    if tool == "tee" and len([t for t in tokens[1:] if not t.startswith("-")]) > 0:
        return True
    if re.search(r"(^|\s)(echo|printf|cat)\b.*?(>>?)\s+\S+", text, re.S):
        return True
    return False


def semantic_tool_from_command(command: str | None, wrapper: str | None = None) -> str:
    tokens = shell_split(command)
    while tokens and "=" in tokens[0] and not tokens[0].startswith("-"):
        tokens = tokens[1:]
    while tokens and basename(tokens[0]) in {"env", "time", "timeout", "sudo"}:
        tokens = tokens[1:]
        while tokens and tokens[0].startswith("-"):
            tokens = tokens[1:]
    if tokens:
        first = basename(tokens[0])
        if first in {"bash", "sh"} and "-c" in tokens:
            idx = tokens.index("-c")
            if idx + 1 < len(tokens):
                return semantic_tool_from_command(tokens[idx + 1], wrapper)
        if first in {"python", "python3"}:
            if "-m" in tokens:
                idx = tokens.index("-m")
                if idx + 1 < len(tokens) and basename(tokens[idx + 1]) in {"pytest", "unittest"}:
                    return basename(tokens[idx + 1])
            if any("pytest" in t for t in tokens[1:]):
                return "pytest"
            return "python"
        if first == "go" and len(tokens) > 1 and tokens[1] == "test":
            return "go_test"
        if first == "npm" and len(tokens) > 1 and tokens[1] == "test":
            return "npm_test"
        if first == "sed":
            return "edit" if has_explicit_write(command, "sed") else "sed_read"
        if first == "echo":
            return "write_file" if has_explicit_write(command, "echo") else "echo"
        if first in TEST_CMDS:
            return first
        if first in READ_CMDS | EXEC_CMDS | NET_CMDS | FINAL_CMDS | WRITE_CMDS:
            return {"py.test": "pytest", "ripgrep": "rg", "ag": "grep"}.get(first, first)
        return first if re.match(r"^[a-zA-Z0-9_.+-]+$", first) else UNKNOWN
    wrapper_clean = clean_name(wrapper)
    if wrapper_clean in {"bash", "bash_command", "execute_bash", "shell"}:
        return UNKNOWN
    return wrapper_clean


def semantic_tool_clean(tool: str | None) -> str:
    t = clean_name(tool)
    if t in {"grep", "rg", "ack", "findstr", "search", "search_file", "search_dir"}:
        return "search"
    if t in {"cat", "head", "tail", "less", "more", "open", "view", "read", "sed_read"}:
        return "read_file"
    if t in {"ls", "find", "fd", "tree", "find_file", "pwd"}:
        return "list_files"
    if t in {"edit", "apply_patch", "write_file", "create", "touch", "mkdir", "mv", "cp", "rm", "tee"}:
        return "edit"
    if t in {"pytest", "unittest", "tox", "nox", "go_test", "npm_test"}:
        return "test"
    if t in {"python", "node", "npm", "npx", "yarn", "pnpm", "make", "go", "cargo", "mvn", "gradle", "bash", "sh"}:
        return "execute"
    if t in NET_CMDS:
        return "retrieve"
    if t in FINAL_CMDS:
        return "final"
    if t in {"echo"}:
        return UNKNOWN
    return t or UNKNOWN


def phase_from_tool(tool_clean: str, command: str | None = None) -> tuple[str, str]:
    t = semantic_tool_clean(tool_clean)
    if t in {"search", "read_file", "list_files"}:
        return "explore/read", "semantic_tool"
    if t == "edit":
        return "edit/write", "semantic_tool"
    if t in {"test", "execute"}:
        return "execute/test", "semantic_tool"
    if t == "retrieve":
        return "retrieve/browser", "semantic_tool"
    if t == "final":
        return "verify/final", "semantic_tool"
    return UNKNOWN, UNKNOWN


def is_no_tool_step(wrapper: str | None, command: str | None, tool_clean: str | None) -> bool:
    return clean_name(wrapper) in {UNKNOWN, "", "none", "nan"} and not command and semantic_tool_clean(tool_clean) == UNKNOWN


def is_tool_action(wrapper: str | None, command: str | None, tool_clean: str | None) -> bool:
    if is_no_tool_step(wrapper, command, tool_clean):
        return False
    if command_artifact_flag(command, tool_clean):
        return False
    return semantic_tool_clean(tool_clean) != UNKNOWN


def extract_paths(text: str | None, limit: int = 64) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in PATH_RE.finditer(text or ""):
        p = m.group(1).strip(".,;:()[]{}'\"")
        if p and p not in seen:
            seen.add(p)
            out.append(p)
            if len(out) >= limit:
                break
    return out


def extract_urls(text: str | None, limit: int = 32) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in URL_RE.finditer(text or ""):
        u = m.group(0).rstrip(".,;)")
        if u not in seen:
            seen.add(u)
            out.append(u)
            if len(out) >= limit:
                break
    return out


def extract_test_ids(text: str | None, limit: int = 16) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in TEST_ID_RE.finditer(text or ""):
        tid = m.group(1).strip(".,;")
        if tid not in seen:
            seen.add(tid)
            out.append(tid)
            if len(out) >= limit:
                break
    return out


def error_signature(text: str | None) -> str | None:
    text = text or ""
    for m in TRACEBACK_RE.finditer(text):
        if m.group(1):
            return f"traceback:{PurePosixPath(m.group(1)).name}:line"
        if m.group(3):
            return f"exception:{m.group(3)}"
    failed = extract_test_ids(text, limit=1)
    if failed:
        return f"failed_test:{failed[0]}"
    if re.search(r"\b(timeout|timed out)\b", text, re.I):
        return "error:timeout"
    if re.search(r"\b(error|failed|exception)\b", text, re.I):
        return "error:generic"
    return None


def object_prefix(path: str) -> str:
    p = PurePosixPath(path)
    parts = [part for part in p.parts if part not in {".", "..", "/"}]
    if len(parts) >= 2:
        return "/".join(parts[:2])
    if parts:
        return parts[0]
    return UNKNOWN


def object_dir(path: str) -> str:
    parent = str(PurePosixPath(path).parent)
    return "." if parent in {"", "."} else parent


def object_extension(path: str) -> str:
    return PurePosixPath(path).suffix.lower() or ""


def unique_preserve(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out
