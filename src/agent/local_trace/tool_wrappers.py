import difflib
import glob as globlib
import hashlib
import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Optional


class ToolLogger:
    def __init__(
        self,
        workflow_id: str,
        agent_id: str,
        builder,
        raw_dir: str = "traces/local_h100/raw_tools",
        artifact_dir: str = "traces/local_h100/tool_outputs",
    ):
        self.workflow_id = workflow_id
        self.agent_id = agent_id
        self.builder = builder
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir = Path(artifact_dir) / workflow_id
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def glob(self, pattern: str, cwd: Path, step_id: str):
        start = time.time()
        matches = sorted(globlib.glob(str(cwd / pattern), recursive=True))
        rel = [str(Path(item).resolve().relative_to(cwd.resolve())) for item in matches if Path(item).is_file()]
        output = "\n".join(rel)
        return self._finish("Glob", {"pattern": pattern}, output, start, step_id, "ok", {"paths": rel})

    def grep(self, pattern: str, cwd: Path, step_id: str, include: str = "*.py"):
        start = time.time()
        matches = []
        for path in sorted(cwd.rglob(include)):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for idx, line in enumerate(text.splitlines(), 1):
                if pattern.lower() in line.lower():
                    matches.append(f"{path.relative_to(cwd)}:{idx}:{line}")
        output = "\n".join(matches)
        return self._finish("Grep", {"pattern": pattern, "include": include}, output, start, step_id, "ok", {})

    def read(self, path: Path, cwd: Path, step_id: str, start_line: int = 1, end_line: Optional[int] = None):
        start = time.time()
        rel = str(path.relative_to(cwd)) if path.is_absolute() else str(path)
        full = path if path.is_absolute() else cwd / path
        text = full.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        end_line = end_line or len(lines)
        snippet = "\n".join(f"{idx}: {line}" for idx, line in enumerate(lines[start_line - 1 : end_line], start_line))
        return self._finish(
            "Read",
            {"path": rel, "start_line": start_line, "end_line": end_line},
            snippet,
            start,
            step_id,
            "ok",
            {"path": rel, "line_range": f"{start_line}-{end_line}", "version": file_hash(full)},
            state_type="file_context",
        )

    def edit_replace(self, path: Path, cwd: Path, old: str, new: str, step_id: str):
        start = time.time()
        rel = str(path.relative_to(cwd)) if path.is_absolute() else str(path)
        full = path if path.is_absolute() else cwd / path
        before = full.read_text(encoding="utf-8", errors="replace")
        status = "ok"
        error = None
        if old not in before:
            after = before
            status = "failed"
            error = "old_text_not_found"
        else:
            after = before.replace(old, new, 1)
            full.write_text(after, encoding="utf-8")
        diff = "\n".join(
            difflib.unified_diff(
                before.splitlines(),
                after.splitlines(),
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
                lineterm="",
                n=12,
            )
        )
        return self._finish(
            "Edit",
            {"path": rel, "old_hash": sha(old), "new_hash": sha(new)},
            diff or error or "no diff",
            start,
            step_id,
            status,
            {"path": rel, "error_type": error},
            state_type="edit_diff",
        )

    def bash(self, command: str, cwd: Path, step_id: str, timeout: int = 30, tool_name: str = "Bash"):
        start = time.time()
        env = dict(os.environ)
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        output = (proc.stdout or "") + ("\nSTDERR:\n" + proc.stderr if proc.stderr else "")
        status = "ok" if proc.returncode == 0 else "failed"
        state_type = None
        if "pytest" in command or tool_name.lower() in {"pytest", "test"}:
            tool_name = "Pytest"
            state_type = "tool_observation" if proc.returncode == 0 else "test_failure_summary"
        return self._finish(
            tool_name,
            {"command": command, "timeout": timeout},
            output,
            start,
            step_id,
            status,
            {"command": command, "exit_code": proc.returncode},
            state_type=state_type,
        )

    def _finish(self, tool, args, output, start, step_id, status, metadata, state_type=None):
        end = time.time()
        event_id = f"tool_{uuid.uuid4().hex[:12]}"
        output_path = None
        stored_output = output
        raw_tokens = self.builder.count_tokens(output)
        if len(output) > 12000:
            output_path = self.artifact_dir / f"{event_id}.txt"
            output_path.write_text(output, encoding="utf-8", errors="ignore")
            stored_output = output[:6000] + "\n...[truncated]...\n" + output[-3000:]
        prompt_tokens = self.builder.count_tokens(stored_output)
        states = self.builder.register_tool_output(
            tool,
            stored_output,
            status=status,
            state_type=state_type,
            producer_event_id=event_id,
            metadata={
                **metadata,
                "full_output_path": str(output_path) if output_path else None,
                "raw_tokens": raw_tokens,
                "prompt_tokens": prompt_tokens,
                "compression_ratio": prompt_tokens / raw_tokens if raw_tokens else 1.0,
            },
        )
        state = states[0]
        event = {
            "type": "tool",
            "event_id": event_id,
            "agent": self.agent_id,
            "turn": self.builder.turn,
            "tool": tool,
            "latency": int((end - start) * 1000),
            "output_tokens": state.tokens,
            "status": status,
            "new_state_id": state.state_id,
            "new_state_type": state.state_type,
            "timestamp_start": start,
            "timestamp_end": end,
            "metadata": {
                "workflow_id": self.workflow_id,
                "step_id": step_id,
                "args": args,
                "latency_ms": (end - start) * 1000,
                "stdout_stderr_hash": sha(output),
                "truncated_output": stored_output[:1000],
                "raw_tokens": raw_tokens,
                "prompt_tokens": prompt_tokens,
                "compression_ratio": prompt_tokens / raw_tokens if raw_tokens else 1.0,
                "full_output_path": str(output_path) if output_path else None,
                **metadata,
            },
        }
        self._append_raw({**event, "output_state_id": state.state_id, "output_state_type": state.state_type})
        return [item.event() for item in states], event, state

    def _append_raw(self, record: dict):
        path = self.raw_dir / f"{self.workflow_id}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def sha(text: object) -> str:
    return hashlib.sha256(("" if text is None else str(text)).encode("utf-8", errors="ignore")).hexdigest()


def file_hash(path: Path) -> str:
    try:
        return sha(path.read_text(encoding="utf-8", errors="replace"))[:16]
    except Exception:
        return "unknown"
