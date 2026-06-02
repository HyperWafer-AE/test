import json
import time
import uuid
from pathlib import Path
from typing import Iterable, Optional
from urllib import request as urlrequest


class LocalOpenAIClientWrapper:
    """Small OpenAI-compatible chat client that logs every request."""

    def __init__(
        self,
        server_url: str,
        model: str,
        output_dir: str = "traces/local_h100/raw_requests",
        server_backend: str = "local",
        timeout: float = 120.0,
    ):
        self.server_url = server_url.rstrip("/")
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.server_backend = server_backend
        self.timeout = float(timeout)

    def chat_completion(
        self,
        messages: list,
        workflow_id: str,
        agent_id: str,
        step_id: str,
        input_segments: Optional[list] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 64,
        stream: bool = False,
        extra_body: Optional[dict] = None,
    ) -> dict:
        request_id = uuid.uuid4().hex
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": bool(stream),
        }
        if extra_body:
            payload.update(extra_body)

        start = time.time()
        first_token = None
        error = None
        raw_response = None
        content = ""
        usage = {}
        retry_info = {"attempt": 1, "retries": 0}
        try:
            if stream:
                raw_response, content, usage, first_token = self._stream(payload, start)
            else:
                raw_response, content, usage = self._post_json(payload)
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
        end = time.time()

        prompt_tokens = _int_usage(usage, "prompt_tokens", _segment_tokens(input_segments))
        completion_tokens = _int_usage(usage, "completion_tokens", _estimate_tokens(content))
        total_tokens = _int_usage(usage, "total_tokens", prompt_tokens + completion_tokens)
        ttft_ms = ((first_token or end) - start) * 1000.0 if not error else None
        total_latency_ms = (end - start) * 1000.0
        tps = completion_tokens / (end - (first_token or start)) if completion_tokens and end > (first_token or start) else 0.0

        record = {
            "request_id": request_id,
            "workflow_id": workflow_id,
            "agent_id": agent_id,
            "step_id": step_id,
            "timestamp_start": start,
            "timestamp_first_token": first_token,
            "timestamp_end": end,
            "input_messages": messages,
            "input_segments": input_segments or [],
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "ttft_ms": ttft_ms,
            "total_latency_ms": total_latency_ms,
            "tokens_per_second": tps,
            "model": self.model,
            "server_backend": self.server_backend,
            "stream": bool(stream),
            "error": error,
            "retry_info": retry_info,
            "raw_response": raw_response,
        }
        self._append_record(record)
        return {"content": content, "usage": usage, "error": error, "request_id": request_id, "record": record}

    def _post_json(self, payload: dict) -> tuple:
        data = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(
            f"{self.server_url}/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlrequest.urlopen(req, timeout=self.timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        parsed = json.loads(body)
        content = _extract_content(parsed)
        usage = parsed.get("usage") or {}
        return parsed, content, usage

    def _stream(self, payload: dict, start: float) -> tuple:
        data = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(
            f"{self.server_url}/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        chunks = []
        content_parts = []
        usage = {}
        first_token = None
        with urlrequest.urlopen(req, timeout=self.timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                item = line[len("data:") :].strip()
                if item == "[DONE]":
                    break
                chunks.append(item)
                parsed = json.loads(item)
                usage = parsed.get("usage") or usage
                delta = (((parsed.get("choices") or [{}])[0]).get("delta") or {}).get("content")
                if delta:
                    if first_token is None:
                        first_token = time.time()
                    content_parts.append(delta)
        return chunks, "".join(content_parts), usage, first_token or start

    def _append_record(self, record: dict):
        path = self.output_dir / f"{record['workflow_id']}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _extract_content(parsed: dict) -> str:
    choices = parsed.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        return "\n".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)
    return "" if content is None else str(content)


def _int_usage(usage: dict, key: str, fallback: int) -> int:
    try:
        return int(usage.get(key, fallback))
    except Exception:
        return int(fallback)


def _segment_tokens(segments: Optional[Iterable[dict]]) -> int:
    return sum(int(item.get("tokens", 0) or 0) for item in (segments or []))


def _estimate_tokens(text: object) -> int:
    text = "" if text is None else str(text)
    return max(1, len(text.split()) or len(text) // 4 or 1)

