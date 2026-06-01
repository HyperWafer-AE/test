import math
from typing import Dict, Sequence


PHASES = ("explore", "execute", "verify", "failure", "finalize")

READ_TOOLS = {"Read", "Grep", "Glob", "WebFetch", "WebSearch"}
EXEC_TOOLS = {"Edit", "Write", "Bash"}
VERIFY_TOOLS = {"pytest", "test", "Verifier"}


def _softmax(scores: Dict[str, float]) -> Dict[str, float]:
    max_score = max(scores.values())
    exp_scores = {key: math.exp(value - max_score) for key, value in scores.items()}
    denom = sum(exp_scores.values())
    return {key: exp_scores[key] / denom for key in PHASES}


def detect_phase(recent_events: Sequence[dict], window: int = 8) -> Dict[str, float]:
    if not recent_events:
        return {
            "explore": 0.70,
            "execute": 0.10,
            "verify": 0.08,
            "failure": 0.04,
            "finalize": 0.08,
        }

    scores = {phase: 0.1 for phase in PHASES}
    for idx, event in enumerate(list(recent_events)[-window:]):
        weight = 1.0 + idx / max(1, window)
        tool = str(event.get("tool") or event.get("tool_name") or event.get("name") or "")
        event_type = str(event.get("type", ""))
        status = str(event.get("status", "ok")).lower()
        text = " ".join(str(event.get(key, "")) for key in ("message", "command", "new_state_type"))
        lowered = f"{event_type} {tool} {status} {text}".lower()

        if tool in READ_TOOLS:
            scores["explore"] += 1.8 * weight
        if tool in EXEC_TOOLS:
            scores["execute"] += 1.5 * weight
        if tool in VERIFY_TOOLS or "pytest" in lowered or "test" in lowered or "verifier" in lowered:
            scores["verify"] += 1.7 * weight
        if status not in {"ok", "success", ""}:
            scores["failure"] += 2.0 * weight
        if any(token in lowered for token in ("error", "exception", "traceback", "failed", "failure")):
            scores["failure"] += 1.5 * weight
        if any(token in lowered for token in ("final", "finalize", "report", "answer")):
            scores["finalize"] += 1.4 * weight

        if event_type == "llm":
            scores["execute"] += 0.4 * weight
        elif event_type == "tool":
            scores["explore"] += 0.2 * weight

    return _softmax(scores)

