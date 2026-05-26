"""Conservative prompt decomposition helpers.

This module intentionally avoids semantic merging. It only splits prompts when
the caller provides explicit segment markers or section labels.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptSegment:
    segment_id: str
    kind: str
    text: str

    @property
    def token_estimate(self) -> int:
        return max(1, len(self.text.split()))


def decompose_prompt(prompt: str, marker: str = "\n---STATE:") -> list[PromptSegment]:
    """Split a prompt into explicit state segments.

    A segment header has the form `---STATE:kind:id---`. If the marker is not
    present, the whole prompt is returned as one opaque segment.
    """

    if marker not in prompt:
        return [PromptSegment("prompt", "opaque_prompt", prompt)]

    segments: list[PromptSegment] = []
    preamble, *parts = prompt.split(marker)
    if preamble.strip():
        segments.append(PromptSegment("preamble", "opaque_prompt", preamble.strip()))
    for index, part in enumerate(parts):
        header, _, body = part.partition("---")
        fields = [field.strip() for field in header.strip(": \n").split(":") if field.strip()]
        kind = fields[0] if fields else "unknown"
        segment_id = fields[1] if len(fields) > 1 else f"segment_{index}"
        segments.append(PromptSegment(segment_id, kind, body.strip()))
    return segments
