from __future__ import annotations

from dataclasses import dataclass

from waferagent.graph_ir import AgentNode
from waferagent.utils import sha256_text


FILLER = (
    "The system studies wafer-scale serving for multi-agent language model workflows. "
    "Agents exchange evidence, intermediate reasoning, tool observations, and summaries. "
    "The prompt is deterministic so token lengths and shared prefixes are reproducible. "
)


@dataclass(frozen=True)
class PromptBundle:
    prompt: str
    prompt_hash: str
    shared_prefix_text: str
    shared_prefix_hash: str
    actual_prompt_tokens: int
    actual_shared_prefix_tokens: int
    actual_private_tokens: int


def _encode(tokenizer, text: str) -> list[int]:
    if tokenizer is None:
        return text.split()
    return tokenizer.encode(text, add_special_tokens=False)


def _decode(tokenizer, tokens: list[int] | list[str]) -> str:
    if tokenizer is None:
        return " ".join(str(t) for t in tokens)
    return tokenizer.decode(tokens, skip_special_tokens=True)


def text_for_token_len(tokenizer, target_tokens: int, seed_text: str) -> str:
    target = max(1, int(target_tokens))
    text = (seed_text + " " + FILLER) * max(1, target // 32 + 2)
    tokens = _encode(tokenizer, text)
    while len(tokens) < target:
        text += " " + FILLER
        tokens = _encode(tokenizer, text)
    return _decode(tokenizer, tokens[:target])


def prompt_for_node(tokenizer, node: AgentNode) -> PromptBundle:
    shared_tokens = max(0, int(node.shared_prefix_token_len))
    private_tokens = max(1, int(node.input_token_len) - shared_tokens)
    shared_seed = f"job shared context role={node.role} agent={node.agent_id}"
    private_seed = f"private task node={node.node_id} round={node.round_id}"
    shared_text = text_for_token_len(tokenizer, shared_tokens, shared_seed) if shared_tokens else ""
    private_text = text_for_token_len(tokenizer, private_tokens, private_seed)
    prompt = shared_text + ("\n" if shared_text else "") + private_text
    actual_prompt = len(_encode(tokenizer, prompt))
    actual_shared = len(_encode(tokenizer, shared_text)) if shared_text else 0
    actual_private = max(0, actual_prompt - actual_shared)
    return PromptBundle(
        prompt=prompt,
        prompt_hash=sha256_text(prompt),
        shared_prefix_text=shared_text,
        shared_prefix_hash=sha256_text(shared_text) if shared_text else "",
        actual_prompt_tokens=actual_prompt,
        actual_shared_prefix_tokens=actual_shared,
        actual_private_tokens=actual_private,
    )
