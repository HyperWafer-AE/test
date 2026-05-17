from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from waferagent.graph_ir import AgentNode, NodeType
from waferagent.kv_model import ModelKVConfig, estimate_kv_bytes
from waferagent.trace_schema import TraceRecord
from waferagent.utils import sha256_text


@dataclass
class RunnerConfig:
    engine: str = "synthetic"
    model_name: str = "synthetic"
    model_path: str = ""
    gpu_id: int | None = None
    seed: int = 0
    synthetic_prefill_ms_per_token: float = 0.012
    synthetic_decode_ms_per_token: float = 0.42
    synthetic_batch_penalty: float = 1.0


class SyntheticRunner:
    def __init__(self, config: RunnerConfig, model_cfg: ModelKVConfig | None = None):
        self.config = config
        self.model_cfg = model_cfg or ModelKVConfig()

    def run_node(self, run_id: str, workload: str, node: AgentNode) -> TraceRecord:
        start = time.time()
        if node.node_type == NodeType.TOOL_CALL:
            ttft = 0.0
            decode = 0.0
            total = float(node.tool_latency_ms)
        else:
            prefill = (
                self.config.synthetic_prefill_ms_per_token * node.input_token_len
                + 1e-6 * node.input_token_len * node.input_token_len
            )
            decode = self.config.synthetic_decode_ms_per_token * max(1, node.actual_output_token_len)
            ttft = prefill + self.config.synthetic_decode_ms_per_token
            total = ttft + max(0.0, decode - self.config.synthetic_decode_ms_per_token)
        end = start + total / 1000.0
        output_key = f"{run_id}|{node.node_id}|{node.actual_output_token_len}|synthetic"
        kv_bytes = estimate_kv_bytes(node.input_token_len, self.model_cfg)
        return TraceRecord(
            schema_version="1.0",
            run_id=run_id,
            job_id=node.job_id,
            workload=workload,
            node_id=node.node_id,
            node_type=node.node_type.value,
            agent_id=node.agent_id,
            role=node.role,
            round_id=node.round_id,
            deps=list(node.deps),
            model_name=self.config.model_name,
            model_path=self.config.model_path,
            engine="synthetic",
            gpu_id=self.config.gpu_id,
            input_tokens=node.input_token_len,
            output_tokens=node.actual_output_token_len,
            shared_prefix_ids=list(node.shared_prefix_ids),
            private_prefix_ids=list(node.private_prefix_ids),
            prompt_hash=node.prompt_hash or sha256_text(node.node_id),
            start_time_unix=start,
            end_time_unix=end,
            ttft_ms=ttft,
            decode_ms=max(0.0, total - ttft),
            total_ms=total,
            tool_latency_ms=node.tool_latency_ms,
            kv_bytes_estimated=kv_bytes,
            cache_hit_tag="not_applicable",
            scheduler_tag="synthetic",
            output_hash=sha256_text(output_key),
            quality_proxy=None,
            shared_prefix_token_len=node.shared_prefix_token_len,
            private_prefix_token_len=node.private_prefix_token_len,
            metadata={},
        )


class HFRunner:
    def __init__(self, config: RunnerConfig, model_cfg: ModelKVConfig | None = None):
        self.config = config
        self.model_cfg = model_cfg or ModelKVConfig()
        self.tokenizer = None
        self.model = None
        self.torch = None
        self._load()

    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.torch = torch
            path = self.config.model_path or self.config.model_name
            self.tokenizer = AutoTokenizer.from_pretrained(
                path, trust_remote_code=True, local_files_only=Path(path).exists()
            )
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                path,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                local_files_only=Path(path).exists(),
            )
            self.model.eval()
        except Exception as exc:
            raise RuntimeError(f"HF model load failed: {exc}") from exc

    def _prompt(self, node: AgentNode) -> str:
        # Compact deterministic prompt; token length is controlled by repeated filler.
        body = " ".join(["context"] * max(1, node.input_token_len))
        return f"Role: {node.role}\nNode: {node.node_id}\n{body}\nAnswer briefly."

    def _generate_once(self, prompt: str, max_new_tokens: int) -> tuple[str, float]:
        assert self.tokenizer is not None and self.model is not None
        torch = self.torch
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.time() - start) * 1000.0
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return text, elapsed_ms

    def run_node(self, run_id: str, workload: str, node: AgentNode) -> TraceRecord:
        if node.node_type == NodeType.TOOL_CALL:
            return SyntheticRunner(
                RunnerConfig(engine="synthetic", model_name=self.config.model_name, model_path=self.config.model_path),
                self.model_cfg,
            ).run_node(run_id, workload, node)
        prompt = self._prompt(node)
        # Warmup is intentionally tiny per node; calibration script does explicit warmup.
        start = time.time()
        _, ttft = self._generate_once(prompt, 1)
        text, total = self._generate_once(prompt, max(1, node.actual_output_token_len))
        end = time.time()
        return TraceRecord(
            schema_version="1.0",
            run_id=run_id,
            job_id=node.job_id,
            workload=workload,
            node_id=node.node_id,
            node_type=node.node_type.value,
            agent_id=node.agent_id,
            role=node.role,
            round_id=node.round_id,
            deps=list(node.deps),
            model_name=self.config.model_name,
            model_path=self.config.model_path,
            engine="hf",
            gpu_id=self.config.gpu_id,
            input_tokens=node.input_token_len,
            output_tokens=node.actual_output_token_len,
            shared_prefix_ids=list(node.shared_prefix_ids),
            private_prefix_ids=list(node.private_prefix_ids),
            prompt_hash=node.prompt_hash or sha256_text(prompt),
            start_time_unix=start,
            end_time_unix=end,
            ttft_ms=ttft,
            decode_ms=max(0.0, total - ttft),
            total_ms=total,
            tool_latency_ms=node.tool_latency_ms,
            kv_bytes_estimated=estimate_kv_bytes(node.input_token_len, self.model_cfg),
            cache_hit_tag="unknown",
            scheduler_tag="hf_naive",
            output_hash=sha256_text(text),
            quality_proxy=None,
            shared_prefix_token_len=node.shared_prefix_token_len,
            private_prefix_token_len=node.private_prefix_token_len,
            metadata={},
        )


class VLLMRunner:
    def __init__(self, config: RunnerConfig, model_cfg: ModelKVConfig | None = None):
        self.config = config
        self.model_cfg = model_cfg or ModelKVConfig()
        try:
            from vllm import LLM, SamplingParams

            self.LLM = LLM
            self.SamplingParams = SamplingParams
            self.llm = LLM(model=config.model_path or config.model_name)
        except Exception as exc:
            raise RuntimeError(f"vLLM unavailable: {exc}") from exc

    def run_node(self, run_id: str, workload: str, node: AgentNode) -> TraceRecord:
        prompt = " ".join(["context"] * max(1, node.input_token_len))
        start = time.time()
        params = self.SamplingParams(max_tokens=max(1, node.actual_output_token_len), temperature=0.0)
        outputs = self.llm.generate([prompt], params)
        total = (time.time() - start) * 1000.0
        text = outputs[0].outputs[0].text if outputs else ""
        return TraceRecord(
            schema_version="1.0",
            run_id=run_id,
            job_id=node.job_id,
            workload=workload,
            node_id=node.node_id,
            node_type=node.node_type.value,
            agent_id=node.agent_id,
            role=node.role,
            round_id=node.round_id,
            deps=list(node.deps),
            model_name=self.config.model_name,
            model_path=self.config.model_path,
            engine="vllm",
            gpu_id=self.config.gpu_id,
            input_tokens=node.input_token_len,
            output_tokens=node.actual_output_token_len,
            shared_prefix_ids=list(node.shared_prefix_ids),
            private_prefix_ids=list(node.private_prefix_ids),
            prompt_hash=node.prompt_hash or sha256_text(prompt),
            start_time_unix=start,
            end_time_unix=start + total / 1000.0,
            ttft_ms=total,
            decode_ms=0.0,
            total_ms=total,
            tool_latency_ms=node.tool_latency_ms,
            kv_bytes_estimated=estimate_kv_bytes(node.input_token_len, self.model_cfg),
            cache_hit_tag="unknown",
            scheduler_tag="vllm",
            output_hash=sha256_text(text),
            quality_proxy=None,
            shared_prefix_token_len=node.shared_prefix_token_len,
            private_prefix_token_len=node.private_prefix_token_len,
            metadata={},
        )


def make_runner(config: RunnerConfig, model_cfg: ModelKVConfig | None = None):
    if config.engine == "hf":
        return HFRunner(config, model_cfg)
    if config.engine == "vllm":
        return VLLMRunner(config, model_cfg)
    return SyntheticRunner(config, model_cfg)
