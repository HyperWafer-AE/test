from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from waferagent.graph_ir import AgentNode, NodeType
from waferagent.kv_model import ModelKVConfig, estimate_kv_bytes
from waferagent.prompts import prompt_for_node
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
            actual_prompt_tokens=node.input_token_len,
            actual_completion_tokens=node.actual_output_token_len,
            measured_ttft_ms=ttft,
            measured_tpot_ms=(max(0.0, total - ttft) / max(1, node.actual_output_token_len - 1)),
            measured_total_ms=total,
            peak_gpu_mem_bytes=None,
            real_trace=False,
            fallback_used=False,
            actual_shared_prefix_tokens=node.shared_prefix_token_len,
            actual_private_tokens=node.private_prefix_token_len,
            shared_prefix_hash=node.shared_prefix_ids[0] if node.shared_prefix_ids else "",
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

    def _generate_once(self, prompt: str, max_new_tokens: int) -> tuple[str, float, int, int]:
        assert self.tokenizer is not None and self.model is not None
        torch = self.torch
        inputs = self.tokenizer(prompt, return_tensors="pt")
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
        else:
            starter = ender = None
        start = time.time()
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        if torch.cuda.is_available():
            ender.record()
            torch.cuda.synchronize()
            elapsed_ms = float(starter.elapsed_time(ender))
            peak_mem = int(torch.cuda.max_memory_allocated())
        else:
            elapsed_ms = (time.time() - start) * 1000.0
            peak_mem = 0
        completion_tokens = max(0, int(out.shape[-1]) - prompt_tokens)
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return text, elapsed_ms, completion_tokens, peak_mem

    def run_node(self, run_id: str, workload: str, node: AgentNode) -> TraceRecord:
        if node.node_type == NodeType.TOOL_CALL:
            rec = SyntheticRunner(
                RunnerConfig(engine="synthetic", model_name=self.config.model_name, model_path=self.config.model_path),
                self.model_cfg,
            ).run_node(run_id, workload, node)
            rec.engine = "hf"
            rec.real_trace = False
            return rec
        bundle = prompt_for_node(self.tokenizer, node)
        # Warmup is intentionally tiny per node; calibration script does explicit warmup.
        start = time.time()
        _, ttft, _, mem1 = self._generate_once(bundle.prompt, 1)
        text, total, completion_tokens, mem2 = self._generate_once(bundle.prompt, max(1, node.actual_output_token_len))
        end = time.time()
        decode_ms = max(0.0, total - ttft)
        tpot = decode_ms / max(1, completion_tokens - 1)
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
            input_tokens=bundle.actual_prompt_tokens,
            output_tokens=completion_tokens,
            shared_prefix_ids=list(node.shared_prefix_ids),
            private_prefix_ids=list(node.private_prefix_ids),
            prompt_hash=bundle.prompt_hash,
            start_time_unix=start,
            end_time_unix=end,
            ttft_ms=ttft,
            decode_ms=decode_ms,
            total_ms=total,
            tool_latency_ms=node.tool_latency_ms,
            kv_bytes_estimated=estimate_kv_bytes(bundle.actual_prompt_tokens, self.model_cfg),
            cache_hit_tag="unknown",
            scheduler_tag="hf_naive",
            output_hash=sha256_text(text),
            quality_proxy=None,
            shared_prefix_token_len=bundle.actual_shared_prefix_tokens,
            private_prefix_token_len=bundle.actual_private_tokens,
            actual_prompt_tokens=bundle.actual_prompt_tokens,
            actual_completion_tokens=completion_tokens,
            measured_ttft_ms=ttft,
            measured_tpot_ms=tpot,
            measured_total_ms=total,
            peak_gpu_mem_bytes=max(mem1, mem2),
            real_trace=True,
            fallback_used=False,
            actual_shared_prefix_tokens=bundle.actual_shared_prefix_tokens,
            actual_private_tokens=bundle.actual_private_tokens,
            shared_prefix_hash=bundle.shared_prefix_hash,
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
            actual_prompt_tokens=node.input_token_len,
            actual_completion_tokens=node.actual_output_token_len,
            measured_ttft_ms=total,
            measured_tpot_ms=total / max(1, node.actual_output_token_len),
            measured_total_ms=total,
            peak_gpu_mem_bytes=None,
            real_trace=True,
            fallback_used=False,
            actual_shared_prefix_tokens=node.shared_prefix_token_len,
            actual_private_tokens=node.private_prefix_token_len,
            shared_prefix_hash=node.shared_prefix_ids[0] if node.shared_prefix_ids else "",
            metadata={},
        )


def make_runner(config: RunnerConfig, model_cfg: ModelKVConfig | None = None):
    if config.engine == "hf":
        return HFRunner(config, model_cfg)
    if config.engine == "vllm":
        return VLLMRunner(config, model_cfg)
    return SyntheticRunner(config, model_cfg)
