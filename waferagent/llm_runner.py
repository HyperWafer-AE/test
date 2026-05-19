from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from waferagent.graph_ir import AgentNode, NodeType
from waferagent.kv_model import ModelKVConfig, estimate_kv_bytes
from waferagent.prompts import PromptBundle, prompt_for_node
from waferagent.trace_schema import TraceRecord
from waferagent.utils import PROJECT_ROOT, configure_project_env, sha256_text


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
    max_new_tokens: int | None = None
    max_input_tokens: int | None = None


def _cap_prompt_bundle(tokenizer, bundle: PromptBundle, max_input_tokens: int | None) -> PromptBundle:
    if not max_input_tokens or bundle.actual_prompt_tokens <= max_input_tokens:
        return bundle
    ids = tokenizer.encode(bundle.prompt, add_special_tokens=False) if tokenizer is not None else bundle.prompt.split()
    ids = ids[: max(1, int(max_input_tokens))]
    prompt = tokenizer.decode(ids, skip_special_tokens=True) if tokenizer is not None else " ".join(ids)
    actual_prompt = len(ids)
    actual_shared = min(bundle.actual_shared_prefix_tokens, actual_prompt)
    return PromptBundle(
        prompt=prompt,
        prompt_hash=sha256_text(prompt),
        shared_prefix_text=bundle.shared_prefix_text,
        shared_prefix_hash=bundle.shared_prefix_hash,
        actual_prompt_tokens=actual_prompt,
        actual_shared_prefix_tokens=actual_shared,
        actual_private_tokens=max(0, actual_prompt - actual_shared),
    )


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
            timing_source="synthetic_model",
            timing_scope="node",
            timing_quality="exact_forward",
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
            rec.timing_source = "synthetic_model"
            rec.timing_quality = "walltime_approx"
            return rec
        bundle = _cap_prompt_bundle(self.tokenizer, prompt_for_node(self.tokenizer, node), self.config.max_input_tokens)
        # Warmup is intentionally tiny per node; calibration script does explicit warmup.
        start = time.time()
        _, ttft, _, mem1 = self._generate_once(bundle.prompt, 1)
        max_new = min(max(1, node.actual_output_token_len), int(self.config.max_new_tokens or node.actual_output_token_len))
        text, total, completion_tokens, mem2 = self._generate_once(bundle.prompt, max_new)
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
            timing_source="hf_generate_walltime",
            timing_scope="node",
            timing_quality="walltime_approx",
            metadata={},
        )


class VLLMRunner:
    def __init__(self, config: RunnerConfig, model_cfg: ModelKVConfig | None = None):
        configure_project_env()
        self.config = config
        self.model_cfg = model_cfg or ModelKVConfig()
        self.tokenizer = None
        try:
            from vllm import LLM, SamplingParams

            self.LLM = LLM
            self.SamplingParams = SamplingParams
            self.llm = LLM(
                model=config.model_path or config.model_name,
                download_dir=str(PROJECT_ROOT / ".cache" / "vllm"),
            )
            if hasattr(self.llm, "get_tokenizer"):
                self.tokenizer = self.llm.get_tokenizer()
        except Exception as exc:
            raise RuntimeError(f"vLLM unavailable: {exc}") from exc

    def run_node(self, run_id: str, workload: str, node: AgentNode) -> TraceRecord:
        return self.run_nodes(run_id, workload, [node])[0]

    def run_nodes(self, run_id: str, workload: str, nodes: list[AgentNode]) -> list[TraceRecord]:
        tool_records: list[tuple[int, TraceRecord]] = []
        llm_items: list[tuple[int, AgentNode, Any]] = []
        for idx, node in enumerate(nodes):
            if node.node_type == NodeType.TOOL_CALL:
                rec = SyntheticRunner(
                    RunnerConfig(engine="synthetic", model_name=self.config.model_name, model_path=self.config.model_path),
                    self.model_cfg,
                ).run_node(run_id, workload, node)
                rec.engine = "vllm"
                rec.real_trace = False
                rec.timing_source = "synthetic_model"
                rec.timing_scope = "node"
                rec.timing_quality = "walltime_approx"
                tool_records.append((idx, rec))
            else:
                llm_items.append((idx, node, _cap_prompt_bundle(self.tokenizer, prompt_for_node(self.tokenizer, node), self.config.max_input_tokens)))
        if not llm_items:
            return [rec for _, rec in sorted(tool_records)]

        prompts = [bundle.prompt for _, _, bundle in llm_items]
        max_tokens = max(1, max(min(node.actual_output_token_len, int(self.config.max_new_tokens or node.actual_output_token_len)) for _, node, _ in llm_items))
        start = time.time()
        params = self.SamplingParams(max_tokens=max_tokens, temperature=0.0)
        outputs = self.llm.generate(prompts, params)
        total = (time.time() - start) * 1000.0
        batch_records: list[tuple[int, TraceRecord]] = []
        batch_layer_id = sha256_text("|".join(node.node_id for _, node, _ in llm_items))[:24]
        prompt_tokens_total = sum(bundle.actual_prompt_tokens for _, _, bundle in llm_items)
        completion_counts = []
        for output in outputs:
            choice = output.outputs[0] if output.outputs else None
            token_ids = getattr(choice, "token_ids", None) if choice else None
            completion_counts.append(len(token_ids) if token_ids is not None else 0)
        completion_tokens_total = sum(completion_counts)
        for (idx, node, bundle), output in zip(llm_items, outputs):
            choice = output.outputs[0] if output.outputs else None
            text = choice.text if choice else ""
            token_ids = getattr(choice, "token_ids", None) if choice else None
            completion_tokens = len(token_ids) if token_ids is not None else node.actual_output_token_len
            metrics = getattr(output, "metrics", None)
            first_token_time = getattr(metrics, "first_token_time", None) if metrics is not None else None
            finished_time = getattr(metrics, "finished_time", None) if metrics is not None else None
            scheduler_time = getattr(metrics, "scheduler_time", None) if metrics is not None else None
            queue_time = getattr(metrics, "queue_time", None) if metrics is not None else None
            if first_token_time is not None and finished_time is not None:
                ttft = max(0.0, (float(first_token_time) - start) * 1000.0)
                node_total = max(ttft, (float(finished_time) - start) * 1000.0)
                decode = max(0.0, node_total - ttft)
                tpot = decode / max(1, completion_tokens - 1)
                timing_unavailable = False
            else:
                node_total = total
                ttft = -1.0
                decode = -1.0
                tpot = -1.0
                timing_unavailable = True
            batch_records.append(
                (
                    idx,
                    TraceRecord(
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
                        input_tokens=bundle.actual_prompt_tokens,
                        output_tokens=completion_tokens,
                        shared_prefix_ids=list(node.shared_prefix_ids),
                        private_prefix_ids=list(node.private_prefix_ids),
                        prompt_hash=bundle.prompt_hash,
                        start_time_unix=start,
                        end_time_unix=start + node_total / 1000.0,
                        ttft_ms=ttft,
                        decode_ms=decode,
                        total_ms=node_total,
                        tool_latency_ms=node.tool_latency_ms,
                        kv_bytes_estimated=estimate_kv_bytes(bundle.actual_prompt_tokens, self.model_cfg),
                        cache_hit_tag="unknown",
                        scheduler_tag="vllm_batched_layer",
                        output_hash=sha256_text(text),
                        quality_proxy=None,
                        shared_prefix_token_len=bundle.actual_shared_prefix_tokens,
                        private_prefix_token_len=bundle.actual_private_tokens,
                        actual_prompt_tokens=bundle.actual_prompt_tokens,
                        actual_completion_tokens=completion_tokens,
                        measured_ttft_ms=None if timing_unavailable else ttft,
                        measured_tpot_ms=None if timing_unavailable else tpot,
                        measured_total_ms=node_total,
                        peak_gpu_mem_bytes=None,
                        real_trace=True,
                        fallback_used=False,
                        actual_shared_prefix_tokens=bundle.actual_shared_prefix_tokens,
                        actual_private_tokens=bundle.actual_private_tokens,
                        shared_prefix_hash=bundle.shared_prefix_hash,
                        timing_source="vllm_engine_metrics",
                        timing_scope="batch_layer",
                        timing_quality="engine_reported" if not timing_unavailable else "walltime_approx",
                        metadata={
                            "timing_unavailable": timing_unavailable,
                            "scheduler_time": scheduler_time,
                            "queue_time": queue_time,
                            "batch_size": len(llm_items),
                            "batch_layer_id": batch_layer_id,
                            "batch_layer_walltime_ms": total,
                            "num_prompts": len(llm_items),
                            "prompt_tokens_total": prompt_tokens_total,
                            "completion_tokens_total": completion_tokens_total,
                            "tokens_per_second": (prompt_tokens_total + completion_tokens_total) / max(1e-9, total / 1000.0),
                        },
                    ),
                )
            )
        return [rec for _, rec in sorted(tool_records + batch_records, key=lambda x: x[0])]


def make_runner(config: RunnerConfig, model_cfg: ModelKVConfig | None = None):
    if config.engine == "hf":
        return HFRunner(config, model_cfg)
    if config.engine == "vllm":
        return VLLMRunner(config, model_cfg)
    return SyntheticRunner(config, model_cfg)
