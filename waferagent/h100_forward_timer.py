from __future__ import annotations

import gc
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from waferagent.utils import configure_project_env


@dataclass
class ForwardTimingResult:
    input_len: int
    output_len: int
    batch_size: int
    rep: int
    prefill_ms: float
    decode_ms: float
    tpot_ms: float
    total_ms: float
    peak_gpu_mem_bytes: int
    oom: bool
    dtype: str
    device: str
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class H100ForwardTimer:
    def __init__(
        self,
        model_path: str,
        gpu_id: int = 0,
        dtype: str = "bfloat16",
        seed: int = 0,
    ):
        configure_project_env()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; forward calibration requires a GPU")
        self.torch = torch
        self.device = torch.device(f"cuda:{gpu_id}")
        self.dtype_name = dtype
        self.seed = int(seed)
        dtype_obj = torch.bfloat16 if dtype == "bfloat16" else torch.float16 if dtype == "float16" else torch.float32
        local = Path(model_path).exists()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=local
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype_obj,
            local_files_only=local,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()
        self.vocab_size = int(getattr(self.model.config, "vocab_size", self.tokenizer.vocab_size))

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def _input_ids(self, batch_size: int, input_len: int, rep: int):
        torch = self.torch
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed + rep + input_len * 997 + batch_size * 37)
        low = 0
        high = max(2, self.vocab_size)
        return torch.randint(low, high, (batch_size, input_len), generator=generator, device=self.device)

    def run_case(
        self,
        input_len: int,
        output_len: int,
        batch_size: int,
        rep: int,
        warmup: int = 1,
    ) -> ForwardTimingResult:
        torch = self.torch
        try:
            input_ids = self._input_ids(batch_size, input_len, rep)
            decode_token = input_ids[:, -1:].contiguous()
            torch.cuda.reset_peak_memory_stats(self.device)
            for _ in range(max(0, warmup)):
                with torch.no_grad():
                    out = self.model(input_ids, use_cache=True)
                    past = out.past_key_values
                    token = decode_token
                    for _i in range(min(2, max(1, output_len))):
                        out = self.model(token, use_cache=True, past_key_values=past)
                        past = out.past_key_values
                        token = out.logits[:, -1:].argmax(dim=-1)
                del out, past
                torch.cuda.synchronize(self.device)

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
            pre_start = torch.cuda.Event(enable_timing=True)
            pre_end = torch.cuda.Event(enable_timing=True)
            pre_start.record()
            with torch.no_grad():
                out = self.model(input_ids, use_cache=True)
            pre_end.record()
            torch.cuda.synchronize(self.device)
            prefill_ms = float(pre_start.elapsed_time(pre_end))
            past = out.past_key_values
            token = decode_token

            dec_start = torch.cuda.Event(enable_timing=True)
            dec_end = torch.cuda.Event(enable_timing=True)
            dec_start.record()
            with torch.no_grad():
                for _ in range(max(0, output_len)):
                    out = self.model(token, use_cache=True, past_key_values=past)
                    past = out.past_key_values
                    token = out.logits[:, -1:].argmax(dim=-1)
            dec_end.record()
            torch.cuda.synchronize(self.device)
            decode_ms = float(dec_start.elapsed_time(dec_end))
            peak = int(torch.cuda.max_memory_allocated(self.device))
            return ForwardTimingResult(
                input_len=input_len,
                output_len=output_len,
                batch_size=batch_size,
                rep=rep,
                prefill_ms=prefill_ms,
                decode_ms=decode_ms,
                tpot_ms=decode_ms / max(1, output_len),
                total_ms=prefill_ms + decode_ms,
                peak_gpu_mem_bytes=peak,
                oom=False,
                dtype=self.dtype_name,
                device=str(self.device),
            )
        except RuntimeError as exc:
            msg = str(exc)
            is_oom = "out of memory" in msg.lower() or "cuda error" in msg.lower()
            if is_oom:
                torch.cuda.empty_cache()
                return ForwardTimingResult(
                    input_len=input_len,
                    output_len=output_len,
                    batch_size=batch_size,
                    rep=rep,
                    prefill_ms=0.0,
                    decode_ms=0.0,
                    tpot_ms=0.0,
                    total_ms=0.0,
                    peak_gpu_mem_bytes=int(torch.cuda.max_memory_allocated(self.device)),
                    oom=True,
                    dtype=self.dtype_name,
                    device=str(self.device),
                    error=msg[:1000],
                )
            raise


def summarize_forward_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[int, int, int], list[dict]] = {}
    for row in rows:
        groups.setdefault((int(row["input_len"]), int(row["output_len"]), int(row["batch_size"])), []).append(row)
    out: list[dict] = []
    for (input_len, output_len, batch_size), items in sorted(groups.items()):
        valid = [r for r in items if not r.get("oom")]
        base = {
            "input_len": input_len,
            "output_len": output_len,
            "batch_size": batch_size,
            "reps": len(items),
            "valid_reps": len(valid),
            "oom_reps": len(items) - len(valid),
        }
        if not valid:
            out.append({**base, "oom": True})
            continue
        for key in ["prefill_ms", "decode_ms", "tpot_ms", "total_ms", "peak_gpu_mem_bytes"]:
            vals = sorted(float(r[key]) for r in valid)
            base[f"{key}_median"] = statistics.median(vals)
            base[f"{key}_p90"] = vals[min(len(vals) - 1, int(0.9 * (len(vals) - 1)))]
            base[f"{key}_p99"] = vals[min(len(vals) - 1, int(0.99 * (len(vals) - 1)))]
        base["prefill_tokens_per_s_median"] = (
            input_len * batch_size / max(1e-9, base["prefill_ms_median"] / 1000.0)
        )
        base["decode_tokens_per_s_median"] = (
            output_len * batch_size / max(1e-9, base["decode_ms_median"] / 1000.0)
        )
        base["oom"] = False
        out.append(base)
    return out
