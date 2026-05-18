from __future__ import annotations

from dataclasses import asdict, dataclass

from waferagent.h100_forward_timer import H100ForwardTimer


@dataclass
class PrefixExtensionTimingResult:
    prefix_len: int
    private_len: int
    output_len: int
    batch_size: int
    rep: int
    full_prefill_ms: float
    prefix_prefill_ms: float
    extend_prefill_ms: float
    decode_ms: float
    decode_tpot_ms: float
    peak_gpu_mem_bytes: int
    oom: bool
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class PrefixExtensionTimer(H100ForwardTimer):
    def run_extension_case(
        self,
        prefix_len: int,
        private_len: int,
        output_len: int,
        batch_size: int,
        rep: int,
        warmup: int = 0,
    ) -> PrefixExtensionTimingResult:
        torch = self.torch
        try:
            full = self.run_case(prefix_len + private_len, 1, batch_size, rep, warmup)
            prefix = self.run_case(max(1, prefix_len), 1, batch_size, rep + 10000, warmup)
            prefix_ids = self._input_ids(batch_size, max(1, prefix_len), rep + 20000)
            private_ids = self._input_ids(batch_size, max(1, private_len), rep + 30000)
            with torch.no_grad():
                out = self.model(prefix_ids, use_cache=True)
                past = out.past_key_values
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.no_grad():
                out = self.model(private_ids, use_cache=True, past_key_values=past)
            end.record()
            torch.cuda.synchronize(self.device)
            extend_ms = float(start.elapsed_time(end))
            past = out.past_key_values
            token = private_ids[:, -1:].contiguous()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.no_grad():
                for _ in range(max(0, output_len)):
                    out = self.model(token, use_cache=True, past_key_values=past)
                    past = out.past_key_values
                    token = out.logits[:, -1:].argmax(dim=-1)
            end.record()
            torch.cuda.synchronize(self.device)
            decode_ms = float(start.elapsed_time(end))
            return PrefixExtensionTimingResult(
                prefix_len,
                private_len,
                output_len,
                batch_size,
                rep,
                full.prefill_ms,
                prefix.prefill_ms,
                extend_ms,
                decode_ms,
                decode_ms / max(1, output_len),
                int(torch.cuda.max_memory_allocated(self.device)),
                False,
            )
        except RuntimeError as exc:
            msg = str(exc)
            if "out of memory" in msg.lower() or "cuda" in msg.lower():
                torch.cuda.empty_cache()
                return PrefixExtensionTimingResult(
                    prefix_len,
                    private_len,
                    output_len,
                    batch_size,
                    rep,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    int(torch.cuda.max_memory_allocated(self.device)),
                    True,
                    msg[:1000],
                )
            raise
