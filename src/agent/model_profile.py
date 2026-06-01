from dataclasses import dataclass


@dataclass
class ModelProfile:
    n_layers: int = 32
    hidden_size: int = 4096
    dtype_bytes: int = 2
    prefill_cycles_per_token: int = 1000
    decode_cycles_per_token: int = 5000

    def kv_bytes(self, tokens: int) -> int:
        return int(2 * self.n_layers * self.hidden_size * self.dtype_bytes * tokens)

    def prefill_cycles(self, tokens: int) -> int:
        return int(tokens * self.prefill_cycles_per_token)

    def decode_cycles(self, tokens: int) -> int:
        return int(tokens * self.decode_cycles_per_token)

