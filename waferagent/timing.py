from __future__ import annotations

import time
from contextlib import contextmanager


@contextmanager
def cuda_timer():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            holder = {"elapsed_ms": 0.0}
            yield holder
            end_ev.record()
            torch.cuda.synchronize()
            holder["elapsed_ms"] = float(start_ev.elapsed_time(end_ev))
            return
    except Exception:
        pass
    start = time.time()
    holder = {"elapsed_ms": 0.0}
    yield holder
    holder["elapsed_ms"] = (time.time() - start) * 1000.0
