"""Unit helpers for KVRing reports."""

from __future__ import annotations

KiB = 1024
MiB = 1024**2
GiB = 1024**3
TiB = 1024**4
GB_DEC = 10**9
TB_DEC = 10**12


def gib(x: float) -> float:
    return x / GiB


def tib(x: float) -> float:
    return x / TiB


def fmt_bytes(x: float) -> str:
    ax = abs(x)
    if ax >= TiB:
        return f"{x / TiB:.3f} TiB"
    if ax >= GiB:
        return f"{x / GiB:.3f} GiB"
    if ax >= MiB:
        return f"{x / MiB:.3f} MiB"
    if ax >= KiB:
        return f"{x / KiB:.3f} KiB"
    return f"{x:.0f} B"


def fmt_seconds(x: float) -> str:
    if x >= 1.0:
        return f"{x:.6f} s"
    if x >= 1e-3:
        return f"{x * 1e3:.6f} ms"
    if x >= 1e-6:
        return f"{x * 1e6:.6f} us"
    return f"{x * 1e9:.6f} ns"
