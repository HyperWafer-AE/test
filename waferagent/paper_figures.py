from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def save_fig(fig, out_base: str | Path) -> None:
    out = Path(out_base)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=180)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def bar_from_csv(csv_path: str | Path, x: str, y: str, out_base: str | Path, hue: str | None = None, ylabel: str | None = None) -> None:
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7, 4))
    if hue and hue in df.columns:
        pivot = df.pivot_table(index=x, columns=hue, values=y, aggfunc="mean")
        pivot.plot(kind="bar", ax=ax)
    else:
        df.groupby(x)[y].mean().plot(kind="bar", ax=ax, color="#4c78a8")
    ax.set_ylabel(ylabel or y)
    ax.tick_params(axis="x", labelrotation=25)
    save_fig(fig, out_base)


def line_from_csv(csv_path: str | Path, x: str, y: str, out_base: str | Path, hue: str | None = None, ylabel: str | None = None) -> None:
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7, 4))
    if hue and hue in df.columns:
        for name, sub in df.groupby(hue):
            sub = sub.sort_values(x)
            ax.plot(sub[x].astype(str), sub[y], marker="o", label=str(name))
        ax.legend(fontsize=7)
    else:
        df = df.sort_values(x)
        ax.plot(df[x].astype(str), df[y], marker="o")
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel or y)
    save_fig(fig, out_base)


def stacked_cache_gap(csv_path: str | Path, out_base: str | Path) -> None:
    df = pd.read_csv(csv_path)
    cols = [
        c
        for c in ["prefill_compute_ms_saved", "decode_shared_kv_read_bytes", "mesh_traffic_bytes", "sram_reload_bytes"]
        if c in df.columns
    ]
    data = df.groupby("baseline")[cols].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    data.plot(kind="bar", ax=ax)
    ax.set_ylabel("normalized metric units")
    ax.tick_params(axis="x", labelrotation=25)
    save_fig(fig, out_base)
