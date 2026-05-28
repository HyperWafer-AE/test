#!/usr/bin/env python
"""Run the full public agent trace profiling pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.basic_stats import run_basic_stats
from src.analysis.fingerprint import run_fingerprint_analysis
from src.analysis.object_locality import run_object_locality_analysis
from src.analysis.success_failure import run_success_failure_analysis
from src.analysis.transitions import run_transition_analysis
from src.loaders.swe_agent import DATASET_NAME as SWE_DATASET
from src.loaders.swe_agent import load_swe_agent, mock_swe_agent_rows
from src.loaders.terminalbench import DATASET_NAME as TB_DATASET
from src.loaders.terminalbench import load_terminalbench, mock_terminalbench_rows
from src.normalize.normalizer import concat_normalized, normalize_rows
from src.sim.wafer_proxy import run_wafer_proxy_sim

LOGGER = logging.getLogger("agent_trace_profile")


def _parse_datasets(value: str) -> list[str]:
    aliases = []
    for raw in value.split(","):
        name = raw.strip().lower().replace("-", "_")
        if not name:
            continue
        if name in {"tb", "terminal_bench", "terminalbench", "yoonholee/terminalbench_trajectories"}:
            aliases.append("terminalbench")
        elif name in {"swe", "swe_agent", "sweagent", "nebius/swe_agent_trajectories"}:
            aliases.append("swe_agent")
        elif name in {"mock", "offline"}:
            aliases.append("mock")
        else:
            aliases.append(name)
    return aliases or ["terminalbench"]


def _load_dataset_rows(
    dataset: str,
    sample_size: int,
    cache_dir: str | None,
    streaming: bool,
    offline_mock: bool,
    seed: int,
) -> tuple[str, list[dict[str, Any]], list[str], bool]:
    if dataset == "terminalbench":
        result = load_terminalbench(
            sample_size=sample_size,
            streaming=streaming,
            cache_dir=cache_dir,
            seed=seed,
            offline_mock=offline_mock,
        )
        return result.dataset, result.rows, result.warnings, result.used_mock
    if dataset == "swe_agent":
        result = load_swe_agent(
            sample_size=sample_size,
            streaming=streaming,
            cache_dir=cache_dir,
            seed=seed,
            offline_mock=offline_mock,
            sample_mode=True,
        )
        return result.dataset, result.rows, result.warnings, result.used_mock
    if dataset == "mock":
        rows = mock_terminalbench_rows()[:sample_size] + mock_swe_agent_rows()[:sample_size]
        return "mock", rows, ["mock: using bundled TerminalBench/SWE-agent style rows."], True
    raise ValueError(f"Unsupported dataset alias: {dataset}")


def _ensure_dirs(outdir: Path) -> dict[str, Path]:
    dirs = {
        "figures": outdir / "figures",
        "tables": outdir / "tables",
        "reports": outdir / "reports",
        "normalized": outdir / "data" / "normalized",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _support_label(score: str) -> str:
    return score


def _hypothesis_status(metrics: dict[str, Any]) -> dict[str, str]:
    transition = metrics.get("transition", {})
    recall = transition.get("next_tool_recall", pd.DataFrame())
    top1 = _safe_float(transition.get("top1_recall"))
    top3 = _safe_float(transition.get("top3_recall"))
    global_top1 = np.nan
    if isinstance(recall, pd.DataFrame) and not recall.empty:
        rows = recall[recall["k"] == 1]
        if not rows.empty:
            global_top1 = _safe_float(rows["global_recall"].iloc[0])
    h1 = "支持" if (top1 > global_top1 + 0.05 or top3 >= 0.6) else "部分支持" if top3 >= 0.3 else "不支持"

    fp = metrics.get("fingerprint", {}).get("fingerprint_metrics", pd.DataFrame())
    if isinstance(fp, pd.DataFrame) and not fp.empty:
        best_cos = _safe_float(fp["future_tool_cosine"].max())
        best_auc = _safe_float(fp["long_auc"].max())
        best_acc = _safe_float(fp["long_accuracy"].max())
        fp_score = max(best_auc if not np.isnan(best_auc) else 0, best_acc if not np.isnan(best_acc) else 0)
    else:
        best_cos = np.nan
        fp_score = 0
    h2 = "支持" if best_cos >= 0.35 and fp_score >= 0.65 else "部分支持" if best_cos >= 0.15 or fp_score >= 0.55 else "不支持"

    basic = metrics.get("basic", {})
    obs_p50 = _safe_float(basic.get("obs_p50"), 0)
    obs_p95 = _safe_float(basic.get("obs_p95"), 0)
    object_loc = metrics.get("object_locality", {})
    reuse_events = int(object_loc.get("reuse_events", 0) or 0)
    short_reuse = _safe_float(object_loc.get("short_reuse_fraction"))
    h3 = (
        "支持"
        if obs_p95 > max(1000, 4 * max(obs_p50, 1)) and reuse_events > 0
        else "部分支持"
        if obs_p95 > max(200, 2 * max(obs_p50, 1)) or (reuse_events > 0 and short_reuse >= 0.3)
        else "不支持"
    )

    sf = metrics.get("success_failure", {}).get("success_failure_metrics", pd.DataFrame())
    h4 = "不支持"
    if isinstance(sf, pd.DataFrame) and not sf.empty:
        deltas = sf["delta_failure_minus_success"].dropna().abs()
        pvals = sf["mannwhitney_p"].dropna()
        if (not pvals.empty and (pvals < 0.05).any()) or (not deltas.empty and deltas.max() > 0.25):
            h4 = "支持"
        elif not deltas.empty and deltas.max() > 0:
            h4 = "部分支持"

    wafer = metrics.get("wafer", {}).get("wafer_proxy_results", pd.DataFrame())
    h5 = "不支持"
    if isinstance(wafer, pd.DataFrame) and not wafer.empty:
        island = wafer[wafer["strategy"] == "agent_island"]
        if not island.empty:
            reduction = _safe_float(island["movement_reduction_vs_random"].iloc[0], 0)
            h5 = "支持" if reduction >= 0.2 else "部分支持" if reduction > 0 else "不支持"
    return {"H1": h1, "H2": h2, "H3": h3, "H4": h4, "H5": h5}


def _rel(path: Path, base: Path) -> str:
    try:
        return os.path.relpath(path, base)
    except Exception:
        return str(path)


def write_report(
    outdir: Path,
    dirs: dict[str, Path],
    traces_df: pd.DataFrame,
    steps_df: pd.DataFrame,
    objects_df: pd.DataFrame,
    metrics: dict[str, Any],
    warnings: list[str],
    used_mocks: dict[str, bool],
    args: argparse.Namespace,
) -> Path:
    report_path = dirs["reports"] / "agent_trace_profile.md"
    statuses = _hypothesis_status(metrics)
    report_dir = report_path.parent

    summary = metrics.get("basic", {}).get("dataset_summary", pd.DataFrame())
    summary_md = summary.to_markdown(index=False) if isinstance(summary, pd.DataFrame) and not summary.empty else "No dataset rows."

    transition = metrics.get("transition", {})
    recall = transition.get("next_tool_recall", pd.DataFrame())
    recall_md = recall.to_markdown(index=False) if isinstance(recall, pd.DataFrame) and not recall.empty else "No transition recall data."

    fp = metrics.get("fingerprint", {}).get("fingerprint_metrics", pd.DataFrame())
    if isinstance(fp, pd.DataFrame) and not fp.empty:
        fp_show = fp[["K", "future_tool_cosine", "future_tool_top3_recall", "remaining_steps_r2", "long_auc", "long_accuracy"]]
        fp_md = fp_show.to_markdown(index=False)
    else:
        fp_md = "No fingerprint metrics."

    sf = metrics.get("success_failure", {}).get("success_failure_metrics", pd.DataFrame())
    sf_md = sf.to_markdown(index=False) if isinstance(sf, pd.DataFrame) and not sf.empty else "No success/failure split."

    wafer = metrics.get("wafer", {}).get("wafer_proxy_results", pd.DataFrame())
    wafer_md = wafer.to_markdown(index=False) if isinstance(wafer, pd.DataFrame) and not wafer.empty else "No wafer proxy results."

    top_pairs = transition.get("top_pairs", pd.DataFrame())
    top_pair_line = "无 transition pair。"
    if isinstance(top_pairs, pd.DataFrame) and not top_pairs.empty:
        row = top_pairs.iloc[0]
        top_pair_line = f"最高频工具转移是 `{row['cur_tool']} -> {row['next_tool']}`，count={int(row['count'])}，P={float(row['probability']):.3f}。"

    basic = metrics.get("basic", {})
    obs_p50 = _safe_float(basic.get("obs_p50"), 0)
    obs_p95 = _safe_float(basic.get("obs_p95"), 0)
    object_loc = metrics.get("object_locality", {})
    reuse_events = int(object_loc.get("reuse_events", 0) or 0)
    median_reuse = _safe_float(object_loc.get("median_reuse_distance"))

    warning_text = "\n".join(f"- {w}" for w in warnings[:50]) if warnings else "- 无。"
    mock_text = ", ".join(f"{k}={'mock' if v else 'real'}" for k, v in used_mocks.items()) or "none"

    fig = dirs["figures"]
    tab = dirs["tables"]
    norm = dirs["normalized"]
    body = f"""# Agent Trace Profile Report

生成时间由本地运行决定。命令参数：

```bash
python scripts/run_all.py --datasets {args.datasets} --sample-size {args.sample_size} --outdir {outdir}
```

## 数据集和样本量

请求数据集：`{args.datasets}`  
加载模式：`streaming={args.streaming}`, `offline_mock={args.offline_mock}`  
真实/兜底状态：`{mock_text}`  
Canonical rows：traces={len(traces_df)}, steps={len(steps_df)}, object_accesses={len(objects_df)}

{summary_md}

Canonical CSV：

- `{_rel(norm / "traces.csv", report_dir)}`
- `{_rel(norm / "steps.csv", report_dir)}`
- `{_rel(norm / "object_accesses.csv", report_dir)}`

## Hypothesis Verdicts

| Hypothesis | Verdict | Evidence |
|---|---:|---|
| H1: tool/phase 时间局部性 | {_support_label(statuses["H1"])} | {top_pair_line} 见 `{_rel(fig / "tool_transition_heatmap.png", report_dir)}`、`{_rel(fig / "phase_transition_heatmap.png", report_dir)}`、`{_rel(fig / "topk_next_tool_recall.png", report_dir)}`。 |
| H2: early fingerprint 可预测未来 | {_support_label(statuses["H2"])} | 前 K 步特征的 tool cosine、remaining step R2、long trajectory AUC/accuracy 见 `{_rel(tab / "fingerprint_metrics.csv", report_dir)}` 和 `{_rel(fig / "early_fingerprint_predictability.png", report_dir)}`。 |
| H3: observation/file/test/browser 是可复用状态对象 | {_support_label(statuses["H3"])} | observation p50={obs_p50:.1f} chars, p95={obs_p95:.1f} chars；reuse events={reuse_events}, median reuse distance={median_reuse:.2f}。见 `{_rel(fig / "observation_size_cdf.png", report_dir)}` 和 `{_rel(fig / "object_reuse_distance_cdf.png", report_dir)}`。 |
| H4: 成功/失败轨迹状态成本不同 | {_support_label(statuses["H4"])} | step 数、tool entropy、observation bytes、重复 action、error rate 的分组统计见 `{_rel(tab / "success_failure_metrics.csv", report_dir)}` 和 `{_rel(fig / "success_vs_failure_bars.png", report_dir)}`。 |
| H5: Agent Island placement 降低 proxy movement | {_support_label(statuses["H5"])} | 5x5 mesh proxy simulator 对比 random/session/island，见 `{_rel(tab / "wafer_proxy_results.csv", report_dir)}` 和 `{_rel(fig / "wafer_proxy_movement_reduction.png", report_dir)}`。 |

## 关键发现

1. Tool locality：条件转移预测的 top-k recall 如下。若 `conditional_recall` 高于 global baseline，说明当前 tool 对下一 tool 有可利用信号。

{recall_md}

2. Early fingerprint：前 1/2/3/5 步的简单统计已能给出轻量预测基线。它不是最终模型，但能回答“早期状态是否带有可调度信号”。

{fp_md}

3. Observation 长尾与对象复用：observation size CDF 展示状态对象大小分布；object reuse distance CDF 使用文件路径、test case、URL、observation hash 和 large-observation bucket 的规则近似复用。

4. 成功/失败差异：

{sf_md}

5. Wafer proxy movement：

{wafer_md}

## 输出索引

Figures：

- `{_rel(fig / "trajectory_length_cdf.png", report_dir)}`
- `{_rel(fig / "tool_calls_per_trace_cdf.png", report_dir)}`
- `{_rel(fig / "observation_size_cdf.png", report_dir)}`
- `{_rel(fig / "tool_transition_heatmap.png", report_dir)}`
- `{_rel(fig / "phase_transition_heatmap.png", report_dir)}`
- `{_rel(fig / "topk_next_tool_recall.png", report_dir)}`
- `{_rel(fig / "early_fingerprint_predictability.png", report_dir)}`
- `{_rel(fig / "object_reuse_distance_cdf.png", report_dir)}`
- `{_rel(fig / "success_vs_failure_bars.png", report_dir)}`
- `{_rel(fig / "wafer_proxy_movement_reduction.png", report_dir)}`

Tables：

- `{_rel(tab / "dataset_summary.csv", report_dir)}`
- `{_rel(tab / "top_tools.csv", report_dir)}`
- `{_rel(tab / "tool_transition_top_pairs.csv", report_dir)}`
- `{_rel(tab / "phase_transition_matrix.csv", report_dir)}`
- `{_rel(tab / "fingerprint_metrics.csv", report_dir)}`
- `{_rel(tab / "success_failure_metrics.csv", report_dir)}`
- `{_rel(tab / "wafer_proxy_results.csv", report_dir)}`

## 限制

- 公开 agent trace 不是生产底层硬件 trace，不能直接推出真实 wafer runtime 的延迟。
- 当前数据没有真实 KV cache object、MoE expert ID、die placement 或 D2D traffic counter。
- Observation 可能被 harness 截断或格式化，`observation_text` 里的对象识别是规则近似。
- Wafer simulator 是 trace-level proxy model，只估计相对 movement/hops，不代表物理网络、cache coherence、DMA 或调度开销。
- 若本次运行使用 mock fallback，报告只验证 pipeline 可运行，不应作为论文证据。

## 下一步

1. 补采开源 MoE expert trace 和 KV/cache access trace，把 `ObjectAccess` 从文本规则升级为真实 runtime object。
2. 加入真实 latency replay：按工具调用、observation size、KV movement、D2D hops 校准 simulator。
3. 扩展到 ClawsBench 和 Applied Compute released jsonl，并把 schema adapter 做成插件式 loader。
4. 对 TerminalBench/SWE-agent 做更大样本的 bootstrap 置信区间和跨模型/跨 agent 分层分析。

## Warnings

{warning_text}
"""
    report_path.write_text(body, encoding="utf-8")
    return report_path


def run_pipeline(args: argparse.Namespace) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    outdir = Path(args.outdir).resolve()
    dirs = _ensure_dirs(outdir)

    all_warnings: list[str] = []
    used_mocks: dict[str, bool] = {}
    normalized_parts: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    datasets = _parse_datasets(args.datasets)

    for dataset in datasets:
        LOGGER.info("Loading dataset=%s sample_size=%s", dataset, args.sample_size)
        raw_dataset, rows, load_warnings, used_mock = _load_dataset_rows(
            dataset=dataset,
            sample_size=args.sample_size,
            cache_dir=args.cache_dir,
            streaming=args.streaming,
            offline_mock=args.offline_mock,
            seed=args.seed,
        )
        all_warnings.extend(load_warnings)
        used_mocks[dataset] = used_mock
        if raw_dataset == "mock":
            tb_rows = [r for r in rows if "trial_id" in r]
            swe_rows = [r for r in rows if "instance_id" in r]
            for real_dataset, real_rows in ((TB_DATASET, tb_rows), (SWE_DATASET, swe_rows)):
                traces, steps, objects, norm_warnings = normalize_rows(real_dataset, real_rows)
                all_warnings.extend(norm_warnings)
                normalized_parts.append((traces, steps, objects))
        else:
            traces, steps, objects, norm_warnings = normalize_rows(raw_dataset, rows)
            all_warnings.extend(norm_warnings)
            normalized_parts.append((traces, steps, objects))

    traces_df, steps_df, objects_df = concat_normalized(normalized_parts)
    traces_df.to_csv(dirs["normalized"] / "traces.csv", index=False)
    steps_df.to_csv(dirs["normalized"] / "steps.csv", index=False)
    objects_df.to_csv(dirs["normalized"] / "object_accesses.csv", index=False)
    (dirs["normalized"] / "metadata.json").write_text(
        json.dumps(
            {
                "datasets": datasets,
                "sample_size": args.sample_size,
                "num_traces": int(len(traces_df)),
                "num_steps": int(len(steps_df)),
                "num_object_accesses": int(len(objects_df)),
                "used_mocks": used_mocks,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (dirs["reports"] / "warnings.log").write_text("\n".join(all_warnings), encoding="utf-8")

    metrics: dict[str, Any] = {}
    metrics["basic"] = run_basic_stats(traces_df, steps_df, dirs["figures"], dirs["tables"])
    metrics["transition"] = run_transition_analysis(steps_df, dirs["figures"], dirs["tables"], seed=args.seed)
    metrics["fingerprint"] = run_fingerprint_analysis(
        traces_df, steps_df, dirs["figures"], dirs["tables"], seed=args.seed
    )
    metrics["object_locality"] = run_object_locality_analysis(objects_df, dirs["figures"])
    metrics["success_failure"] = run_success_failure_analysis(
        traces_df, steps_df, dirs["figures"], dirs["tables"]
    )
    metrics["wafer"] = run_wafer_proxy_sim(
        objects_df,
        dirs["figures"],
        dirs["tables"],
        mesh_size=args.mesh_size,
        island_size=args.island_size,
    )

    report_path = write_report(
        outdir=outdir,
        dirs=dirs,
        traces_df=traces_df,
        steps_df=steps_df,
        objects_df=objects_df,
        metrics=metrics,
        warnings=all_warnings,
        used_mocks=used_mocks,
        args=args,
    )
    LOGGER.info("Wrote report: %s", report_path)
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile public agent traces for wafer-locality hypotheses.")
    parser.add_argument("--datasets", default="terminalbench", help="Comma list: terminalbench,swe_agent,mock")
    parser.add_argument("--sample-size", type=int, default=5000, help="Max rows per requested real dataset.")
    parser.add_argument("--outdir", default=".", help="Output directory containing figures/tables/reports/data.")
    parser.add_argument("--cache-dir", default=str(ROOT / ".cache" / "huggingface"), help="HF cache directory.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mesh-size", type=int, default=5)
    parser.add_argument("--island-size", type=int, default=3, choices=[2, 3])
    parser.add_argument("--offline-mock", action="store_true", help="Skip network and use bundled mock traces.")
    parser.add_argument("--no-streaming", dest="streaming", action="store_false", help="Disable HF streaming.")
    parser.set_defaults(streaming=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.sample_size <= 0:
        parser.error("--sample-size must be positive")
    run_pipeline(args)


if __name__ == "__main__":
    main()
    logging.shutdown()
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        # Some HuggingFace/pyarrow builds leave background native state that can
        # abort during interpreter finalization after all artifacts are written.
        os._exit(0)
