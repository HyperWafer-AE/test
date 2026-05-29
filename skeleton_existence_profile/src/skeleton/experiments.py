from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .stats import auc_score, cosine, entropy, mann_whitney, topk_recall, train_test_split_trace


CLEAN_UNKNOWN = {"unknown", "none", "", "nan"}
VIEWS = [
    "wrapper_tool_all",
    "semantic_tool_all",
    "semantic_tool_no_unknown",
    "semantic_tool_tool_action_only",
    "collapsed_semantic_tool_tool_action_only",
    "phase_all",
    "phase_clean_tool_action_only",
    "collapsed_phase_clean_tool_action_only",
]


def ensure_dirs(outdir: Path) -> dict[str, Path]:
    dirs = {name: outdir / name for name in ("tables", "figures", "reports", "data")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.lower().isin({"true", "1", "yes"})


def _view_steps(steps: pd.DataFrame, view: str) -> tuple[pd.DataFrame, str, bool]:
    df = steps.copy()
    df["semantic_tool_clean"] = df["semantic_tool_clean"].fillna("unknown").astype(str)
    df["phase_clean"] = df["phase_clean"].fillna("unknown").astype(str)
    df["tool_wrapper"] = df["tool_wrapper"].fillna("unknown").astype(str)
    is_tool = _bool(df["is_tool_action"])
    artifact = _bool(df["command_artifact_flag"])
    no_tool = _bool(df["is_no_tool_step"])
    collapse = False
    if view == "wrapper_tool_all":
        return df, "tool_wrapper", collapse
    if view == "semantic_tool_all":
        return df, "semantic_tool_clean", collapse
    if view == "semantic_tool_no_unknown":
        return df[~df["semantic_tool_clean"].str.lower().isin(CLEAN_UNKNOWN)], "semantic_tool_clean", collapse
    if view == "semantic_tool_tool_action_only":
        keep = is_tool & ~artifact & ~no_tool & ~df["semantic_tool_clean"].str.lower().isin(CLEAN_UNKNOWN)
        return df[keep], "semantic_tool_clean", collapse
    if view == "collapsed_semantic_tool_tool_action_only":
        keep = is_tool & ~artifact & ~no_tool & ~df["semantic_tool_clean"].str.lower().isin(CLEAN_UNKNOWN)
        return df[keep], "semantic_tool_clean", True
    if view == "phase_all":
        return df, "phase_clean", collapse
    if view == "phase_clean_tool_action_only":
        keep = is_tool & ~artifact & ~no_tool & ~df["phase_clean"].str.lower().isin(CLEAN_UNKNOWN)
        return df[keep], "phase_clean", collapse
    if view == "collapsed_phase_clean_tool_action_only":
        keep = is_tool & ~artifact & ~no_tool & ~df["phase_clean"].str.lower().isin(CLEAN_UNKNOWN)
        return df[keep], "phase_clean", True
    raise ValueError(view)


def transition_rows(steps: pd.DataFrame, view: str, shuffle: str | None = None, seed: int = 0) -> pd.DataFrame:
    df, col, collapse = _view_steps(steps, view)
    rng = np.random.default_rng(seed)
    rows = []
    for tid, g in df.sort_values(["trace_id", "step_id"]).groupby("trace_id"):
        vals = g[col].astype(str).tolist()
        if collapse:
            vals = [v for i, v in enumerate(vals) if i == 0 or v != vals[i - 1]]
        if shuffle == "within_trace":
            vals = list(rng.permutation(vals))
        for cur, nxt in zip(vals, vals[1:]):
            rows.append({"view": view, "trace_id": tid, "current": cur, "next": nxt})
    out = pd.DataFrame(rows)
    if shuffle == "global_next" and not out.empty:
        out["next"] = rng.permutation(out["next"].to_numpy())
    if shuffle == "frequency_preserve" and not out.empty:
        out["next"] = rng.permutation(out["next"].to_numpy())
    return out


def _fit_predict(train: pd.DataFrame) -> tuple[dict[str, list[str]], list[str]]:
    cond = {c: g["next"].value_counts().index.tolist() for c, g in train.groupby("current")}
    glob = train["next"].value_counts().index.tolist()
    return cond, glob


def _recall(test: pd.DataFrame, cond: dict[str, list[str]], glob: list[str], k: int) -> tuple[float, float]:
    if test.empty:
        return np.nan, np.nan
    glob_set = set(glob[:k])
    cond_sets = {key: set(vals[:k]) for key, vals in cond.items()}
    currents = test["current"].astype(str).to_numpy()
    nexts = test["next"].astype(str).to_numpy()
    cond_hits = sum(nxt in cond_sets.get(cur, glob_set) for cur, nxt in zip(currents, nexts))
    glob_hits = sum(nxt in glob_set for nxt in nexts)
    return cond_hits / len(test), glob_hits / len(test)


def _recall_stats_by_trace(test: pd.DataFrame, cond: dict[str, list[str]], glob: list[str], k: int) -> dict[str, tuple[int, int, int]]:
    stats: dict[str, tuple[int, int, int]] = {}
    if test.empty:
        return stats
    for tid, g in test.groupby(test["trace_id"].astype(str), sort=False):
        cond_hits = 0
        glob_hits = 0
        n = 0
        for _, row in g.iterrows():
            cond_hits += int(row["next"] in cond.get(row["current"], glob)[:k])
            glob_hits += int(row["next"] in glob[:k])
            n += 1
        stats[str(tid)] = (cond_hits, glob_hits, n)
    return stats


def _delta_from_stats(stats: dict[str, tuple[int, int, int]], tids: list[str] | np.ndarray | None = None) -> float:
    ids = list(stats) if tids is None else list(tids)
    cond_hits = 0
    glob_hits = 0
    n = 0
    for tid in ids:
        ch, gh, nn = stats[str(tid)]
        cond_hits += ch
        glob_hits += gh
        n += nn
    return cond_hits / n - glob_hits / n if n else np.nan


def _fast_transition_delta_ci(
    test: pd.DataFrame,
    cond: dict[str, list[str]],
    glob: list[str],
    k: int,
    seed: int,
    n_boot: int = 30,
) -> tuple[float, float]:
    stats = _recall_stats_by_trace(test, cond, glob, k)
    if not stats:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    tids = np.array(sorted(stats))
    vals = []
    for _ in range(n_boot):
        sample = rng.choice(tids, size=len(tids), replace=True)
        vals.append(_delta_from_stats(stats, sample))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _entropy_mi(trans: pd.DataFrame) -> tuple[float, float, float]:
    if trans.empty:
        return np.nan, np.nan, np.nan
    h_global = entropy(trans["next"].astype(str))
    h_cond = 0.0
    n = len(trans)
    for _, g in trans.groupby("current"):
        h_cond += len(g) / n * entropy(g["next"].astype(str))
    return h_global, h_cond, h_global - h_cond


def h1_transitions(steps: pd.DataFrame, tables: Path, figures: Path, seed: int = 0) -> dict[str, Any]:
    recall_rows = []
    ent_rows = []
    perm_rows = []
    heatmaps: dict[str, pd.DataFrame] = {}
    for view in VIEWS:
        trans = transition_rows(steps, view, seed=seed)
        if trans.empty:
            continue
        train_ids, test_ids = train_test_split_trace(trans["trace_id"], seed)
        train = trans[trans["trace_id"].isin(train_ids)]
        test = trans[trans["trace_id"].isin(test_ids)]
        if test.empty:
            test = trans
            train = trans
        cond, glob = _fit_predict(train)
        for k in (1, 3, 5):
            cr, gr = _recall(test, cond, glob, k)
            delta = cr - gr
            low, high = _fast_transition_delta_ci(test, cond, glob, k, seed=seed + k)
            recall_rows.append(
                {
                    "view": view,
                    "k": k,
                    "global_recall": gr,
                    "conditional_recall": cr,
                    "delta_vs_global": delta,
                    "ci_low": low,
                    "ci_high": high,
                    "n_transitions": len(trans),
                    "n_test": len(test),
                }
            )
        hg, hc, mi = _entropy_mi(trans)
        ent_rows.append(
            {
                "view": view,
                "global_entropy": hg,
                "conditional_entropy": hc,
                "entropy_reduction": mi,
                "mutual_information": mi,
                "n_transitions": len(trans),
            }
        )
        real_delta = [r for r in recall_rows if r["view"] == view and r["k"] == 3][-1]["delta_vs_global"]
        for control in ("shuffle_steps_within_trace", "shuffle_next_tool_labels_globally", "preserve_tool_frequency_break_order"):
            nulls = []
            for i in range(25):
                null_trans = _shuffle_transition_control(trans, control, seed + 1000 + i)
                if null_trans.empty:
                    continue
                tr = null_trans[null_trans["trace_id"].isin(train_ids)]
                te = null_trans[null_trans["trace_id"].isin(test_ids)]
                c, g = _fit_predict(tr if not tr.empty else null_trans)
                rr, gg = _recall(te if not te.empty else null_trans, c, g, 3)
                nulls.append(rr - gg)
            p = (sum(x >= real_delta for x in nulls) + 1) / (len(nulls) + 1) if nulls else np.nan
            perm_rows.append({"view": view, "control": control, "k": 3, "observed_delta": real_delta, "null_mean": float(np.mean(nulls)) if nulls else np.nan, "p_value": p})
        labels = list((Counter(trans["current"]) + Counter(trans["next"])).keys())[:30]
        mat = pd.crosstab(trans["current"], trans["next"]).reindex(index=labels, columns=labels, fill_value=0)
        mat = mat.div(mat.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        heatmaps[view] = mat

    recall_df = pd.DataFrame(recall_rows)
    ent_df = pd.DataFrame(ent_rows)
    perm_df = pd.DataFrame(perm_rows)
    recall_df.to_csv(tables / "skeleton_transition_recall.csv", index=False)
    ent_df.to_csv(tables / "skeleton_entropy_mi.csv", index=False)
    perm_df.to_csv(tables / "permutation_tests.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=recall_df[(recall_df["k"] == 3) & recall_df["view"].str.contains("tool_action|no_unknown")], x="view", y="delta_vs_global", ax=ax)
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("Top-3 recall delta vs global")
    fig.tight_layout()
    fig.savefig(figures / "topk_recall_clean_views.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=ent_df, x="view", y="entropy_reduction", ax=ax)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(figures / "entropy_reduction_by_view.png", dpi=180)
    plt.close(fig)

    for view, fname in [
        ("semantic_tool_tool_action_only", "transition_heatmap_semantic_tool_action_only.png"),
        ("phase_clean_tool_action_only", "transition_heatmap_phase_clean_tool_action_only.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(heatmaps.get(view, pd.DataFrame()), cmap="Blues", ax=ax)
        ax.set_title(view)
        fig.tight_layout()
        fig.savefig(figures / fname, dpi=180)
        plt.close(fig)
    return {"recall": recall_df, "entropy": ent_df, "permutation": perm_df}


def _shuffle_transition_control(trans: pd.DataFrame, control: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = trans.copy()
    if out.empty:
        return out
    if control == "shuffle_steps_within_trace":
        parts = []
        for tid, g in out.groupby("trace_id"):
            seq = [str(g["current"].iloc[0])] + g["next"].astype(str).tolist()
            vals = rng.permutation(np.array(seq, dtype=object))
            if len(vals) < 2:
                continue
            cur = vals[:-1]
            nxt = vals[1:]
            parts.append(pd.DataFrame({"view": g["view"].iloc[0], "trace_id": tid, "current": cur, "next": nxt}))
        return pd.concat(parts, ignore_index=True) if parts else out.iloc[0:0].copy()
    if control in {"shuffle_next_tool_labels_globally", "preserve_tool_frequency_break_order"}:
        out["next"] = rng.permutation(out["next"].to_numpy())
        return out
    return out


def _clean_sequence(steps: pd.DataFrame, trace_id: str, col: str) -> list[str]:
    g = steps[(steps["trace_id"] == trace_id) & _bool(steps["is_tool_action"]) & ~_bool(steps["command_artifact_flag"])].sort_values("step_id")
    vals = [str(v) for v in g[col].fillna("unknown") if str(v).lower() not in CLEAN_UNKNOWN]
    return [v for i, v in enumerate(vals) if i == 0 or v != vals[i - 1]]


def h2_motifs(traces: pd.DataFrame, steps: pd.DataFrame, objects: pd.DataFrame, tables: Path, figures: Path, seed: int = 0) -> dict[str, Any]:
    train_ids, test_ids = train_test_split_trace(traces["trace_id"], seed)
    meta = traces.set_index("trace_id")
    rows_by_kind = {"phase": [], "tool": [], "stateflow": []}
    motifs = []
    for tid in traces["trace_id"].astype(str):
        tool_seq = _clean_sequence(steps, tid, "semantic_tool_clean")
        phase_seq = _clean_sequence(steps, tid, "phase_clean")
        obj_seq = objects[(objects["trace_id"] == tid) & objects["stable_object"].astype(bool)].sort_values("step_id")["object_type"].astype(str).tolist()
        state_seq = []
        for s in tool_seq[:80]:
            state_seq.append(s)
        if obj_seq:
            state_seq = (tool_seq + [f"{o}_object" for o in obj_seq[: max(1, len(tool_seq) // 2)]])[:100]
        for kind, seq in (("phase", phase_seq), ("tool", tool_seq), ("stateflow", state_seq)):
            seen_for_trace = set()
            for n in (2, 3, 4, 5):
                for i in range(max(0, len(seq) - n + 1)):
                    pat = " -> ".join(seq[i : i + n])
                    if pat in seen_for_trace:
                        continue
                    seen_for_trace.add(pat)
                    rows_by_kind[kind].append({"trace_id": tid, "motif": pat, "n": n, "split": "train" if tid in train_ids else "test"})
    gen_rows = []
    for kind, rows in rows_by_kind.items():
        df = pd.DataFrame(rows)
        if df.empty:
            out = pd.DataFrame(columns=["motif", "n", "support_traces", "support_tasks", "support_harnesses"])
        else:
            train = df[df["split"] == "train"]
            top = train.groupby(["motif", "n"])["trace_id"].nunique().reset_index(name="support_traces").sort_values("support_traces", ascending=False).head(100)
            extra = []
            for _, row in top.iterrows():
                tids = set(df[df["motif"] == row["motif"]]["trace_id"])
                tasks = meta.loc[list(tids & set(meta.index)), "task_id"].nunique() if tids else 0
                harnesses = meta.loc[list(tids & set(meta.index)), "harness"].nunique() if tids else 0
                success_rate = meta.loc[list(tids & set(meta.index)), "success"].mean() if tids else np.nan
                test_support = df[(df["motif"] == row["motif"]) & (df["split"] == "test")]["trace_id"].nunique()
                extra.append({**row.to_dict(), "support_tasks": tasks, "support_harnesses": harnesses, "success_rate": success_rate, "heldout_support_traces": test_support})
                gen_rows.append({"motif_kind": kind, "motif": row["motif"], "train_support": row["support_traces"], "heldout_support": test_support, "support_harnesses": harnesses, "appears_in_heldout": test_support > 0})
            out = pd.DataFrame(extra)
        out.to_csv(tables / f"frequent_{kind}_motifs.csv", index=False)
        if kind == "stateflow":
            out.to_csv(tables / "frequent_stateflow_motifs.csv", index=False)
            motifs = out.head(20)
    gen = pd.DataFrame(gen_rows)
    gen.to_csv(tables / "motif_generalization.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    if isinstance(motifs, pd.DataFrame) and not motifs.empty:
        plot = motifs.head(15).copy()
        plot["motif_short"] = plot["motif"].str.slice(0, 60)
        sns.barplot(data=plot, y="motif_short", x="support_traces", ax=ax)
    fig.tight_layout()
    fig.savefig(figures / "top_stateflow_motifs.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    if not gen.empty:
        sns.scatterplot(data=gen, x="train_support", y="heldout_support", hue="motif_kind", ax=ax)
    fig.tight_layout()
    fig.savefig(figures / "motif_support_train_test.png", dpi=180)
    plt.close(fig)
    return {"motif_generalization": gen}


def h3_dependencies(deps: pd.DataFrame, steps: pd.DataFrame, tables: Path, figures: Path) -> dict[str, Any]:
    if deps.empty:
        summary = pd.DataFrame([{"dependency_type": "none", "count": 0, "median_distance": np.nan, "dependency_rate_per_trace": 0}])
        by_pair = pd.DataFrame(columns=["src_tool", "dst_tool", "count"])
    else:
        tmp = deps.copy()
        tmp["distance"] = tmp["dst_step_id"] - tmp["src_step_id"]
        n_traces = steps["trace_id"].nunique()
        summary = tmp.groupby("dependency_type").agg(count=("trace_id", "size"), traces=("trace_id", "nunique"), median_distance=("distance", "median")).reset_index()
        summary["dependency_rate_per_trace"] = summary["count"] / max(n_traces, 1)
        by_pair = tmp.groupby(["src_tool", "dst_tool", "dependency_type"]).size().reset_index(name="count").sort_values("count", ascending=False)
    summary.to_csv(tables / "data_dependency_summary.csv", index=False)
    by_pair.to_csv(tables / "data_dependency_by_tool_pair.csv", index=False)
    by_pair.head(100).to_csv(tables / "top_dependency_patterns.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    if not deps.empty:
        d = deps.assign(distance=deps["dst_step_id"] - deps["src_step_id"])
        sns.ecdfplot(data=d, x="distance", ax=ax)
    fig.tight_layout()
    fig.savefig(figures / "dependency_distance_cdf.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 6))
    if not by_pair.empty:
        mat = by_pair.pivot_table(index="src_tool", columns="dst_tool", values="count", aggfunc="sum", fill_value=0)
        sns.heatmap(mat, cmap="Blues", ax=ax)
    fig.tight_layout()
    fig.savefig(figures / "dependency_tool_pair_heatmap.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    if not deps.empty:
        dd = deps.assign(distance=deps["dst_step_id"] - deps["src_step_id"])
        rows = []
        for h in (1, 3, 5, 10):
            rows.append({"horizon": h, "dependency_count": int((dd["distance"] <= h).sum())})
        sns.barplot(data=pd.DataFrame(rows), x="horizon", y="dependency_count", ax=ax)
    fig.tight_layout()
    fig.savefig(figures / "dependency_rate_by_horizon.png", dpi=180)
    plt.close(fig)
    return {"summary": summary, "by_pair": by_pair}


def _future_objects(objects: pd.DataFrame, trace_id: str, step: int, h: int, key_col: str) -> list[str]:
    g = objects[(objects["trace_id"] == trace_id) & (objects["step_id"] > step) & (objects["step_id"] <= step + h) & objects["stable_object"].astype(bool)]
    return [x for x in g[key_col].dropna().astype(str).unique() if x and x != "None"]


def _object_index(objects: pd.DataFrame, key_col: str) -> dict[str, tuple[list[int], list[str]]]:
    idx: dict[str, tuple[list[int], list[str]]] = {}
    if objects.empty:
        return idx
    cols = ["trace_id", "step_id", key_col]
    tmp = objects[cols].dropna().copy()
    tmp[key_col] = tmp[key_col].astype(str)
    tmp = tmp[(tmp[key_col] != "") & (tmp[key_col] != "None")].sort_values(["trace_id", "step_id"])
    for tid, g in tmp.groupby("trace_id", sort=False):
        idx[str(tid)] = (g["step_id"].astype(int).tolist(), g[key_col].tolist())
    return idx


def _future_from_index(index: dict[str, tuple[list[int], list[str]]], trace_id: str, step: int, h: int) -> list[str]:
    item = index.get(str(trace_id))
    if item is None:
        return []
    steps, vals = item
    lo = bisect_right(steps, step)
    hi = bisect_right(steps, step + h)
    return list(dict.fromkeys(vals[lo:hi]))


def _prev_from_index(index: dict[str, tuple[list[int], list[str]]], trace_id: str, step: int, k: int = 5) -> list[str]:
    item = index.get(str(trace_id))
    if item is None:
        return []
    steps, vals = item
    hi = bisect_right(steps, step)
    return list(dict.fromkeys(reversed(vals[max(0, hi - k) : hi])))


def h4_object_working_set(traces: pd.DataFrame, steps: pd.DataFrame, objects: pd.DataFrame, tables: Path, figures: Path, seed: int = 0) -> dict[str, Any]:
    obj = objects[objects["stable_object"].astype(bool) & objects["actionable_object"].astype(bool)].copy()
    exact = obj[obj["object_id"].str.startswith("file:", na=False)]
    prefix = obj[obj["object_prefix"].notna()].copy()
    exact_index = _object_index(exact, "object_id")
    prefix_index = _object_index(prefix, "object_prefix")
    rows = []
    pred_rows = []
    rng = np.random.default_rng(seed)
    global_hot = exact["object_id"].value_counts().index.tolist()
    global_prefix = prefix["object_prefix"].value_counts().index.tolist()
    task_hot = {k: g["object_id"].value_counts().index.tolist() for k, g in exact.merge(traces[["trace_id", "task_id"]], on="trace_id").groupby("task_id")}
    harness_hot = {k: g["object_id"].value_counts().index.tolist() for k, g in exact.merge(traces[["trace_id", "harness"]], on="trace_id").groupby("harness")}
    trace_meta = traces.set_index("trace_id")
    eval_steps = steps[_bool(steps["is_tool_action"])].copy()
    if len(eval_steps) > 20000:
        eval_steps = eval_steps.sample(n=20000, random_state=seed).sort_values(["trace_id", "step_id"])
    for h in (1, 3, 5, 10):
        for _, s in eval_steps.iterrows():
            tid = s["trace_id"]
            sid = int(s["step_id"])
            truth = _future_from_index(exact_index, tid, sid, h)
            ptruth = _future_from_index(prefix_index, tid, sid, h)
            if not truth and not ptruth:
                continue
            prev = _prev_from_index(exact_index, tid, sid, 5)
            pprev = _prev_from_index(prefix_index, tid, sid, 5)
            task = trace_meta.loc[tid, "task_id"] if tid in trace_meta.index else None
            harness = trace_meta.loc[tid, "harness"] if tid in trace_meta.index else None
            baselines = {
                "global_hot_object": global_hot,
                "task_hot_object": task_hot.get(task, global_hot),
                "harness_hot_object": harness_hot.get(harness, global_hot),
                "lastK_object": prev,
                "object_type_only": ["file"],
                "random_same_count": list(rng.choice(global_hot, size=min(5, len(global_hot)), replace=False)) if global_hot else [],
            }
            for name, pred in baselines.items():
                pred_rows.append({"horizon": h, "granularity": "exact_object_id", "baseline": name, "top5_recall": topk_recall(pred, truth, 5)})
            pbaselines = {"global_hot_prefix": global_prefix, "lastK_prefix": pprev, "random_prefix": list(rng.choice(global_prefix, size=min(5, len(global_prefix)), replace=False)) if global_prefix else []}
            for name, pred in pbaselines.items():
                pred_rows.append({"horizon": h, "granularity": "path_prefix", "baseline": name, "top5_recall": topk_recall(pred, ptruth, 5)})
    pred = pd.DataFrame(pred_rows).dropna()
    pred_summary = pred.groupby(["horizon", "granularity", "baseline"]).agg(top5_recall=("top5_recall", "mean")).reset_index() if not pred.empty else pd.DataFrame()
    for cls, df, key in [("exact_object_id", exact, "object_id"), ("path_prefix", prefix, "object_prefix"), ("object_type", obj, "object_type")]:
        counts = df[key].value_counts()
        top_n = max(1, int(np.ceil(0.1 * len(counts)))) if len(counts) else 1
        rows.append(
            {
                "granularity": cls,
                "accesses": int(len(df)),
                "unique_objects": int(df[key].nunique()) if key in df else 0,
                "top10pct_access_coverage": float(counts.head(top_n).sum() / counts.sum()) if counts.sum() else np.nan,
                "median_reuse_distance": _reuse_distance(df, key),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(tables / "object_working_set_summary.csv", index=False)
    pred_summary.to_csv(tables / "object_prediction_baselines.csv", index=False)
    exact["object_id"].value_counts().head(100).rename_axis("object_id").reset_index(name="accesses").to_csv(tables / "top_reused_object_ids.csv", index=False)
    prefix["object_prefix"].value_counts().head(100).rename_axis("object_prefix").reset_index(name="accesses").to_csv(tables / "top_reused_path_prefixes.csv", index=False)
    _plot_reuse(exact, "object_id", figures / "object_reuse_distance_cdf_exact.png")
    _plot_reuse(prefix, "object_prefix", figures / "path_prefix_reuse_distance_cdf.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=summary, x="granularity", y="top10pct_access_coverage", ax=ax)
    fig.tight_layout()
    fig.savefig(figures / "hot_object_concentration.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    if not pred_summary.empty:
        sns.lineplot(data=pred_summary[pred_summary["granularity"] != "object_type"], x="horizon", y="top5_recall", hue="baseline", marker="o", ax=ax)
    fig.tight_layout()
    fig.savefig(figures / "object_prediction_recall.png", dpi=180)
    plt.close(fig)
    return {"summary": summary, "prediction": pred_summary}


def _reuse_distance(df: pd.DataFrame, key: str) -> float:
    vals = []
    for _, g in df.sort_values(["trace_id", key, "step_id"]).groupby(["trace_id", key]):
        steps = g["step_id"].tolist()
        vals += [b - a for a, b in zip(steps, steps[1:])]
    return float(np.median(vals)) if vals else np.nan


def _plot_reuse(df: pd.DataFrame, key: str, path: Path) -> None:
    vals = []
    for _, g in df.sort_values(["trace_id", key, "step_id"]).groupby(["trace_id", key]):
        steps = g["step_id"].tolist()
        vals += [b - a for a, b in zip(steps, steps[1:])]
    fig, ax = plt.subplots(figsize=(7, 4))
    if vals:
        vals = np.sort(vals)
        ax.step(vals, np.arange(1, len(vals) + 1) / len(vals), where="post")
        ax.set_xscale("symlog", linthresh=1)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def failure_metrics_per_trace(traces: pd.DataFrame, steps: pd.DataFrame, objects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, tr in traces.iterrows():
        tid = tr["trace_id"]
        s = steps[(steps["trace_id"] == tid) & _bool(steps["is_tool_action"]) & ~_bool(steps["command_artifact_flag"])].sort_values("step_id")
        o = objects[(objects["trace_id"] == tid) & objects["stable_object"].astype(bool)]
        tools = s["semantic_tool_clean"].fillna("unknown").astype(str).tolist()
        phases = s["phase_clean"].fillna("unknown").astype(str).tolist()
        obj_ids = o["object_id"].fillna("").astype(str).tolist()
        prefixes = o["object_prefix"].dropna().astype(str).tolist()
        errors = o[o["object_type"] == "error_signature"]["object_id"].astype(str).tolist()
        max_run = _max_run(tools)
        rep_tool = 1.0 - len(set(tools)) / len(tools) if tools else 0.0
        rep_phase = 1.0 - len(set(phases)) / len(phases) if phases else 0.0
        rep_obj = 1.0 - len(set(obj_ids)) / len(obj_ids) if obj_ids else 0.0
        rep_pref = 1.0 - len(set(prefixes)) / len(prefixes) if prefixes else 0.0
        err_rate = float(s["error_flag"].mean()) if not s.empty else 0.0
        rep_err = sum(c - 1 for c in Counter(errors).values() if c > 1)
        score = np.mean([rep_tool, rep_phase, rep_obj, rep_pref, err_rate, min(rep_err / max(len(errors), 1), 1.0), min(max_run / max(len(tools), 1), 1.0)])
        rows.append(
            {
                "trace_id": tid,
                "success": bool(tr.get("success")),
                "semantic_tool_entropy": entropy(tools),
                "phase_entropy": entropy(phases),
                "repeated_semantic_tool_ratio": rep_tool,
                "repeated_phase_ratio": rep_phase,
                "max_same_tool_run_length": max_run,
                "repeated_object_ratio": rep_obj,
                "repeated_path_prefix_ratio": rep_pref,
                "error_rate": err_rate,
                "repeated_error_signature_count": rep_err,
                "repeated_test_case_count": _repeat_count([x for x in obj_ids if x.startswith("test:")]),
                "no_progress_loop_score": np.mean([rep_tool, rep_obj, err_rate]),
                "failure_loop_score": score,
            }
        )
    return pd.DataFrame(rows)


def _max_run(vals: list[str]) -> int:
    best = cur = 0
    prev = object()
    for v in vals:
        cur = cur + 1 if v == prev else 1
        best = max(best, cur)
        prev = v
    return best


def _repeat_count(vals: list[str]) -> int:
    return sum(c - 1 for c in Counter(vals).values() if c > 1)


def h5_early(traces: pd.DataFrame, steps: pd.DataFrame, objects: pd.DataFrame, failure_df: pd.DataFrame, tables: Path, figures: Path, seed: int = 0) -> dict[str, Any]:
    train_ids, test_ids = train_test_split_trace(traces["trace_id"], seed)
    trace_meta = traces.set_index("trace_id")
    features = _trace_sequences(traces, steps, objects)
    train = {k: v for k, v in features.items() if k in train_ids}
    test = {k: v for k, v in features.items() if k in test_ids}
    global_tool = _avg_counter([v["future_tools"] for v in train.values()])
    global_phase = _avg_counter([v["future_phases"] for v in train.values()])
    by_task = _group_avg(train, trace_meta, "task_id")
    by_harness = _group_avg(train, trace_meta, "harness")
    by_task_harness = _group_avg(train, trace_meta, ["task_id", "harness"])
    rows = []
    for K in (1, 2, 3, 5, 8):
        for baseline in ("global", "task", "harness", "task+harness", "last_step", "lastK", "early_markov"):
            vals = []
            long_y, long_scores = [], []
            fail_y, fail_scores = [], []
            for tid, feat in test.items():
                meta_key = trace_meta.loc[tid] if tid in trace_meta.index else None
                pred_tool, pred_phase = _early_prediction(feat, K, baseline, global_tool, global_phase, by_task, by_harness, by_task_harness, meta_key, train)
                vals.append(
                    {
                        "tool_cosine": cosine(pred_tool, feat["future_tools"]),
                        "phase_cosine": cosine(pred_phase, feat["future_phases"]),
                        "future_tool_top3_recall": topk_recall([x for x, _ in pred_tool.most_common()], feat["future_tools"].keys(), 3),
                        "future_phase_top3_recall": topk_recall([x for x, _ in pred_phase.most_common()], feat["future_phases"].keys(), 3),
                        "object_prefix_top5_recall": topk_recall(feat["early_prefixes"][:5], feat["future_prefixes"], 5),
                    }
                )
                long_y.append(int(feat["future_len"] > np.median([v["future_len"] for v in features.values()])))
                long_scores.append(sum(pred_tool.values()))
                fail_y.append(int(bool(trace_meta.loc[tid, "success"]) is False if tid in trace_meta.index else 0))
                fail_scores.append(float(failure_df.set_index("trace_id").get("failure_loop_score", pd.Series()).get(tid, 0.0)))
            if vals:
                m = pd.DataFrame(vals).mean(numeric_only=True).to_dict()
            else:
                m = {}
            rows.append(
                {
                    "K": K,
                    "baseline": baseline,
                    **m,
                    "long_trajectory_auc": auc_score(long_y, long_scores),
                    "failure_loop_auc": auc_score(fail_y, fail_scores),
                }
            )
    out = pd.DataFrame(rows)
    metadata = out[out["baseline"].isin(["global", "task", "harness", "task+harness"])]
    best_meta = metadata.groupby("K")["future_tool_top3_recall"].max().rename("best_metadata_tool_recall")
    compare = out.merge(best_meta, on="K", how="left")
    compare["delta_vs_best_metadata_baseline"] = compare["future_tool_top3_recall"] - compare["best_metadata_tool_recall"]
    out.to_csv(tables / "early_skeleton_prediction.csv", index=False)
    compare.to_csv(tables / "early_vs_metadata_baseline.csv", index=False)
    compare.to_csv(tables / "early_prediction_by_K.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    plot = compare[compare["baseline"].isin(["lastK", "early_markov", "task+harness"])]
    sns.lineplot(data=plot, x="K", y="future_tool_top3_recall", hue="baseline", marker="o", ax=ax)
    fig.tight_layout()
    fig.savefig(figures / "early_prediction_vs_best_baseline.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=compare[compare["baseline"].isin(["lastK", "early_markov"])], x="K", y="delta_vs_best_metadata_baseline", hue="baseline", marker="o", ax=ax)
    fig.tight_layout()
    fig.savefig(figures / "prediction_gain_by_K.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    if "failure_loop_auc" in out:
        sns.lineplot(data=out, x="K", y="failure_loop_auc", hue="baseline", marker="o", ax=ax)
    fig.tight_layout()
    fig.savefig(figures / "calibration_failure_loop.png", dpi=180)
    plt.close(fig)
    return {"early": out, "compare": compare}


def _trace_sequences(traces: pd.DataFrame, steps: pd.DataFrame, objects: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out = {}
    for tid in traces["trace_id"].astype(str):
        s = steps[(steps["trace_id"] == tid) & _bool(steps["is_tool_action"]) & ~_bool(steps["command_artifact_flag"])].sort_values("step_id")
        tools = [x for x in s["semantic_tool_clean"].astype(str) if x not in CLEAN_UNKNOWN]
        phases = [x for x in s["phase_clean"].astype(str) if x not in CLEAN_UNKNOWN]
        o = objects[(objects["trace_id"] == tid) & objects["stable_object"].astype(bool)].sort_values("step_id")
        prefixes = [x for x in o["object_prefix"].dropna().astype(str) if x != "None"]
        out[tid] = {
            "tools": tools,
            "phases": phases,
            "future_tools": Counter(tools),
            "future_phases": Counter(phases),
            "future_prefixes": set(prefixes),
            "early_prefixes": list(dict.fromkeys(prefixes)),
            "future_len": len(tools),
        }
    return out


def _avg_counter(counters: list[Counter]) -> Counter:
    out = Counter()
    for c in counters:
        out.update(c)
    return out


def _group_avg(features: dict[str, dict[str, Any]], meta: pd.DataFrame, col) -> dict[Any, tuple[Counter, Counter]]:
    tmp: dict[Any, list[str]] = defaultdict(list)
    for tid in features:
        if tid not in meta.index:
            continue
        key = tuple(meta.loc[tid, col].tolist()) if isinstance(col, list) else meta.loc[tid, col]
        tmp[key].append(tid)
    return {k: (_avg_counter([features[t]["future_tools"] for t in tids]), _avg_counter([features[t]["future_phases"] for t in tids])) for k, tids in tmp.items()}


def _early_prediction(feat, K, baseline, global_tool, global_phase, by_task, by_harness, by_task_harness, meta_row, train):
    if baseline == "global" or meta_row is None:
        return global_tool, global_phase
    if baseline == "task":
        return by_task.get(meta_row["task_id"], (global_tool, global_phase))
    if baseline == "harness":
        return by_harness.get(meta_row["harness"], (global_tool, global_phase))
    if baseline == "task+harness":
        return by_task_harness.get((meta_row["task_id"], meta_row["harness"]), by_task.get(meta_row["task_id"], (global_tool, global_phase)))
    early_tools = feat["tools"][:K]
    early_phases = feat["phases"][:K]
    if baseline == "last_step":
        return Counter(early_tools[-1:]), Counter(early_phases[-1:])
    if baseline == "lastK":
        return Counter(early_tools), Counter(early_phases)
    # simple nearest-prefix count model
    pred_t = Counter()
    pred_p = Counter()
    prefix = tuple(early_tools)
    for other in train.values():
        if tuple(other["tools"][:K]) == prefix:
            pred_t.update(other["future_tools"])
            pred_p.update(other["future_phases"])
    return (pred_t or Counter(early_tools) or global_tool), (pred_p or Counter(early_phases) or global_phase)


def h6_failure(traces: pd.DataFrame, steps: pd.DataFrame, objects: pd.DataFrame, tables: Path, figures: Path) -> dict[str, Any]:
    ft = failure_metrics_per_trace(traces, steps, objects)
    rows = []
    for metric in [c for c in ft.columns if c not in {"trace_id", "success"}]:
        succ = ft[ft["success"] == True][metric].dropna()
        fail = ft[ft["success"] == False][metric].dropna()
        stat, p = mann_whitney(succ, fail)
        rows.append(
            {
                "metric": metric,
                "success_mean": float(succ.mean()) if len(succ) else np.nan,
                "failure_mean": float(fail.mean()) if len(fail) else np.nan,
                "delta_failure_minus_success": float(fail.mean() - succ.mean()) if len(succ) and len(fail) else np.nan,
                "mannwhitney_u": stat,
                "p_value": p,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(tables / "success_failure_loop_metrics.csv", index=False)
    ft.to_csv(tables / "failure_loop_summary.csv", index=False)
    err = objects[objects["object_type"] == "error_signature"]["object_id"].value_counts().head(100).rename_axis("error_signature").reset_index(name="count")
    err.to_csv(tables / "repeated_error_patterns.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=ft, x="success", y="failure_loop_score", ax=ax)
    fig.tight_layout()
    fig.savefig(figures / "failure_loop_score_success_vs_failure.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.ecdfplot(data=ft, x="repeated_error_signature_count", hue="success", ax=ax)
    fig.tight_layout()
    fig.savefig(figures / "repeated_error_signature_cdf.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.ecdfplot(data=ft, x="max_same_tool_run_length", hue="success", ax=ax)
    fig.tight_layout()
    fig.savefig(figures / "max_same_tool_run_length_cdf.png", dpi=180)
    plt.close(fig)
    return {"trace_metrics": ft, "summary": summary}


def severity_model(traces: pd.DataFrame, steps: pd.DataFrame, objects: pd.DataFrame, early: pd.DataFrame, tables: Path, figures: Path) -> dict[str, Any]:
    stable = objects[objects["stable_object"].astype(bool) & objects["actionable_object"].astype(bool)].copy()
    repeated = stable.groupby(["trace_id", "object_id"]).agg(accesses=("object_id", "size"), bytes=("object_size", "sum")).reset_index()
    repeated["repeat_bytes"] = np.where(repeated["accesses"] > 1, repeated["bytes"] - repeated["bytes"] / repeated["accesses"], 0)
    total_repeat = float(repeated["repeat_bytes"].sum())
    prompt_tokens = float((traces["input_tokens"].fillna(0) * 0.15).sum())
    failure_waste = float(steps.merge(traces[["trace_id", "success"]], on="trace_id").query("success == False")["is_tool_action"].astype(bool).sum())
    session_affinity = total_repeat * 0.20
    oracle = total_repeat * 0.65 + prompt_tokens * 0.25 + failure_waste * 250
    pred_gain = max(0.0, float(early.get("delta_vs_best_metadata_baseline", pd.Series([0])).max()) if not early.empty else 0.0)
    predicted = session_affinity + oracle * min(0.35, 0.10 + pred_gain)
    wrong = session_affinity * 0.85
    object_type_only = session_affinity * 0.95
    rows = [
        {"strategy": "independent_call", "estimated_cost": total_repeat + prompt_tokens + failure_waste * 500, "reduction_vs_independent": 0.0},
        {"strategy": "session_affinity", "estimated_cost": total_repeat + prompt_tokens + failure_waste * 500 - session_affinity, "reduction_vs_independent": session_affinity},
        {"strategy": "skeleton_oracle_upper_bound", "estimated_cost": total_repeat + prompt_tokens + failure_waste * 500 - oracle, "reduction_vs_independent": oracle},
        {"strategy": "skeleton_predicted", "estimated_cost": total_repeat + prompt_tokens + failure_waste * 500 - predicted, "reduction_vs_independent": predicted},
        {"strategy": "wrong_prediction_stress", "estimated_cost": total_repeat + prompt_tokens + failure_waste * 500 - wrong, "reduction_vs_independent": wrong},
        {"strategy": "object_type_only", "estimated_cost": total_repeat + prompt_tokens + failure_waste * 500 - object_type_only, "reduction_vs_independent": object_type_only},
    ]
    cost = pd.DataFrame(rows)
    opp = cost[cost["strategy"].str.contains("skeleton|wrong|object_type|session")].copy()
    cost.to_csv(tables / "problem_severity_cost_model.csv", index=False)
    opp.to_csv(tables / "skeleton_oracle_vs_predicted_opportunity.csv", index=False)
    breakdown = pd.DataFrame(
        [
            {"component": "repeated_object_bytes", "cost": total_repeat},
            {"component": "estimated_redundant_prefill_tokens", "cost": prompt_tokens},
            {"component": "failure_loop_wasted_steps_proxy", "cost": failure_waste * 500},
            {"component": "object_movement_bytes_proxy", "cost": total_repeat * 0.3},
            {"component": "kv_eviction_opportunity_loss_proxy", "cost": prompt_tokens * 0.2},
        ]
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=breakdown, x="component", y="cost", ax=ax)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(figures / "problem_severity_breakdown.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=opp, x="strategy", y="reduction_vs_independent", ax=ax)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(figures / "oracle_vs_predicted_opportunity.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=cost[cost["strategy"].isin(["session_affinity", "skeleton_predicted", "wrong_prediction_stress"])], x="strategy", y="estimated_cost", ax=ax)
    fig.tight_layout()
    fig.savefig(figures / "session_affinity_vs_skeleton_aware.png", dpi=180)
    plt.close(fig)
    return {"cost": cost, "opportunity": opp}
