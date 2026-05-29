#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skeleton.experiments import (
    ensure_dirs,
    h1_transitions,
    h2_motifs,
    h3_dependencies,
    h4_object_working_set,
    h5_early,
    h6_failure,
    severity_model,
)
from skeleton.loader import load_terminalbench_strict_real
from skeleton.normalize import normalize_terminalbench_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="terminalbench")
    p.add_argument("--sample-size", type=int, default=1500)
    p.add_argument("--strict-real", action="store_true")
    p.add_argument("--profile-mode", default="skeleton_existence")
    p.add_argument("--outdir", default="outputs/skeleton_existence_tb1500")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--page-size", type=int, default=10)
    p.add_argument("--resume", action="store_true", default=True)
    return p.parse_args()


def _write_tables(dirs: dict[str, Path], traces: pd.DataFrame, steps: pd.DataFrame, objects: pd.DataFrame, deps: pd.DataFrame) -> None:
    data = dirs["data"]
    traces.to_csv(data / "traces.csv", index=False)
    steps.to_csv(data / "steps.csv", index=False)
    objects.to_csv(data / "object_accesses.csv", index=False)
    deps.to_csv(data / "data_dependencies.csv", index=False)


def _data_quality(traces: pd.DataFrame, steps: pd.DataFrame, objects: pd.DataFrame, load_meta: dict, norm_meta: dict, args: argparse.Namespace) -> pd.DataFrame:
    unknown = float((steps["semantic_tool_clean"].fillna("unknown") == "unknown").mean()) if len(steps) else 0
    artifact = float(steps["command_artifact_flag"].astype(bool).mean()) if len(steps) else 0
    parse = float(steps["command_string"].notna().mean()) if len(steps) else 0
    return pd.DataFrame(
        [
            {
                "dataset": "terminalbench",
                "num_traces": len(traces),
                "num_steps": len(steps),
                "num_tool_action_steps": int(steps["is_tool_action"].astype(bool).sum()) if len(steps) else 0,
                "num_object_accesses": len(objects),
                "used_mock": False,
                "strict_real": bool(args.strict_real),
                "skipped_rows": int(load_meta.get("skipped_rows", 0) + norm_meta.get("skipped_rows", 0)),
                "unknown_rate": unknown,
                "artifact_rate": artifact,
                "command_parse_success_rate": parse,
                "source": load_meta.get("source"),
                "profile_mode": args.profile_mode,
            }
        ]
    )


def _verdicts(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    h1r = results["h1_recall"]
    h1p = results["h1_perm"]
    clean = h1r[(h1r["view"].isin(["semantic_tool_tool_action_only", "collapsed_semantic_tool_tool_action_only"])) & (h1r["k"] == 3)]
    perm = h1p[h1p["view"].isin(clean["view"]) & (h1p["control"] != "wrapper_only")]
    h1 = "supported" if (not clean.empty and clean["delta_vs_global"].max() > 0.08 and clean["ci_low"].max() > 0 and (perm["p_value"] < 0.05).any()) else "not supported"

    mot = results["motif_gen"]
    h2 = "supported" if (not mot.empty and (mot["appears_in_heldout"].sum() >= 3) and (mot["support_harnesses"].max() > 1)) else "partially supported" if not mot.empty and mot["appears_in_heldout"].any() else "not supported"

    dep = results["dep_summary"]
    dep_count = int(dep["count"].sum()) if "count" in dep else 0
    dep_med = float(dep["median_distance"].median()) if "median_distance" in dep and len(dep) else 999
    h3 = "supported" if dep_count > 0 and dep_med <= 5 else "not supported"

    obj_pred = results["object_pred"]
    h4 = "not supported"
    if not obj_pred.empty:
        exact = obj_pred[(obj_pred["granularity"] == "exact_object_id") & (obj_pred["horizon"].isin([3, 5, 10]))]
        lastk = exact[exact["baseline"] == "lastK_object"]["top5_recall"].mean()
        rand = exact[exact["baseline"] == "random_same_count"]["top5_recall"].mean()
        otype = exact[exact["baseline"] == "object_type_only"]["top5_recall"].mean()
        if pd.notna(lastk) and lastk > max(rand if pd.notna(rand) else -1, otype if pd.notna(otype) else -1) + 0.05:
            h4 = "supported"

    early = results["early_compare"]
    h5 = "not supported"
    if not early.empty:
        candidate = early[(early["baseline"].isin(["lastK", "early_markov"])) & (early["K"].isin([3, 5, 8]))]
        if (candidate["delta_vs_best_metadata_baseline"] > 0).sum() >= 2:
            h5 = "supported"
        elif (candidate["delta_vs_best_metadata_baseline"] > 0).any():
            h5 = "partially supported"

    fail = results["failure_summary"]
    h6 = "not supported"
    row = fail[fail["metric"] == "failure_loop_score"] if not fail.empty else pd.DataFrame()
    if not row.empty and float(row["delta_failure_minus_success"].iloc[0]) > 0 and float(row["p_value"].iloc[0]) < 0.05:
        h6 = "supported"
    elif not row.empty and float(row["delta_failure_minus_success"].iloc[0]) > 0:
        h6 = "partially supported"

    return pd.DataFrame(
        [
            {"hypothesis": "H1 Tool/Phase Skeleton", "verdict": h1},
            {"hypothesis": "H2 Cross-task Motifs", "verdict": h2},
            {"hypothesis": "H3 Data Dependency", "verdict": h3},
            {"hypothesis": "H4 Object Working Set", "verdict": h4},
            {"hypothesis": "H5 Early Skeleton Matching", "verdict": h5},
            {"hypothesis": "H6 Failure-loop Skeleton", "verdict": h6},
        ]
    )


def _write_reports(outdir: Path, dq: pd.DataFrame, verdicts: pd.DataFrame, results: dict[str, pd.DataFrame], args: argparse.Namespace) -> None:
    reports = outdir / "reports"
    tables = outdir / "tables"
    sev = results["severity"]
    predicted = sev[sev["strategy"] == "skeleton_predicted"]["reduction_vs_independent"].iloc[0]
    session = sev[sev["strategy"] == "session_affinity"]["reduction_vs_independent"].iloc[0]
    wrong = sev[sev["strategy"] == "wrong_prediction_stress"]["reduction_vs_independent"].iloc[0]
    enough = (
        (verdicts["verdict"] == "supported").sum() >= 4
        and predicted > session
        and predicted > wrong
    )
    conclusion = (
        "问题存在。动态 agent workflow 中存在可利用的静态 stateflow skeleton，可以进入 SkeletonFlow 算法设计阶段。"
        if enough
        else "问题没有被充分证明，暂时不应进入系统算法设计。"
    )
    h1 = results["h1_recall"]
    h1_clean = h1[(h1["view"] == "semantic_tool_tool_action_only") & (h1["k"] == 3)].head(1)
    dep = results["dep_summary"]
    obj = results["object_pred"]
    fail = results["failure_summary"]
    report = f"""# Problem Existence Report

## 1. Research Question

We test, from scratch, whether dynamic agent workflows contain reusable static stateflow skeletons. No previous profile result is used as a premise.

## 2. Data and Cleaning

{dq.to_markdown(index=False)}

Strict-real is `{bool(dq['strict_real'].iloc[0])}` and `used_mock=false`.

## 3. Hypothesis Results

{verdicts.to_markdown(index=False)}

H1 metric: semantic tool-action-only Top-3 conditional recall delta vs global = `{float(h1_clean['delta_vs_global'].iloc[0]) if not h1_clean.empty else float('nan'):.4f}`, CI = `[{float(h1_clean['ci_low'].iloc[0]) if not h1_clean.empty else float('nan'):.4f}, {float(h1_clean['ci_high'].iloc[0]) if not h1_clean.empty else float('nan'):.4f}]`. Baseline: global next-tool frequency. Negative controls: within-trace shuffle, global next-label shuffle, frequency-preserving temporal break.

H2 metric: held-out motif reproduction from train-mined motifs. Baseline/control: held-out trace split and harness support count. See `{tables / 'motif_generalization.csv'}`.

H3 metric: dependency count `{int(dep['count'].sum()) if not dep.empty and 'count' in dep else 0}`, median dependency distance `{float(dep['median_distance'].median()) if not dep.empty and 'median_distance' in dep else float('nan'):.2f}`. Baseline/control: observation-only path mentions are not counted; dependency requires later command use.

H4 metric: object prediction baselines at exact object-id and path-prefix granularity. Main comparison: last-K exact objects vs random and object_type-only. See `{tables / 'object_prediction_baselines.csv'}`.

H5 metric: early K=1/2/3/5/8 prediction delta vs best metadata baseline. See `{tables / 'early_vs_metadata_baseline.csv'}`.

H6 metric: failure_loop_score success vs failure, Mann-Whitney p-value `{float(fail[fail['metric']=='failure_loop_score']['p_value'].iloc[0]) if not fail.empty and (fail['metric']=='failure_loop_score').any() else float('nan'):.4g}`. Baseline: success traces.

## 4. Static Skeleton Evidence

- Phase/tool skeleton: see H1 transition recall, entropy/MI, permutation tests.
- Cross-task motif skeleton: see frequent motifs and held-out support.
- Data dependency skeleton: file/url/test/error dependencies require future command use.
- Object working set skeleton: exact object_id and path_prefix are analyzed separately from object_type.
- KV skeleton opportunity: estimated only as repeated prompt/prefix opportunity proxy, not as proven KV reuse.
- Failure-loop skeleton: compared with success/failure statistics, not observation-byte volume.

## 5. Problem Severity

{sev.to_markdown(index=False)}

Predicted skeleton-aware opportunity is `{predicted:.2f}` vs session-affinity `{session:.2f}` and wrong-prediction stress `{wrong:.2f}`.

## 6. What Is Proven

Only hypotheses with `supported` in the table above should be treated as proven for this TerminalBench sample.

## 7. What Is Not Proven

- Real wafer acceleration is not proven.
- Object/KV reuse in a production runtime is not proven.
- Generality across all agent domains is not proven.
- Speculation safety is not proven.
- End-to-end task success preservation is not proven.

## 8. Next Step

{conclusion}
"""
    (reports / "problem_existence_report.md").write_text(report, encoding="utf-8")
    audit = f"""# Problem Existence Audit

- Required strict-real run: {bool(dq['strict_real'].iloc[0])}
- Mock fallback used: {bool(dq['used_mock'].iloc[0])}
- Tool-action-only filters exclude unknown/no-tool/artifact steps.
- Object reuse is reported at exact object-id and path-prefix levels.
- Negative controls are in `tables/permutation_tests.csv` and cost stress rows.
- Main command: `python scripts/run_all.py --datasets {args.datasets} --sample-size {args.sample_size} --strict-real --profile-mode {args.profile_mode} --outdir {args.outdir}`
"""
    (reports / "problem_existence_audit.md").write_text(audit, encoding="utf-8")
    limitations = """# Limitations

- TerminalBench observations are truncated to 5,000 chars, so dependencies are lower-bound estimates.
- Dataset-server partial loading may produce fewer rows if the network fails; metadata records the actual count.
- Confidence intervals are trace-bootstrap estimates, not a full hierarchical model.
- The severity model is a problem-opportunity proxy, not a runtime speedup experiment.
- KV stability is inferred from repeated prefixes/tokens and is not direct KV-cache telemetry.
- Object future-window prediction uses all tool steps up to 20,000 evaluation points for runtime control on large samples.
"""
    (reports / "limitations.md").write_text(limitations, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.datasets.strip().lower() != "terminalbench":
        raise SystemExit("This skeleton_existence first-stage runner currently supports --datasets terminalbench only.")
    if args.profile_mode != "skeleton_existence":
        raise SystemExit("Use --profile-mode skeleton_existence for this runner.")
    if not args.strict_real:
        raise SystemExit("strict-real is required; mock fallback is disabled for this task.")
    outdir = Path(args.outdir)
    dirs = ensure_dirs(outdir)
    raw_path = dirs["data"] / "terminalbench_raw.jsonl"

    print("[run_all] loading strict-real TerminalBench rows", flush=True)
    rows, load_meta = load_terminalbench_strict_real(args.sample_size, raw_path, page_size=args.page_size, resume=args.resume)
    print(f"[run_all] normalizing {len(rows)} rows", flush=True)
    traces, steps, objects, deps, norm_meta = normalize_terminalbench_rows(rows)
    _write_tables(dirs, traces, steps, objects, deps)

    dq = _data_quality(traces, steps, objects, load_meta, norm_meta, args)
    dq.to_csv(dirs["tables"] / "data_quality_summary.csv", index=False)
    metadata = dq.iloc[0].to_dict()
    metadata.update(load_meta)
    metadata.update(norm_meta)
    metadata["used_mock"] = False
    metadata["strict_real"] = True
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")

    print("[run_all] H1 transitions/entropy/permutation", flush=True)
    h1 = h1_transitions(steps, dirs["tables"], dirs["figures"], args.seed)
    print("[run_all] H2 motifs", flush=True)
    h2 = h2_motifs(traces, steps, objects, dirs["tables"], dirs["figures"], args.seed)
    print("[run_all] H3 data dependencies", flush=True)
    h3 = h3_dependencies(deps, steps, dirs["tables"], dirs["figures"])
    print("[run_all] H4 object working set", flush=True)
    h4 = h4_object_working_set(traces, steps, objects, dirs["tables"], dirs["figures"], args.seed)
    print("[run_all] H6 failure loops", flush=True)
    h6 = h6_failure(traces, steps, objects, dirs["tables"], dirs["figures"])
    print("[run_all] H5 early skeleton matching", flush=True)
    h5 = h5_early(traces, steps, objects, h6["trace_metrics"], dirs["tables"], dirs["figures"], args.seed)
    print("[run_all] severity model and reports", flush=True)
    sev = severity_model(traces, steps, objects, h5["compare"], dirs["tables"], dirs["figures"])

    results = {
        "h1_recall": h1["recall"],
        "h1_perm": h1["permutation"],
        "motif_gen": h2["motif_generalization"],
        "dep_summary": h3["summary"],
        "object_pred": h4["prediction"],
        "early_compare": h5["compare"],
        "failure_summary": h6["summary"],
        "severity": sev["cost"],
    }
    verdicts = _verdicts(results)
    verdicts.to_csv(dirs["tables"] / "hypothesis_verdicts.csv", index=False)
    _write_reports(outdir, dq, verdicts, results, args)
    print(json.dumps({"outdir": str(outdir), "num_traces": len(traces), "num_steps": len(steps), "num_tool_action_steps": int(steps["is_tool_action"].astype(bool).sum()), "used_mock": False}, indent=2))


if __name__ == "__main__":
    main()
