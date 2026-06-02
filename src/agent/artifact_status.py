import argparse
import csv
import json
from pathlib import Path


def _load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_is_placeholder(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return len(rows) == 0
    except Exception:
        return False


def _trace_status(root: Path):
    selection = _load_json(root / "agent_results" / "trace_selection_hf" / "selection_report.json")
    high_quality_dir = root / "traces" / "real_high_quality" / "opportunity"
    hf_usable = int(selection.get("num_paper_usable", 0)) if selection else 0
    matched_control = int(selection.get("num_matched_control", 0)) if selection else 0
    local_high_quality_files = list(high_quality_dir.glob("*.json")) if high_quality_dir.exists() else []
    return {
        "hf_selection_report_exists": selection is not None,
        "hf_paper_usable_traces": hf_usable,
        "hf_matched_control_traces": matched_control,
        "local_high_quality_opportunity_files": len(local_high_quality_files),
        "current_data_quality": "paper_usable_partial" if hf_usable else "smoke_only_or_missing",
    }


def _synthetic_status(root: Path):
    manifest = _load_json(root / "traces" / "synthetic_agent" / "synthetic_manifest.json")
    selection = _load_json(root / "agent_results" / "trace_selection_synthetic" / "selection_report.json")
    eval_dir = root / "agent_results" / "paper_eval_synthetic"
    summaries = [str(item) for item in eval_dir.glob("*summary.csv")]
    figures = list((root / "agent_results" / "paper_figures_synthetic").glob("*.pdf"))
    profile = _load_json(root / "agent_results" / "synthetic_trace_profile.json")
    return {
        "synthetic_manifest_exists": manifest is not None,
        "generated_trace_count": len(manifest.get("scenarios", [])) if manifest else 0,
        "paper_usable_for_testing": int((manifest.get("audit_summary") or {}).get("paper_usable_trace_count", 0)) if manifest else 0,
        "selection_report_exists": selection is not None,
        "selected_opportunity": int(selection.get("num_opportunity_rich", 0)) if selection else 0,
        "selected_control": int(selection.get("num_matched_control", 0)) if selection else 0,
        "paper_eval_summaries": summaries,
        "figure_count": len(figures),
        "delayed_reuse_ratio": (profile or {}).get("delayed_reuse_ratio"),
        "LRU_regret_tokens": (profile or {}).get("LRU_regret_tokens"),
        "can_support_real_paper_claims": False,
    }


def build_artifact_status(root: Path):
    backend_smoke = _load_json(root / "agent_results" / "backend_smoke.json")
    backend_smoke_passed = bool(backend_smoke and backend_smoke.get("status") == "passed")
    paper_eval_dirs = [
        root / "agent_results" / "paper_eval_hf_tight_memory",
        root / "agent_results" / "paper_eval_hf",
        root / "agent_results" / "paper_eval",
    ]
    paper_eval_summaries = []
    for path in paper_eval_dirs:
        if (path / "summary.csv").exists():
            paper_eval_summaries.append(str(path / "summary.csv"))
        paper_eval_summaries.extend(str(item) for item in path.glob("*summary.csv"))
    placeholder_csvs = []
    skipped_reports = []
    for path in paper_eval_dirs:
        for name in ("mapping_ablation.csv", "prefetch_ablation.csv"):
            candidate = path / name
            if _csv_is_placeholder(candidate):
                placeholder_csvs.append(str(candidate))
        for name in ("mapping_ablation_skipped.json", "prefetch_ablation_skipped.json"):
            candidate = path / name
            if candidate.exists():
                skipped_reports.append(str(candidate))

    trace_status = _trace_status(root)
    synthetic_status = _synthetic_status(root)
    if not backend_smoke_passed:
        readiness = "prototype"
    elif trace_status["hf_paper_usable_traces"] > 0 and trace_status["hf_matched_control_traces"] > 0 and paper_eval_summaries and not placeholder_csvs:
        readiness = "paper-ready partial"
    else:
        readiness = "prototype / smoke-test only"

    return {
        "status": "computed",
        "readiness": readiness,
        "backend_smoke_passed": backend_smoke_passed,
        "backend_smoke_path": "agent_results/backend_smoke.json",
        "trace_status": trace_status,
        "synthetic_status": synthetic_status,
        "paper_eval_summaries": paper_eval_summaries,
        "placeholder_csvs": placeholder_csvs,
        "ablation_skipped_reports": skipped_reports,
        "code_path": [
            "ASG Builder",
            "Persistent State Planner",
            "Wafer Mapper",
            "Event Compiler",
            "BusyBarn Event Backend",
        ],
        "backend_mode": "event-level BusyBarn backend with fixed-duration analytical LLM prefill/decode compute events",
        "backend_boundary": "KV movement uses BusyBarn communication/routing/link scheduling; LLM prefill/decode do not use BusyBarn's original operator-level partition pipeline.",
        "notes": [
            "Do not claim full paper-readiness without matched controls and broader paper-usable traces.",
            "Synthetic traces can validate algorithm behavior but cannot support real-trace paper claims.",
            "Skipped ablation JSON files are acceptable; empty placeholder ablation CSVs are not.",
        ],
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Report Agent-on-Wafer artifact readiness.")
    parser.add_argument("--output", default="agent_results/artifact_status.json")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    report = build_artifact_status(root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"artifact_status readiness={report['readiness']} output={output}")


if __name__ == "__main__":
    main()
