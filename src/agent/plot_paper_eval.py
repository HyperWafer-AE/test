import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


FIGURE_NAMES = [
    "fig_problem_motivation",
    "fig_workload_characterization",
    "fig_concurrency_memory_sweep",
    "fig_online_vs_oracle_gap",
    "fig_opportunity_vs_control",
    "fig_mapping_ablation",
    "fig_prefetch_ablation",
]

POLICY_LABELS = {
    "nocache": "No cache",
    "lru-system": "LRU",
    "kvflow-like": "KVFlow-like",
    "asg-retention-v2-graph-only": "ASG graph",
    "asg-retention-v2-online": "ASG online",
    "asg-retention-v2-trace-stats": "ASG stats",
    "asg-retention-v2-oracle": "ASG oracle",
    "asg-placement-v2-online": "ASG place",
    "asg-prefetch-v2-online": "ASG prefetch",
}

PALETTE = {
    "gray": "#4D4D4D",
    "light_gray": "#C7C7C7",
    "blue": "#2F6B9A",
    "orange": "#D4852B",
    "green": "#4E8A57",
    "red": "#B04A3F",
    "purple": "#6A5C9F",
}


def plot_paper_eval(input_dir, output_dir, trace_selection_dir=None):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    skip_file = input_dir / "skipped_missing_opportunity_traces.json"
    summaries = list(input_dir.glob("*summary.csv")) + list(input_dir.glob("*sweep.csv")) + list(input_dir.glob("*ablation.csv"))
    if skip_file.exists() or not summaries:
        report = {
            "status": "skipped_missing_paper_usable_data",
            "input_dir": str(input_dir),
            "reason": "No paper-usable evaluation CSVs were available; figures were not generated.",
        }
        (output_dir / "skipped_missing_paper_usable_data.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        report = {"status": "skipped_plot_dependency_missing", "reason": str(exc)}
        (output_dir / "skipped_missing_paper_usable_data.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    _set_isca_style(plt)
    selection_dir = Path(trace_selection_dir) if trace_selection_dir else _find_trace_selection_dir(input_dir)
    context = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "selection_dir": selection_dir,
        "memory_rows": _load_eval_csv(input_dir, "memory_sweep.csv", include_siblings=True),
        "concurrency_rows": _load_eval_csv(input_dir, "concurrency_sweep.csv", include_siblings=True),
        "arrival_rows": _load_eval_csv(input_dir, "arrival_model_sweep.csv", include_siblings=True),
        "full_rows": _load_eval_csv(input_dir, "full_set_summary.csv", include_siblings=False),
        "opportunity_rows": _load_eval_csv(input_dir, "opportunity_subset_summary.csv", include_siblings=False),
        "control_rows": _load_eval_csv(input_dir, "control_subset_summary.csv", include_siblings=False),
        "selection_rows": _load_selection_rows(selection_dir),
        "selection_report": _load_json(selection_dir / "selection_report.json") if selection_dir else {},
    }
    context["evidence_class"] = _evidence_class(context)

    generated = []
    data_files = []
    evidence = {}
    for name, plotter in [
        ("fig_problem_motivation", _plot_problem_motivation),
        ("fig_workload_characterization", _plot_workload_characterization),
        ("fig_concurrency_memory_sweep", _plot_concurrency_memory_sweep),
        ("fig_online_vs_oracle_gap", _plot_online_vs_oracle_gap),
        ("fig_opportunity_vs_control", _plot_opportunity_vs_control),
        ("fig_mapping_ablation", _plot_mapping_ablation),
        ("fig_prefetch_ablation", _plot_prefetch_ablation),
    ]:
        result = plotter(plt, context, name)
        generated.extend(result["figures"])
        data_files.append(result["data_csv"])
        evidence[name] = result["evidence"]

    report = {
        "status": "generated",
        "input_dir": str(input_dir),
        "trace_selection_dir": str(selection_dir) if selection_dir else None,
        "evidence_class": context["evidence_class"],
        "can_support_real_paper_claims": context["evidence_class"] == "real",
        "figures": generated,
        "data_csv": data_files,
        "evidence": evidence,
        "limitations": _limitations(context),
    }
    (output_dir / "figure_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _plot_problem_motivation(plt, context, name):
    memory = _memory_policy_table(context["memory_rows"])
    selected = [row for row in context["selection_rows"] if _truthy(row.get("paper_usable"))]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.2))

    trace_rows = []
    ax = axes[0][0]
    if selected:
        labels = [f"T{i + 1}" for i in range(len(selected))]
        pressure = [_num(row, "estimated_memory_pressure_ratio") for row in selected]
        regret = [_num(row, "LRU_regret_tokens") for row in selected]
        ax.bar(labels, pressure, color=PALETTE["blue"], edgecolor="black", linewidth=0.6)
        ax.axhline(1.0, color=PALETTE["red"], linestyle="--", linewidth=1.0)
        ax.set_ylabel("Live KV / budget")
        ax.set_title(f"(a) {_trace_label(context)} traces exceed KV budget")
        for x, y, reg in zip(labels, pressure, regret):
            ax.text(x, y + max(pressure) * 0.035, f"{int(reg)} tok", ha="center", va="bottom", fontsize=7)
            trace_rows.append({"panel": "trace_pressure", "trace": x, "live_kv_over_budget": y, "lru_regret_tokens": reg})
    else:
        _missing_axis(ax, f"No {_trace_label(context).lower()} trace rows")

    ax = axes[0][1]
    mem_points = _policy_points(memory, "lru-system", "cache_byte_miss", scale=1e9)
    if mem_points:
        _plot_line(ax, mem_points, "LRU", PALETTE["gray"], marker="o")
        asg_points = _policy_points(memory, "asg-retention-v2-online", "cache_byte_miss", scale=1e9)
        _plot_line(ax, asg_points, "ASG online", PALETTE["blue"], marker="s")
        ax.set_xscale("log")
        ax.set_xlabel("State memory budget (GB)")
        ax.set_ylabel("KV miss traffic (GB)")
        ax.set_title("(b) Limited state memory causes miss traffic")
        ax.legend(frameon=False, fontsize=7)
        for mem, value in mem_points:
            trace_rows.append({"panel": "memory_cliff", "memory_budget_gb": mem, "policy": "lru-system", "cache_byte_miss_gb": value})
    else:
        _missing_axis(ax, "No memory sweep")

    ax = axes[1][0]
    policies = ["lru-system", "asg-retention-v2-online", "asg-retention-v2-oracle"]
    colors = [PALETTE["gray"], PALETTE["blue"], PALETTE["green"]]
    plotted = False
    for policy, color in zip(policies, colors):
        points = _policy_points(memory, policy, "LRU_regret_tokens_preserved")
        if points:
            _plot_line(ax, points, POLICY_LABELS[policy], color, marker="o")
            plotted = True
            for mem, value in points:
                trace_rows.append({"panel": "delayed_reuse_retention", "memory_budget_gb": mem, "policy": policy, "tokens_preserved": value})
    if plotted:
        ax.set_xscale("log")
        ax.set_xlabel("State memory budget (GB)")
        ax.set_ylabel("Delayed-reuse tokens kept")
        ax.set_title("(c) LRU drops valuable delayed state")
        ax.legend(frameon=False, fontsize=7)
    else:
        _missing_axis(ax, "No delayed-reuse rows")

    ax = axes[1][1]
    placement = _policy_points(memory, "asg-placement-v2-online", "remote_read_bytes", scale=1e9)
    prefetch = _policy_points(memory, "asg-prefetch-v2-online", "prefetch_migration_bytes", scale=1e9)
    if placement or prefetch:
        _plot_line(ax, placement, "Remote reads", PALETTE["orange"], marker="o")
        _plot_line(ax, prefetch, "Prefetch moves", PALETTE["purple"], marker="s")
        ax.set_xscale("log")
        ax.set_xlabel("State memory budget (GB)")
        ax.set_ylabel("KV movement (GB)")
        ax.set_title("(d) Placement exposes topology traffic")
        ax.legend(frameon=False, fontsize=7)
        for mem, value in placement:
            trace_rows.append({"panel": "topology_traffic", "memory_budget_gb": mem, "metric": "remote_read_gb", "value": value})
        for mem, value in prefetch:
            trace_rows.append({"panel": "topology_traffic", "memory_budget_gb": mem, "metric": "prefetch_move_gb", "value": value})
    else:
        _missing_axis(ax, "No topology rows")

    _finish_grid(fig, axes)
    data_csv = context["output_dir"] / f"{name}.csv"
    _write_rows(data_csv, trace_rows)
    figures = _save_figure(fig, context["output_dir"], name)
    plt.close(fig)
    return {
        "figures": figures,
        "data_csv": str(data_csv),
        "evidence": {
            "num_paper_usable_traces": len(selected),
            "max_live_kv_over_budget": max([_num(row, "estimated_memory_pressure_ratio") for row in selected], default=0.0),
            "max_lru_cache_miss_gb": max([v for _, v in mem_points], default=0.0),
        },
    }


def _plot_workload_characterization(plt, context, name):
    rows = context["selection_rows"]
    report = context["selection_report"]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.25))
    data = []

    ax = axes[0]
    if report:
        labels = ["Loaded", "Smoke", "Unusable", "Opp.", "Ctrl"]
        values = [
            report.get("num_loaded", 0),
            report.get("num_smoke_only", 0),
            report.get("num_unusable", 0),
            report.get("num_opportunity_rich", report.get("num_paper_usable", 0)),
            report.get("num_matched_control", 0),
        ]
    else:
        counts = Counter(row.get("selection_label", "unknown") for row in rows)
        labels = list(counts)
        values = [counts[label] for label in labels]
    ax.bar(labels, values, color=[PALETTE["light_gray"], PALETTE["gray"], "#E2E2E2", PALETTE["blue"], PALETTE["green"]], edgecolor="black", linewidth=0.5)
    ax.set_ylabel("# traces")
    ax.set_title(f"(a) {_trace_label(context)} selection funnel")
    ax.tick_params(axis="x", rotation=25)
    for label, value in zip(labels, values):
        data.append({"panel": "selection_funnel", "label": label, "value": value})

    ax = axes[1]
    if rows:
        groups = {
            "smoke": [row for row in rows if row.get("selection_label") == "smoke_only"],
            "opp.": [row for row in rows if row.get("selection_label") == "opportunity_rich"],
        }
        box_data = [[_num(row, "estimated_memory_pressure_ratio") for row in group] for group in groups.values()]
        ax.boxplot(box_data, tick_labels=list(groups), widths=0.55, patch_artist=True, boxprops={"facecolor": "#F0F0F0", "edgecolor": "black"}, medianprops={"color": PALETTE["red"]})
        ax.axhline(1.0, color=PALETTE["red"], linestyle="--", linewidth=0.9)
        ax.set_ylabel("Live KV / budget")
        ax.set_title("(b) KV pressure distribution")
        for label, group in groups.items():
            vals = [_num(row, "estimated_memory_pressure_ratio") for row in group]
            data.append({"panel": "pressure_distribution", "label": label, "count": len(vals), "mean": _mean(vals), "max": max(vals, default=0.0)})
    else:
        _missing_axis(ax, "No selection rows")

    ax = axes[2]
    if rows:
        smoke = [row for row in rows if row.get("selection_label") == "smoke_only"]
        opp = [row for row in rows if row.get("selection_label") == "opportunity_rich"]
        metrics = [
            ("Opportunity\nscore", "asg_opportunity_score"),
            ("Delayed\nreuse", "delayed_reuse_ratio"),
            ("LRU-regret\nkTok", "LRU_regret_tokens"),
        ]
        x = range(len(metrics))
        smoke_vals = [_scaled_mean(smoke, key) for _, key in metrics]
        opp_vals = [_scaled_mean(opp, key) for _, key in metrics]
        width = 0.36
        ax.bar([i - width / 2 for i in x], smoke_vals, width=width, label="Smoke", color=PALETTE["light_gray"], edgecolor="black", linewidth=0.5)
        ax.bar([i + width / 2 for i in x], opp_vals, width=width, label="Opportunity", color=PALETTE["blue"], edgecolor="black", linewidth=0.5)
        ax.set_xticks(list(x))
        ax.set_xticklabels([m[0] for m in metrics])
        ax.set_ylabel("Scaled mean")
        ax.set_title("(c) Opportunity traces have delayed reuse")
        ax.legend(frameon=False, fontsize=7)
        for metric, key in metrics:
            data.append({"panel": "opportunity_vs_smoke", "metric": key, "smoke_mean": _scaled_mean(smoke, key), "opportunity_mean": _scaled_mean(opp, key)})
    else:
        _missing_axis(ax, "No selection rows")

    _finish_grid(fig, [axes])
    data_csv = context["output_dir"] / f"{name}.csv"
    _write_rows(data_csv, data)
    figures = _save_figure(fig, context["output_dir"], name)
    plt.close(fig)
    return {"figures": figures, "data_csv": str(data_csv), "evidence": _selection_evidence(context)}


def _plot_concurrency_memory_sweep(plt, context, name):
    memory = _memory_policy_table(context["memory_rows"])
    concurrency = _group_policy_table(context["concurrency_rows"], "concurrency")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.4))
    data = []

    ax = axes[0]
    for policy, color, marker in [
        ("lru-system", PALETTE["gray"], "o"),
        ("asg-retention-v2-online", PALETTE["blue"], "s"),
        ("nocache", PALETTE["red"], "^"),
    ]:
        points = _policy_points(memory, policy, "cache_byte_miss", scale=1e9)
        if points:
            _plot_line(ax, points, POLICY_LABELS[policy], color, marker=marker)
            for mem, value in points:
                data.append({"panel": "memory_miss", "memory_budget_gb": mem, "policy": policy, "cache_byte_miss_gb": value})
    ax.set_xscale("log")
    ax.set_xlabel("State memory budget (GB)")
    ax.set_ylabel("KV miss traffic (GB)")
    ax.set_title("(a) Memory capacity cliff")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    plotted = False
    for policy, color, marker in [
        ("lru-system", PALETTE["gray"], "o"),
        ("asg-retention-v2-online", PALETTE["blue"], "s"),
        ("kvflow-like", PALETTE["orange"], "^"),
    ]:
        points = _policy_points(concurrency, policy, "effective_prefill_tokens", normalize_by=("nocache", "effective_prefill_tokens"))
        if points:
            _plot_line(ax, points, POLICY_LABELS[policy], color, marker=marker)
            plotted = True
            for c, value in points:
                data.append({"panel": "concurrency_prefill", "concurrency": c, "policy": policy, "prefill_vs_nocache": value})
    if plotted:
        ax.set_xlabel("Concurrent workflows")
        ax.set_ylabel("Prefill tokens / no-cache")
        ax.set_title("(b) Reuse pressure under serving replay")
        ax.set_ylim(bottom=0.94, top=1.01)
        ax.legend(frameon=False, fontsize=7)
    else:
        _missing_axis(ax, "No concurrency sweep")

    _finish_grid(fig, [axes])
    data_csv = context["output_dir"] / f"{name}.csv"
    _write_rows(data_csv, data)
    figures = _save_figure(fig, context["output_dir"], name)
    plt.close(fig)
    return {"figures": figures, "data_csv": str(data_csv), "evidence": {"num_memory_points": len(memory), "num_concurrency_points": len(concurrency)}}


def _plot_online_vs_oracle_gap(plt, context, name):
    memory = _memory_policy_table(context["memory_rows"])
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.4))
    data = []

    ax = axes[0]
    for policy, color, marker in [
        ("lru-system", PALETTE["gray"], "o"),
        ("asg-retention-v2-online", PALETTE["blue"], "s"),
        ("asg-retention-v2-oracle", PALETTE["green"], "^"),
    ]:
        points = _policy_points(memory, policy, "LRU_regret_tokens_preserved")
        if points:
            _plot_line(ax, points, POLICY_LABELS[policy], color, marker=marker)
            for mem, value in points:
                data.append({"panel": "tokens_preserved", "memory_budget_gb": mem, "policy": policy, "tokens_preserved": value})
    ax.set_xscale("log")
    ax.set_xlabel("State memory budget (GB)")
    ax.set_ylabel("Delayed-reuse tokens kept")
    ax.set_title("(a) Retention of LRU-regret state")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    for policy, color, marker in [
        ("asg-retention-v2-graph-only", PALETTE["purple"], "D"),
        ("asg-retention-v2-online", PALETTE["blue"], "s"),
        ("asg-retention-v2-oracle", PALETTE["green"], "^"),
    ]:
        points = _improvement_vs_policy(memory, policy, "lru-system", "effective_prefill_tokens")
        if points:
            _plot_line(ax, points, POLICY_LABELS[policy], color, marker=marker)
            for mem, value in points:
                data.append({"panel": "prefill_reduction", "memory_budget_gb": mem, "policy": policy, "reduction_vs_lru_pct": value})
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("State memory budget (GB)")
    ax.set_ylabel("Prefill reduction vs LRU (%)")
    ax.set_title("(b) Current online gap is small")
    ax.legend(frameon=False, fontsize=7)

    _finish_grid(fig, [axes])
    data_csv = context["output_dir"] / f"{name}.csv"
    _write_rows(data_csv, data)
    figures = _save_figure(fig, context["output_dir"], name)
    plt.close(fig)
    return {"figures": figures, "data_csv": str(data_csv), "evidence": {"max_prefill_reduction_vs_lru_pct": max([float(row.get("reduction_vs_lru_pct", 0.0)) for row in data if row.get("panel") == "prefill_reduction"], default=0.0)}}


def _plot_opportunity_vs_control(plt, context, name):
    rows = context["selection_rows"]
    report = context["selection_report"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.4))
    data = []

    ax = axes[0]
    groups = {
        "Smoke": [row for row in rows if row.get("selection_label") == "smoke_only"],
        "Opportunity": [row for row in rows if row.get("selection_label") == "opportunity_rich"],
        "Control": [row for row in rows if row.get("selection_label") == "matched_control"],
    }
    metric_specs = [
        ("Delayed reuse", "delayed_reuse_ratio", 1.0),
        ("LRU regret\n(kTok)", "LRU_regret_tokens", 1000.0),
        ("Opportunity\nscore", "asg_opportunity_score", 1.0),
    ]
    width = 0.24
    x = list(range(len(metric_specs)))
    for offset, (label, group, color) in zip([-width, 0, width], [(k, v, c) for (k, v), c in zip(groups.items(), [PALETTE["light_gray"], PALETTE["blue"], PALETTE["green"]])]):
        vals = [_mean([_num(row, key) / scale for row in group]) for _, key, scale in metric_specs]
        ax.bar([i + offset for i in x], vals, width=width, label=label, color=color, edgecolor="black", linewidth=0.5)
        for spec, value in zip(metric_specs, vals):
            data.append({"panel": "intrinsic_metrics", "group": label, "metric": spec[1], "mean": value, "count": len(group)})
    ax.set_xticks(x)
    ax.set_xticklabels([spec[0] for spec in metric_specs])
    ax.set_ylabel("Mean")
    ax.set_title("(a) Opportunity traces expose delayed reuse")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    labels = ["Candidates", "Paper\nusable", "Opportunity", "Matched\ncontrol"]
    values = [
        report.get("num_candidates", len(rows)),
        report.get("num_paper_usable", sum(1 for row in rows if _truthy(row.get("paper_usable")))),
        report.get("num_opportunity_rich", len(groups["Opportunity"])),
        report.get("num_matched_control", len(groups["Control"])),
    ]
    ax.bar(labels, values, color=[PALETTE["gray"], PALETTE["blue"], PALETTE["blue"], PALETTE["green"]], edgecolor="black", linewidth=0.5)
    ax.set_ylabel("# traces")
    ax.set_title("(b) Public control traces are scarce")
    for label, value in zip(labels, values):
        data.append({"panel": "sample_counts", "label": label.replace("\n", " "), "value": value})
        ax.text(label, value + max(values or [1]) * 0.03, str(value), ha="center", va="bottom", fontsize=7)

    _finish_grid(fig, [axes])
    data_csv = context["output_dir"] / f"{name}.csv"
    _write_rows(data_csv, data)
    figures = _save_figure(fig, context["output_dir"], name)
    plt.close(fig)
    return {"figures": figures, "data_csv": str(data_csv), "evidence": {"num_matched_control": values[-1], "num_opportunity": values[-2]}}


def _plot_mapping_ablation(plt, context, name):
    memory = _memory_policy_table(context["memory_rows"])
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.4))
    data = []

    ax = axes[0]
    for policy, metric, color, marker in [
        ("asg-placement-v2-online", "remote_read_bytes", PALETTE["orange"], "o"),
        ("asg-prefetch-v2-online", "remote_read_bytes", PALETTE["purple"], "s"),
    ]:
        points = _policy_points(memory, policy, metric, scale=1e9)
        if points:
            _plot_line(ax, points, f"{POLICY_LABELS[policy]} remote", color, marker=marker)
            for mem, value in points:
                data.append({"panel": "remote_read", "memory_budget_gb": mem, "policy": policy, "remote_read_gb": value})
    ax.set_xscale("log")
    ax.set_xlabel("State memory budget (GB)")
    ax.set_ylabel("Remote-read bytes (GB)")
    ax.set_title("(a) Topology-aware state access is nonzero")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    for policy, color, marker in [
        ("asg-placement-v2-online", PALETTE["orange"], "o"),
        ("asg-prefetch-v2-online", PALETTE["purple"], "s"),
    ]:
        points = _policy_points(memory, policy, "model_comm_cycles", scale=1e6)
        if points:
            _plot_line(ax, points, POLICY_LABELS[policy], color, marker=marker)
            for mem, value in points:
                data.append({"panel": "comm_cycles", "memory_budget_gb": mem, "policy": policy, "model_comm_mcycles": value})
    ax.set_xscale("log")
    ax.set_xlabel("State memory budget (GB)")
    ax.set_ylabel("Comm cycles (M)")
    ax.set_title("(b) State placement creates communication cost")
    ax.legend(frameon=False, fontsize=7)

    _finish_grid(fig, [axes])
    data_csv = context["output_dir"] / f"{name}.csv"
    _write_rows(data_csv, data)
    figures = _save_figure(fig, context["output_dir"], name)
    plt.close(fig)
    return {"figures": figures, "data_csv": str(data_csv), "evidence": {"max_remote_read_gb": max([float(row.get("remote_read_gb", 0.0)) for row in data if row.get("panel") == "remote_read"], default=0.0)}}


def _plot_prefetch_ablation(plt, context, name):
    memory = _memory_policy_table(context["memory_rows"])
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.4))
    data = []

    prefetch_rows = [(mem, policies["asg-prefetch-v2-online"]) for mem, policies in sorted(memory.items()) if "asg-prefetch-v2-online" in policies]
    ax = axes[0]
    if prefetch_rows:
        xs = [mem for mem, _ in prefetch_rows]
        hidden = [_num(row, "prefetch_hidden_cycles") / 1e6 for _, row in prefetch_rows]
        exposed = [_num(row, "prefetch_exposed_cycles") / 1e6 for _, row in prefetch_rows]
        width = 0.18
        pos = list(range(len(xs)))
        ax.bar([p - width / 2 for p in pos], hidden, width=width, label="Hidden", color=PALETTE["green"], edgecolor="black", linewidth=0.5)
        ax.bar([p + width / 2 for p in pos], exposed, width=width, label="Exposed", color=PALETTE["red"], edgecolor="black", linewidth=0.5)
        ax.set_xticks(pos)
        ax.set_xticklabels([_fmt_budget(x) for x in xs])
        ax.set_xlabel("State memory budget (GB)")
        ax.set_ylabel("Prefetch cycles (M)")
        ax.set_title("(a) Tool waits can hide only part of prefetch")
        ax.legend(frameon=False, fontsize=7)
        for mem, h, e in zip(xs, hidden, exposed):
            data.append({"panel": "hidden_vs_exposed", "memory_budget_gb": mem, "hidden_mcycles": h, "exposed_mcycles": e})
    else:
        _missing_axis(ax, "No prefetch rows")

    ax = axes[1]
    move = _policy_points(memory, "asg-prefetch-v2-online", "prefetch_migration_bytes", scale=1e9)
    unused = _policy_points(memory, "asg-prefetch-v2-online", "unused_prefetch_events")
    if move:
        _plot_line(ax, move, "Migration bytes", PALETTE["purple"], marker="s")
        ax.set_xscale("log")
        ax.set_xlabel("State memory budget (GB)")
        ax.set_ylabel("Prefetch migration (GB)")
        ax2 = ax.twinx()
        _plot_line(ax2, unused, "Unused events", PALETTE["gray"], marker="o", linestyle="--")
        ax2.set_ylabel("Unused prefetches")
        ax.set_title("(b) Prefetch adds wafer traffic risk")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, frameon=False, fontsize=7, loc="upper left")
        for mem, value in move:
            data.append({"panel": "prefetch_migration", "memory_budget_gb": mem, "prefetch_migration_gb": value})
        for mem, value in unused:
            data.append({"panel": "unused_prefetch", "memory_budget_gb": mem, "unused_prefetch_events": value})
    else:
        _missing_axis(ax, "No prefetch movement rows")

    _finish_grid(fig, [axes])
    data_csv = context["output_dir"] / f"{name}.csv"
    _write_rows(data_csv, data)
    figures = _save_figure(fig, context["output_dir"], name)
    plt.close(fig)
    return {"figures": figures, "data_csv": str(data_csv), "evidence": {"max_prefetch_migration_gb": max([float(row.get("prefetch_migration_gb", 0.0)) for row in data if row.get("panel") == "prefetch_migration"], default=0.0)}}


def _set_isca_style(plt):
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _finish_grid(fig, axes_grid):
    for row in axes_grid:
        for ax in row:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(width=0.7, length=3)
    fig.tight_layout(pad=0.9)


def _plot_line(ax, points, label, color, marker="o", linestyle="-"):
    if not points:
        return
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    ax.plot(xs, ys, label=label, color=color, marker=marker, markersize=4, linewidth=1.4, linestyle=linestyle)


def _missing_axis(ax, label):
    ax.text(0.5, 0.5, label, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def _save_figure(fig, output_dir, name):
    saved = []
    for suffix in ("pdf", "svg", "png"):
        path = output_dir / f"{name}.{suffix}"
        fig.savefig(path, bbox_inches="tight")
        saved.append(str(path))
    return saved


def _find_trace_selection_dir(input_dir):
    candidates = [
        input_dir.parent / "trace_selection_hf",
        input_dir.parent / "trace_selection",
        input_dir.parent / "trace_selection_hf_tight_memory",
    ]
    for candidate in candidates:
        if (candidate / "all_trace_scores.csv").exists():
            return candidate
    return None


def _candidate_eval_dirs(input_dir):
    dirs = [input_dir]
    for name in ("paper_eval_hf_tight_memory", "paper_eval_hf", "paper_eval"):
        candidate = input_dir.parent / name
        if candidate.exists() and candidate not in dirs:
            dirs.append(candidate)
    return dirs


def _load_eval_csv(input_dir, filename, include_siblings):
    rows = []
    dirs = _candidate_eval_dirs(input_dir) if include_siblings else [input_dir]
    for directory in dirs:
        path = directory / filename
        if not path.exists():
            continue
        for row in _read_csv(path):
            row["_source_csv"] = str(path)
            rows.append(row)
    if filename == "memory_sweep.csv":
        return _dedupe_rows(rows, ["memory_budget_gb", "policy"], prefer_source=str(input_dir))
    if filename == "concurrency_sweep.csv":
        return _dedupe_rows(rows, ["concurrency", "policy"], prefer_source=str(input_dir))
    return rows


def _dedupe_rows(rows, keys, prefer_source):
    chosen = {}
    for row in rows:
        key = tuple(row.get(k, "") for k in keys)
        score = 1 if row.get("_source_csv", "").startswith(prefer_source) else 0
        if key not in chosen or score > chosen[key][0]:
            chosen[key] = (score, row)
    return [value[1] for value in chosen.values()]


def _load_selection_rows(selection_dir):
    if not selection_dir:
        return []
    path = selection_dir / "all_trace_scores.csv"
    if not path.exists():
        return []
    return _read_csv(path)


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_json(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["status"]
        rows = [{"status": "no_data"}]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _memory_policy_table(rows):
    return _group_policy_table(rows, "memory_budget_gb")


def _group_policy_table(rows, group_key):
    table = defaultdict(dict)
    for row in rows:
        group = _num(row, group_key)
        policy = row.get("policy", "")
        if policy:
            table[group][policy] = row
    return dict(sorted(table.items()))


def _policy_points(table, policy, metric, scale=1.0, normalize_by=None):
    points = []
    for group, policies in sorted(table.items()):
        if policy not in policies:
            continue
        value = _num(policies[policy], metric) / scale
        if normalize_by:
            base_policy, base_metric = normalize_by
            base = _num(policies.get(base_policy, {}), base_metric)
            if base == 0:
                continue
            value = _num(policies[policy], metric) / base
        points.append((group, value))
    return points


def _improvement_vs_policy(table, policy, baseline_policy, metric):
    points = []
    for group, policies in sorted(table.items()):
        if policy not in policies or baseline_policy not in policies:
            continue
        baseline = _num(policies[baseline_policy], metric)
        value = _num(policies[policy], metric)
        if baseline == 0:
            continue
        points.append((group, (baseline - value) / baseline * 100.0))
    return points


def _selection_evidence(context):
    rows = context["selection_rows"]
    report = context["selection_report"]
    pressure = [_num(row, "estimated_memory_pressure_ratio") for row in rows if _truthy(row.get("paper_usable"))]
    return {
        "num_candidates": report.get("num_candidates", len(rows)),
        "num_paper_usable": report.get("num_paper_usable", sum(1 for row in rows if _truthy(row.get("paper_usable")))),
        "num_matched_control": report.get("num_matched_control", 0),
        "max_paper_usable_live_kv_over_budget": max(pressure, default=0.0),
    }


def _evidence_class(context):
    for key in ("full_rows", "opportunity_rows", "control_rows", "memory_rows", "concurrency_rows", "arrival_rows"):
        for row in context.get(key, []):
            value = row.get("evidence_class")
            if value:
                return value
    return "real"


def _trace_label(context):
    return "Synthetic" if context.get("evidence_class") == "synthetic" else "Paper-usable"


def _limitations(context):
    report = context["selection_report"]
    limitations = []
    if context.get("evidence_class") == "synthetic":
        limitations.append("Synthetic traces validate controlled algorithm behavior only; they cannot support real-trace paper claims.")
        return limitations
    if report.get("num_paper_usable", 0) < 10:
        limitations.append("Only a small number of paper-usable public traces are available locally; treat current plots as problem-existence evidence, not a broad workload study.")
    if report.get("num_matched_control", 0) == 0:
        limitations.append("No matched-control public traces were found in the current local selection run.")
    if not context["control_rows"]:
        limitations.append("Control replay CSV is absent, so opportunity-vs-control speedup claims are not generated.")
    return limitations


def _num(row, key, default=0.0):
    try:
        value = row.get(key, default)
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _truthy(value):
    return str(value).lower() in {"1", "true", "yes", "y"}


def _mean(values):
    values = [v for v in values if not math.isnan(v)]
    return sum(values) / len(values) if values else 0.0


def _scaled_mean(rows, key):
    scale = 1000.0 if key == "LRU_regret_tokens" else 1.0
    return _mean([_num(row, key) / scale for row in rows])


def _fmt_budget(value):
    return f"{value:g}"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate ISCA-style paper-evaluation figures from paper-usable Agent-on-Wafer CSVs.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trace-selection-dir", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = plot_paper_eval(args.input_dir, args.output_dir, trace_selection_dir=args.trace_selection_dir)
    print(f"paper_figures status={report['status']}")


if __name__ == "__main__":
    main()
