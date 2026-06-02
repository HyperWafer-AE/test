import argparse
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from .trace_audit import audit_single_trace, audit_traces
from .trace_profile import profile_traces
from .trace_schema import LLMEvent, StateEvent, ToolEvent, validate_trace


@dataclass(frozen=True)
class SyntheticTraceSpec:
    scenario: str
    index: int
    agents: int
    turns: int
    core_states: int
    filler_states: int
    core_tokens: int
    filler_tokens: int
    tool_tokens: int
    delayed_gap: int
    cross_agent: bool
    tool_waits: bool
    phase_shift: bool


def generate_synthetic_trace_suite(
    output_dir,
    opportunity_count: int = 12,
    control_count: int = 6,
    stress_count: int = 4,
    turns: int = 28,
    agents: int = 4,
    seed: int = 0,
    clean: bool = True,
):
    output_dir = Path(output_dir)
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    raw_dir = output_dir / "raw"
    opportunity_dir = output_dir / "opportunity"
    control_dir = output_dir / "control"
    stress_dir = output_dir / "stress"
    for path in (raw_dir, opportunity_dir, control_dir, stress_dir):
        path.mkdir(parents=True, exist_ok=True)

    manifest = {
        "status": "generated",
        "synthetic": True,
        "seed": seed,
        "warning": "Synthetic traces are for algorithm and pipeline testing only; do not mix with real-trace paper evidence.",
        "scenarios": [],
    }
    all_traces = []

    for idx in range(opportunity_count):
        spec = SyntheticTraceSpec(
            scenario="opportunity_delayed_reuse",
            index=idx,
            agents=agents,
            turns=turns + (idx % 3) * 2,
            core_states=4 + (idx % 3),
            filler_states=16 + (idx % 5) * 2,
            core_tokens=1536 + (idx % 4) * 512,
            filler_tokens=256 + (idx % 3) * 128,
            tool_tokens=1024 + (idx % 4) * 256,
            delayed_gap=10 + (idx % 4),
            cross_agent=True,
            tool_waits=True,
            phase_shift=True,
        )
        trace = generate_trace(spec)
        _write_trace(trace, opportunity_dir / f"{idx:04d}_{spec.scenario}.normalized.json")
        _write_trace(trace, raw_dir / f"opportunity_{idx:04d}.normalized.json")
        manifest["scenarios"].append(_manifest_entry(spec, trace, "opportunity"))
        all_traces.append(trace)

    for idx in range(control_count):
        spec = SyntheticTraceSpec(
            scenario="control_short_reuse",
            index=idx,
            agents=max(2, agents // 2),
            turns=max(12, turns // 2 + idx % 3),
            core_states=2,
            filler_states=4 + idx % 2,
            core_tokens=384,
            filler_tokens=128,
            tool_tokens=256,
            delayed_gap=2,
            cross_agent=idx % 2 == 0,
            tool_waits=True,
            phase_shift=False,
        )
        trace = generate_trace(spec)
        _write_trace(trace, control_dir / f"{idx:04d}_{spec.scenario}.normalized.json")
        _write_trace(trace, raw_dir / f"control_{idx:04d}.normalized.json")
        manifest["scenarios"].append(_manifest_entry(spec, trace, "control"))
        all_traces.append(trace)

    for idx in range(stress_count):
        spec = SyntheticTraceSpec(
            scenario="stress_memory_pressure",
            index=idx,
            agents=agents + 2,
            turns=turns + 12 + idx * 2,
            core_states=8,
            filler_states=28 + idx * 4,
            core_tokens=2048 + idx * 512,
            filler_tokens=512,
            tool_tokens=2048,
            delayed_gap=12,
            cross_agent=True,
            tool_waits=True,
            phase_shift=True,
        )
        trace = generate_trace(spec)
        _write_trace(trace, stress_dir / f"{idx:04d}_{spec.scenario}.normalized.json")
        _write_trace(trace, raw_dir / f"stress_{idx:04d}.normalized.json")
        manifest["scenarios"].append(_manifest_entry(spec, trace, "stress"))
        all_traces.append(trace)

    audit = audit_traces(all_traces, min_turns=4, allow_reconstructed_full_history=False, delayed_reuse_k=8)
    profile = profile_traces(all_traces, delayed_reuse_k=8)
    profile = {key: value for key, value in profile.items() if key not in {"state_rows", "reuse_rows"}}
    manifest["audit_summary"] = {
        "paper_usable_trace_count": audit["paper_usable_trace_count"],
        "smoke_test_trace_count": audit["smoke_test_trace_count"],
        "unusable_trace_count": audit["unusable_trace_count"],
        "exclusion_reason_counts": audit["exclusion_reason_counts"],
    }
    manifest["profile_summary"] = {
        "num_traces": profile["num_traces"],
        "num_llm_events": profile["num_llm_events"],
        "num_tool_events": profile["num_tool_events"],
        "delayed_reuse_ratio": profile["delayed_reuse_ratio"],
        "cross_agent_reuse_ratio": profile["cross_agent_reuse_ratio"],
        "LRU_regret_tokens": profile["LRU_regret_tokens"],
        "num_lru_regret_candidates": profile["num_lru_regret_candidates"],
    }
    (output_dir / "synthetic_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "synthetic_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (output_dir / "synthetic_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return manifest


def generate_trace(spec: SyntheticTraceSpec):
    events = []
    state_ids = set()
    prefix = f"syn_{spec.scenario}_{spec.index:04d}"

    def add_state(state_id, state_type, owner, tokens, semantic_key=None):
        if state_id in state_ids:
            return state_id
        state_ids.add(state_id)
        exact = _stable_hash(prefix, state_id, state_type, tokens)
        events.append(
            StateEvent(
                state_id=state_id,
                state_type=state_type,
                owner=owner,
                tokens=tokens,
                semantic_key=semantic_key or f"{spec.scenario}:{state_type}:{state_id}",
                exact_token_hash=exact,
                metadata={
                    "synthetic": True,
                    "scenario": spec.scenario,
                    "semantic_key": semantic_key or f"{spec.scenario}:{state_type}:{state_id}",
                    "exact_token_hash": exact,
                },
            ).to_dict()
        )
        return state_id

    system = add_state(f"{prefix}:system", "system_prefix", "shared", 1024, "shared:system")
    task = add_state(f"{prefix}:task", "task_prefix", "shared", 768, f"{prefix}:task")
    shared_plan = add_state(f"{prefix}:shared_plan", "summary_state", "shared", 1024, f"{prefix}:shared_plan")
    roles = [
        add_state(f"{prefix}:agent_{agent}:role", "agent_role", f"agent_{agent}", 128, f"{prefix}:role:{agent}")
        for agent in range(spec.agents)
    ]
    core_states = [
        add_state(
            f"{prefix}:core_{idx}",
            _core_state_type(idx),
            "shared" if spec.cross_agent and idx % 2 == 0 else f"agent_{idx % spec.agents}",
            spec.core_tokens + (idx % 2) * 256,
            f"{prefix}:core_semantic_{idx}",
        )
        for idx in range(spec.core_states)
    ]
    filler_states = [
        add_state(
            f"{prefix}:filler_{idx}",
            "dialogue_delta" if idx % 3 else "assistant_delta",
            f"agent_{idx % spec.agents}",
            spec.filler_tokens,
            f"{prefix}:filler_semantic_{idx}",
        )
        for idx in range(spec.filler_states)
    ]

    recent_outputs = []
    for turn in range(spec.turns):
        agent_idx = turn % spec.agents
        agent = f"agent_{agent_idx}"
        phase = _phase_for_turn(turn, spec.turns, spec.phase_shift)
        inputs = [system, task, shared_plan, roles[agent_idx]]

        if spec.scenario == "control_short_reuse":
            inputs.append(core_states[turn % len(core_states)])
            if turn > 0:
                inputs.append(recent_outputs[-1])
        else:
            active_core = core_states[turn % len(core_states)]
            delayed_core = core_states[(turn // max(1, spec.delayed_gap)) % len(core_states)]
            inputs.extend([active_core, delayed_core])
            if turn >= spec.delayed_gap:
                inputs.append(core_states[(turn - spec.delayed_gap) % len(core_states)])
            inputs.extend(filler_states[(turn + offset) % len(filler_states)] for offset in range(0, min(6, len(filler_states)), 2))
            if recent_outputs and turn % 5 == 0:
                inputs.append(recent_outputs[max(0, len(recent_outputs) - spec.delayed_gap // 2)])

        inputs = list(dict.fromkeys(inputs))
        llm_state = f"{prefix}:llm_out_{turn}"
        events.append(
            LLMEvent(
                event_id=f"{prefix}:llm:{turn}",
                agent=agent,
                turn=turn,
                input_state_ids=inputs,
                append_tokens=64 + 8 * (turn % 5),
                output_tokens=96 + 16 * (turn % 4),
                new_state_id=llm_state,
                new_state_type="assistant_delta",
                phase=phase,
                metadata=_event_metadata(spec, turn, prompt_reconstruction="explicit"),
            ).to_dict()
        )
        recent_outputs.append(llm_state)

        if spec.tool_waits and (turn % 3 == 1 or (spec.scenario.startswith("stress") and turn % 4 == 2)):
            tool_state_type = _tool_state_type(turn)
            tool_state = f"{prefix}:tool_out_{turn}"
            latency = 80 + (turn % 5) * 40
            events.append(
                ToolEvent(
                    event_id=f"{prefix}:tool:{turn}",
                    agent=agent,
                    turn=turn,
                    tool=_tool_name(turn),
                    latency=latency,
                    output_tokens=spec.tool_tokens + (turn % 3) * 128,
                    status="error" if tool_state_type in {"failure_summary", "test_failure_summary", "raw_error_log"} else "ok",
                    new_state_id=tool_state,
                    new_state_type=tool_state_type,
                    phase=phase,
                    metadata=_event_metadata(spec, turn, prompt_reconstruction="explicit"),
                ).to_dict()
            )
            if spec.scenario != "control_short_reuse":
                core_states.append(tool_state)
            else:
                filler_states.append(tool_state)

    return validate_trace(events)


def _manifest_entry(spec: SyntheticTraceSpec, trace, split: str) -> dict:
    audit = audit_single_trace(trace, trace_idx=spec.index, min_turns=4, allow_reconstructed_full_history=False, delayed_reuse_k=8)
    return {
        "split": split,
        "spec": asdict(spec),
        "quality_label": audit["quality_label"],
        "exclusion_reasons": audit["exclusion_reasons"],
        "reuse_quality": audit["reuse_quality"],
        "num_events": len(trace),
    }


def _write_trace(trace, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace, indent=2), encoding="utf-8")


def _event_metadata(spec: SyntheticTraceSpec, turn: int, prompt_reconstruction: str) -> dict:
    return {
        "synthetic": True,
        "scenario": spec.scenario,
        "synthetic_index": spec.index,
        "turn": turn,
        "trace_success": True,
        "prompt_reconstruction": prompt_reconstruction,
        "full_history_likely": False,
        "monotonic_context_growth_score": 0.0,
    }


def _core_state_type(idx: int) -> str:
    types = ["file_context", "test_failure_summary", "edit_diff", "summary_state", "raw_error_log"]
    return types[idx % len(types)]


def _tool_state_type(turn: int) -> str:
    types = ["tool_observation", "failure_summary", "web_result", "test_failure_summary", "raw_error_log"]
    return types[turn % len(types)]


def _tool_name(turn: int) -> str:
    names = ["read_file", "run_tests", "search", "edit_file", "web_lookup"]
    return names[turn % len(names)]


def _phase_for_turn(turn: int, turns: int, phase_shift: bool) -> str:
    if not phase_shift:
        return "execute"
    frac = turn / max(1, turns - 1)
    if frac < 0.25:
        return "plan"
    if frac < 0.55:
        return "execute"
    if frac < 0.80:
        return "failure"
    return "repair"


def _stable_hash(*parts) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate synthetic Agent-on-Wafer normalized traces for controlled testing.")
    parser.add_argument("--output-dir", default="traces/synthetic_agent")
    parser.add_argument("--opportunity-count", type=int, default=12)
    parser.add_argument("--control-count", type=int, default=6)
    parser.add_argument("--stress-count", type=int, default=4)
    parser.add_argument("--turns", type=int, default=28)
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-clean", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    manifest = generate_synthetic_trace_suite(
        output_dir=args.output_dir,
        opportunity_count=args.opportunity_count,
        control_count=args.control_count,
        stress_count=args.stress_count,
        turns=args.turns,
        agents=args.agents,
        seed=args.seed,
        clean=not args.no_clean,
    )
    print(
        "synthetic_traces status={status} opportunity={opportunity} control={control} stress={stress} paper_usable={paper_usable}".format(
            status=manifest["status"],
            opportunity=args.opportunity_count,
            control=args.control_count,
            stress=args.stress_count,
            paper_usable=manifest["audit_summary"]["paper_usable_trace_count"],
        )
    )


if __name__ == "__main__":
    main()
