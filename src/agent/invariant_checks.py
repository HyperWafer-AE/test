from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from .backend_contract import validate_event_location


@dataclass
class InvariantReport:
    name: str
    passed: bool
    failures: List[str] = field(default_factory=list)

    def to_dict(self):
        return {"name": self.name, "passed": self.passed, "failures": list(self.failures)}


def check_event_graph(events_dict, hardware_platform) -> InvariantReport:
    failures = []
    for tag, event in events_dict.items():
        if getattr(event, "event_tag", tag) != tag:
            failures.append(f"event key {tag} does not match event_tag {getattr(event, 'event_tag', None)}")
        for dep in getattr(event, "dependency_set", set()):
            if dep not in events_dict:
                failures.append(f"event {tag} depends on missing tag {dep}")
            if dep == tag:
                failures.append(f"event {tag} has self dependency")
        for target in getattr(event, "issue_set", set()):
            if target not in events_dict:
                failures.append(f"event {tag} issues missing target {target}")
            if target == tag:
                failures.append(f"event {tag} issues itself")
        try:
            validate_event_location(event, hardware_platform)
        except ValueError as exc:
            failures.append(f"event {tag}: {exc}")
        if getattr(event, "event_type", None) == "communication":
            if int(getattr(event, "comm_bytes", 0)) < 0:
                failures.append(f"communication event {tag} has negative bytes")
    return InvariantReport("event_graph", not failures, failures)


def check_asg_graph(graph) -> InvariantReport:
    failures = []
    for state_id, state in graph.states.items():
        if state.state_id != state_id:
            failures.append(f"state key {state_id} does not match state_id {state.state_id}")
        if state.token_len < 0:
            failures.append(f"state {state_id} has negative token_len")
        if state.kv_bytes < 0:
            failures.append(f"state {state_id} has negative kv_bytes")
        if state.producer_exec_id and state.producer_exec_id not in graph.execs:
            failures.append(f"state {state_id} producer_exec_id missing: {state.producer_exec_id}")
    for exec_id, exec_node in graph.execs.items():
        if exec_node.exec_id != exec_id:
            failures.append(f"exec key {exec_id} does not match exec_id {exec_node.exec_id}")
        for state_id in exec_node.input_states:
            if state_id not in graph.states:
                failures.append(f"exec {exec_id} input state missing: {state_id}")
        for state_id in exec_node.output_states:
            if state_id not in graph.states:
                failures.append(f"exec {exec_id} output state missing: {state_id}")
    for state_id, exec_ids in graph.dep_edges.items():
        if state_id not in graph.states:
            failures.append(f"dep edge source state missing: {state_id}")
        for exec_id in exec_ids:
            if exec_id not in graph.execs:
                failures.append(f"dep edge target exec missing: {exec_id}")
    for exec_id, state_ids in graph.gen_edges.items():
        if exec_id not in graph.execs:
            failures.append(f"gen edge source exec missing: {exec_id}")
        for state_id in state_ids:
            if state_id not in graph.states:
                failures.append(f"gen edge target state missing: {state_id}")
    return InvariantReport("asg_graph", not failures, failures)


def check_state_locations(graph, hardware_platform, events_dict=None) -> InvariantReport:
    failures = []
    nodes = set(getattr(hardware_platform, "nodes_set", set()))
    event_tags = set(events_dict or {})
    for state in graph.states.values():
        if state.loc is not None and tuple(state.loc) not in nodes:
            failures.append(f"state {state.state_id} loc is not a wafer node: {state.loc}")
        for replica in state.replica_locs:
            if tuple(replica) not in nodes:
                failures.append(f"state {state.state_id} replica loc is not a wafer node: {replica}")
        if events_dict is not None and state.available_event_tag is not None and state.available_event_tag not in event_tags:
            failures.append(f"state {state.state_id} available_event_tag missing: {state.available_event_tag}")
    return InvariantReport("state_locations", not failures, failures)


def check_policy_outputs(metrics, graph, policy_name: str, timing: Optional[dict] = None, paper_usable: Optional[bool] = None, events_dict=None) -> InvariantReport:
    failures = []
    metrics_dict = metrics.to_dict() if hasattr(metrics, "to_dict") else dict(metrics)
    if policy_name == "nocache":
        if int(metrics_dict.get("cache_hits", 0)) != 0:
            failures.append("nocache reported resident cache hits")
        if int(metrics_dict.get("local_state_hits", 0)) != 0 or int(metrics_dict.get("remote_state_hits", 0)) != 0:
            failures.append("nocache reported local/remote state hits")
    has_comm = (
        int(metrics_dict.get("remote_read_bytes", 0)) > 0
        or int(metrics_dict.get("demand_migration_bytes", 0)) > 0
        or int(metrics_dict.get("prefetch_migration_bytes", 0)) > 0
    )
    if has_comm and events_dict is not None:
        comm_events = [
            event
            for event in events_dict.values()
            if getattr(event, "event_type", None) == "communication"
            and int(getattr(event, "comm_bytes", 0)) > 0
        ]
        if not comm_events:
            failures.append("communication bytes exist but no BusyBarn communication events were emitted")
    if timing:
        for key in ("total_cycles", "pure_comp_cycles", "pure_comm_cycles"):
            if key in timing and int(timing.get(key, 0)) < 0:
                failures.append(f"{key} is negative")
    if paper_usable is False and "speedup" in metrics_dict:
        failures.append("non-paper-usable run contains speedup metric")
    return InvariantReport("policy_outputs", not failures, failures)


def run_invariant_checks(events_dict, graph, metrics, hardware_platform, policy_name: str, timing: Optional[dict] = None, paper_usable: Optional[bool] = None) -> List[InvariantReport]:
    return [
        check_event_graph(events_dict, hardware_platform),
        check_asg_graph(graph),
        check_state_locations(graph, hardware_platform, events_dict=events_dict),
        check_policy_outputs(metrics, graph, policy_name, timing=timing, paper_usable=paper_usable, events_dict=events_dict),
    ]


def assert_invariant_reports(reports: Iterable[InvariantReport]):
    failures = []
    for report in reports:
        failures.extend(f"{report.name}: {failure}" for failure in report.failures)
    if failures:
        raise AssertionError("Invariant checks failed:\n" + "\n".join(failures))
