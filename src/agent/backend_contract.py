"""Event-level BusyBarn backend contract for Agent-on-Wafer.

Agent-on-Wafer currently uses BusyBarn as an event-level wafer backend. It
does not use BusyBarn's original operator-level LLM partition pipeline for
prefill/decode.

Event meanings:

fixed_compute:
    Analytical LLM prefill/decode event. It occupies a BusyBarn compute
    module. Duration comes from ModelProfile.

communication:
    KV remote-read, migration, replicate, or prefetch. It uses BusyBarn NoC
    routing and link scheduling.

external_wait:
    Tool/environment latency. It does not occupy compute or links, but creates
    dependency delay.
"""

from typing import Optional, Tuple

from src.scheduling.event_notation import (
    communication_notation,
    external_wait_notation,
    fixed_compute_notation,
)


def emit_fixed_compute_event(
    events_dict: dict,
    event_tag: int,
    name: str,
    comp_device: str,
    comp_location: Tuple,
    duration: int,
    metadata: Optional[dict] = None,
):
    event = fixed_compute_notation(
        comp_name=name,
        comp_tag=event_tag,
        comp_device=comp_device,
        comp_location=tuple(comp_location),
        duration=max(1, int(duration)),
        metadata=metadata or {},
    )
    events_dict[event.event_tag] = event
    return event


def emit_external_wait_event(
    events_dict: dict,
    event_tag: int,
    name: str,
    duration: int,
    metadata: Optional[dict] = None,
):
    event = external_wait_notation(
        wait_name=name,
        wait_tag=event_tag,
        duration=max(0, int(duration)),
        metadata=metadata or {},
    )
    events_dict[event.event_tag] = event
    return event


def emit_communication_event(
    events_dict: dict,
    event_tag: int,
    name: str,
    source_location: Tuple,
    target_location: Tuple,
    comm_bytes: int,
    metadata: Optional[dict] = None,
):
    event = communication_notation(
        comm_name=name,
        comm_tag=event_tag,
        source_location=tuple(source_location),
        target_location=tuple(target_location),
        comm_bytes=max(0, int(comm_bytes)),
    )
    event.metadata = metadata or {}
    events_dict[event.event_tag] = event
    return event


def add_dependency(events_dict: dict, parent_tag: Optional[int], child_event):
    if parent_tag is None:
        return child_event
    if parent_tag not in events_dict:
        raise ValueError(f"Dependency parent tag {parent_tag} does not exist.")
    if parent_tag == child_event.event_tag:
        raise ValueError(f"Self dependency on event tag {parent_tag}.")
    child_event.dependency_set.add(parent_tag)
    events_dict[parent_tag].issue_set.add(child_event.event_tag)
    return child_event


def validate_event_location(event, hardware_platform):
    event_type = getattr(event, "event_type", None)
    if event_type == "fixed_compute":
        modules = getattr(hardware_platform, "modules_dict", {})
        if event.comp_device not in modules:
            raise ValueError(f"Unknown compute device {event.comp_device}.")
        if tuple(event.comp_location) not in modules[event.comp_device]:
            raise ValueError(f"Unknown compute location {event.comp_location}.")
    elif event_type == "communication":
        nodes = set(getattr(hardware_platform, "nodes_set", set()))
        if tuple(event.source_location) not in nodes:
            raise ValueError(f"Unknown communication source {event.source_location}.")
        targets = event.target_location if isinstance(event.target_location, set) else {event.target_location}
        for target in targets:
            if tuple(target) not in nodes:
                raise ValueError(f"Unknown communication target {target}.")
    return True
