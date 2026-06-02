from . import backend_contract


class EventCompiler:
    """Compile agent-runtime decisions into BusyBarn event-level events."""

    def __init__(self, events_dict: dict, next_tag):
        self.events_dict = events_dict
        self._next_tag = next_tag

    def fixed_compute(self, name: str, device: str, location, duration: int, metadata=None):
        return backend_contract.emit_fixed_compute_event(
            self.events_dict,
            self._next_tag(),
            name,
            device,
            location,
            duration,
            metadata=metadata,
        )

    def external_wait(self, name: str, duration: int, metadata=None):
        return backend_contract.emit_external_wait_event(
            self.events_dict,
            self._next_tag(),
            name,
            duration,
            metadata=metadata,
        )

    def communication(self, name: str, source_location, target_location, comm_bytes: int, metadata=None):
        return backend_contract.emit_communication_event(
            self.events_dict,
            self._next_tag(),
            name,
            source_location,
            target_location,
            comm_bytes,
            metadata=metadata,
        )

    def add_dependency(self, parent_tag, child_event):
        return backend_contract.add_dependency(self.events_dict, parent_tag, child_event)

    def validate_event(self, event, hardware_platform):
        return backend_contract.validate_event_location(event, hardware_platform)
