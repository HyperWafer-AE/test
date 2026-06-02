from .placement import (
    choose_exec_location,
    choose_shared_anchor,
    choose_state_location,
)


class WaferMapperRuntime:
    """Own topology decisions without emitting BusyBarn events."""

    def __init__(self, hardware_platform, node_memory, module_load):
        self.hardware_platform = hardware_platform
        self.node_memory = node_memory
        self.module_load = module_load
        self.shared_anchor = None

    def choose_shared_anchor(self):
        self.shared_anchor = choose_shared_anchor(self.hardware_platform)
        return self.shared_anchor

    def choose_exec_location(
        self,
        input_states,
        agent_home,
        affinity_states=None,
        locality_weight=1.0,
        home_weight=0.25,
        affinity_weight=0.5,
        load_weight=0.1,
    ):
        return choose_exec_location(
            input_states,
            self.hardware_platform,
            agent_home=agent_home,
            affinity_states=affinity_states,
            module_load=self.module_load,
            locality_weight=locality_weight,
            home_weight=home_weight,
            affinity_weight=affinity_weight,
            load_weight=load_weight,
        )

    def choose_state_location(self, state, predicted_exec_loc, agent_home=None, shared_anchor=None):
        return choose_state_location(
            state,
            predicted_exec_loc,
            self.hardware_platform,
            agent_home=agent_home,
            shared_anchor=shared_anchor if shared_anchor is not None else self.shared_anchor,
            per_node_used=self.node_memory.node_used,
            per_node_budget=self.node_memory.per_node_budget_bytes,
        )
