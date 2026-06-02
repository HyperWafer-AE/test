from .state_node import ExecNode


class ASGBuilderRuntime:
    """Own ASG mutations without creating BusyBarn events."""

    def __init__(self, graph, model_profile):
        self.graph = graph
        self.model_profile = model_profile

    def build_or_get_state(
        self,
        state_id: str,
        state_type: str,
        owner: str,
        token_len: int,
        kv_bytes=None,
        semantic_id=None,
        exact_kv_id=None,
        metadata=None,
    ):
        return self.graph.build_or_get_state(
            state_id=state_id,
            state_type=state_type,
            owner=owner,
            token_len=token_len,
            kv_bytes=int(kv_bytes if kv_bytes is not None else self.model_profile.kv_bytes(token_len)),
            semantic_id=semantic_id,
            exact_kv_id=exact_kv_id,
            metadata=metadata,
        )

    def add_llm_exec(self, exec_id: str, agent: str, phase: str, input_ids, input_tokens: int, append_tokens: int, output_tokens: int, metadata: dict, status: str = "ok"):
        exec_node = ExecNode(
            exec_id=exec_id,
            exec_type="llm",
            agent_id=agent,
            phase=phase,
            input_states=list(input_ids),
            input_tokens=input_tokens,
            append_tokens=append_tokens,
            output_tokens=output_tokens,
            status=status,
            metadata=metadata,
        )
        self.graph.add_exec(exec_node)
        for state_id in input_ids:
            self.graph.add_dependency(state_id, exec_id)
            self.graph.add_affinity(state_id, exec_id, 1.0)
        return exec_node

    def add_tool_exec(self, exec_id: str, agent: str, phase: str, tool_name: str, latency: int, output_tokens: int, metadata: dict, status: str = "ok"):
        exec_node = ExecNode(
            exec_id=exec_id,
            exec_type="tool",
            agent_id=agent,
            phase=phase,
            output_tokens=output_tokens,
            tool_name=tool_name,
            tool_latency=latency,
            status=status,
            metadata=metadata,
        )
        return self.graph.add_exec(exec_node)

    def add_generation(self, exec_id: str, state_id: str):
        self.graph.add_generation(exec_id, state_id)
        self.graph.add_affinity(exec_id, state_id, 1.0)
