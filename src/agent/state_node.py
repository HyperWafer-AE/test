from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


STATE_TYPES = {
    "system_prefix",
    "agent_role",
    "task_prefix",
    "shared_prefix",
    "dialogue_delta",
    "assistant_delta",
    "tool_observation",
    "file_context",
    "edit_diff",
    "test_failure_summary",
    "failure_summary",
    "raw_error_log",
    "skill_core",
    "speculative_state",
}


@dataclass
class StateNode:
    state_id: str
    state_type: str
    owner: str
    token_len: int
    kv_bytes: int
    birth_step: int
    last_access: int
    access_count: int = 0

    reuse_prob: float = 0.0
    next_use: float = float("inf")
    retention_score: float = 0.0
    phase_value: float = 0.0

    loc: Optional[Tuple] = None
    tier: str = "cold"
    resident: bool = False

    exact_kv_id: Optional[str] = None
    semantic_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    producer_exec_id: Optional[str] = None
    producer_event_tag: Optional[int] = None
    available_event_tag: Optional[int] = None
    replica_locs: Set[Tuple] = field(default_factory=set)
    pinned: bool = False
    anchored: bool = False
    summarized: bool = False
    predicted_future_accesses: float = 1.0


@dataclass
class ExecNode:
    exec_id: str
    exec_type: str
    agent_id: str
    phase: str
    input_states: List[str] = field(default_factory=list)
    output_states: List[str] = field(default_factory=list)

    input_tokens: int = 0
    append_tokens: int = 0
    output_tokens: int = 0
    tool_name: Optional[str] = None
    tool_latency: int = 0
    status: str = "ok"

    assigned_loc: Optional[Tuple] = None
    metadata: Dict = field(default_factory=dict)
