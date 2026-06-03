from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple


Node = Tuple[int, int]
Device = Tuple[int, int, int]
KVKind = Literal[
    "global",
    "workflow_shared",
    "private_hot",
    "private_warm",
    "private_cold",
]
Policy = Literal["central", "full_replication", "krd_selective"]


@dataclass(frozen=True)
class AgentSpec:
    agent_id: int
    workflow_id: int
    role: str
    expected_decode_tokens: int
    expected_return_time: float
    private_blocks: int
    required_state_ids: Tuple[int, ...]
    decode_node: Optional[Node] = None
    krd_id: Optional[int] = None


@dataclass(frozen=True)
class KVStateSpec:
    state_id: int
    name: str
    kind: KVKind
    owner_agent_id: Optional[int]
    workflow_id: Optional[int]
    num_blocks: int
    block_elems: int
    dtype: str = "bf16"
    reuse_count: float = 1.0
    data_tag: Optional[Tuple[int, int]] = None

    @property
    def block_bytes(self) -> int:
        elem_bytes = 2 if self.dtype in {"bf16", "bfloat16", "fp16", "float16"} else 4
        return self.block_elems * elem_bytes

    @property
    def total_bytes(self) -> int:
        return self.num_blocks * self.block_bytes


@dataclass
class KRD:
    krd_id: int
    agent_ids: List[int]
    workflow_ids: List[int]
    regions: List[Node] = field(default_factory=list)
    anchor: Optional[Node] = None


@dataclass
class PlacementPlan:
    policy: str
    state_locations: Dict[int, List[Node]]
    agent_decode_nodes: Dict[int, Node]
    krds: List[KRD]
    replica_bytes: int = 0

