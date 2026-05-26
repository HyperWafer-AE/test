"""Request-, operator-, and state-centric workflow schedulers."""

from __future__ import annotations

from dataclasses import dataclass

from .hotness import HotnessConfig, HotnessTracker
from .ir import StateAccessGraph


@dataclass(frozen=True)
class Wave:
    wave_id: int
    scheduler: str
    operator_ids: tuple[str, ...]
    seed_state_id: str | None
    benefit: float
    wait_penalty: float

    @property
    def batch_size(self) -> int:
        return len(self.operator_ids)


class RequestCentricScheduler:
    name = "request_centric"

    def __init__(self) -> None:
        self.wave_counter = 0

    def next_wave(
        self, graph: StateAccessGraph, completed_ops: set[str], current_time: float = 0.0
    ) -> Wave:
        ready = graph.ready_operators(completed_ops)
        if not ready:
            return Wave(self.wave_counter, self.name, tuple(), None, 0.0, 0.0)
        op_id = ready[0]
        wave = Wave(self.wave_counter, self.name, (op_id,), None, 0.0, 0.0)
        self.wave_counter += 1
        return wave


class OperatorCentricScheduler:
    name = "operator_centric"

    def __init__(self, max_wave_size: int = 8) -> None:
        self.max_wave_size = max_wave_size
        self.wave_counter = 0

    def next_wave(
        self, graph: StateAccessGraph, completed_ops: set[str], current_time: float = 0.0
    ) -> Wave:
        ready = graph.ready_operators(completed_ops)[: self.max_wave_size]
        wave = Wave(self.wave_counter, self.name, tuple(ready), None, float(len(ready)), 0.0)
        self.wave_counter += 1
        return wave


class StateCentricWaveScheduler:
    name = "WaferStateFlow"

    def __init__(
        self,
        max_wave_size: int = 8,
        hotness_config: HotnessConfig | None = None,
        min_wave_hotness: float = 2500.0,
        critical_wait_threshold: float = 4.0,
    ) -> None:
        self.max_wave_size = max_wave_size
        self.hotness_config = hotness_config or HotnessConfig()
        self.min_wave_hotness = min_wave_hotness
        self.critical_wait_threshold = critical_wait_threshold
        self.wave_counter = 0
        self._tracker: HotnessTracker | None = None

    def next_wave(
        self, graph: StateAccessGraph, completed_ops: set[str], current_time: float = 0.0
    ) -> Wave:
        if self._tracker is None or self._tracker.graph is not graph:
            self._tracker = HotnessTracker(graph, self.hotness_config)

        ready = graph.ready_operators(completed_ops)
        if not ready:
            return Wave(self.wave_counter, self.name, tuple(), None, 0.0, 0.0)

        overdue = [
            op_id
            for op_id in ready
            if graph.operators[op_id].criticality >= 2.0
            and current_time - float(graph.operators[op_id].ready_time) > self.critical_wait_threshold
        ]
        if overdue:
            op_id = sorted(overdue, key=lambda oid: graph.operators[oid].criticality, reverse=True)[0]
            wave = Wave(self.wave_counter, self.name, (op_id,), None, 0.0, self.critical_wait_threshold)
            self.wave_counter += 1
            return wave

        state_to_ready_ops: dict[str, list[str]] = {}
        for op_id in ready:
            for state_id in graph.operators[op_id].input_states:
                state_to_ready_ops.setdefault(state_id, []).append(op_id)

        candidates = []
        for state_id, op_ids in state_to_ready_ops.items():
            if len(op_ids) < 2:
                continue
            state = graph.states[state_id]
            hotness = max(self._tracker.combined_hotness(state_id), state.token_size * (len(op_ids) - 1))
            batching_gain = len(op_ids) * 0.5
            reuse_gain = state.token_size * (len(op_ids) - 1)
            wait_penalty = sum(max(0.0, current_time - float(graph.operators[op].ready_time)) for op in op_ids)
            benefit = batching_gain + reuse_gain + hotness - wait_penalty
            candidates.append((benefit, hotness, state_id, op_ids, wait_penalty))

        if candidates:
            benefit, hotness, state_id, op_ids, wait_penalty = max(candidates)
            if hotness >= self.min_wave_hotness and benefit > 0:
                selected = tuple(op_ids[: self.max_wave_size])
                wave = Wave(self.wave_counter, self.name, selected, state_id, benefit, wait_penalty)
                self.wave_counter += 1
                return wave

        op_id = max(ready, key=lambda oid: graph.operators[oid].criticality)
        wave = Wave(self.wave_counter, self.name, (op_id,), None, 0.0, 0.0)
        self.wave_counter += 1
        return wave


def make_scheduler(name: str):
    normalized = name.lower()
    if normalized in {"flat_sequential", "request_centric", "request_parallel_gpu_like", "wafer_request_centric"}:
        return RequestCentricScheduler()
    if normalized in {"operator_centric", "helium_like_operator_schedule", "kvflow_like_future_eviction"}:
        return OperatorCentricScheduler()
    if normalized in {"waferstateflow", "state_centric", "state_centric_wave"}:
        return StateCentricWaveScheduler()
    raise ValueError(f"unknown scheduler {name}")
