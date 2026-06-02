from bisect import bisect_left
from collections import defaultdict
from typing import Dict, List, Optional, Set


class FutureDemandIndex:
    """Oracle-only index over replay-trace future state consumers."""

    def __init__(self, trace: list[dict], horizon: int = 16):
        self.trace = trace
        self.horizon = int(horizon)
        self._state_steps: Dict[str, List[int]] = defaultdict(list)
        self._state_consumers: Dict[str, List[dict]] = defaultdict(list)
        self._tool_steps_by_agent: Dict[str, List[int]] = defaultdict(list)
        self._tool_events_by_agent: Dict[str, List[dict]] = defaultdict(list)
        self._build()

    def _build(self):
        for step, event in enumerate(self.trace):
            event_type = event.get("type")
            if event_type == "llm":
                agent = event.get("agent")
                for state_id in event.get("input_state_ids", []):
                    consumer = dict(event)
                    consumer["step"] = step
                    consumer["agent"] = agent
                    self._state_steps[state_id].append(step)
                    self._state_consumers[state_id].append(consumer)
            elif event_type == "tool":
                agent = event.get("agent")
                if agent is None:
                    continue
                tool_event = dict(event)
                tool_event["step"] = step
                tool_event["latency"] = int(event.get("latency", event.get("tool_latency", 0)))
                self._tool_steps_by_agent[agent].append(step)
                self._tool_events_by_agent[agent].append(tool_event)

    def _limit(self, step: int, horizon: Optional[int]) -> int:
        active_horizon = self.horizon if horizon is None else int(horizon)
        return int(step) + active_horizon

    def next_use_distance(self, step: int, state_id: str) -> float:
        steps = self._state_steps.get(state_id, [])
        idx = bisect_left(steps, int(step))
        if idx >= len(steps):
            return float("inf")
        return float(steps[idx] - int(step))

    def future_access_count(self, step: int, state_id: str, horizon: Optional[int] = None) -> int:
        steps = self._state_steps.get(state_id, [])
        start = bisect_left(steps, int(step))
        end = bisect_left(steps, self._limit(step, horizon) + 1)
        return max(0, end - start)

    def future_consumers(self, step: int, state_id: str, horizon: Optional[int] = None) -> List[dict]:
        steps = self._state_steps.get(state_id, [])
        consumers = self._state_consumers.get(state_id, [])
        start = bisect_left(steps, int(step))
        end = bisect_left(steps, self._limit(step, horizon) + 1)
        return [dict(item) for item in consumers[start:end]]

    def future_agents(self, step: int, state_id: str, horizon: Optional[int] = None) -> Set[str]:
        return {
            consumer["agent"]
            for consumer in self.future_consumers(step, state_id, horizon)
            if consumer.get("agent") is not None
        }

    def future_tool_wait_windows(self, step: int, agent, horizon: Optional[int] = None) -> List[dict]:
        steps = self._tool_steps_by_agent.get(agent, [])
        events = self._tool_events_by_agent.get(agent, [])
        start = bisect_left(steps, int(step) + 1)
        end = bisect_left(steps, self._limit(step, horizon) + 1)
        return [dict(item) for item in events[start:end]]
