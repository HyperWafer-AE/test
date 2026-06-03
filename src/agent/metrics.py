from typing import Dict, List

from .types import KVStateSpec


def _to_number(value):
    try:
        return value.item()
    except AttributeError:
        return value


def _kv_events_from_metadata(event_dict: Dict):
    return [
        event
        for event in event_dict.values()
        if getattr(event, "event_type", None) == "communication" and getattr(event, "is_kv_comm", False)
    ]


def _kv_events_from_data_dict(event_dict: Dict, data_dict: Dict, states: List[KVStateSpec]):
    kv_tags = set()
    for state in states:
        tensor = data_dict[state.data_tag]
        for users in tensor.used_splitted_tag_dict.values():
            for tag in users:
                if tag in event_dict and getattr(event_dict[tag], "event_type", None) == "communication":
                    kv_tags.add(tag)
    return [event_dict[tag] for tag in kv_tags]


def collect_kv_metrics(event_dict, data_dict, states, communication_loads_dict, plan=None) -> dict:
    total_comm_events = sum(
        1 for event in event_dict.values() if getattr(event, "event_type", None) == "communication"
    )
    kv_events = _kv_events_from_metadata(event_dict)
    if not kv_events:
        kv_events = _kv_events_from_data_dict(event_dict, data_dict, states)

    loads = [_to_number(load) for load in communication_loads_dict.values()]
    active_loads = [float(load) for load in loads if float(load) > 0]
    metrics = {
        "total_comm_events": int(total_comm_events),
        "kv_comm_events": int(len(kv_events)),
        "kv_comm_bytes": int(sum(event.comm_bytes for event in kv_events)),
        "kv_hop_bytes": int(sum(getattr(event, "hops", 0) * event.comm_bytes for event in kv_events)),
        "kv_distance": float(sum(getattr(event, "communication_distances", 0) for event in kv_events)),
        "max_link_load": float(max(loads) if loads else 0.0),
        "avg_link_load": float(sum(active_loads) / len(active_loads) if active_loads else 0.0),
        "resident_bytes": 0,
        "unique_state_bytes": 0,
        "extra_replica_bytes": 0,
        "capacity_violations": 0,
        "max_region_used_bytes": 0,
        "avg_region_used_bytes": 0.0,
        "sram_capacity_bytes": 0,
        "num_krds": 0,
    }
    if plan is not None:
        metrics.update(
            {
                "resident_bytes": int(plan.resident_bytes),
                "unique_state_bytes": int(plan.unique_state_bytes),
                "extra_replica_bytes": int(plan.extra_replica_bytes),
                "capacity_violations": int(plan.capacity_violations),
                "max_region_used_bytes": int(plan.max_region_used_bytes),
                "avg_region_used_bytes": float(plan.avg_region_used_bytes),
                "sram_capacity_bytes": int(plan.sram_capacity_bytes),
                "num_krds": int(len(plan.krds)),
            }
        )
    return metrics
