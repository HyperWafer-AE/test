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


def collect_kv_metrics(event_dict, data_dict, states, communication_loads_dict) -> dict:
    total_comm_events = sum(
        1 for event in event_dict.values() if getattr(event, "event_type", None) == "communication"
    )
    kv_events = _kv_events_from_metadata(event_dict)
    if not kv_events:
        kv_events = _kv_events_from_data_dict(event_dict, data_dict, states)

    loads = [_to_number(load) for load in communication_loads_dict.values()]
    active_loads = [float(load) for load in loads if float(load) > 0]
    return {
        "total_comm_events": int(total_comm_events),
        "kv_comm_events": int(len(kv_events)),
        "kv_comm_bytes": int(sum(event.comm_bytes for event in kv_events)),
        "kv_hop_bytes": int(sum(getattr(event, "hops", 0) * event.comm_bytes for event in kv_events)),
        "kv_distance": float(sum(getattr(event, "communication_distances", 0) for event in kv_events)),
        "max_link_load": float(max(loads) if loads else 0.0),
        "avg_link_load": float(sum(active_loads) / len(active_loads) if active_loads else 0.0),
        "replica_bytes": 0,
        "num_krds": 0,
    }

